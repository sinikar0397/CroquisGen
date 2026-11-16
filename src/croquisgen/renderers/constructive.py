"""Constructive anatomy - the body as spheres, cylinders and ellipsoids.

Owner: 3802 김민재.

No learned model; it is geometry driven by the detected joints. Four stages, in
the order the plan lists them:

1. 골격 전처리 - landmarks are smoothed across frames (z is several times noisier
   than xy and gets a heavier filter) and a body unit, the head height H, is
   estimated from the torso. Every size below is a multiple of H taken from an
   8-head figure, never the raw distance between two detected joints.
2. 도형 생성 - head, ribcage and pelvis as ellipsoids, joints as spheres, limbs
   as tapered cylinders. Centre, size and rotation come from the joints plus the
   canonical proportions.
3. 인체구조 보정 - a limb that projects longer than its canonical length is a
   detection error, not a long arm: the distal joint is pulled in and the rest of
   the chain moves with it. A limb that projects shorter is foreshortened, which
   is real and is drawn as such.
4. 렌더링 - solid forms, not wireframes. Every form is collected with a depth,
   then painted far to near: filled with paper colour first, outlined second, so
   a nearer form hides the lines of whatever is behind it (painter's algorithm).
   Outlines carry the construction cues artists use: end caps whose roundness
   follows the foreshortening, drawn in full at the near end and as a half ring
   at the far end; cross-contour rings along the limbs; centre lines on head and
   torso that shift toward the side facing the camera.

Depth comes from the landmark z, which MediaPipe scales like x. It decides the
paint order, which end of a limb is nearer and which way a mass faces, so its
noise shows up as an occasional swap of two forms at nearly the same depth, not
as distorted geometry.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..types import (
    L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_FOOT_INDEX, L_HEEL, L_HIP, L_INDEX, L_KNEE,
    L_PINKY, L_SHOULDER, L_WRIST, NOSE, PerceptionResult, R_ANKLE, R_EAR, R_ELBOW,
    R_EYE, R_FOOT_INDEX, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_PINKY, R_SHOULDER,
    R_WRIST, Target,
)
from .base import blank_paper, scaled_points, visible

INK = config.INK_COLOR
PAPER = config.PAPER_COLOR

# --- canonical proportions, in head heights H (8-head figure) ----------------
TORSO_H = 2.5           # shoulder centre -> hip centre
SHOULDER_SPAN_H = 1.8
# Projected lengths above canonical x this are treated as detection errors.
LENGTH_TOLERANCE = 1.15

# A chain is (joint indices, per-segment (length, proximal radius, distal radius)).
_ARM = ((1.5, 0.22, 0.16), (1.25, 0.16, 0.11))
_LEG = ((2.0, 0.34, 0.21), (1.9, 0.22, 0.13))
_BODY_CHAINS = (
    ((L_SHOULDER, L_ELBOW, L_WRIST), _ARM),
    ((R_SHOULDER, R_ELBOW, R_WRIST), _ARM),
    ((L_HIP, L_KNEE, L_ANKLE), _LEG),
    ((R_HIP, R_KNEE, R_ANKLE), _LEG),
)
# Points that ride along when the joint they hang from is corrected.
_BODY_FOLLOWERS = {
    L_WRIST: (L_PINKY, L_INDEX), R_WRIST: (R_PINKY, R_INDEX),
    L_ANKLE: (L_HEEL, L_FOOT_INDEX), R_ANKLE: (R_HEEL, R_FOOT_INDEX),
}
_BODY_JOINTS = {
    L_SHOULDER: 0.24, R_SHOULDER: 0.24, L_ELBOW: 0.17, R_ELBOW: 0.17,
    L_WRIST: 0.12, R_WRIST: 0.12, L_HIP: 0.28, R_HIP: 0.28,
    L_KNEE: 0.23, R_KNEE: 0.23, L_ANKLE: 0.14, R_ANKLE: 0.14,
}

# Hand proportions, in palm lengths P (wrist -> middle MCP).
PALM_WIDTH_P = 0.75     # index MCP -> pinky MCP
_FINGER = ((0.42, 0.10, 0.085), (0.27, 0.085, 0.075), (0.22, 0.075, 0.065))
_THUMB_CHAIN = ((1, 2, 3, 4), ((0.45, 0.14, 0.12), (0.33, 0.12, 0.10), (0.27, 0.10, 0.085)))
# Index to pinky, in order: the palm's top edge is built from these bases.
_FINGER_CHAINS = (
    ((5, 6, 7, 8), _FINGER),
    ((9, 10, 11, 12), ((0.46, 0.10, 0.085), (0.30, 0.085, 0.075), (0.24, 0.075, 0.065))),
    ((13, 14, 15, 16), _FINGER),
    ((17, 18, 19, 20), ((0.34, 0.09, 0.075), (0.22, 0.075, 0.065), (0.20, 0.065, 0.055))),
)
_HAND_CHAINS = (_THUMB_CHAIN, *_FINGER_CHAINS)


# --- stage 1: preprocessing --------------------------------------------------

class _Smoother:
    """Exponential smoothing over strictly increasing timestamps.

    A timestamp that does not advance, or jumps by more than `max_gap_ms`, resets
    the filter. That makes a still image or a test (timestamp 0 every call) render
    unsmoothed, and a style switch after a pause start clean instead of sliding in
    from a stale pose.
    """

    def __init__(self, alpha_xy: float = 0.55, alpha_z: float = 0.25,
                 max_gap_ms: int = 500) -> None:
        self._alpha = np.array([alpha_xy, alpha_xy, alpha_z], dtype=np.float32)
        self._max_gap = max_gap_ms
        self._ts = 0
        self._points: np.ndarray | None = None

    def update(self, timestamp_ms: int, points: np.ndarray) -> np.ndarray:
        previous = self._points
        if (previous is None or previous.shape != points.shape
                or timestamp_ms <= self._ts or timestamp_ms - self._ts > self._max_gap):
            self._points = points.astype(np.float32).copy()
        else:
            self._points = self._alpha * points + (1 - self._alpha) * previous
        self._ts = timestamp_ms
        return self._points


def _head_unit(points: np.ndarray, result: PerceptionResult) -> float:
    """Head height in pixels, estimated from the torso.

    Each measurement can only shrink under foreshortening, so the largest of them
    is the closest to the truth.
    shortcut: torso-only estimate, so a torso foreshortened in both axes shrinks
    the limbs; upgrade to pose_world lengths once depth is smoothed enough.
    """
    candidates = [0.0]
    if visible(result, L_SHOULDER, R_SHOULDER):
        candidates.append(
            float(np.linalg.norm(points[L_SHOULDER] - points[R_SHOULDER])) / SHOULDER_SPAN_H)
        if visible(result, L_HIP, R_HIP):
            shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
            hips = (points[L_HIP] + points[R_HIP]) / 2
            candidates.append(float(np.linalg.norm(shoulders - hips)) / TORSO_H)
    return max(candidates)


# --- stage 3: proportion correction -----------------------------------------

def _correct_proportions(points: np.ndarray, chains, unit: float,
                         followers: dict | None = None) -> np.ndarray:
    """Pull in every segment that projects longer than its canonical length.

    Joints further down the chain move by the same offset, so the next segment
    keeps its own direction and length, and `followers` (hand and foot points)
    move with the joint they hang from.
    """
    points = points.copy()
    followers = followers or {}
    for chain, segments in chains:
        delta = np.zeros(2, dtype=np.float32)
        for (a, b), (length, _, _) in zip(zip(chain, chain[1:]), segments):
            points[b] += delta
            v = points[b] - points[a]
            dist = float(np.linalg.norm(v))
            limit = length * unit * LENGTH_TOLERANCE
            if dist > limit > 0:
                pull = v * (limit / dist - 1)
                points[b] += pull
                delta += pull
            for f in followers.get(b, ()):
                points[f] += delta
    return points


def _foreshortening(start: np.ndarray, end: np.ndarray, canonical: float) -> float:
    """|cos| of the angle between the limb and the image plane, in [0, 1].

    0 = lying in the picture plane (end caps are flat lines), 1 = pointing at the
    camera (end caps are full circles). Comes from the projected / canonical
    length ratio, which is steadier than the z coordinate.
    """
    if canonical <= 0:
        return 0.0
    ratio = min(1.0, float(np.linalg.norm(end - start)) / canonical)
    return math.sqrt(1.0 - ratio * ratio)


# --- stage 4: solid forms ----------------------------------------------------

Painter = Callable[[np.ndarray], None]


class _Scene:
    """Painter's algorithm over simple forms.

    Each form is a depth plus two painters. paint() runs them far to near, fill
    then outline, so a nearer form's paper-coloured fill erases the lines of
    anything behind it. That is the whole of the hidden-line removal.
    """

    def __init__(self) -> None:
        self._forms: list[tuple[float, Painter, Painter]] = []

    def add(self, depth: float, fill: Painter, stroke: Painter) -> None:
        self._forms.append((float(depth), fill, stroke))

    def paint(self, canvas: np.ndarray) -> None:
        for _, fill, stroke in sorted(self._forms, key=lambda f: -f[0]):
            fill(canvas)
            stroke(canvas)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _axes(a: float, b: float) -> tuple[int, int]:
    return max(1, int(round(a))), max(1, int(round(b)))


def _half_ellipse(canvas, centre, along, half_along: float, half_across: float,
                  thickness: int, full: bool = False) -> None:
    """Ellipse with its first axis on `along`; only the half displaced toward
    +along unless `full`. Every ring, cap and centre line is one of these."""
    angle = math.degrees(math.atan2(float(along[1]), float(along[0])))
    cv2.ellipse(canvas, _pt(centre), _axes(half_along, half_across), angle,
                0 if full else -90, 360 if full else 90, INK, thickness, cv2.LINE_AA)


def _add_cylinder(scene: _Scene, depth: float, start, end, r_start: float, r_end: float,
                  squash: float, near_is_end: bool, thick: int, thin: int, *,
                  cap_start: bool = True, cap_end: bool = True, rings: int = 0) -> None:
    """A tapered tube with the cues that make it read as a solid.

    The cap at the near end is a full ellipse; the far cap and any mid rings show
    only the half displaced away from the near end, the half that is actually in
    view on a solid. `squash` is the cap's minor/major ratio, i.e. foreshortening.
    """
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        return
    axis = direction / length
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    toward_far = -axis if near_is_end else axis
    quad = np.array([start + normal * r_start, end + normal * r_end,
                     end - normal * r_end, start - normal * r_start], dtype=np.int32)
    cap_angle = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) + 90

    def fill(c):
        cv2.fillConvexPoly(c, quad, PAPER, cv2.LINE_8)
        for centre, r in ((start, r_start), (end, r_end)):
            cv2.ellipse(c, _pt(centre), _axes(r * squash, r), cap_angle, 0, 360, PAPER, -1, cv2.LINE_8)

    def stroke(c):
        for sign in (1, -1):
            cv2.line(c, _pt(start + normal * r_start * sign), _pt(end + normal * r_end * sign),
                     INK, thick, cv2.LINE_AA)
        if cap_start:
            _half_ellipse(c, start, toward_far, r_start * squash, r_start, thick, full=not near_is_end)
        if cap_end:
            _half_ellipse(c, end, toward_far, r_end * squash, r_end, thick, full=near_is_end)
        for k in range(1, rings + 1):
            t = k / (rings + 1)
            r = r_start + (r_end - r_start) * t
            _half_ellipse(c, start + direction * t, toward_far, r * squash, r, thin)

    scene.add(depth, fill, stroke)


def _add_sphere(scene: _Scene, depth: float, centre, r: float, thickness: int) -> None:
    radius = max(2, int(r))
    # A sphere's surface is nearer than its centre by r, which is what lets the
    # ball of a joint sit on top of the cylinders that meet there.
    scene.add(depth - r,
              lambda c: cv2.circle(c, _pt(centre), radius, PAPER, -1, cv2.LINE_8),
              lambda c: cv2.circle(c, _pt(centre), radius, INK, thickness, cv2.LINE_AA))


def _add_ellipse(scene: _Scene, depth: float, centre, axes: tuple[int, int], angle: float,
                 thick: int, extra: Painter | None = None) -> None:
    def stroke(c):
        cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, INK, thick, cv2.LINE_AA)
        if extra:
            extra(c)

    scene.add(depth,
              lambda c: cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, PAPER, -1, cv2.LINE_8),
              stroke)


def _add_polygon(scene: _Scene, depth: float, pts: np.ndarray, thick: int,
                 extra: Painter | None = None) -> None:
    poly = pts.astype(np.int32)

    def stroke(c):
        cv2.polylines(c, [poly], True, INK, thick, cv2.LINE_AA)
        if extra:
            extra(c)

    scene.add(depth, lambda c: cv2.fillPoly(c, [poly], PAPER, cv2.LINE_8), stroke)


def _add_ellipsoid(scene: _Scene, depth: float, centre, up, half_w: float, half_h: float,
                   depth_r: float, facing: float, thick: int, thin: int,
                   ring_at: float, ring_squash: float, ring_toward: float) -> None:
    """An ellipsoid: outline, a meridian centre line, and one latitude ring.

    `facing` is sin(yaw) along `right`; it widens the silhouette by the ellipsoid's
    depth and shifts the centre line toward the side turned to the camera.
    `ring_at` places the ring along `up` (fraction of half_h), `ring_squash` is its
    minor/major ratio, `ring_toward` the sign of `up` its visible half faces.
    """
    right = np.array([-up[1], up[0]], dtype=np.float32)
    facing = max(-1.0, min(1.0, facing))
    proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
    angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
    side = 1.0 if facing >= 0 else -1.0

    def lines(c):
        _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin)
        _half_ellipse(c, centre + up * (half_h * ring_at), up * ring_toward,
                      max(proj_w * ring_squash, proj_w * 0.12), proj_w, thin)

    _add_ellipse(scene, depth, centre, _axes(proj_w, half_h), angle, thick, extra=lines)


def _add_chain(scene: _Scene, points, depth, chain, segments, unit: float, thick: int,
               thin: int, *, rings: int, tips_only: bool, ok=lambda a, b: True) -> None:
    pairs = list(zip(chain, chain[1:]))
    for i, ((a, b), (length, r0, r1)) in enumerate(zip(pairs, segments)):
        if not ok(a, b):
            continue
        last = i == len(pairs) - 1
        _add_cylinder(
            scene, (depth[a] + depth[b]) / 2, points[a], points[b], r0 * unit, r1 * unit,
            squash=_foreshortening(points[a], points[b], length * unit),
            near_is_end=bool(depth[b] < depth[a]), thick=thick, thin=thin,
            # In tips_only mode the base of the chain still gets its cap: that
            # line is the finger's root on the palm, shared with the palm outline.
            cap_start=i == 0 or not tips_only, cap_end=last or not tips_only, rings=rings,
        )


# --- the renderer -------------------------------------------------------------

class ConstructiveRenderer:
    name = "Constructive Anatomy"

    def __init__(self) -> None:
        self._pose_filter = _Smoother()
        self._hand_filter = _Smoother()

    def render(
        self, result: PerceptionResult, target: Target, size: tuple[int, int]
    ) -> np.ndarray:
        canvas = blank_paper(size)
        scene = _Scene()
        if target is Target.HAND:
            self._build_hand(scene, result, size)
        else:
            self._build_body(scene, result, size)
        scene.paint(canvas)
        return canvas

    # -- full body ---------------------------------------------------------

    def _build_body(self, scene: _Scene, result: PerceptionResult, size: tuple[int, int]) -> None:
        if result.pose is None:
            return
        pose = self._pose_filter.update(result.timestamp_ms, result.pose)
        points = scaled_points(pose, size)
        depth = pose[:, 2] * size[0]  # z is scaled like x, so width converts it to pixels
        H = _head_unit(points, result)
        if H < 3:
            return
        points = _correct_proportions(points, _BODY_CHAINS, H, _BODY_FOLLOWERS)
        thick = max(2, int(round(H * 0.035)))
        thin = max(1, thick // 2)
        seen = lambda *idx: visible(result, *idx)  # noqa: E731

        shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
        if seen(L_HIP, R_HIP):
            hips = (points[L_HIP] + points[R_HIP]) / 2
            self._torso(scene, points, depth, shoulders, hips, H, thick, thin)
            up = shoulders - hips
        else:
            up = np.array([0.0, -1.0], dtype=np.float32)
        up = up / (np.linalg.norm(up) or 1.0)

        self._head(scene, points, depth, result, shoulders, up, H, thick, thin)

        for chain, segments in _BODY_CHAINS:
            _add_chain(scene, points, depth, chain, segments, H, thick, thin,
                       rings=1, tips_only=False, ok=seen)

        # Hands as mittens, feet as wedges: enough to show where the limb ends and
        # which way the extremity points, without competing with hand mode.
        for wrist, pinky, index in ((L_WRIST, L_PINKY, L_INDEX), (R_WRIST, R_PINKY, R_INDEX)):
            if seen(wrist, pinky, index):
                tip = (points[pinky] + points[index]) / 2
                along = tip - points[wrist]
                length = float(np.linalg.norm(along))
                # A hand pointed at the camera is already told by the full wrist
                # cap; a mitten on top of it just stacks a third circle there.
                if length > H * 0.25:
                    _add_ellipse(scene, depth[wrist], (points[wrist] + tip) / 2,
                                 _axes(length / 2, H * 0.2),
                                 math.degrees(math.atan2(float(along[1]), float(along[0]))), thick)
        for ankle, heel, toe in ((L_ANKLE, L_HEEL, L_FOOT_INDEX), (R_ANKLE, R_HEEL, R_FOOT_INDEX)):
            if seen(ankle, heel, toe):
                _add_polygon(scene, depth[ankle],
                             np.array([points[ankle], points[heel], points[toe]]), thick)

        for idx, r in _BODY_JOINTS.items():
            if seen(idx):
                _add_sphere(scene, depth[idx], points[idx], r * H, thick)

    def _torso(self, scene, points, depth, shoulders, hips, H, thick, thin) -> None:
        spine = hips - shoulders
        length = float(np.linalg.norm(spine)) or 1.0
        up = -spine / length
        right = np.array([-up[1], up[0]], dtype=np.float32)

        def facing(a: int, b: int) -> float:
            # The front turns toward the nearer of the two joints. sin(yaw) is
            # positive toward b, then re-signed onto the torso's `right` axis.
            ab = points[b] - points[a]
            d = math.hypot(float(np.linalg.norm(ab)), float(depth[b] - depth[a])) or 1.0
            toward_b = float(depth[a] - depth[b]) / d
            return toward_b * (1.0 if float(np.dot(ab, right)) >= 0 else -1.0)

        # Pitch: leaning toward the camera tilts the horizontal cross-sections open.
        z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
        z_hips = float(depth[L_HIP] + depth[R_HIP]) / 2
        dz = z_hips - z_shoulders
        pitch = abs(dz) / math.hypot(length, dz)
        ring_toward = -1.0 if dz > 0 else 1.0  # visible half faces away from the near end

        # Ribcage and pelvis as separate masses; the waist between them is where
        # the body twists, which is the point of drawing them apart.
        _add_ellipsoid(scene, z_shoulders, shoulders + spine * 0.34, up, H * 0.72, H * 0.85,
                       H * 0.55, facing(L_SHOULDER, R_SHOULDER), thick, thin,
                       ring_at=-0.6, ring_squash=pitch, ring_toward=ring_toward)
        _add_ellipsoid(scene, z_hips, shoulders + spine * 0.86, up, H * 0.60, H * 0.45,
                       H * 0.50, facing(L_HIP, R_HIP), thick, thin,
                       ring_at=0.0, ring_squash=pitch, ring_toward=ring_toward)

    def _head(self, scene, points, depth, result, shoulders, up, H, thick, thin) -> None:
        if visible(result, L_EAR, R_EAR):
            centre = (points[L_EAR] + points[R_EAR]) / 2
            z = float(depth[L_EAR] + depth[R_EAR]) / 2
        elif visible(result, NOSE):
            centre = points[NOSE] + up * H * 0.15
            z = float(depth[NOSE])
        else:
            return
        right = np.array([-up[1], up[0]], dtype=np.float32)
        half_w, half_h, depth_r = H * 0.37, H * 0.5, H * 0.45

        # The nose fixes the facial centre line directly, no yaw estimate needed.
        offset = float(np.dot(points[NOSE] - centre, right)) if visible(result, NOSE) else 0.0
        facing = max(-1.0, min(1.0, offset / depth_r))
        proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
        angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
        side = 1.0 if offset >= 0 else -1.0
        eye_level = (float(np.dot((points[L_EYE] + points[R_EYE]) / 2 - centre, up))
                     if visible(result, L_EYE, R_EYE) else None)

        def lines(c):
            _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin)
            if eye_level is not None:
                _half_ellipse(c, centre + up * eye_level, -up, proj_w * 0.18, proj_w * 0.95, thin)

        _add_ellipse(scene, z, centre, _axes(proj_w, half_h), angle, thick, extra=lines)

        # Neck: a short cylinder from under the skull to the shoulder line. A head
        # further than 1.5 H from the shoulders is a partial or mis-detected pose,
        # and a long neck would only advertise it.
        if np.linalg.norm(centre - shoulders) <= H * 1.5:
            z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
            # Just behind both, so the skull and the ribcage cover its ends.
            _add_cylinder(scene, max(z, z_shoulders) + 1.0, centre - up * half_h * 0.9,
                          shoulders, H * 0.2, H * 0.22, squash=0.0, near_is_end=False,
                          thick=thick, thin=thin, cap_start=False, cap_end=False)

    # -- hand ----------------------------------------------------------------

    def _build_hand(self, scene: _Scene, result: PerceptionResult, size: tuple[int, int]) -> None:
        if result.hand is None:
            return
        hand = self._hand_filter.update(result.timestamp_ms, result.hand)
        points = scaled_points(hand, size, fit=0.12)
        depth = hand[:, 2]
        # Palm length, measured after the fit-to-canvas zoom. Width is the fallback
        # for a palm seen edge-on.
        P = max(float(np.linalg.norm(points[0] - points[9])),
                float(np.linalg.norm(points[5] - points[17])) / PALM_WIDTH_P)
        if P < 3:
            return
        points = _correct_proportions(points, _HAND_CHAINS, P)
        thick = max(2, int(round(P * 0.022)))
        thin = max(1, thick // 2)

        # The palm's top edge is the row of finger bases: each finger's proximal
        # cylinder starts on a segment across its MCP, and those segments, joined
        # by short webs, are the palm's upper outline. So the palm's corners and
        # the fingers' base corners are the same points.
        across = points[17] - points[5]
        across /= np.linalg.norm(across) or 1.0
        top: list[np.ndarray] = []
        for chain, segments in _FINGER_CHAINS:
            base, nxt = points[chain[0]], points[chain[1]]
            axis = nxt - base
            axis = axis / (np.linalg.norm(axis) or 1.0)
            normal = np.array([-axis[1], axis[0]], dtype=np.float32) * segments[0][1] * P
            top += sorted((base + normal, base - normal), key=lambda p: float(np.dot(p, across)))
        wrist_index = points[0] - across * P * 0.28
        wrist_pinky = points[0] + across * P * 0.28
        palm = np.array([wrist_index, *top, wrist_pinky])
        mid_a, mid_b = (wrist_index + top[0]) / 2, (wrist_pinky + top[-1]) / 2
        _add_polygon(scene, float(np.mean(depth[[0, 5, 9, 13, 17]])), palm, thick,
                     extra=lambda c: cv2.line(c, _pt(mid_a), _pt(mid_b), INK, thin, cv2.LINE_AA))

        for chain, segments in _HAND_CHAINS:
            _add_chain(scene, points, depth, chain, segments, P, thick, thin,
                       rings=0, tips_only=True)
            # Spheres at every finger joint, the reference-drawing convention.
            for idx, (_, r0, _) in zip(chain[1:-1], segments[1:]):
                _add_sphere(scene, depth[idx], points[idx], r0 * P * 1.05, thin)
