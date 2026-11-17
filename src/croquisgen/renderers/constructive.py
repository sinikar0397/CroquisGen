<<<<<<< HEAD
# 인체 도형화 (Constructive Anatomy) 렌더러 - 3802 김민재
# 관절 좌표만 가지고 머리/몸통은 타원체, 관절은 구, 팔다리는 원기둥으로 그린다.
# 학습 모델 없음. 랜드마크 번호는 MediaPipe pose(33점)/hand(21점) 기준.
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
=======
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
4. 렌더링 - outlines plus the construction lines artists actually use: end caps
   whose roundness follows the foreshortening, drawn in full at the near end and
   as a half ring at the far end; cross-contour rings along the limbs; centre
   lines on head and torso that shift toward the side facing the camera.

Depth comes from the landmark z, which MediaPipe scales like x. It decides only
which end of a limb is nearer and which way a mass faces, so its noise is far
less visible than if it drove the geometry directly.
"""
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
from __future__ import annotations

import math

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

<<<<<<< HEAD
INK = config.INK_COLOR
PAPER = config.PAPER_COLOR
GHOST = tuple(int(p * 0.68 + i * 0.32) for p, i in zip(PAPER, INK))  #가려진 선용 연한 잉크

# 비율표. 단위 H = 머리 높이 (8등신 기준)
TORSO_H = 2.5  #어깨 중심 -> 골반 중심
SHOULDER_SPAN_H = 1.8
LENGTH_TOLERANCE = 1.15  #표준 길이의 이 배수를 넘으면 검출 오류로 봄

# (관절 번호들), ((길이, 시작 반지름, 끝 반지름), ...)
=======
# --- canonical proportions, in head heights H (8-head figure) ----------------
TORSO_H = 2.5           # shoulder centre -> hip centre
SHOULDER_SPAN_H = 1.8
# Projected lengths above canonical x this are treated as detection errors.
LENGTH_TOLERANCE = 1.15

# A chain is (joint indices, per-segment (length, proximal radius, distal radius)).
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
_ARM = ((1.5, 0.22, 0.16), (1.25, 0.16, 0.11))
_LEG = ((2.0, 0.34, 0.21), (1.9, 0.22, 0.13))
_BODY_CHAINS = (
    ((L_SHOULDER, L_ELBOW, L_WRIST), _ARM),
    ((R_SHOULDER, R_ELBOW, R_WRIST), _ARM),
    ((L_HIP, L_KNEE, L_ANKLE), _LEG),
    ((R_HIP, R_KNEE, R_ANKLE), _LEG),
<<<<<<< HEAD
=======
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
_HAND_CHAINS = (
    ((1, 2, 3, 4), ((0.45, 0.14, 0.12), (0.33, 0.12, 0.10), (0.27, 0.10, 0.085))),
    ((5, 6, 7, 8), _FINGER),
    ((9, 10, 11, 12), ((0.46, 0.10, 0.085), (0.30, 0.085, 0.075), (0.24, 0.075, 0.065))),
    ((13, 14, 15, 16), _FINGER),
    ((17, 18, 19, 20), ((0.34, 0.09, 0.075), (0.22, 0.075, 0.065), (0.20, 0.065, 0.055))),
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
)
_BODY_FOLLOWERS = {  #관절을 보정하면 같이 따라 움직이는 점들
    L_WRIST: (L_PINKY, L_INDEX), R_WRIST: (R_PINKY, R_INDEX),
    L_ANKLE: (L_HEEL, L_FOOT_INDEX), R_ANKLE: (R_HEEL, R_FOOT_INDEX),
}
_BODY_JOINTS = {  #관절 구 반지름
    L_SHOULDER: 0.24, R_SHOULDER: 0.24, L_ELBOW: 0.17, R_ELBOW: 0.17,
    L_WRIST: 0.12, R_WRIST: 0.12, L_HIP: 0.28, R_HIP: 0.28,
    L_KNEE: 0.23, R_KNEE: 0.23, L_ANKLE: 0.14, R_ANKLE: 0.14,
}

<<<<<<< HEAD
# 손은 단위 P = 손바닥 길이 (손목 -> 중지 MCP)
PALM_WIDTH_P = 0.75  #검지 MCP -> 소지 MCP
_FINGER = ((0.42, 0.10, 0.085), (0.27, 0.085, 0.075), (0.22, 0.075, 0.065))
_THUMB_CHAIN = ((2, 3, 4), ((0.33, 0.12, 0.10), (0.27, 0.10, 0.085)))  #1->2는 손바닥 안이라 뺌
_FINGER_CHAINS = (  #검지부터 소지 순서. 손바닥 윗변을 이 순서로 만든다
    ((5, 6, 7, 8), _FINGER),
    ((9, 10, 11, 12), ((0.46, 0.10, 0.085), (0.30, 0.085, 0.075), (0.24, 0.075, 0.065))),
    ((13, 14, 15, 16), _FINGER),
    ((17, 18, 19, 20), ((0.34, 0.09, 0.075), (0.22, 0.075, 0.065), (0.20, 0.065, 0.055))),
)
_HAND_CHAINS = (_THUMB_CHAIN, *_FINGER_CHAINS)


class _Smoother:  #프레임 간 지수 스무딩. z가 더 튀어서 z만 alpha를 낮게
    def __init__(self, alpha_xy=0.55, alpha_z=0.25, max_gap_ms=500):
        self._alpha = np.array([alpha_xy, alpha_xy, alpha_z], dtype=np.float32)
        self._max_gap = max_gap_ms
        self._ts = 0
        self._points = None

    def update(self, timestamp_ms, points):
        previous = self._points
        # timestamp가 안 늘거나(정지 이미지, 테스트) 너무 벌어지면 리셋
        if (previous is None or previous.shape != points.shape
                or timestamp_ms <= self._ts or timestamp_ms - self._ts > self._max_gap):
            self._points = points.astype(np.float32).copy()
        else:
            self._points = self._alpha * points + (1 - self._alpha) * previous
        self._ts = timestamp_ms
        return self._points


def _head_unit(points, result):  #머리 높이(px) 추정
    # 어깨 폭, 몸통 길이 둘 다 원근으로 줄어들기만 하니까 큰 쪽이 진짜에 가깝다
    # TODO: 몸통이 양 방향으로 다 줄어드는 자세(카메라 쪽으로 숙임)는 pose_world 길이로 바꿔야 함
=======

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
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
    candidates = [0.0]
    if visible(result, L_SHOULDER, R_SHOULDER):
        candidates.append(
            float(np.linalg.norm(points[L_SHOULDER] - points[R_SHOULDER])) / SHOULDER_SPAN_H)
        if visible(result, L_HIP, R_HIP):
            shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
            hips = (points[L_HIP] + points[R_HIP]) / 2
            candidates.append(float(np.linalg.norm(shoulders - hips)) / TORSO_H)
    return max(candidates)


<<<<<<< HEAD
def _correct_proportions(points, chains, unit, followers=None):
    # 표준 길이보다 긴 마디는 끝 관절을 당겨 넣는다. 짧은 건 원근이니까 그대로 둠.
    # 당긴 만큼 그 아래 관절(과 손발 점)도 같이 이동시켜 다음 마디 모양은 유지
=======
# --- stage 3: proportion correction -----------------------------------------

def _correct_proportions(points: np.ndarray, chains, unit: float,
                         followers: dict | None = None) -> np.ndarray:
    """Pull in every segment that projects longer than its canonical length.

    Joints further down the chain move by the same offset, so the next segment
    keeps its own direction and length, and `followers` (hand and foot points)
    move with the joint they hang from.
    """
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
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


<<<<<<< HEAD
def _foreshortening(start, end, canonical):
    # 투영 길이 / 표준 길이로 |cos| 계산. 0 = 화면에 누움(캡이 납작), 1 = 카메라를 향함(캡이 원)
=======
def _foreshortening(start: np.ndarray, end: np.ndarray, canonical: float) -> float:
    """|cos| of the angle between the limb and the image plane, in [0, 1].

    0 = lying in the picture plane (end caps are flat lines), 1 = pointing at the
    camera (end caps are full circles). Comes from the projected / canonical
    length ratio, which is steadier than the z coordinate.
    """
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
    if canonical <= 0:
        return 0.0
    ratio = min(1.0, float(np.linalg.norm(end - start)) / canonical)
    return math.sqrt(1.0 - ratio * ratio)


<<<<<<< HEAD
class _Scene:
    # 도형을 (깊이, 채우기, 외곽선)으로 모아뒀다가 먼 것부터 그린다 (painter's algorithm).
    # 채우기는 종이색이라 가까운 도형이 뒤 선을 지움. 지워진 선은 두 번째 패스에서 연하게 다시 그림.
    def __init__(self):
        self._forms = []

    def add(self, depth, fill, stroke):  #stroke(canvas, ink, ghost) - ghost면 가려진 반쪽까지 다 그림
        self._forms.append((float(depth), fill, stroke))

    def paint(self, canvas):
        ordered = sorted(self._forms, key=lambda f: -f[0])
        for _, fill, stroke in ordered:
            fill(canvas)
            stroke(canvas, INK, False)
        ghost = np.empty_like(canvas)
        ghost[:] = PAPER
        for _, _, stroke in ordered:
            stroke(ghost, GHOST, True)
        np.minimum(canvas, ghost, out=canvas)  #어두운 쪽이 이김 -> 보이는 선은 진하게, 가려진 선은 연하게


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def _axes(a, b):
    return max(1, int(round(a))), max(1, int(round(b)))


def _half_ellipse(canvas, centre, along, half_along, half_across, thickness, ink=INK, full=False):
    # 첫 축을 along 방향으로 둔 타원. full이 아니면 +along 쪽 절반만 (캡, 단면선, 중심선 전부 이걸로)
    angle = math.degrees(math.atan2(float(along[1]), float(along[0])))
    cv2.ellipse(canvas, _pt(centre), _axes(half_along, half_across), angle,
                0 if full else -90, 360 if full else 90, ink, thickness, cv2.LINE_AA)


def _add_cylinder(scene, depth, start, end, r_start, r_end, squash, near_is_end, thick, thin, *,
                  cap_start=True, cap_end=True, rings=0):
    # 끝이 가늘어지는 원기둥. 가까운 끝 캡은 온전한 타원, 먼 끝 캡과 중간 단면선은 바깥쪽 절반만
=======
# --- stage 4: drawing primitives ---------------------------------------------

def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _half_ellipse(canvas, centre, along, half_along: float, half_across: float,
                  thickness: int, full: bool = False) -> None:
    """Ellipse with its first axis on `along`; only the half displaced toward
    +along unless `full`. Every ring, cap and centre line is one of these."""
    angle = math.degrees(math.atan2(float(along[1]), float(along[0])))
    axes = (max(1, int(round(half_along))), max(1, int(round(half_across))))
    cv2.ellipse(canvas, _pt(centre), axes, angle, 0 if full else -90, 360 if full else 90,
                config.INK_COLOR, thickness, cv2.LINE_AA)


def _cylinder(canvas, start, end, r_start: float, r_end: float, squash: float,
              near_is_end: bool, thick: int, thin: int, *,
              cap_start: bool = True, cap_end: bool = True, rings: int = 0) -> None:
    """A tapered tube with the cues that make it read as a solid.

    The cap at the near end is a full ellipse; the far cap and any mid rings show
    only the half displaced away from the near end, the half that is actually in
    view on a solid. `squash` is the cap's minor/major ratio, i.e. foreshortening.
    """
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        return
    axis = direction / length
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
<<<<<<< HEAD
    toward_far = -axis if near_is_end else axis
    quad = np.array([start + normal * r_start, end + normal * r_end,
                     end - normal * r_end, start - normal * r_start], dtype=np.int32)
    cap_angle = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) + 90

    def fill(c):
        cv2.fillConvexPoly(c, quad, PAPER, cv2.LINE_8)
        for centre, r in ((start, r_start), (end, r_end)):
            cv2.ellipse(c, _pt(centre), _axes(r * squash, r), cap_angle, 0, 360, PAPER, -1, cv2.LINE_8)

    def stroke(c, ink, ghost):
        for sign in (1, -1):
            cv2.line(c, _pt(start + normal * r_start * sign), _pt(end + normal * r_end * sign),
                     ink, thick, cv2.LINE_AA)
        if cap_start:
            _half_ellipse(c, start, toward_far, r_start * squash, r_start, thick, ink,
                          full=ghost or not near_is_end)
        if cap_end:
            _half_ellipse(c, end, toward_far, r_end * squash, r_end, thick, ink,
                          full=ghost or near_is_end)
        for k in range(1, rings + 1):
            t = k / (rings + 1)
            r = r_start + (r_end - r_start) * t
            _half_ellipse(c, start + direction * t, toward_far, r * squash, r, thin, ink,
                          full=ghost)

    scene.add(depth, fill, stroke)


def _add_sphere(scene, depth, centre, r, thickness):
    radius = max(2, int(r))
    # 반지름만큼 뒤로 밀어서 원기둥이 구 위에 칠해지게. 튀어나온 부분만 진한 선으로 남는다
    scene.add(depth + r,
              lambda c: cv2.circle(c, _pt(centre), radius, PAPER, -1, cv2.LINE_8),
              lambda c, ink, ghost: cv2.circle(c, _pt(centre), radius, ink, thickness, cv2.LINE_AA))


def _add_ellipse(scene, depth, centre, axes, angle, thick, extra=None):
    def stroke(c, ink, ghost):
        cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, ink, thick, cv2.LINE_AA)
        if extra:
            extra(c, ink, ghost)

    scene.add(depth,
              lambda c: cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, PAPER, -1, cv2.LINE_8),
              stroke)


def _add_polygon(scene, depth, pts, thick, extra=None):
    poly = pts.astype(np.int32)

    def stroke(c, ink, ghost):
        cv2.polylines(c, [poly], True, ink, thick, cv2.LINE_AA)
        if extra:
            extra(c, ink, ghost)

    scene.add(depth, lambda c: cv2.fillPoly(c, [poly], PAPER, cv2.LINE_8), stroke)


def _add_ellipsoid(scene, depth, centre, up, half_w, half_h, depth_r, facing, thick, thin,
                   ring_at, ring_squash, ring_toward):
    # 타원체 = 윤곽 + 중심선(카메라 향한 쪽으로 쏠림) + 가로 단면선 하나
    # facing = right 방향 sin(yaw). 돌아선 만큼 폭이 depth_r 쪽으로 바뀌고 중심선이 옆으로 간다
    right = np.array([-up[1], up[0]], dtype=np.float32)
    facing = max(-1.0, min(1.0, facing))
    proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
    angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
    side = 1.0 if facing >= 0 else -1.0

    def lines(c, ink, ghost):
        _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin, ink,
                      full=ghost)
        _half_ellipse(c, centre + up * (half_h * ring_at), up * ring_toward,
                      max(proj_w * ring_squash, proj_w * 0.12), proj_w, thin, ink, full=ghost)

    _add_ellipse(scene, depth, centre, _axes(proj_w, half_h), angle, thick, extra=lines)


def _base_edge(points, chain, r, order):  #마디 첫 원기둥의 밑동 양 끝점. order 방향으로 정렬
    base = points[chain[0]]
    axis = points[chain[1]] - base
    axis = axis / (np.linalg.norm(axis) or 1.0)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32) * r
    return sorted((base + normal, base - normal), key=lambda p: float(np.dot(p, order)))


def _palm_outline(points, P):
    # 손바닥 다각형: 손목 모서리 -> (엄지 밑동) -> 손가락 밑동들 -> 손목 모서리
    # 윗변을 손가락 밑동 선분으로 만들어서 손바닥 꼭짓점 = 손가락 밑동 꼭짓점이 되게 함
    across = points[17] - points[5]
    across = across / (np.linalg.norm(across) or 1.0)
    top = []
    for chain, segments in _FINGER_CHAINS:
        top += _base_edge(points, chain, segments[0][1] * P, across)
    wrist_index = points[0] - across * P * 0.28
    wrist_pinky = points[0] + across * P * 0.28

    edge = top[0] - wrist_index  #검지 쪽 가장자리
    def side(p):  #edge 기준 어느 쪽인지 (외적 부호)
        d = p - wrist_index
        return float(edge[0] * d[1] - edge[1] * d[0])
    centre = points[[0, 5, 9, 13, 17]].mean(axis=0)
    # 엄지가 손바닥 안으로 접히면 MCP가 안쪽에 들어와서 다각형이 꼬임 -> 그땐 윤곽에서 빼고 위에 얹기만
    thumb_out = side(points[_THUMB_CHAIN[0][0]]) * side(centre) < 0
    thumb = (_base_edge(points, _THUMB_CHAIN[0], _THUMB_CHAIN[1][0][1] * P, edge)
             if thumb_out else [])
    return np.array([wrist_index, *thumb, *top, wrist_pinky])


def _add_chain(scene, points, depth, chain, segments, unit, thick, thin, *,
               rings, tips_only, ok=lambda a, b: True):
    pairs = list(zip(chain, chain[1:]))
    for i, ((a, b), (length, r0, r1)) in enumerate(zip(pairs, segments)):
        if not ok(a, b):
            continue
        last = i == len(pairs) - 1
        _add_cylinder(
            scene, (depth[a] + depth[b]) / 2, points[a], points[b], r0 * unit, r1 * unit,
            squash=_foreshortening(points[a], points[b], length * unit),
            near_is_end=bool(depth[b] < depth[a]), thick=thick, thin=thin,
            # tips_only(손)여도 첫 마디 밑동 캡은 그림 - 손바닥 윗변과 겹치는 그 선
            cap_start=i == 0 or not tips_only, cap_end=last or not tips_only, rings=rings,
        )
=======
    for sign in (1, -1):
        cv2.line(canvas, _pt(start + normal * r_start * sign), _pt(end + normal * r_end * sign),
                 config.INK_COLOR, thick, cv2.LINE_AA)

    toward_far = -axis if near_is_end else axis
    if cap_start:
        _half_ellipse(canvas, start, toward_far, r_start * squash, r_start, thick,
                      full=not near_is_end)
    if cap_end:
        _half_ellipse(canvas, end, toward_far, r_end * squash, r_end, thick, full=near_is_end)
    for k in range(1, rings + 1):
        t = k / (rings + 1)
        r = r_start + (r_end - r_start) * t
        _half_ellipse(canvas, start + direction * t, toward_far, r * squash, r, thin)
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)


def _draw_chain(canvas, points, depth, chain, segments, unit: float, thick: int, thin: int,
                *, rings: int, tips_only: bool, ok=lambda a, b: True) -> None:
    pairs = list(zip(chain, chain[1:]))
    for i, ((a, b), (length, r0, r1)) in enumerate(zip(pairs, segments)):
        if not ok(a, b):
            continue
        last = i == len(pairs) - 1
        _cylinder(
            canvas, points[a], points[b], r0 * unit, r1 * unit,
            squash=_foreshortening(points[a], points[b], length * unit),
            near_is_end=bool(depth[b] < depth[a]), thick=thick, thin=thin,
            cap_start=not tips_only, cap_end=last or not tips_only, rings=rings,
        )


def _mass(canvas, centre, up, half_w: float, half_h: float, depth: float, facing: float,
          thick: int, thin: int, ring_at: float, ring_squash: float, ring_toward: float) -> None:
    """An ellipsoid: outline, a meridian centre line, and one latitude ring.

    `facing` is sin(yaw) along `right`; it widens the silhouette by the ellipsoid's
    depth and shifts the centre line toward the side turned to the camera.
    `ring_at` places the ring along `up` (fraction of half_h), `ring_squash` is its
    minor/major ratio, `ring_toward` the sign of `up` its visible half faces.
    """
    right = np.array([-up[1], up[0]], dtype=np.float32)
    facing = max(-1.0, min(1.0, facing))
    proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth ** 2 * facing ** 2)
    angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
    cv2.ellipse(canvas, _pt(centre), (max(1, int(proj_w)), max(1, int(half_h))), angle,
                0, 360, config.INK_COLOR, thick, cv2.LINE_AA)
    side = 1.0 if facing >= 0 else -1.0
    _half_ellipse(canvas, centre, right * side, abs(facing) * depth, half_h, thin)
    _half_ellipse(canvas, centre + up * (half_h * ring_at), up * ring_toward,
                  max(proj_w * ring_squash, proj_w * 0.12), proj_w, thin)


# --- the renderer -------------------------------------------------------------

class ConstructiveRenderer:
    name = "Constructive Anatomy"

<<<<<<< HEAD
    def __init__(self):
        self._pose_filter = _Smoother()
        self._hand_filter = _Smoother()

    def render(self, result: PerceptionResult, target: Target, size: tuple[int, int]) -> np.ndarray:
=======
    def __init__(self) -> None:
        self._pose_filter = _Smoother()
        self._hand_filter = _Smoother()

    def render(
        self, result: PerceptionResult, target: Target, size: tuple[int, int]
    ) -> np.ndarray:
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        canvas = blank_paper(size)
        scene = _Scene()
        if target is Target.HAND:
            self._build_hand(scene, result, size)
        else:
            self._build_body(scene, result, size)
        scene.paint(canvas)
        return canvas

<<<<<<< HEAD
    def _build_body(self, scene, result, size):
=======
    # -- full body ---------------------------------------------------------

    def _draw_body(self, canvas, result: PerceptionResult, size: tuple[int, int]) -> None:
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        if result.pose is None:
            return
        pose = self._pose_filter.update(result.timestamp_ms, result.pose)
        points = scaled_points(pose, size)
<<<<<<< HEAD
        depth = pose[:, 2] * size[0]  #z는 x와 같은 스케일이라 width 곱하면 px
=======
        depth = pose[:, 2] * size[0]  # z is scaled like x, so width converts it to pixels
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
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
<<<<<<< HEAD
            self._torso(scene, points, depth, shoulders, hips, H, thick, thin)
=======
            self._torso(canvas, points, depth, shoulders, hips, H, thick, thin)
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
            up = shoulders - hips
        else:
            up = np.array([0.0, -1.0], dtype=np.float32)
        up = up / (np.linalg.norm(up) or 1.0)

<<<<<<< HEAD
        self._head(scene, points, depth, result, shoulders, up, H, thick, thin)

        for chain, segments in _BODY_CHAINS:
            _add_chain(scene, points, depth, chain, segments, H, thick, thin,
                       rings=1, tips_only=False, ok=seen)

        # 손은 미튼 타원, 발은 쐐기. 손 모드랑 겹치지 않을 만큼만
=======
        self._head(canvas, points, depth, result, shoulders, up, H, thick, thin)

        for chain, segments in _BODY_CHAINS:
            _draw_chain(canvas, points, depth, chain, segments, H, thick, thin,
                        rings=1, tips_only=False, ok=seen)

        # Hands as mittens, feet as wedges: enough to show where the limb ends and
        # which way the extremity points, without competing with hand mode.
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        for wrist, pinky, index in ((L_WRIST, L_PINKY, L_INDEX), (R_WRIST, R_PINKY, R_INDEX)):
            if seen(wrist, pinky, index):
                tip = (points[pinky] + points[index]) / 2
                along = tip - points[wrist]
                length = float(np.linalg.norm(along))
<<<<<<< HEAD
                if length > H * 0.25:  #카메라를 향한 손은 손목 캡이 이미 원이라 미튼까지 그리면 원 세 겹
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

    def _torso(self, scene, points, depth, shoulders, hips, H, thick, thin):
=======
                if length > 1:
                    _half_ellipse(canvas, (points[wrist] + tip) / 2, along / length,
                                  max(length / 2, H * 0.2), H * 0.2, thick, full=True)
        for ankle, heel, toe in ((L_ANKLE, L_HEEL, L_FOOT_INDEX), (R_ANKLE, R_HEEL, R_FOOT_INDEX)):
            if seen(ankle, heel, toe):
                wedge = np.array([points[ankle], points[heel], points[toe]], dtype=np.int32)
                cv2.polylines(canvas, [wedge], True, config.INK_COLOR, thick, cv2.LINE_AA)

        for idx, r in _BODY_JOINTS.items():
            if seen(idx):
                cv2.circle(canvas, _pt(points[idx]), max(2, int(r * H)), config.INK_COLOR,
                           thick, cv2.LINE_AA)

    def _torso(self, canvas, points, depth, shoulders, hips, H, thick, thin) -> None:
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        spine = hips - shoulders
        length = float(np.linalg.norm(spine)) or 1.0
        up = -spine / length
        right = np.array([-up[1], up[0]], dtype=np.float32)

<<<<<<< HEAD
        def facing(a, b):  #앞면은 더 가까운 관절 쪽으로 돈다. sin(yaw)를 right 축 부호로 맞춤
=======
        def facing(a: int, b: int) -> float:
            # The front turns toward the nearer of the two joints. sin(yaw) is
            # positive toward b, then re-signed onto the torso's `right` axis.
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
            ab = points[b] - points[a]
            d = math.hypot(float(np.linalg.norm(ab)), float(depth[b] - depth[a])) or 1.0
            toward_b = float(depth[a] - depth[b]) / d
            return toward_b * (1.0 if float(np.dot(ab, right)) >= 0 else -1.0)

<<<<<<< HEAD
        # 앞뒤로 숙이면 가로 단면이 벌어져 보임
        z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
        z_hips = float(depth[L_HIP] + depth[R_HIP]) / 2
        dz = z_hips - z_shoulders
        pitch = abs(dz) / math.hypot(length, dz)
        ring_toward = -1.0 if dz > 0 else 1.0  #보이는 반쪽은 가까운 끝 반대편

        # 흉곽과 골반을 따로. 허리에서 비틀리는 게 보여야 하니까
        _add_ellipsoid(scene, z_shoulders, shoulders + spine * 0.34, up, H * 0.72, H * 0.85,
                       H * 0.55, facing(L_SHOULDER, R_SHOULDER), thick, thin,
                       ring_at=-0.6, ring_squash=pitch, ring_toward=ring_toward)
        _add_ellipsoid(scene, z_hips, shoulders + spine * 0.86, up, H * 0.60, H * 0.45,
                       H * 0.50, facing(L_HIP, R_HIP), thick, thin,
                       ring_at=0.0, ring_squash=pitch, ring_toward=ring_toward)

    def _head(self, scene, points, depth, result, shoulders, up, H, thick, thin):
        if visible(result, L_EAR, R_EAR):
            centre = (points[L_EAR] + points[R_EAR]) / 2
            z = float(depth[L_EAR] + depth[R_EAR]) / 2
        elif visible(result, NOSE):
            centre = points[NOSE] + up * H * 0.15
            z = float(depth[NOSE])
=======
        # Pitch: leaning toward the camera tilts the horizontal cross-sections open.
        dz = float(depth[L_HIP] + depth[R_HIP] - depth[L_SHOULDER] - depth[R_SHOULDER]) / 2
        pitch = abs(dz) / math.hypot(length, dz)
        ring_toward = -1.0 if dz > 0 else 1.0  # visible half faces away from the near end

        # Ribcage and pelvis as separate masses; the waist between them is where
        # the body twists, which is the point of drawing them apart.
        _mass(canvas, shoulders + spine * 0.34, up, H * 0.72, H * 0.85, H * 0.55,
              facing(L_SHOULDER, R_SHOULDER), thick, thin,
              ring_at=-0.6, ring_squash=pitch, ring_toward=ring_toward)
        _mass(canvas, shoulders + spine * 0.86, up, H * 0.60, H * 0.45, H * 0.50,
              facing(L_HIP, R_HIP), thick, thin,
              ring_at=0.0, ring_squash=pitch, ring_toward=ring_toward)
        cv2.line(canvas, _pt(shoulders), _pt(hips), config.INK_COLOR, thin, cv2.LINE_AA)

    def _head(self, canvas, points, depth, result, shoulders, up, H, thick, thin) -> None:
        if visible(result, L_EAR, R_EAR):
            centre = (points[L_EAR] + points[R_EAR]) / 2
        elif visible(result, NOSE):
            centre = points[NOSE] + up * H * 0.15
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        else:
            return
        right = np.array([-up[1], up[0]], dtype=np.float32)
        half_w, half_h, depth_r = H * 0.37, H * 0.5, H * 0.45

<<<<<<< HEAD
        # 얼굴 중심선은 코 위치로 바로 정함 (yaw 추정 필요 없음)
=======
        # The nose fixes the facial centre line directly, no yaw estimate needed.
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        offset = float(np.dot(points[NOSE] - centre, right)) if visible(result, NOSE) else 0.0
        facing = max(-1.0, min(1.0, offset / depth_r))
        proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
        angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
<<<<<<< HEAD
        side = 1.0 if offset >= 0 else -1.0
        eye_level = (float(np.dot((points[L_EYE] + points[R_EYE]) / 2 - centre, up))
                     if visible(result, L_EYE, R_EYE) else None)

        def lines(c, ink, ghost):
            _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin, ink,
                          full=ghost)
            if eye_level is not None:
                _half_ellipse(c, centre + up * eye_level, -up, proj_w * 0.18, proj_w * 0.95,
                              thin, ink, full=ghost)

        _add_ellipse(scene, z, centre, _axes(proj_w, half_h), angle, thick, extra=lines)

        # 목. 머리가 어깨에서 1.5H 넘게 떨어져 있으면 검출이 이상한 거라 안 그림
        if np.linalg.norm(centre - shoulders) <= H * 1.5:
            z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
            _add_cylinder(scene, max(z, z_shoulders) + 1.0, centre - up * half_h * 0.9,  #머리/흉곽 뒤에
                          shoulders, H * 0.2, H * 0.22, squash=0.0, near_is_end=False,
                          thick=thick, thin=thin, cap_start=False, cap_end=False)

    def _build_hand(self, scene, result, size):
=======
        cv2.ellipse(canvas, _pt(centre), (max(1, int(proj_w)), max(1, int(half_h))), angle,
                    0, 360, config.INK_COLOR, thick, cv2.LINE_AA)
        side = 1.0 if offset >= 0 else -1.0
        _half_ellipse(canvas, centre, right * side, abs(facing) * depth_r, half_h, thin)
        if visible(result, L_EYE, R_EYE):
            level = float(np.dot((points[L_EYE] + points[R_EYE]) / 2 - centre, up))
            _half_ellipse(canvas, centre + up * level, -up, proj_w * 0.18, proj_w * 0.95, thin)

        # Neck: a short cylinder from under the skull to the shoulder line. A head
        # further than 1.5 H from the shoulders is a partial or mis-detected pose,
        # and a long neck would only advertise it.
        if np.linalg.norm(centre - shoulders) <= H * 1.5:
            _cylinder(canvas, centre - up * half_h * 0.9, shoulders, H * 0.2, H * 0.22,
                      squash=0.0, near_is_end=False, thick=thick, thin=thin,
                      cap_start=False, cap_end=False)

    # -- hand ----------------------------------------------------------------

    def _draw_hand(self, canvas, result: PerceptionResult, size: tuple[int, int]) -> None:
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        if result.hand is None:
            return
        hand = self._hand_filter.update(result.timestamp_ms, result.hand)
        points = scaled_points(hand, size, fit=0.12)
        depth = hand[:, 2]
<<<<<<< HEAD
        # 손바닥 길이 (fit 확대 후 기준). 옆에서 본 손은 폭으로 대신
=======
        # Palm length, measured after the fit-to-canvas zoom. Width is the fallback
        # for a palm seen edge-on.
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
        P = max(float(np.linalg.norm(points[0] - points[9])),
                float(np.linalg.norm(points[5] - points[17])) / PALM_WIDTH_P)
        if P < 3:
            return
        points = _correct_proportions(points, _HAND_CHAINS, P)
        thick = max(2, int(round(P * 0.022)))
        thin = max(1, thick // 2)

<<<<<<< HEAD
        palm = _palm_outline(points, P)
        mid_a, mid_b = (palm[0] + palm[-9]) / 2, (palm[-1] + palm[-2]) / 2  #palm[-9]가 검지 밑동 바깥점
        _add_polygon(scene, float(np.mean(depth[[0, 2, 5, 9, 13, 17]])), palm, thick,
                     extra=lambda c, ink, ghost: cv2.line(c, _pt(mid_a), _pt(mid_b), ink, thin,
                                                          cv2.LINE_AA))

        for chain, segments in _HAND_CHAINS:
            _add_chain(scene, points, depth, chain, segments, P, thick, thin,
                       rings=0, tips_only=True)
            for idx, (_, r0, _) in zip(chain[1:-1], segments[1:]):  #마디마다 관절 구
                _add_sphere(scene, depth[idx], points[idx], r0 * P * 1.05, thin)
=======
        # Palm as a slab: knuckle line across the MCPs, two corners at the wrist, and
        # a cross line at mid-palm. The thenar pad is the one round mass on it.
        across = points[5] - points[17]
        across /= np.linalg.norm(across) or 1.0
        wrist_i = points[0] + across * P * 0.28
        wrist_p = points[0] - across * P * 0.28
        slab = np.array([wrist_i, points[5], points[9], points[13], points[17], wrist_p])
        cv2.polylines(canvas, [slab.astype(np.int32)], True, config.INK_COLOR, thick, cv2.LINE_AA)
        cv2.line(canvas, _pt((wrist_i + points[5]) / 2), _pt((wrist_p + points[17]) / 2),
                 config.INK_COLOR, thin, cv2.LINE_AA)
        cv2.circle(canvas, _pt((points[1] + points[2]) / 2), max(2, int(P * 0.2)),
                   config.INK_COLOR, thin, cv2.LINE_AA)

        for chain, segments in _HAND_CHAINS:
            _draw_chain(canvas, points, depth, chain, segments, P, thick, thin,
                        rings=0, tips_only=True)
            # Spheres at every finger joint, the reference-drawing convention.
            for idx, (_, r0, _) in zip(chain[1:-1], segments[1:]):
                cv2.circle(canvas, _pt(points[idx]), max(2, int(r0 * P * 1.2)),
                           config.INK_COLOR, thin, cv2.LINE_AA)
>>>>>>> 0ff5eb7 (인체 도형화(Constructive Anatomy) 렌더러 구현)
