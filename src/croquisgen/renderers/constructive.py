"""Constructive anatomy - the body as spheres, cylinders and boxes.

Owner: 김민재.

This one needs no learned model. It is geometry: joint coordinates decide where
each primitive sits, how large it is and which way it points. What is here is the
2D first pass the plan calls for - ellipses for head and joints, an ellipse for
the ribcage and pelvis, tapered cylinders for limbs.

Two things are deliberately left for the next pass:

- Depth. PerceptionResult.pose_world holds metric 3D coordinates, so cylinder
  cross-sections can become ellipses that narrow as a limb turns away from the
  camera. The z axis is noticeably noisier than x and y, so it needs heavier
  temporal smoothing than the 2D coordinates before it is usable.
- Proportion correction. Limb lengths should be checked against canonical body
  ratios so a bad detection does not produce a forearm longer than an upper arm.
  The reference image is meant to be anatomically trustworthy, which means
  rejecting implausible skeletons rather than drawing them.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..types import (
    INDEX_MCP,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    PINKY_MCP,
    PerceptionResult,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    Target,
    WRIST,
)
from .base import blank_paper, body_scale, scaled_points, visible

# (start, end, start_radius, end_radius) as fractions of shoulder span. Limbs are
# thicker at the joint they hang from and thinner at the far end.
_LIMBS = (
    (L_SHOULDER, L_ELBOW, 0.17, 0.13),
    (L_ELBOW, L_WRIST, 0.13, 0.09),
    (R_SHOULDER, R_ELBOW, 0.17, 0.13),
    (R_ELBOW, R_WRIST, 0.13, 0.09),
    (L_HIP, L_KNEE, 0.21, 0.15),
    (L_KNEE, L_ANKLE, 0.15, 0.10),
    (R_HIP, R_KNEE, 0.21, 0.15),
    (R_KNEE, R_ANKLE, 0.15, 0.10),
)

_JOINTS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
           L_HIP, R_HIP, L_KNEE, R_KNEE)


def _cylinder(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
    thickness: int,
) -> None:
    """A limb as a tapered tube: two side walls plus an elliptical end cap.

    The end cap is what makes it read as a 3D form rather than a flat shape - it
    is the cue that tells the viewer which way the cylinder is facing.
    """
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)

    for sign in (1, -1):
        p0 = (start + normal * r_start * sign).astype(int)
        p1 = (end + normal * r_end * sign).astype(int)
        cv2.line(canvas, tuple(p0), tuple(p1), config.INK_COLOR, thickness, cv2.LINE_AA)

    angle = float(np.degrees(np.arctan2(unit[1], unit[0])))
    # Minor axis is a guess at foreshortening until 3D landmarks drive it.
    cv2.ellipse(
        canvas,
        tuple(end.astype(int)),
        (int(r_end), int(r_end * 0.42)),
        angle + 90,
        0,
        360,
        config.INK_COLOR,
        thickness,
        cv2.LINE_AA,
    )


class ConstructiveRenderer:
    name = "Constructive Anatomy"

    def render(
        self, result: PerceptionResult, target: Target, size: tuple[int, int]
    ) -> np.ndarray:
        canvas = blank_paper(size)
        if target is Target.HAND:
            self._draw_hand(canvas, result, size)
        else:
            self._draw_body(canvas, result, size)
        return canvas

    def _draw_body(
        self, canvas: np.ndarray, result: PerceptionResult, size: tuple[int, int]
    ) -> None:
        if result.pose is None:
            return

        points = scaled_points(result.pose, size)
        scale = body_scale(points)
        thickness = max(2, int(scale * 0.045))

        if visible(result, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            self._torso(canvas, points, scale, thickness)

        for a, b, r0, r1 in _LIMBS:
            if not visible(result, a, b):
                continue
            _cylinder(canvas, points[a], points[b], scale * r0, scale * r1, thickness)

        # Joints as spheres, the pivots the forms rotate around.
        for idx in _JOINTS:
            if not visible(result, idx):
                continue
            cv2.circle(
                canvas,
                tuple(points[idx].astype(int)),
                max(2, int(scale * 0.07)),
                config.INK_COLOR,
                thickness,
                cv2.LINE_AA,
            )

        head = points[NOSE]
        cv2.ellipse(
            canvas,
            tuple(head.astype(int)),
            (int(scale * 0.26), int(scale * 0.33)),
            0, 0, 360,
            config.INK_COLOR, thickness, cv2.LINE_AA,
        )
        # Centre line across the face: the standard cue for which way a head turns.
        cv2.line(
            canvas,
            tuple((head - np.array([0, scale * 0.33])).astype(int)),
            tuple((head + np.array([0, scale * 0.33])).astype(int)),
            config.INK_COLOR, max(1, thickness // 2), cv2.LINE_AA,
        )

    def _torso(
        self, canvas: np.ndarray, points: np.ndarray, scale: float, thickness: int
    ) -> None:
        shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
        hips = (points[L_HIP] + points[R_HIP]) / 2
        spine = hips - shoulders
        angle = float(np.degrees(np.arctan2(spine[1], spine[0]))) - 90

        # Ribcage and pelvis as separate masses. Keeping them separate is the point
        # of the exercise: the waist between them is where the body twists.
        ribcage = shoulders + spine * 0.28
        pelvis = shoulders + spine * 0.85
        cv2.ellipse(
            canvas, tuple(ribcage.astype(int)),
            (int(scale * 0.52), int(scale * 0.42)), angle, 0, 360,
            config.INK_COLOR, thickness, cv2.LINE_AA,
        )
        cv2.ellipse(
            canvas, tuple(pelvis.astype(int)),
            (int(scale * 0.44), int(scale * 0.30)), angle, 0, 360,
            config.INK_COLOR, thickness, cv2.LINE_AA,
        )
        cv2.line(
            canvas, tuple(shoulders.astype(int)), tuple(hips.astype(int)),
            config.INK_COLOR, max(1, thickness // 2), cv2.LINE_AA,
        )

    def _draw_hand(
        self, canvas: np.ndarray, result: PerceptionResult, size: tuple[int, int]
    ) -> None:
        if result.hand is None:
            return
        points = scaled_points(result.hand, size, fit=0.12)
        # Palm length, measured after the fit-to-canvas zoom.
        span = float(np.linalg.norm(points[WRIST] - points[INDEX_MCP])) or 1.0
        thickness = max(2, int(span * 0.014))

        # Palm as one slab, then fingers as chains of tapering boxes - the standard
        # way hands are taught, and the structure generative models usually miss.
        palm = np.array([points[WRIST], points[INDEX_MCP], points[PINKY_MCP]], dtype=np.int32)
        cv2.polylines(canvas, [palm], True, config.INK_COLOR, thickness, cv2.LINE_AA)

        for base in (1, 5, 9, 13, 17):
            chain = range(base, base + 4)
            radii = np.linspace(span * 0.10, span * 0.06, 4)
            for i, (a, b) in enumerate(zip(chain, list(chain)[1:])):
                _cylinder(
                    canvas, points[a], points[b],
                    float(radii[i]), float(radii[i + 1]), thickness,
                )
