"""Gesture croquis - the flow of the pose in a few loose lines.

Owner: 박준현.

What is here now is a geometric placeholder, not the planned model. It draws the
line of action and tapered limb strokes directly from landmarks, which is enough
for the rest of the app to run end to end and gives the model something to be
compared against.

The planned replacement is an unpaired skeleton -> croquis model trained
CycleGAN-style, since no paired dataset exists. When it lands, it only has to keep
the render() signature below; everything it needs is already in PerceptionResult
(2D landmarks, visibility, metric 3D landmarks, segmentation mask).

Dataset preparation runs offline and is not bound by the 30 ms frame budget, so it
is free to use heavier pose models than the one driving the live app. Note that
neither MediaPipe nor OpenPose extracts reliable skeletons from line art, so
pulling skeletons out of scraped croquis images is the risky half of the data plan
- rendered 3D poses are the dependable source.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..types import (
    HAND_EDGES,
    MIDDLE_TIP,
    WRIST,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    PerceptionResult,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    Target,
)
from .base import blank_paper, body_scale, scaled_points, tapered_line, visible
from .outline import draw_silhouette

# Limb chains drawn as single continuous strokes. Drawing a limb as one stroke
# rather than two segments is what gives it flow.
_LIMB_CHAINS = (
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
)


class GestureCroquisRenderer:
    name = "Gesture Croquis"

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

        draw_silhouette(canvas, result.segmentation, thickness=2, alpha=0.25)

        points = scaled_points(result.pose, size)
        scale = body_scale(points)

        # Line of action: head -> chest -> pelvis, the curve the whole pose hangs
        # off. Drawn heaviest because it is the thing a croquis is actually about.
        chest = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
        pelvis = (points[L_HIP] + points[R_HIP]) / 2
        if visible(result, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            tapered_line(canvas, points[NOSE], chest, scale * 0.10, scale * 0.16)
            tapered_line(canvas, chest, pelvis, scale * 0.16, scale * 0.11)

        # Limbs taper toward the extremities, the way a brush empties out.
        for chain in _LIMB_CHAINS:
            if not visible(result, *chain):
                continue
            widths = np.linspace(scale * 0.11, scale * 0.04, len(chain))
            for i in range(len(chain) - 1):
                tapered_line(
                    canvas,
                    points[chain[i]],
                    points[chain[i + 1]],
                    float(widths[i]),
                    float(widths[i + 1]),
                )

        head_radius = int(scale * 0.30)
        if head_radius > 1:
            cv2.circle(
                canvas,
                tuple(points[NOSE].astype(int)),
                head_radius,
                config.INK_COLOR,
                max(2, int(scale * 0.05)),
                lineType=cv2.LINE_AA,
            )

    def _draw_hand(
        self, canvas: np.ndarray, result: PerceptionResult, size: tuple[int, int]
    ) -> None:
        if result.hand is None:
            return
        points = scaled_points(result.hand, size, fit=0.12)
        # Hand length, measured after the fit-to-canvas zoom.
        span = float(np.linalg.norm(points[WRIST] - points[MIDDLE_TIP])) or 1.0
        for a, b in HAND_EDGES:
            tapered_line(canvas, points[a], points[b], span * 0.026, span * 0.015)
