"""Gesture croquis - the flow of the pose in a few loose lines.

Owner: 박준현.

Two engines, picked by config.GESTURE_CROQUIS_ENGINE.

**"s4"** (default) runs this package's real abstraction stage,
s4_gesture.build_croquis, in its 2D mode: candidate line-of-action chains scored
and fitted as cubic Béziers, rhythm lines per limb, torso squash, and the
line-weight model. About 56 ms, which the threaded loop absorbs by letting the
render stage fall behind the camera - roughly 13 croquis a second. See realtime.py
for how a PerceptionResult becomes the PoseFrame it wants, and what 2D mode gives
up (novel views and the depth-ordered weight terms).

**"placeholder"** draws the line of action and tapered limb strokes straight from
the landmarks. Geometric, about 7 ms, and the only one that fits a 30 fps budget
on its own. It is what `--placeholder` selects, what the hand target always
draws, and where a frame S4 cannot handle falls back to.

S4 is still missing S5, the stroke stylisation, and it runs without S3, so the
effort and support terms in the weight model read as zero: the line weights come
from role, taper and 2D compression only.

The planned further replacement is an unpaired skeleton -> croquis model trained
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

from core import config
from core.landmarks import HAND_EDGES, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, MIDDLE_TIP, NOSE, R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, WRIST
from core.types import PerceptionResult, Target
from render.base import blank_paper, body_scale, scaled_points, tapered_line, visible
from render.outline import draw_silhouette

from . import realtime

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
    # No frame-to-frame state, so the render stage may run several of these at
    # once. The constructive renderer smooths across frames and cannot.
    stateless = True

    def __init__(self) -> None:
        # Why the last S4 failure is kept instead of raised: a renderer that
        # throws kills the render thread, and an unusual pose is not a bug worth
        # stopping the app for. Read it when a croquis looks wrong.
        self.last_s4_error: str | None = None

    def render(
        self, result: PerceptionResult, target: Target, size: tuple[int, int]
    ) -> np.ndarray:
        # Read per frame so --s4 can switch engines without rebuilding anything.
        # The hand target has no S4 path: every stage of it is about a body.
        if config.GESTURE_CROQUIS_ENGINE == "s4" and target is not Target.HAND:
            canvas = self._draw_s4(result, size)
            if canvas is not None:
                return canvas

        canvas = blank_paper(size)
        if target is Target.HAND:
            self._draw_hand(canvas, result, size)
        else:
            self._draw_body(canvas, result, size)
        return canvas

    def _draw_s4(
        self, result: PerceptionResult, size: tuple[int, int]
    ) -> np.ndarray | None:
        """build_croquis in 2D mode, rasterised onto paper. None = fall back.

        Imported here rather than at module scope so the placeholder engine
        never pays for loading S4 and its dependencies.
        """
        from . import s4_gesture

        frame = realtime.pose_frame(result, size)
        if frame is None:
            return None  # nobody in the frame: the placeholder returns blank paper
        try:
            croquis = s4_gesture.build_croquis(frame, layers=config.S4_LAYERS)
            # fit=False keeps the strokes in the canvas's own pixels, which is
            # what realtime.pose_frame set img_hw to. supersample=1 because the
            # 2x pass costs more than the anti-aliasing is worth here.
            canvas = s4_gesture.render_preview(
                croquis, size=(size[1], size[0]), bg=blank_paper(size), fit=False,
                supersample=1, color=config.INK_COLOR,
            )
        except Exception as exc:  # noqa: BLE001 - see last_s4_error above
            self.last_s4_error = f"{type(exc).__name__}: {exc}"
            return None
        self.last_s4_error = None
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
