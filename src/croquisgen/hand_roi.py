"""Hand ROI from the body skeleton.

Task 2 in the plan ("전신 사진에서 손 ROI 지정하기") originally assumed a separate
YOLO hand detector. It does not need one: the forearm direction already says where
the hand is. Extending the elbow->wrist vector past the wrist lands on the palm,
and forearm length is a decent proxy for hand size, which is the same trick
MediaPipe's own Holistic pipeline uses.

Used when the app is in HAND target mode and needs to crop before rendering.
"""
from __future__ import annotations

import numpy as np

from .types import (
    L_ELBOW,
    L_WRIST,
    R_ELBOW,
    R_WRIST,
    PerceptionResult,
)

# Palm centre sits roughly this far past the wrist, in forearm lengths.
_PALM_OFFSET = 0.33
# Crop half-width, also in forearm lengths. Generous so a spread hand still fits.
_CROP_SCALE = 0.85


def hand_roi_from_pose(
    result: PerceptionResult,
    prefer: str | None = None,
    min_visibility: float = 0.5,
) -> tuple[int, int, int, int] | None:
    """Return a square (x, y, w, h) pixel box around the hand, or None.

    `prefer` picks a side ("Left"/"Right" as MediaPipe labels them); otherwise the
    arm whose wrist the pose model is most confident about wins.
    """
    if result.pose is None or result.pose_visibility is None:
        return None

    sides = {"Left": (L_ELBOW, L_WRIST), "Right": (R_ELBOW, R_WRIST)}
    if prefer in sides:
        candidates = [sides[prefer]]
    else:
        candidates = sorted(
            sides.values(),
            key=lambda pair: float(result.pose_visibility[pair[1]]),
            reverse=True,
        )

    points = result.pose_px()
    assert points is not None

    for elbow_idx, wrist_idx in candidates:
        if float(result.pose_visibility[wrist_idx]) < min_visibility:
            continue

        elbow = points[elbow_idx]
        wrist = points[wrist_idx]
        forearm = wrist - elbow
        length = float(np.linalg.norm(forearm))
        if length < 1.0:  # degenerate: arm pointing straight at the camera
            continue

        centre = wrist + forearm * _PALM_OFFSET
        half = length * _CROP_SCALE
        x = int(round(centre[0] - half))
        y = int(round(centre[1] - half))
        size = int(round(half * 2))
        return x, y, size, size

    return None


def crop_with_padding(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop `box` from `frame`, padding with edge pixels when it runs off-frame.

    The ROI routinely extends past the frame when a hand is near the border, and a
    clamped crop would change aspect ratio and shift the hand off-centre.
    """
    x, y, w, h = box
    height, width = frame.shape[:2]

    pad_left = max(0, -x)
    pad_top = max(0, -y)
    pad_right = max(0, x + w - width)
    pad_bottom = max(0, y + h - height)

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(h, 1), max(w, 1), frame.shape[2]), dtype=frame.dtype)

    import cv2

    crop = frame[y0:y1, x0:x1]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    return crop
