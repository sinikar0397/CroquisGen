"""Data passed between pipeline stages.

PerceptionResult is the only thing renderers see. It holds plain numpy arrays and
floats rather than MediaPipe objects, so renderers stay independent of the capture
backend - the same renderer code can later run against landmarks produced by
MediaPipe's JS build in a browser.

Which index means which joint lives in core/landmarks.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Target(Enum):
    """What the croquis is being generated from."""

    FULL_BODY = "full_body"
    HAND = "hand"


class Style(Enum):
    """How the croquis is drawn. Order defines the radial menu layout."""

    GESTURE_CROQUIS = "gesture_croquis"
    CONSTRUCTIVE = "constructive"


@dataclass
class PerceptionResult:
    """One frame's worth of analysis.

    Landmark arrays are normalised to [0, 1] in image space, so renderers scale
    them by their own output size instead of assuming the capture resolution.
    """

    width: int
    height: int
    timestamp_ms: int

    # (33, 3) normalised xy + relative depth, or None when no person is in frame.
    pose: np.ndarray | None = None
    # (33,) per-landmark confidence, used to skip limbs the model is guessing at.
    pose_visibility: np.ndarray | None = None
    # (33, 3) metric coordinates in metres, origin at the hip centre. Depth is
    # noisier than xy - smooth it harder before using it for 3D construction.
    pose_world: np.ndarray | None = None
    # (H, W) float32 person probability, 1.0 inside the silhouette.
    segmentation: np.ndarray | None = None

    # (21, 3) normalised landmarks of the tracked hand, or None.
    hand: np.ndarray | None = None
    handedness: str | None = None
    # Label from MediaPipe's built-in classifier, e.g. "Open_Palm", "Pointing_Up".
    gesture: str | None = None
    gesture_score: float = 0.0

    def has_pose(self) -> bool:
        return self.pose is not None

    def has_hand(self) -> bool:
        return self.hand is not None

    def pose_px(self) -> np.ndarray | None:
        """Pose landmarks in pixel coordinates of this frame."""
        if self.pose is None:
            return None
        return self.pose[:, :2] * np.array([self.width, self.height], dtype=np.float32)

    def hand_px(self) -> np.ndarray | None:
        if self.hand is None:
            return None
        return self.hand[:, :2] * np.array([self.width, self.height], dtype=np.float32)


@dataclass
class AppState:
    """Everything the UI needs to draw itself."""

    target: Target = Target.FULL_BODY
    style: Style = Style.GESTURE_CROQUIS
    menu_open: bool = False
    menu_anchor: tuple[float, float] | None = None  # pixel centre of the radial menu
    menu_highlight: Style | None = None
    # Frames remaining on the "이미지가 캡처되었습니다" toast.
    toast_frames: int = 0
    toast_thumbnail: np.ndarray | None = field(default=None, repr=False)
