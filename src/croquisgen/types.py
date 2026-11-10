"""Data passed between pipeline stages.

PerceptionResult is the only thing renderers see. It holds plain numpy arrays and
floats rather than MediaPipe objects, so renderers stay independent of the capture
backend - the same renderer code can later run against landmarks produced by
MediaPipe's JS build in a browser.
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


# Pose landmark indices (MediaPipe Pose, 33 points).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Torso-and-limbs skeleton. Face/finger detail is left out on purpose - croquis
# cares about the body's flow, not features.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
)

# Hand landmark indices (MediaPipe Hands, 21 points).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP = 13
PINKY_MCP, PINKY_TIP = 17, 20

HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


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
