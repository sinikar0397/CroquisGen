"""Bridge from the app's PerceptionResult to this package's PoseFrame.

The S1-S4 pipeline was written for photographs: S1 detects keypoints with
RTMPose, S2 fits a metric 3D skeleton, S3 solves the static load, and S4 turns
the result into strokes. Only S4 is cheap enough for the live loop, and this
module is what lets it run there without S1-S3.

**Why 2D mode.** S4's BodyView takes either a fitted 3D skeleton or, when
frame.joints3d is None, the 2D keypoints alone - "photo view only, no depth
cues". The live loop only ever wants the photo view, so the 2D path costs
nothing in features we would have used.

MediaPipe does hand us metric 3D (pose_world), which looks like it should feed
the 3D path and skip S2 entirely. It cannot, not directly: those coordinates are
centred on the hips, while BodyView.proj divides by z as a distance from the
camera. Placing the body at its true depth means solving for the translation that
reprojects onto the 2D landmarks - a small fit, but a fit, and that belongs in S2
rather than here. Until then 3D-only features (novel views, depth-ordered line
weights, front/behind accents) stay off in the live loop.

Confidences below config.MIN_VISIBILITY are zeroed rather than passed through.
S4 treats a zero-confidence joint as missing and leaves it out, which is the
house rule: a limb the model invented draws an angle the subject never made.
"""
from __future__ import annotations

import numpy as np

from core import config
from core.landmarks import (
    L_EAR, L_EYE, L_HEEL, L_INDEX, L_PINKY, NOSE, R_EAR, R_EYE, R_HEEL, R_INDEX, R_PINKY,
)
from core.types import PerceptionResult

from .poseframe import PoseFrame
from .skeleton import IDX, J

# MediaPipe pose index -> canonical joint name. Imported from core.landmarks so
# the two layouts stay described in one place each.
_DIRECT: dict[int, str] = {
    NOSE: "nose", L_EYE: "leye", R_EYE: "reye", L_EAR: "lear", R_EAR: "rear",
    11: "lsho", 12: "rsho", 13: "lelb", 14: "relb", 15: "lwri", 16: "rwri",
    23: "lhip", 24: "rhip", 25: "lknee", 26: "rknee", 27: "lank", 28: "rank",
    L_HEEL: "lheel", R_HEEL: "rheel", 31: "lbigtoe", 32: "rbigtoe",
}
# The canonical hand point is the middle-finger MCP, which the pose model does
# not have; the midpoint of its index and pinky extremities sits close enough for
# a 5 cm bone.
_HAND_FROM = {"lhand": (L_INDEX, L_PINKY), "rhand": (R_INDEX, R_PINKY)}
# No small toe in MediaPipe's 33 points, and no joint between the face points and
# the crown. Both are derived or left missing.
_FACE = ("nose", "leye", "reye", "lear", "rear")


def to_canonical(pose: np.ndarray, visibility: np.ndarray | None,
                 size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe's 33 normalised landmarks -> canonical (J,2) px and (J,) conf."""
    width, height = size
    px = pose[:, :2] * np.array([width, height], dtype=np.float32)
    conf = (np.ones(len(pose), dtype=np.float32) if visibility is None
            else np.asarray(visibility, dtype=np.float32))
    conf = np.where(conf >= config.MIN_VISIBILITY, conf, 0.0)

    k = np.zeros((J, 2), dtype=np.float32)
    c = np.zeros(J, dtype=np.float32)
    for source, name in _DIRECT.items():
        k[IDX[name]] = px[source]
        c[IDX[name]] = conf[source]
    for name, (a, b) in _HAND_FROM.items():
        k[IDX[name]] = (px[a] + px[b]) / 2
        c[IDX[name]] = min(conf[a], conf[b])

    # Centres, the way skeleton._derive_centres does it for the RTMPose layouts.
    # Repeated here rather than called because that helper also fills the slots
    # this layout has no source for.
    k[IDX["neck"]] = (k[IDX["lsho"]] + k[IDX["rsho"]]) / 2
    c[IDX["neck"]] = min(c[IDX["lsho"]], c[IDX["rsho"]])
    k[IDX["pelvis"]] = (k[IDX["lhip"]] + k[IDX["rhip"]]) / 2
    c[IDX["pelvis"]] = min(c[IDX["lhip"]], c[IDX["rhip"]])

    face = [IDX[n] for n in _FACE]
    weights = c[face][:, None] + 1e-6
    centre = (k[face] * weights).sum(0) / weights.sum()
    k[IDX["head"]] = centre + (centre - k[IDX["neck"]]) * 0.35  # push toward the crown
    c[IDX["head"]] = 0.5 * float(np.mean(c[face]))              # derived: lower trust
    return k, c


def pose_frame(result: PerceptionResult, size: tuple[int, int],
               body_height_m: float = config.BODY_HEIGHT_M) -> PoseFrame | None:
    """A 2D-mode PoseFrame for S4, or None when there is nobody to draw.

    img_hw follows the requested render size, not the capture size, so the
    strokes come back in the canvas's own pixels and need no rescaling.
    """
    if result.pose is None:
        return None
    kpts2d, conf = to_canonical(result.pose, result.pose_visibility, size)
    if conf[IDX["neck"]] <= 0.0 or conf[IDX["pelvis"]] <= 0.0:
        return None  # no trunk, so no spine and no line of action
    return PoseFrame(
        img_hw=(size[1], size[0]),
        kpts2d=kpts2d,
        conf=conf,
        s1_layout="mediapipe33",
        body_height_m=body_height_m,
    )
