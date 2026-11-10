"""Tunables in one place.

Numbers that affect the frame budget are commented with what they cost. Measured
on an M-series CPU at 1280x720 with scripts/bench_perception.py.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
CAPTURE_DIR = PROJECT_ROOT / "captures"

# "full" costs ~25 ms/frame, "lite" ~15 ms. Drop to lite if the renderers push the
# loop under 30 fps - lite loses accuracy mostly on occluded limbs.
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"

CAMERA_INDEX = 0
# Capture resolution barely affects inference time - MediaPipe resizes to the
# model's own input internally - so there is no speed reason to capture small.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIRROR = True  # selfie view: the user's right hand should appear on the right

# Pose runs every frame; hand runs every Nth. Gestures are slow enough to track at
# ~10 fps, and this is what keeps the combined loop above 30 fps:
# pose 25 ms + hand 19/3 ms = ~31 ms.
HAND_EVERY_N_FRAMES = 3

# Segmentation adds ~2 ms in VIDEO mode and gives a clean person silhouette, which
# is far more robust than running an edge filter over the raw frame.
OUTPUT_SEGMENTATION = True

POSE_MIN_CONFIDENCE = 0.5
HAND_MIN_CONFIDENCE = 0.5
# Landmarks below this confidence are not drawn - better a missing limb than one
# the model invented.
MIN_VISIBILITY = 0.5

# --- gesture event tuning -------------------------------------------------
# A label must hold for this many consecutive hand reads before it fires, which
# filters the single-frame misclassifications that happen during motion blur.
GESTURE_STABLE_READS = 2
# No second event for this long after one fires.
GESTURE_COOLDOWN_MS = 800
# "손가락 하나 세워서 1초 이상" from the plan.
MENU_HOLD_MS = 1000
# Fingertip must travel this far from the menu centre, as a fraction of frame
# height, before a sector counts as chosen.
MENU_SELECT_RADIUS = 0.12

# --- rendering ------------------------------------------------------------
PAPER_COLOR = (238, 240, 244)  # BGR, warm off-white
INK_COLOR = (48, 44, 40)
TOAST_FRAMES = 60
