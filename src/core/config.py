"""Tunables in one place.

Numbers that affect the frame budget are commented with what they cost. Measured
on an M-series CPU at 1280x720 with tools/bench_perception.py.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
CAPTURE_DIR = PROJECT_ROOT / "captures"

# Which landmark source perception.get_backend() builds. This is the line that
# changes when the RTMPose stack in gesture_croquis/ grows a real-time path;
# everything downstream sees the same PerceptionResult either way.
POSE_BACKEND = "mediapipe"

# "full" costs ~25 ms/frame, "lite" ~15 ms. Drop to lite if the renderers push the
# loop under 30 fps - lite loses accuracy mostly on occluded limbs.
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"
# GestureRecognizer is a superset of HandLandmarker - same 21 landmarks plus a
# pre-trained label at the same cost - so hand_landmarker.task is not needed.
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"

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

# How many threads the render stage runs. Only read by the threaded loop, and
# only honoured for renderers that declare themselves stateless - the
# constructive renderer smooths across frames and has to stay on one thread.
# Workers claim different frames and drop their result if a later frame was
# already published, so extra workers buy throughput, never ordering.
#
# 1 is right on this machine (4 performance cores), including for the S4 engine.
# S4 alone scales 1.4x on two threads and 1.7x on three, but that disappears
# once the perception stage is running beside it: 1.2x on two, and three threads
# come out *worse* than two. Measured with tools/bench_threading.py and
# tools/smoke_threading.py --s4 --workers 1 2 3. Worth retrying on a machine with
# more performance cores.
RENDER_WORKERS = 1

# Which gesture-croquis implementation renders.
#   "s4"          ~56 ms  the real S1-S4 abstraction, in its 2D mode
#   "placeholder"  ~7 ms  geometric lines straight from the landmarks
# S4 is the default: it is the drawing the project is actually for, and the
# placeholder was only ever a stand-in. It costs most of twice the whole frame
# budget, so the croquis updates about 13 times a second in the threaded loop
# while the camera runs at 49 and the gesture UI stays at the display rate.
# Holding a pose works; a moving pose visibly trails. Getting it to 30 means
# cutting S4's own cost - the line-of-action search is 39 ms of the 56, fitting a
# cubic Bezier per candidate chain. `--placeholder` goes back to the cheap one,
# which is also what the hand target always draws. See
# gesture_croquis/realtime.py for what 2D mode gives up.
GESTURE_CROQUIS_ENGINE = "s4"
# S4 layers: A line of action, B rhythm/torso, C construction volumes. "C" is a
# mannequin hint and costs little, but it fights the loose look of a croquis.
S4_LAYERS = "AB"
# Subject height prior. Sets S4's stroke scale; it does not need to be exact.
BODY_HEIGHT_M = 1.70
