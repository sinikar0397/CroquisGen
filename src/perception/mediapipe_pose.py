"""MediaPipe Tasks backend: pose every frame, hands on a slower cadence.

Selected via perception.get_backend("mediapipe").

Two things here are not obvious and matter a lot:

1. Both landmarkers run in VIDEO mode, not IMAGE mode. VIDEO mode reuses the
   previous frame's result to track instead of re-running the detector, which is
   worth roughly 1.6-2x: pose 40 -> 25 ms, hand 40 -> 19 ms.
2. VIDEO mode requires strictly increasing timestamps. Passing a stale or equal
   timestamp makes MediaPipe raise, so the timestamp comes from a monotonic
   counter rather than wall-clock time.
"""
from __future__ import annotations

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core import config
from core.types import PerceptionResult


def _landmarks_to_array(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


class MediaPipePerception:
    """Runs pose every frame and hands on a slower cadence."""

    def __init__(
        self,
        pose_model=config.POSE_MODEL,
        gesture_model=config.GESTURE_MODEL,
        hand_every_n: int = config.HAND_EVERY_N_FRAMES,
        segmentation: bool = config.OUTPUT_SEGMENTATION,
    ) -> None:
        for path in (pose_model, gesture_model):
            if not path.exists():
                raise FileNotFoundError(
                    f"model not found: {path}\nrun: python scripts/download_models.py"
                )

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=config.POSE_MIN_CONFIDENCE,
                output_segmentation_masks=segmentation,
            )
        )
        # GestureRecognizer is a superset of HandLandmarker: same 21 landmarks plus
        # a pre-trained label, at the same cost. Open_Palm and Pointing_Up - two of
        # the three gestures the project needs - come out of it with no training.
        self._gesture = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(gesture_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=config.HAND_MIN_CONFIDENCE,
            )
        )
        self._hand_every_n = max(1, hand_every_n)
        self._frame_index = 0
        self._timestamp_ms = 0
        # Hand results are reused on frames where the recogniser does not run,
        # otherwise the radial menu would flicker at a third of the frame rate.
        self._last_hand: np.ndarray | None = None
        self._last_handedness: str | None = None
        self._last_gesture: str | None = None
        self._last_score: float = 0.0

    def process(self, frame_bgr: np.ndarray) -> PerceptionResult:
        import cv2  # local import keeps this module importable without a camera stack

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic, and spaced so MediaPipe's tracker sees a plausible frame rate.
        self._timestamp_ms += 33
        ts = self._timestamp_ms

        result = PerceptionResult(width=width, height=height, timestamp_ms=ts)

        pose_result = self._pose.detect_for_video(mp_image, ts)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            result.pose = _landmarks_to_array(lm)
            result.pose_visibility = np.array([p.visibility for p in lm], dtype=np.float32)
            if pose_result.pose_world_landmarks:
                result.pose_world = _landmarks_to_array(pose_result.pose_world_landmarks[0])
            if pose_result.segmentation_masks:
                result.segmentation = pose_result.segmentation_masks[0].numpy_view()

        if self._frame_index % self._hand_every_n == 0:
            gesture_result = self._gesture.recognize_for_video(mp_image, ts)
            if gesture_result.hand_landmarks:
                self._last_hand = _landmarks_to_array(gesture_result.hand_landmarks[0])
                self._last_handedness = gesture_result.handedness[0][0].category_name
                if gesture_result.gestures:
                    top = gesture_result.gestures[0][0]
                    self._last_gesture = top.category_name
                    self._last_score = top.score
                else:
                    self._last_gesture, self._last_score = None, 0.0
            else:
                self._last_hand = None
                self._last_handedness = None
                self._last_gesture, self._last_score = None, 0.0

        result.hand = self._last_hand
        result.handedness = self._last_handedness
        result.gesture = self._last_gesture
        result.gesture_score = self._last_score

        self._frame_index += 1
        return result

    def hand_ran_this_frame(self) -> bool:
        """True when the last process() call actually refreshed the hand result."""
        return (self._frame_index - 1) % self._hand_every_n == 0

    def close(self) -> None:
        self._pose.close()
        self._gesture.close()

    def __enter__(self) -> "MediaPipePerception":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
