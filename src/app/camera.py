"""Webcam source."""
from __future__ import annotations

import cv2
import numpy as np

from . import config


class Camera:
    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        mirror: bool = config.MIRROR,
    ) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"cannot open camera {index}. On macOS, the terminal or IDE running "
                "this needs camera permission in System Settings > Privacy & Security."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._mirror = mirror

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        # Selfie view: without this the user reaches left and the hand goes right.
        return cv2.flip(frame, 1) if self._mirror else frame

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
