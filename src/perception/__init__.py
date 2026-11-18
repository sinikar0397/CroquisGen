"""Landmark supply. The one place a different detector gets swapped in.

Backends are not interchangeable in cost, only in interface. MediaPipe runs at
~25 ms/frame and drives the live loop. The RTMPose stack in gesture_croquis/ is
far more accurate but its 3D fit takes 250-600 ms per frame, so it has no
real-time path yet; when it grows one it lands here as a second backend and
config.POSE_BACKEND selects it.

Backends return core.types.PerceptionResult with landmarks in the
core.landmarks layout, whatever the underlying model's own layout is.
"""
from __future__ import annotations

from typing import Protocol

from core import config
from core.types import PerceptionResult


class PoseBackend(Protocol):
    """What the app loop needs from a landmark source."""

    def process(self, frame_bgr) -> PerceptionResult:
        """One frame in, one frame's landmarks out. Never raises on an empty frame."""
        ...

    def close(self) -> None:
        ...


def get_backend(name: str = config.POSE_BACKEND, **kwargs) -> PoseBackend:
    """Build the configured backend. Imported lazily so an unused backend's
    dependencies (onnxruntime, rtmlib) are not required to start the app."""
    if name == "mediapipe":
        from .mediapipe_pose import MediaPipePerception

        return MediaPipePerception(**kwargs)
    raise ValueError(f"unknown pose backend: {name!r} (available: 'mediapipe')")


__all__ = ["PoseBackend", "get_backend"]
