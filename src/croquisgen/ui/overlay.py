"""Status bar, capture toast and the camera inset, as in the screen design."""
from __future__ import annotations

import cv2
import numpy as np

from ..renderers import renderer_name
from ..types import AppState, Target

_BAR_HEIGHT = 34
_BAR_COLOR = (28, 24, 22)
_TEXT = (238, 238, 238)
_MUTED = (150, 146, 142)
_OK = (90, 200, 110)


def draw_status_bar(frame: np.ndarray, state: AppState, fps: float, camera_ok: bool) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - _BAR_HEIGHT), (width, height), _BAR_COLOR, -1)

    target = "FULL BODY" if state.target is Target.FULL_BODY else "HAND"
    label = f"{target}  -  {renderer_name(state.style).upper()}"
    cv2.putText(frame, label, (14, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _TEXT, 1, cv2.LINE_AA)

    right = f"{fps:4.1f} fps"
    size, _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, right, (width - size[0] - 36, height - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _MUTED, 1, cv2.LINE_AA)
    cv2.circle(frame, (width - 18, height - 16), 5, _OK if camera_ok else (80, 80, 200), -1,
               cv2.LINE_AA)


def draw_toast(frame: np.ndarray, state: AppState) -> None:
    """Capture confirmation with a thumbnail, bottom right."""
    if state.toast_frames <= 0 or state.toast_thumbnail is None:
        return

    thumb = state.toast_thumbnail
    th, tw = thumb.shape[:2]
    pad = 12
    box_w, box_h = tw + 240, th + pad * 2
    x = frame.shape[1] - box_w - 20
    y = frame.shape[0] - box_h - _BAR_HEIGHT - 20
    if x < 0 or y < 0:
        return

    # Fades out over the last third of its life so it does not just vanish.
    alpha = min(1.0, state.toast_frames / 20.0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (38, 34, 32), -1)
    overlay[y + pad:y + pad + th, x + pad:x + pad + tw] = thumb
    cv2.putText(overlay, "Image captured", (x + tw + pad * 2, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT, 1, cv2.LINE_AA)
    cv2.circle(overlay, (x + box_w - 28, y + box_h // 2), 11, _OK, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_camera_inset(canvas: np.ndarray, frame: np.ndarray, scale: float = 0.22) -> None:
    """Live camera thumbnail over the croquis, so the user can see their pose.

    The mockups show the croquis filling the window, but while posing you cannot
    tell what the camera is actually seeing, so this sits in the corner.
    """
    height, width = canvas.shape[:2]
    inset_w = int(width * scale)
    inset_h = int(frame.shape[0] * inset_w / frame.shape[1])
    inset = cv2.resize(frame, (inset_w, inset_h))

    x, y = width - inset_w - 18, 18
    canvas[y:y + inset_h, x:x + inset_w] = inset
    cv2.rectangle(canvas, (x, y), (x + inset_w, y + inset_h), (90, 86, 82), 1, cv2.LINE_AA)
