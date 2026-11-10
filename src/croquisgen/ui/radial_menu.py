"""The radial menu that opens around the fingertip.

Drawn over the live camera view, not the croquis: the user is looking at their own
hand while choosing, so the menu has to appear where the hand is.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..types import Style

_SECTOR_LABELS = {
    Style.GESTURE_CROQUIS: ("Gesture", "Croquis"),
    Style.CONSTRUCTIVE: ("Constructive", "Anatomy"),
}

_ACCENT = (209, 199, 61)      # BGR teal
_DIM = (120, 112, 104)
_TEXT = (245, 245, 245)


def draw_hold_progress(
    frame: np.ndarray, tip: tuple[float, float], progress: float, radius: int = 34
) -> None:
    """Arc that fills while the user holds a raised finger.

    Without it a one-second hold feels like the app is ignoring them.
    """
    if progress <= 0.0:
        return
    centre = (int(tip[0]), int(tip[1]))
    cv2.circle(frame, centre, radius, _DIM, 2, cv2.LINE_AA)
    cv2.ellipse(frame, centre, (radius, radius), -90, 0, 360 * progress, _ACCENT, 4, cv2.LINE_AA)


def draw_menu(
    frame: np.ndarray,
    anchor: tuple[float, float],
    active: Style,
    highlight: Style | None = None,
    radius: int = 150,
) -> None:
    """Two half-circle sectors, upper and lower, matching style_for_direction()."""
    overlay = frame.copy()
    centre = (int(anchor[0]), int(anchor[1]))
    order = [Style.GESTURE_CROQUIS, Style.CONSTRUCTIVE]

    for i, style in enumerate(order):
        start_angle = 180 if i == 0 else 0
        selected = style is highlight
        color = _ACCENT if selected else (64, 60, 56)
        cv2.ellipse(overlay, centre, (radius, radius), 0, start_angle,
                    start_angle + 180, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.circle(frame, centre, radius, _ACCENT, 2, cv2.LINE_AA)
    cv2.line(frame, (centre[0] - radius, centre[1]), (centre[0] + radius, centre[1]),
             _ACCENT, 1, cv2.LINE_AA)

    for i, style in enumerate(order):
        offset = -radius // 2 if i == 0 else radius // 2
        lines = _SECTOR_LABELS[style]
        for j, text in enumerate(lines):
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            position = (centre[0] - size[0] // 2, centre[1] + offset + j * 20 - 6)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _TEXT if style is not active else _ACCENT, 1, cv2.LINE_AA)

    # Dead zone: inside this the fingertip has not committed to a direction yet.
    cv2.circle(frame, centre, 22, _TEXT, 1, cv2.LINE_AA)
