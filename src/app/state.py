"""Gesture events applied to application state, plus capture to disk."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from core import config
from .gestures import Event, GestureEvent
from core.types import AppState, Style, Target


class StateMachine:
    def __init__(self, capture_dir: Path = config.CAPTURE_DIR) -> None:
        self.state = AppState()
        self._capture_dir = capture_dir

    def apply(self, events: list[GestureEvent], croquis: np.ndarray | None) -> list[Path]:
        """Fold events into state. Returns paths of any files written."""
        written: list[Path] = []
        for event in events:
            if event.kind is Event.CAPTURE:
                path = self._save(croquis)
                if path is not None:
                    written.append(path)
            elif event.kind is Event.TOGGLE_TARGET:
                self.state.target = (
                    Target.HAND if self.state.target is Target.FULL_BODY else Target.FULL_BODY
                )
            elif event.kind is Event.MENU_OPENED:
                self.state.menu_open = True
                self.state.menu_anchor = event.position
            elif event.kind is Event.MENU_CANCELLED:
                self.state.menu_open = False
                self.state.menu_anchor = None
                self.state.menu_highlight = None
            elif event.kind is Event.STYLE_SELECTED and event.style is not None:
                self.state.style = event.style
                self.state.menu_open = False
                self.state.menu_anchor = None
                self.state.menu_highlight = None
        return written

    def capture(self, croquis: np.ndarray | None) -> Path | None:
        """Save the current croquis, as the capture gesture would."""
        return self._save(croquis)

    def tick(self) -> None:
        if self.state.toast_frames > 0:
            self.state.toast_frames -= 1
            if self.state.toast_frames == 0:
                self.state.toast_thumbnail = None

    def set_style(self, style: Style) -> None:
        self.state.style = style

    def toggle_target(self) -> None:
        self.state.target = (
            Target.HAND if self.state.target is Target.FULL_BODY else Target.FULL_BODY
        )

    def _save(self, croquis: np.ndarray | None) -> Path | None:
        if croquis is None:
            return None
        import cv2

        self._capture_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self._capture_dir / f"{self.state.target.value}_{self.state.style.value}_{stamp}.png"
        cv2.imwrite(str(path), croquis)

        thumb_height = 96
        scale = thumb_height / croquis.shape[0]
        self.state.toast_thumbnail = cv2.resize(
            croquis, (max(1, int(croquis.shape[1] * scale)), thumb_height)
        )
        self.state.toast_frames = config.TOAST_FRAMES
        return path
