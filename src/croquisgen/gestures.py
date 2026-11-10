"""Hand gestures -> application events.

MediaPipe's built-in classifier covers two of the three gestures in the plan
(Open_Palm = capture, Pointing_Up = open menu) with no training, so the work here
is temporal, not classification: a label is only an event once it has been stable
for a few reads, and a held pose is a different event from a momentary one.

The finger snap is the exception and is handled separately - see SnapDetector.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from . import config
from .types import INDEX_TIP, MIDDLE_TIP, PerceptionResult, Style, THUMB_TIP, WRIST

# MediaPipe GestureRecognizer category names this app reacts to.
GESTURE_CAPTURE = "Open_Palm"
GESTURE_POINT = "Pointing_Up"


class Event(Enum):
    CAPTURE = auto()
    TOGGLE_TARGET = auto()
    MENU_OPENED = auto()
    MENU_CANCELLED = auto()
    STYLE_SELECTED = auto()


@dataclass
class GestureEvent:
    kind: Event
    style: Style | None = None
    position: tuple[float, float] | None = None


class SnapDetector:
    """Finger snap, by heuristic.

    This is the weak point of the gesture set and is worth validating early. A snap
    takes a few tens of milliseconds, so at 30 fps it spans 1-2 frames and motion
    blur degrades exactly the landmarks needed to see it. The heuristic below looks
    for the two-phase shape of a snap: thumb and middle fingertip touching, then the
    middle fingertip collapsing toward the palm within a couple of frames.

    Distances are normalised by hand size so they do not depend on how far the user
    is from the camera.

    If this proves unreliable in testing, the fallback that needs no new model is to
    drop the snap and add the full-body/hand toggle as a third sector in the radial
    menu, which is already a reliable interaction.
    """

    # Thumb-middle distance below this, relative to hand size, counts as "touching".
    TOUCH_RATIO = 0.25
    # Middle fingertip must then come at least this close to the palm base.
    COLLAPSE_RATIO = 0.55
    # The collapse has to happen within this many hand reads of the touch.
    MAX_FRAMES = 3

    def __init__(self) -> None:
        self._armed_age: int | None = None

    def update(self, hand: np.ndarray | None) -> bool:
        if hand is None:
            self._armed_age = None
            return False

        # Wrist-to-middle-knuckle span, a scale reference that a snap does not change.
        scale = float(np.linalg.norm(hand[MIDDLE_TIP][:2] - hand[WRIST][:2]))
        if scale < 1e-6:
            self._armed_age = None
            return False

        thumb_middle = float(np.linalg.norm(hand[THUMB_TIP][:2] - hand[MIDDLE_TIP][:2])) / scale
        middle_palm = float(np.linalg.norm(hand[MIDDLE_TIP][:2] - hand[WRIST][:2])) / scale

        if self._armed_age is not None:
            self._armed_age += 1
            if middle_palm < self.COLLAPSE_RATIO:
                self._armed_age = None
                return True
            if self._armed_age > self.MAX_FRAMES:
                self._armed_age = None

        if thumb_middle < self.TOUCH_RATIO:
            self._armed_age = 0

        return False


class GestureInterpreter:
    """Turns per-frame recognition results into discrete events."""

    def __init__(self, enable_snap: bool = True) -> None:
        self._current_label: str | None = None
        self._stable_reads = 0
        self._label_start_ms: int | None = None
        self._cooldown_until_ms = 0
        self._menu_open = False
        self._menu_anchor: tuple[float, float] | None = None
        self._snap = SnapDetector() if enable_snap else None

    @property
    def menu_anchor(self) -> tuple[float, float] | None:
        return self._menu_anchor

    def update(self, result: PerceptionResult) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        now = result.timestamp_ms
        label = result.gesture

        if label != self._current_label:
            self._current_label = label
            self._stable_reads = 1
            self._label_start_ms = now
        else:
            self._stable_reads += 1

        stable = self._stable_reads >= config.GESTURE_STABLE_READS
        held_ms = now - self._label_start_ms if self._label_start_ms is not None else 0

        # The menu owns the hand while it is open, so nothing else fires.
        if self._menu_open:
            if label != GESTURE_POINT or not result.has_hand():
                self._menu_open = False
                self._menu_anchor = None
                events.append(GestureEvent(Event.MENU_CANCELLED))
                return events

            selected = self._menu_selection(result)
            if selected is not None:
                self._menu_open = False
                self._menu_anchor = None
                self._cooldown_until_ms = now + config.GESTURE_COOLDOWN_MS
                events.append(GestureEvent(Event.STYLE_SELECTED, style=selected))
            return events

        if now < self._cooldown_until_ms:
            return events

        if self._snap is not None and self._snap.update(result.hand):
            self._cooldown_until_ms = now + config.GESTURE_COOLDOWN_MS
            events.append(GestureEvent(Event.TOGGLE_TARGET))
            return events

        if stable and label == GESTURE_CAPTURE:
            self._cooldown_until_ms = now + config.GESTURE_COOLDOWN_MS
            events.append(GestureEvent(Event.CAPTURE))
            return events

        if stable and label == GESTURE_POINT and held_ms >= config.MENU_HOLD_MS:
            tip = self._index_tip_px(result)
            if tip is not None:
                self._menu_open = True
                self._menu_anchor = tip
                events.append(GestureEvent(Event.MENU_OPENED, position=tip))

        return events

    def menu_progress(self, result: PerceptionResult) -> float:
        """0..1 fill for the ring that shows the 1-second hold building up."""
        if self._menu_open or self._current_label != GESTURE_POINT:
            return 0.0
        if self._label_start_ms is None:
            return 0.0
        return min(1.0, (result.timestamp_ms - self._label_start_ms) / config.MENU_HOLD_MS)

    def _menu_selection(self, result: PerceptionResult) -> Style | None:
        tip = self._index_tip_px(result)
        if tip is None or self._menu_anchor is None:
            return None
        dx = tip[0] - self._menu_anchor[0]
        dy = tip[1] - self._menu_anchor[1]
        if np.hypot(dx, dy) < config.MENU_SELECT_RADIUS * result.height:
            return None
        return style_for_direction(dx, dy)

    @staticmethod
    def _index_tip_px(result: PerceptionResult) -> tuple[float, float] | None:
        points = result.hand_px()
        if points is None:
            return None
        return float(points[INDEX_TIP][0]), float(points[INDEX_TIP][1])


def style_for_direction(dx: float, dy: float) -> Style:
    """Which sector a fingertip offset points at.

    Two styles split the circle into upper and lower halves, matching the mockup
    where Gesture Croquis sits above Constructive Anatomy. Adding a third style
    means replacing this with an angular bucket over len(Style) sectors.
    """
    return Style.GESTURE_CROQUIS if dy < 0 else Style.CONSTRUCTIVE
