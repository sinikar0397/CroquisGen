"""Tests that run without a camera or a model download.

Everything here feeds synthetic landmarks straight into the pure parts of the
pipeline. Anything that needs MediaPipe belongs in a script, not a test.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.gestures import (  # noqa: E402
    GESTURE_CAPTURE,
    GESTURE_POINT,
    Event,
    GestureInterpreter,
    style_for_direction,
)
from app.hand_roi import crop_with_padding, hand_roi_from_pose  # noqa: E402
from render.registry import get_renderer  # noqa: E402
from app.state import StateMachine  # noqa: E402
from core.landmarks import INDEX_TIP, L_ELBOW, L_WRIST  # noqa: E402
from core.types import PerceptionResult, Style, Target  # noqa: E402

SIZE = (320, 240)


def make_pose() -> np.ndarray:
    """A roughly upright figure, normalised.

    Every landmark the renderers read is filled in. Leaving one at the 0.5
    default puts it in the middle of the chest, which looks like a plausible
    coordinate but silently changes what the renderer draws - the constructive
    renderer reads ears, heels and foot tips as well as the main joints.
    """
    pose = np.full((33, 3), 0.5, dtype=np.float32)
    pose[0] = (0.50, 0.15, 0.0)    # nose
    pose[2] = (0.48, 0.14, 0.0)    # left eye
    pose[5] = (0.52, 0.14, 0.0)
    pose[7] = (0.46, 0.15, 0.0)    # left ear
    pose[8] = (0.54, 0.15, 0.0)
    pose[11] = (0.42, 0.30, 0.0)   # left shoulder
    pose[12] = (0.58, 0.30, 0.0)
    pose[13] = (0.38, 0.45, 0.0)   # left elbow
    pose[14] = (0.62, 0.45, 0.0)
    pose[15] = (0.35, 0.58, 0.0)   # left wrist
    pose[16] = (0.65, 0.58, 0.0)
    pose[17] = (0.34, 0.61, 0.0)   # left pinky (pose model's hand extremity)
    pose[18] = (0.66, 0.61, 0.0)
    pose[19] = (0.35, 0.62, 0.0)   # left index
    pose[20] = (0.65, 0.62, 0.0)
    pose[23] = (0.45, 0.58, 0.0)   # left hip
    pose[24] = (0.55, 0.58, 0.0)
    pose[25] = (0.44, 0.75, 0.0)   # left knee
    pose[26] = (0.56, 0.75, 0.0)
    pose[27] = (0.43, 0.92, 0.0)   # left ankle
    pose[28] = (0.57, 0.92, 0.0)
    pose[29] = (0.43, 0.95, 0.0)   # left heel
    pose[30] = (0.57, 0.95, 0.0)
    pose[31] = (0.45, 0.97, 0.0)   # left foot index
    pose[32] = (0.55, 0.97, 0.0)
    return pose


def make_hand(tip: tuple[float, float] = (0.5, 0.3)) -> np.ndarray:
    hand = np.linspace(0.4, 0.6, 21 * 3).reshape(21, 3).astype(np.float32)
    hand[INDEX_TIP] = (tip[0], tip[1], 0.0)
    return hand


def make_result(**kwargs) -> PerceptionResult:
    base = dict(width=SIZE[0], height=SIZE[1], timestamp_ms=0)
    base.update(kwargs)
    return PerceptionResult(**base)


# --- renderers ------------------------------------------------------------

@pytest.mark.parametrize("style", list(Style))
def test_renders_body_at_requested_size(style):
    result = make_result(pose=make_pose(), pose_visibility=np.ones(33, dtype=np.float32))
    image = get_renderer(style).render(result, Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert image.dtype == np.uint8
    # Something was actually drawn, i.e. the canvas is not uniform paper.
    assert image.std() > 1.0


@pytest.mark.parametrize("style", list(Style))
def test_renders_hand(style):
    result = make_result(hand=make_hand())
    image = get_renderer(style).render(result, Target.HAND, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert image.std() > 1.0


@pytest.mark.parametrize("style", list(Style))
def test_blank_input_does_not_raise(style):
    """No person in frame is normal, not an error."""
    image = get_renderer(style).render(make_result(), Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)


@pytest.mark.parametrize("style", list(Style))
def test_low_visibility_limbs_are_skipped(style):
    """A limb the model is unsure about should not be drawn."""
    visibility = np.ones(33, dtype=np.float32)
    confident = get_renderer(style).render(
        make_result(pose=make_pose(), pose_visibility=visibility), Target.FULL_BODY, SIZE
    )
    visibility[[L_ELBOW, L_WRIST]] = 0.1
    partial = get_renderer(style).render(
        make_result(pose=make_pose(), pose_visibility=visibility), Target.FULL_BODY, SIZE
    )
    assert not np.array_equal(confident, partial)


# --- hand ROI -------------------------------------------------------------

def test_hand_roi_sits_beyond_the_wrist():
    result = make_result(pose=make_pose(), pose_visibility=np.ones(33, dtype=np.float32))
    box = hand_roi_from_pose(result, prefer="Left")
    assert box is not None
    x, y, w, h = box
    assert w == h and w > 0
    points = result.pose_px()
    centre = (x + w / 2, y + h / 2)
    # The ROI centre must be further from the elbow than the wrist is.
    assert np.linalg.norm(np.array(centre) - points[L_ELBOW]) > np.linalg.norm(
        points[L_WRIST] - points[L_ELBOW]
    )


def test_hand_roi_returns_none_without_pose():
    assert hand_roi_from_pose(make_result()) is None


def test_crop_keeps_requested_shape_off_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    crop = crop_with_padding(frame, (-40, -30, 100, 100))
    assert crop.shape[:2] == (100, 100)


# --- gestures -------------------------------------------------------------

def test_open_palm_fires_capture_once():
    interpreter = GestureInterpreter(enable_snap=False)
    kinds = []
    for i in range(6):
        result = make_result(timestamp_ms=i * 33, hand=make_hand(), gesture=GESTURE_CAPTURE)
        kinds += [e.kind for e in interpreter.update(result)]
    # Cooldown must stop a held palm from firing every frame.
    assert kinds.count(Event.CAPTURE) == 1


def test_menu_needs_a_sustained_hold():
    interpreter = GestureInterpreter(enable_snap=False)
    early = interpreter.update(
        make_result(timestamp_ms=100, hand=make_hand(), gesture=GESTURE_POINT)
    )
    interpreter.update(make_result(timestamp_ms=400, hand=make_hand(), gesture=GESTURE_POINT))
    late = interpreter.update(
        make_result(timestamp_ms=1500, hand=make_hand(), gesture=GESTURE_POINT)
    )
    assert [e.kind for e in early] == []
    assert [e.kind for e in late] == [Event.MENU_OPENED]


def test_flick_selects_the_style_in_that_direction():
    interpreter = GestureInterpreter(enable_snap=False)
    for ts in (0, 100, 1500):
        interpreter.update(make_result(timestamp_ms=ts, hand=make_hand((0.5, 0.5)),
                                       gesture=GESTURE_POINT))
    assert interpreter.menu_anchor is not None

    # Fingertip moves well above the anchor -> upper sector.
    events = interpreter.update(
        make_result(timestamp_ms=1600, hand=make_hand((0.5, 0.1)), gesture=GESTURE_POINT)
    )
    assert [e.kind for e in events] == [Event.STYLE_SELECTED]
    assert events[0].style is Style.GESTURE_CROQUIS


def test_losing_the_hand_cancels_the_menu():
    interpreter = GestureInterpreter(enable_snap=False)
    for ts in (0, 100, 1500):
        interpreter.update(make_result(timestamp_ms=ts, hand=make_hand(), gesture=GESTURE_POINT))
    events = interpreter.update(make_result(timestamp_ms=1600))
    assert [e.kind for e in events] == [Event.MENU_CANCELLED]


def test_sector_directions():
    assert style_for_direction(0, -100) is Style.GESTURE_CROQUIS
    assert style_for_direction(0, 100) is Style.CONSTRUCTIVE


# --- state ----------------------------------------------------------------

def test_capture_writes_a_file_and_raises_a_toast(tmp_path):
    machine = StateMachine(capture_dir=tmp_path)
    croquis = np.zeros((120, 160, 3), dtype=np.uint8)
    written = machine.capture(croquis)
    assert written is not None and written.exists()
    assert machine.state.toast_frames > 0
    assert machine.state.toast_thumbnail is not None


def test_toast_expires():
    machine = StateMachine()
    machine.state.toast_frames = 2
    machine.tick()
    machine.tick()
    assert machine.state.toast_frames == 0
    assert machine.state.toast_thumbnail is None


def test_target_toggles():
    machine = StateMachine()
    assert machine.state.target is Target.FULL_BODY
    machine.toggle_target()
    assert machine.state.target is Target.HAND
    machine.toggle_target()
    assert machine.state.target is Target.FULL_BODY
