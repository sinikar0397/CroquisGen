"""The S4 gesture-croquis engine in the live loop, and the adapter feeding it.

No camera, no models: synthetic MediaPipe landmarks go through
gesture_croquis.realtime into s4_gesture.build_croquis.

    python -m pytest tests -q -k croquis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import config  # noqa: E402
from core.landmarks import L_ELBOW, L_SHOULDER, L_WRIST, NOSE  # noqa: E402
from core.types import PerceptionResult, Style, Target  # noqa: E402
from gesture_croquis import realtime  # noqa: E402
from gesture_croquis.skeleton import IDX  # noqa: E402
from render.registry import get_renderer  # noqa: E402

from test_pipeline import make_pose  # noqa: E402  (same directory)

SIZE = (320, 240)


@pytest.fixture
def s4_engine(monkeypatch):
    monkeypatch.setattr(config, "GESTURE_CROQUIS_ENGINE", "s4")


def body(visibility=None) -> PerceptionResult:
    return PerceptionResult(
        width=SIZE[0], height=SIZE[1], timestamp_ms=0, pose=make_pose(),
        pose_visibility=np.ones(33, dtype=np.float32) if visibility is None else visibility)


# --- the adapter ----------------------------------------------------------

def test_landmarks_land_on_the_right_canonical_joints():
    pose = make_pose()
    kpts, conf = realtime.to_canonical(pose, np.ones(33, np.float32), SIZE)
    expected = pose[L_SHOULDER, :2] * np.array(SIZE, np.float32)
    assert np.allclose(kpts[IDX["lsho"]], expected)
    assert np.allclose(kpts[IDX["nose"]], pose[NOSE, :2] * np.array(SIZE, np.float32))
    assert conf[IDX["lsho"]] == 1.0


def test_centres_are_derived():
    kpts, conf = realtime.to_canonical(make_pose(), np.ones(33, np.float32), SIZE)
    assert np.allclose(kpts[IDX["neck"]],
                       (kpts[IDX["lsho"]] + kpts[IDX["rsho"]]) / 2)
    assert np.allclose(kpts[IDX["pelvis"]],
                       (kpts[IDX["lhip"]] + kpts[IDX["rhip"]]) / 2)
    # The crown sits above the face, i.e. further from the neck (smaller y).
    assert kpts[IDX["head"]][1] < kpts[IDX["nose"]][1]
    assert conf[IDX["head"]] < conf[IDX["nose"]], "derived joints get lower trust"


def test_joints_below_min_visibility_are_dropped():
    """S4 leaves out zero-confidence joints, which is the house rule."""
    visibility = np.ones(33, dtype=np.float32)
    visibility[[L_ELBOW, L_WRIST]] = config.MIN_VISIBILITY - 0.1
    _, conf = realtime.to_canonical(make_pose(), visibility, SIZE)
    assert conf[IDX["lelb"]] == 0.0 and conf[IDX["lwri"]] == 0.0
    assert conf[IDX["relb"]] > 0.0, "the confident side survives"


def test_smalltoe_has_no_source_and_stays_missing():
    _, conf = realtime.to_canonical(make_pose(), np.ones(33, np.float32), SIZE)
    assert conf[IDX["lsmalltoe"]] == 0.0 and conf[IDX["rsmalltoe"]] == 0.0


def test_pose_frame_uses_the_render_size_not_the_capture_size():
    """Strokes come back in canvas pixels, so img_hw follows the canvas."""
    result = PerceptionResult(width=1280, height=720, timestamp_ms=0, pose=make_pose(),
                              pose_visibility=np.ones(33, dtype=np.float32))
    frame = realtime.pose_frame(result, SIZE)
    assert frame.img_hw == (SIZE[1], SIZE[0])


def test_pose_frame_is_none_without_a_pose():
    assert realtime.pose_frame(
        PerceptionResult(width=SIZE[0], height=SIZE[1], timestamp_ms=0), SIZE) is None


def test_pose_frame_is_none_without_a_trunk():
    """No shoulders or hips means no spine and no line of action."""
    visibility = np.ones(33, dtype=np.float32)
    visibility[[11, 12]] = 0.0  # both shoulders
    assert realtime.pose_frame(body(visibility), SIZE) is None


# --- the engine through the renderer contract -----------------------------

def test_s4_honours_the_render_contract(s4_engine):
    image = get_renderer(Style.GESTURE_CROQUIS).render(body(), Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert image.dtype == np.uint8
    assert image.std() > 1.0, "nothing was drawn"


def test_s4_draws_something_other_than_the_placeholder(s4_engine, monkeypatch):
    renderer = get_renderer(Style.GESTURE_CROQUIS)
    with_s4 = renderer.render(body(), Target.FULL_BODY, SIZE)
    assert renderer.last_s4_error is None, renderer.last_s4_error
    monkeypatch.setattr(config, "GESTURE_CROQUIS_ENGINE", "placeholder")
    placeholder = renderer.render(body(), Target.FULL_BODY, SIZE)
    assert not np.array_equal(with_s4, placeholder)


def test_s4_falls_back_to_the_placeholder_without_a_pose(s4_engine):
    """An empty frame is normal; blank paper, no exception."""
    empty = PerceptionResult(width=SIZE[0], height=SIZE[1], timestamp_ms=0)
    image = get_renderer(Style.GESTURE_CROQUIS).render(empty, Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)


def test_hand_target_stays_on_the_placeholder(s4_engine):
    """Every S4 stage is about a body; there is no hand path."""
    hand = np.linspace(0.4, 0.6, 21 * 3).reshape(21, 3).astype(np.float32)
    result = PerceptionResult(width=SIZE[0], height=SIZE[1], timestamp_ms=0, hand=hand)
    renderer = get_renderer(Style.GESTURE_CROQUIS)
    image = renderer.render(result, Target.HAND, SIZE)
    assert image.std() > 1.0, "the hand should still be drawn"


def test_an_s4_failure_is_recorded_and_degrades(s4_engine, monkeypatch):
    """A renderer that raises would kill the render thread."""
    from gesture_croquis import s4_gesture

    def explode(*_args, **_kwargs):
        raise ValueError("no valid line-of-action chain")

    monkeypatch.setattr(s4_gesture, "build_croquis", explode)
    renderer = get_renderer(Style.GESTURE_CROQUIS)
    image = renderer.render(body(), Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert "no valid line-of-action chain" in renderer.last_s4_error


def test_s4_is_the_default_engine():
    assert config.GESTURE_CROQUIS_ENGINE == "s4"


def test_the_placeholder_still_renders_when_selected(monkeypatch):
    """The escape hatch --placeholder selects, and S4 falls back to."""
    monkeypatch.setattr(config, "GESTURE_CROQUIS_ENGINE", "placeholder")
    image = get_renderer(Style.GESTURE_CROQUIS).render(body(), Target.FULL_BODY, SIZE)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    assert image.std() > 1.0


def test_the_gesture_renderer_declares_itself_stateless():
    """What lets the render stage run several workers over different frames."""
    assert get_renderer(Style.GESTURE_CROQUIS).stateless is True
    assert get_renderer(Style.CONSTRUCTIVE).stateless is False
