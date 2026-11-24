"""The threaded loop and its plumbing, with no camera and no models.

The stages are injected, so the real wiring runs against stubs: a camera that
hands out frames, a perception that returns prepared landmarks, and a display
that records what it was asked to show instead of opening a window.

    python -m pytest tests -q -k thread
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import loop as app_loop  # noqa: E402
from app.gestures import GestureInterpreter  # noqa: E402
from app.pipeline import Pipeline, Slot  # noqa: E402
from app.state import StateMachine  # noqa: E402
from core.types import PerceptionResult, Style, Target  # noqa: E402

from test_pipeline import make_pose  # noqa: E402  (same directory)

SIZE = (160, 120)


# --- Slot -----------------------------------------------------------------

def test_slot_starts_empty():
    value, seq = Slot().latest()
    assert value is None and seq == 0


def test_slot_wait_returns_the_new_value_and_a_new_seq():
    slot: Slot[str] = Slot()
    slot.put("a")
    value, seq = slot.wait(0, timeout=1.0)
    assert value == "a" and seq == 1


def test_slot_wait_times_out_with_the_same_seq():
    """An unchanged seq is how a stage tells 'nothing new' from 'new value'."""
    slot: Slot[str] = Slot()
    slot.put("a")
    _, seq = slot.latest()
    start = time.perf_counter()
    value, again = slot.wait(seq, timeout=0.05)
    assert again == seq and value == "a"
    assert time.perf_counter() - start >= 0.04


def test_slot_keeps_only_the_newest_value():
    slot: Slot[int] = Slot()
    for i in range(5):
        slot.put(i)
    value, seq = slot.latest()
    assert value == 4 and seq == 5


def test_slot_wakes_a_waiting_thread():
    slot: Slot[str] = Slot()
    seen: list[str | None] = []

    def consume() -> None:
        value, _ = slot.wait(0, timeout=2.0)
        seen.append(value)

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.02)
    slot.put("ready")
    thread.join(2.0)
    assert seen == ["ready"]


def test_slot_repeated_value_still_counts_as_new():
    """A still pose renders an identical croquis; the seq has to move anyway."""
    slot: Slot[str] = Slot()
    slot.put("same")
    slot.put("same")
    value, seq = slot.wait(1, timeout=1.0)
    assert value == "same" and seq == 2


def test_put_newer_drops_a_late_worker():
    """A render worker that fell behind must not replace a newer croquis."""
    slot: Slot[str] = Slot()
    assert slot.put_newer("frame 7", 7) is True
    assert slot.put_newer("frame 9", 9) is True
    assert slot.put_newer("frame 8", 8) is False  # started earlier, finished later
    assert slot.put_newer("frame 9 again", 9) is False
    value, _ = slot.latest()
    assert value == "frame 9"


# --- Pipeline -------------------------------------------------------------

def test_pipeline_runs_a_step_repeatedly_until_stopped():
    pipe = Pipeline()
    calls = []
    pipe.spawn("counter", lambda: calls.append(1))
    time.sleep(0.05)
    pipe.stop()
    pipe.join()
    assert len(calls) > 1


def test_step_returning_false_ends_the_run():
    pipe = Pipeline()

    def step() -> bool:
        return False

    pipe.spawn("once", step)
    pipe.join()
    assert not pipe.running
    pipe.raise_if_failed()  # ending on purpose is not a failure


def test_a_failing_stage_is_re_raised_on_the_caller():
    pipe = Pipeline()

    def step() -> None:
        raise ValueError("model exploded")

    pipe.spawn("perception", step)
    pipe.join()
    with pytest.raises(RuntimeError, match="perception stage failed") as info:
        pipe.raise_if_failed()
    assert isinstance(info.value.__cause__, ValueError)


def test_one_stage_failing_stops_the_others():
    """Otherwise the main thread waits on a slot nobody fills again."""
    pipe = Pipeline()
    ticks = []
    pipe.spawn("worker", lambda: (ticks.append(1), time.sleep(0.005))[0])
    pipe.spawn("doomed", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    pipe.join()
    assert not pipe.running
    before = len(ticks)
    time.sleep(0.03)
    assert len(ticks) == before


# --- the threaded loop ----------------------------------------------------

class StubCamera:
    """Hands out frames until told to stop, then reports the camera is done."""

    def __init__(self, size=SIZE, limit=None):
        self.size = size
        self.done = threading.Event()
        self.reads = 0
        self._limit = limit

    def read(self):
        if self.done.is_set() or (self._limit is not None and self.reads >= self._limit):
            return None
        self.reads += 1
        time.sleep(0.002)  # a camera does not return instantly
        return np.full((self.size[1], self.size[0], 3), 90, dtype=np.uint8)


class StubPerception:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        if self.result is not None:
            return self.result
        return PerceptionResult(
            width=SIZE[0], height=SIZE[1], timestamp_ms=self.calls * 33,
            pose=make_pose(), pose_visibility=np.ones(33, dtype=np.float32))


class Display:
    """Stands in for cv2.imshow/waitKey, and ends the loop on a condition."""

    def __init__(self, stop_after_drawn=3, max_passes=2000):
        self.shown: list[np.ndarray] = []
        self.drawn = 0
        self.passes = 0
        self._stop_after = stop_after_drawn
        self._max = max_passes

    def imshow(self, _window, image):
        self.shown.append(image)
        if image.std() > 1.0:  # something other than blank paper
            self.drawn += 1

    def wait_key(self, _delay):
        self.passes += 1
        if self.drawn >= self._stop_after or self.passes >= self._max:
            return ord("q")
        return 255


@pytest.fixture
def threaded(monkeypatch):
    """Run _run_threaded with stubs, returning the pieces for assertions."""

    def run(perception=None, show_camera=False, stop_after_drawn=3, render_workers=1):
        camera = StubCamera()
        perception = perception or StubPerception()
        display = Display(stop_after_drawn=stop_after_drawn)
        monkeypatch.setattr(app_loop.cv2, "imshow", display.imshow)
        monkeypatch.setattr(app_loop.cv2, "waitKey", display.wait_key)

        machine = StateMachine()
        args = argparse.Namespace(show_camera=show_camera, render_workers=render_workers)
        finished = threading.Event()

        def target() -> None:
            try:
                app_loop._run_threaded(camera, perception, machine, GestureInterpreter(),
                                       app_loop.FrameClock(), args)
            finally:
                finished.set()

        runner = threading.Thread(target=target)
        runner.start()
        assert finished.wait(15.0), "the threaded loop did not finish"
        runner.join(5.0)
        camera.done.set()
        return camera, perception, display, machine

    return run


def test_threaded_loop_draws_and_exits_cleanly(threaded):
    camera, perception, display, _ = threaded()
    assert camera.reads > 0, "capture stage never ran"
    assert perception.calls > 0, "perception stage never ran"
    assert display.drawn >= 3, "render stage produced nothing"


def test_threaded_loop_survives_an_empty_frame(threaded):
    """No person in frame is normal; the loop keeps showing blank paper."""
    perception = StubPerception(
        result=PerceptionResult(width=SIZE[0], height=SIZE[1], timestamp_ms=0))
    _, _, display, _ = threaded(perception=perception, stop_after_drawn=0)
    assert display.shown, "nothing was displayed"
    assert all(image.shape == (SIZE[1], SIZE[0], 3) for image in display.shown)


def test_threaded_loop_ends_when_the_camera_stops():
    """A dead camera has to end the run instead of leaving main waiting."""
    pipe = Pipeline()
    pipe.spawn("capture", lambda: False)
    pipe.join()
    assert not pipe.running


def test_renderer_sees_the_style_the_state_machine_holds(threaded, monkeypatch):
    seen: list[tuple[Style, Target]] = []
    real = app_loop.get_renderer

    def spy(style):
        renderer = real(style)

        class Recording:
            def render(self, result, target, size):
                seen.append((style, target))
                return renderer.render(result, target, size)

        return Recording()

    monkeypatch.setattr(app_loop, "get_renderer", spy)
    threaded()
    assert seen, "the render stage never asked for a renderer"
    assert all(entry == (Style.GESTURE_CROQUIS, Target.FULL_BODY) for entry in seen)


def test_show_camera_displays_the_frame(threaded):
    _, _, display, _ = threaded(show_camera=True, stop_after_drawn=0)
    # The stub camera's frame is flat grey 90; the croquis would be paper-coloured.
    assert any(abs(float(image[0, 0, 0]) - 90.0) < 1.0 for image in display.shown)


def test_several_render_workers_share_the_frames(threaded, monkeypatch):
    """Workers must claim different frames, not all redo the newest one."""
    rendered: list[int] = []
    lock = threading.Lock()
    real = app_loop.get_renderer

    def spy(style):
        renderer = real(style)

        class Recording:
            def render(self, result, target, size):
                with lock:
                    rendered.append(result.timestamp_ms)
                time.sleep(0.01)  # long enough that workers overlap
                return renderer.render(result, target, size)

        return Recording()

    monkeypatch.setattr(app_loop, "get_renderer", spy)
    threaded(render_workers=3, stop_after_drawn=6)
    assert len(rendered) >= 2
    # StubPerception stamps each result; a frame claimed twice means two workers
    # duplicated the work instead of splitting it.
    assert len(set(rendered)) == len(rendered), f"a frame was rendered twice: {rendered}"


def test_a_stateful_renderer_is_never_called_concurrently(threaded, monkeypatch):
    """Extra workers have to queue up on a renderer that smooths across frames."""
    overlaps = []
    active = [0]
    lock = threading.Lock()
    real = app_loop.get_renderer

    def spy(style):
        renderer = real(style)

        class Stateful:
            stateless = False  # what ConstructiveRenderer declares

            def render(self, result, target, size):
                with lock:
                    active[0] += 1
                    overlaps.append(active[0])
                time.sleep(0.01)
                with lock:
                    active[0] -= 1
                return renderer.render(result, target, size)

        return Stateful()

    monkeypatch.setattr(app_loop, "get_renderer", spy)
    threaded(render_workers=3, stop_after_drawn=5)
    assert overlaps, "the render stage never ran"
    assert max(overlaps) == 1, f"stateful renderer ran {max(overlaps)} at once"


# --- the single-threaded loop (--no-threads) ------------------------------

def test_single_threaded_loop_still_draws(monkeypatch):
    """The baseline path the threaded numbers are measured against."""
    camera = StubCamera(limit=5)
    perception = StubPerception()
    display = Display(stop_after_drawn=1000)  # let the camera end the run instead
    monkeypatch.setattr(app_loop.cv2, "imshow", display.imshow)
    monkeypatch.setattr(app_loop.cv2, "waitKey", display.wait_key)

    app_loop._run_single_threaded(camera, perception, StateMachine(), GestureInterpreter(),
                                  app_loop.FrameClock(),
                                  argparse.Namespace(show_camera=False, render_workers=1))

    assert camera.reads == 5
    assert perception.calls == 5
    assert display.drawn == 5, "every frame should have been rendered and shown"
