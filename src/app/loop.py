"""Main loop: camera -> perception -> gesture events -> render -> display.

Run with `python main.py` from the repository root.

Keyboard shortcuts mirror every gesture. They are not a convenience - debugging a
renderer while holding poses in front of a camera is impractical, and the gesture
layer needs to be bypassable when the thing under test is the drawing code.

There are two loops here, picked with --no-threads.

**Threaded (default).** Capture, perception and rendering each get a thread, so
the frame cost is the slowest stage instead of their sum:

    capture thread      read -> flip        -> newest frame
    perception thread   pose, then hand     -> newest PerceptionResult
    render thread(s)    renderer.render()   -> newest croquis
    main thread         gestures, overlay, imshow

--render-workers runs more than one render thread. It only applies to a renderer
with no frame-to-frame state (CroquisRenderer.stateless); the others are
serialised on a lock, because the style can change mid-run. Workers claim
different frames and tag what they publish, so one that falls behind drops its
croquis rather than replacing a newer one. On this machine it is not worth using
- see the measurements on config.RENDER_WORKERS - but the plumbing is here for a
machine with more performance cores.

This works because the expensive work is native and releases the GIL (MediaPipe
inference, OpenCV drawing), which tools/bench_threading.py measures directly
rather than assuming. Display stays on the main thread because cv2.imshow talks
to Cocoa, and on macOS that is only legal there.

The croquis lags the camera by up to one stage per hop. For a pose the user is
holding that is invisible, and the gesture affordances do not lag: the main
thread draws the hold ring and the radial menu straight from the newest
perception result, not from the rendered image.

**Single-threaded (--no-threads).** The original straight-line loop, kept as the
baseline the threaded numbers are compared against and as the thing to fall back
to when a renderer misbehaves and threads are in the way of reading the error.
"""
from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from core import config
from core.landmarks import INDEX_TIP
from core.types import PerceptionResult, Style
from perception import get_backend
from render.base import blank_paper
from render.registry import get_renderer

from .camera import Camera
from .gestures import GestureInterpreter
from .pipeline import Pipeline, Slot
from .state import StateMachine
from .ui.overlay import draw_camera_inset, draw_status_bar, draw_toast
from .ui.radial_menu import draw_hold_progress, draw_menu

WINDOW = "CroquisGen"

# How long a worker blocks on an empty slot before re-checking the stop flag.
# Only bounds shutdown time; the slots wake their waiters as soon as a value
# lands, so this is not a polling interval.
_STAGE_TIMEOUT = 0.1
# The main thread wakes on every new croquis. This caps how often it redraws when
# no new one has arrived, which is what keeps the keyboard responsive without
# spinning on waitKey.
_DISPLAY_TIMEOUT = 1 / 60


class FrameClock:
    """Frames per second, counted over a sliding window.

    This used to smooth 1/dt exponentially, which reads high whenever the
    interval jitters - the mean of 1/dt is not 1/(mean dt), and the gap grows
    with the jitter. The threaded loop jitters more than the single-threaded one,
    so that bias would have made the status bar flatter the very thing it is
    there to measure. Counting frames over a window has no such bias and makes
    the two paths comparable.
    """

    def __init__(self, window: float = 0.5) -> None:
        self._window = window
        self._start = time.perf_counter()
        self._frames = 0
        self.fps = 0.0

    def tick(self, frames: int = 1) -> float:
        """Record `frames` frames. The threaded loop can see several at once."""
        self._frames += max(frames, 1)
        now = time.perf_counter()
        elapsed = now - self._start
        if elapsed >= self._window:
            self.fps = self._frames / elapsed
            self._start, self._frames = now, 0
        return self.fps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time croquis reference generator")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--lite", action="store_true",
                        help="use the lite pose model (~15 ms vs ~25 ms per frame)")
    parser.add_argument("--no-snap", action="store_true",
                        help="disable snap detection; use the 'm' key to switch target")
    parser.add_argument("--show-camera", action="store_true",
                        help="draw the live camera feed instead of the croquis")
    parser.add_argument("--no-threads", action="store_true",
                        help="run capture, perception and rendering in one thread")
    parser.add_argument("--render-workers", type=int, default=config.RENDER_WORKERS,
                        metavar="N",
                        help=f"render threads (default {config.RENDER_WORKERS}); "
                             "ignored with --no-threads")
    engine = parser.add_mutually_exclusive_group()
    engine.add_argument("--s4", dest="engine", action="store_const", const="s4",
                        help="gesture croquis via the real S4 stage (the default; "
                             "~56 ms/frame, so the croquis updates ~13 times a second)")
    engine.add_argument("--placeholder", dest="engine", action="store_const",
                        const="placeholder",
                        help="gesture croquis via the cheap geometric stand-in (~7 ms)")
    args = parser.parse_args(argv)

    if args.engine is not None:
        # The renderer reads this per frame, so setting it here is enough.
        config.GESTURE_CROQUIS_ENGINE = args.engine

    pose_model = config.MODEL_DIR / (
        "pose_landmarker_lite.task" if args.lite else "pose_landmarker_full.task"
    )

    machine = StateMachine()
    interpreter = GestureInterpreter(enable_snap=not args.no_snap)
    clock = FrameClock()

    with Camera(index=args.camera) as camera, get_backend(pose_model=pose_model) as perception:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        run = _run_single_threaded if args.no_threads else _run_threaded
        run(camera, perception, machine, interpreter, clock, args)

    cv2.destroyAllWindows()
    return 0


def _run_single_threaded(camera, perception, machine, interpreter, clock, args) -> None:
    """One stage after another, as the app ran before the threaded loop."""
    size = camera.size

    while True:
        frame = camera.read()
        if frame is None:
            break

        result = perception.process(frame)
        events = interpreter.update(result)

        renderer = get_renderer(machine.state.style)
        croquis = renderer.render(result, machine.state.target, size)

        # Capture saves the croquis, not the camera frame - the drawing is the
        # artefact the user wants.
        machine.apply(events, croquis)
        machine.tick()
        clock.tick()

        # The croquis is this iteration's own canvas, so the overlay can go
        # straight onto it; the threaded loop has to copy instead.
        display = _compose(croquis, frame, result, machine, interpreter, clock.fps, args,
                           size, shared=False)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        _handle_key(key, machine, croquis)


def _run_threaded(camera, perception, machine, interpreter, clock, args) -> None:
    """Three worker threads feeding newest-value slots, UI on the main thread."""
    size = camera.size
    frames: Slot[np.ndarray] = Slot()
    results: Slot[PerceptionResult] = Slot()
    drawings: Slot[np.ndarray] = Slot()
    pipe = Pipeline()

    def capture_step() -> bool | None:
        frame = camera.read()
        if frame is None:
            return False  # camera gone: ends the run
        frames.put(frame)
        return None

    seen_frame = 0

    def perception_step() -> None:
        nonlocal seen_frame
        frame, seq = frames.wait(seen_frame, _STAGE_TIMEOUT)
        if seq == seen_frame or frame is None:
            return
        seen_frame = seq
        # pose and hand stay together here, cadence and all, so this stage is
        # the one place MediaPipe's stateful VIDEO-mode trackers are touched.
        results.put(perception.process(frame))

    # Shared by the render workers so each one picks up a different frame. With
    # one worker this is just the "have I seen it" counter the other stages use.
    claimed = 0
    claim_lock = threading.Lock()
    # Serialises the renderers that keep state between frames. The style can
    # change mid-run from the radial menu, so the worker count cannot be decided
    # once at start-up from whichever renderer happened to be selected: extra
    # workers queue up here on a stateful renderer and run free on a stateless
    # one.
    stateful_lock = threading.Lock()

    def render_step() -> None:
        nonlocal claimed
        result, seq = results.wait(claimed, _STAGE_TIMEOUT)
        if result is None:
            return
        with claim_lock:
            if seq <= claimed:
                return  # another worker already took this frame
            claimed = seq
        # Two attribute reads of immutable enums the main thread may have just
        # replaced. The cost of being stale is drawing one frame in the style
        # selected a moment ago, which is not worth a lock.
        state = machine.state
        renderer = get_renderer(state.style)
        if getattr(renderer, "stateless", False):
            image = renderer.render(result, state.target, size)
        else:
            with stateful_lock:
                image = renderer.render(result, state.target, size)
        # Tagged with the frame it came from: a worker that fell behind drops its
        # result instead of replacing a newer croquis.
        drawings.put_newer(image, seq)

    pipe.spawn("capture", capture_step)
    pipe.spawn("perception", perception_step)
    for index in range(max(1, int(args.render_workers))):
        pipe.spawn(f"render{index}", render_step)

    croquis: np.ndarray | None = None
    seen_drawing = 0
    handled_result = 0

    try:
        while pipe.running:
            drawing, seq = drawings.wait(seen_drawing, _DISPLAY_TIMEOUT)
            if seq != seen_drawing and drawing is not None:
                # Ticked per croquis, not per display pass: the status bar should
                # read as the rate the drawing updates, and toast_frames is
                # counted in croquis frames the way the single-threaded loop had
                # it. The seq delta covers croquis the render stage produced
                # while main was busy with the previous one.
                clock.tick(seq - seen_drawing)
                seen_drawing, croquis = seq, drawing
                machine.tick()

            result, result_seq = results.latest()
            if result is not None and result_seq != handled_result:
                handled_result = result_seq
                machine.apply(interpreter.update(result), croquis)

            frame, _ = frames.latest()
            display = _compose(croquis, frame, result, machine, interpreter, clock.fps, args,
                               size, shared=True)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            _handle_key(key, machine, croquis)
    finally:
        pipe.stop()
        pipe.join()

    # A stage that raised has stopped the run quietly; surface it as a traceback.
    pipe.raise_if_failed()


def _compose(croquis, frame, result, machine, interpreter, fps, args, size, *, shared: bool):
    """Build the image to show: croquis or camera, plus every overlay."""
    if args.show_camera:
        display = frame.copy() if frame is not None else blank_paper(size)
    elif croquis is None:
        display = blank_paper(size)  # nothing rendered yet
    else:
        # In the threaded loop the same croquis is shown across several display
        # passes, so overlays have to go on a copy or they accumulate.
        display = croquis.copy() if shared else croquis
        if frame is not None:
            draw_camera_inset(display, frame)

    if result is not None:
        _draw_hand_affordances(display, result, interpreter, machine)
    draw_status_bar(display, machine.state, fps, camera_ok=True)
    draw_toast(display, machine.state)
    return display


def _draw_hand_affordances(display, result, interpreter, machine) -> None:
    """Hold ring and radial menu, both anchored to the fingertip."""
    points = result.hand_px()
    if points is None:
        return
    tip = (float(points[INDEX_TIP][0]), float(points[INDEX_TIP][1]))

    if machine.state.menu_open and machine.state.menu_anchor is not None:
        draw_menu(display, machine.state.menu_anchor, machine.state.style,
                  highlight=machine.state.menu_highlight)
        cv2.circle(display, (int(tip[0]), int(tip[1])), 8, (209, 199, 61), -1, cv2.LINE_AA)
    else:
        draw_hold_progress(display, tip, interpreter.menu_progress(result))


def _handle_key(key: int, machine: StateMachine, croquis) -> None:
    if key == ord("m"):
        machine.toggle_target()
    elif key == ord("1"):
        machine.set_style(Style.GESTURE_CROQUIS)
    elif key == ord("2"):
        machine.set_style(Style.CONSTRUCTIVE)
    elif key == ord("c"):
        machine.capture(croquis)  # same path the capture gesture takes
