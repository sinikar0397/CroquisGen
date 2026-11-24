"""Plumbing for the threaded loop: latest-value slots and worker threads.

Only app/loop.py uses this. It is here rather than in core/ because threading is
an app concern - the web build gets its concurrency from the browser instead, and
this file is part of what gets thrown away then.

Two decisions are worth knowing about.

**Slots hold one value, not a queue.** A queue in front of a slow stage grows,
and every frame it holds is latency the user sees in the croquis. Dropping all
but the newest frame keeps the lag bounded by the stage times. Stages are
therefore allowed to run at different rates and to miss each other's output;
nothing in the pipeline needs every frame.

**A stage that dies stops the run.** Without that, main would block on a slot
nobody fills any more. Exceptions are kept and re-raised on the main thread, so
a MediaPipe error still shows a traceback instead of a silent freeze.

Python threads overlap here because the expensive work is native and releases the
GIL - MediaPipe inference and OpenCV drawing both do. Measured with
tools/bench_threading.py; a counter in another thread keeps 91-98% of its idle
rate during a pose call.
"""
from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Slot(Generic[T]):
    """A mailbox holding only the newest value, with a sequence number.

    The sequence number is what lets a consumer tell a fresh value from the one
    it already handled: values repeat (a still pose renders the same croquis),
    so comparing the values themselves would not work.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: T | None = None
        self._seq = 0
        # Highest tag accepted by put_newer; None while only put() has been used.
        self._tag: int | None = None

    def put(self, value: T) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._tag = None
            self._cond.notify_all()

    def put_newer(self, value: T, tag: int) -> bool:
        """Accept `value` only if `tag` beats every tag put so far.

        Several render workers process different frames at once and can finish
        out of order: a worker that started on an older frame must not overwrite
        a newer croquis that landed while it was busy. Its work is wasted, which
        is the right trade - the alternative is the drawing jumping backwards.
        """
        with self._cond:
            if self._tag is not None and tag <= self._tag:
                return False
            self._value = value
            self._tag = tag
            self._seq += 1
            self._cond.notify_all()
            return True

    def latest(self) -> tuple[T | None, int]:
        """The current value and its sequence number. Never blocks."""
        with self._cond:
            return self._value, self._seq

    def wait(self, seen: int, timeout: float) -> tuple[T | None, int]:
        """Block until the sequence number differs from `seen`, or timeout.

        A timeout returns the unchanged sequence number, which is how a caller
        tells "nothing new" from "something new" - and why every stage gets back
        to its stop check even when upstream has gone quiet.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seen, timeout)
            return self._value, self._seq


class Pipeline:
    """Stop flag and error capture shared by the worker threads."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._errors: list[tuple[str, BaseException]] = []
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def spawn(self, name: str, step: Callable[[], bool | None]) -> None:
        """Call step() in a loop on its own thread until the pipeline stops.

        `step` returning False ends the run - that is how the capture stage
        reports that the camera is finished.
        """

        def loop() -> None:
            try:
                while self.running:
                    if step() is False:
                        break
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                with self._lock:
                    self._errors.append((name, exc))
            finally:
                # Whatever ended this stage ends the run: downstream stages would
                # otherwise wait on a slot that will never be filled again.
                self.stop()

        thread = threading.Thread(target=loop, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def join(self, timeout: float = 2.0) -> None:
        for thread in self._threads:
            thread.join(timeout)

    def raise_if_failed(self) -> None:
        """Re-raise the first stage exception, named after the stage."""
        with self._lock:
            if not self._errors:
                return
            name, exc = self._errors[0]
        raise RuntimeError(f"{name} stage failed") from exc
