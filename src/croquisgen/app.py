"""Main loop: camera -> perception -> gesture events -> render -> display.

Run with `python -m croquisgen.app` from src/, or `python scripts/run.py`.

Keyboard shortcuts mirror every gesture. They are not a convenience - debugging a
renderer while holding poses in front of a camera is impractical, and the gesture
layer needs to be bypassable when the thing under test is the drawing code.
"""
from __future__ import annotations

import argparse

import cv2

from . import config
from .capture import Camera
from .gestures import GestureInterpreter
from .perception import FrameClock, Perception
from .renderers import get_renderer
from .state import StateMachine
from .types import INDEX_TIP, Style
from .ui.overlay import draw_camera_inset, draw_status_bar, draw_toast
from .ui.radial_menu import draw_hold_progress, draw_menu
     
WINDOW = "CroquisGen"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time croquis reference generator")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--lite", action="store_true",
                        help="use the lite pose model (~15 ms vs ~25 ms per frame)")
    parser.add_argument("--no-snap", action="store_true",
                        help="disable snap detection; use the 'm' key to switch target")
    parser.add_argument("--show-camera", action="store_true",
                        help="draw the live camera feed instead of the croquis")
    args = parser.parse_args(argv)

    pose_model = config.MODEL_DIR / (
        "pose_landmarker_lite.task" if args.lite else "pose_landmarker_full.task"
    )

    machine = StateMachine()
    interpreter = GestureInterpreter(enable_snap=not args.no_snap)
    clock = FrameClock()

    with Camera(index=args.camera) as camera, Perception(pose_model=pose_model) as perception:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        width, height = camera.size

        while True:
            frame = camera.read()
            if frame is None:
                break

            result = perception.process(frame)
            events = interpreter.update(result)

            renderer = get_renderer(machine.state.style)
            croquis = renderer.render(result, machine.state.target, (width, height))

            # Capture saves the croquis, not the camera frame - the drawing is the
            # artefact the user wants.
            machine.apply(events, croquis)
            machine.tick()

            display = frame.copy() if args.show_camera else croquis
            if not args.show_camera:
                draw_camera_inset(display, frame)

            _draw_hand_affordances(display, result, interpreter, machine)
            draw_status_bar(display, machine.state, clock.tick(), camera_ok=True)
            draw_toast(display, machine.state)

            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            _handle_key(key, machine, croquis)

    cv2.destroyAllWindows()
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
