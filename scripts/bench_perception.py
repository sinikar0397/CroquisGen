"""Measure the frame budget on this machine.

Rerun after changing models or cadence. The numbers quoted in config.py and
CLAUDE.md came from this script on an M-series CPU at 1280x720.

    python scripts/bench_perception.py photo.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from croquisgen import config

RUNS = 20


def _image(frame: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def bench_pose(frame, model: Path, seg: bool, video: bool) -> float:
    image = _image(frame)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=seg,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        return _time(landmarker, image, video, "detect")


def bench_hand(frame, model: Path, video: bool) -> float:
    image = _image(frame)
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=1,
    )
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        return _time(recognizer, image, video, "recognize")


def _time(obj, image, video: bool, verb: str) -> float:
    if video:
        call = getattr(obj, f"{verb}_for_video")
        for i in range(3):
            call(image, i)
        start = time.perf_counter()
        for i in range(RUNS):
            call(image, 100 + i * 33)
    else:
        call = getattr(obj, verb)
        call(image)
        start = time.perf_counter()
        for _ in range(RUNS):
            call(image)
    return (time.perf_counter() - start) / RUNS * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="a still with a person in it")
    args = parser.parse_args(argv)

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 1
    print(f"{frame.shape[1]}x{frame.shape[0]}, {RUNS} runs each\n")

    rows = []
    for mode_name, video in (("IMAGE", False), ("VIDEO", True)):
        for model_name in ("pose_landmarker_full.task", "pose_landmarker_lite.task"):
            for seg in (True, False):
                ms = bench_pose(frame, config.MODEL_DIR / model_name, seg, video)
                rows.append((f"pose {model_name.split('_')[-1][:-5]:<5} seg={seg!s:<5}",
                             mode_name, ms))
        rows.append(("hand + gesture        ", mode_name,
                     bench_hand(frame, config.GESTURE_MODEL, video)))

    for label, mode, ms in rows:
        print(f"  {label}  {mode}  {ms:6.1f} ms  ({1000 / ms:4.0f} fps)")

    pose = next(ms for label, mode, ms in rows
                if label.startswith("pose full  seg=True") and mode == "VIDEO")
    hand = next(ms for label, mode, ms in rows
                if label.startswith("hand") and mode == "VIDEO")
    combined = pose + hand / config.HAND_EVERY_N_FRAMES
    print(f"\n  app loop estimate: pose {pose:.0f} + hand {hand:.0f}/"
          f"{config.HAND_EVERY_N_FRAMES} = {combined:.0f} ms "
          f"({1000 / combined:.0f} fps) before rendering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
