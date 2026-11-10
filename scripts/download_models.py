"""Fetch the MediaPipe task bundles into models/.

The bundles are ~31 MB together and are not in the repository.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/mediapipe-models"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

MODELS = {
    "pose_landmarker_full.task": "pose_landmarker/pose_landmarker_full/float16/latest",
    "pose_landmarker_lite.task": "pose_landmarker/pose_landmarker_lite/float16/latest",
    "hand_landmarker.task": "hand_landmarker/hand_landmarker/float16/latest",
    "gesture_recognizer.task": "gesture_recognizer/gesture_recognizer/float16/latest",
}


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in MODELS.items():
        target = MODEL_DIR / name
        if target.exists():
            print(f"  have  {name}")
            continue
        url = f"{BASE_URL}/{path}/{name}"
        print(f"  get   {name} ...", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f" failed: {exc}")
            return 1
        print(f" {target.stat().st_size / 1e6:.1f} MB")
    print(f"\nmodels in {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
