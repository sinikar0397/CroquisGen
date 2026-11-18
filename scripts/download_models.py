"""Fetch the MediaPipe task bundles into models/.

The bundles are ~23 MB together and are not in the repository.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config import MODEL_DIR  # noqa: E402  (path set above)

BASE_URL = "https://storage.googleapis.com/mediapipe-models"

# hand_landmarker is deliberately absent: GestureRecognizer returns the same 21
# landmarks plus a pre-trained label at the same cost, so nothing ever loaded it.
MODELS = {
    "pose_landmarker_full.task": "pose_landmarker/pose_landmarker_full/float16/latest",
    "pose_landmarker_lite.task": "pose_landmarker/pose_landmarker_lite/float16/latest",
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
