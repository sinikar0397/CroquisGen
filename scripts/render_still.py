"""Render a croquis from a still image instead of the webcam.

The fast way to work on a renderer: one fixed input, both styles side by side, no
posing in front of a camera and no frame budget. Also the only way to work on the
drawing code on a machine without a webcam.

    python scripts/render_still.py photo.jpg -o out.png
    python scripts/render_still.py photo.jpg --target hand
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from croquisgen import config
from croquisgen.perception import Perception
from croquisgen.renderers import get_renderer, renderer_name
from croquisgen.types import Style, Target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("still_croquis.png"))
    parser.add_argument("--target", choices=[t.value for t in Target],
                        default=Target.FULL_BODY.value)
    parser.add_argument("--style", choices=[s.value for s in Style],
                        help="render one style instead of a side-by-side of all")
    args = parser.parse_args(argv)

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 1

    target = Target(args.target)
    styles = [Style(args.style)] if args.style else list(Style)

    with Perception() as perception:
        # Perception runs in VIDEO mode, which tracks across calls. On a still
        # image the first call is a cold detect, so run it a few times and keep the
        # last result to match what the app sees mid-stream.
        for _ in range(3):
            result = perception.process(frame)

    print(f"pose: {'yes' if result.has_pose() else 'no'}   "
          f"hand: {'yes' if result.has_hand() else 'no'}   "
          f"gesture: {result.gesture or '-'} ({result.gesture_score:.2f})")
    if target is Target.FULL_BODY and not result.has_pose():
        print("no person detected", file=sys.stderr)
        return 1
    if target is Target.HAND and not result.has_hand():
        print("no hand detected", file=sys.stderr)
        return 1

    height, width = frame.shape[:2]
    panels = []
    for style in styles:
        panel = get_renderer(style).render(result, target, (width, height))
        cv2.putText(panel, renderer_name(style), (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, config.INK_COLOR, 2, cv2.LINE_AA)
        panels.append(panel)

    out = np.hstack(panels) if len(panels) > 1 else panels[0]
    cv2.imwrite(str(args.output), out)
    print(f"wrote {args.output} ({out.shape[1]}x{out.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
