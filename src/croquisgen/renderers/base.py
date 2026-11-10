"""The renderer contract.

A renderer takes a PerceptionResult and returns an image. It never touches the
camera, the OpenCV window or application state. Two reasons that boundary is
strict: the two conversion models are being written by different people and only
have to satisfy this one signature, and a pure landmarks->image function is what
ports cleanly to the browser later, where MediaPipe produces the same landmarks
from JS.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .. import config
from ..types import PerceptionResult, Target


class CroquisRenderer(Protocol):
    """Implemented by each croquis style."""

    name: str

    def render(
        self, result: PerceptionResult, target: Target, size: tuple[int, int]
    ) -> np.ndarray:
        """Draw one frame.

        Args:
            result: landmarks for this frame. May have no pose or no hand - return
                blank paper rather than raising.
            target: whether to draw the whole body or just the hand.
            size: (width, height) of the image to return.

        Returns:
            A BGR uint8 image of exactly `size`.
        """
        ...


def blank_paper(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = config.PAPER_COLOR
    return canvas


def scaled_points(
    normalised: np.ndarray, size: tuple[int, int], fit: float | None = None
) -> np.ndarray:
    """Normalised landmarks -> pixel coordinates in an image of `size`.

    `fit` is a margin fraction; passing it re-centres and zooms the subject to fill
    the canvas, which is what hand mode needs - a hand occupies a small part of the
    frame, and drawing it at frame scale would waste most of the paper. Note that
    stroke widths then have to be derived from the returned points, not from the
    frame size, or they end up scaled for the wrong subject.
    """
    width, height = size
    points = normalised[:, :2].astype(np.float32).copy()

    if fit is not None:
        margin = fit
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        scale = (1.0 - 2 * margin) / span.max()
        centre = (lo + hi) / 2.0
        points = (points - centre) * scale + 0.5

    points[:, 0] *= width
    points[:, 1] *= height
    return points


def tapered_line(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
    color: tuple[int, int, int] = config.INK_COLOR,
) -> None:
    """A stroke that changes thickness along its length.

    Uniform-width lines are what make generated skeletons read as diagrams rather
    than drawings; varying the width is most of the difference.
    """
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        return
    normal = np.array([-direction[1], direction[0]], dtype=np.float32) / length

    quad = np.array(
        [
            start + normal * start_width / 2,
            end + normal * end_width / 2,
            end - normal * end_width / 2,
            start - normal * start_width / 2,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, quad, color, lineType=cv2.LINE_AA)


def body_scale(points: np.ndarray) -> float:
    """A length to size strokes and shapes against: the shoulder span.

    Everything derived from this stays consistent whether the subject is close to
    the camera or far from it.
    """
    from ..types import L_SHOULDER, R_SHOULDER

    span = float(np.linalg.norm(points[L_SHOULDER] - points[R_SHOULDER]))
    return max(span, 1.0)


def visible(
    result: PerceptionResult, *indices: int, threshold: float = config.MIN_VISIBILITY
) -> bool:
    """True when every listed pose landmark is confidently seen.

    Skipping low-confidence limbs is better than drawing them: a guessed elbow
    produces a bent arm that is not there, and the point of the reference image is
    anatomical correctness.
    """
    if result.pose_visibility is None:
        return False
    return all(float(result.pose_visibility[i]) >= threshold for i in indices)
