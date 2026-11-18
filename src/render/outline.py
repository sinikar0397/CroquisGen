"""Silhouette outline from the pose segmentation mask.

Shared by both styles rather than being a style of its own.

The plan's original approach was a gradient or Laplacian filter over the raw
frame. That picks up whatever is behind the subject - furniture, wall texture,
clothing folds - and the croquis ends up full of lines that are not the body.
PoseLandmarker already returns a person probability mask for about 2 ms, so the
contour comes from the mask boundary instead and contains only the body.
"""
from __future__ import annotations

import cv2
import numpy as np

from core import config


def silhouette_contours(
    segmentation: np.ndarray,
    size: tuple[int, int],
    threshold: float = 0.5,
    smoothing: float = 0.004,
    min_area_ratio: float = 0.01,
) -> list[np.ndarray]:
    """Contours of the person, in pixel coordinates of an image of `size`.

    `smoothing` is a fraction of the image diagonal fed to the polygon simplifier.
    Raising it gives fewer, straighter edges, which reads as a more confident line;
    lowering it follows clothing folds.
    """
    width, height = size
    mask = (segmentation.squeeze() > threshold).astype(np.uint8)
    if not mask.any():
        return []

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # Close pinholes so the outline does not break at thin limbs or between fingers.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diagonal = float(np.hypot(width, height))
    min_area = min_area_ratio * width * height

    contours = []
    for contour in found:
        if cv2.contourArea(contour) < min_area:
            continue
        contours.append(cv2.approxPolyDP(contour, smoothing * diagonal, closed=True))
    return contours


def draw_silhouette(
    canvas: np.ndarray,
    segmentation: np.ndarray | None,
    thickness: int = 2,
    color: tuple[int, int, int] = config.INK_COLOR,
    alpha: float = 0.35,
) -> None:
    """Draw the outline faintly over `canvas`, in place.

    Kept light on purpose - it is context for the structural lines, not the
    subject of the drawing.
    """
    if segmentation is None:
        return
    height, width = canvas.shape[:2]
    contours = silhouette_contours(segmentation, (width, height))
    if not contours:
        return

    overlay = canvas.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)
