"""Constructive anatomy: the pure geometry behind the drawing.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from croquisgen.renderers.constructive import (  # noqa: E402
    _BODY_CHAINS,
    _BODY_FOLLOWERS,
    LENGTH_TOLERANCE,
    _Smoother,
    _correct_proportions,
    _foreshortening,
)
from croquisgen.types import L_ELBOW, L_INDEX, L_PINKY, L_SHOULDER, L_WRIST  # noqa: E402


def test_overlong_forearm_is_pulled_in_and_hand_follows():
    unit = 10.0  # head height in px; canonical forearm is 1.25 * unit
    points = np.zeros((33, 2), dtype=np.float32)
    points[L_SHOULDER] = (0, 0)
    points[L_ELBOW] = (0, 15)
    points[L_WRIST] = (0, 15 + 60)      # 4.8 heads: a detection error
    points[L_PINKY] = points[L_INDEX] = (0, 80)

    fixed = _correct_proportions(points, _BODY_CHAINS, unit, _BODY_FOLLOWERS)

    forearm = np.linalg.norm(fixed[L_WRIST] - fixed[L_ELBOW])
    assert forearm <= 1.25 * unit * LENGTH_TOLERANCE + 1e-4
    # Upper arm was plausible and must not move.
    assert np.allclose(fixed[L_ELBOW], points[L_ELBOW])
    # The hand points move with the wrist, keeping their offset from it.
    assert np.allclose(fixed[L_PINKY] - fixed[L_WRIST], points[L_PINKY] - points[L_WRIST])


def test_foreshortening_follows_projected_length():
    a = np.zeros(2, dtype=np.float32)
    assert _foreshortening(a, np.array([10.0, 0.0]), canonical=10.0) == 0.0
    assert abs(_foreshortening(a, np.array([5.0, 0.0]), canonical=10.0) - 0.866) < 1e-3
    assert _foreshortening(a, a, canonical=10.0) == 1.0
    # Longer than canonical never goes negative.
    assert _foreshortening(a, np.array([20.0, 0.0]), canonical=10.0) == 0.0


def test_smoother_only_blends_across_advancing_timestamps():
    smoother = _Smoother(alpha_xy=0.5, alpha_z=0.5)
    first = np.zeros((3, 3), dtype=np.float32)
    second = np.ones((3, 3), dtype=np.float32)

    assert np.array_equal(smoother.update(0, first), first)
    # Same timestamp again (a still image, a test): no blending.
    assert np.array_equal(smoother.update(0, second), second)
    # Advancing timestamp: halfway between.
    assert np.allclose(smoother.update(33, first), 0.5)
    # A long gap resets to the new input.
    assert np.array_equal(smoother.update(5000, second), second)
