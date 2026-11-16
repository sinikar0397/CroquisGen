# 인체 도형화 렌더러의 기하 함수 테스트 (카메라/모델 필요 없음)
#   python -m pytest tests -q
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
    _palm_outline,
)
from croquisgen.types import L_ELBOW, L_INDEX, L_PINKY, L_SHOULDER, L_WRIST  # noqa: E402


def test_overlong_forearm_is_pulled_in_and_hand_follows():
    unit = 10.0  #머리 높이 px. 표준 전완 길이 = 1.25 * unit
    points = np.zeros((33, 2), dtype=np.float32)
    points[L_SHOULDER] = (0, 0)
    points[L_ELBOW] = (0, 15)
    points[L_WRIST] = (0, 15 + 60)  #4.8H짜리 전완 -> 검출 오류
    points[L_PINKY] = points[L_INDEX] = (0, 80)

    fixed = _correct_proportions(points, _BODY_CHAINS, unit, _BODY_FOLLOWERS)

    forearm = np.linalg.norm(fixed[L_WRIST] - fixed[L_ELBOW])
    assert forearm <= 1.25 * unit * LENGTH_TOLERANCE + 1e-4
    assert np.allclose(fixed[L_ELBOW], points[L_ELBOW])  #상완은 정상이라 안 움직여야 함
    # 손 점들은 손목이랑 같이 움직여서 상대 위치 유지
    assert np.allclose(fixed[L_PINKY] - fixed[L_WRIST], points[L_PINKY] - points[L_WRIST])


def test_foreshortening_follows_projected_length():
    a = np.zeros(2, dtype=np.float32)
    assert _foreshortening(a, np.array([10.0, 0.0]), canonical=10.0) == 0.0  #화면에 누움
    assert abs(_foreshortening(a, np.array([5.0, 0.0]), canonical=10.0) - 0.866) < 1e-3
    assert _foreshortening(a, a, canonical=10.0) == 1.0  #카메라 정면
    assert _foreshortening(a, np.array([20.0, 0.0]), canonical=10.0) == 0.0  #표준보다 길어도 음수 X


def test_smoother_only_blends_across_advancing_timestamps():
    smoother = _Smoother(alpha_xy=0.5, alpha_z=0.5)
    first = np.zeros((3, 3), dtype=np.float32)
    second = np.ones((3, 3), dtype=np.float32)

    assert np.array_equal(smoother.update(0, first), first)
    assert np.array_equal(smoother.update(0, second), second)  #같은 timestamp(정지 이미지) -> 섞지 않음
    assert np.allclose(smoother.update(33, first), 0.5)  #다음 프레임 -> 절반씩
    assert np.array_equal(smoother.update(5000, second), second)  #오래 끊기면 리셋


def _flat_hand(thumb_mcp):  #손가락은 위로 쭉 편 손. 엄지 MCP 위치만 바꿔서 씀
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = (50, 90)
    for base, x in ((5, 40), (9, 48), (13, 56), (17, 64)):
        for i in range(4):
            pts[base + i] = (x, 55 - 12 * i)
    pts[1] = (44, 80)
    pts[2] = thumb_mcp
    pts[3] = pts[2] + (-6, -6)
    pts[4] = pts[3] + (-5, -5)
    return pts


def test_palm_outline_includes_thumb_only_when_it_sticks_out():
    spread = _palm_outline(_flat_hand((30, 70)), P=40.0)  #엄지가 손바닥 옆으로 나감
    folded = _palm_outline(_flat_hand((50, 70)), P=40.0)  #엄지 MCP가 손바닥 안
    # 손목 모서리 2 + 손가락 4개 x 밑동 2 = 10, 엄지가 나가 있으면 +2
    assert len(folded) == 10
    assert len(spread) == 12
