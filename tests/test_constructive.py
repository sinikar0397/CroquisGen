# 인체 도형화 렌더러의 geometric function test
#   python -m pytest tests -q
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constructive_anatomy.renderer import (  # noqa: E402
    _BODY_FOLLOWERS,
    _BUILDS,
    LENGTH_TOLERANCE,
    _Smoother,
    _correct_proportions,
    _foreshortening,
    _palm_normal,
    _palm_outline,
    _slab_outline,
)
from core.landmarks import L_ELBOW, L_INDEX, L_PINKY, L_SHOULDER, L_WRIST  # noqa: E402


def test_overlong_forearm_is_pulled_in_and_hand_follows():
    unit = 10.0  #머리 높이 px. 표준 전완 길이 = 1.25 * unit
    points = np.zeros((33, 2), dtype=np.float32)
    points[L_SHOULDER] = (0, 0)
    points[L_ELBOW] = (0, 15)
    points[L_WRIST] = (0, 15 + 60)  #4.8H짜리 전완 -> 검출 오류
    points[L_PINKY] = points[L_INDEX] = (0, 80)

    fixed = _correct_proportions(points, _BUILDS['adult']['chains'], unit, _BODY_FOLLOWERS)

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
    # 엄지가 손가락 쪽으로 누우면(손등 보이는 포즈) 밑동이 윤곽을 파고들어 홈이 생기니 빼야 함
    pts = _flat_hand((36, 62))
    pts[3], pts[4] = (28, 74), (22, 84)
    down = _palm_outline(pts, P=40.0)
    assert len(down) == 10


def test_palm_slab_gets_thickness_when_seen_edge_on():
    import cv2
    square = np.array([(20, 20), (40, 20), (40, 40), (20, 40)], np.float32)
    flat = _slab_outline(square, np.zeros(2, np.float32), (100, 100))  #정면 -> 그냥 사각형
    side = _slab_outline(square, np.array([10.0, 0.0], np.float32), (100, 100))  #옆으로 10px 두께
    assert len(flat) == 4
    assert cv2.boundingRect(side)[2] == cv2.boundingRect(flat)[2] + 10

    # 법선은 항상 카메라 쪽(z가 작은 쪽)을 향해야 함
    pts = _flat_hand((30, 70))
    z = np.zeros(21, np.float32)
    z[17] = 20  #소지 쪽이 멀어진 손
    n = _palm_normal(pts, z)
    assert n[2] < 0
    assert _palm_normal(pts, -z)[2] < 0


def test_hand_drawing_stays_inside_the_canvas():
    # 손가락이 카메라 쪽으로 누워 랜드마크 상자가 납작한 손: fit 확대가 커져서 손바닥 모서리가 화면을 넘기 쉬움
    from constructive_anatomy.renderer import PAPER, ConstructiveRenderer
    from core.types import PerceptionResult, Target
    pts = _flat_hand((30, 70))
    for base in (5, 9, 13, 17):
        pts[base:base + 4, 1] = 55
    pts[0] = (50, 62)
    hand = np.column_stack([pts / 100.0, np.zeros(21)]).astype(np.float32)
    img = ConstructiveRenderer().render(
        PerceptionResult(width=640, height=480, timestamp_ms=1, hand=hand), Target.HAND, (640, 480))
    border = np.concatenate([img[:2].reshape(-1, 3), img[-2:].reshape(-1, 3),
                             img[:, :2].reshape(-1, 3), img[:, -2:].reshape(-1, 3)])
    assert (border == PAPER).all()


def test_cylinder_fill_stays_inside_the_cylinder():
    # 납작하게 누운 원기둥(squash 0)의 끝 바로 너머에 있는 선은 지워지면 안 된다
    import cv2
    from constructive_anatomy.renderer import INK, PAPER, _Scene, _add_cylinder
    canvas = np.full((100, 100, 3), PAPER, np.uint8)
    cv2.line(canvas, (56, 20), (56, 80), INK, 1)  #끝(x=50)에서 6px 너머
    scene = _Scene()
    _add_cylinder(scene, 0.0, np.array([10.0, 50.0]), np.array([50.0, 50.0]), 10, 10,
                  squash=0.0, near_is_end=False, thick=1, thin=1)
    scene.paint(canvas)
    assert (canvas[50, 56] < 200).any()  #아직 잉크


def _world_figure(heads):  #등신 수만큼 키운 3D 모형 (미터). 머리 0.23m 고정, 몸통:다리 = 2.5:3.9
    from core.landmarks import L_ANKLE, L_EAR, L_HIP, L_KNEE, R_ANKLE, R_EAR, R_HIP, R_KNEE
    head = 0.23
    body = (heads - 1.6) * head
    torso, legs = body * 2.5 / 6.4, body * 3.9 / 6.4
    w = np.zeros((33, 3), dtype=np.float32)
    w[L_EAR], w[R_EAR] = (-0.5 * 0.66 * head, 0, 0), (0.5 * 0.66 * head, 0, 0)
    w[L_SHOULDER], w[12] = (-0.2, 0.2, 0), (0.2, 0.2, 0)
    w[L_HIP], w[R_HIP] = (-0.1, 0.2 + torso, 0), (0.1, 0.2 + torso, 0)
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        w[knee] = w[hip] + (0, legs / 2, 0)
        w[ankle] = w[hip] + (0, legs, 0)
    return w


def _full_result(world):
    from core.types import PerceptionResult
    return PerceptionResult(width=640, height=480, timestamp_ms=0, pose=np.zeros((33, 3), np.float32),
                            pose_visibility=np.ones(33, np.float32), pose_world=world)


def test_heads_estimate_recovers_the_figure_height():
    from constructive_anatomy.renderer import _heads_estimate
    for heads in (6.0, 7.0, 8.0):
        assert abs(_heads_estimate(_full_result(None), _world_figure(heads)) - heads) < 1e-3
    assert _heads_estimate(_full_result(None), None) is None  #3D 없으면 판정 안 함


def test_build_picker_needs_sustained_evidence_to_switch():
    from constructive_anatomy.renderer import HEADS_HOLD_FRAMES, _BuildPicker
    picker = _BuildPicker()  #기본 성인
    child, adult = _world_figure(6.0), _world_figure(8.0)
    r = _full_result(None)
    picker.update(r, child)  #한 프레임으로는 안 바뀜
    assert picker.name == "adult"
    for _ in range(HEADS_HOLD_FRAMES + 10):  #EMA가 내려오고 유지 프레임 채우면 바뀜
        picker.update(r, child)
    assert picker.name == "child"
    for _ in range(3):
        picker.update(r, adult)
    assert picker.name == "child"  #다시 몇 프레임으로는 안 돌아감
