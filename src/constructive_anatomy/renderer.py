# 인체 도형화 (Constructive Anatomy) 렌더러 - 3802 김민재
# AI prompt: rivise a code with reflecting 3D distance (for avoiding shortened leg/arm), 
# and based on checking the cam, set the criteria for splitting figure(3) and setting freesets.
# 참고) MediaPipe pose(33 points)/hand(21 points)
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
from __future__ import annotations

import math

import cv2
import numpy as np

from core import config
from core.landmarks import L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_FOOT_INDEX, L_HEEL, L_HIP, L_INDEX, L_KNEE, L_PINKY, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_EAR, R_ELBOW, R_EYE, R_FOOT_INDEX, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_PINKY, R_SHOULDER, R_WRIST
from core.types import PerceptionResult, Target
from render.base import blank_paper, scaled_points, visible

INK = config.INK_COLOR
PAPER = config.PAPER_COLOR
GHOST = tuple(int(p * 0.68 + i * 0.32) for p, i in zip(PAPER, INK))  #가려진 선용 연한 잉크

# 비율표. 단위 H = 머리 높이. 체형(등신)별로 한 벌씩
LENGTH_TOLERANCE = 1.15  #표준 길이의 이 배수를 넘으면 검출 오류로 봄
EAR_SPAN_H = 0.66  #귀 사이 폭 / 머리 높이. 3D 좌표에서 머리 높이를 역산할 때 씀

_JOINTS = {  #관절 구 반지름 (성인 기준, 체형별 thickness를 곱함)
    L_SHOULDER: 0.24, R_SHOULDER: 0.24, L_ELBOW: 0.17, R_ELBOW: 0.17,
    L_WRIST: 0.12, R_WRIST: 0.12, L_HIP: 0.28, R_HIP: 0.28,
    L_KNEE: 0.23, R_KNEE: 0.23, L_ANKLE: 0.14, R_ANKLE: 0.14,
}


def _build(torso, shoulder, arm, leg, thickness):
    # arm/leg = ((길이, 시작 반지름, 끝 반지름), ...). 반지름은 머리 대비라 어릴수록 가늘게
    def thin(segs):
        return tuple((L, r0 * thickness, r1 * thickness) for L, r0, r1 in segs)
    return dict(
        torso=torso, shoulder=shoulder,  #어깨 중심->골반 중심, 어깨 폭 (H 단위)
        chains=(
            ((L_SHOULDER, L_ELBOW, L_WRIST), thin(arm)),
            ((R_SHOULDER, R_ELBOW, R_WRIST), thin(arm)),
            ((L_HIP, L_KNEE, L_ANKLE), thin(leg)),
            ((R_HIP, R_KNEE, R_ANKLE), thin(leg)),
        ),
        joints={k: v * thickness for k, v in _JOINTS.items()},
    )


# 등신이 낮을수록 머리가 크고 몸통/팔다리가 짧다.
_BUILDS = {
    "child": _build(1.8, 1.4, ((1.1, 0.22, 0.16), (0.9, 0.16, 0.11)),
                    ((1.25, 0.34, 0.21), (1.2, 0.22, 0.13)), 0.85),  #6등신
    "teen": _build(2.2, 1.6, ((1.3, 0.22, 0.16), (1.1, 0.16, 0.11)),
                   ((1.65, 0.34, 0.21), (1.55, 0.22, 0.13)), 0.93),  #7등신
    "adult": _build(2.5, 1.8, ((1.5, 0.22, 0.16), (1.25, 0.16, 0.11)),
                    ((2.0, 0.34, 0.21), (1.9, 0.22, 0.13)), 1.0),  #8등신
}
# 등신 경계: 성인 1명 웹캠 평균 7.27
# (p10 7.02, p90 8.11), 성인 경계를 6.9등신으로 설정
HEADS_CHILD_MAX = 6.3
HEADS_TEEN_MAX = 6.9
HEADS_MARGIN = 0.15  #히스테리시스값 설정
HEADS_HOLD_FRAMES = 15  #이만큼 연속으로 다른 구간이어야 바꿈

_BODY_FOLLOWERS = {  #관절을 보정하면 같이 따라 움직이는 점들
    L_WRIST: (L_PINKY, L_INDEX), R_WRIST: (R_PINKY, R_INDEX),
    L_ANKLE: (L_HEEL, L_FOOT_INDEX), R_ANKLE: (R_HEEL, R_FOOT_INDEX),
}

# 손은 단위 P = 손바닥 길이 (손목 -> 중지 MCP)
PALM_WIDTH_P = 0.75  #검지 MCP -> 소지 MCP
PALM_THICK_P = 0.25  #손바닥 두께. 옆에서 보면 이 폭의 판으로 보여야 함
_FINGER = ((0.42, 0.10, 0.085), (0.27, 0.085, 0.075), (0.22, 0.075, 0.065))
_THUMB_CHAIN = ((2, 3, 4), ((0.33, 0.12, 0.10), (0.27, 0.10, 0.085)))  #1->2는 손바닥 안이라 뺌
_FINGER_CHAINS = (  #검지부터 소지 순서. 손바닥 윗변을 이 순서로 제작
    ((5, 6, 7, 8), _FINGER),
    ((9, 10, 11, 12), ((0.46, 0.10, 0.085), (0.30, 0.085, 0.075), (0.24, 0.075, 0.065))),
    ((13, 14, 15, 16), _FINGER),
    ((17, 18, 19, 20), ((0.34, 0.09, 0.075), (0.22, 0.075, 0.065), (0.20, 0.065, 0.055))),
)
_HAND_CHAINS = (_THUMB_CHAIN, *_FINGER_CHAINS)


class _Smoother:  #프레임 간 지수 스무딩
    def __init__(self, alpha_xy=0.55, alpha_z=0.25, max_gap_ms=500):
        self._alpha = np.array([alpha_xy, alpha_xy, alpha_z], dtype=np.float32)
        self._max_gap = max_gap_ms
        self._ts = 0
        self._points = None

    def update(self, timestamp_ms, points):
        previous = self._points
        if (previous is None or previous.shape != points.shape
                or timestamp_ms <= self._ts or timestamp_ms - self._ts > self._max_gap):
            self._points = points.astype(np.float32).copy()
        else:
            self._points = self._alpha * points + (1 - self._alpha) * previous
        self._ts = timestamp_ms
        return self._points


def _dist(arr, a, b):
    return float(np.linalg.norm(arr[a] - arr[b]))


def _mid_dist(arr, a1, a2, b1, b2):  #(a1,a2) 중점과 (b1,b2) 중점 사이 거리
    return float(np.linalg.norm((arr[a1] + arr[a2]) / 2 - (arr[b1] + arr[b2]) / 2))


def _heads_estimate(result, world):
    # 몇 등신인지 3D(pose_world, 미터)로 잰다. 2D는 숙이기만 해도 줄어들어서 못 씀
    # 등신 ~= (몸통 + 다리) / 머리높이 + 1.6 (머리+목 1.3, 발목~바닥 0.3)
    if world is None or not visible(result, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return None
    legs = [_dist(world, h, k) + _dist(world, k, a)
            for h, k, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
            if visible(result, h, k, a)]
    if not legs:
        return None
    head = _dist(world, L_EAR, R_EAR) / EAR_SPAN_H
    if head < 1e-3:
        return None
    torso = _mid_dist(world, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    return (torso + sum(legs) / len(legs)) / head + 1.6


class _BuildPicker:  #등신 측정값으로 체형 고르기. 경계에서 이상해지지 않게 히스테리시스 + 유지 프레임
    def __init__(self, name="adult"):
        self.name = name
        self.heads = None  #스무딩된 측정값 (상태 표시, 디버그용)
        self._pending = None
        self._count = 0

    def update(self, result, world):
        h = _heads_estimate(result, world)
        if h is None:  #못 재면 지금 체형 유지
            return _BUILDS[self.name]
        self.heads = h if self.heads is None else 0.8 * self.heads + 0.2 * h
        lo, hi = HEADS_CHILD_MAX, HEADS_TEEN_MAX
        if self.name == "child":
            lo += HEADS_MARGIN
        elif self.name == "adult":
            hi -= HEADS_MARGIN
        else:
            lo, hi = lo - HEADS_MARGIN, hi + HEADS_MARGIN
        want = "child" if self.heads < lo else "adult" if self.heads > hi else "teen"
        if want == self.name:
            self._pending, self._count = None, 0
        elif want == self._pending:
            self._count += 1
            if self._count >= HEADS_HOLD_FRAMES:
                self.name, self._pending, self._count = want, None, 0
        else:
            self._pending, self._count = want, 1
        return _BUILDS[self.name]


def _head_unit(points, result, world, build):  #머리 높이(px) 추정
    # 2D 어깨 폭, 몸통 길이는 원근으로 줄어들기만 하니까 큰 쪽이 진짜에 가까움
    candidates = [0.0]
    if not visible(result, L_SHOULDER, R_SHOULDER):
        return 0.0
    hips = visible(result, L_HIP, R_HIP)
    candidates.append(_dist(points, L_SHOULDER, R_SHOULDER) / build["shoulder"])
    if hips:
        candidates.append(_mid_dist(points, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) / build["torso"])
    # 3D가 있으면 px/m 스케일로도 추정. 카메라 쪽으로 숙여서 2D가 둘 다 줄어든 자세용
    if world is not None:
        scales = [_dist(points, L_SHOULDER, R_SHOULDER) / max(_dist(world, L_SHOULDER, R_SHOULDER), 1e-3)]
        if hips:
            scales.append(_mid_dist(points, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
                          / max(_mid_dist(world, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP), 1e-3))
        # 머리 높이(m)는 프리셋 비율표로 역산.
        head_m = _dist(world, L_SHOULDER, R_SHOULDER) / build["shoulder"]
        if hips:
            head_m = max(head_m, _mid_dist(world, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) / build["torso"])
        candidates.append(max(scales) * head_m)
    return max(candidates)


def _correct_proportions(points, chains, unit, followers=None):
    # 표준 길이보다 긴 마디는 끝 관절을 당겨 넣음. 짧은 건 원근이니까 그대로 둠.
    # 당긴 만큼 그 아래 관절(과 손발 점)도 같이 이동시켜 다음 마디 모양은 유지
    points = points.copy()
    followers = followers or {}
    for chain, segments in chains:
        delta = np.zeros(2, dtype=np.float32)
        for (a, b), (length, _, _) in zip(zip(chain, chain[1:]), segments):
            points[b] += delta
            v = points[b] - points[a]
            dist = float(np.linalg.norm(v))
            limit = length * unit * LENGTH_TOLERANCE
            if dist > limit > 0:
                pull = v * (limit / dist - 1)
                points[b] += pull
                delta += pull
            for f in followers.get(b, ()):
                points[f] += delta
    return points


def _foreshortening(start, end, canonical):
    # 투영 길이 / 표준 길이로 |cos| 계산. 0 = 화면에 누움(캡이 납작), 1 = 카메라를 향함(캡이 원)
    if canonical <= 0:
        return 0.0
    ratio = min(1.0, float(np.linalg.norm(end - start)) / canonical)
    return math.sqrt(1.0 - ratio * ratio)


class _Scene:
    # 도형을 (깊이, 채우기, 외곽선)으로 모아뒀다가 먼 것부터 그린다 (painter's algorithm).
    # 채우기는 종이색이라 가까운 도형이 뒤 선을 지움. 지워진 선은 두 번째 패스에서 연하게 다시 그림.
    def __init__(self):
        self._forms = []

    def add(self, depth, fill, stroke):  #stroke(canvas, ink, ghost) - ghost면 가려진 반쪽까지 다 그림
        self._forms.append((float(depth), fill, stroke))

    def paint(self, canvas):
        ordered = sorted(self._forms, key=lambda f: -f[0])
        for _, fill, stroke in ordered:
            fill(canvas)
            stroke(canvas, INK, False)
        ghost = np.empty_like(canvas)
        ghost[:] = PAPER
        for _, _, stroke in ordered:
            stroke(ghost, GHOST, True)
        np.minimum(canvas, ghost, out=canvas)  #어두운 쪽이 이김 -> 보이는 선은 진하게, 가려진 선은 연하게


def _pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def _axes(a, b):
    return max(1, int(round(a))), max(1, int(round(b)))


def _half_ellipse(canvas, centre, along, half_along, half_across, thickness, ink=INK, full=False):
    # 첫 축을 along 방향으로 둔 타원. full이 아니면 +along 쪽 절반만 (캡, 단면선, 중심선 전부 이걸로)
    angle = math.degrees(math.atan2(float(along[1]), float(along[0])))
    cv2.ellipse(canvas, _pt(centre), _axes(half_along, half_across), angle,
                0 if full else -90, 360 if full else 90, ink, thickness, cv2.LINE_AA)


def _add_cylinder(scene, depth, start, end, r_start, r_end, squash, near_is_end, thick, thin, *,
                  cap_start=True, cap_end=True, rings=0):
    # 끝이 가늘어지는 원기둥. 가까운 끝 캡은 온전한 타원, 먼 끝 캡과 중간 단면선은 바깥쪽 절반만
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        return
    axis = direction / length
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    toward_far = -axis if near_is_end else axis
    quad = np.array([start + normal * r_start, end + normal * r_end,
                     end - normal * r_end, start - normal * r_start], dtype=np.int32)
    cap_angle = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) + 90

    def fill(c):
        cv2.fillConvexPoly(c, quad, PAPER, cv2.LINE_8)
        for centre, r in ((start, r_start), (end, r_end)):
            cv2.ellipse(c, _pt(centre), _axes(r, r * squash), cap_angle, 0, 360, PAPER, -1, cv2.LINE_8)

    def stroke(c, ink, ghost):
        for sign in (1, -1):
            cv2.line(c, _pt(start + normal * r_start * sign), _pt(end + normal * r_end * sign),
                     ink, thick, cv2.LINE_AA)
        if cap_start:
            _half_ellipse(c, start, toward_far, r_start * squash, r_start, thick, ink,
                          full=ghost or not near_is_end)
        if cap_end:
            _half_ellipse(c, end, toward_far, r_end * squash, r_end, thick, ink,
                          full=ghost or near_is_end)
        for k in range(1, rings + 1):
            t = k / (rings + 1)
            r = r_start + (r_end - r_start) * t
            _half_ellipse(c, start + direction * t, toward_far, r * squash, r, thin, ink,
                          full=ghost)

    scene.add(depth, fill, stroke)


def _add_sphere(scene, depth, centre, r, thickness):
    radius = max(2, int(r))
    # 반지름만큼 뒤로 밀어서 원기둥이 구 위에 칠해지게. 튀어나온 부분만 진한 선으로 남는다
    scene.add(depth + r,
              lambda c: cv2.circle(c, _pt(centre), radius, PAPER, -1, cv2.LINE_8),
              lambda c, ink, ghost: cv2.circle(c, _pt(centre), radius, ink, thickness, cv2.LINE_AA))


def _add_ellipse(scene, depth, centre, axes, angle, thick, extra=None):
    def stroke(c, ink, ghost):
        cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, ink, thick, cv2.LINE_AA)
        if extra:
            extra(c, ink, ghost)

    scene.add(depth,
              lambda c: cv2.ellipse(c, _pt(centre), axes, angle, 0, 360, PAPER, -1, cv2.LINE_8),
              stroke)


def _add_polygon(scene, depth, pts, thick, extra=None):
    poly = pts.astype(np.int32)

    def stroke(c, ink, ghost):
        cv2.polylines(c, [poly], True, ink, thick, cv2.LINE_AA)
        if extra:
            extra(c, ink, ghost)

    scene.add(depth, lambda c: cv2.fillPoly(c, [poly], PAPER, cv2.LINE_8), stroke)


def _slab_outline(poly, offset, size):
    # 다각형을 offset만큼 밀어낸 판(앞면 poly+offset/2, 뒷면 poly-offset/2)의 바깥 윤곽
    # 앞면·뒷면·옆면을 마스크에 다 채우고 외곽선만 뽑는다. 정면이면 poly 그대로 나옴
    near, far = poly + offset / 2, poly - offset / 2
    mask = np.zeros((size[1], size[0]), np.uint8)
    for quad in zip(near, np.roll(near, -1, axis=0), np.roll(far, -1, axis=0), far):
        cv2.fillPoly(mask, [np.array(quad, np.int32)], 255, cv2.LINE_8)
    cv2.fillPoly(mask, [near.astype(np.int32)], 255, cv2.LINE_8)
    cv2.fillPoly(mask, [far.astype(np.int32)], 255, cv2.LINE_8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else near.astype(np.int32)


def _add_slab(scene, depth, poly, offset, thick, size, extra=None):
    outline = _slab_outline(poly, offset, size)
    near = (poly + offset / 2).astype(np.int32)
    far = (poly - offset / 2).astype(np.int32)

    def stroke(c, ink, ghost):
        cv2.polylines(c, [outline], True, ink, thick, cv2.LINE_AA)
        cv2.polylines(c, [near], True, ink, thick, cv2.LINE_AA)
        if ghost:  #뒷면은 가려진 선으로만
            cv2.polylines(c, [far], True, ink, thick, cv2.LINE_AA)
        if extra:
            extra(c, ink, ghost)

    scene.add(depth, lambda c: cv2.fillPoly(c, [outline], PAPER, cv2.LINE_8), stroke)


def _add_ellipsoid(scene, depth, centre, up, half_w, half_h, depth_r, facing, thick, thin,
                   ring_at, ring_squash, ring_toward):
    # 타원체 = 윤곽 + 중심선(카메라 향한 쪽으로 쏠림) + 가로 단면선 하나
    # facing = right 방향 sin(yaw). 돌아선 만큼 폭이 depth_r 쪽으로 바뀌고 중심선이 옆으로 간다
    right = np.array([-up[1], up[0]], dtype=np.float32)
    facing = max(-1.0, min(1.0, facing))
    proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
    angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
    side = 1.0 if facing >= 0 else -1.0

    def lines(c, ink, ghost):
        _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin, ink,
                      full=ghost)
        _half_ellipse(c, centre + up * (half_h * ring_at), up * ring_toward,
                      max(proj_w * ring_squash, proj_w * 0.12), proj_w, thin, ink, full=ghost)

    _add_ellipse(scene, depth, centre, _axes(proj_w, half_h), angle, thick, extra=lines)


def _base_edge(points, chain, r, order):  #마디 첫 원기둥의 밑동 양 끝점. order 방향으로 정렬
    base = points[chain[0]]
    axis = points[chain[1]] - base
    axis = axis / (np.linalg.norm(axis) or 1.0)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32) * r
    return sorted((base + normal, base - normal), key=lambda p: float(np.dot(p, order)))


def _palm_outline(points, P):
    # 손바닥 다각형: 손목 모서리 -> (엄지 밑동) -> 손가락 밑동들 -> 손목 모서리
    # 윗변을 손가락 밑동 선분으로 만들어서 손바닥 꼭짓점 = 손가락 밑동 꼭짓점이 되게 함
    across = points[17] - points[5]
    across = across / (np.linalg.norm(across) or 1.0)
    top = []
    for chain, segments in _FINGER_CHAINS:
        top += _base_edge(points, chain, segments[0][1] * P, across)
    wrist_index = points[0] - across * P * 0.28
    wrist_pinky = points[0] + across * P * 0.28

    edge = top[0] - wrist_index  #검지 쪽 가장자리
    def side(p):  #edge 기준 어느 쪽인지 (외적 부호)
        d = p - wrist_index
        return float(edge[0] * d[1] - edge[1] * d[0])
    centre = points[[0, 5, 9, 13, 17]].mean(axis=0)
    # 엄지가 손바닥 안으로 접히거나 손가락 쪽으로 누우면 밑동 끝점이 가장자리 안쪽에 들어와 다각형이 꼬임
    # -> 밑동 양 끝이 다 바깥에 있을 때만 윤곽에 넣고, 아니면 위에 얹기만
    thumb = _base_edge(points, _THUMB_CHAIN[0], _THUMB_CHAIN[1][0][1] * P, edge)
    if any(side(p) * side(centre) >= 0 for p in thumb):
        thumb = []
    return np.array([wrist_index, *thumb, *top, wrist_pinky])


def _palm_normal(points, z):  #손바닥 평면 법선. 카메라 쪽(z 작은 쪽)을 향하게 뒤집음
    p = np.column_stack([points, z])
    n = np.cross(p[5] - p[0], p[17] - p[0])
    n = n / (np.linalg.norm(n) or 1.0)
    return -n if n[2] > 0 else n


def _add_chain(scene, points, depth, chain, segments, unit, thick, thin, *,
               rings, tips_only, ok=lambda a, b: True):
    pairs = list(zip(chain, chain[1:]))
    for i, ((a, b), (length, r0, r1)) in enumerate(zip(pairs, segments)):
        if not ok(a, b):
            continue
        last = i == len(pairs) - 1
        _add_cylinder(
            scene, (depth[a] + depth[b]) / 2, points[a], points[b], r0 * unit, r1 * unit,
            squash=_foreshortening(points[a], points[b], length * unit),
            near_is_end=bool(depth[b] < depth[a]), thick=thick, thin=thin,
            # tips_only(손)여도 첫 마디 밑동 캡은 그림 - 손바닥 윗변과 겹치는 그 선
            cap_start=i == 0 or not tips_only, cap_end=last or not tips_only, rings=rings,
        )


class ConstructiveRenderer:
    name = "Constructive Anatomy"
    #프레임 간 스무딩과 체형 선택 상태를 들고 있어 호출 순서가 결과를 바꾼다.
    #렌더 스테이지는 이 값을 보고 워커를 하나로 묶는다 (render/base.py 참고)
    stateless = False

    def __init__(self):
        self._pose_filter = _Smoother()
        self._world_filter = _Smoother(alpha_xy=0.3, alpha_z=0.3)  #3D는 전체적으로 더 튀어서 더 세게
        self._hand_filter = _Smoother()
        self._picker = _BuildPicker()

    @property
    def build(self):  #지금 쓰는 체형 이름 (상태 표시, 디버그용)
        return self._picker.name

    def render(self, result: PerceptionResult, target: Target, size: tuple[int, int]) -> np.ndarray:
        canvas = blank_paper(size)
        scene = _Scene()
        if target is Target.HAND:
            self._build_hand(scene, result, size)
        else:
            self._build_body(scene, result, size)
        scene.paint(canvas)
        return canvas

    def _build_body(self, scene, result, size):
        if result.pose is None:
            return
        pose = self._pose_filter.update(result.timestamp_ms, result.pose)
        world = (self._world_filter.update(result.timestamp_ms, result.pose_world)
                 if result.pose_world is not None else None)
        points = scaled_points(pose, size)
        depth = pose[:, 2] * size[0]  #z는 x와 같은 스케일이라 width 곱하면 px
        build = self._picker.update(result, world)
        H = _head_unit(points, result, world, build)
        if H < 3:
            return
        points = _correct_proportions(points, build["chains"], H, _BODY_FOLLOWERS)
        thick = max(2, int(round(H * 0.035)))
        thin = max(1, thick // 2)
        seen = lambda *idx: visible(result, *idx)  # noqa: E731

        shoulders = (points[L_SHOULDER] + points[R_SHOULDER]) / 2
        if seen(L_HIP, R_HIP):
            hips = (points[L_HIP] + points[R_HIP]) / 2
            self._torso(scene, points, depth, shoulders, hips, H, thick, thin)
            up = shoulders - hips
        else:
            up = np.array([0.0, -1.0], dtype=np.float32)
        up = up / (np.linalg.norm(up) or 1.0)

        self._head(scene, points, depth, result, shoulders, up, H, thick, thin)

        for chain, segments in build["chains"]:
            _add_chain(scene, points, depth, chain, segments, H, thick, thin,
                       rings=1, tips_only=False, ok=seen)

        # 손은 미튼 타원, 발은 쐐기. 손 모드랑 겹치지 않을 만큼만
        for wrist, pinky, index in ((L_WRIST, L_PINKY, L_INDEX), (R_WRIST, R_PINKY, R_INDEX)):
            if seen(wrist, pinky, index):
                tip = (points[pinky] + points[index]) / 2
                along = tip - points[wrist]
                length = float(np.linalg.norm(along))
                if length > H * 0.25:  #카메라를 향한 손은 손목 캡이 이미 원이라 미튼까지 그리면 원 세 겹
                    _add_ellipse(scene, depth[wrist], (points[wrist] + tip) / 2,
                                 _axes(length / 2, H * 0.2),
                                 math.degrees(math.atan2(float(along[1]), float(along[0]))), thick)
        for ankle, heel, toe in ((L_ANKLE, L_HEEL, L_FOOT_INDEX), (R_ANKLE, R_HEEL, R_FOOT_INDEX)):
            if seen(ankle, heel, toe):
                _add_polygon(scene, depth[ankle],
                             np.array([points[ankle], points[heel], points[toe]]), thick)

        for idx, r in build["joints"].items():
            if seen(idx):
                _add_sphere(scene, depth[idx], points[idx], r * H, thick)

    def _torso(self, scene, points, depth, shoulders, hips, H, thick, thin):
        spine = hips - shoulders
        length = float(np.linalg.norm(spine)) or 1.0
        up = -spine / length
        right = np.array([-up[1], up[0]], dtype=np.float32)

        def facing(a, b):  #앞면은 더 가까운 관절 쪽으로 돈다. sin(yaw)를 right 축 부호로 맞춤
            ab = points[b] - points[a]
            d = math.hypot(float(np.linalg.norm(ab)), float(depth[b] - depth[a])) or 1.0
            toward_b = float(depth[a] - depth[b]) / d
            return toward_b * (1.0 if float(np.dot(ab, right)) >= 0 else -1.0)

        # 앞뒤로 숙이면 가로 단면이 벌어져 보임
        z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
        z_hips = float(depth[L_HIP] + depth[R_HIP]) / 2
        dz = z_hips - z_shoulders
        pitch = abs(dz) / math.hypot(length, dz)
        ring_toward = -1.0 if dz > 0 else 1.0  #보이는 반쪽은 가까운 끝 반대편

        # 흉곽과 골반을 따로. 허리에서 비틀리는 게 보여야 하니까
        _add_ellipsoid(scene, z_shoulders, shoulders + spine * 0.34, up, H * 0.72, H * 0.85,
                       H * 0.55, facing(L_SHOULDER, R_SHOULDER), thick, thin,
                       ring_at=-0.6, ring_squash=pitch, ring_toward=ring_toward)
        _add_ellipsoid(scene, z_hips, shoulders + spine * 0.86, up, H * 0.60, H * 0.45,
                       H * 0.50, facing(L_HIP, R_HIP), thick, thin,
                       ring_at=0.0, ring_squash=pitch, ring_toward=ring_toward)

    def _head(self, scene, points, depth, result, shoulders, up, H, thick, thin):
        if visible(result, L_EAR, R_EAR):
            centre = (points[L_EAR] + points[R_EAR]) / 2
            z = float(depth[L_EAR] + depth[R_EAR]) / 2
        elif visible(result, NOSE):
            centre = points[NOSE] + up * H * 0.15
            z = float(depth[NOSE])
        else:
            return
        right = np.array([-up[1], up[0]], dtype=np.float32)
        half_w, half_h, depth_r = H * 0.37, H * 0.5, H * 0.45

        # 얼굴 중심선은 코 위치로 바로 정함 (yaw 추정 필요 없음)
        offset = float(np.dot(points[NOSE] - centre, right)) if visible(result, NOSE) else 0.0
        facing = max(-1.0, min(1.0, offset / depth_r))
        proj_w = math.sqrt(half_w ** 2 * (1 - facing ** 2) + depth_r ** 2 * facing ** 2)
        angle = math.degrees(math.atan2(float(right[1]), float(right[0])))
        side = 1.0 if offset >= 0 else -1.0
        eye_level = (float(np.dot((points[L_EYE] + points[R_EYE]) / 2 - centre, up))
                     if visible(result, L_EYE, R_EYE) else None)

        def lines(c, ink, ghost):
            _half_ellipse(c, centre, right * side, abs(facing) * depth_r, half_h, thin, ink,
                          full=ghost)
            if eye_level is not None:
                _half_ellipse(c, centre + up * eye_level, -up, proj_w * 0.18, proj_w * 0.95,
                              thin, ink, full=ghost)

        _add_ellipse(scene, z, centre, _axes(proj_w, half_h), angle, thick, extra=lines)

        # 목. 머리가 어깨에서 1.5H 넘게 떨어져 있으면 검출이 이상한 거라 안 그림
        if np.linalg.norm(centre - shoulders) <= H * 1.5:
            z_shoulders = float(depth[L_SHOULDER] + depth[R_SHOULDER]) / 2
            _add_cylinder(scene, max(z, z_shoulders) + 1.0, centre - up * half_h * 0.9,  #머리/흉곽 뒤에
                          shoulders, H * 0.2, H * 0.22, squash=0.0, near_is_end=False,
                          thick=thick, thin=thin, cap_start=False, cap_end=False)

    def _build_hand(self, scene, result, size):
        if result.hand is None:
            return
        hand = self._hand_filter.update(result.timestamp_ms, result.hand)
        points = scaled_points(hand, size, fit=0.12)
        depth = hand[:, 2]
        # z를 px 단위로 맞춰서 손바닥 법선을 구함 (z는 x와 같은 스케일)
        px_per_unit = (np.linalg.norm(points[9] - points[0])
                       / (np.linalg.norm(hand[9, :2] - hand[0, :2]) or 1.0))
        normal = _palm_normal(points, depth * px_per_unit)
        # 손바닥 길이 (fit 확대 후 기준). 옆에서 본 손은 폭으로 대신
        P = max(float(np.linalg.norm(points[0] - points[9])),
                float(np.linalg.norm(points[5] - points[17])) / PALM_WIDTH_P)
        if P < 3:
            return
        points = _correct_proportions(points, _HAND_CHAINS, P)
        # 손목 모서리(0.28P)랑 손바닥 두께가 랜드마크 상자 밖으로 나가니, 그만큼 여유가 없으면 화면 중심 기준으로 줄임
        need = points.max(axis=0) - points.min(axis=0) + 0.8 * P
        shrink = min(1.0, float(np.min(np.array(size) * 0.96 / need)))
        if shrink < 1.0:
            centre = np.array(size, dtype=np.float32) / 2
            points = (points - centre) * shrink + centre
            P *= shrink
        thick = max(2, int(round(P * 0.022)))
        thin = max(1, thick // 2)

        palm = _palm_outline(points, P)
        offset = normal[:2] * PALM_THICK_P * P  #두께를 화면에 투영한 벡터. 정면이면 0에 가까움
        mid_a, mid_b = (palm[0] + points[5]) / 2, (palm[-1] + points[17]) / 2  #손목 모서리~검지/소지 MCP 중간
        mid_a, mid_b = mid_a + offset / 2, mid_b + offset / 2  #앞면 위에
        _add_slab(scene, float(np.mean(depth[[0, 2, 5, 9, 13, 17]])), palm, offset, thick, size,
                  extra=lambda c, ink, ghost: cv2.line(c, _pt(mid_a), _pt(mid_b), ink, thin,
                                                       cv2.LINE_AA))

        for chain, segments in _HAND_CHAINS:
            _add_chain(scene, points, depth, chain, segments, P, thick, thin,
                       rings=0, tips_only=True)
            for idx, (_, r0, _) in zip(chain[1:-1], segments[1:]):  #마디마다 관절 구
                _add_sphere(scene, depth[idx], points[idx], r0 * P * 1.05, thin)
