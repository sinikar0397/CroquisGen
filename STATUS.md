# 개발 진척도

2026-11-24 기준. 계획서(`3402김주환 - 모션 인식 프로젝트 계획서.docx`)의 기능 단위로
정리한다. 뒤쪽에 11/24 스레드 파이프라인과 11/18 폴더 구조 개편의 경로 대응표가 있다.

실행은 `python main.py`. 설치는 `pip install -r requirements.txt` 후
`python scripts/download_models.py`.

---

## 1. 기능별 상태

### 완성

| 기능 | 구현 위치 |
|---|---|
| 웹캠 입력 (셀피 미러링, macOS 권한 안내) | `app/camera.py` |
| 전신 골격 추출 (33관절 + 신뢰도 + 미터 3D + 인물 마스크) | `perception/mediapipe_pose.py` |
| 손 골격 추출 (21관절, 3프레임마다 실행 후 재사용) | 같음 |
| 손바닥 → 캡처 (PNG 저장 + 썸네일 알림) | `app/gestures.py`, `app/state.py` |
| 검지 1초 홀드 → 원형 툴바 (홀드 중 게이지) | `app/gestures.py`, `app/ui/radial_menu.py` |
| 툴바에서 스타일 선택 (상/하 섹터, 데드존, 취소) | 같음 |
| 제스처 오발화 방지 (라벨 안정화 + 800 ms 쿨다운) | `app/gestures.py` |
| 전신 사진에서 손 ROI 지정 — 계획서 과제 2 | `app/hand_roi.py` |
| 몸 윤곽선 추출 (세그멘테이션 마스크 경계) | `render/outline.py` |
| 상태 표시·카메라 인셋 | `app/ui/overlay.py` |
| 키보드 단축키 (모든 제스처에 1:1 대응) | `app/loop.py` |
| 스레드 파이프라인 (캡처·인식·렌더 분리) | `app/loop.py`, `app/pipeline.py` |
| 동세 크로키 S4를 실시간 루프에 연결 | `gesture_croquis/realtime.py`, `gesture_croquis/renderer.py` |

### 진행 중

**인체 도형화 (김민재)** — `constructive_anatomy/renderer.py`, 542줄

프레임 간 지수 스무딩(z는 더 약하게) → 머리 높이 H 기준 비율표로 도형 크기 결정 →
표준 길이를 넘는 사지는 검출 오류로 보고 보정 → depth 순서대로 쌓아 합성. 가려지는 선은
지우지 않고 연한 잉크로 남기고, 원기둥 끝 캡은 단축률에 따라 납작해지며 가까운 쪽은
전체 링, 먼 쪽은 반 링으로 그린다.

11/18 이후 추가분: **체형 프리셋 3종**(`child` 6등신 / `teen` 7등신 / `adult` 8등신).
비율표·관절 반지름·두께를 한 벌씩 갖고, 머리 높이 대비 등신 수로 자동 선택한다
(경계 6.3 / 6.9등신, 성인 웹캠 평균 7.27 측정값 주석). 사지 길이 보정도 3D 좌표
기반으로 바뀌었다. 캡 채우기 방향도 수정됐다.

수정 예정: 인체 세분화, 도형 구체화. 세부 범위는 담당자 확인 필요.

**동세 크로키 (박준현)** — S4까지 구현됐고 S5가 남았다.

| 단계 | 상태 | 위치 |
|---|---|---|
| S1~S3 분석 | 완료 | `gesture_croquis/` |
| S4 추상화 (선 생성) | 완료 (932줄 + 테스트 122줄) | `gesture_croquis/s4_gesture.py` |
| S5 스트로크 스타일화 | 미착수 | — |
| 실시간 경로 | S4 2D 모드로 연결, 기본 엔진 | `gesture_croquis/realtime.py` |

분석 단계는 사진 한 장에서 2D 키포인트를 뽑고, 뼈 길이 사전확률·관절 가동범위·근육 이완
자세·중력 토크·균형·바닥 평면을 한 목적함수에 넣어 미터 단위 3D 골격을 맞춘다. depth
모호성은 투시 기하의 두 해를 모두 탐색해 타당성 에너지로 고르고, 팔이 몸통 앞/뒤인지는
이미지 윤곽의 엣지 연속성으로 판정한다. 척추 곡선과 관절별 토크·근육 스트레스까지 낸다.

S4는 관절을 움직이지 않는다. 투영된 골격에서 선만 뽑는다 — 후보 체인을 점수 매겨 고른
line of action(과장 포함), 사지별 리듬선(근육 활성·이완으로 휘는 방향 결정), 토르소
스쿼시, 선 굵기 모델, novel view. 뷰당 30~60 ms.

**S2에 JAX 정확 야코비안이 들어왔다.** `s2_jax.py`(796줄)가 같은 잔차를 `jax.numpy`로
다시 쓰고 전진 모드 야코비안을 `least_squares`에 넘긴다. 자체 Levenberg–Marquardt
솔버도 포함. S2 전체 15.6초 → 1.6초. 다만 **`py312` 환경에 jax가 없고
`requirements.txt`에도 없다** — 지금 돌리면 유한차분 폴백이고 `test_s2_jax.py`의
11개가 skip된다. 쓸 거면 requirements에 넣고, 안 쓸 거면 skip이 정상이라는 걸 적어둬야
한다.

### 해결이 필요한 것

**프레임 예산.** 스레드 파이프라인으로 두 스타일 모두 30 fps를 넘겼지만, S4를 기본
엔진으로 올리면서 동세 크로키는 13 fps가 됐다. 수치는 2절에 있다.

**S4가 S3 없이 돈다.** 실시간 경로는 2D 모드라 선 굵기 모델의 effort·support 항이
0이다. 굵기가 role·taper·2D compression에서만 나온다.

**LOA 승자 플리핑.** 후보 체인 점수가 비슷하면 프레임마다 승자가 바뀌어 그림이 튈 수
있다. 합성 포즈에서 상위 둘이 0.68 동률이었다. 웹캠으로 확인이 필요하고, 실제로 튀면
히스테리시스가 답인데 S4 설계 영역이다.

**핑거 스냅 미검증.** 계획서의 제스처 3개 중 전신↔손 전환만 사전학습 카테고리가 없어
직접 구현했다. 스냅은 수십 ms 안에 끝나 30 fps에서 1~2프레임이고, 모션 블러가 판단에
필요한 랜드마크를 망가뜨린다. 2단계 휴리스틱이 들어가 있지만 실사용 신뢰도가 확인되지
않았다. 대안은 이 전환을 툴바의 세 번째 섹터로 옮기는 것 — 새 모델이 필요 없고 툴바는
이미 안정적이다.

**제스처 안정화 필터가 실질적으로 무효다.** 손 인식 결과가 3프레임 재사용되는데 이벤트
해석은 매 결과마다 돌아, 같은 결과가 연속으로 세어져 2회 임계값이 즉시 충족된다. 오분류
한 번이 3프레임으로 증폭된다. `MediaPipePerception.hand_ran_this_frame()`이 이 문제를
위해 있지만 호출되지 않는다. 스레드 전환으로도 해결되지 않았다 — pose와 hand가 한
스테이지에 묶여 있어 재사용 구조가 그대로다.

**툴바 섹터 하이라이트가 없다.** `AppState.menu_highlight`를 설정하는 코드가 없어 손가락이
어느 섹터를 향하는지 표시되지 않고, 선택되면 바로 적용된다.

**손 ROI가 연결되지 않았다.** `app/hand_roi.py`는 구현과 테스트가 끝났지만 호출부가 없다.
손 모드는 전신 프레임을 크롭하지 않고 손 랜드마커 21점을 캔버스에 맞춰 확대해 그린다.

**문서와 코드 불일치.** README는 손바닥 인식 후 5초 카운트다운을 적어두었고, 코드는 안정화
후 즉시 저장한다. 어느 쪽으로 맞출지 미정.

### 계획서에서 수단이 바뀐 항목

기능은 유지하고 구현 수단만 교체했다.

| 계획서 | 현재 | 이유 |
|---|---|---|
| YOLO 손 검출 | 제거 | MediaPipe 손 파이프라인에 palm detector가 이미 있고, 일반 YOLO 가중치에 hand 클래스가 없다 |
| OpenPose 골격 추출 | MediaPipe Pose Landmarker | OpenPose는 CUDA 전제라 Mac에 가속 경로가 없고 비상업 학술 라이선스다 |
| gradient/Laplacian 윤곽 | 세그멘테이션 마스크 윤곽 | 엣지 필터는 배경 가구·벽 무늬·옷 주름을 전부 선으로 넣는다 |
| 커스텀 제스처 분류 모델 | 사전학습 카테고리 | `GestureRecognizer`가 `Open_Palm`·`Pointing_Up`을 학습 없이 준다 |
| `mp.solutions` 기반 코드 | Tasks API | mediapipe 1.x에서 `mp.solutions`가 제거되어 import 자체가 안 된다 |

### 미착수

| 기능 | 비고 |
|---|---|
| 동세 크로키 S5 (스트로크 스타일화) | S4가 내는 스트로크 JSON을 그대로 받으면 된다. skia/pycairo, 레이어별 SVG |
| 학습 데이터 준비 | 인터넷 크로키에서 골격 추출은 실사 학습 모델이라 line art에서 잘 안 된다. 렌더링된 3D 포즈가 신뢰할 수 있는 출처. 오프라인 작업이라 프레임 예산과 무관 |
| 웹 전환 | 브라우저 MediaPipe가 동일 랜드마크를 주므로 `core/`와 `render/`는 그대로 따라간다. `app/`은 버려지는 쪽 |
| S2/S3 테스트 이전 | 독립 프로젝트 시절의 검증 7개가 아직 `tests/`에 없다 |

### 테스트

카메라와 모델 파일 없이 도는 60개가 통과한다 (`python -m pytest tests -q`).

- `tests/test_pipeline.py` — 렌더러 출력 규약(크기·dtype·빈 입력 무예외·저신뢰 사지 생략),
  제스처 쿨다운·홀드·섹터 선택·취소, 손 ROI 기하, 상태 전이와 캡처
- `tests/test_constructive.py` — 인체 도형화 내부 기하(길이 보정, 단축률, 프레임 간 스무딩,
  손바닥 윤곽, 체형 프리셋)
- `tests/test_threading.py` — 최신값 슬롯과 워커 수명, 단계 하나가 죽으면 전체가 멈추는지,
  스텁 카메라·인식으로 스레드 루프 전체를 돌린 통합, `--no-threads` 경로
- `tests/test_gesture_croquis.py` — MediaPipe → canonical 어댑터(매핑, 파생 중심,
  저신뢰 제거), S4 엔진의 렌더러 계약 준수, 실패 시 placeholder로 degrade

`src/gesture_croquis/`에는 패키지 자체 테스트가 따로 있다
(`cd src && python -m pytest gesture_croquis -q`): 10개 통과, 11개 skip(jax 미설치).

미검증 영역은 제스처 안정화 필터의 프레임 카운트, 툴바 하이라이트, 실제 웹캠에서의
MediaPipe 호출과 S4 선 안정성이다.

---

## 2. 11/24 스레드 파이프라인과 S4 이식

### 왜 스레드인가

파이썬 스레드는 네이티브 호출이 GIL을 놓을 때만 의미가 있다. 이 프로젝트는 비싼 두
단계가 모두 그렇다. MediaPipe 호출 중 별도 스레드에서 파이썬 카운터를 돌리면 유휴
대비 91~98%가 유지되고, OpenCV 드로잉도 같다. S4도 98%다(Bézier 피팅의 numpy 덕).

### 구조

```
캡처 스레드      read → flip            → [최신 프레임 1칸]
인식 스레드      pose → hand            → [최신 PerceptionResult]
렌더 스레드      renderer.render()      → [최신 크로키]
메인 스레드      제스처 · 상태 · 오버레이 · imshow
```

단계 사이는 **큐가 아니라 락 하나 달린 1칸짜리 최신값 상자**(`app/pipeline.py`의
`Slot`)다. 큐를 쓰면 느린 단계 앞에 프레임이 쌓여 지연이 계속 늘어난다. 늦은 프레임은
버린다. 값이 같아도 새 결과인지 구분해야 하므로 시퀀스 번호를 같이 둔다 — 가만히 있는
포즈는 같은 크로키를 낸다.

`Pipeline`은 정지 플래그와 에러 수집이다. **단계 하나가 죽으면 전체를 멈춘다.** 안
그러면 메인이 아무도 채우지 않는 슬롯에서 영원히 기다린다. 예외는 모아뒀다가 메인
스레드에서 다시 던져 조용한 멈춤 대신 트레이스백이 나오게 한다.

화면 출력이 메인에 남는 것은 선택이 아니다. macOS에서 `cv2.imshow`/`waitKey`는 Cocoa를
건드리므로 메인 스레드여야 한다.

`--no-threads`로 기존 직선 루프를 그대로 쓸 수 있다. 스레드 수치를 비교할 기준선이고,
렌더러가 오동작할 때 스레드가 오류 읽기를 방해하지 않게 하는 용도다.

### 측정

1280x720, 합성 랜드마크, `tools/smoke_threading.py --force-pose`.

| 스타일 | 단일 스레드 | 스레드 |
|---|---|---|
| 동세 크로키 (S4, 기본) | 8.5 fps | **13.4 fps** |
| 동세 크로키 (placeholder) | 19.3 fps | **41.5 fps** |
| 인체 도형화 | 14.4 fps | **38.0 fps** |

S4 기본에서도 카메라 49/s, 인식 40.8/s, 디스플레이 53/s로 돈다. 뒤처지는 건 크로키
그림뿐이고, 제스처 UI·홀드 링·툴바는 메인이 최신 인식 결과로 직접 그려 반응성이
유지된다. 포즈를 잡고 보는 용도는 문제없고 움직이면 끌린다.

단계별 비용(사람 없는 프레임이라 인식은 검출기 비용만):

| | 비용 |
|---|---|
| pose (full) | 15.4~15.7 ms |
| hand + gesture | 21.8~22.0 ms |
| 렌더 동세 크로키 placeholder | 7.3~8.1 ms |
| 렌더 동세 크로키 S4 | 56 ms + 래스터화 9 ms |
| 렌더 인체 도형화 | 24.4~31.0 ms |

**11/18 STATUS의 렌더 11.8 ms는 틀렸다.** 앱이 실제로 쓰는 1280x720에서 인체 도형화는
24~31 ms다. 체형 프리셋이 들어간 뒤 재측정이 안 됐었다.

### 렌더 워커를 늘리는 건 이 기계에서 값을 못 한다

`--render-workers N`과 그 배관(`CroquisRenderer.stateless`, `Slot.put_newer`)은 들어가
있지만 **기본값은 1**이다.

| 워커 | S4 단독 | 인식 스레드와 함께 |
|---|---|---|
| 1 | 18.4/s | 17.7/s |
| 2 | 25.9/s (1.41×) | 22.3/s (1.21×) |
| 3 | 31.5/s (1.72×) | 20.1/s (1.09×) |

격리하면 붙는데 인식이 옆에서 돌면 거의 사라지고, 3워커는 2워커보다 느리다(성능 코어
4개). `--lite`로도, BLAS 스레드를 1로 묶어도 같았다. 성능 코어가 더 많은 기계에서는
다시 볼 값어치가 있다. 30 fps로 가려면 S4 자체를 줄여야 하고, 56 ms 중 **39 ms가
line-of-action 탐색**(후보 체인마다 cubic Bézier 피팅)이다.

워커를 여러 개 돌릴 때를 위해 두 가지가 들어갔다. `CroquisRenderer.stateless` 선언은
프레임 간 상태를 들고 있는지를 알린다 — 인체 도형화는 `False`라 락으로 직렬화되고,
툴바로 스타일이 바뀌므로 시작 시점에 정할 수 없다. `Slot.put_newer(value, tag)`는
뒤처진 워커가 더 새 크로키를 덮어쓰지 않게 버린다.

### S4 실시간 경로

`gesture_croquis/realtime.py`가 `PerceptionResult` → `PoseFrame` 변환을 맡는다.
MediaPipe 33점을 canonical 28관절로 옮기고 neck·pelvis·head를 파생시킨다. 손 관절은
중지 MCP인데 pose 모델에 없어 index·pinky 중점으로 근사했고, small toe는 소스가 없어
비워둔다. **`MIN_VISIBILITY` 미만은 0으로 떨어뜨린다** — S4는 신뢰도 0을 없는 관절로
보고 빼므로, 모델이 지어낸 관절을 그리지 않는다는 규칙이 지켜진다.

**2D 모드를 쓴다.** S4의 `BodyView`는 `joints3d`가 있으면 3D, 없으면 2D 키포인트만으로
돈다. 실시간 루프는 photo view만 쓰므로 2D에서 잃는 게 거의 없다. MediaPipe가 미터 단위
3D(`pose_world`)를 주니 S2를 건너뛰고 3D 모드로 갈 수 있을 것 같지만 안 된다 — 그
좌표는 골반 중심인데 `BodyView.proj`는 z를 카메라까지의 거리로 보고 나눈다. 쓰려면 2D
랜드마크에 재투영되는 평행이동을 풀어야 하고 그건 S2의 일이다. 그래서 novel view와 깊이
기반 선 굵기(front/behind 강조)는 꺼져 있다.

엔진은 `core/config.py`의 `GESTURE_CROQUIS_ENGINE`으로 고른다. **S4가 기본**이고
`--placeholder`로 가벼운 기하 근사로 돌아간다. placeholder는 세 군데서 계속 쓰인다 —
`--placeholder`로 고를 때, 손 모드(S4는 전 단계가 몸에 대한 것이라 손 경로가 없다),
S4가 처리 못 한 프레임의 폴백(사람이 없거나 몸통 신뢰도가 없거나 예외가 났을 때).
렌더러가 예외를 던지면 렌더 스레드가 죽으므로 `last_s4_error`에 남기고 넘어간다.

### 그 외 변경

| 항목 | 내용 |
|---|---|
| `FrameClock` | 지수평활 `1/dt` → 창 기반 계수. 간격이 흔들리면 평균의 역수와 역수의 평균이 달라 fps가 과대평가된다(스레드 43을 48.6으로, 단일 스레드도 19.0을 20.1로 읽고 있었다). 지금은 실측과 ±1 fps |
| 상태바 fps | 디스플레이 패스가 아니라 **새 크로키마다** 센다. `TOAST_FRAMES`도 크로키 프레임 기준을 유지 |
| 오버레이 | 스레드 경로는 같은 크로키를 여러 번 그리므로 복사본에 오버레이를 올린다. 원본에 직접 그리면 상태바와 토스트가 누적된다 |
| 측정 도구 | `tools/bench_threading.py`(GIL 해제·단계별·중첩 비용), `tools/smoke_threading.py`(실제 MediaPipe로 두 경로 끝까지). `tools/`는 저장소에서 추적하지 않는다 |

---

## 3. 11/18 폴더 구조 개편

담당자 한 명이 폴더 하나를 보면 되도록 나눴다. 기능 변경은 없고 위치와 import 경로만
바뀌었다.

```
main.py                        진입점
src/
  core/                        단계 간 계약 (공용)
    types.py                   PerceptionResult, AppState, Style, Target
    landmarks.py               랜드마크 인덱스와 골격 엣지
    config.py                  튜너블
  app/                         실시간 루프·제스처·상태·UI       김주환
  perception/                  랜드마크 공급, 백엔드 교체 지점    김주환
  render/                      드로잉 계약·공용 도구·레지스트리    공용
  gesture_croquis/             동세 크로키                     박준현
  constructive_anatomy/        인체 도형화                     김민재
scripts/  tools/  tests/
```

### 경로 대응표

| 이전 | 현재 |
|---|---|
| `src/croquisgen/app.py` | `src/app/loop.py` |
| `src/croquisgen/capture.py` | `src/app/camera.py` |
| `src/croquisgen/gestures.py` | `src/app/gestures.py` |
| `src/croquisgen/state.py` | `src/app/state.py` |
| `src/croquisgen/hand_roi.py` | `src/app/hand_roi.py` |
| `src/croquisgen/ui/` | `src/app/ui/` |
| `src/croquisgen/types.py` | `src/core/types.py` + `src/core/landmarks.py` |
| `src/croquisgen/config.py` | `src/core/config.py` |
| `src/croquisgen/perception.py` | `src/perception/mediapipe_pose.py` |
| `src/croquisgen/renderers/base.py` | `src/render/base.py` |
| `src/croquisgen/renderers/outline.py` | `src/render/outline.py` |
| `src/croquisgen/renderers/__init__.py` | `src/render/registry.py` |
| `src/croquisgen/renderers/gesture_croquis.py` | `src/gesture_croquis/renderer.py` |
| `src/croquisgen/renderers/constructive.py` | `src/constructive_anatomy/renderer.py` |
| `scripts/run.py` | `main.py` (저장소 루트) |
| `scripts/_bootstrap.py` | 삭제 |
| `scripts/bench_perception.py` | `tools/bench_perception.py` (저장소에서 추적 해제) |

`git log --follow <파일>`로 이력이 그대로 따라온다.

### import 변경 규칙

```python
from croquisgen import config              →  from core import config
from croquisgen.types import PerceptionResult  →  from core.types import PerceptionResult
from croquisgen.types import L_SHOULDER    →  from core.landmarks import L_SHOULDER
from croquisgen.renderers import get_renderer   →  from render.registry import get_renderer

# 렌더러 안에서
from .base import blank_paper              →  from render.base import blank_paper
from .outline import draw_silhouette       →  from render.outline import draw_silhouette
```

`core/types.py`에는 데이터 계약(`PerceptionResult`, `AppState`, `Style`, `Target`)만 있고,
랜드마크 인덱스 상수와 엣지 테이블은 `core/landmarks.py`로 갈라졌다.

### 주의할 점 두 가지

**렌더러에서 `from render import ...`를 쓰면 순환 import가 된다.** `render/registry.py`가
렌더러를 import하므로, 렌더러는 `render.base` / `render.outline`을 직접 지정해야 한다.
`render/__init__.py`는 이 이유로 비어 있다.

**`src/` 바로 아래에 표준 라이브러리와 같은 이름의 모듈을 두면 안 된다.** `src/`가
`sys.path[0]`이므로 최상위 `types.py`는 표준 라이브러리 `types`를 가리고, numpy·cv2·
mediapipe 내부의 `import types`가 그 파일을 집어 전부 깨진다. 계약이 `core/` 안에 있는
이유다. 새 폴더를 만들 때 `types`, `io`, `json`, `copy`, `queue` 등을 피한다.

### 그 외 변경

| 항목 | 내용 |
|---|---|
| 진입점 | `python main.py`. `_bootstrap.py`를 없애고 `sys.path` 설정을 `main.py` 안에 직접 둬서 import 순서 의존을 제거 |
| `perception/` | `get_backend()`로 검출기를 선택한다. `core/config.py`의 `POSE_BACKEND = "mediapipe"` 한 줄이 전환점. RTMPose 실시간 경로가 생기면 두 번째 백엔드로 들어온다 |
| 클래스명 | `Perception` → `MediaPipePerception`. `FrameClock`은 `app/loop.py`로 이동 |
| 모델 | `hand_landmarker.task` 삭제 (`GestureRecognizer`가 같은 21점 + 라벨을 같은 비용으로 주므로 호출부가 없었다). 다운로드 3개, 약 23 MB |
| 병합 충돌 | `constructive.py`에 커밋돼 있던 충돌 마커 21군데 해소. 해소 전에는 clone 시 import 단계에서 실패했다 |
| 테스트 | 삭제됐던 `tests/` 복구. 픽스처에 눈·귀·손끝·발꿈치·발끝 추가 — 기본값 `0.5`는 가슴 중앙이라 그럴듯한 좌표처럼 보이면서 렌더 결과를 조용히 바꾼다 |
| `.gitignore` | `.ipynb_checkpoints/`, `*.Rhistory`, `tools/`, `out/` 추가 |

---

## 다음 작업 전 확인

- 각자 작업 브랜치가 있으면 위 개편 커밋들을 먼저 merge하고 import 경로를 규칙대로 고친다
- `python -m pytest tests -q`로 60개 통과를 확인한다
- 렌더러만 보려면 `python scripts/render_still.py 사진.jpg` (카메라 불필요, 두 스타일 나란히)
- 동세 크로키 사진 파이프라인은 `cd src && python -m gesture_croquis.run --demo`
- 프레임 예산을 건드렸으면 `python tools/bench_threading.py 사진.jpg`와
  `python tools/smoke_threading.py 사진.jpg`를 같이 본다. 사진 없이 돌리면 사람이 검출되지
  않아 인식 절대값이 검출기 비용만 나온다 — 비율은 유효하고 밀리초는 아니다
