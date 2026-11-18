# 개발 진척도

2026-11-18 기준. 계획서(`3402김주환 - 모션 인식 프로젝트 계획서.docx`)의 기능 단위로
정리한다. 뒤쪽에 11/18 폴더 구조 개편의 경로 대응표가 있다.

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

### 진행 중

**인체 도형화 (김민재)** — `constructive_anatomy/renderer.py`, 444줄

현재 들어간 것: 프레임 간 지수 스무딩(z는 더 약하게) → 머리 높이 H 기준 8등신 비율표로
도형 크기 결정 → 표준 길이를 넘는 사지는 검출 오류로 보고 보정 → depth 순서대로 쌓아
합성. 가려지는 선은 지우지 않고 연한 잉크로 남기고, 원기둥 끝 캡은 단축률에 따라
납작해지며 가까운 쪽은 전체 링, 먼 쪽은 반 링으로 그린다.

수정 예정: 인체 세분화, 도형 구체화. 세부 범위는 담당자 확인 필요.

**동세 크로키 (박준현)** — 세 단계로 나뉘며 앞단만 끝났다.

| 단계 | 상태 | 위치 |
|---|---|---|
| S1~S3 분석 | 완료 (13개 모듈, 2,964줄) | `gesture_croquis/` |
| S4~S5 선 생성 | 미착수 | — |
| 렌더러 | placeholder (108줄, 기하학적 근사) | `gesture_croquis/renderer.py` |

분석 단계는 사진 한 장에서 2D 키포인트를 뽑고, 뼈 길이 사전확률·관절 가동범위·근육 이완
자세·중력 토크·균형·바닥 평면을 한 목적함수에 넣어 미터 단위 3D 골격을 맞춘다. depth
모호성은 투시 기하의 두 해를 모두 탐색해 타당성 에너지로 고르고, 팔이 몸통 앞/뒤인지는
이미지 윤곽의 엣지 연속성으로 판정한다. 척추 곡선과 관절별 토크·근육 스트레스까지 낸다.

사진 입력 전용이고 실시간 루프에 연결되어 있지 않다. S2 단독으로 250~600 ms다.

### 해결이 필요한 것

**프레임 예산 초과.** 30 fps는 프레임당 33 ms인데 인체 도형화 스타일에서 약 43 ms(약 23 fps)다.

| | 인식 | 렌더 | 합계 |
|---|---|---|---|
| 동세 크로키 (placeholder) | ~31 ms | 5.5 ms | ~37 ms |
| 인체 도형화 | ~31 ms | 11.8 ms (손 14.3) | ~43 ms |

예산을 세울 때 두 렌더러가 placeholder여서 렌더링 비용을 계산에 넣지 않았다. 세분화와
도형 구체화가 진행되면 이 몫은 더 커진다. 줄일 수 있는 순서는 `--lite` 관절 모델
(25 → 15 ms, 합계 약 35 ms), 손 인식 간격 3 → 4프레임(약 1.6 ms), 렌더러 내부 최적화
(depth 정렬 합성과 윤곽 추출이 비용의 대부분). 위 렌더 수치는 합성 랜드마크 측정값이고
인식 수치는 이전 측정값이므로, 실제 사진으로 `tools/bench_perception.py`를 돌려 합계를
확인해야 한다.

**핑거 스냅 미검증.** 계획서의 제스처 3개 중 전신↔손 전환만 사전학습 카테고리가 없어
직접 구현했다. 스냅은 수십 ms 안에 끝나 30 fps에서 1~2프레임이고, 모션 블러가 판단에
필요한 랜드마크를 망가뜨린다. 2단계 휴리스틱이 들어가 있지만 실사용 신뢰도가 확인되지
않았다. 대안은 이 전환을 툴바의 세 번째 섹터로 옮기는 것 — 새 모델이 필요 없고 툴바는
이미 안정적이다.

**제스처 안정화 필터가 실질적으로 무효다.** 손 인식 결과가 3프레임 재사용되는데 이벤트
해석은 매 프레임 돌아, 같은 결과가 연속으로 세어져 2회 임계값이 즉시 충족된다. 오분류
한 번이 3프레임으로 증폭된다. `MediaPipePerception.hand_ran_this_frame()`이 이 문제를
위해 있지만 호출되지 않는다.

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
| 동세 크로키 선 생성 (S4~S5) | 분석 결과를 쓸지, 실시간 랜드마크만 쓸지 미정 |
| 학습 데이터 준비 | 인터넷 크로키에서 골격 추출은 실사 학습 모델이라 line art에서 잘 안 된다. 렌더링된 3D 포즈가 신뢰할 수 있는 출처. 오프라인 작업이라 프레임 예산과 무관 |
| 웹 전환 | 브라우저 MediaPipe가 동일 랜드마크를 주므로 `core/`와 `render/`는 그대로 따라간다. `app/`은 버려지는 쪽 |
| S2/S3 테스트 이전 | 독립 프로젝트 시절의 검증 7개가 아직 `tests/`에 없다 |

### 테스트

카메라와 모델 파일 없이 도는 23개가 통과한다.

- `tests/test_pipeline.py` — 렌더러 출력 규약(크기·dtype·빈 입력 무예외·저신뢰 사지 생략),
  제스처 쿨다운·홀드·섹터 선택·취소, 손 ROI 기하, 상태 전이와 캡처
- `tests/test_constructive.py` — 인체 도형화 내부 기하(길이 보정, 단축률, 프레임 간 스무딩,
  손바닥 윤곽)

미검증 영역은 제스처 안정화 필터의 프레임 카운트, 툴바 하이라이트, MediaPipe 호출 자체다.

---

## 2. 11/18 폴더 구조 개편

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

### 다음 작업 전 확인

- 각자 작업 브랜치가 있으면 이 개편 커밋을 먼저 merge하고 import 경로를 위 규칙대로 고친다
- `python -m pytest tests -q`로 23개 통과를 확인한다
- 렌더러만 보려면 `python scripts/render_still.py 사진.jpg` (카메라 불필요, 두 스타일 나란히)
- 동세 크로키 분석은 `cd src && python -m gesture_croquis.run --demo`
