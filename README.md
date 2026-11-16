# CroquisGen

웹캠에 잡힌 사람을 분석해 크로키 연습용 레퍼런스 이미지를 실시간으로 생성한다.
스타일 전환·캡처는 버튼이 아니라 손동작으로 한다.

생성 스타일은 두 가지다.

- **동세 크로키 (Gesture Croquis)** — 인체의 흐름을 간결한 선으로 추상화
- **인체 도형화 (Constructive Anatomy)** — 구·원기둥·타원체로 부위별 비율과 입체 구조를 단순화

각 스타일은 전신과 손, 두 대상에 모두 적용된다. 생성형 AI가 특히 약한 손 구조를
별도 모드로 분리한 이유다.

## 설치

Python 3.12, macOS(Apple Silicon)에서 확인했다. GPU는 필요 없다.

```bash
pip install -r requirements.txt
python scripts/download_models.py   # MediaPipe 모델 4개, 약 31 MB
```

## 실행

```bash
python scripts/run.py
```

| 옵션 | 효과 |
|---|---|
| `--lite` | 가벼운 pose 모델 사용 (프레임당 약 25 ms → 15 ms, 정확도 하락) |
| `--show-camera` | 크로키 대신 카메라 원본을 표시 (제스처 디버깅용) |
| `--no-snap` | 핑거스냅 인식 비활성화 |
| `--camera N` | 카메라 장치 번호 |

## 조작

| 손동작 | 동작 |
|---|---|
| 손바닥을 카메라로 내밀기 | 현재 크로키를 `captures/`에 저장 |
| 검지를 1초 이상 세우기 | 손가락 주위에 원형 툴바 표시 |
| 툴바가 열린 상태에서 손가락을 위/아래로 이동 | 위 = 동세 크로키, 아래 = 인체 도형화 |
| 핑거 스냅 | 전신 ↔ 손 모드 전환 |

모든 동작에는 키보드 대응이 있다. 렌더러를 수정하면서 매번 카메라 앞에서 포즈를
잡을 수는 없기 때문이다.

| 키 | 동작 |
|---|---|
| `1` / `2` | 동세 크로키 / 인체 도형화 |
| `m` | 전신 ↔ 손 전환 |
| `c` | 캡처 |
| `q` 또는 `esc` | 종료 |

## 웹캠 없이 작업하기

렌더러 개발은 정지 이미지로 하는 쪽이 빠르다. 두 스타일을 나란히 출력한다.

```bash
python scripts/render_still.py 사진.jpg -o out.png
python scripts/render_still.py 사진.jpg --target hand
```

## 구조

```
src/croquisgen/
  app.py          메인 루프
  perception.py   MediaPipe pose / hand 래퍼
  gestures.py     손동작 → 이벤트 변환
  hand_roi.py     전신 골격에서 손 ROI 추출
  state.py        모드·스타일 상태와 캡처 저장
  renderers/      스타일별 생성기 (landmarks → 이미지)
  ui/             원형 툴바, 상태 표시줄, 캡처 알림
scripts/          실행·모델 다운로드·벤치마크·정지 이미지 렌더링
```

렌더러는 `PerceptionResult`를 받아 이미지를 돌려주는 순수 함수다. 카메라나 화면,
앱 상태를 건드리지 않는다. 변환 모델 두 개를 각자 작성하기 위한 경계이자, 나중에
웹으로 옮길 때 그대로 따라오는 부분이다.

## 테스트

카메라도 모델 파일도 필요 없다. 합성 랜드마크만 사용한다.

```bash
python -m pytest tests -q
```

## 담당

| 영역 | 담당 |
|---|---|
| 동세 크로키 변환 모델 (`renderers/gesture_croquis.py`) | 3308 박준현 |
| 인체 도형화 변환 모델 (`renderers/constructive.py`) | 3802 김민재 |
| 손 ROI, 제스처 인식, 화면 구현, 전체 연결 | 3402 김주환 |

동세 크로키 변환 모델은 현재 기하학적 placeholder가 들어 있다. 앱이 끝에서 끝까지
돌아가게 해두고 교체하면 된다.

인체 도형화는 학습 모델 없이 관절 좌표와 8등신 비율표로 도형을 만든다. 프레임 간
스무딩, 비율을 벗어난 사지 길이 보정, 투영 길이로 추정한 원근(끝단 타원의 눌림,
가까운 쪽은 온전한 원·먼 쪽은 반원), 몸통·머리의 방향을 보여주는 중심선과 단면선이
들어 있다. 자세한 단계는 `renderers/constructive.py` 상단 주석에 있다.
