# CroquisGen

웹캠에 잡힌 사람을 분석해 크로키 연습용 레퍼런스 이미지를 실시간으로 생성한다.
스타일 전환/캡처는 손동작으로 한다.

두 가지 생성 스타일이 존재한다.

- **동세 크로키 (Gesture Croquis)** — 인체의 흐름을 간결한 선으로 추상화
- **인체 도형화 (Constructive Anatomy)** — 구/원기둥/타원체로 부위별 비율과 입체 구조를 단순화

## 설치

```bash
pip install -r requirements.txt
python scripts/download_models.py   # MediaPipe 모델 3개, 약 23 MB
```

## 실행

```bash
python main.py
```

| 옵션 | 효과 |
|---|---|
| `--lite` | 가벼운 pose 모델 사용 (프레임당 약 25 ms → 15 ms, 정확도 하락) |
| `--show-camera` | 크로키 대신 카메라 원본을 표시 (제스처 디버깅용) |
| `--no-snap` | 핑거스냅 인식 비활성화 |
| `--camera N` | 카메라 장치 번호 |
| `--placeholder` | 동세 크로키를 가벼운 기하 근사로 (기본은 S4) |
| `--no-threads` | 캡처·인식·렌더를 한 스레드에서 (기준선) |
| `--render-workers N` | 렌더 스레드 수 (기본 1) |

## 조작

| 손동작 | 동작 |
|---|---|
| 손바닥을 카메라로 내밀기 | 5초 카운트다운 뒤 그 순간의 크로키를 `captures/`에 저장 |
| 검지를 1초 이상 세우기 | 손가락 주위에 원형 툴바 표시 |
| 툴바가 열린 상태에서 손가락을 위/아래로 이동 | 위 = 동세 크로키, 아래 = 인체 도형화 |
| 핑거 스냅 | 전신 ↔ 손 모드 전환 |


| 키 | 동작 |
|---|---|
| `1` / `2` | 동세 크로키 / 인체 도형화 |
| `m` | 전신 ↔ 손 전환 |
| `c` | 캡처 |
| `q` 또는 `esc` | 종료 |


## 구조

```
main.py                  진입점
src/
  core/                  단계 간 계약 — 나머지 전부가 이것만 공유한다
    types.py             PerceptionResult, AppState, Style, Target
    landmarks.py         랜드마크 인덱스와 골격 엣지
    config.py            튜너블 (프레임 예산에 영향을 주는 값은 비용 주석과 함께)
  app/                   실시간 루프
    loop.py              스레드 루프 · 단일 스레드 루프 · 키 입력
    pipeline.py          단계 사이의 최신값 슬롯과 워커 스레드
    camera.py            웹캠 입력
    gestures.py          손동작 → 이벤트 변환
    state.py             모드·스타일 상태와 캡처 저장
    hand_roi.py          전신 골격에서 손 ROI 추출
    ui/                  원형 툴바, 상태 표시줄, 캡처 알림
  perception/            랜드마크 공급. 검출기를 갈아끼우는 지점
    mediapipe_pose.py    MediaPipe Tasks pose / gesture
  render/                드로잉 계약과 공용 도구
    base.py              렌더러 프로토콜, 종이, 좌표 스케일, 획
    outline.py           세그멘테이션 마스크 윤곽
    registry.py          Style → 렌더러
  gesture_croquis/       동세 크로키
    renderer.py          실시간 렌더러 (S4 엔진 / 기하 근사 두 가지)
    realtime.py          PerceptionResult → PoseFrame 어댑터
    s1_keypoints.py …    사진 한 장에서 3D 골격까지 (S1~S3)
    s4_gesture.py        골격 → 스트로크 추상화 (S4)
  constructive_anatomy/  인체 도형화 렌더러
scripts/                 모델 다운로드, 정지 이미지 렌더링
tests/                   카메라·모델 없이 도는 테스트
tools/                   프레임 예산 측정 (저장소에서 추적하지 않음)
```

