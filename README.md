# CroquisGen

웹캠에 잡힌 사람을 분석해 크로키 연습용 레퍼런스 이미지를 실시간으로 생성한다.
스타일 전환/캡처는 손동작으로 한다.

두 가지 생성 스타일이 존재한다.

- **동세 크로키 (Gesture Croquis)** — 인체의 흐름을 간결한 선으로 추상화
- **인체 도형화 (Constructive Anatomy)** — 구/원기둥/타원체로 부위별 비율과 입체 구조를 단순화

## 설치

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
