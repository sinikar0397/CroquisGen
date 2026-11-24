#!/usr/bin/env python3
"""CroquisGen entry point.

    python main.py                      웹캠 실행
    python main.py --lite               가벼운 pose 모델 (프레임당 25 ms -> 15 ms)
    python main.py --show-camera        크로키 대신 카메라 원본 표시
    python main.py --no-snap            핑거스냅 인식 끄기
    python main.py --camera 1           카메라 장치 번호
    python main.py --no-threads         캡처·인식·렌더를 한 스레드에서 (기준선)
    python main.py --placeholder        동세 크로키를 가벼운 기하 근사로 (S4가 기본)

src/ is put on the path here rather than in a helper module, so there is no
import-order requirement: sys.path has to be set before `from app.loop import
main` runs, and a plain statement cannot be reordered by an import sorter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.loop import main  # noqa: E402  (path set above)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
