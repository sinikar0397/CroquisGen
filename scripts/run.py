"""Entry point: python scripts/run.py [--lite] [--show-camera]"""
import sys

import _bootstrap  # noqa: F401

from croquisgen.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
