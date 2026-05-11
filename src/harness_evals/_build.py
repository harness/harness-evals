"""Build entry point for `poetry run build`."""

import subprocess
import sys


def main():
    sys.exit(subprocess.call([sys.executable, "-m", "build", "."]))
