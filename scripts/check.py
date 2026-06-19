"""Run lightweight repository checks."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_python_files() -> list[Path]:
    dirs = [ROOT / "src", ROOT / "scripts", ROOT / "examples" / "2d-gravity"]
    files: list[Path] = []
    for directory in dirs:
        if directory.exists():
            files.extend(sorted(directory.glob("*.py")))
    return files


def main() -> None:
    for path in iter_python_files():
        py_compile.compile(str(path), doraise=True)

    generated = ROOT / "data" / "generated"
    if not generated.exists():
        print("data/generated is missing; run `make data` or `python src/run_all.py` before self_check.")
        return

    subprocess.run([sys.executable, str(ROOT / "src" / "self_check.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
