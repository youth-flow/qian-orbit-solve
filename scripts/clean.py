"""Remove generated files created by the build workflow."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    ROOT / "build",
    ROOT / "data" / "generated",
    ROOT / "src" / "__pycache__",
    ROOT / "scripts" / "__pycache__",
    ROOT / "examples" / "2d-gravity" / "__pycache__",
]

PATTERNS = [
    "*.aux",
    "*.bbl",
    "*.bcf",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.lof",
    "*.log",
    "*.lot",
    "*.out",
    "*.run.xml",
    "*.synctex.gz",
    "*.toc",
    "*.xdv",
    "report.pdf",
]


def main() -> None:
    for path in DIRS:
        shutil.rmtree(path, ignore_errors=True)
    for pattern in PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                path.unlink()


if __name__ == "__main__":
    main()
