"""ParaViT-Lab — entry point for the unified CLI: ``uv run paravit.py <command>``.

Thin launcher so the whole project has one terminal entry point. The commands
live in ``src/cli.py``. Run ``uv run paravit.py --help`` for the full list.

``tfg.py`` is kept as a backward-compatible alias of this launcher.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import app

if __name__ == "__main__":
    app()
