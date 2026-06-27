"""Backward-compatible alias of the ParaViT-Lab launcher.

The canonical entry point is ``paravit.py`` (``uv run paravit.py <command>``).
This file is kept so that the older ``uv run tfg.py <command>`` invocation, used
throughout the project history and runbooks, keeps working unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import app

if __name__ == "__main__":
    app()
