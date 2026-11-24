"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

import pytest

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _no_machine_profile(monkeypatch):
    """Tests never pick up a machine profile from the host they run on (``none`` disables it)."""
    monkeypatch.setenv("PARAVIT_MACHINE", "none")
