"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
