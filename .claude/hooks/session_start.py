#!/usr/bin/env python3
"""SessionStart hook: inject the "Estado actual" section of docs/roadmap.md.

Whoever (or whatever agent) opens a session starts from the project's real state,
which lives in the repository, not in a previous chat.
"""
import os
import re
from pathlib import Path

root = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
roadmap = root / "docs" / "roadmap.md"
try:
    text = roadmap.read_text(encoding="utf-8")
except OSError:
    raise SystemExit(0)

match = re.search(r"^## Estado actual\n(.*?)(?=^## |\Z)", text, re.S | re.M)
if match:
    print("Estado actual del proyecto (docs/roadmap.md):\n")
    print(match.group(1).strip())
