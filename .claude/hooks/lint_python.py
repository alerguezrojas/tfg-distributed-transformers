#!/usr/bin/env python3
"""PostToolUse(Edit|Write) hook: run ruff on the Python file just edited.

Only real errors (pyflakes + syntax, see [tool.ruff] in pyproject.toml). Problems
are reported back to Claude so they are fixed in the same turn.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    raise SystemExit(0)
path = (payload.get("tool_input") or {}).get("file_path", "")
if not path.endswith(".py") or not Path(path).exists():
    raise SystemExit(0)

root = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
venv_ruff = root / ".venv" / "bin" / "ruff"
if venv_ruff.exists():
    cmd = [str(venv_ruff)]
elif shutil.which("ruff"):
    cmd = ["ruff"]
else:
    raise SystemExit(0)                       # no ruff available: never block editing

result = subprocess.run([*cmd, "check", "--output-format", "concise", path],
                        cwd=root, capture_output=True, text=True)
if result.returncode == 1 and result.stdout.strip():
    print(json.dumps({
        "decision": "block",
        "reason": "ruff ha encontrado errores en el fichero editado:\n" + result.stdout.strip(),
    }))
