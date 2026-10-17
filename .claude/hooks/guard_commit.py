#!/usr/bin/env python3
"""PreToolUse(Bash) hook: enforce the project's commit conventions.

Blocks `git commit` commands whose message carries AI attribution
(Co-Authored-By / "Generated with"): the author does not want it anywhere.
"""
import json
import re
import sys

try:
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
except (json.JSONDecodeError, AttributeError):
    raise SystemExit(0)

if re.search(r"\bgit\s+(?:-\S+\s+)*commit\b", command) and re.search(
        r"co-authored-by|generated with", command, re.I):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            "Convención del proyecto (AGENTS.md): los commits no llevan "
            "'Co-Authored-By' ni 'Generated with'. Quítalo del mensaje."),
    }}))
