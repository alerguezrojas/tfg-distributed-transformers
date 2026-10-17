#!/usr/bin/env python3
"""PreToolUse(Bash) hook: enforce the project's commit conventions.

Blocks `git commit` commands whose message carries AI attribution
(Co-Authored-By / "Generated with"): the author does not want it anywhere.

Only the arguments of the `git commit` itself are inspected, so a command that
merely mentions those words elsewhere (a PR body, a grep, a test fixture) is not
blocked.
"""
import json
import re
import shlex
import sys

_ATTRIBUTION = re.compile(r"co-authored-by|generated with", re.I)
_OPERATORS = {"&&", "||", ";", "|", "&", ";;", "(", ")"}


def commit_arguments(command: str) -> list[list[str]]:
    """Arguments of every `git commit` in a shell command line."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:                     # unbalanced quotes: fall back to plain split
        tokens = command.split()
    commits, i = [], 0
    while i < len(tokens):
        if tokens[i] == "git":
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-"):   # git -C dir, -c k=v …
                j += 2 if tokens[j] in ("-C", "-c") else 1
            if j < len(tokens) and tokens[j] == "commit":
                k = j + 1
                while k < len(tokens) and tokens[k] not in _OPERATORS:
                    k += 1
                commits.append(tokens[j + 1:k])
                i = k
                continue
        i += 1
    return commits


def is_blocked(command: str) -> bool:
    return any(_ATTRIBUTION.search(" ".join(args)) for args in commit_arguments(command))


def main() -> None:
    try:
        command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    except (json.JSONDecodeError, AttributeError):
        return
    if is_blocked(command):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Convención del proyecto (AGENTS.md): los commits no llevan "
                "'Co-Authored-By' ni 'Generated with'. Quítalo del mensaje."),
        }}))


if __name__ == "__main__":
    main()
