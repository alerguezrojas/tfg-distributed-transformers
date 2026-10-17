"""Hooks of the AI harness (.claude/hooks): the commit guard only judges the
`git commit` message, never other text in the same command."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "guard_commit.py"
_spec = importlib.util.spec_from_file_location("guard_commit", _PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.mark.parametrize("command", [
    'git commit -m "feat: x\n\nCo-Authored-By: Someone <a@b.c>"',
    'git -C repo commit -m "fix: y (Generated with a tool)"',
    'git add . && git commit -q -m "docs: z\n\nco-authored-by: x"',
])
def test_blocks_attribution_in_commit_messages(command):
    assert guard.is_blocked(command)


@pytest.mark.parametrize("command", [
    'git commit -m "feat: limpio"',
    # The false positive that motivated the parser: the words appear in a PR body.
    'git commit -m "chore: x" && gh pr create --body "rechaza un mensaje con Co-Authored-By"',
    'grep -n "Co-Authored-By" CHANGELOG.md',
    'echo "Generated with" > /tmp/f && git status',
    'git log --grep "Co-Authored-By"',
])
def test_lets_other_commands_through(command):
    assert not guard.is_blocked(command)
