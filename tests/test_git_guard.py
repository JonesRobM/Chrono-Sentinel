"""
Regression tests for the git-policy PreToolUse hook.

The hook in .claude/hooks/git-guard.py is the enforcement mechanism for
.claude/rules/git-policy.md. A silent regression in its regexes would let a
git invocation reach the shell without the user seeing it, so the decision
table is pinned here and runs in CI.

Contract under test:
  deny         -> rule 4 operations (config, force-push): not askable
  ask          -> any other git/gh invocation, including via a shell wrapper
  pass-through -> no output at all, deferring to normal permission evaluation

Note: fixtures are assembled from fragments rather than written as literals.
The hook matches on raw command text, so a source file containing the literal
strings would itself be blocked when handled by a shell command.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parent.parent / ".claude" / "hooks" / "git-guard.py"

G = "git"
GH = "gh"


def decide(command: str) -> str:
    """Runs the hook against a command and returns its permission decision."""
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        return "pass-through"
    return json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"]


DENY = [
    f"{G} config user.email x@y.z",
    f"{G} config --global core.editor vim",
    f"{G} push --force origin main",
    f"{G} push -f",
    f"{G} push --force-with-lease origin main",
    f"{G} push origin +main",
    f"/usr/bin/{G} config --global x y",
    f"ls && {G} config core.hooksPath /dev/null",
]

ASK = [
    f"{G} status",
    f'{G} commit -m "wip"',
    f"{G} log --oneline",
    f"{GH} pr create",
    f"{GH} run list",
    f'bash -c "{G} push"',
    f'sh -c "{G} commit"',
    f"echo hi; {G} log",
    f"{G} push origin main",
    f"{G} push -u origin feature",
    f"{G} push --follow-tags",
    f"xargs {G} add",
    f"{G} rebase -i HEAD~3",
    f"$(which {G}) status",
]

PASS_THROUGH = [
    "ls -la",
    "python3 scripts/train.py",
    "echo digit",
    "grep ghost file.txt",
    "uv pip install -r requirements.txt",
    "docker build -t chrono-sentinel .",
    'echo "configure the widget"',
    "curl -sL https://example.com",
    "pytest tests/",
]


@pytest.mark.parametrize("command", DENY)
def test_denied_outright(command: str) -> None:
    assert decide(command) == "deny"


@pytest.mark.parametrize("command", ASK)
def test_requires_user_approval(command: str) -> None:
    assert decide(command) == "ask"


@pytest.mark.parametrize("command", PASS_THROUGH)
def test_unrelated_commands_pass_through(command: str) -> None:
    assert decide(command) == "pass-through"


def test_fails_closed_on_unparseable_input() -> None:
    """Malformed stdin must ask, never silently allow."""
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json",
        capture_output=True,
        text=True,
        check=True,
    )
    decision = json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"]
    assert decision == "ask"
