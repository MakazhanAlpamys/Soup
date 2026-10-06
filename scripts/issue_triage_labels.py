"""Keep the triage labels of an issue consistent.

Run by ``.github/workflows/issue-triage.yml`` on ``issues`` events:

* a new issue that has no ``difficulty:N`` label gets ``needs-triage``;
* ``needs-triage`` is removed as soon as a ``difficulty:N`` label is set;
* more than one ``difficulty:N`` label, or an unknown ``difficulty:...`` label, fails the run,
  so whoever added the second label sees it.

The script only ever adds or removes ``needs-triage``. It reads nothing but the issue number,
the repository and the event action from the environment, and it asks ``gh`` for the CURRENT
labels instead of trusting the event payload, so a burst of label changes cannot act on stale
state. It never reads issue text.

The ``opened`` run and a ``labeled`` run for the same issue can overlap (they are in different
concurrency groups), so after adding ``needs-triage`` the labels are read once more: a level
that was set while the first read was in flight clears it again. A level set after that second
read starts its own run, which then finds ``needs-triage`` already there and removes it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass

NEEDS_TRIAGE = "needs-triage"
ACTIONS = frozenset({"opened", "labeled", "unlabeled"})
_LEVEL = re.compile(r"difficulty:(?:[1-9]|10)")
# A path component made only of dots ("." or "..") is never a repository name.
_COMPONENT = r"(?!\.+(?:/|\Z))[A-Za-z0-9_.][A-Za-z0-9_.-]*"
_REPO = re.compile(_COMPONENT + "/" + _COMPONENT)
_NUMBER = re.compile(r"[1-9][0-9]{0,9}")


@dataclass(frozen=True)
class Decision:
    """What to do to one issue: labels to add, labels to remove, and a problem to report."""

    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    error: str = ""


def _level_number(label: str) -> int:
    return int(label.split(":", 1)[1])


def decide(labels: Iterable[str], action: str) -> Decision:
    """Work out the label changes for an issue with these labels after this event action."""
    names = set(labels)
    levels = sorted((name for name in names if _LEVEL.fullmatch(name)), key=_level_number)
    unknown = sorted(
        name for name in names if name.startswith("difficulty:") and not _LEVEL.fullmatch(name)
    )

    problems = []
    if len(levels) > 1:
        problems.append("more than one difficulty label: " + ", ".join(levels))
    if unknown:
        problems.append("unknown difficulty label: " + ", ".join(unknown))

    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    if levels and NEEDS_TRIAGE in names:
        remove = (NEEDS_TRIAGE,)
    elif action == "opened" and not levels and NEEDS_TRIAGE not in names:
        add = (NEEDS_TRIAGE,)
    return Decision(add=add, remove=remove, error="; ".join(problems))


def _escape(text: str) -> str:
    """Make text safe to print on a line the runner reads for workflow commands.

    The runner treats a stdout line that starts with ``::`` as a command (``::error::``,
    ``::add-mask::`` and so on), and ``%25``, ``%0D``, ``%0A`` are its escapes. A label name
    with a line break must not be able to start a second line.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _say(text: str) -> None:
    print(_escape(text))


def _error(text: str) -> None:
    print("::error::" + _escape(text))


def _gh(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _report_gh_failure(what: str, result: subprocess.CompletedProcess) -> int:
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    _error(f"{what} failed: {detail[0] if detail else 'no output'}")
    return 1


def _read_labels(repo: str, number: str) -> list[str] | int:
    """The issue's current label names, or an exit code when they cannot be read."""
    view = _gh(["gh", "issue", "view", number, "--repo", repo, "--json", "labels"])
    if view.returncode != 0:
        return _report_gh_failure("gh issue view", view)
    try:
        return [item["name"] for item in json.loads(view.stdout)["labels"]]
    except (ValueError, KeyError, TypeError):
        _error("gh issue view returned something that is not a label list")
        return 1


def _apply(repo: str, number: str, decision: Decision) -> int:
    """Run the edit a decision asks for; 0 when done or when there was nothing to do."""
    if not (decision.add or decision.remove):
        return 0
    command = ["gh", "issue", "edit", number, "--repo", repo]
    for name in decision.add:
        command += ["--add-label", name]
    for name in decision.remove:
        command += ["--remove-label", name]
    edit = _gh(command)
    if edit.returncode != 0:
        return _report_gh_failure("gh issue edit", edit)
    changes = [f"added {name}" for name in decision.add]
    changes += [f"removed {name}" for name in decision.remove]
    _say(f"issue #{number}: " + ", ".join(changes))
    return 0


def main() -> int:
    repo = os.environ.get("REPO", "")
    number = os.environ.get("NUMBER", "")
    action = os.environ.get("EVENT_ACTION", "")

    if not _REPO.fullmatch(repo):
        _error("REPO is missing or malformed")
        return 2
    if not _NUMBER.fullmatch(number):
        _error("NUMBER is missing or malformed")
        return 2
    if action not in ACTIONS:
        _error("EVENT_ACTION is missing or not an action this workflow handles")
        return 2

    labels = _read_labels(repo, number)
    if isinstance(labels, int):
        return labels
    decision = decide(labels, action)
    failed = _apply(repo, number, decision)
    if failed:
        return failed
    if not (decision.add or decision.remove):
        _say(f"issue #{number}: nothing to change")

    if decision.add:
        # The opened run can overlap the run of a level that was set a moment later.
        labels = _read_labels(repo, number)
        if isinstance(labels, int):
            return labels
        recheck = decide(labels, "labeled")
        failed = _apply(repo, number, recheck)
        if failed:
            return failed
        decision = Decision(error=recheck.error)

    if decision.error:
        _error(f"issue #{number}: {decision.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
