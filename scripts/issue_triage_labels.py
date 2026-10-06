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
_LEVEL = re.compile(r"difficulty:(?:[1-9]|10)")
_REPO = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*/[A-Za-z0-9_.][A-Za-z0-9_.-]*")
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


def _gh(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _report_gh_failure(what: str, result: subprocess.CompletedProcess) -> int:
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    print(f"::error::{what} failed: {detail[0] if detail else 'no output'}")
    return 1


def main() -> int:
    repo = os.environ.get("REPO", "")
    number = os.environ.get("NUMBER", "")
    action = os.environ.get("EVENT_ACTION", "")

    for name, value, pattern in (
        ("REPO", repo, _REPO),
        ("NUMBER", number, _NUMBER),
        ("EVENT_ACTION", action, re.compile(r"[a-z_]{1,32}")),
    ):
        if not pattern.fullmatch(value):
            print(f"::error::{name} is missing or malformed")
            return 2

    view = _gh(["gh", "issue", "view", number, "--repo", repo, "--json", "labels"])
    if view.returncode != 0:
        return _report_gh_failure("gh issue view", view)
    try:
        labels = [item["name"] for item in json.loads(view.stdout)["labels"]]
    except (ValueError, KeyError, TypeError):
        print("::error::gh issue view returned something that is not a label list")
        return 1

    decision = decide(labels, action)
    if decision.add or decision.remove:
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
        print(f"issue #{number}: " + ", ".join(changes))
    else:
        print(f"issue #{number}: nothing to change")

    if decision.error:
        print(f"::error::issue #{number}: {decision.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
