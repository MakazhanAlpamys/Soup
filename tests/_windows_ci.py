"""THE ``windows-latest`` illegal-instruction guard (#382). One definition.

Imported by every test file that reaches a real ``trainer.train()``. It lives
here rather than in one of those files because the guard has now been needed in
two of them, and this repo has been bitten three times by a predicate that was
copied instead of shared (#372, #392, #424). ``test_windows_ci_guard_is_not_
duplicated`` fails if a second copy appears.
"""

from __future__ import annotations

import os
import sys

import pytest


def _windows_ci() -> bool:
    """True on a GitHub ``windows-latest`` runner, false on a Windows dev box.

    #382 — part of GitHub's ``windows-latest`` fleet lacks an instruction the
    bitsandbytes wheel emits, so the first test that reaches a real
    ``trainer.train()`` dies with ``Windows fatal exception: code 0xc000001d``
    (ILLEGAL_INSTRUCTION): a faulthandler dump, no Python exception, and a dead
    interpreter. Occurrences so far span py3.10, py3.11 and py3.12, so it tracks
    the runner CPU rather than the interpreter; the control is that ``1715261``
    is the crashing tree ``7c9a931`` plus one line of markdown and came back
    green on the same pool.

    Skipping it is not hiding a failure. The crash kills the process, so the
    cell reports NOTHING about the ~17,000 tests it had not reached — one test
    censoring the whole Windows matrix, which is the reverse of what a test
    suite is for. What is skipped stays covered on ubuntu 3.10/3.11/3.12 and
    macos 3.10/3.11/3.12, and the assertions are platform-independent.

    Deliberately narrow: ``CI`` is set to ``true`` by GitHub Actions and by
    nothing on a developer's machine, so the maintainer's own Windows box —
    where the CPU is known — keeps running these tests. What is excluded is an
    unknown CPU, not a platform.
    """
    return sys.platform == "win32" and os.environ.get("CI") == "true"


skip_on_windows_ci = pytest.mark.skipif(
    _windows_ci(),
    reason=(
        "#382: a real trainer.train() hits an illegal instruction on part "
        "of GitHub's windows-latest fleet and kills the interpreter, which "
        "censors every other test in the cell. NOT a statement about this "
        "code path on Windows: still covered on ubuntu + macos, and still "
        "live on a local Windows box."
    ),
)
