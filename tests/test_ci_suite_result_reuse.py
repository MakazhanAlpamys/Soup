"""A pull request's ubuntu / 3.12 cell does not run the suite twice for one merge commit.

The first run of a pull request is QUICK: ``lint`` and ``test (ubuntu-latest, 3.12)``. When a
maintainer then adds ``ci:full``, the APPROVAL run starts the same cell again. Measured
2026-11-12..15 (72 hours, 105 approval runs): 34 approval runs started on a head whose quick
run was already green, and in 14 of 14 sampled pairs (gaps of 0.4 to 6.1 hours) the two runs had
checked out the SAME merge commit (the log line ``HEAD is now at ... Merge <head> into <main>``
named one SHA twice: a ``labeled`` run reuses the merge commit of the push, it does not rebuild
it). The second run is therefore a repeat: 807 job-minutes of ``test (ubuntu-latest, 3.12)`` in
72 hours, about 270 a day. (The approval run's wall time does not shrink: the macOS cell, 43
minutes on average, is its long pole. What shrinks is the queue that every other run waits in.)

So the cell records one marker when it ends green (``actions/cache/save``, a few bytes, keyed on
``github.sha``: for a pull request that is the MERGE commit, so a push, a moved ``main`` or a
different pull request is a different key) and looks for it before it does anything else. On a
hit every other step is skipped and the job still reports SUCCESS under the same name, which is
what the merge gate reads: the required context ``test (ubuntu-latest, 3.12)`` is neither
renamed, nor skipped by a job-level ``if:``, nor missing.

What reuse gives up, and why it is accepted: the second run would have resolved the packages
that ``pinned-upstream.txt`` does not pin (numpy, pydantic, ...) a few hours later; a green
result describes the environment of the first run. The tree, the cell and the pinned stack are
the same. A result is never reused unless the first run ended green (the marker is written by a
step that needs ``success()``), never outside a pull request, and never for another cell.

Nothing here talks to GitHub: the workflow is read as YAML and the expressions are evaluated by
the interpreter of tests/test_ci_full_matrix_on_approval.py.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    ALL_MATRICES,
    CONTRIBUTING,
    _cell_dicts,
    _Expression,
    _jobs,
    _matrix,
    _render,
    _truthy,
)

LOOKUP = "Look for the green result of this merge commit"
NOTICE = "Say which run's result is reused"
MARKER = "Record the green result of this merge commit"
SAVE = "Save the green result of this merge commit"
HIT_GUARD = "steps.reuse.outputs.cache-hit != 'true'"
SUITE_GUARD = "env.RUN_SUITE == 'true'"
#: The one cell that repeats itself: the only one a quick run has.
REUSE_CELL = {"os": "ubuntu-latest", "python-version": "3.12"}
EXEMPT = (LOOKUP, NOTICE)


def _steps() -> list[dict[str, Any]]:
    return list(_jobs()["test"]["steps"])


def _step(name: str) -> dict[str, Any]:
    found = [step for step in _steps() if step.get("name") == name]
    assert len(found) == 1, (name, found)
    return found[0]


def _names() -> list[str]:
    return [str(step.get("name") or step.get("uses")) for step in _steps()]


def _contexts(
    cell: dict[str, str], *, event: str, hit: str | None = None, docs_only: str = "false"
) -> dict[str, Any]:
    """What a step's ``if:`` sees: the matrix cell, the plan outputs, ``RUN_SUITE`` and
    ``REUSE_CELL`` as the job's own ``env:`` renders them (never recomputed here, so a change
    to either expression changes what these tests see), and the lookup step's output (absent
    when the step did not run)."""
    context: dict[str, Any] = {
        "github": {
            "event_name": event,
            "ref": "refs/pull/7/merge" if event == "pull_request" else "refs/heads/main",
            "repository": "MakazhanAlpamys/Soup",
            "sha": "b" * 40,
        },
        "matrix": cell,
        "needs": {"plan": {"outputs": {"docs_only": docs_only, "full": "true"}}},
        "steps": {
            "warm": {"outputs": {"warmed": "all"}},
            "hf-cache": {"outputs": {"cache-hit": "false"}},
        },
    }
    env = _jobs()["test"]["env"]
    context["env"] = {
        name: _render(str(env[name]), context) for name in ("RUN_SUITE", "REUSE_CELL")
    }
    if hit is not None:
        context["steps"]["reuse"] = {"outputs": {"cache-hit": hit}}
    return context


def _runs(step: dict[str, Any], context: dict[str, Any]) -> bool:
    """A step condition, evaluated. ``success()`` is true in a run that has not failed."""
    condition = str(step.get("if", "true"))
    plain = re.sub(r"\b(?:always|success)\(\)\s*&&\s*", "", condition)
    assert "success()" not in plain and "always()" not in plain, condition
    return _truthy(_Expression(plain, context).value())


# --- The shape ---------------------------------------------------------------------------


class TestTheLookupAndTheMarker:
    def test_the_lookup_is_the_first_step_and_a_cache_restore(self) -> None:
        first = _steps()[0]
        assert first["name"] == LOOKUP
        assert first["id"] == "reuse"
        assert first["uses"] == "actions/cache/restore@v4"

    def test_the_save_is_the_last_step_and_a_cache_save(self) -> None:
        last = _steps()[-1]
        assert last["name"] == SAVE
        assert last["uses"] == "actions/cache/save@v4"

    def test_the_marker_is_written_just_before_it_is_saved(self) -> None:
        names = _names()
        assert names.index(MARKER) + 1 == names.index(SAVE)
        assert all(
            names.index(MARKER) > names.index(test)
            for test in ("Run unit tests with coverage", "Run unit tests without coverage")
        )

    def test_lookup_and_save_use_one_key_and_one_path(self) -> None:
        lookup, save = _step(LOOKUP)["with"], _step(SAVE)["with"]
        assert lookup["key"] == save["key"]
        assert lookup["path"] == save["path"]

    def test_the_key_names_the_merge_commit_and_the_cell(self) -> None:
        """``github.sha`` of a pull request is the merge commit: another push, another base and
        another pull request get another key. The cell is in it so no other cell can match."""
        key = str(_step(LOOKUP)["with"]["key"])
        assert "${{ github.sha }}" in key
        assert "${{ matrix.os }}" in key
        assert "${{ matrix.python-version }}" in key

    def test_the_key_has_no_restore_keys_and_no_prefix_match(self) -> None:
        """A prefix match would hand a result of another commit to this one."""
        for name in (LOOKUP, SAVE):
            assert "restore-keys" not in _step(name)["with"], name

    def test_the_marker_names_the_run_it_came_from(self) -> None:
        script = str(_step(MARKER)["run"])
        assert "GITHUB_RUN_ID" in script and "GITHUB_RUN_ATTEMPT" in script
        assert "suite-green" in script
        assert str(_step(SAVE)["with"]["path"]).endswith("/suite-green")

    def test_the_notice_prints_that_marker(self) -> None:
        script = str(_step(NOTICE)["run"])
        assert "::notice" in script and "suite-green" in script

    def test_the_marker_step_runs_in_bash_on_every_runner(self) -> None:
        assert _step(MARKER)["shell"] == "bash"
        assert _step(NOTICE)["shell"] == "bash"


class TestWhoLooksAndWhoRecords:
    @pytest.mark.parametrize("matrix", ALL_MATRICES)
    @pytest.mark.parametrize("event", ("pull_request", "push", "schedule", "workflow_dispatch"))
    def test_only_a_pull_request_cell_of_ubuntu_3_12_looks(self, matrix: str, event: str) -> None:
        for cell in _cell_dicts(_matrix(matrix)):
            context = _contexts(cell, event=event)
            expected = event == "pull_request" and cell == REUSE_CELL
            assert _runs(_step(LOOKUP), context) is expected, (matrix, event, cell)

    @pytest.mark.parametrize("matrix", ALL_MATRICES)
    @pytest.mark.parametrize("event", ("pull_request", "push", "schedule", "workflow_dispatch"))
    def test_only_that_cell_records_and_only_after_a_green_suite(
        self, matrix: str, event: str
    ) -> None:
        for cell in _cell_dicts(_matrix(matrix)):
            context = _contexts(cell, event=event, hit="false")
            expected = event == "pull_request" and cell == REUSE_CELL
            for name in (MARKER, SAVE):
                assert str(_step(name)["if"]).startswith("success() && "), name
                assert _runs(_step(name), context) is expected, (name, matrix, event, cell)

    def test_a_hit_is_never_recorded_again(self) -> None:
        context = _contexts(REUSE_CELL, event="pull_request", hit="true")
        for name in (MARKER, SAVE):
            assert _runs(_step(name), context) is False, name

    def test_the_lookup_follows_the_docs_only_rule_of_the_cell(self) -> None:
        """The kept cell of a docs-only run is this very cell: it may look and record."""
        context = _contexts(REUSE_CELL, event="pull_request", docs_only="true")
        assert _runs(_step(LOOKUP), context) is True

    def test_the_cell_flag_is_one_expression_in_the_job_env(self) -> None:
        expression = str(_jobs()["test"]["env"]["REUSE_CELL"])
        assert expression == (
            "${{ github.event_name == 'pull_request' && matrix.os == 'ubuntu-latest' && "
            "matrix.python-version == '3.12' }}"
        )


class TestAHitSkipsTheWholeCell:
    @pytest.mark.parametrize("event", ("pull_request",))
    def test_on_a_hit_only_the_lookup_and_the_notice_run(self, event: str) -> None:
        context = _contexts(REUSE_CELL, event=event, hit="true")
        running = [
            str(step.get("name") or step.get("uses"))
            for step in _steps()
            if _runs(step, context)
        ]
        assert running == [LOOKUP, NOTICE], running

    @pytest.mark.parametrize("hit", ("false", "", None))
    def test_on_a_miss_the_cell_does_what_it_did_before(self, hit: str | None) -> None:
        """No marker, an empty output and a lookup that never ran all mean: run the suite."""
        context = _contexts(REUSE_CELL, event="pull_request", hit=hit)
        running = [
            str(step.get("name") or step.get("uses"))
            for step in _steps()
            if _runs(step, context)
        ]
        assert NOTICE not in running
        assert "Run unit tests without coverage" in running
        assert "Install dependencies" in running
        assert "actions/checkout@v4" in running

    def test_every_other_cell_is_untouched_by_a_hit_somewhere_else(self) -> None:
        """A cell that did not look has no ``reuse`` output at all; the guard must read true
        for it, or its steps would vanish. Compared with the same conditions without it."""
        for matrix in ("QUICK_MATRIX", "APPROVAL_MATRIX"):
            for cell in _cell_dicts(_matrix(matrix)):
                if cell == REUSE_CELL:
                    continue
                context = _contexts(cell, event="pull_request")
                for step in _steps():
                    if step.get("name") in EXEMPT:
                        continue
                    without = {"if": str(step["if"]).replace(HIT_GUARD, "'a' == 'a'")}
                    assert _runs(step, context) is _runs(without, context), (
                        cell,
                        step.get("name") or step.get("uses"),
                    )

    @pytest.mark.parametrize("event", ("push", "schedule", "workflow_dispatch"))
    def test_a_run_that_is_not_a_pull_request_never_reuses(self, event: str) -> None:
        for cell in _cell_dicts(_matrix("FULL_MATRIX")):
            context = _contexts(cell, event=event)
            assert _runs(_step(LOOKUP), context) is False
            assert _runs(_step(NOTICE), context) is False


class TestEveryOtherStepIsGuarded:
    def test_every_step_carries_the_suite_guard(self) -> None:
        for step in _steps():
            assert SUITE_GUARD in str(step.get("if", "")), step.get("name") or step.get("uses")

    def test_every_step_that_is_not_the_lookup_or_the_notice_carries_the_hit_guard(self) -> None:
        for step in _steps():
            label = step.get("name") or step.get("uses")
            if label in EXEMPT:
                continue
            assert HIT_GUARD in str(step.get("if", "")), label

    def test_the_lookup_and_the_notice_do_not_carry_it(self) -> None:
        assert HIT_GUARD not in str(_step(LOOKUP).get("if", ""))
        assert "cache-hit == 'true'" in str(_step(NOTICE)["if"])

    def test_the_suite_steps_are_the_ones_that_were_there_before(self) -> None:
        """Reuse added three steps and removed none."""
        names = _names()
        for kept in (
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "actions/setup-node@v4",
            "Install dependencies",
            "Warm HF Hub cache (models the tests load)",
            "Run unit tests with coverage",
            "Run unit tests without coverage",
            "Upload coverage to Codecov",
            "Update test count badge",
        ):
            assert kept in names, kept

    @pytest.mark.parametrize("matrix", ("QUICK_MATRIX", "APPROVAL_MATRIX"))
    def test_a_miss_adds_the_lookup_in_front_and_the_two_recording_steps_at_the_end(
        self, matrix: str
    ) -> None:
        """With no marker the kept cell runs the steps it ran before, in the same order."""
        assert REUSE_CELL in _cell_dicts(_matrix(matrix))
        context = _contexts(REUSE_CELL, event="pull_request", hit="false")
        running = [
            str(step.get("name") or step.get("uses")) for step in _steps() if _runs(step, context)
        ]
        assert running[0] == LOOKUP
        assert running[1] == "actions/checkout@v4"
        assert running[-2:] == [MARKER, SAVE]
        before = [name for name in running if name not in (LOOKUP, MARKER, SAVE)]
        assert before == [
            str(step.get("name") or step.get("uses"))
            for step in _steps()
            if step.get("name") not in (LOOKUP, NOTICE, MARKER, SAVE)
            and _runs(step, context)
        ]


# --- What a contributor is told ---------------------------------------------------------------


def _contributing() -> str:
    return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())


def test_contributing_says_a_green_quick_cell_is_reused_at_approval() -> None:
    text = _contributing()
    assert "A green `test (ubuntu-latest, 3.12)` is reused at approval." in text
    assert "same merge commit" in text
    assert "`github.sha`" in text
    assert "still reports" in text


def test_truthy_helper_is_the_interpreters() -> None:
    """CONTROL: a step whose condition is false must read false, or the tests above prove
    nothing."""
    assert _runs({"if": "'a' == 'b'"}, {}) is False
    assert _runs({"if": "'a' == 'a'"}, {}) is True
    assert _runs({"if": "success() && 'a' == 'a'"}, {}) is True
