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

import copy
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.test_ci_full_matrix_on_approval import (
    ALL_MATRICES,
    BASH,
    CONTRIBUTING,
    NEEDS_BASH,
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
SUITE = "Run unit tests without coverage"
COVERAGE_SUITE = "Run unit tests with coverage"
#: The steps that are machinery of the reuse and never a reason to fail a required job.
OPTIONAL = (LOOKUP, NOTICE, MARKER, SAVE)
#: Steps that only print a fact about the runner; optional like the machinery, never a gate.
DIAGNOSTICS = ("Print the host CPU",)
HIT_GUARD = "steps.reuse.outputs.cache-hit != 'true'"
SUITE_PASSED = "steps.suite.outcome == 'success'"
MARKER_WRITTEN = "steps.marker.outcome == 'success'"
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
    cell: dict[str, str],
    *,
    event: str,
    hit: str | None = None,
    docs_only: str = "false",
    suite: str = "success",
    marker: str = "success",
) -> dict[str, Any]:
    """What a step's ``if:`` sees: the matrix cell, the plan outputs, ``RUN_SUITE`` and
    ``REUSE_CELL`` as the job's own ``env:`` renders them (never recomputed here, so a change
    to either expression changes what these tests see), the outcome of the suite step and of
    the marker step (``success`` unless asked), and the lookup step's output (absent when the
    step did not run). A STATIC view: which steps run one after the other, with ``success()``
    and skipped steps, is ``_simulate`` below."""
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
            "suite": {"outcome": suite},
            "marker": {"outcome": marker},
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


_STATUS_FUNCTION = re.compile(r"\b(success|always|failure|cancelled)\(\)")
#: What the steps with an id other than the three below write to `steps.<id>.outputs`.
_OTHER_OUTPUTS = {"warm": {"warmed": "all"}, "hf-cache": {"cache-hit": "false"}}


@dataclass
class Simulation:
    """The outcome of every step of one cell (``success`` / ``failure`` / ``skipped``, by name)
    and whether the job went red."""

    ran: dict[str, str]
    failed: bool


def _run_steps(
    steps: list[dict[str, Any]],
    context: dict[str, Any],
    behave: Callable[[dict[str, Any]], tuple[str, dict[str, str]]],
) -> Simulation:
    """Run *steps* in order the way the runner does. A step without a status function carries
    an implicit ``success()``; an explicit ``success()`` is true while no earlier step FAILED,
    which includes the case where every earlier step was SKIPPED (the coincidence this file
    must not rely on); ``always()`` is true. *behave* gives a step that runs its outcome and
    outputs; a failed step turns the job red unless it has ``continue-on-error``, and then the
    step reads ``failure`` as its ``outcome`` and still lets later steps run."""
    context = {**context, "steps": {}}
    failed = False
    ran: dict[str, str] = {}
    for step in steps:
        label = str(step.get("name") or step.get("uses"))
        condition = str(step.get("if", "true"))
        used = set(_STATUS_FUNCTION.findall(condition))
        assert used <= {"success", "always"}, condition
        if used:
            plain = condition.replace("success()", "false" if failed else "true")
            plain = plain.replace("always()", "true")
            runs = _truthy(_Expression(plain, context).value())
        else:
            runs = not failed and _truthy(_Expression(condition, context).value())
        outcome, outputs = "skipped", {}
        if runs:
            outcome, outputs = behave(step)
            if outcome == "failure" and not step.get("continue-on-error"):
                failed = True
        if step.get("id"):
            context["steps"][step["id"]] = {"outcome": outcome, "outputs": outputs}
        ran[label] = outcome
    return Simulation(ran, failed)


def _simulate(
    cell: dict[str, str],
    *,
    event: str,
    docs_only: str = "false",
    lookup: str = "miss",
    suite: str = "success",
    marker: str = "success",
    steps: list[dict[str, Any]] | None = None,
) -> Simulation:
    """One cell of one run. *lookup* is ``miss``, ``hit`` or ``error`` (the restore step
    fails and writes no output); *suite* and *marker* are the outcomes of the steps with those
    ids when they run; every other step succeeds."""

    def behave(step: dict[str, Any]) -> tuple[str, dict[str, str]]:
        identity = step.get("id")
        if identity == "reuse":
            if lookup == "error":
                return "failure", {}
            return "success", {"cache-hit": "true" if lookup == "hit" else "false"}
        if identity == "suite":
            return suite, {}
        if identity == "marker":
            return marker, {}
        return "success", dict(_OTHER_OUTPUTS.get(str(identity), {}))

    context = _contexts(cell, event=event, docs_only=docs_only)
    return _run_steps(_steps() if steps is None else steps, context, behave)


#: Every cell any run can have, once.
def _unique_cells() -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for matrix in ALL_MATRICES:
        for cell in _cell_dicts(_matrix(matrix)):
            if cell not in cells:
                cells.append(cell)
    return cells


CELLS = _unique_cells()
EVENTS_GRID = ("pull_request", "push", "schedule", "workflow_dispatch")


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
                red = _contexts(cell, event=event, hit="false", suite="failure", marker="failure")
                assert _runs(_step(name), red) is False, (name, matrix, event, cell)

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


class TestTheMarkerIsWrittenOnlyAfterAGreenSuite:
    """``success()`` is true when every earlier step was SKIPPED, so a record step guarded by it
    and by a predicate copied from the suite step is right only while the two predicates agree.
    It is guarded by the suite step's own outcome."""

    def test_the_suite_step_has_an_id_and_both_guards_read_its_outcome(self) -> None:
        assert _step(SUITE)["id"] == "suite"
        assert _step(MARKER)["id"] == "marker"
        for name in (MARKER, SAVE):
            assert SUITE_PASSED in str(_step(name)["if"]), name
        assert MARKER_WRITTEN in str(_step(SAVE)["if"])
        assert MARKER_WRITTEN not in str(_step(MARKER)["if"])

    def test_the_coverage_step_cannot_be_the_suite_of_the_reuse_cell(self) -> None:
        """The cell that reuses is a pull request's ubuntu / 3.12; the measuring step is
        ubuntu / 3.11 of a run that is not a pull request: never the same cell."""
        for event in EVENTS_GRID:
            for cell in CELLS:
                ran = _simulate(cell, event=event).ran
                if event == "pull_request" and cell == REUSE_CELL:
                    assert ran[COVERAGE_SUITE] == "skipped"
                    assert ran[SUITE] == "success"

    def test_success_is_true_after_skipped_steps_in_the_model(self) -> None:
        """CONTROL: the simulator treats `success()` the way GitHub does, so the grid below
        can catch a guard that leans on it."""
        steps = [
            {"name": "a", "if": "'a' == 'b'"},
            {"name": "b", "if": "'a' == 'b'"},
            {"name": "c", "if": "success() && 'a' == 'a'"},
            {"name": "d", "if": "'a' == 'a'"},
            {"name": "e", "if": "always()"},
        ]
        result = _run_steps(steps, {}, lambda step: ("success", {}))
        assert result.ran == {
            "a": "skipped", "b": "skipped", "c": "success", "d": "success", "e": "success"
        }
        steps.insert(2, {"name": "boom", "if": "true"})

        def boom(step: dict[str, Any]) -> tuple[str, dict[str, str]]:
            return ("failure", {}) if step["name"] == "boom" else ("success", {})

        red = _run_steps(steps, {}, boom)
        assert red.failed
        assert red.ran["c"] == "skipped" and red.ran["d"] == "skipped" and red.ran["e"] == "success"

    @pytest.mark.parametrize("event", EVENTS_GRID)
    @pytest.mark.parametrize("docs_only", ("false", "true"))
    @pytest.mark.parametrize("lookup", ("miss", "hit", "error"))
    @pytest.mark.parametrize("suite", ("success", "failure"))
    def test_the_save_runs_only_if_this_cells_suite_ran_and_passed(
        self, event: str, docs_only: str, lookup: str, suite: str
    ) -> None:
        if docs_only == "true" and event != "pull_request":
            return  # only a pull request can be docs-only
        for cell in CELLS:
            ran = _simulate(cell, event=event, docs_only=docs_only, lookup=lookup, suite=suite).ran
            if ran[SAVE] != "skipped":
                where = (event, docs_only, lookup, suite, cell)
                assert event == "pull_request" and cell == REUSE_CELL, where
                assert ran[SUITE] == "success", where
                assert ran[MARKER] == "success", where
                assert lookup != "hit", where
            if ran[MARKER] != "skipped":
                assert ran[SUITE] == "success", (event, docs_only, lookup, suite, cell)

    @pytest.mark.parametrize("docs_only", ("false", "true"))
    @pytest.mark.parametrize("lookup", ("miss", "error"))
    def test_a_green_suite_of_the_reuse_cell_is_recorded_and_saved(
        self, docs_only: str, lookup: str
    ) -> None:
        """Not vacuous: where the rule says the marker is written, it is."""
        ran = _simulate(REUSE_CELL, event="pull_request", docs_only=docs_only, lookup=lookup).ran
        assert ran[SUITE] == "success"
        assert ran[MARKER] == "success"
        assert ran[SAVE] == "success"

    @pytest.mark.parametrize("lookup", ("miss", "error"))
    def test_a_red_suite_is_never_recorded(self, lookup: str) -> None:
        sim = _simulate(REUSE_CELL, event="pull_request", lookup=lookup, suite="failure")
        assert sim.failed
        assert sim.ran[SUITE] == "failure"
        assert sim.ran[MARKER] == "skipped" and sim.ran[SAVE] == "skipped"

    def test_a_hit_runs_nothing_and_records_nothing(self) -> None:
        sim = _simulate(REUSE_CELL, event="pull_request", lookup="hit")
        assert not sim.failed
        assert {name for name, outcome in sim.ran.items() if outcome != "skipped"} == {
            LOOKUP, NOTICE
        }

    @pytest.mark.parametrize("drift", ("false", "env.RUN_SUITE == 'false'"))
    def test_a_suite_condition_that_drifts_away_does_not_drag_the_marker_along(
        self, drift: str
    ) -> None:
        """The record guards must not be a copy of the suite step's predicate: with the suite
        step skipped (here by a changed condition) every step before the record steps is skipped
        or green, `success()` is true, and only the suite step's own outcome is left to say no."""
        steps = copy.deepcopy(_steps())
        for step in steps:
            if step.get("id") == "suite":
                step["if"] = drift
        sim = _simulate(REUSE_CELL, event="pull_request", steps=steps)
        assert sim.ran[SUITE] == "skipped"
        assert sim.ran[MARKER] == "skipped" and sim.ran[SAVE] == "skipped"

    def test_a_marker_that_could_not_be_written_is_not_saved(self) -> None:
        sim = _simulate(REUSE_CELL, event="pull_request", marker="failure")
        assert sim.ran[MARKER] == "failure"
        assert sim.ran[SAVE] == "skipped"
        assert not sim.failed, "a record step that fails must not turn a green suite red"


class TestTheSuiteIsNotMaskedAndTheMachineryIsNotFatal:
    @pytest.mark.parametrize("name", (SUITE, COVERAGE_SUITE))
    def test_a_suite_step_carries_no_continue_on_error_and_no_fallback(self, name: str) -> None:
        """A failing suite that exits 0 would be recorded as green and reused."""
        step = _step(name)
        assert "continue-on-error" not in step, name
        script = str(step["run"])
        assert not re.search(r"\|\|\s*(true|:|exit\s+0)\b", script), script
        assert "set +e" not in script and "|| echo" not in script, script

    def test_only_the_machinery_and_the_diagnostic_steps_may_continue_on_error(self) -> None:
        """The suite steps never carry it. The four machinery steps do, and so does the one
        diagnostic (`tests/test_ci_windows_host_cpu_print.py`): it prints the host CPU of a
        Windows cell and changes nothing. A step added to this list needs a reason written
        beside it."""
        carrying = [
            str(step.get("name") or step.get("uses"))
            for step in _steps()
            if "continue-on-error" in step
        ]
        assert sorted(carrying) == sorted([*OPTIONAL, *DIAGNOSTICS])

    @pytest.mark.parametrize("name", OPTIONAL)
    def test_each_machinery_step_is_optional_and_bounded(self, name: str) -> None:
        """The lookup is the first step of a required job: an outage of the cache service must
        read as a miss, not as a red `test (ubuntu-latest, 3.12)`, and a hung restore must not
        hold a runner for the job's 105 minutes."""
        step = _step(name)
        assert step["continue-on-error"] is True, name
        assert step["timeout-minutes"] == 5, name

    @pytest.mark.parametrize("event", EVENTS_GRID)
    def test_a_failed_lookup_leaves_every_later_step_running_as_after_a_miss(
        self, event: str
    ) -> None:
        for cell in CELLS:
            error = _simulate(cell, event=event, lookup="error")
            miss = _simulate(cell, event=event, lookup="miss")
            assert not error.failed, (event, cell)
            assert {k: v for k, v in error.ran.items() if k != LOOKUP} == {
                k: v for k, v in miss.ran.items() if k != LOOKUP
            }, (event, cell)
            if event == "pull_request" and cell == REUSE_CELL:
                assert error.ran[LOOKUP] == "failure"
                assert error.ran[SUITE] == "success"

    def test_the_lookup_would_turn_the_job_red_without_continue_on_error(self) -> None:
        """CONTROL: the model does fail a job on a failed step without the flag."""
        steps = copy.deepcopy(_steps())
        for step in steps:
            if step.get("id") == "reuse":
                del step["continue-on-error"]
        sim = _simulate(REUSE_CELL, event="pull_request", lookup="error", steps=steps)
        assert sim.failed
        assert sim.ran[SUITE] == "skipped"


@NEEDS_BASH
class TestTheNoticeEchoesOnlyWhatItCanValidate:
    """The marker file comes out of the Actions cache, which a pull request can write: its text
    goes into a workflow-command line, so only `run <digits> attempt <digits>` is printed."""

    FIXED = "this exact merge commit passed the suite earlier; nothing is run again"
    # No case with a trailing carriage return: Git for Windows' bash drops it from `$(cat ...)`
    # (a marker of ours is written with LF on Linux, where such a file would be refused).

    def _notice(self, tmp_path: Path, content: bytes | None) -> list[str]:
        marker_dir = tmp_path / "suite-green"
        marker_dir.mkdir()
        if content is not None:
            (marker_dir / "result").write_bytes(content)
        script = tmp_path / "notice.sh"
        script.write_bytes(str(_step(NOTICE)["run"]).encode("utf-8"))
        assert BASH is not None
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
            env={**os.environ, "RUNNER_TEMP": tmp_path.as_posix()},
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        return result.stdout.splitlines()

    def test_a_marker_of_this_workflow_is_named_in_the_notice(self, tmp_path: Path) -> None:
        lines = self._notice(tmp_path, b"run 4242 attempt 2\n")
        assert len(lines) == 1
        assert lines[0].startswith("::notice title=Suite result reused::")
        assert "passed the suite in run 4242 attempt 2;" in lines[0]

    @pytest.mark.parametrize(
        "content",
        [
            None,
            b"",
            b"::error::boom",
            b"run 1 attempt 1\n::error::boom\n",
            b"run 1 attempt 1 extra",
            b"run abc attempt 1",
            b"run 1 attempt",
            b"::set-output name=x::y",
            b"run 9 attempt 9;touch x",
        ],
        ids=[
            "missing", "empty", "command", "second-line", "trailing-words", "letters",
            "no-number", "set-output", "semicolon",
        ],
    )
    def test_anything_else_prints_a_fixed_sentence(
        self, tmp_path: Path, content: bytes | None
    ) -> None:
        lines = self._notice(tmp_path, content)
        assert lines == [f"::notice title=Suite result reused::{self.FIXED}"], lines

    def test_the_script_takes_no_expression_from_the_workflow(self) -> None:
        assert "${{" not in str(_step(NOTICE)["run"])


# --- What a contributor is told ---------------------------------------------------------------


def _contributing() -> str:
    return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())


def test_contributing_says_a_green_quick_cell_is_reused_at_approval() -> None:
    text = _contributing()
    assert "A green `test (ubuntu-latest, 3.12)` is reused at approval." in text
    assert "same merge commit" in text
    assert "`github.sha`" in text
    assert "still reports" in text


def test_contributing_says_what_is_true_about_a_moved_main() -> None:
    """A `labeled` run does not rebuild the merge commit, so a moved `main` alone is NOT a miss:
    the label run reuses the result of the commit it would have tested itself."""
    text = _contributing()
    assert "A moved `main` (a rebuilt merge commit)" not in text
    assert "adding `ci:full` does not rebuild it" in text
    assert "A moved `main` alone is not a miss" in text
    assert "exactly what the quick run tested" in text


def test_contributing_names_the_real_key_and_how_to_force_a_fresh_suite() -> None:
    text = _contributing()
    key = str(_step(LOOKUP)["with"]["key"])
    assert key == "suite-green-v1-${{ github.sha }}-${{ matrix.os }}-${{ matrix.python-version }}"
    shown = "suite-green-v1-<merge sha>-ubuntu-latest-3.12"
    assert shown in text
    assert f"`gh cache delete {shown} --ref refs/pull/<number>/merge`" in text
    assert "`gh cache list --key suite-green-v1- --ref refs/pull/<number>/merge`" in text
    assert "then re-run the job" in text


def test_contributing_says_a_dispatch_on_a_pull_request_branch_is_not_the_label_run() -> None:
    text = _contributing()
    assert "tests the branch tip, not its merge with `main`" in text
    assert "does not replace the label run" in text


def test_truthy_helper_is_the_interpreters() -> None:
    """CONTROL: a step whose condition is false must read false, or the tests above prove
    nothing."""
    assert _runs({"if": "'a' == 'b'"}, {}) is False
    assert _runs({"if": "'a' == 'a'"}, {}) is True
    assert _runs({"if": "success() && 'a' == 'a'"}, {}) is True
