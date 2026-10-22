"""A docs-only pull request runs the suite once, and a hung test is capped.

Measured 2026-10-12..19 (re-derived 2026-10-20): about 136 job-minutes a day went to the
Windows, macOS and ubuntu / 3.10 test cells and the three smoke jobs of pull requests that
changed nothing but ``docs/**``, ``changelog.d/**``, ``CONTRIBUTORS.md`` and
``CONTRIBUTING.md`` (about 185 before the warm-cache fix of 2026-10-19). So ``plan`` computes
``docs_only`` from the pull request's own file list, and when it is true:

* ``test (ubuntu-latest, 3.12)`` runs exactly what it ran before: the full suite;
* every step of the other test cells and of the three smoke jobs is skipped by its own
  ``if:``, so those jobs still REPORT -- as success, not as "skipped" -- and every one of the
  eight required contexts is still created on every pull request.

The merge gate is why it is done with steps and not with a ``paths`` filter or a job-level
``if:``: a workflow filtered out by ``paths`` reports nothing, and a required context nothing
reported stays "Expected" for ever (see tests/test_ci_full_matrix_on_approval.py).

The second half is the hang cap. In the nightly run of 2026-10-20 ``test (windows-latest,
3.10)`` stayed in one test until the job's 90-minute limit, and a macOS cell of a pull request
held its runner for 46 minutes on 2026-10-13. The test steps now carry ``--timeout=600
--timeout-method=thread --max-worker-restart=2`` (pytest-timeout, in the dev extra) and
``timeout-minutes: 45``. The caps sit far above a healthy run: the slowest single test of
four recent full cells took under 51 s and the longest test step of the last 7 days 27.7
minutes. Under ``--dist loadfile`` xdist hands a crashed test to the replacement worker again
(measured with xdist 3.8.0), which is why the restarts are limited.

Nothing here talks to GitHub: the workflow is read as YAML, the ``docs`` step's own shell runs
under a local bash with ``gh`` replaced by a function, and the expressions are evaluated by the
interpreter of tests/test_ci_full_matrix_on_approval.py.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    _STATUS_FUNCTION,
    ALL_MATRICES,
    BASH,
    CONTRIBUTING,
    GATE,
    NEEDS_BASH,
    REQUIRED_TEST_CONTEXTS,
    ROOT,
    _cell_dicts,
    _cells,
    _Expression,
    _jobs,
    _matrix,
    _plan,
    _render,
    _triggers,
    _truthy,
)

SMOKE_JOBS = ("mlx-smoke", "pytorch-smoke", "transformers-floor")
#: The jobs that read `docs_only`, and no other job may.
SKIPPING_JOBS = {"plan", "test", *SMOKE_JOBS}
SMOKE_GUARD = "needs.plan.outputs.docs_only != 'true'"
SUITE_GUARD = "env.RUN_SUITE == 'true'"
#: The one cell that keeps its work in a docs-only run.
KEPT_CELL = {"os": "ubuntu-latest", "python-version": "3.12"}
#: The filter the `docs` step hands `gh api --jq`: one row per file, `filename<TAB>previous`
#: (null prints as nothing). The fake `gh` below reproduces its output; a change to the filter
#: must change this line and the rows `_rows` writes.
JQ_FILTER = ".[] | [.filename, .previous_filename] | @tsv"
RUN_STEPS = ("Run unit tests with coverage", "Run unit tests without coverage")
PYPROJECT = ROOT / "pyproject.toml"

#: Measured 2026-10-20 (see the module docstring); the caps must not be able to fire on them.
SLOWEST_TEST_SECONDS = 51
LONGEST_TEST_STEP_MINUTES = 27.7
SLOWEST_TEST_JOB_MINUTES = 37.5


def _steps(job: str) -> list[dict[str, Any]]:
    return list(_jobs()[job]["steps"])


def _label(step: dict[str, Any]) -> str:
    return str(step.get("name") or step.get("uses") or step.get("run"))


def _docs_step() -> dict[str, Any]:
    found = [step for step in _plan()["steps"] if step.get("id") == "docs"]
    assert len(found) == 1, "the plan job must have exactly one step with `id: docs`"
    return found[0]


def _option(args: list[str], flag: str) -> list[str]:
    """Every value given to ``flag``, as ``flag value``."""
    return [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == flag]


def _per_test_cap(step: dict[str, Any]) -> int:
    (cap,) = [
        int(arg.partition("=")[2])
        for arg in shlex.split(str(step["run"]))
        if arg.startswith("--timeout=")
    ]
    return cap


def _holds(condition: str, context: dict[str, Any]) -> bool:
    """A step condition, evaluated. `always()` / `success()` are true in a run that has not
    failed, which is the case every test here asks about."""
    plain = re.sub(r"\b(?:always|success)\(\)\s*&&\s*", "", condition)
    assert not _STATUS_FUNCTION.search(plain), plain
    return _truthy(_Expression(plain, context).value())


def _plan_outputs(docs_only: str) -> dict[str, Any]:
    return {"needs": {"plan": {"outputs": {"docs_only": docs_only, "full": "true"}}}}


def _suite_runs(cell: dict[str, str], docs_only: str) -> str:
    """`RUN_SUITE` for *cell*, rendered the way the runner renders the job's `env:`."""
    context = {**_plan_outputs(docs_only), "matrix": cell}
    return _render(str(_jobs()["test"]["env"]["RUN_SUITE"]), context)


def _steps_that_run(
    cell: dict[str, str],
    docs_only: str,
    *,
    event_name: str = "pull_request",
    ref: str = "refs/pull/7/merge",
) -> list[str]:
    """The steps of the `test` job that run in *cell* of a run: a pull-request run unless the
    event is given (a pull request is the only run that can be docs-only)."""
    context = {
        **_plan_outputs(docs_only),
        "matrix": cell,
        "env": {"RUN_SUITE": _suite_runs(cell, docs_only)},
        "github": {
            "event_name": event_name,
            "ref": ref,
            "repository": "MakazhanAlpamys/Soup",
        },
        "steps": {
            "warm": {"outputs": {"warmed": "all"}},
            "hf-cache": {"outputs": {"cache-hit": "false"}},
        },
    }
    return [
        _label(step)
        for step in _steps("test")
        if _holds(str(step.get("if", "true")), context)
    ]


# --- What `plan` computes ------------------------------------------------------


class TestThePlanJobGrantsOnlyWhatTheFileListNeeds:
    def test_plan_may_read_pull_requests_and_nothing_else(self) -> None:
        """Naming one permission makes every other one `none`: no `contents`, no `actions`,
        and in particular no write."""
        assert _plan()["permissions"] == {"pull-requests": "read"}

    def test_no_other_job_widened_its_permissions(self) -> None:
        for name, job in _jobs().items():
            if name != "plan":
                assert "permissions" not in job, name

    def test_the_decision_is_exported_as_an_output(self) -> None:
        assert _plan()["outputs"]["docs_only"] == "${{ steps.docs.outputs.docs_only }}"

    def test_the_step_is_given_its_inputs_through_env_and_never_through_the_script(self) -> None:
        env = _docs_step()["env"]
        assert env == {
            "GH_TOKEN": "${{ github.token }}",
            "EVENT_NAME": "${{ github.event_name }}",
            "REPOSITORY": "${{ github.repository }}",
            "PR_NUMBER": "${{ github.event.pull_request.number }}",
            "CHANGED_FILES": "${{ github.event.pull_request.changed_files }}",
        }

    def test_no_script_of_the_plan_job_interpolates_an_expression(self) -> None:
        """An expression written into `run:` is code that a pull request's title, branch or
        label could shape. Values travel in `env:`."""
        for step in _plan()["steps"]:
            assert "${{" not in str(step.get("run", "")), _label(step)

    def test_the_only_source_is_the_pull_requests_own_file_list(self) -> None:
        """Not `git diff` against a base that may have moved, not the head commit's files."""
        script = str(_docs_step()["run"])
        assert 'gh api "repos/${REPOSITORY}/pulls/${PR_NUMBER}/files" --paginate' in script
        assert f"--jq '{JQ_FILTER}'" in script
        for other in ("git ", "head_commit", "github.event.commits", "compare/"):
            assert other not in script, other

    def test_the_pull_request_trigger_is_still_unfiltered(self) -> None:
        triggers = _triggers()
        assert "pull_request_target" not in triggers
        assert "paths" not in triggers["pull_request"], "a filtered workflow reports nothing"
        assert "paths-ignore" not in triggers["pull_request"]


class _Fake:
    """The result of running the `docs` step's script against a canned file list."""

    def __init__(self, output: list[str], gh_arguments: list[str] | None, log: str) -> None:
        self.output = output
        self.gh_arguments = gh_arguments
        self.log = log  # what the step printed: the runner's workflow commands

    @property
    def docs_only(self) -> str:
        assert len(self.output) == 1 and self.output[0].startswith("docs_only="), self.output
        return self.output[0].partition("=")[2]


def _rows(files: list[tuple[str, str]]) -> bytes:
    """What `gh api --jq JQ_FILTER` prints for *files*, `(filename, previous_filename)`."""
    return "".join(f"{name}\t{previous}\n" for name, previous in files).encode("utf-8")


def _run_docs_step(
    tmp_path: Path,
    *,
    files: list[tuple[str, str]],
    changed: str | None = None,
    event: str = "pull_request",
    number: str = "7",
    gh_status: int = 0,
) -> _Fake:
    assert BASH is not None
    rows = tmp_path / "rows.tsv"
    rows.write_bytes(_rows(files))
    log = tmp_path / "gh_arguments"
    output = tmp_path / "github_output"
    output.write_bytes(b"")
    stub = tmp_path / "gh_stub.sh"
    stub.write_bytes(
        b"gh() {\n"
        b'  printf "%s\\n" "$@" > "$FAKE_GH_LOG"\n'
        b'  if [ "$FAKE_GH_STATUS" != "0" ]; then\n'
        b'    echo "gh: HTTP 502" >&2\n'
        b'    return "$FAKE_GH_STATUS"\n'
        b"  fi\n"
        b'  cat "$FAKE_GH_ROWS"\n'
        b"}\n"
    )
    script = tmp_path / "docs.sh"
    script.write_bytes(str(_docs_step()["run"]).encode("utf-8"))
    env = {
        **os.environ,
        "BASH_ENV": stub.as_posix(),
        "GH_TOKEN": "not-a-token",
        "EVENT_NAME": event,
        "REPOSITORY": "MakazhanAlpamys/Soup",
        # Set even for an event that has no pull request: only the event may start the call.
        "PR_NUMBER": number,
        "CHANGED_FILES": str(len(files)) if changed is None else changed,
        "GITHUB_OUTPUT": output.as_posix(),
        "FAKE_GH_ROWS": rows.as_posix(),
        "FAKE_GH_LOG": log.as_posix(),
        "FAKE_GH_STATUS": str(gh_status),
    }
    # The runner's default shell for a step without `shell:` is `bash -e {0}`.
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-e", script.as_posix()],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    arguments = log.read_text(encoding="utf-8").splitlines() if log.exists() else None
    return _Fake(output.read_text(encoding="utf-8").splitlines(), arguments, result.stdout)


def _same(*names: str) -> list[tuple[str, str]]:
    return [(name, "") for name in names]


DOCS_FILES = [
    "docs/evaluation.md",
    "docs/sub/dir/page.md",
    "changelog.d/0.76.1/2040.fixed.md",
    "CONTRIBUTORS.md",
    "CONTRIBUTING.md",
]


@NEEDS_BASH
class TestDocsOnlyIsTrueOnlyForADocsOnlyPullRequest:
    """Run the step's own shell, as the runner would, against a fake `gh` and `$GITHUB_OUTPUT`."""

    def test_every_docs_path_together_is_docs_only(self, tmp_path: Path) -> None:
        assert _run_docs_step(tmp_path, files=_same(*DOCS_FILES)).docs_only == "true"

    @pytest.mark.parametrize("name", DOCS_FILES)
    def test_each_docs_path_alone_is_docs_only(self, tmp_path: Path, name: str) -> None:
        assert _run_docs_step(tmp_path, files=_same(name)).docs_only == "true", name

    def test_the_file_list_is_read_from_this_pull_request(self, tmp_path: Path) -> None:
        fake = _run_docs_step(tmp_path, files=_same("docs/a.md"), number="1234")
        assert fake.gh_arguments is not None
        path = "repos/MakazhanAlpamys/Soup/pulls/1234/files"
        assert fake.gh_arguments == ["api", path, "--paginate", "--jq", JQ_FILTER]

    @pytest.mark.parametrize(
        "name",
        [
            "src/soup_cli/cli.py",
            "tests/test_cli.py",
            ".github/workflows/ci.yml",
            "pyproject.toml",
            # README and benchmarks are tests' input: not docs for this purpose.
            "README.md",
            "README.tr.md",
            "benchmarks/harness/mlx_sft_smoke.py",
            "benchmarks/README.md",
            # Lookalikes of the four docs paths.
            "docsx/a.md",
            "mydocs/a.md",
            "src/docs/a.md",
            "docs.md",
            "docs",
            "changelog.d",
            "changelog.dx/1.fixed.md",
            "CHANGELOG.md",
            "CONTRIBUTORS.md.bak",
            "CONTRIBUTING.md/x",
            "contributing.md",
            "NOTICE",
        ],
    )
    def test_one_other_file_among_docs_files_is_not_docs_only(
        self, tmp_path: Path, name: str
    ) -> None:
        files = _same("docs/a.md", name, "CONTRIBUTORS.md")
        assert _run_docs_step(tmp_path, files=files).docs_only == "false", name

    @pytest.mark.parametrize("position", ("first", "last"))
    def test_the_position_of_the_odd_file_does_not_matter(
        self, tmp_path: Path, position: str
    ) -> None:
        files = _same("docs/a.md", "docs/b.md", "docs/c.md")
        files.insert(0 if position == "first" else len(files), ("src/x.py", ""))
        assert _run_docs_step(tmp_path, files=files).docs_only == "false"

    def test_a_rename_between_two_docs_paths_is_docs_only(self, tmp_path: Path) -> None:
        assert _run_docs_step(tmp_path, files=[("docs/new.md", "docs/old.md")]).docs_only == "true"

    def test_a_file_renamed_into_docs_still_deletes_a_source_file(self, tmp_path: Path) -> None:
        """`filename` is the new name only; the old one is `previous_filename`."""
        fake = _run_docs_step(tmp_path, files=[("docs/new.md", "src/soup_cli/old.py")])
        assert fake.docs_only == "false"

    def test_a_file_renamed_out_of_docs_is_not_docs_only(self, tmp_path: Path) -> None:
        fake = _run_docs_step(tmp_path, files=[("src/soup_cli/new.py", "docs/old.md")])
        assert fake.docs_only == "false"

    @pytest.mark.parametrize("changed", (None, "0", "", "1"))
    def test_an_empty_file_list_is_not_docs_only(
        self, tmp_path: Path, changed: str | None
    ) -> None:
        """Nothing was checked: not when the pull request says it changed no file, and not
        when it says one (an empty answer from the API reads as one blank row)."""
        assert _run_docs_step(tmp_path, files=[], changed=changed).docs_only == "false"

    @pytest.mark.parametrize("changed", ("3", "1", "", "many", "0"))
    def test_a_list_that_disagrees_with_the_pull_requests_file_count_is_not_trusted(
        self, tmp_path: Path, changed: str
    ) -> None:
        """The endpoint stops at 3000 files: a longer pull request would look shorter."""
        fake = _run_docs_step(tmp_path, files=_same("docs/a.md", "docs/b.md"), changed=changed)
        assert fake.docs_only == "false", changed

    def test_an_api_error_runs_the_full_set_and_says_so(self, tmp_path: Path) -> None:
        fake = _run_docs_step(tmp_path, files=_same("docs/a.md"), gh_status=1)
        assert fake.docs_only == "false"
        assert "::warning title=Docs-only check::" in fake.log, fake.log

    @pytest.mark.parametrize("event", ("push", "schedule", "workflow_dispatch"))
    def test_a_run_that_is_not_a_pull_request_is_never_docs_only(
        self, tmp_path: Path, event: str
    ) -> None:
        """Pushes, the nightly run, a manual dispatch and a release branch: no API call, no
        skipped cell."""
        fake = _run_docs_step(tmp_path, files=_same("docs/a.md"), event=event)
        assert fake.docs_only == "false"
        assert fake.gh_arguments is None, "gh was called outside a pull request"

    def test_a_pull_request_without_a_number_is_not_docs_only(self, tmp_path: Path) -> None:
        fake = _run_docs_step(tmp_path, files=_same("docs/a.md"), number="")
        assert fake.docs_only == "false"
        assert fake.gh_arguments is None

    def test_exactly_one_line_is_written_to_the_output(self, tmp_path: Path) -> None:
        for docs, files in (("true", _same("docs/a.md")), ("false", _same("src/a.py"))):
            fake = _run_docs_step(tmp_path, files=files)
            assert fake.output == [f"docs_only={docs}"]


# --- What the jobs do with it -----------------------------------------------------


class TestTheTestCellsSkipOnlyInADocsOnlyRun:
    @pytest.mark.parametrize("matrix", ALL_MATRICES)
    @pytest.mark.parametrize("docs_only", ("false", "", "no"))
    def test_every_cell_works_unless_the_run_is_docs_only(
        self, matrix: str, docs_only: str
    ) -> None:
        for cell in _cell_dicts(_matrix(matrix)):
            assert _suite_runs(cell, docs_only) == "true", (cell, docs_only)

    @pytest.mark.parametrize("matrix", ALL_MATRICES)
    def test_in_a_docs_only_run_only_ubuntu_3_12_keeps_its_work(self, matrix: str) -> None:
        for cell in _cell_dicts(_matrix(matrix)):
            expected = "true" if cell == KEPT_CELL else "false"
            assert _suite_runs(cell, "true") == expected, cell

    def test_the_kept_cell_is_in_every_run_that_has_a_docs_only_pull_request(self) -> None:
        """A pull request runs QUICK or APPROVAL; both have it, so the suite always runs once."""
        for matrix in ("QUICK_MATRIX", "APPROVAL_MATRIX"):
            assert KEPT_CELL in _cell_dicts(_matrix(matrix)), matrix

    @pytest.mark.parametrize("matrix", ("QUICK_MATRIX", "APPROVAL_MATRIX"))
    def test_ubuntu_3_12_runs_every_step_it_ran_before(self, matrix: str) -> None:
        """The same steps, in the same order, docs-only or not."""
        assert KEPT_CELL in _cell_dicts(_matrix(matrix))
        docs = _steps_that_run(KEPT_CELL, "true")
        assert docs == _steps_that_run(KEPT_CELL, "false")
        assert "Run unit tests without coverage" in docs
        assert "Install dependencies" in docs
        assert docs[0] == "actions/checkout@v4"

    @pytest.mark.parametrize("matrix", ("QUICK_MATRIX", "APPROVAL_MATRIX"))
    def test_every_other_required_cell_runs_no_step_of_a_docs_only_run(self, matrix: str) -> None:
        for cell in _cell_dicts(_matrix(matrix)):
            if cell != KEPT_CELL:
                assert _steps_that_run(cell, "true") == [], cell

    def test_the_approval_run_loses_the_work_of_exactly_three_cells(self) -> None:
        idle = [
            cell
            for cell in _cell_dicts(_matrix("APPROVAL_MATRIX"))
            if _suite_runs(cell, "true") == "false"
        ]
        assert sorted(f"{cell['os']} {cell['python-version']}" for cell in idle) == [
            "macos-latest 3.12",
            "ubuntu-latest 3.11",
            "windows-latest 3.12",
        ]

    def test_every_step_of_the_test_job_is_guarded(self) -> None:
        """A step added later without the guard would install and test on the cells that
        were meant to do nothing."""
        for step in _steps("test"):
            assert SUITE_GUARD in str(step.get("if", "")), _label(step)

    def test_the_guard_is_written_once_in_the_job_env(self) -> None:
        assert str(_jobs()["test"]["env"]["RUN_SUITE"]) == (
            "${{ needs.plan.outputs.docs_only != 'true' || "
            "(matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12') }}"
        )

    def test_the_coverage_cell_and_the_badge_step_never_run_in_a_docs_only_run(self) -> None:
        """No coverage report exists to upload when the cell that measures it did nothing."""
        coverage_cell = {"os": "ubuntu-latest", "python-version": "3.11"}
        assert _steps_that_run(coverage_cell, "true") == []

    def test_no_pull_request_measures_or_uploads_coverage_and_the_nightly_run_does(self) -> None:
        """2026-10-21: the approval run's ubuntu / 3.11 (the cell that measured and uploaded
        coverage) runs the suite with `--no-cov`, docs-only or not, and uploads nothing; the
        same cell of the nightly run measures, checks the floor and uploads."""
        coverage_cell = {"os": "ubuntu-latest", "python-version": "3.11"}
        for docs_only in ("true", "false"):
            steps = _steps_that_run(coverage_cell, docs_only)
            assert "Run unit tests with coverage" not in steps, docs_only
            assert "Upload coverage to Codecov" not in steps, docs_only
        pull_request = _steps_that_run(coverage_cell, "false")
        assert "Run unit tests without coverage" in pull_request
        nightly = _steps_that_run(
            coverage_cell, "false", event_name="schedule", ref="refs/heads/main"
        )
        assert "Run unit tests with coverage" in nightly
        assert "Upload coverage to Codecov" in nightly
        assert "Run unit tests without coverage" not in nightly

    @pytest.mark.parametrize("matrix", ("QUICK_MATRIX", "APPROVAL_MATRIX"))
    def test_a_docs_only_run_still_creates_every_context_branch_protection_waits_on(
        self, matrix: str
    ) -> None:
        """The matrix does not look at `docs_only`: a cell skipped by the matrix would be a
        context nobody reports, and the merge button would stay locked."""
        test = _jobs()["test"]
        assert test["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
        assert "if" not in test, "a job skipped by its `if:` reads 'skipped', not success"
        assert "docs_only" not in str(test["strategy"])
        assert "docs_only" not in str(test["runs-on"])
        if matrix == "APPROVAL_MATRIX":
            assert _cells(_matrix(matrix)) == REQUIRED_TEST_CONTEXTS


class TestTheSmokeJobsSkipOnlyInADocsOnlyRun:
    @pytest.mark.parametrize("name", SMOKE_JOBS)
    def test_every_step_is_guarded(self, name: str) -> None:
        for step in _steps(name):
            assert SMOKE_GUARD in str(step.get("if", "")), (name, _label(step))

    @pytest.mark.parametrize("name", SMOKE_JOBS)
    @pytest.mark.parametrize(("docs_only", "runs"), (("true", False), ("false", True), ("", True)))
    def test_no_step_runs_in_a_docs_only_run_and_every_step_runs_otherwise(
        self, name: str, docs_only: str, runs: bool
    ) -> None:
        context = {
            **_plan_outputs(docs_only),
            "steps": {"hf-cache": {"outputs": {"cache-hit": "false"}}},
        }
        for step in _steps(name):
            assert _holds(str(step["if"]), context) is runs, (name, _label(step), docs_only)

    @pytest.mark.parametrize("name", SMOKE_JOBS)
    def test_the_job_itself_is_not_skipped_it_reports(self, name: str) -> None:
        """Skipped by its own `if:` the job would read 'skipped' on a pull request that is
        docs-only; with its steps skipped it reads success, like every other required context."""
        job = _jobs()[name]
        assert job["if"] == GATE
        assert "plan" in (job["needs"] if isinstance(job["needs"], list) else [job["needs"]])

    def test_the_cache_save_of_the_pytorch_smoke_cannot_run_on_a_skipped_restore(self) -> None:
        """`success()` is true when every earlier step was skipped, and the restore step's
        `cache-hit` output is then empty: only the docs-only guard keeps the save from
        writing an empty cache."""
        save = next(
            step
            for step in _steps("pytorch-smoke")
            if step.get("name") == "Save PyTorch smoke HF cache"
        )
        assert str(save["if"]).startswith(SMOKE_GUARD + " && success()")

    def test_only_the_planned_jobs_read_docs_only(self) -> None:
        """`lint` still does its work on every pull request, and `core-floor` was left alone."""
        reading = {name for name, job in _jobs().items() if "docs_only" in str(job)}
        assert reading == SKIPPING_JOBS

    def test_no_job_level_condition_or_matrix_reads_it(self) -> None:
        for name, job in _jobs().items():
            for key in ("if", "strategy", "needs", "runs-on", "timeout-minutes"):
                assert "docs_only" not in str(job.get(key, "")), (name, key)


# --- The hang caps -----------------------------------------------------------------


class TestAHungTestIsCapped:
    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_the_test_step_has_a_time_limit_of_its_own(self, step_name: str) -> None:
        step = next(step for step in _steps("test") if step.get("name") == step_name)
        assert step["timeout-minutes"] == 45

    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_each_test_gets_a_per_test_cap_that_works_on_windows_and_in_a_worker(
        self, step_name: str
    ) -> None:
        """`thread` is the method with no SIGALRM, and the one that ends a worker instead of
        leaving it wedged; xdist then reports the crash on that test and starts another."""
        step = next(step for step in _steps("test") if step.get("name") == step_name)
        args = shlex.split(str(step["run"]))
        assert "--timeout=600" in args, args
        assert "--timeout-method=thread" in args, args

    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_a_crashed_worker_is_replaced_twice_at_most(self, step_name: str) -> None:
        """Measured with xdist 3.8.0: under `--dist loadfile` the replacement worker runs the
        crashed test again, so a test that hangs every time fails on every replacement. At the
        default of four per worker that is 17 tries of ten minutes, which outlasts the step;
        with 2 it is three tries, thirty minutes."""
        step = next(step for step in _steps("test") if step.get("name") == step_name)
        assert "--max-worker-restart=2" in shlex.split(str(step["run"]))

    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_the_stacks_of_a_slow_test_are_logged_before_the_cap_ends_it(
        self, step_name: str
    ) -> None:
        """`faulthandler_timeout` is pytest's own (no dependency) and, like the cap, works in an
        xdist worker: a hang leaves its location in the log of the cell that hit it."""
        step = next(step for step in _steps("test") if step.get("name") == step_name)
        args = shlex.split(str(step["run"]))
        assert "faulthandler_timeout=300" in _option(args, "-o"), args
        assert 300 < _per_test_cap(step), "the dump must come before the cap ends the test"

    def test_the_caps_are_far_above_a_healthy_run(self) -> None:
        for step in _steps("test"):
            if step.get("name") in RUN_STEPS:
                assert _per_test_cap(step) >= 10 * SLOWEST_TEST_SECONDS
                assert step["timeout-minutes"] >= 1.5 * LONGEST_TEST_STEP_MINUTES
                assert step["timeout-minutes"] < _jobs()["test"]["timeout-minutes"]

    def test_the_job_limit_is_above_every_job_measured_and_above_the_step_limit(self) -> None:
        """60 minutes: 1.6 times the slowest successful `test` job of 2026-10-12..19 (37.5
        minutes, 1145 jobs), with room above the 45-minute step for checkout, install and the
        cache warm-up. It was 90."""
        limit = _jobs()["test"]["timeout-minutes"]
        assert limit == 60
        assert limit >= 1.5 * SLOWEST_TEST_JOB_MINUTES
        step_limits = [
            step["timeout-minutes"] for step in _steps("test") if "timeout-minutes" in step
        ]
        assert step_limits and limit >= max(step_limits) + 10

    def test_no_other_job_passes_the_flag_to_a_pytest_that_may_not_have_the_plugin(self) -> None:
        """mlx-smoke and core-floor install `pytest` alone: `--timeout` there is an error."""
        for name, job in _jobs().items():
            for step in job.get("steps", []):
                if "--timeout" in str(step.get("run", "")):
                    assert name == "test", (name, _label(step))

    def test_the_plugin_comes_with_the_dev_extra_the_test_job_installs(self) -> None:
        text = PYPROJECT.read_text(encoding="utf-8")
        table = re.search(
            r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S
        )
        assert table is not None
        dev = re.search(r"^dev = \[(.*?)\]\s*$", table.group(1), re.M)
        assert dev is not None
        requirements = re.findall(r'"([^"\n]*)"', dev.group(1))
        assert any(req.startswith("pytest-timeout>=2") for req in requirements), requirements
        install = next(
            step for step in _steps("test") if step.get("name") == "Install dependencies"
        )
        assert ".[dev]" in shlex.split(str(install["run"]))

    def test_a_local_run_needs_no_plugin_and_has_no_cap(self) -> None:
        """The flags are a CI choice, as `-n logical` is: in `addopts` they would break a
        checkout that has not installed the plugin, and cap every local run."""
        text = PYPROJECT.read_text(encoding="utf-8")
        table = re.search(r"^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
        assert table is not None
        assert "timeout" not in table.group(1).lower()


# --- What a contributor is told ------------------------------------------------------


def _contributing() -> str:
    return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())


def test_contributing_says_what_a_docs_only_pull_request_runs() -> None:
    text = _contributing()
    assert "A docs-only pull request runs the suite once" in text
    for path in ("`docs/`", "`changelog.d/`", "`CONTRIBUTORS.md`", "`CONTRIBUTING.md`"):
        assert path in text, path
    assert "`README*.md`" in text and "`benchmarks/`" in text


def test_contributing_says_what_caps_a_slow_test() -> None:
    text = _contributing()
    assert "A test that runs for ten minutes fails" in text
    assert "`@pytest.mark.timeout(" in text
    assert "`--timeout=600 --timeout-method=thread --max-worker-restart=2`" in text
    assert "`-o faulthandler_timeout=300`" in text
    assert "a 60-minute job limit" in text
