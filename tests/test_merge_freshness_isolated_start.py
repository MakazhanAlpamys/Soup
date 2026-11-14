"""The merge-freshness workflow starts its gate script in isolated mode, from outside the checkout.

Both jobs of ``.github/workflows/merge-freshness.yml`` (``gate`` on a pull request,
``reeval-open-prs`` on a push to ``main``) write ``main``'s copy of
``scripts/merge_freshness.py`` to ``$RUNNER_TEMP`` and run it as
``python -I "$RUNNER_TEMP/merge_freshness.py" ...``. Isolated mode (``-I`` implies ``-E``
and ``-s``) leaves the script's own directory, the current directory and every ``PYTHON*``
variable off ``sys.path``; the interpreter's own site-packages stay, so an installed
``ruff`` still resolves.

Two kinds of test:

* the workflow file is parsed and every ``run:`` that mentions the script is classified,
  so a step added later that starts it some other way fails here by name;
* the script itself is run in a subprocess next to decoy modules named like one of its
  stdlib imports, with and without ``-I``, so the isolation is observed, not assumed.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "merge-freshness.yml"
SCRIPT = ROOT / "scripts" / "merge_freshness.py"
SCRIPT_NAME = "merge_freshness.py"

#: The path every start and every copy uses, with ``${{ runner.temp }}`` folded into it.
TEMP_PREFIX = "$RUNNER_TEMP/"
GIT_SOURCE = "origin/main:scripts/merge_freshness.py"
COPY_LINE = f'git show {GIT_SOURCE} > "{TEMP_PREFIX}{SCRIPT_NAME}"'
JOBS_THAT_RUN_THE_SCRIPT = ("gate", "reeval-open-prs")

_RUNNER_TEMP_EXPRESSION = re.compile(r"\$\{\{\s*runner\.temp\s*\}\}")
_PYTHON = re.compile(r"^python(\d+(\.\d+)*)?$")
#: The decoy is named after a stdlib module the script imports at module scope and that
#: the interpreter itself does not need to start: ``tarfile``.
DECOY_MODULE = "tarfile"


# --------------------------------------------------------------------------------------
# helpers for the workflow file
# --------------------------------------------------------------------------------------


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _run_steps(job_id: str) -> list[tuple[int, str, str]]:
    """``(index in the job, step name, run text)`` for every ``run:`` step of a job."""
    steps = _workflow()["jobs"][job_id]["steps"]
    return [
        (index, str(step.get("name", "")), str(step["run"]))
        for index, step in enumerate(steps)
        if "run" in step
    ]


def _folded(run: str) -> str:
    """One line, single spaces, with ``${{ runner.temp }}`` spelled ``$RUNNER_TEMP``."""
    return " ".join(_RUNNER_TEMP_EXPRESSION.sub("$RUNNER_TEMP", run).split())


def _tokens(run: str) -> list[str]:
    """Shell tokens of a ``run:`` text; ``>`` and ``&&`` are tokens of their own."""
    text = _RUNNER_TEMP_EXPRESSION.sub("$RUNNER_TEMP", run)
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _references(run: str) -> list[tuple[str, str, list[str]]]:
    """Every token that names the script, classified.

    Returns ``(kind, token, flags)``: ``start`` for ``python [flags] <script>``,
    ``redirect`` for the target of ``>``, ``git-show`` for the ``git show`` source and
    ``other`` for anything else (which no step is allowed to have).
    """
    tokens = _tokens(run)
    found: list[tuple[str, str, list[str]]] = []
    for index, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1].rsplit(":", 1)[-1] != SCRIPT_NAME:
            continue
        flags: list[str] = []
        back = index - 1
        while back >= 0 and tokens[back].startswith("-"):
            flags.append(tokens[back])
            back -= 1
        before = tokens[back] if back >= 0 else ""
        if _PYTHON.match(before):
            found.append(("start", token, flags))
        elif index and tokens[index - 1] == ">":
            found.append(("redirect", token, []))
        elif token == GIT_SOURCE and index >= 2 and tokens[index - 2 : index] == ["git", "show"]:
            found.append(("git-show", token, []))
        else:
            found.append(("other", token, flags))
    return found


def _all_references() -> list[tuple[str, int, str, str, list[str]]]:
    """``(job id, step index, kind, token, flags)`` over both jobs."""
    out: list[tuple[str, int, str, str, list[str]]] = []
    for job_id in _workflow()["jobs"]:
        for index, _name, run in _run_steps(job_id):
            for kind, token, flags in _references(run):
                out.append((job_id, index, kind, token, flags))
    return out


# --------------------------------------------------------------------------------------
# the workflow file
# --------------------------------------------------------------------------------------


class TestWorkflowStart:
    def test_both_jobs_start_the_script_exactly_once(self):
        starts = [ref for ref in _all_references() if ref[2] == "start"]
        assert sorted(ref[0] for ref in starts) == sorted(JOBS_THAT_RUN_THE_SCRIPT), starts

    def test_every_start_is_isolated_and_under_runner_temp(self):
        starts = [ref for ref in _all_references() if ref[2] == "start"]
        assert starts, "the workflow no longer starts merge_freshness.py at all"
        for job_id, _index, _kind, token, flags in starts:
            assert "-I" in flags, f"{job_id}: python must run with -I, got flags {flags}"
            assert token == f"{TEMP_PREFIX}{SCRIPT_NAME}", (
                f"{job_id}: the script must start from {TEMP_PREFIX}{SCRIPT_NAME}, got {token}"
            )

    def test_no_step_runs_the_script_from_the_workspace(self):
        for job_id, index, kind, token, _flags in _all_references():
            if kind == "git-show":
                continue
            assert token.startswith(TEMP_PREFIX), (
                f"{job_id} step {index}: {kind} of {token!r} is outside {TEMP_PREFIX}"
            )
        text = WORKFLOW.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert "python scripts/merge_freshness.py" not in line, line
            assert "python -I scripts/merge_freshness.py" not in line, line

    def test_no_reference_is_unclassified(self):
        others = [ref for ref in _all_references() if ref[2] == "other"]
        assert not others, f"a step names the script in a way this test cannot vouch for: {others}"

    def test_no_step_checks_the_script_out_into_the_working_tree(self):
        for job_id in _workflow()["jobs"]:
            for index, _name, run in _run_steps(job_id):
                if SCRIPT_NAME not in run:
                    continue
                for command in re.split(r"[;&|\n]+", run):
                    assert not re.search(r"\bgit\s+(checkout|restore|switch)\b", command), (
                        f"{job_id} step {index} checks the script out: {command.strip()}"
                    )

    def test_each_job_copies_the_script_from_origin_main_before_starting_it(self):
        for job_id in JOBS_THAT_RUN_THE_SCRIPT:
            copies = [
                index
                for index, _name, run in _run_steps(job_id)
                if _folded(run) == COPY_LINE
            ]
            starts = [
                index
                for _job, index, kind, _token, _flags in _all_references()
                if _job == job_id and kind == "start"
            ]
            assert len(copies) == 1, f"{job_id}: expected one copy step, found {copies}"
            assert len(starts) == 1
            assert copies[0] < starts[0], f"{job_id}: the copy must come before the start"

    def test_the_copy_is_written_to_a_file_named_like_the_script(self):
        redirects = [ref for ref in _all_references() if ref[2] == "redirect"]
        assert sorted(ref[0] for ref in redirects) == sorted(JOBS_THAT_RUN_THE_SCRIPT)
        assert {ref[3] for ref in redirects} == {f"{TEMP_PREFIX}{SCRIPT_NAME}"}


class TestWorkflowKeepsEverythingElse:
    """Hardening how the gate starts changes no name, trigger or permission."""

    def test_names_triggers_and_permissions_are_unchanged(self):
        data = _workflow()
        assert data["name"] == "merge freshness"
        assert set(data["jobs"]) == set(JOBS_THAT_RUN_THE_SCRIPT)
        assert data["permissions"] == {
            "contents": "read",
            "checks": "write",
            "pull-requests": "read",
        }
        assert "name" not in data["jobs"]["gate"]
        assert "name" not in data["jobs"]["reeval-open-prs"]
        on = data.get("on", data.get(True))
        assert set(on) == {"pull_request", "push", "workflow_dispatch"}

    def test_the_existing_step_names_are_unchanged(self):
        names = {
            job_id: [step.get("name") for step in _workflow()["jobs"][job_id]["steps"]]
            for job_id in JOBS_THAT_RUN_THE_SCRIPT
        }
        assert "Fetch current main" in names["gate"]
        assert "Take the gate script from main, not from the PR" in names["gate"]
        assert "Gate the live merge" in names["gate"]
        assert "Re-evaluate open PRs against live main" in names["reeval-open-prs"]

    def test_the_arguments_after_the_script_are_unchanged(self):
        gate_run = _folded(
            next(run for _i, name, run in _run_steps("gate") if name == "Gate the live merge")
        )
        assert gate_run == (
            'python -I "$RUNNER_TEMP/merge_freshness.py" --max-behind 10 --repo-root . '
            "gate --base origin/main --head HEAD "
            "--head-sha ${{ github.event.pull_request.head.sha }} --post-check"
        )
        reeval_run = _folded(
            next(
                run
                for _i, name, run in _run_steps("reeval-open-prs")
                if name == "Re-evaluate open PRs against live main"
            )
        )
        assert reeval_run == (
            'python -I "$RUNNER_TEMP/merge_freshness.py" --max-behind 10 --repo-root . '
            "reeval --base origin/main"
        )


# --------------------------------------------------------------------------------------
# the script runs in isolated mode and needs nothing from the tree
# --------------------------------------------------------------------------------------


def _imported_top_level_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import needs the package next to the script"
            names.add((node.module or "").split(".")[0])
    return names


class TestScriptNeedsNothingFromTheTree:
    def test_it_imports_the_standard_library_only(self):
        imported = _imported_top_level_names(SCRIPT.read_text(encoding="utf-8"))
        assert imported, "no imports found: the scan is broken"
        assert DECOY_MODULE in imported, "the behavioural tests below need this import"
        outside = sorted(name for name in imported if name not in sys.stdlib_module_names)
        assert not outside, f"merge_freshness.py imports {outside}, which -I may not find"

    def test_it_does_not_touch_sys_path(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        touched = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ]
        assert not touched, f"merge_freshness.py uses sys.path on lines {touched}"


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _decoy(path: Path, marker: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{DECOY_MODULE}.py").write_text(
        f"open({str(marker)!r}, 'a', encoding='utf-8').write({label + chr(10)!r})\n",
        encoding="utf-8",
    )


def _clean_env(**extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(extra)
    return env


class _Layout:
    """The copy outside any tree, and a decoy in each place Python could find one."""

    def __init__(self, root: Path) -> None:
        self.marker = root / "marker.txt"
        self.copy_dir = root / "runner-temp"
        self.cwd = root / "workspace"
        self.pythonpath = root / "on-pythonpath"
        self.copy_dir.mkdir()
        self.script = self.copy_dir / SCRIPT_NAME
        shutil.copyfile(SCRIPT, self.script)
        _decoy(self.copy_dir, self.marker, "script-dir")
        _decoy(self.cwd, self.marker, "cwd")
        _decoy(self.pythonpath, self.marker, "pythonpath")

    def run(
        self, *args: str, isolated: bool, cwd: Path | None = None, with_pythonpath: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable] + (["-I"] if isolated else []) + [str(self.script), *args]
        env = _clean_env(**({"PYTHONPATH": str(self.pythonpath)} if with_pythonpath else {}))
        return subprocess.run(
            command,
            cwd=cwd or self.cwd,
            env=env,
            capture_output=True,
            # -I ignores PYTHONIOENCODING, so on Windows the child writes its console
            # code page: read it leniently, the assertions below are ASCII.
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )

    def imported(self) -> list[str]:
        if not self.marker.exists():
            return []
        return self.marker.read_text(encoding="utf-8").split()


@pytest.fixture
def layout(tmp_path: Path) -> _Layout:
    return _Layout(tmp_path)


class TestIsolatedModeIgnoresDecoys:
    def test_isolated_help_imports_no_decoy_and_still_works(self, layout):
        result = layout.run("--help", isolated=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "usage:" in result.stdout
        assert layout.imported() == [], "a decoy module was imported under python -I"

    def test_isolated_decide_runs_the_verdict_logic_outside_any_tree(self, layout):
        result = layout.run("decide", "--behind", "3", isolated=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert _plain(result.stdout).startswith("success: 3 commits behind main")
        assert layout.imported() == []

    @pytest.mark.parametrize(
        "label, with_pythonpath",
        [("script-dir", False), ("pythonpath", True)],
    )
    def test_without_dash_i_the_same_decoy_is_imported(self, layout, label, with_pythonpath):
        """The control: the decoys above are in places an ordinary start does search."""
        if label == "script-dir":
            # only the decoy next to the script is left in reach
            shutil.rmtree(layout.pythonpath)
        else:
            (layout.copy_dir / f"{DECOY_MODULE}.py").unlink()
        result = layout.run("--help", isolated=False, with_pythonpath=with_pythonpath)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert label in layout.imported(), (layout.imported(), result.stderr)


def _git(repo: Path, *args: str) -> None:
    env = _clean_env(
        GIT_AUTHOR_NAME="soup-test",
        GIT_AUTHOR_EMAIL="soup@test",
        GIT_COMMITTER_NAME="soup-test",
        GIT_COMMITTER_EMAIL="soup@test",
        GIT_TERMINAL_PROMPT="0",
    )
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


class TestGateRunsEndToEndUnderIsolatedMode:
    def test_gate_on_a_real_repository_from_the_outside_copy(self, layout):
        """What the ``gate`` job does: a repository as cwd, the script from outside it."""
        repo = layout.cwd
        _git(repo, "init", "-b", "main")
        (repo / "README.md").write_text("base\n", encoding="utf-8")
        _git(repo, "add", "README.md")
        _git(repo, "commit", "-m", "base")
        _git(repo, "branch", "pr")
        (repo / "NOTES.md").write_text("main moved\n", encoding="utf-8")
        _git(repo, "add", "NOTES.md")
        _git(repo, "commit", "-m", "main moves on")
        result = layout.run(
            "--max-behind", "10", "--repo-root", ".", "gate", "--base", "main", "--head", "pr",
            isolated=True,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert _plain(result.stdout).startswith("success: 1 commits behind main"), result.stdout
        assert layout.imported() == [], "a decoy module was imported under python -I"
