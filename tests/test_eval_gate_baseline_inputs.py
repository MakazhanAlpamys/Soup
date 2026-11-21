"""A baseline output must not replace the suite or a task's local inputs."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import Result
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.gate import GateTask, write_baseline_file
from soup_cli.utils import paths
from tests.conftest import strip_ansi

SUITE = "inputs/suite[red].yaml"
CUSTOM = "inputs/tasks.jsonl"
JUDGE = "inputs/prompts.jsonl"
INPUTS = [SUITE, CUSTOM, JUDGE]
FIELDS = ["--suite", "task 'custom' --tasks", "task 'judge' --prompts"]


def normalized(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    Path("inputs").mkdir()
    Path(SUITE).write_text(
        "suite: protected\ntasks:\n"
        f"  - {{name: custom, type: custom, tasks: {CUSTOM}, scorer: exact, threshold: 0.5}}\n"
        f"  - {{name: judge, type: judge, prompts: {JUDGE}, "
        "judge_model: 'ollama://synthetic', threshold: 0.5}\n",
        encoding="utf-8",
    )
    Path(CUSTOM).write_text('{"prompt":"custom-control","expected":"gold"}\n', encoding="utf-8")
    Path(JUDGE).write_text('{"prompt":"judge-control"}\n', encoding="utf-8")
    generated = []
    judged = []

    def generate(prompt: str) -> str:
        generated.append(prompt)
        return "gold"

    def judge(task: GateTask, generate_fn: Callable[[str], str]) -> float:
        judged.append(task.name)
        generate_fn("judge-control")
        return 1.0

    factory = Mock(return_value=generate)
    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", factory)
    monkeypatch.setattr("soup_cli.eval.gate._run_judge_task", judge)
    return {
        "before": {path: Path(path).read_bytes() for path in INPUTS},
        "factory": factory, "generated": generated, "judged": judged,
    }


def invoke(output: str, *extra: str) -> Result:
    return CliRunner().invoke(app, [
        "eval", "gate", "--suite", SUITE, "--model", "synthetic",
        "--write-baseline", output, *extra,
    ])


def preserved(fixture: dict) -> None:
    assert {path: Path(path).read_bytes() for path in INPUTS} == fixture["before"]


def assert_collision(fixture: dict, output: str, source: str, field: str) -> None:
    result = invoke(output)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert normalized(result.output) == (
        f"--write-baseline '{output}' aliases {field} '{source}'; "
        "choose a different --write-baseline"
    )
    fixture["factory"].assert_not_called()
    assert fixture["generated"] == [] and fixture["judged"] == []
    preserved(fixture)


@pytest.mark.parametrize(
    "source,field", list(zip(INPUTS, FIELDS)), ids=["suite", "custom", "judge"],
)
@pytest.mark.parametrize("spelling", ["same", "relative", "parent"])
def test_colliding_output_is_refused_before_model(
    fixture: dict, source: str, field: str, spelling: str,
) -> None:
    output = source
    if spelling == "relative":
        output = "./" + source
    elif spelling == "parent":
        Path("temporary").mkdir()
        output = "temporary/../" + source
    assert_collision(fixture, output, source, field)


@pytest.mark.requires_symlink
@pytest.mark.parametrize(
    "source,field", list(zip(INPUTS, FIELDS)), ids=["suite", "custom", "judge"],
)
def test_parent_directory_alias_is_refused(fixture: dict, source: str, field: str) -> None:
    os.symlink("inputs", "alias", target_is_directory=True)
    assert_collision(fixture, source.replace("inputs/", "alias/", 1), source, field)


@pytest.mark.parametrize(
    "source,field", list(zip(INPUTS, FIELDS)), ids=["suite", "custom", "judge"],
)
def test_case_alias_is_refused_using_entry_identity(
    fixture: dict, source: str, field: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = source.upper()
    real_stat = os.stat
    alias_absolute = os.path.abspath(alias)

    def stat_with_alias(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and os.fspath(path) == alias_absolute:
            return real_stat(source, *args, **kwargs)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(paths.os, "stat", stat_with_alias)
    assert_collision(fixture, alias, source, field)


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS path lookup")
@pytest.mark.requires_hardlink
@pytest.mark.parametrize("lookup", ["native", "no-link-id", "no-link-id-or-listing"])
def test_case_alias_and_separate_link_on_case_insensitive_volume(
    fixture: dict, monkeypatch: pytest.MonkeyPatch, lookup: str,
) -> None:
    alias = str(Path(CUSTOM).with_name("TASKS.JSONL"))
    if not Path(alias).exists():
        pytest.skip("the test volume is case-sensitive")
    separate = str(Path(CUSTOM).with_name("copy.jsonl"))
    os.link(CUSTOM, separate)
    if lookup == "native":
        read_link_id = getattr(paths, "_macos_link_id", lambda _: 0)
        if read_link_id(CUSTOM) is None:
            pytest.skip("the test volume does not supply a link ID")
    else:
        monkeypatch.setattr(paths, "_macos_link_id", lambda _: None, raising=False)

    directory = Path(CUSTOM).parent
    original_mode = directory.stat().st_mode & 0o777
    try:
        if lookup == "no-link-id-or-listing":
            directory.chmod(0o300)
            try:
                os.listdir(directory)
            except PermissionError:
                pass
            else:
                pytest.skip("the current user can list a write/search-only directory")
        assert_collision(fixture, alias, CUSTOM, FIELDS[1])
        result = invoke(separate.upper())
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert json.loads(Path(separate).read_text())["scores"] == {"custom": 1.0, "judge": 1.0}
        preserved(fixture)
    finally:
        if lookup == "no-link-id-or-listing":
            directory.chmod(original_mode)


def test_fresh_baseline_preserves_inputs(fixture: dict) -> None:
    result = invoke("baseline.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path("baseline.json").read_text())["scores"] == {"custom": 1.0, "judge": 1.0}
    fixture["factory"].assert_called_once_with("synthetic")
    assert fixture["generated"] == ["custom-control", "judge-control"]
    assert fixture["judged"] == ["judge"]
    preserved(fixture)


def test_intentional_baseline_refresh_is_allowed(fixture: dict) -> None:
    write_baseline_file("baseline.json", {"custom": 0.5, "judge": 0.5})
    result = invoke("./baseline.json", "--baseline", "baseline.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path("baseline.json").read_text())["scores"] == {"custom": 1.0, "judge": 1.0}
    preserved(fixture)


@pytest.mark.requires_hardlink
@pytest.mark.parametrize("source", INPUTS, ids=["suite", "custom", "judge"])
def test_atomic_hard_link_output_preserves_original(fixture: dict, source: str) -> None:
    os.link(source, "baseline.json")
    result = invoke("baseline.json")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(Path("baseline.json").read_text())["scores"] == {"custom": 1.0, "judge": 1.0}
    preserved(fixture)


@pytest.mark.requires_symlink
def test_final_component_symlink_is_refused_before_model(fixture: dict) -> None:
    os.symlink(CUSTOM, "baseline.json")
    result = invoke("baseline.json")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "symlink" in normalized(result.output).lower()
    fixture["factory"].assert_not_called()
    preserved(fixture)


def test_output_directory_is_refused_before_model(fixture: dict) -> None:
    result = invoke("inputs")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "directory" in normalized(result.output).lower()
    fixture["factory"].assert_not_called()
    preserved(fixture)
