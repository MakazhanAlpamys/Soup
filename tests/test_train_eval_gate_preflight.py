"""``soup train`` refuses a ``training.eval_gate.suite`` it cannot load (#2231).

``--gate <path>`` has always validated its suite up front and exited 1. A gate
enabled in ``soup.yaml`` got no such check: the callback logged the loader's
error at WARNING and trained with no gate, so ``on_regression: stop`` could
never stop the run. These tests drive ``soup train --dry-run`` through the CLI
and assert the refusal arrives before the dataset loader or ``build_trainer``
is touched. Removing ``_preflight_config_eval_gate`` from
``soup_cli.commands.train`` fails every refusal test below.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

_CONFIG = """
base: Qwen/Qwen2.5-0.5B-Instruct
task: sft
data:
  train: train.jsonl
  format: alpaca
training:
  epochs: 1
  lr: 1.0e-4
  eval_gate:
    enabled: {enabled}
    suite: {suite}
  lora:
    r: 8
    alpha: 16
output: out
"""

_VALID_TASK = {
    "type": "custom",
    "name": "exact",
    "threshold": 0.5,
    "tasks": "tasks.jsonl",
    "scorer": "exact",
}


def _plain(text: str) -> str:
    """Strip ANSI escapes and collapse whitespace (Rich colours and wraps on CI)."""
    return " ".join(strip_ansi(text).split())


def _write_config(tmp_path: Path, *, suite: str, enabled: bool = True) -> Path:
    (tmp_path / "train.jsonl").write_text(
        '{"instruction": "q", "output": "a"}\n' * 4, encoding="utf-8"
    )
    path = tmp_path / "soup.yaml"
    path.write_text(_CONFIG.format(enabled=str(enabled).lower(), suite=suite), encoding="utf-8")
    return path


def _write_suite(tmp_path: Path, name: str, payload: object) -> str:
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    (tmp_path / name).write_text(text, encoding="utf-8")
    return name


@pytest.fixture
def heavy_steps_are_never_reached(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count calls to the dataset loader and ``build_trainer``.

    ``load_dataset`` is bound at import time in ``soup_cli.commands.train``;
    ``build_trainer`` is imported lazily from ``soup_cli.trainer.dispatch``, so a
    stub module stands in for it.
    """
    calls = {"load_dataset": 0, "build_trainer": 0}

    def _load_dataset(*args: object, **kwargs: object) -> object:
        calls["load_dataset"] += 1
        raise AssertionError("load_dataset must not run after a refused eval gate")

    def _build_trainer(*args: object, **kwargs: object) -> object:
        calls["build_trainer"] += 1
        raise AssertionError("build_trainer must not run after a refused eval gate")

    monkeypatch.setattr("soup_cli.commands.train.load_dataset", _load_dataset)
    monkeypatch.setitem(
        sys.modules,
        "soup_cli.trainer.dispatch",
        types.SimpleNamespace(build_trainer=_build_trainer),
    )
    return calls


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)


def _dry_run(config: Path, *extra: str):
    return CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run", *extra])


_BAD_SUITES: list[tuple[str, object, str]] = [
    (
        "no_tasks",
        {"suite": "empty", "tasks": []},
        "field 'tasks' must contain at least one task",
    ),
    (
        "repeated_task_name",
        {"suite": "twice", "tasks": [_VALID_TASK, dict(_VALID_TASK)]},
        "repeats task name 'exact'",
    ),
    (
        "malformed_yaml",
        "suite: [unclosed\ntasks:\n  - :\n",
        "is not valid YAML",
    ),
    (
        "invalid_task_field",
        {"suite": "bad", "tasks": [{**_VALID_TASK, "threshold": 5000}]},
        "threshold",
    ),
]


@pytest.mark.parametrize(
    "payload, fragment",
    [(payload, fragment) for _, payload, fragment in _BAD_SUITES],
    ids=[name for name, _, _ in _BAD_SUITES],
)
def test_a_bad_suite_in_soup_yaml_is_refused_before_dry_run_returns(
    tmp_path: Path,
    heavy_steps_are_never_reached: dict[str, int],
    payload: object,
    fragment: str,
) -> None:
    suite = _write_suite(tmp_path, "gate.yaml", payload)
    config = _write_config(tmp_path, suite=suite)

    result = _dry_run(config)

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Invalid training.eval_gate.suite:" in output, output
    assert fragment in output, output
    for later_line in ("Dry run", "Data OK", "Config valid.", "Ready to train"):
        assert later_line not in output, (later_line, output)
    assert heavy_steps_are_never_reached == {"load_dataset": 0, "build_trainer": 0}


def test_a_missing_suite_file_in_soup_yaml_is_refused(
    tmp_path: Path, heavy_steps_are_never_reached: dict[str, int]
) -> None:
    config = _write_config(tmp_path, suite="missing.yaml")

    result = _dry_run(config)

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Invalid training.eval_gate.suite:" in output, output
    assert "eval-gate suite not found: missing.yaml" in output, output
    assert heavy_steps_are_never_reached == {"load_dataset": 0, "build_trainer": 0}


def test_a_valid_suite_in_soup_yaml_still_dry_runs(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, "gate.yaml", {"suite": "ok", "tasks": [_VALID_TASK]})
    config = _write_config(tmp_path, suite=suite)

    result = _dry_run(config)

    output = _plain(result.output)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Invalid training.eval_gate.suite:" not in output, output
    assert "Config valid. Ready to train!" in output, output


def test_a_disabled_gate_with_a_bad_suite_is_not_checked(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, "gate.yaml", {"suite": "empty", "tasks": []})
    config = _write_config(tmp_path, suite=suite, enabled=False)

    result = _dry_run(config)

    output = _plain(result.output)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Invalid training.eval_gate.suite:" not in output, output
    assert "Config valid. Ready to train!" in output, output


def test_gate_flag_with_a_valid_suite_wins_over_a_bad_suite_in_soup_yaml(
    tmp_path: Path,
) -> None:
    """``--gate`` replaces ``training.eval_gate`` and behaves as it does today."""
    bad = _write_suite(tmp_path, "bad.yaml", {"suite": "empty", "tasks": []})
    good = _write_suite(tmp_path, "good.yaml", {"suite": "ok", "tasks": [_VALID_TASK]})
    config = _write_config(tmp_path, suite=bad)

    result = _dry_run(config, "--gate", good)

    output = _plain(result.output)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Invalid training.eval_gate.suite:" not in output, output
    assert "Eval gate enabled with suite: good.yaml" in output, output
    assert "Config valid. Ready to train!" in output, output


def test_gate_flag_with_a_bad_suite_keeps_its_own_message(
    tmp_path: Path, heavy_steps_are_never_reached: dict[str, int]
) -> None:
    bad = _write_suite(tmp_path, "bad.yaml", {"suite": "empty", "tasks": []})
    config = _write_config(tmp_path, suite=bad)

    result = _dry_run(config, "--gate", bad)

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Invalid --gate suite:" in output, output
    assert "Invalid training.eval_gate.suite:" not in output, output
    assert heavy_steps_are_never_reached == {"load_dataset": 0, "build_trainer": 0}
