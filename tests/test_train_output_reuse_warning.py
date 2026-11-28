"""Issue #3299 — `soup train` warns before replacing a previous SFT adapter.

A fresh SFT run whose output directory already holds final adapter files
replaces them on save. The run must print one notice naming the resolved
destination and the existing files before the trainer starts, so the operator
can move them aside. A clean destination and a checkpoint-only directory stay
silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.commands.train import _warn_existing_output_files
from soup_cli.config.schema import SoupConfig
from tests.conftest import make_cli_runner, strip_ansi

_ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer_config.json",
)


def _output(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _squash(text: str) -> str:
    """Remove all whitespace, so a Rich line-wrap cannot split a token.

    The module-level console freezes its width from COLUMNS at import time,
    so a test cannot widen it afterwards; with a long tmp_path the notice
    can still break mid-word on a narrow console. Comparing without
    whitespace rejoins any such break.
    """
    return "".join(text.split())


class _StubTrainer:
    """Stand in for the real trainer: record that it started, then finish."""

    def __init__(self, calls: list, output_dir: str) -> None:
        self._calls = calls
        self._output_dir = output_dir

    def setup(self, dataset: dict) -> None:
        self._calls.append("setup")
        print("stub trainer starting")

    def train(self, **kwargs) -> dict:
        self._calls.append("train")
        return {
            "output_dir": self._output_dir,
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 1,
            "duration_secs": 1.0,
        }


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("soup_cli.utils.launcher.is_in_distributed", lambda: False)
    monkeypatch.setattr("soup_cli.commands.train.detect_device", lambda **_: ("cpu", "CPU"))
    monkeypatch.setattr(
        "soup_cli.commands.train.get_gpu_info", lambda **_: {"memory_total": "N/A"}
    )
    Path("data.jsonl").write_text('{"instruction":"hi", "output":"hello"}\n', encoding="utf-8")
    config = tmp_path / "soup.yaml"
    config.write_text(
        "base: owner/base\ntask: sft\nbackend: transformers\n"
        "data: {train: data.jsonl, format: alpaca, val_split: 0}\n"
        "training: {epochs: 1, quantization: none, batch_size: 1}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n"
        "experiment_name: exp1\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out" / "exp1"
    calls: list[str] = []
    monkeypatch.setattr(
        "soup_cli.trainer.dispatch.build_trainer",
        lambda *_a, **_kw: _StubTrainer(calls, str(dest)),
    )
    return config, dest, calls


def test_fresh_run_warns_about_existing_adapter_files(run) -> None:
    config, dest, calls = run
    dest.mkdir(parents=True)
    for name in _ADAPTER_FILES:
        (dest / name).write_text("x", encoding="utf-8")

    result = make_cli_runner().invoke(app, ["train", "--config", str(config), "--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["setup", "train"]
    output = _output(result)
    notice = (
        f"Warning: {dest} already has adapter_model.safetensors, "
        "adapter_config.json and tokenizer_config.json. This run may replace them."
    )
    squashed = _squash(output)
    assert _squash(notice) in squashed
    assert squashed.index(_squash(notice)) < squashed.index("stubtrainerstarting")


def test_clean_destination_stays_silent(run) -> None:
    config, dest, calls = run

    result = make_cli_runner().invoke(app, ["train", "--config", str(config), "--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["setup", "train"]
    assert "already has" not in _output(result)
    assert "This run may replace them" not in _output(result)


def test_checkpoint_only_directory_stays_silent(run) -> None:
    config, dest, calls = run
    (dest / "checkpoint-100").mkdir(parents=True)

    result = make_cli_runner().invoke(app, ["train", "--config", str(config), "--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["setup", "train"]
    assert "already has" not in _output(result)
    assert "This run may replace them" not in _output(result)


def _silent_cfg(
    tmp_path: Path,
    task: str = "sft",
    backend: str = "transformers",
    output: str | None = None,
) -> SoupConfig:
    return SoupConfig.model_validate(
        {
            "base": "owner/base",
            "task": task,
            "backend": backend,
            "data": {"train": "data.jsonl", "format": "alpaca", "val_split": 0},
            "training": {"epochs": 1, "quantization": "none", "batch_size": 1},
            "output": output if output is not None else str(tmp_path / "out"),
            "experiment_name": "exp1",
        }
    )


def _seed_adapter_files(dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in _ADAPTER_FILES:
        (dest / name).write_text("x", encoding="utf-8")


@pytest.mark.parametrize(
    ("task", "backend", "resume", "hf_resume"),
    [
        ("sft", "transformers", "checkpoint-100", False),
        ("sft", "transformers", None, True),
        ("dpo", "transformers", None, False),
        ("sft", "mlx", None, False),
    ],
)
def test_silent_branches_stay_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture, task, backend, resume, hf_resume,
) -> None:
    # The files exist, so silence here proves the branch — not an empty directory.
    _seed_adapter_files(tmp_path / "out" / "exp1")
    _warn_existing_output_files(
        _silent_cfg(tmp_path, task, backend), resume=resume, hf_resume=hf_resume,
    )
    captured = capsys.readouterr()
    assert "already has" not in captured.out + captured.err
    assert "This run may replace them" not in captured.out + captured.err


def test_helper_warns_on_the_fresh_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Control for the silence test above: the same call on the fresh path
    # does print, so silence above is the branch working, not lost output.
    _seed_adapter_files(tmp_path / "out" / "exp1")
    _warn_existing_output_files(_silent_cfg(tmp_path), resume=None, hf_resume=False)
    captured = capsys.readouterr()
    assert "already has adapter_model.safetensors" in captured.out + captured.err


def test_only_chief_worker_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # In a multi-worker launch every worker reaches the warning, so only the
    # chief prints: across a RANK=0 and a RANK=1 call there is one notice.
    # A plain single-node launch without RANK still warns (LOCAL_RANK == 0),
    # while a non-chief LOCAL_RANK stays silent.
    _seed_adapter_files(tmp_path / "out" / "exp1")
    cfg = _silent_cfg(tmp_path)
    for rank_env in (
        {"RANK": "0", "WORLD_SIZE": "2"},
        {"RANK": "1", "WORLD_SIZE": "2"},
        {"LOCAL_RANK": "1"},
        {},
    ):
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            monkeypatch.delenv(key, raising=False)
        for key, value in rank_env.items():
            monkeypatch.setenv(key, value)
        _warn_existing_output_files(cfg, resume=None, hf_resume=False)
    captured = capsys.readouterr()
    assert (captured.out + captured.err).count("already has") == 2


def test_fresh_run_warns_once_on_chief_rank(run, monkeypatch: pytest.MonkeyPatch) -> None:
    config, dest, calls = run
    dest.mkdir(parents=True)
    for name in _ADAPTER_FILES:
        (dest / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("RANK", "0")

    result = make_cli_runner().invoke(app, ["train", "--config", str(config), "--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["setup", "train"]
    squashed = _squash(_output(result))
    notice = (
        f"Warning: {dest} already has adapter_model.safetensors, "
        "adapter_config.json and tokenizer_config.json. This run may replace them."
    )
    assert squashed.count(_squash(notice)) == 1
    assert squashed.index(_squash(notice)) < squashed.index("stubtrainerstarting")


def test_fresh_run_stays_silent_on_non_chief_rank(run, monkeypatch: pytest.MonkeyPatch) -> None:
    config, dest, calls = run
    dest.mkdir(parents=True)
    for name in _ADAPTER_FILES:
        (dest / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("RANK", "1")

    result = make_cli_runner().invoke(app, ["train", "--config", str(config), "--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["setup", "train"]
    assert "already has" not in _output(result)
    assert "This run may replace them" not in _output(result)


@pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not a valid Windows filename character"
)
def test_helper_prints_emoji_alias_path_as_supplied(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # A POSIX destination containing a Rich emoji alias must appear exactly
    # as supplied: without literal rendering, ":smile:" is expanded to the
    # emoji and the rendered path differs from the directory the save uses.
    dest = tmp_path / "out:smile:" / "exp1"
    _seed_adapter_files(dest)
    _warn_existing_output_files(
        _silent_cfg(tmp_path, output=str(tmp_path / "out:smile:")),
        resume=None, hf_resume=False,
    )
    captured = capsys.readouterr()
    assert str(dest) in _squash(strip_ansi(captured.out + captured.err))
    assert "😄" not in captured.out + captured.err


def test_helper_prints_bracketed_path_as_supplied(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # A destination containing Rich markup text must appear exactly as
    # supplied: without escaping, "[yellow]" is read as a style tag and the
    # rendered path differs from the directory the save will use.
    dest = tmp_path / "out[yellow]" / "exp1"
    _seed_adapter_files(dest)
    _warn_existing_output_files(
        _silent_cfg(tmp_path, output=str(tmp_path / "out[yellow]")),
        resume=None, hf_resume=False,
    )
    captured = capsys.readouterr()
    assert str(dest) in _squash(strip_ansi(captured.out + captured.err))
