"""Issue #3299 — `soup train` warns before replacing a previous SFT adapter.

A fresh SFT run whose output directory already holds final adapter files
replaces them on save. The run must print one notice naming the resolved
destination and the existing files before the trainer starts, so the operator
can move them aside. A clean destination and a checkpoint-only directory stay
silent.
"""

from __future__ import annotations

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
    # Pin the console wide: the notice names a tmp_path, and on a narrow
    # console Rich breaks the path mid-word, so the notice assertion fails.
    monkeypatch.setenv("COLUMNS", "500")
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
    assert notice in output
    assert output.index(notice) < output.index("stub trainer starting")


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


def _silent_cfg(tmp_path: Path, task: str = "sft", backend: str = "transformers") -> SoupConfig:
    return SoupConfig.model_validate(
        {
            "base": "owner/base",
            "task": task,
            "backend": backend,
            "data": {"train": "data.jsonl", "format": "alpaca", "val_split": 0},
            "training": {"epochs": 1, "quantization": "none", "batch_size": 1},
            "output": str(tmp_path / "out"),
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
