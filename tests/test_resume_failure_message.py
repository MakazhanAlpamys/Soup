"""Regression tests for the ``--resume`` failure message (#2296).

``_resolve_checkpoint`` / ``_resolve_mlx_checkpoint`` return ``None`` for
several different situations, and ``_resolve_resume_or_exit`` used to print one
flat sentence for all of them. These pin the sentence that names the path or
directory the resolver searched, for both backend layouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from rich.console import Console

import soup_cli.commands.train as train_module
from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

from .conftest import strip_ansi

PREFIX = "No checkpoint found to resume from:"


def _cfg(
    output: Path,
    backend: str = "transformers",
    experiment_name: str | None = None,
) -> SoupConfig:
    return SoupConfig(
        base="meta-llama/Llama-3.1-8B-Instruct",
        task="sft",
        backend=backend,
        data=DataConfig(train="./data/train.jsonl", format="chatml"),
        training=TrainingConfig(),
        output=str(output),
        experiment_name=experiment_name,
    )


def _refusal(monkeypatch, capsys, cfg: SoupConfig, resume: str) -> str:
    """Run the CLI helper, assert exit 1, and return the collapsed message."""
    monkeypatch.setattr(train_module, "console", Console(width=200))
    with pytest.raises(typer.Exit) as caught:
        train_module._resolve_resume_or_exit(resume, cfg)
    assert caught.value.exit_code == 1
    return " ".join(strip_ansi(capsys.readouterr().out).split())


def test_direct_checkpoint_that_does_not_exist_names_the_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    missing = tmp_path / "runs" / "does-not-exist"
    out = _refusal(monkeypatch, capsys, _cfg(tmp_path / "out"), str(missing))
    assert f"{PREFIX} '{missing}' does not exist" in out


def test_direct_checkpoint_that_is_a_file_says_not_a_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.write_text("not a directory")
    out = _refusal(monkeypatch, capsys, _cfg(tmp_path / "out"), str(checkpoint))
    assert f"{PREFIX} '{checkpoint}' is not a directory" in out


def test_auto_with_a_missing_output_directory_names_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    out = _refusal(monkeypatch, capsys, _cfg(output), "auto")
    assert f"{PREFIX} '{output}' does not exist" in out


def test_auto_with_no_checkpoint_directories_names_the_output_dir(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    out = _refusal(monkeypatch, capsys, _cfg(output), "auto")
    assert f"{PREFIX} no checkpoint-N directory in '{output}'" in out


def test_auto_with_an_experiment_name_names_the_nested_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    nested = output / "my-run"
    out = _refusal(monkeypatch, capsys, _cfg(output, experiment_name="my-run"), "auto")
    assert f"{PREFIX} '{nested}' does not exist" in out


def test_auto_with_an_experiment_name_searches_the_nested_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    (output / "my-run").mkdir(parents=True)
    out = _refusal(monkeypatch, capsys, _cfg(output, experiment_name="my-run"), "auto")
    assert f"{PREFIX} no checkpoint-N directory in '{output / 'my-run'}'" in out


def test_mlx_direct_checkpoint_that_does_not_exist_names_the_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    missing = tmp_path / "adapters.safetensors"
    out = _refusal(monkeypatch, capsys, _cfg(tmp_path / "out", backend="mlx"), str(missing))
    assert f"{PREFIX} '{missing}' does not exist" in out


def test_mlx_direct_checkpoint_that_is_a_directory_says_not_a_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    out = _refusal(monkeypatch, capsys, _cfg(tmp_path / "out", backend="mlx"), str(checkpoint))
    assert "--resume on MLX takes an adapter file" in out
    assert f"'{checkpoint}' is not a file" in out


def test_mlx_auto_with_a_missing_output_directory_names_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    out = _refusal(monkeypatch, capsys, _cfg(output, backend="mlx"), "auto")
    assert f"{PREFIX} '{output}' does not exist" in out


def test_mlx_auto_with_no_adapter_file_names_the_output_dir(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    out = _refusal(monkeypatch, capsys, _cfg(output, backend="mlx"), "auto")
    assert f"{PREFIX} no adapter file in '{output}'" in out


HOSTILE = "bad\x1b[bold red]x"


@pytest.mark.parametrize(
    ("backend", "resume", "output_name", "experiment_name"),
    [
        ("transformers", HOSTILE, "out", None),  # explicit path
        ("transformers", "auto", "out", HOSTILE),  # auto, experiment_name applied
        ("mlx", HOSTILE, "out", None),  # MLX explicit adapter path
        ("mlx", "auto", HOSTILE, None),  # MLX auto output directory
    ],
)
def test_every_branch_escapes_terminal_control_characters(
    tmp_path: Path, backend, resume, output_name, experiment_name
) -> None:
    output = tmp_path / output_name
    output.mkdir(parents=True, exist_ok=True)
    reason = train_module._resume_failure_reason(
        resume, str(output), experiment_name, backend=backend
    )
    assert "\x1b" not in reason
    assert "\\[bold red]" in reason


def test_mlx_auto_ignores_experiment_name_like_the_resolver(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    cfg = _cfg(output, backend="mlx", experiment_name="my-run")
    out = _refusal(monkeypatch, capsys, cfg, "auto")
    assert f"{PREFIX} no adapter file in '{output}'" in out


def test_auto_is_case_insensitive_like_the_resolver(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    out = _refusal(monkeypatch, capsys, _cfg(output), "AUTO")
    assert f"{PREFIX} no checkpoint-N directory in '{output}'" in out
