"""Regression tests for #1913 Finding C: with stdin not a TTY and no `--yes`,
`soup train` printed the interactive `Start training? [Y/n]:` confirm and
stalled — a pipe waits until its writer closes (EOF → Aborted) and an
unattended service waits forever.

After the fix the run refuses up front with a diagnostic that names `--yes`.
The pre-fix behaviour of answering the confirm by piping `y` into stdin is
intentionally not preserved: a non-interactive stdin now always refuses, so
the escape hatch is the flag.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from soup_cli.cli import app
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi


class _DatasetLoadReachedError(Exception):
    """Stop before dataset loading can perform any work."""


@pytest.fixture
def loading_spies(monkeypatch):
    loader = Mock(side_effect=_DatasetLoadReachedError)
    builder = Mock(side_effect=AssertionError("The trainer must not be built"))
    monkeypatch.setattr("soup_cli.commands.train.load_dataset", loader)
    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", builder)
    return loader, builder


def _output(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _write_config(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    row = '{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}'
    train.write_text(
        row + "\n" + row + "\n", encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        "base: some-model\ntask: sft\nbackend: transformers\n"
        f"data:\n  train: {train.as_posix()}\n  format: chatml\n"
        "  val_split: 0\n"
        "training:\n  quantization: none\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def test_train_without_tty_and_without_yes_exits_usage_error(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: False)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["train", "--config", str(config)])

    output = _output(result)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "stdin is not a TTY" in output
    assert "--yes" in output
    assert "Start training?" not in output
    loader, builder = loading_spies
    loader.assert_not_called()
    builder.assert_not_called()


def test_train_with_yes_on_nontty_skips_the_confirm(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: False)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--yes"],
    )

    output = _output(result)
    loader, builder = loading_spies
    loader.assert_called_once()
    assert isinstance(result.exception, _DatasetLoadReachedError), (output, repr(result.exception))
    assert "stdin is not a TTY" not in output
    assert "Start training?" not in output
    builder.assert_not_called()


def test_train_dry_run_on_nontty_without_yes_validates(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: False)
    config = _write_config(tmp_path)
    loader, builder = loading_spies
    loader.side_effect = None
    loader.return_value = {"train": [{"text": "sample"}]}

    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])

    output = _output(result)
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "Config valid" in output
    assert "stdin is not a TTY" not in output
    assert "Start training?" not in output
    loader.assert_called_once()
    builder.assert_not_called()


def test_train_on_tty_answers_confirm_normally(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    import soup_cli.commands.train as train_mod

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_mod, "_stdin_isatty", lambda: True)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        app, ["train", "--config", str(config)], input="n\n",
    )

    output = _output(result)
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "Start training?" in output
    assert "Cancelled." in output
    loader, builder = loading_spies
    loader.assert_not_called()
    builder.assert_not_called()
