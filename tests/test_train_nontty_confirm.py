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

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer

from soup_cli.cli import app
from soup_cli.utils import launcher, topology
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

_DISTRIBUTED_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "ACCELERATE_MIXED_PRECISION",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
)


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


def _setup_multi_gpu_env(monkeypatch, gpu_count: int = 3) -> None:
    for var in _DISTRIBUTED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        topology,
        "detect_topology",
        lambda: {"gpu_count": gpu_count, "interconnect": "pcie"},
    )
    monkeypatch.setattr(topology, "resolve_num_gpus", lambda spec: int(spec))


def test_train_gpus_on_tty_asks_confirm_once_and_passes_yes_to_ranks(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    """#2239: `soup train --gpus N` without `--yes` asks the confirmation once
    in the launching process and passes `--yes` in the re-exec argv so the
    spawned ranks do not prompt again.
    """
    _setup_multi_gpu_env(monkeypatch, gpu_count=3)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: True)
    config = _write_config(tmp_path)

    prompts: list[tuple[str, str]] = []
    launched_argvs: list[list[str]] = []

    def _record_confirm(text: str, **kwargs) -> bool:
        prompts.append((os.environ.get("RANK", "launcher"), text))
        return True

    def _fake_run_launcher(argv: list[str]) -> int:
        launched_argvs.append(list(argv))
        train_args = argv[argv.index("train"):]
        for rank in ("0", "1", "2"):
            with monkeypatch.context() as rank_mp:
                rank_mp.setenv("RANK", rank)
                rank_mp.setenv("LOCAL_RANK", rank)
                rank_mp.setenv("WORLD_SIZE", "3")
                rank_res = CliRunner().invoke(app, train_args)
                assert isinstance(rank_res.exception, _DatasetLoadReachedError), (
                    _output(rank_res),
                    repr(rank_res.exception),
                )
        return 0

    monkeypatch.setattr(typer, "confirm", _record_confirm)
    monkeypatch.setattr(launcher, "run_launcher", _fake_run_launcher)

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "3"],
    )

    assert result.exit_code == 0, (_output(result), repr(result.exception))
    assert prompts == [("launcher", "Start training?")]
    assert len(launched_argvs) == 1
    assert "--yes" in launched_argvs[0]
    loader, builder = loading_spies
    assert loader.call_count == 3
    builder.assert_not_called()


def test_train_gpus_on_tty_declining_confirm_cancels_before_launcher(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    _setup_multi_gpu_env(monkeypatch, gpu_count=4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: True)
    config = _write_config(tmp_path)

    prompts: list[str] = []
    launched_argvs: list[list[str]] = []
    monkeypatch.setattr(
        typer, "confirm", lambda text, **kw: prompts.append(text) or False,
    )
    monkeypatch.setattr(
        launcher, "run_launcher", lambda argv: launched_argvs.append(list(argv)) or 0,
    )

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "4"],
    )

    output = _output(result)
    assert result.exit_code == 0, (output, repr(result.exception))
    assert prompts == ["Start training?"]
    assert "Cancelled." in output
    assert launched_argvs == []
    loader, builder = loading_spies
    loader.assert_not_called()
    builder.assert_not_called()


def test_train_gpus_without_tty_and_without_yes_exits_usage_error_before_launcher(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    _setup_multi_gpu_env(monkeypatch, gpu_count=4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: False)
    config = _write_config(tmp_path)

    launched_argvs: list[list[str]] = []
    monkeypatch.setattr(
        launcher, "run_launcher", lambda argv: launched_argvs.append(list(argv)) or 0,
    )

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "4"],
    )

    output = _output(result)
    assert result.exit_code == 3, (output, repr(result.exception))
    assert "stdin is not a TTY" in output
    assert "--yes" in output
    assert launched_argvs == []
    loader, builder = loading_spies
    loader.assert_not_called()
    builder.assert_not_called()


def test_train_gpus_with_yes_or_dry_run_skips_confirm(
    tmp_path: Path, monkeypatch, loading_spies,
) -> None:
    _setup_multi_gpu_env(monkeypatch, gpu_count=4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: False)
    config = _write_config(tmp_path)

    prompts: list[str] = []
    launched_argvs: list[list[str]] = []
    monkeypatch.setattr(
        typer, "confirm", lambda text, **kw: prompts.append(text) or True,
    )
    monkeypatch.setattr(
        launcher, "run_launcher", lambda argv: launched_argvs.append(list(argv)) or 0,
    )

    yes_res = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "4", "--yes"],
    )
    assert yes_res.exit_code == 0, (_output(yes_res), repr(yes_res.exception))
    assert prompts == []
    assert len(launched_argvs) == 1
    assert "--yes" in launched_argvs[0]

    loader, builder = loading_spies
    loader.side_effect = None
    loader.return_value = {"train": [{"text": "sample"}]}

    dry_res = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "4", "--dry-run"],
    )
    assert dry_res.exit_code == 0, (_output(dry_res), repr(dry_res.exception))
    assert prompts == []
    assert len(launched_argvs) == 1
    loader.assert_called_once()
    builder.assert_not_called()


@pytest.mark.parametrize("tty", [False, True])
def test_train_gpus_no_reexec_without_yes_prints_the_command_without_asking(
    tmp_path: Path, monkeypatch, loading_spies, tty,
) -> None:
    """`--no-reexec` only prints the launch command, so the launching process
    must not ask, must not refuse on a non-TTY stdin, and launches nothing."""
    _setup_multi_gpu_env(monkeypatch, gpu_count=4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.train._stdin_isatty", lambda: tty)
    config = _write_config(tmp_path)

    prompts: list[str] = []
    launched_argvs: list[list[str]] = []
    monkeypatch.setattr(
        typer, "confirm", lambda text, **kw: prompts.append(text) or True,
    )
    monkeypatch.setattr(
        launcher, "run_launcher", lambda argv: launched_argvs.append(list(argv)) or 0,
    )

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--gpus", "4", "--no-reexec"],
    )

    output = _output(result)
    assert result.exit_code == 1, (output, repr(result.exception))
    assert "Multi-GPU launch required" in output
    assert "stdin is not a TTY" not in output
    assert prompts == []
    assert launched_argvs == []
    loader, builder = loading_spies
    loader.assert_not_called()
    builder.assert_not_called()
