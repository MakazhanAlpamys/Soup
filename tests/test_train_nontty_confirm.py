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

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


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
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["train", "--config", str(config)])

    output = strip_ansi(result.output)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "stdin is not a TTY" in output
    assert "--yes" in output
    assert "Start training?" not in output


def test_train_with_yes_on_nontty_skips_the_confirm(
    tmp_path: Path, monkeypatch,
) -> None:
    # Control for the "refuse even when --yes is passed" mutation: --yes on a
    # non-TTY stdin must get past the TTY guard without the diagnostic.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        app, ["train", "--config", str(config), "--yes", "--dry-run"],
    )

    output = strip_ansi(result.output)
    assert "stdin is not a TTY" not in output


def test_train_on_tty_answers_confirm_normally(
    tmp_path: Path, monkeypatch,
) -> None:
    # Control for the "refuse even on a TTY" mutation: fake a TTY via the
    # _stdin_isatty helper seam and answer the confirm with "n" — the run
    # must stop via the normal Cancelled path with exit 0.
    import soup_cli.commands.train as train_mod

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_mod, "_stdin_isatty", lambda: True)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(
        app, ["train", "--config", str(config)], input="n\n",
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0
    assert "Cancelled." in output
