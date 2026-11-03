"""kto task must refuse non-kto rows before any model is loaded (#2423).

_validate_preference_dataset_if_applicable is extended to handle task: kto.
It must reject rows that do not contain {prompt, completion, label} (the three
fields the KTO trainer requires), name the detected format, name the three
required fields, and tell the user to set data.format: kto.  The check fires
on both the --dry-run path and the real-run path.  Valid kto-shaped rows
continue to work under both format: auto and format: kto.

Existing tasks (dpo, ipo, simpo, orpo, bco, preference, reward_model) keep
their existing messages unchanged.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.conftest import strip_ansi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER = {"role": "user", "content": "Q"}
GOOD = {"role": "assistant", "content": "A"}
KTO_ROW = {"prompt": "Q", "completion": "A", "label": True}
KTO_ROW_CONV = {"prompt": [USER], "completion": [GOOD], "label": True}


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _normalise(text: str) -> str:
    """Strip ANSI codes and collapse all whitespace to single spaces."""
    return " ".join(strip_ansi(text).split())


@pytest.fixture(autouse=True)
def fixed_console_width(monkeypatch):
    from soup_cli.commands import train

    monkeypatch.setattr(train, "console", Console(width=120, color_system=None))


# ---------------------------------------------------------------------------
# Unit-level tests for _validate_preference_dataset_if_applicable directly
# ---------------------------------------------------------------------------

_NON_KTO_SHAPES = [
    ("llava", {"image": "img.png", "conversations": [
        {"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "A"},
    ]}),
    ("audio", {"audio": "a.wav", "messages": [USER, GOOD]}),
    ("asr", {"audio": "a.wav", "text": "hello"}),
    ("chatml", {"messages": [USER, GOOD]}),
    ("alpaca", {"instruction": "Q", "output": "A"}),
    ("plaintext", {"text": "some text"}),
    ("dpo", {"prompt": "Q", "chosen": "A", "rejected": "B"}),
]


@pytest.mark.parametrize("fmt, row", _NON_KTO_SHAPES)
def test_validator_refuses_non_kto_row_in_train(monkeypatch, capsys, fmt, row):
    """Validator must exit(1) for each non-kto row shape in the train split."""
    import typer

    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    monkeypatch.setattr(
        loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt=fmt)
    )
    cfg = SimpleNamespace(task="kto", data=SimpleNamespace(format="auto"))
    with pytest.raises(typer.Exit) as exc_info:
        _validate_preference_dataset_if_applicable(cfg, {"train": [row]})
    assert exc_info.value.exit_code == 1

    output = _normalise(capsys.readouterr().out)
    assert "kto" in output, output
    assert fmt in output, output
    assert "data.format: kto" in output, output
    assert "prompt" in output, output
    assert "completion" in output, output
    assert "label" in output, output


@pytest.mark.parametrize("missing", ["prompt", "completion", "label", "object"])
def test_validator_refuses_a_kto_row_missing_any_field(monkeypatch, capsys, missing):
    """Dropping any single required field must trigger exit(1)."""
    import typer

    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    bad = (
        {key: value for key, value in KTO_ROW.items() if key != missing}
        if missing != "object" else []
    )
    monkeypatch.setattr(
        loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt="kto")
    )
    cfg = SimpleNamespace(task="kto", data=SimpleNamespace(format="kto"))
    with pytest.raises(typer.Exit) as exc_info:
        _validate_preference_dataset_if_applicable(cfg, {"train": [KTO_ROW, bad]})
    assert exc_info.value.exit_code == 1
    expected = (
        "Invalid kto dataset: train row 2 loaded as format 'kto' lacks prompt, "
        "completion or label. Use data.format: kto with kto rows."
    )
    assert expected in _normalise(capsys.readouterr().out)


def test_validator_names_the_split_and_row_index(monkeypatch, capsys):
    """When a bad row is in the val split the message must name 'val row N'."""
    import typer

    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    bad = {"messages": [USER, GOOD]}  # chatml-shaped, not kto
    monkeypatch.setattr(
        loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt="chatml")
    )
    cfg = SimpleNamespace(task="kto", data=SimpleNamespace(format="auto"))
    with pytest.raises(typer.Exit) as exc_info:
        _validate_preference_dataset_if_applicable(
            cfg, {"train": [KTO_ROW], "val": [KTO_ROW, bad]}
        )
    assert exc_info.value.exit_code == 1

    output = _normalise(capsys.readouterr().out)
    assert "val row 2" in output, output
    assert "chatml" in output, output


def test_validator_accepts_valid_kto_row(monkeypatch):
    """Validator must not raise for rows that have prompt, completion and label."""
    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    monkeypatch.setattr(
        loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt="kto")
    )
    cfg = SimpleNamespace(task="kto", data=SimpleNamespace(format="kto"))
    _validate_preference_dataset_if_applicable(cfg, {"train": [KTO_ROW]})


def test_validator_ignores_other_tasks_without_reading_the_load(monkeypatch):
    """Validator must return before calling last_load_outcome for unrelated tasks.

    Passes an empty LoadOutcome (no fmt) and a config without data so that any
    attempt to read outcome.fmt or cfg.data.format raises AttributeError.
    """
    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    monkeypatch.setattr(loader, "last_load_outcome", lambda: loader.LoadOutcome())
    _validate_preference_dataset_if_applicable(
        SimpleNamespace(task="sft"),
        {"train": [{"messages": [USER, GOOD]}]},
    )


@pytest.mark.parametrize(
    "task",
    ["dpo", "ipo", "simpo", "orpo", "bco", "preference", "reward_model"],
)
def test_validator_still_guards_pair_tasks_with_existing_message(
    monkeypatch, capsys, task
):
    """Existing pair-task guard must be untouched by the kto addition.

    The error message for pair tasks must still say 'data.format: dpo' and name
    'chosen' and 'rejected'; it must NOT say 'data.format: kto'.
    """
    import typer

    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    bad = {"messages": [USER, GOOD]}
    monkeypatch.setattr(
        loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt="chatml")
    )
    cfg = SimpleNamespace(task=task, data=SimpleNamespace(format="auto"))
    with pytest.raises(typer.Exit) as exc_info:
        _validate_preference_dataset_if_applicable(cfg, {"train": [bad]})
    assert exc_info.value.exit_code == 1

    output = _normalise(capsys.readouterr().out)
    assert "data.format: dpo" in output, output
    assert "chosen" in output and "rejected" in output, output
    assert "data.format: kto" not in output, output


# ---------------------------------------------------------------------------
# CLI-level tests (--dry-run and real-run paths)
# ---------------------------------------------------------------------------

_NON_KTO_SHAPES_CLI = [
    ("llava", {"image": "img.png", "conversations": [
        {"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "A"},
    ]}),
    ("audio", {"audio": "a.wav", "messages": [USER, GOOD]}),
    ("asr", {"audio": "a.wav", "text": "hello"}),
    ("chatml", {"messages": [USER, GOOD]}),
    ("alpaca", {"instruction": "Q", "output": "A"}),
    ("plaintext", {"text": "some text"}),
    ("dpo", {"prompt": "Q", "chosen": "A", "rejected": "B"}),
]


@pytest.mark.parametrize("fmt, row", _NON_KTO_SHAPES_CLI)
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_refuses_non_kto_rows_before_trainer(
    tmp_path, monkeypatch, fmt, row, dry_run
):
    """CLI must exit(1) with a helpful message before build_trainer is called."""
    from soup_cli.cli import app
    from soup_cli.trainer import dispatch

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data.jsonl", [row])
    (tmp_path / "soup.yaml").write_text(
        "base: public/model\ntask: kto\n"
        "data: {train: data.jsonl, format: auto, val_split: 0}\n"
        "training: {quantization: none}\noutput: out\n",
        encoding="utf-8",
    )
    build = Mock(side_effect=AssertionError("kto preflight reached a trainer"))
    monkeypatch.setattr(dispatch, "build_trainer", build)

    args = ["train", "--config", "soup.yaml", "--yes"]
    if dry_run:
        args.append("--dry-run")
    result = CliRunner().invoke(app, args)
    output = _normalise(result.output)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert (
        f"train row 1 loaded as format '{fmt}' lacks prompt, completion or label. "
        "Use data.format: kto with kto rows."
    ) in output, output
    build.assert_not_called()


@pytest.mark.parametrize("fmt, kto_row", [
    ("auto", KTO_ROW),
    ("kto", KTO_ROW),
    ("auto", KTO_ROW_CONV),
    ("kto", KTO_ROW_CONV),
])
def test_cli_kto_dry_run_passes_with_valid_rows(tmp_path, monkeypatch, fmt, kto_row):
    """--dry-run must succeed for string and conversational kto rows under auto and kto."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data.jsonl", [kto_row, kto_row])
    (tmp_path / "soup.yaml").write_text(
        f"base: public/model\ntask: kto\n"
        f"data: {{train: data.jsonl, format: {fmt}, val_split: 0}}\n"
        "training: {quantization: none}\noutput: out\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    output = _normalise(result.output)
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "Ready to train" in output
