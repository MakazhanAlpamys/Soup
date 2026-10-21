"""Preference task context must preserve pairs and refuse lossy loads before a model."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import data_config_for_task, load_dataset
from tests.conftest import strip_ansi

TASKS = ("dpo", "ipo", "simpo", "orpo", "bco", "preference", "reward_model")
USER = {"role": "user", "content": "Q"}
SYSTEM = {"role": "system", "content": "Be brief"}
GOOD = {"role": "assistant", "content": "good"}
BAD = {"role": "assistant", "content": "bad"}


@pytest.fixture(autouse=True)
def fixed_console_width(monkeypatch):
    from soup_cli.commands import train

    monkeypatch.setattr(train, "console", Console(width=100, color_system=None))


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("explicit_auto", [False, True])
@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("message_answers", [False, True])
def test_auto_preference_task_preserves_messages_prompt_and_both_answers(
    tmp_path, monkeypatch, task, explicit_auto, system, message_answers,
):
    monkeypatch.chdir(tmp_path)
    prompt = [SYSTEM, USER] if system else [USER]
    row = {"messages": prompt, "chosen": [GOOD] if message_answers else "good",
           "rejected": [BAD] if message_answers else "bad"}
    _write(tmp_path / "pairs.jsonl", [row])
    config = DataConfig(train="pairs.jsonl", val_split=0, **({"format": "auto"} if explicit_auto
                                                           else {}))
    resolved = data_config_for_task(config, task)
    dataset = load_dataset(resolved)
    assert dataset["train"] == [{"prompt": prompt, "chosen": [GOOD], "rejected": [BAD]}]
    assert config.format == "auto", "task resolution must not mutate the caller's config"


@pytest.mark.parametrize("task", TASKS)
def test_mixed_prompt_spellings_load_as_one_conversational_preference_layout(
    tmp_path, monkeypatch, task,
):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "pairs.jsonl", [
        {"prompt": "Q", "chosen": "good", "rejected": "bad"},
        {"messages": [USER], "chosen": "good", "rejected": "bad"},
    ])
    dataset = load_dataset(data_config_for_task(DataConfig(train="pairs.jsonl", val_split=0), task))
    expected = {"prompt": [USER], "chosen": [GOOD], "rejected": [BAD]}
    assert dataset["train"] == [expected, expected]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("prompt", ["explicit", [{"role": "user", "content": "explicit"}]])
def test_explicit_prompt_precedence_is_preserved(tmp_path, monkeypatch, task, prompt):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "pairs.jsonl", [
        {"prompt": prompt, "messages": [USER], "chosen": "good", "rejected": "bad"},
    ])
    rows = load_dataset(data_config_for_task(DataConfig(train="pairs.jsonl", val_split=0), task))
    expected = {"prompt": prompt, "chosen": "good", "rejected": "bad"}
    if isinstance(prompt, list):
        expected.update(chosen=[GOOD], rejected=[BAD])
    assert rows["train"] == [expected]


def test_sft_chatml_and_kto_resolution_keep_their_existing_shapes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "chat.jsonl", [
        {"messages": [USER, GOOD], "chosen": "good", "rejected": "bad"},
    ])
    config = DataConfig(train="chat.jsonl", val_split=0)
    assert data_config_for_task(config, "sft") is config
    rows = load_dataset(data_config_for_task(config, "sft"))["train"]
    assert rows == [{"messages": [USER, GOOD]}]
    _write(tmp_path / "kto.jsonl", [{"prompt": "Q", "completion": "good", "label": True}])
    kto = DataConfig(train="kto.jsonl", val_split=0)
    assert data_config_for_task(kto, "kto") is kto
    assert load_dataset(data_config_for_task(kto, "kto"))["train"] == [
        {"prompt": "Q", "completion": "good", "label": True},
    ]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("fmt", ["chatml", "alpaca"])
def test_cli_refuses_lossy_preference_load_before_any_trainer(
    tmp_path, monkeypatch, task, dry_run, fmt,
):
    from soup_cli.cli import app
    from soup_cli.trainer import dispatch

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data.jsonl", [
        {"messages": [USER, GOOD], "chosen": "good", "rejected": "bad"},
    ])
    (tmp_path / "soup.yaml").write_text(
        f"base: public/model\ntask: {task}\n"
        f"data: {{train: data.jsonl, format: {fmt}, val_split: 0}}\n"
        "training: {quantization: none"
        + (", preference_loss: dpo" if task == "preference" else "")
        + "}\noutput: out\n", encoding="utf-8",
    )
    build = Mock(side_effect=AssertionError("preference preflight reached a trainer"))
    monkeypatch.setattr(dispatch, "build_trainer", build)
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"]
                                + (["--dry-run"] if dry_run else []))
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (output, result.exception)
    assert fmt in output and "data.format: dpo" in output, output
    assert "chosen" in output and "rejected" in output, output
    build.assert_not_called()


@pytest.mark.parametrize("task", TASKS)
def test_cli_auto_preference_dry_run_keeps_pairs(tmp_path, monkeypatch, task):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data.jsonl", [{"messages": [USER], "chosen": "good", "rejected": "bad"}])
    (tmp_path / "soup.yaml").write_text(
        f"base: public/model\ntask: {task}\n"
        "data: {train: data.jsonl, val_split: 0}\ntraining: {quantization: none"
        + (", preference_loss: dpo" if task == "preference" else "")
        + "}\noutput: out\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, (output, result.exception)
    from soup_cli.data.loader import last_load_outcome

    assert last_load_outcome().fmt == "dpo"
    assert "Ready to train" in output


@pytest.mark.parametrize("missing", ["prompt", "chosen", "rejected", "object"])
def test_preflight_checks_validation_rows_and_names_the_split(monkeypatch, capsys, missing):
    import typer

    from soup_cli.commands.train import _validate_preference_dataset_if_applicable
    from soup_cli.data import loader

    good = {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    bad = (
        {key: value for key, value in good.items() if key != missing}
        if missing != "object" else []
    )
    monkeypatch.setattr(loader, "last_load_outcome", lambda: loader.LoadOutcome(fmt="chatml"))
    cfg = SimpleNamespace(task="dpo", data=SimpleNamespace(format="auto"))
    with pytest.raises(typer.Exit) as error:
        _validate_preference_dataset_if_applicable(cfg, {"train": [good], "val": [good, bad]})
    assert error.value.exit_code == 1
    output = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert "val row 2" in output
    assert "chatml" in output and "data.format: dpo" in output


def test_preflight_keeps_valid_pairs_and_non_preference_datasets_unchanged():
    from soup_cli.commands.train import _validate_preference_dataset_if_applicable

    pair = {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    dataset = {"train": [pair], "val": [pair]}
    cfg = SimpleNamespace(task="dpo", data=SimpleNamespace(format="dpo"))
    _validate_preference_dataset_if_applicable(cfg, dataset)
    assert dataset == {"train": [pair], "val": [pair]}
    _validate_preference_dataset_if_applicable(SimpleNamespace(task="sft"), {"train": [{}]})
