"""`soup registry push --data`: record the training input file's hash on a run's entry (#1764)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.hashing import hash_entry, hash_file
from tests.conftest import strip_ansi

runner = CliRunner()

CONFIG = {"base": "llama", "task": "sft"}


def _finished_run(tmp_path: Path) -> str:
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    try:
        run_id = tracker.start_run(
            config_dict=CONFIG, device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
    finally:
        tracker.close()
    return run_id


@pytest.fixture()
def work(tmp_path, monkeypatch):
    """A working directory with isolated experiment and registry databases."""
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    monkeypatch.chdir(cwd)
    return cwd


def _push(run_id: str, *extra: str):
    return runner.invoke(
        app, ["registry", "push", "--run-id", run_id, "--name", "chat", "--tag", "v1", *extra],
    )


def _entry():
    from soup_cli.registry.store import RegistryStore

    with RegistryStore() as store:
        return store.get(store.resolve("chat:v1"))


def test_push_with_data_records_the_file_hash(work):
    run_id = _finished_run(work.parent)
    data = work / "train.jsonl"
    data.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")

    result = _push(run_id, "--data", "train.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    entry = _entry()
    assert entry["data_hash"] == hash_file(str(data))
    assert entry["entry_hash"] == hash_entry(
        config=CONFIG, data_path=str(data), base_model="llama",
    )
    assert entry["entry_hash"] != hash_entry(config=CONFIG, data_path=None, base_model="llama")


def test_push_without_data_is_unchanged(work):
    run_id = _finished_run(work.parent)

    result = _push(run_id)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    entry = _entry()
    assert entry["data_hash"] is None
    assert entry["entry_hash"] == hash_entry(config=CONFIG, data_path=None, base_model="llama")


def test_show_prints_the_data_hash(work):
    run_id = _finished_run(work.parent)
    data = work / "train.jsonl"
    data.write_text('{"text": "a"}\n', encoding="utf-8")
    assert _push(run_id, "--data", "train.jsonl").exit_code == 0

    result = runner.invoke(app, ["registry", "show", "chat:v1"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "data_hash" in result.output
    assert hash_file(str(data))[:16] in result.output


def test_show_marks_a_missing_data_hash(work):
    run_id = _finished_run(work.parent)
    assert _push(run_id).exit_code == 0

    result = runner.invoke(app, ["registry", "show", "chat:v1"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    row = next(line for line in result.output.splitlines() if "data_hash" in line)
    assert "-" in row.replace("data_hash", "")


def _assert_refused_before_the_registry_opens(result, work, needle: str):
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert needle in " ".join(strip_ansi(result.output).split())
    assert not (work.parent / "reg.db").exists()


def test_missing_data_file_is_refused(work):
    run_id = _finished_run(work.parent)

    result = _push(run_id, "--data", "nope.jsonl")

    _assert_refused_before_the_registry_opens(result, work, "not found")


def test_data_directory_is_refused(work):
    run_id = _finished_run(work.parent)
    (work / "data").mkdir()

    result = _push(run_id, "--data", "data")

    _assert_refused_before_the_registry_opens(result, work, "not a file")


@pytest.mark.parametrize("spelling", ["absolute", "relative"])
def test_data_outside_the_working_directory_is_refused(work, spelling):
    run_id = _finished_run(work.parent)
    outside = work.parent / "outside.jsonl"
    outside.write_text('{"text": "a"}\n', encoding="utf-8")
    path = str(outside) if spelling == "absolute" else os.path.join("..", "outside.jsonl")

    result = _push(run_id, "--data", path)

    _assert_refused_before_the_registry_opens(result, work, "under the current working directory")


@pytest.mark.requires_symlink
def test_symlink_to_a_file_outside_is_refused(work):
    run_id = _finished_run(work.parent)
    outside = work.parent / "outside.jsonl"
    outside.write_text('{"text": "a"}\n', encoding="utf-8")
    (work / "train.jsonl").symlink_to(outside)

    result = _push(run_id, "--data", "train.jsonl")

    _assert_refused_before_the_registry_opens(result, work, "under the current working directory")


def test_register_from_run_takes_an_optional_data_path(work):
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.registry.store import RegistryStore

    run_id = _finished_run(work.parent)
    data = work / "train.jsonl"
    data.write_text('{"text": "a"}\n', encoding="utf-8")
    tracker = ExperimentTracker()
    try:
        with RegistryStore() as store:
            with_data = store.register_from_run(
                tracker, run_id, name="chat", tag="v1", data_path=str(data),
            )
            without = store.register_from_run(tracker, run_id, name="chat", tag="v2")
            assert store.get(with_data)["data_hash"] == hash_file(str(data))
            assert store.get(without)["data_hash"] is None
    finally:
        tracker.close()


def test_help_describes_what_data_records():
    result = runner.invoke(app, ["registry", "push", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = " ".join(strip_ansi(result.output).split())
    assert "--data" in text
    assert "training input" in text
    assert "not proof" in text

