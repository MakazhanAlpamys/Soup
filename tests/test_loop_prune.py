from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.loop_prune import build_prune_plan, execute_prune

runner = CliRunner()


def _manifest(path: Path, iteration_id: str, run_id: str) -> None:
    directory = path / iteration_id
    directory.mkdir(parents=True)
    (directory / "iteration.json").write_text(json.dumps({
        "iteration_id": iteration_id,
        "run_id": run_id,
    }), encoding="utf-8")


def test_plan_keeps_runs_and_pair_refs_then_applies_only_unreferenced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "loop-train-kept")
    _manifest(root, "iter-20261117T120000-00000002", "loop-train-old")
    (root / "adapters" / "loop-train-old").mkdir(parents=True)
    (root / "adapters" / "loop-train-kept").mkdir()
    (root / "pairs").mkdir()
    kept_pair = "pairs-a.jsonl"
    removed_pair = "pairs-b.jsonl"
    (root / "pairs" / kept_pair).write_text("kept\n", encoding="utf-8")
    (root / "pairs" / removed_pair).write_text("old\n", encoding="utf-8")
    (root / "iterations.jsonl").write_text(
        json.dumps({"iteration_id": "iter-20261118T120000-00000001",
                    "dataset": {"path": f".soup-loops/pairs/{kept_pair}"}}) + "\n",
        encoding="utf-8",
    )

    plan = build_prune_plan(keep_runs=1, keep_log_lines=1)

    paths = {entry.path for entry in plan}
    assert ".soup-loops/iter-20261117T120000-00000002" in paths
    assert ".soup-loops/adapters/loop-train-old" in paths
    assert ".soup-loops/pairs/pairs-b.jsonl" in paths
    assert ".soup-loops/pairs/pairs-a.jsonl" not in paths
    assert (root / "iter-20261117T120000-00000002").exists()  # planning is read-only

    execute_prune(plan, keep_log_lines=1)

    assert not (root / "iter-20261117T120000-00000002").exists()
    assert not (root / "adapters" / "loop-train-old").exists()
    assert not (root / "pairs" / removed_pair).exists()
    assert (root / "pairs" / kept_pair).exists()


def test_bad_retained_iteration_line_fails_closed_for_pair_deletion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "pairs").mkdir(parents=True)
    pair = root / "pairs" / "pairs-x.jsonl"
    pair.write_text("data\n", encoding="utf-8")
    (root / "iterations.jsonl").write_bytes(b"not json\n")

    plan = build_prune_plan(keep_log_lines=1)

    assert str(pair) not in {entry.path for entry in plan}


def test_latest_version_transitions_keep_their_old_run_and_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    old_id = "iter-20261117T120000-00000002"
    new_id = "iter-20261118T120000-00000001"
    _manifest(root, old_id, "loop-train-old")
    _manifest(root, new_id, "loop-train-new")
    (root / "adapters" / "loop-train-old").mkdir(parents=True)
    (root / "adapters" / "loop-train-new").mkdir()
    (root / "versions.jsonl").write_text(
        json.dumps({"iteration_id": old_id}) + "\n",
        encoding="utf-8",
    )

    plan = build_prune_plan(keep_runs=1)

    paths = {entry.path for entry in plan}
    assert f".soup-loops/{old_id}" not in paths
    assert ".soup-loops/adapters/loop-train-old" not in paths


def test_bad_version_history_keeps_all_run_and_adapter_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    old_id = "iter-20261117T120000-00000002"
    new_id = "iter-20261118T120000-00000001"
    _manifest(root, old_id, "loop-train-old")
    _manifest(root, new_id, "loop-train-new")
    (root / "adapters" / "loop-train-old").mkdir(parents=True)
    (root / "adapters" / "loop-train-new").mkdir()
    (root / "versions.jsonl").write_text("not json\n", encoding="utf-8")

    plan = build_prune_plan(keep_runs=1)

    paths = {entry.path for entry in plan}
    assert f".soup-loops/{old_id}" not in paths
    assert f".soup-loops/{new_id}" not in paths
    assert ".soup-loops/adapters/loop-train-old" not in paths
    assert ".soup-loops/adapters/loop-train-new" not in paths


def test_log_plan_keeps_only_the_requested_recent_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    root.mkdir()
    log = root / "progress.jsonl"
    log.write_text("old-1\nold-2\nrecent-1\nrecent-2\n", encoding="utf-8")

    plan = build_prune_plan(keep_log_lines=2)
    log_entry = next(entry for entry in plan if entry.path == ".soup-loops/progress.jsonl")

    assert log_entry.reason.startswith("trim 2 old log lines")
    execute_prune([log_entry], keep_log_lines=2)

    assert log.read_text(encoding="utf-8") == "recent-1\nrecent-2\n"


def test_prune_refuses_symlinked_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    try:
        (tmp_path / ".soup-loops").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is not permitted on this Windows host: {exc}")

    with pytest.raises(ValueError, match="symlink|junction"):
        build_prune_plan()


def test_cli_refuses_to_prune_while_daemon_is_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "loop-train-kept")
    monkeypatch.setattr("soup_cli.commands.loop.daemon_is_running", lambda: True)

    result = runner.invoke(app, ["loop", "prune", "--yes"])

    assert result.exit_code == 2
    assert "daemon is running" in result.output
    assert (root / "iter-20261118T120000-00000001" / "iteration.json").exists()

