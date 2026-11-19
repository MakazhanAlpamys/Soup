from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.loop_detach import ABSENT, RUNNING, STALE, DaemonInfo
from soup_cli.utils.loop_prune import build_prune_plan, daemon_is_running, execute_prune

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


@pytest.mark.parametrize("prefix", ["pairs", "kto", "sft"])
def test_retained_dataset_refs_survive_log_trimming_for_each_format(
    tmp_path, monkeypatch, prefix
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    iteration_id = "iter-20261118T120000-00000001"
    _manifest(root, iteration_id, "loop-train-kept")
    (root / "pairs").mkdir()
    referenced = f"{prefix}-kept.jsonl"
    (root / "pairs" / referenced).write_text("data\n", encoding="utf-8")
    (root / "pairs" / "pairs-unused.jsonl").write_text("old\n", encoding="utf-8")
    (root / "iterations.jsonl").write_text(
        json.dumps({"iteration_id": iteration_id,
                    "dataset": {"path": f".soup-loops/pairs/{referenced}"}}) + "\n"
        + json.dumps({"iteration_id": "old", "dataset": {"path": ".soup-loops/pairs/old.jsonl"}})
        + "\n",
        encoding="utf-8",
    )

    plan = build_prune_plan(keep_runs=1, keep_log_lines=1)
    entries = {entry.path: entry for entry in plan}
    paths = set(entries)

    assert f".soup-loops/pairs/{referenced}" not in paths
    assert ".soup-loops/pairs/pairs-unused.jsonl" in paths
    assert "not referenced by retained iteration log lines" in entries[
        ".soup-loops/pairs/pairs-unused.jsonl"
    ].reason


def test_missing_retained_manifest_preserves_all_adapter_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "iter-20261118T120000-00000001").mkdir(parents=True)
    (root / "adapters" / "loop-train-old").mkdir(parents=True)
    (root / "verdicts").mkdir()
    (root / "verdicts" / "loop-train-old.json").write_text("{}", encoding="utf-8")
    (root / "versions.jsonl").write_text(
        json.dumps({"iteration_id": "iter-20261118T120000-00000001"}) + "\n",
        encoding="utf-8",
    )

    plan = build_prune_plan(keep_runs=1)
    paths = {entry.path for entry in plan}

    assert ".soup-loops/adapters/loop-train-old" not in paths
    assert ".soup-loops/verdicts/loop-train-old.json" not in paths


def test_missing_iteration_log_preserves_datasets_for_retained_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    iteration_id = "iter-20261118T120000-00000001"
    _manifest(root, iteration_id, "loop-train-kept")
    pair = root / "pairs" / "pairs-unknown.jsonl"
    pair.parent.mkdir()
    pair.write_text("data\n", encoding="utf-8")

    plan = build_prune_plan(keep_runs=1)

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


def test_prune_keeps_current_rollback_targets_outside_recent_version_window(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    rollback_id = "iter-20261001T120000-00000001"
    transition_ids = [
        f"iter-202611{day:02}T120000-{day:08}" for day in range(1, 22)
    ]
    current_id = transition_ids[-1]
    for iteration_id in transition_ids:
        run_id = f"adapter-{iteration_id[-8:]}"
        _manifest(root, iteration_id, run_id)
        (root / "adapters" / run_id).mkdir(parents=True)
    _manifest(root, rollback_id, "adapter-rollback")
    (root / "adapters" / "adapter-rollback").mkdir()
    versions = [json.dumps({"iteration_id": iteration_id}) for iteration_id in transition_ids]
    (root / "versions.jsonl").write_text("\n".join(versions) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=current_id,
            previous_production_iteration_id=rollback_id,
            canary_iteration=None,
            production_adapter=".soup-loops/adapters/adapter-current",
            previous_production_adapter=".soup-loops/adapters/adapter-rollback",
            canary_adapter=None,
        ),
    )

    plan = build_prune_plan(keep_runs=1)
    paths = {entry.path for entry in plan}

    assert ".soup-loops/iter-20261001T120000-00000001" not in paths
    assert f".soup-loops/adapters/adapter-{transition_ids[0][-8:]}" in paths
    assert f".soup-loops/adapters/adapter-{current_id[-8:]}" not in paths
    assert ".soup-loops/adapters/adapter-rollback" not in paths

    execute_prune(plan)

    assert (root / "adapters" / "adapter-rollback").is_dir()
    assert (root / "adapters" / f"adapter-{current_id[-8:]}").is_dir()
    assert not (root / "adapters" / f"adapter-{transition_ids[0][-8:]}").exists()


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


@pytest.mark.parametrize(
    ("state", "expected"),
    [(RUNNING, True), (STALE, False), (ABSENT, False)],
)
def test_prune_daemon_guard_only_blocks_a_running_pid_file(monkeypatch, state, expected):
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.inspect_pid_file",
        lambda _path=None: DaemonInfo(state),
    )

    assert daemon_is_running() is expected
