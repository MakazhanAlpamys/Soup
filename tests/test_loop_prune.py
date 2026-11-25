from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.loop_detach import ABSENT, RUNNING, STALE, DaemonInfo
from soup_cli.utils.loop_prune import (
    PruneEntry,
    build_prune_plan,
    daemon_is_running,
    execute_prune,
)

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _manifest(path: Path, iteration_id: str, run_id: str) -> None:
    directory = path / iteration_id
    directory.mkdir(parents=True)
    (directory / "iteration.json").write_text(json.dumps({
        "iteration_id": iteration_id,
        "run_id": run_id,
    }), encoding="utf-8")


def _stamp_oldest_first(*directories: Path) -> None:
    """Give each directory its own mtime, two seconds apart, oldest first.

    The planner ranks adapter folders by mtime. Folders created back to back can share one
    mtime on a coarse-clock filesystem (Linux CI), so a test that depends on "newest" must
    say which folder that is.
    """
    base_ns = min(path.stat().st_mtime_ns for path in directories)
    for step, path in enumerate(directories):
        stamp_ns = base_ns + step * 2_000_000_000
        os.utime(path, ns=(stamp_ns, stamp_ns))


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
    _stamp_oldest_first(root / "adapters" / "loop-train-old", root / "adapters" / "loop-train-kept")

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

    assert ".soup-loops/pairs/pairs-x.jsonl" not in {entry.path for entry in plan}


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


def test_active_and_pending_state_targets_are_never_planned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    targets = {
        "production": ("iter-production", "run-production"),
        "previous": ("iter-previous", "run-previous"),
        "canary": ("iter-canary", "run-canary"),
        "pending": ("iter-pending", "run-pending"),
        "newest": ("iter-newest", "run-newest"),
    }
    for iteration_id, run_id in targets.values():
        _manifest(root, iteration_id, run_id)
        (root / "adapters" / run_id).mkdir(parents=True)
        (root / "adapters" / f"{run_id}.recipe.json").write_text("{}", encoding="utf-8")
        (root / "verdicts" / f"{run_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "verdicts" / f"{run_id}.json").write_text("{}", encoding="utf-8")
    production = targets["production"]
    previous = targets["previous"]
    canary = targets["canary"]
    pending = targets["pending"]
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=production[0],
            previous_production_iteration_id=previous[0],
            canary_iteration={"iteration_id": canary[0], "outcome": "canary"},
            production_adapter=f".soup-loops/adapters/{production[1]}",
            previous_production_adapter=f".soup-loops/adapters/{previous[1]}",
            canary_adapter=f".soup-loops/adapters/{canary[1]}",
            pending_deploy={
                "iteration_id": pending[0],
                "train": {"adapter_path": f".soup-loops/adapters/{pending[1]}"},
            },
        ),
    )

    plan = build_prune_plan(keep_runs=1)
    paths = {entry.path for entry in plan}

    for iteration_id, run_id in list(targets.values())[:-1]:
        assert f".soup-loops/{iteration_id}" not in paths
        assert f".soup-loops/adapters/{run_id}" not in paths
        assert f".soup-loops/adapters/{run_id}.recipe.json" not in paths
        assert f".soup-loops/verdicts/{run_id}.json" not in paths


def test_pending_adapter_path_survives_when_manifest_run_id_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    newest_id = "iter-newest"
    pending_id = "iter-pending"
    _manifest(root, newest_id, "run-newest")
    _manifest(root, pending_id, "manifest-run-id")
    (root / "adapters" / "run-newest").mkdir(parents=True)
    pending_adapter = root / "adapters" / "pending-adapter-path"
    pending_adapter.mkdir(parents=True)
    unrelated_newest = root / "adapters" / "unrelated-newest"
    unrelated_newest.mkdir()
    unrelated_oldest = root / "adapters" / "unrelated-oldest"
    unrelated_oldest.mkdir()
    now_ns = pending_adapter.stat().st_mtime_ns
    os.utime(pending_adapter, ns=(now_ns, now_ns))
    os.utime(unrelated_newest, ns=(now_ns + 2_000_000_000, now_ns + 2_000_000_000))
    os.utime(unrelated_oldest, ns=(now_ns - 2_000_000_000, now_ns - 2_000_000_000))
    os.utime(root / "adapters" / "run-newest", ns=(now_ns + 4_000_000_000,
                                                     now_ns + 4_000_000_000))
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=None,
            previous_production_iteration_id=None,
            canary_iteration=None,
            production_adapter=None,
            previous_production_adapter=None,
            canary_adapter=None,
            pending_deploy={
                "iteration_id": pending_id,
                "train": {"adapter_path": f".soup-loops/adapters/{pending_adapter.name}"},
            },
        ),
    )

    result = runner.invoke(app, ["loop", "prune", "--keep-runs", "1"])

    assert result.exit_code == 0
    output = _plain(result.output)
    assert f".soup-loops/adapters/{pending_adapter.name}" not in output
    assert f".soup-loops/adapters/{unrelated_oldest.name}" in output


def test_canary_iteration_dict_preserves_iteration_adapter_and_sidecars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    canary_id = "iter-canary"
    canary_run = "run-canary"
    newest_id = "iter-newest"
    _manifest(root, canary_id, canary_run)
    _manifest(root, newest_id, "run-newest")
    (root / "adapters" / canary_run).mkdir(parents=True)
    (root / "adapters" / f"{canary_run}.recipe.json").write_text("{}")
    (root / "adapters" / f"{canary_run}.train.log").write_text("train")
    (root / "verdicts" / f"{canary_run}.json").parent.mkdir(parents=True)
    (root / "verdicts" / f"{canary_run}.json").write_text("{}")
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=None,
            previous_production_iteration_id=None,
            canary_iteration={"iteration_id": canary_id, "outcome": "canary"},
            production_adapter=None,
            previous_production_adapter=None,
            canary_adapter=f".soup-loops/adapters/{canary_run}",
            pending_deploy=None,
        ),
    )

    plan = build_prune_plan(keep_runs=1)
    planned = {entry.path for entry in plan}

    assert f".soup-loops/{canary_id}" not in planned
    assert f".soup-loops/adapters/{canary_run}" not in planned
    assert f".soup-loops/adapters/{canary_run}.recipe.json" not in planned
    assert f".soup-loops/adapters/{canary_run}.train.log" not in planned
    assert f".soup-loops/verdicts/{canary_run}.json" not in planned
    execute_prune(plan)
    assert (root / canary_id / "iteration.json").is_file()
    assert (root / "adapters" / canary_run).is_dir()
    assert (root / "adapters" / f"{canary_run}.recipe.json").is_file()
    assert (root / "adapters" / f"{canary_run}.train.log").is_file()
    assert (root / "verdicts" / f"{canary_run}.json").is_file()


@pytest.mark.parametrize("listing_order", ["ascending", "descending"])
def test_equal_mtime_adapters_are_ranked_by_name_not_listing_order(
    tmp_path, monkeypatch, listing_order
):
    monkeypatch.chdir(tmp_path)
    adapters = tmp_path / ".soup-loops" / "adapters"
    names = ["run-a", "run-b", "run-c"]
    for name in names:
        (adapters / name).mkdir(parents=True, exist_ok=True)
    tie_ns = (adapters / names[0]).stat().st_mtime_ns
    for name in names:
        os.utime(adapters / name, ns=(tie_ns, tie_ns))
    real_iterdir = Path.iterdir

    def iterdir(self):
        return iter(sorted(
            real_iterdir(self), key=lambda path: path.name,
            reverse=listing_order == "descending",
        ))

    monkeypatch.setattr(Path, "iterdir", iterdir)

    paths = {entry.path for entry in build_prune_plan(keep_runs=1)}

    assert ".soup-loops/adapters/run-c" not in paths
    assert ".soup-loops/adapters/run-a" in paths
    assert ".soup-loops/adapters/run-b" in paths


def test_unknown_adapter_and_verdict_notes_are_kept_during_apply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "run-new")
    _manifest(root, "iter-20261117T120000-00000002", "run-old")
    run_old = root / "adapters" / "run-old"
    run_new = root / "adapters" / "run-new"
    run_old.mkdir(parents=True)
    run_new.mkdir()
    oldest_ns = run_old.stat().st_mtime_ns
    os.utime(run_old, ns=(oldest_ns, oldest_ns))
    os.utime(run_new, ns=(oldest_ns + 2_000_000_000, oldest_ns + 2_000_000_000))
    adapter_note = root / "adapters" / "notes.txt"
    adapter_note.write_text("keep", encoding="utf-8")
    (root / "verdicts").mkdir()
    verdict_note = root / "verdicts" / "notes.txt"
    verdict_note.write_text("keep", encoding="utf-8")

    result = execute_prune(build_prune_plan(keep_runs=1))

    assert not (root / "adapters" / "run-old").exists()
    assert adapter_note.read_text(encoding="utf-8") == "keep"
    assert verdict_note.read_text(encoding="utf-8") == "keep"
    assert not result.failed


def test_missing_iteration_log_preserves_datasets_for_retained_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    iteration_id = "iter-20261118T120000-00000001"
    _manifest(root, iteration_id, "loop-train-kept")
    pair = root / "pairs" / "pairs-unknown.jsonl"
    pair.parent.mkdir()
    pair.write_text("data\n", encoding="utf-8")

    plan = build_prune_plan(keep_runs=1)

    assert ".soup-loops/pairs/pairs-unknown.jsonl" not in {entry.path for entry in plan}


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


def test_log_changed_after_planning_is_skipped_and_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    root.mkdir()
    log = root / "progress.jsonl"
    log.write_text("old\nnew\n", encoding="utf-8")
    entry = next(
        row for row in build_prune_plan(keep_log_lines=1)
        if row.path == ".soup-loops/progress.jsonl"
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write("concurrent append\n")

    result = execute_prune([entry], keep_log_lines=1)

    assert result.skipped and result.skipped[0].path == entry.path
    assert "concurrent append" in log.read_text(encoding="utf-8")


def test_same_size_log_rewrite_with_changed_mtime_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    root.mkdir()
    log = root / "progress.jsonl"
    log.write_text("old\nnew\n", encoding="utf-8")
    entry = next(row for row in build_prune_plan(keep_log_lines=1)
                 if row.path == ".soup-loops/progress.jsonl")
    old = log.stat()
    log.write_text("bad\nnew\n", encoding="utf-8")
    os.utime(log, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
    assert log.stat().st_size == entry.expected_size

    result = execute_prune([entry], keep_log_lines=1)

    assert result.skipped and result.skipped[0].path == entry.path
    assert "bad\nnew\n" == log.read_text(encoding="utf-8")


def test_log_size_change_with_restored_mtime_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    root.mkdir()
    log = root / "progress.jsonl"
    log.write_text("old\nnew\n", encoding="utf-8")
    entry = next(row for row in build_prune_plan(keep_log_lines=1)
                 if row.path == ".soup-loops/progress.jsonl")
    mtime_ns = log.stat().st_mtime_ns
    log.write_text("changed-size\nnew\n", encoding="utf-8")
    os.utime(log, ns=(mtime_ns, mtime_ns))
    assert log.stat().st_mtime_ns == entry.expected_mtime_ns

    result = execute_prune([entry], keep_log_lines=1)

    assert result.skipped and result.skipped[0].path == entry.path
    assert "changed-size\nnew\n" == log.read_text(encoding="utf-8")


def test_iteration_log_trim_keeps_rows_for_retained_runs_and_their_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    old_id = "iter-20261117T120000-00000002"
    recent_id = "iter-20261118T120000-00000001"
    _manifest(root, old_id, "run-old")
    _manifest(root, recent_id, "run-recent")
    pairs = root / "pairs"
    pairs.mkdir()
    (pairs / "pairs-old.jsonl").write_text("old data\n", encoding="utf-8")
    (pairs / "pairs-recent.jsonl").write_text("recent data\n", encoding="utf-8")
    rows = [
        {"iteration_id": old_id, "dataset": {"path": ".soup-loops/pairs/pairs-old.jsonl"}},
        {"iteration_id": "unrelated", "dataset": {"path": ".soup-loops/pairs/unused.jsonl"}},
        {"iteration_id": recent_id, "dataset": {"path": ".soup-loops/pairs/pairs-recent.jsonl"}},
    ]
    (root / "iterations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=old_id,
            previous_production_iteration_id=None,
            canary_iteration=None,
            production_adapter=None,
            previous_production_adapter=None,
            canary_adapter=None,
            pending_deploy=None,
        ),
    )

    plan = build_prune_plan(keep_runs=1, keep_log_lines=1)
    by_path = {entry.path: entry for entry in plan}
    log_entry = by_path[".soup-loops/iterations.jsonl"]

    assert log_entry.action == "trim"
    assert len(log_entry.replacement.splitlines()) == 2
    assert ".soup-loops/pairs/pairs-old.jsonl" not in by_path
    assert ".soup-loops/pairs/pairs-recent.jsonl" not in by_path
    assert ".soup-loops/pairs/unused.jsonl" not in by_path
    execute_prune([log_entry], keep_log_lines=1)
    stored = [json.loads(line) for line in (root / "iterations.jsonl").read_text().splitlines()]
    assert {row["iteration_id"] for row in stored} == {old_id, recent_id}


def test_cli_dry_run_explains_pairs_kept_for_missing_retained_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    retained_id = "iter-retained-no-row"
    _manifest(root, retained_id, "run-retained")
    (root / "pairs").mkdir()
    pair = root / "pairs" / "pairs-unknown.jsonl"
    pair.write_text("data\n", encoding="utf-8")
    iteration_log = root / "iterations.jsonl"
    iteration_log.write_text(
        json.dumps({"iteration_id": "other", "dataset": {"path": "ignored"}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "soup_cli.utils.loop_prune.read_state",
        lambda: SimpleNamespace(
            production_iteration_id=retained_id,
            previous_production_iteration_id=None,
            canary_iteration=None,
            production_adapter=None,
            previous_production_adapter=None,
            canary_adapter=None,
            pending_deploy=None,
        ),
    )

    result = runner.invoke(app, ["loop", "prune"])
    output = _plain(result.output)

    assert result.exit_code == 0
    assert "pairs kept because retained iteration history is incomplete" in output
    assert pair.exists()

    unused_pair = root / "pairs" / "pairs-now-unused.jsonl"
    unused_pair.write_text("stale data\n", encoding="utf-8")
    iteration_log.write_text(
        json.dumps({
            "iteration_id": retained_id,
            "dataset": {"path": ".soup-loops/pairs/pairs-unknown.jsonl"},
        }) + "\n",
        encoding="utf-8",
    )
    complete = runner.invoke(app, ["loop", "prune"])
    complete_output = _plain(complete.output)
    assert complete.exit_code == 0
    assert "pairs kept because retained iteration history is incomplete" not in complete_output
    assert ".soup-loops/pairs/pairs-now-unused.jsonl" in complete_output
    assert ".soup-loops/pairs/pairs-unknown.jsonl" not in complete_output
    assert pair.exists()


@pytest.mark.requires_symlink
def test_prune_refuses_symlinked_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    (tmp_path / ".soup-loops").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|junction"):
        build_prune_plan()


@pytest.mark.requires_symlink
def test_apply_refuses_a_symlink_nested_in_a_pruned_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "new")
    old_id = "iter-20261117T120000-00000002"
    _manifest(root, old_id, "old")
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    link = root / old_id / "escape.txt"
    link.symlink_to(outside)

    plan = build_prune_plan(keep_runs=1)
    result = execute_prune(plan)

    assert result.failed and "escape.txt" in result.failed[0].reason
    assert (root / old_id).exists()
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_apply_refuses_a_plan_entry_outside_the_loop_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")

    result = execute_prune([PruneEntry("../outside.txt", "test", False)])

    assert result.failed and "outside" in result.failed[0].reason
    assert outside.read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_apply_refuses_a_junction_nested_in_a_pruned_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "new")
    old_id = "iter-20261117T120000-00000002"
    _manifest(root, old_id, "old")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("untouched", encoding="utf-8")
    junction = root / old_id / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True, capture_output=True, text=True,
    )

    result = execute_prune(build_prune_plan(keep_runs=1))

    assert result.failed and "junction" in result.failed[0].reason
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "untouched"


def test_cli_refuses_to_prune_while_daemon_is_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    _manifest(root, "iter-20261118T120000-00000001", "loop-train-kept")
    monkeypatch.setattr("soup_cli.commands.loop.daemon_is_running", lambda: True)

    result = runner.invoke(app, ["loop", "prune", "--yes"])

    assert result.exit_code == 2
    assert "detached loop daemon is running" in _plain(result.output)
    assert (root / "iter-20261118T120000-00000001" / "iteration.json").exists()


def test_cli_dry_run_is_read_only_and_terminal_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "pairs").mkdir(parents=True)
    hostile = ".soup-loops/pairs/[bold]x.txt"
    monkeypatch.setattr(
        "soup_cli.commands.loop.build_prune_plan",
        lambda **_kwargs: [SimpleNamespace(
            path=hostile, reason="hostile \x1b[red]reason[/]", is_dir=False, action="remove"
        )],
    )

    result = runner.invoke(app, ["loop", "prune"])
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "dry run:" in output
    assert "[bold]x.txt" in output
    assert "\\[bold]" not in output
    assert "hostile [red]reason[/]" in output
    assert "\x1b" not in result.output


def test_cli_failed_line_strips_controls_and_renders_markup_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.loop.daemon_is_running", lambda: False)
    monkeypatch.setattr(
        "soup_cli.commands.loop.build_prune_plan",
        lambda **_kwargs: [PruneEntry(".soup-loops/pairs/[bold]x.txt", "remove", False)],
    )
    monkeypatch.setattr(
        "soup_cli.commands.loop.execute_prune",
        lambda *_args, **_kwargs: SimpleNamespace(
            removed=0,
            bytes_freed=0,
            skipped=[],
            failed=[SimpleNamespace(
                path=".soup-loops/pairs/[bold]x.txt",
                reason="hostile \x1b [red]reason[/]",
            )],
        ),
    )

    result = runner.invoke(app, ["loop", "prune", "--yes"])
    output = _plain(result.output)

    assert result.exit_code == 1
    assert "failed:" in output
    assert "[bold]x.txt" in output
    assert "hostile [red]reason[/]" in output
    assert "\\[bold]" not in output
    assert "\x1b" not in result.output


def test_cli_dry_run_leaves_paths_sizes_and_mtimes_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "pairs").mkdir(parents=True)
    (root / "pairs" / "pairs-unused.jsonl").write_text("data", encoding="utf-8")
    (root / "pairs" / "notes.txt").write_text("keep", encoding="utf-8")

    def snapshot():
        return {
            path.relative_to(tmp_path).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()
        }

    before = snapshot()
    result = runner.invoke(app, ["loop", "prune"])
    after = snapshot()

    assert result.exit_code == 0
    assert before == after
    assert ".soup-loops/pairs/pairs-unused.jsonl" in _plain(result.output)
    assert ".soup-loops/pairs/notes.txt" in _plain(result.output)


def test_cli_yes_removes_only_the_printed_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "pairs").mkdir(parents=True)
    remove_me = root / "pairs" / "pairs-old.jsonl"
    keep_me = root / "pairs" / "notes.txt"
    remove_me.write_text("data", encoding="utf-8")
    keep_me.write_text("note", encoding="utf-8")

    dry_run = runner.invoke(app, ["loop", "prune"])
    planned = _plain(dry_run.output)
    assert ".soup-loops/pairs/pairs-old.jsonl" in planned
    assert ".soup-loops/pairs/notes.txt" in planned

    applied = runner.invoke(app, ["loop", "prune", "--yes"])

    assert applied.exit_code == 0
    assert not remove_me.exists()
    assert keep_me.exists()


def test_cli_two_pass_prune_preserves_twenty_version_lineage_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.loop.daemon_is_running", lambda: False)
    root = tmp_path / ".soup-loops"
    version_rows = []
    iteration_ids = []
    (root / "verdicts").mkdir(parents=True)
    for index in range(1, 26):
        iteration_id = f"iter-202611{index:02d}T120000-{index:08d}"
        run_id = f"loop-train-{index:02d}"
        iteration_ids.append(iteration_id)
        _manifest(root, iteration_id, run_id)
        adapter = root / "adapters" / run_id
        adapter.mkdir(parents=True)
        (root / "adapters" / f"{run_id}.recipe.json").write_text("{}", encoding="utf-8")
        (root / "verdicts" / f"{run_id}.json").write_text("{}", encoding="utf-8")
        version_rows.append(json.dumps({"iteration_id": iteration_id}))
    version_log = root / "versions.jsonl"
    version_log.write_text("\n".join(version_rows) + "\n", encoding="utf-8")
    progress_log = root / "progress.jsonl"
    progress_log.write_text("{}\n" * 25, encoding="utf-8")

    first = runner.invoke(
        app, ["loop", "prune", "--keep-runs", "1", "--keep-log-lines", "5", "--yes"]
    )

    assert first.exit_code == 0, _plain(first.output)
    first_output = _plain(first.output)
    assert f".soup-loops/{iteration_ids[4]}" in first_output
    assert ".soup-loops/adapters/loop-train-05" in first_output
    assert ".soup-loops/adapters/loop-train-05.recipe.json" in first_output
    assert ".soup-loops/verdicts/loop-train-05.json" in first_output
    assert len(version_log.read_text(encoding="utf-8").splitlines()) == 20
    assert len(progress_log.read_text(encoding="utf-8").splitlines()) == 5
    assert all((root / iteration_id / "iteration.json").exists()
               for iteration_id in iteration_ids[5:])

    second = runner.invoke(app, ["loop", "prune", "--keep-runs", "1", "--yes"])

    assert second.exit_code == 0, _plain(second.output)
    second_output = _plain(second.output)
    assert not any(iteration_id in second_output for iteration_id in iteration_ids[5:])
    assert all((root / iteration_id / "iteration.json").exists()
               for iteration_id in iteration_ids[5:])
    assert len(version_log.read_text(encoding="utf-8").splitlines()) == 20


def test_cli_reports_partial_failures_and_continues(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".soup-loops"
    (root / "pairs").mkdir(parents=True)
    failed = root / "pairs" / "pairs-fail.jsonl"
    successful = root / "pairs" / "pairs-success.jsonl"
    failed.write_text("data", encoding="utf-8")
    successful.write_text("data", encoding="utf-8")
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name == failed.name:
            raise PermissionError("denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(failed), "unlink", unlink)
    result = runner.invoke(app, ["loop", "prune", "--yes"])
    output = _plain(result.output)

    assert result.exit_code == 1
    assert "failed:" in output and "pairs-fail.jsonl" in output and "PermissionError" in output
    assert not successful.exists()
    assert failed.exists()


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
