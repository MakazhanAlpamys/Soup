"""Compare persisted local experiments without requiring a registry push (#2675)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi


@pytest.fixture
def tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=240))
    tracker = ExperimentTracker(db_path=tmp_path / "runs.db")
    try:
        yield tracker
    finally:
        tracker.close()


def _start(tracker: ExperimentTracker, run_id: str, config: dict) -> str:
    return tracker.start_run(
        config_dict=config, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id,
    )


def _compare(first: str, second: str) -> str:
    result = CliRunner().invoke(app, ["runs", "compare", first, second])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return re.sub(r"\s+", " ", strip_ansi(result.output))


def test_dropout_only_difference_is_visible_in_real_cli(tracker):
    _start(tracker, "first", {"training": {"lora": {"dropout": 0.05}}})
    _start(tracker, "second", {"training": {"lora": {"dropout": 0.1}}})
    text = _compare("fir", "sec")
    assert "Run Comparison" in text and "LoRA alpha" in text
    assert "Config Differences" in text
    assert re.search(r"training.lora.dropout.*changed.*0.05.*0.1", text)


def test_nested_mapping_lists_and_missing_are_distinct_from_null(tracker):
    _start(tracker, "first", {"training": {"seed": 0, "lora": {"targets": ["q", "v"]}},
                              "removed": None})
    _start(tracker, "second", {"training": {"seed": 42, "lora": {"targets": ["k"]}},
                               "added": None})
    text = _compare("first", "second")
    assert "training.seed" in text and "training.lora.targets" in text
    assert '["q", "v"]' in text and '["k"]' in text
    assert re.search(r"added.*not stored.*<not stored>.*null", text)
    assert re.search(r"removed.*not stored.*null.*<not stored>", text)


def test_empty_mapping_is_a_real_config_difference(tracker):
    _start(tracker, "first", {"removed": {}})
    _start(tracker, "second", {"added": {}})
    text = _compare("first", "second")
    assert re.search(r"added.*not stored.*<not stored>.*\{\}", text)
    assert re.search(r"removed.*not stored.*\{\}.*<not stored>", text)


def test_shared_registry_diff_preserves_empty_mapping_leaves():
    from soup_cli.registry.diff import ConfigChange, config_diff

    assert config_diff({}, {}) == []
    assert config_diff({}, {"nested": {}}) == [ConfigChange("nested", "added", None, {})]
    assert config_diff({"nested": {}}, {}) == [ConfigChange("nested", "removed", {}, None)]
    assert config_diff({"nested": None}, {"nested": {}}) == [
        ConfigChange("nested", "changed", None, {}),
    ]


def test_dataset_hashes_are_full_and_unknown_is_explicit(tracker):
    _start(tracker, "first", {})
    _start(tracker, "second", {})
    tracker.record_training_data_hash("first", "a" * 64)
    tracker.record_training_data_hash("second", "b" * 64)
    text = _compare("first", "second")
    assert "Dataset SHA-256" in text and "a" * 64 in text and "b" * 64 in text
    tracker.record_training_data_hash("second", None)
    assert re.search(r"Dataset SHA-256.*unknown", _compare("first", "second"))


def test_latest_stored_evaluation_scores_and_missing_side(tracker):
    _start(tracker, "first", {})
    _start(tracker, "second", {})
    for run_id, benchmark, score in [("first", "accuracy", 0.1), ("first", "accuracy", 0.4),
                                     ("second", "accuracy", 0.7), ("first", "left-only", 0.0)]:
        tracker.save_eval_result(
            model_path="local/model", benchmark=benchmark, score=score,
            details={"provenance": {}}, run_id=run_id,
        )
    # Equal timestamps must still select the latest insertion, not an arbitrary row.
    tracker._get_conn().execute("UPDATE eval_results SET created_at = '2026-11-10'")
    tracker._get_conn().commit()
    text = _compare("fir", "sec")
    assert "Evaluation Comparison" in text
    assert re.search(r"accuracy.*0.4000.*0.7000.*\+0.3000", text)
    assert re.search(r"left-only.*0.0000.*-.*-", text)


def test_matching_configs_and_absent_evaluations_are_explicit(tracker):
    _start(tracker, "first", {"training": {"seed": 7}})
    _start(tracker, "second", {"training": {"seed": 7}})
    text = _compare("first", "second")
    assert "No config differences." in text
    assert "No stored evaluation scores." in text


@pytest.mark.parametrize("stored", ["null", "broken{", "[]"])
def test_unavailable_config_is_not_reported_as_identical(tracker, stored):
    _start(tracker, "first", {})
    _start(tracker, "second", {"unique_config_key": 1})
    tracker._get_conn().execute("UPDATE runs SET config_json=? WHERE run_id='first'", (stored,))
    tracker._get_conn().commit()
    text = _compare("first", "second")
    assert "Config comparison unavailable: config for first not stored." in text
    assert "No config differences." not in text
    assert "unique_config_key" not in text


def test_config_paths_values_and_benchmarks_render_literally(tracker):
    _start(tracker, "first", {"[red]:warning:": "[bold]left[/bold]"})
    _start(tracker, "second", {"[red]:warning:": "[blue]right[/blue]"})
    tracker.save_eval_result(model_path="local", benchmark="[green]:warning:", score=0.5,
                             details={"provenance": {}}, run_id="first")
    text = _compare("first", "second")
    assert "[red]:warning:" in text and "[bold]left[/bold]" in text
    assert "[green]:warning:" in text and "[blue]right[/blue]" in text
