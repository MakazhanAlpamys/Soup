"""#1773 — refuse non-finite baseline evidence at read, write, and gate boundaries.

An unusable baseline (NaN, -inf, or +inf) must be refused before evaluation or reporting
a verdict, rather than silently passing the gate or misclassifying an invalid input
as a regression.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.gate import (
    EvalSuite,
    GateTask,
    resolve_baseline,
    run_gate,
    stamp_baseline_scores,
    write_baseline_file,
)
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

runner = CliRunner()


def _make_suite_and_tasks(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"prompt": "control", "expected": "gold", "scoring": "exact"}) + "\n",
        encoding="utf-8",
    )
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(
        yaml.safe_dump({
            "suite": "finite-baseline-probe",
            "tasks": [
                {
                    "type": "custom",
                    "name": "control",
                    "threshold": 0.0,
                    "tasks": str(tasks_file),
                    "scorer": "exact",
                }
            ],
        }),
        encoding="utf-8",
    )
    return suite_file, tasks_file


class TestWriterHelperRefusal:
    """stamp_baseline_scores and write_baseline_file must reject non-finite scores."""

    @pytest.mark.parametrize("bad_val", [float("nan"), float("-inf"), float("inf")])
    def test_stamp_baseline_scores_rejects_non_finite(self, bad_val):
        with pytest.raises(ValueError, match="control") as exc_info:
            stamp_baseline_scores({"control": bad_val})
        assert "finite" in str(exc_info.value).lower()

    def test_stamp_baseline_scores_names_offending_task_in_multi_entry_map(self):
        with pytest.raises(ValueError, match="bad_task") as exc_info:
            stamp_baseline_scores({"ok1": 0.5, "bad_task": float("nan"), "ok2": 0.9})
        assert "finite" in str(exc_info.value).lower()

    @pytest.mark.parametrize("bad_val", [float("nan"), float("-inf"), float("inf")])
    def test_write_baseline_file_rejects_non_finite_and_creates_no_file(self, tmp_path, bad_val):
        target = tmp_path / "out.json"
        with pytest.raises(ValueError, match="control"):
            write_baseline_file(str(target), {"control": bad_val})
        assert not target.exists()


class TestReaderRefusal:
    """resolve_baseline must reject non-finite scores in files and registry."""

    @pytest.mark.parametrize("raw_num", ["-1e999", "1e999", "NaN", "Infinity", "-Infinity"])
    def test_resolve_baseline_legacy_file_rejects_non_finite(self, tmp_path, monkeypatch, raw_num):
        monkeypatch.chdir(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text(f'{{"control": {raw_num}}}', encoding="utf-8")
        with pytest.raises(ValueError, match="control") as exc_info:
            resolve_baseline(str(base_file.name))
        assert "finite" in str(exc_info.value).lower()

    @pytest.mark.parametrize("raw_num", ["-1e999", "1e999", "NaN", "Infinity", "-Infinity"])
    def test_resolve_baseline_stamped_envelope_rejects_non_finite(
        self, tmp_path, monkeypatch, raw_num
    ):
        monkeypatch.chdir(tmp_path)
        base_file = tmp_path / "baseline_stamped.json"
        base_file.write_text(
            f'{{"scores": {{"control": {raw_num}}}, '
            f'"provenance": {{"soup_version": "x", "scorer_revision": 1}}}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="control") as exc_info:
            resolve_baseline(str(base_file.name))
        assert "finite" in str(exc_info.value).lower()

    @pytest.mark.parametrize("bad_score", [float("-inf"), float("inf")])
    def test_resolve_baseline_registry_rejects_infinite_db_rows(
        self, tmp_path, monkeypatch, bad_score
    ):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            {"base": "m", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        conn = tracker._get_conn()
        conn.execute(
            """INSERT INTO eval_results
               (run_id, model_path, benchmark, score, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, "m", "control", bad_score, json.dumps({}), "2026-01-01T00:00:00"),
        )
        conn.commit()

        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(
            name="base", tag="v1", base_model="m", task="sft", run_id=run_id, config={},
        )
        store.close()

        with pytest.raises(ValueError, match="control") as exc_info:
            resolve_baseline(f"registry://{eid}")
        assert "finite" in str(exc_info.value).lower()

    def test_resolve_baseline_registry_rejects_nan_row(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.__enter__.return_value = mock_store
        mock_store.resolve.return_value = "entry_123"
        mock_store.get_eval_results.return_value = [
            {
                "benchmark": "control",
                "score": float("nan"),
                "created_at": "2026-01-01T00:00:00",
                "details_json": "{}",
            }
        ]

        monkeypatch.setattr("soup_cli.registry.store.RegistryStore", lambda **kw: mock_store)

        with pytest.raises(ValueError, match="control") as exc_info:
            resolve_baseline("registry://entry_123")
        assert "finite" in str(exc_info.value).lower()


class TestGateExecutionRefusal:
    """run_gate must refuse non-finite baseline values."""

    @pytest.mark.parametrize("bad_val", [float("nan"), float("-inf"), float("inf")])
    def test_run_gate_rejects_non_finite_baseline(self, tmp_path, bad_val):
        suite = EvalSuite(
            suite="probe",
            tasks=[
                GateTask(
                    type="custom",
                    name="control",
                    threshold=0.0,
                    tasks=str(tmp_path / "dummy.jsonl"),
                    scorer="exact",
                )
            ],
        )
        with pytest.raises(ValueError, match="control") as exc_info:
            run_gate(suite, generate_fn=lambda p: "", baseline={"control": bad_val})
        assert "finite" in str(exc_info.value).lower()

    def test_run_gate_rejects_non_mapping_baseline(self):
        suite = EvalSuite(suite="probe", tasks=[])
        with pytest.raises(TypeError, match="mapping"):
            run_gate(suite, generate_fn=lambda p: "", baseline="not-a-mapping")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", ["not_a_number", None, True, False, []])
    def test_run_gate_rejects_non_number_score_with_type_error(self, bad_val):
        suite = EvalSuite(suite="probe", tasks=[])
        with pytest.raises(TypeError, match="must be a number"):
            run_gate(suite, generate_fn=lambda p: "", baseline={"control": bad_val})  # type: ignore[arg-type]

    def test_run_gate_names_specific_bad_task_in_multi_entry_baseline(self, tmp_path):
        suite = EvalSuite(
            suite="probe",
            tasks=[
                GateTask(
                    type="custom",
                    name="bad_entry",
                    threshold=0.0,
                    tasks=str(tmp_path / "dummy.jsonl"),
                    scorer="exact",
                )
            ],
        )
        with pytest.raises(ValueError, match="bad_entry") as exc_info:
            run_gate(
                suite,
                generate_fn=lambda p: "",
                baseline={"ok_entry": 0.5, "bad_entry": float("nan")},
            )
        assert "finite" in str(exc_info.value).lower()


class TestCliNonSuccess:
    """soup eval gate must refuse non-finite baselines with exit code 3 (EXIT_USAGE_ERROR)."""

    def test_cli_refuses_negative_infinity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"control": -1e999}', encoding="utf-8")

        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "GATE PASSED" not in output
        assert "control" in output
        assert "finite" in output.lower()

    def test_cli_refuses_nan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"control": NaN}', encoding="utf-8")

        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "GATE PASSED" not in output
        assert "control" in output
        assert "finite" in output.lower()

    def test_cli_refuses_positive_infinity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"control": 1e999}', encoding="utf-8")

        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "GATE PASSED" not in output
        assert "control" in output
        assert "finite" in output.lower()

    def test_cli_refuses_stamped_envelope_with_infinity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline_stamped.json"
        base_file.write_text(
            '{"scores": {"control": -1e999}, "provenance": {"scorer_revision": 1}}',
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "GATE PASSED" not in output
        assert "control" in output
        assert "finite" in output.lower()

    def test_cli_refuses_registry_with_non_finite_score(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.__enter__.return_value = mock_store
        mock_store.resolve.return_value = "entry_bad"
        mock_store.get_eval_results.return_value = [
            {
                "benchmark": "control",
                "score": float("nan"),
                "created_at": "2026-01-01T00:00:00",
                "details_json": "{}",
            }
        ]
        monkeypatch.setattr("soup_cli.registry.store.RegistryStore", lambda **kw: mock_store)

        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", "registry://entry_bad"],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "GATE PASSED" not in output
        assert "control" in output
        assert "finite" in output.lower()

    def test_ship_cli_refuses_non_finite_baseline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_file = tmp_path / "b.json"
        base_file.write_text('{"control": -1e999}', encoding="utf-8")
        tasks = tmp_path / "tasks.jsonl"
        tasks.write_text('{"prompt": "x", "expected": "y", "scoring": "exact"}\n', encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "ship",
                "--base", "m",
                "--tuned", "m",
                "--task-eval", str(tasks.name),
                "--baseline", str(base_file.name),
            ],
        )
        assert result.exit_code == EXIT_USAGE_ERROR
        output = strip_ansi(result.output)
        assert "finite" in output.lower()
        assert "control" in output


class TestHealthyControlsPreserved:
    """Healthy finite controls must behave exactly as before."""

    def test_cli_healthy_control_zero_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"control": 0.0}', encoding="utf-8")

        # Without --model, stub produces "" for prompt "control".
        # Exact match with expected "gold" scores 0.0.
        # Score 0.0 >= threshold 0.0, delta = 0.0 - 0.0 = 0.0 -> PASS.
        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_OK
        output = strip_ansi(result.output)
        assert "GATE PASSED" in output

    def test_cli_healthy_control_regression_exits_gate_failed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite_file, _ = _make_suite_and_tasks(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"control": 1.0}', encoding="utf-8")

        # Candidate score 0.0, baseline 1.0 -> delta -1.0 < -0.05 -> regression.
        result = runner.invoke(
            app,
            ["eval", "gate", "--suite", str(suite_file.name), "--baseline", str(base_file.name)],
        )
        assert result.exit_code == EXIT_GATE_FAILED
        output = strip_ansi(result.output)
        assert "GATE FAILED" in output

    def test_resolve_baseline_finite_numbers_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_file = tmp_path / "baseline.json"
        base_file.write_text('{"a": 0, "b": 0.42, "c": -0.5, "d": 100.0}', encoding="utf-8")
        scores = resolve_baseline(str(base_file.name))
        assert scores == {"a": 0.0, "b": 0.42, "c": -0.5, "d": 100.0}

    def test_empty_baseline_resolved_silently(self):
        assert resolve_baseline(None) == {}
        assert resolve_baseline("") == {}
