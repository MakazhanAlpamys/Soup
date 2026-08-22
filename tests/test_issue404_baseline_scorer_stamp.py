"""#404 — baseline scorer/version stamp + revision fingerprint lock."""

from __future__ import annotations

import inspect
import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


class TestBundledScorerRevisionLock:
    def test_fingerprint_matches_locked_constant(self):
        """Fails if a bundled scorer's output moves without bumping revision.

        Update ``BUNDLED_SCORER_REVISION`` and ``BUNDLED_SCORER_FINGERPRINT``
        in the same commit that changes scorer semantics.
        """
        from soup_cli.eval.gate_suites import (
            BUNDLED_SCORER_FINGERPRINT,
            BUNDLED_SCORER_REVISION,
            bundled_scorer_fingerprint,
        )

        assert BUNDLED_SCORER_REVISION >= 1
        assert bundled_scorer_fingerprint() == BUNDLED_SCORER_FINGERPRINT

    def test_every_suite_fingerprint_score_is_strictly_between_0_and_1(self):
        """CONTROL. No suite may pin at 0.0 or 1.0 under the fingerprint corpus."""
        from soup_cli.eval.gate_suites import (
            DEFAULT_GENERAL_SUITE,
            bundled_scorer_fingerprint_scores,
        )

        scores = bundled_scorer_fingerprint_scores()
        assert set(scores) == set(DEFAULT_GENERAL_SUITE)
        assert list(scores) == sorted(DEFAULT_GENERAL_SUITE)
        for name, score in scores.items():
            assert 0.0 < score < 1.0, f"{name} pinned at {score}"

    def test_removing_arguments_requirement_moves_fingerprint(self, monkeypatch):
        """#346 mutation: name-only bare function must change the fingerprint."""
        from soup_cli.eval import gate_suites as gate_suites_mod
        from soup_cli.eval.gate_suites import (
            BUNDLED_SCORER_FINGERPRINT,
            BUNDLED_SCORER_REVISION,
            bundled_scorer_fingerprint,
        )

        before = bundled_scorer_fingerprint()
        assert before == BUNDLED_SCORER_FINGERPRINT

        def name_only(obj: object) -> bool:
            return (
                isinstance(obj, dict)
                and "function" not in obj
                and isinstance(obj.get("name"), str)
            )

        monkeypatch.setattr(gate_suites_mod, "_looks_like_a_bare_function", name_only)
        # Corpus is cached on first use; clear so the mutated unwrap path runs.
        monkeypatch.setattr(gate_suites_mod, "_fingerprint_responses", None)

        after = bundled_scorer_fingerprint()
        assert after != before
        assert BUNDLED_SCORER_REVISION >= 1  # revision unchanged by the mutation


class TestStampBaselineScores:
    def test_envelope_shape(self):
        from soup_cli.eval.gate import stamp_baseline_scores
        from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

        payload = stamp_baseline_scores({"mini_mmlu": 0.5, "mini_tool_call": 1.0})
        assert set(payload) == {"scores", "provenance"}
        assert payload["scores"] == {"mini_mmlu": 0.5, "mini_tool_call": 1.0}
        assert payload["provenance"]["scorer_revision"] == BUNDLED_SCORER_REVISION
        assert isinstance(payload["provenance"]["soup_version"], str)
        assert payload["provenance"]["soup_version"]

    def test_write_and_resolve_round_trip(self, tmp_path, monkeypatch):
        from soup_cli.eval.gate import resolve_baseline, write_baseline_file

        monkeypatch.chdir(tmp_path)
        write_baseline_file("b.json", {"a": 0.1, "b": 0.2})
        seen: list[str] = []
        scores = resolve_baseline("b.json", warn=lambda m: seen.append(m))
        assert scores == {"a": 0.1, "b": 0.2}
        assert seen == []

    def test_rejects_bool_score(self):
        from soup_cli.eval.gate import stamp_baseline_scores

        with pytest.raises(TypeError, match="number"):
            stamp_baseline_scores({"x": True})  # type: ignore[dict-item]

    def test_envelope_with_extra_key_warns_and_resolves(self, tmp_path, monkeypatch):
        from soup_cli.eval.gate import resolve_baseline, stamp_baseline_scores

        monkeypatch.chdir(tmp_path)
        payload = stamp_baseline_scores({"mini_mmlu": 0.42})
        payload["extra_meta"] = "nope"
        (tmp_path / "baseline.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        seen: list[str] = []
        scores = resolve_baseline(
            "baseline.json", warn=lambda msg: seen.append(msg)
        )
        assert scores == {"mini_mmlu": 0.42}
        assert len(seen) == 1
        assert "extra_meta" in seen[0]
        assert "scores" in seen[0] and "provenance" in seen[0]


class TestRegistryBaselineStamp:
    def test_save_eval_result_stamps_details(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))

        from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            {"base": "llama", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        tracker.save_eval_result(
            model_path="m",
            benchmark="mini_mmlu",
            score=0.5,
            details={"note": "hi"},
            run_id=run_id,
        )
        rows = tracker.get_eval_results(run_id=run_id)
        tracker.close()
        assert len(rows) == 1
        details = json.loads(rows[0]["details_json"])
        assert details["note"] == "hi"
        assert details["provenance"]["scorer_revision"] == BUNDLED_SCORER_REVISION

    def test_save_eval_result_rejects_non_dict_details(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            {"base": "llama", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        with pytest.raises(TypeError, match="details must be a dict"):
            tracker.save_eval_result(
                model_path="m",
                benchmark="mini_mmlu",
                score=0.5,
                details=["not", "a", "dict"],  # type: ignore[arg-type]
                run_id=run_id,
            )
        tracker.close()

    def test_registry_baseline_with_stamp_is_silent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

        from soup_cli.eval.gate import resolve_baseline
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            {"base": "llama", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        tracker.save_eval_result(
            model_path="m",
            benchmark="mini_mmlu",
            score=0.55,
            details={},
            run_id=run_id,
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
        tracker.close()

        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(
            name="baseline",
            tag="v1",
            base_model="llama",
            task="sft",
            run_id=run_id,
            config={},
        )
        store.close()

        seen: list[str] = []
        scores = resolve_baseline(
            f"registry://{eid}", warn=lambda m: seen.append(m)
        )
        assert scores == {"mini_mmlu": 0.55}
        assert seen == []

    def test_unstamped_registry_baseline_warns_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

        from soup_cli.eval.gate import resolve_baseline
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            {"base": "llama", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
        )
        # Bypass save_eval_result stamping: insert a raw unstamped row.
        now = __import__("datetime").datetime.now().isoformat()
        conn = tracker._get_conn()
        conn.execute(
            """INSERT INTO eval_results
               (run_id, model_path, benchmark, score, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, "m", "mini_mmlu", 0.61, json.dumps({"note": "old"}), now),
        )
        conn.commit()
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
        tracker.close()

        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(
            name="baseline",
            tag="v1",
            base_model="llama",
            task="sft",
            run_id=run_id,
            config={},
        )
        store.close()

        seen: list[str] = []
        scores = resolve_baseline(
            f"registry://{eid}", warn=lambda m: seen.append(m)
        )
        assert scores == {"mini_mmlu": 0.61}
        assert len(seen) == 1
        assert "unknown provenance" in seen[0]


class TestWriteBaselineCli:
    def _write_suite(self, tmp_path, *, tasks=None):
        if tasks is None:
            tasks = [
                {
                    "type": "custom",
                    "name": "exact_task",
                    "threshold": 0.0,
                    "tasks": "tasks.jsonl",
                    "scorer": "exact",
                }
            ]
        suite = {"suite": "smoke", "tasks": tasks}
        (tmp_path / "suite.yaml").write_text(
            yaml.safe_dump(suite), encoding="utf-8"
        )
        if tasks:
            (tmp_path / "tasks.jsonl").write_text(
                json.dumps({"prompt": "hi", "expected": ""}) + "\n",
                encoding="utf-8",
            )

    def test_write_baseline_without_model_fails_and_creates_no_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        self._write_suite(tmp_path)
        result = runner.invoke(
            app,
            [
                "eval", "gate",
                "--suite", "suite.yaml",
                "--write-baseline", "baseline.json",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "--write-baseline requires --model" in result.output
        assert not (tmp_path / "baseline.json").exists()

    def test_write_baseline_file_rejects_empty_scores(self):
        from soup_cli.eval.gate import write_baseline_file

        with pytest.raises(ValueError, match="empty baseline"):
            write_baseline_file("unused.json", {})

    def test_empty_result_set_refused_creates_no_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        self._write_suite(tmp_path, tasks=[])  # no scored tasks
        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            lambda _model: (lambda _prompt: ""),
        )
        result = runner.invoke(
            app,
            [
                "eval", "gate",
                "--suite", "suite.yaml",
                "--model", "fake-model",
                "--write-baseline", "baseline.json",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Cannot write --write-baseline" in result.output
        assert "empty baseline" in result.output
        assert not (tmp_path / "baseline.json").exists()

    def test_preexisting_empty_baseline_resolves_silently(
        self, tmp_path, monkeypatch
    ):
        """Backward-compatible read: empty scores file stays silent. """
        from soup_cli.eval.gate import resolve_baseline

        monkeypatch.chdir(tmp_path)
        (tmp_path / "empty.json").write_text(
            json.dumps({"scores": {}, "provenance": {"soup_version": "0.0.0"}}),
            encoding="utf-8",
        )
        seen: list[str] = []
        scores = resolve_baseline(
            "empty.json", warn=lambda msg: seen.append(msg)
        )
        assert scores == {}
        assert seen == []

    def test_eval_gate_write_baseline_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_suite(tmp_path)
        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            lambda _model: (lambda _prompt: ""),
        )

        result = runner.invoke(
            app,
            [
                "eval", "gate",
                "--suite", "suite.yaml",
                "--model", "fake-model",
                "--write-baseline", "baseline.json",
            ],
        )
        assert result.exit_code in (0, 1), (result.output, repr(result.exception))
        assert (tmp_path / "baseline.json").exists()
        payload = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert "scores" in payload
        assert "exact_task" in payload["scores"]
        assert "soup_version" in payload["provenance"]
        assert "scorer_revision" in payload["provenance"]

        from soup_cli.eval.gate import resolve_baseline

        seen: list[str] = []
        scores = resolve_baseline(
            "baseline.json", warn=lambda msg: seen.append(msg)
        )
        assert scores == payload["scores"]
        assert seen == []


class TestCliWarnPlumbing:
    def test_ship_passes_warn_callback(self):
        from soup_cli.commands import ship as ship_mod

        src = inspect.getsource(ship_mod._verdict_live)
        assert "warn=" in src
        assert "console.print" in src

    def test_eval_passes_warn_callback(self):
        from soup_cli.commands import eval as eval_mod

        src = inspect.getsource(eval_mod.gate_cmd)
        assert "warn=" in src
        assert "console.print" in src

    def test_eval_gate_cli_shows_unknown_provenance_warning(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        suite = {
            "suite": "smoke",
            "tasks": [
                {
                    "type": "custom",
                    "name": "exact_task",
                    "threshold": 0.0,
                    "tasks": "tasks.jsonl",
                    "scorer": "exact",
                }
            ],
        }
        (tmp_path / "suite.yaml").write_text(
            yaml.safe_dump(suite), encoding="utf-8"
        )
        (tmp_path / "tasks.jsonl").write_text(
            json.dumps({"prompt": "hi", "expected": ""}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "old_baseline.json").write_text(
            json.dumps({"exact_task": 0.9}), encoding="utf-8"
        )
        result = runner.invoke(
            app,
            [
                "eval", "gate",
                "--suite", "suite.yaml",
                "--baseline", "old_baseline.json",
            ],
        )
        assert result.exit_code in (0, 1), (result.output, repr(result.exception))
        assert "Warning" in result.output
        assert "unknown provenance" in result.output

    def test_ship_cli_shows_unknown_provenance_warning(
        self, tmp_path, monkeypatch
    ):
        from pathlib import Path

        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        # Same fake-generator pattern as test_v07125 live baseline tests.
        def factory(model_id, *, adapter=None, device=None, max_new_tokens=64, **kwargs):
            is_tuned = adapter is not None or "tuned" in str(model_id)

            def gen(prompt: str) -> str:
                if "TASKMARK" in prompt:
                    return "the magic widget" if is_tuned else "nope"
                return "B" if is_tuned else "zzz"

            return gen

        monkeypatch.setattr(live_eval, "make_generator", factory)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("tasks.jsonl").write_text(
                "\n".join(
                    json.dumps(r)
                    for r in (
                        {
                            "prompt": "TASKMARK one",
                            "expected": "widget",
                            "scoring": "contains",
                        },
                        {
                            "prompt": "TASKMARK two",
                            "expected": "widget",
                            "scoring": "contains",
                        },
                    )
                ),
                encoding="utf-8",
            )
            Path("baseline.json").write_text(
                json.dumps({"mini_mmlu": 0.2}), encoding="utf-8"
            )
            result = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "fake-base",
                    "--adapter", "fake-adapter",
                    "--task-eval", "tasks.jsonl",
                    "--general-suite", "mini_mmlu",
                    "--baseline", "baseline.json",
                ],
            )
            assert "Warning" in result.output, result.output
            assert "unknown provenance" in result.output
