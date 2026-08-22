"""#404 — baseline scorer/version stamp + revision fingerprint lock."""

from __future__ import annotations

import json

import pytest


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
