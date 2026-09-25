"""#1224 — ``--baseline registry://`` must use the newest row per benchmark.

``RegistryStore.get_eval_results`` returns rows newest first. ``resolve_baseline``
used to keep the *last* row it saw for each benchmark (the oldest) while the
provenance warning was computed from the newest stamped row, so re-measuring a
baseline -- exactly what the warning asks for -- silenced the warning and kept
the stale score. A 0.07 regression then passed the gate.

Rows are inserted directly with explicit ``created_at`` values so ordering and
ties are pinned without sleeping.
"""

from __future__ import annotations

import json

import pytest

from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

_OLD = {"provenance": {"soup_version": "0.60.0", "scorer_revision": 0}}
_CURRENT = {"provenance": {"soup_version": "x", "scorer_revision": BUNDLED_SCORER_REVISION}}


@pytest.fixture
def registry_run(tmp_path, monkeypatch):
    """A tracker run pushed to the registry; yields (insert, spec)."""
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.registry.store import RegistryStore

    tracker = ExperimentTracker()
    run_id = tracker.start_run(
        {"base": "llama", "task": "sft"},
        device="cpu",
        device_name="cpu",
        gpu_info={},
    )

    def insert(benchmark: str, score: float, details: dict, created_at: str) -> None:
        # Bypass save_eval_result so the stamp and the timestamp are ours.
        conn = tracker._get_conn()
        conn.execute(
            """INSERT INTO eval_results
               (run_id, model_path, benchmark, score, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, "m", benchmark, score, json.dumps(details), created_at),
        )
        conn.commit()

    store = RegistryStore(db_path=tmp_path / "reg.db")
    eid = store.push(
        name="baseline", tag="v1", base_model="llama", task="sft",
        run_id=run_id, config={},
    )
    store.close()

    yield insert, f"registry://{eid}"
    tracker.close()


def _resolve(spec: str) -> tuple[dict[str, float], list[str]]:
    from soup_cli.eval.gate import resolve_baseline

    seen: list[str] = []
    scores = resolve_baseline(spec, warn=seen.append)
    return scores, seen


class TestNewestRowWins:
    def test_remeasured_baseline_uses_new_score_and_is_silent(self, registry_run):
        insert, spec = registry_run
        insert("mmlu", 0.30, _OLD, "2026-01-01T00:00:00")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.30}
        assert len(seen) == 1 and "scorer_revision=0" in seen[0]

        # The user follows the warning and re-measures the same run.
        insert("mmlu", 0.45, _CURRENT, "2026-01-02T00:00:00")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.45}
        assert seen == []

    def test_newest_wins_even_when_its_stamp_is_worse(self, registry_run):
        """The rule is "newest wins", not "best stamp wins"."""
        insert, spec = registry_run
        insert("mmlu", 0.45, _CURRENT, "2026-01-01T00:00:00")
        insert("mmlu", 0.30, _OLD, "2026-01-02T00:00:00")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.30}
        assert len(seen) == 1
        assert "scorer_revision=0" in seen[0]
        assert "mmlu" in seen[0]

    def test_only_the_stale_benchmark_is_named(self, registry_run):
        insert, spec = registry_run
        insert("mmlu", 0.30, _OLD, "2026-01-01T00:00:00")
        insert("gsm8k", 0.20, _OLD, "2026-01-01T00:00:01")
        insert("mmlu", 0.45, _CURRENT, "2026-01-02T00:00:00")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.45, "gsm8k": 0.20}
        assert len(seen) == 1
        assert "gsm8k" in seen[0]
        assert "mmlu" not in seen[0]

    def test_benchmarks_sharing_a_problem_share_one_warning(self, registry_run):
        insert, spec = registry_run
        insert("mmlu", 0.30, _OLD, "2026-01-01T00:00:00")
        insert("gsm8k", 0.20, _OLD, "2026-01-01T00:00:01")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.30, "gsm8k": 0.20}
        assert len(seen) == 1
        assert "gsm8k, mmlu" in seen[0]

    def test_newest_row_unstamped_warns_unknown_provenance(self, registry_run):
        """A pre-#404 row (no ``provenance`` in details_json) on top still warns."""
        insert, spec = registry_run
        insert("mmlu", 0.45, _CURRENT, "2026-01-01T00:00:00")
        insert("mmlu", 0.50, {"note": "pre-404"}, "2026-01-02T00:00:00")

        scores, seen = _resolve(spec)
        assert scores == {"mmlu": 0.50}
        assert len(seen) == 1
        assert "unknown provenance" in seen[0]

    def test_identical_timestamps_resolve_to_the_later_insert(self, registry_run):
        insert, spec = registry_run
        insert("mmlu", 0.30, _CURRENT, "2026-01-01T00:00:00")
        insert("mmlu", 0.45, _CURRENT, "2026-01-01T00:00:00")

        for _ in range(3):
            scores, seen = _resolve(spec)
            assert scores == {"mmlu": 0.45}
            assert seen == []


def test_tracker_orders_ties_by_insertion(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    conn = tracker._get_conn()
    for score in (0.1, 0.2, 0.3):
        conn.execute(
            """INSERT INTO eval_results
               (run_id, model_path, benchmark, score, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("R", "m", "mmlu", score, "{}", "2026-01-01T00:00:00"),
        )
    conn.commit()
    assert [r["score"] for r in tracker.get_eval_results("R")] == [0.3, 0.2, 0.1]
    assert [r["score"] for r in tracker.get_eval_results()] == [0.3, 0.2, 0.1]
    tracker.close()


def test_gate_flags_regression_against_remeasured_baseline(
    registry_run, tmp_path, monkeypatch
):
    """A candidate at 0.38 regresses against a re-measured 0.45 baseline."""
    from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

    insert, spec = registry_run
    insert("mmlu", 0.30, _OLD, "2026-01-01T00:00:00")
    insert("mmlu", 0.45, _CURRENT, "2026-01-02T00:00:00")

    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    # 19 of 50 answered correctly -> 0.38.
    tasks_file.write_text(
        "".join(
            json.dumps({"prompt": f"q{i}", "expected": "yes", "scorer": "exact"}) + "\n"
            for i in range(50)
        ),
        encoding="utf-8",
    )
    suite = EvalSuite(
        suite="s",
        tasks=[GateTask(
            type="custom", name="mmlu", tasks=str(tasks_file),
            scorer="exact", threshold=0.0,
        )],
    )
    correct = {f"q{i}" for i in range(19)}

    baseline, _ = _resolve(spec)
    result = run_gate(
        suite,
        generate_fn=lambda p: "yes" if p in correct else "no",
        baseline=baseline,
    )

    task = result.task_results[0]
    assert task.score == pytest.approx(0.38)
    assert task.baseline == pytest.approx(0.45)
    assert task.delta == pytest.approx(-0.07)
    assert result.regression is True
    assert result.passed is False
