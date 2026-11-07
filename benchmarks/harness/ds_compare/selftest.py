#!/usr/bin/env python3
"""CPU-only checks of the harness logic. No GPU, no download, no real training series.

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/src python selftest.py [name-filter]

Runs every ``test_*`` function below and prints PASS / FAIL / SKIP per test. A test that
needs the pinned model's tokenizer reads its location from ``DS_COMPARE_TOKENIZER`` and is
skipped without it. The orchestrator's run loop is exercised with fake child processes and
a fake ``nvidia-smi``, never with a real arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_trace  # noqa: E402
import dry_run_parts  # noqa: E402
import ds_common  # noqa: E402
import gates  # noqa: E402
import make_rows  # noqa: E402
import orchestrate  # noqa: E402
import soup_arm  # noqa: E402
import tiny_fixture  # noqa: E402
from sampler import RunSampler, summarize  # noqa: E402


class SkipTestError(Exception):
    pass


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def test_word_pool_is_256_unique_words() -> None:
    assert len(make_rows.WORD_POOL) == 256
    assert len(set(make_rows.WORD_POOL)) == 256
    assert len(make_rows.TOPICS) == 64 and len(set(make_rows.TOPICS)) == 64


def test_make_rows_unmeasured_is_deterministic() -> None:
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as first,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as second,
    ):
        assert make_rows.main(["--unmeasured", "--out-dir", first]) == 0
        assert make_rows.main(["--unmeasured", "--out-dir", second]) == 0
        for shape, spec in ds_common.SHAPES.items():
            one = (Path(first) / spec["rows_file"]).read_bytes()
            two = (Path(second) / spec["rows_file"]).read_bytes()
            assert one == two, f"shape {shape} differs between two runs"
            rows = ds_common.load_rows(Path(first) / spec["rows_file"])
            assert len(rows) == 64
            assert all(set(row) == {"instruction", "input", "output"} for row in rows)
        manifest = json.loads((Path(first) / make_rows.MANIFEST_NAME).read_text())
        assert manifest["measured"] is False
        assert all(entry["token_len"] is None for entry in manifest["files"].values())


def test_committed_manifest_matches_the_files() -> None:
    manifest = json.loads((HERE / make_rows.MANIFEST_NAME).read_text())
    assert manifest["measured"] is True
    assert manifest["model"]["revision"] == ds_common.MODEL_REVISION
    for name, entry in manifest["files"].items():
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{name} changed since the manifest was written"
        length = entry["token_len"]
        assert length["min"] == length["max"] == entry["target_tokens"], name


def test_chat_template_matches_the_pinned_revision() -> None:
    manifest = json.loads((HERE / make_rows.MANIFEST_NAME).read_text())
    digest = hashlib.sha256(tiny_fixture.NOUS_LLAMA31_TEMPLATE.encode("utf-8")).hexdigest()
    assert digest == manifest["tokenizer"]["chat_template_sha256"]


def test_committed_rows_have_exact_token_counts() -> None:
    location = os.environ.get("DS_COMPARE_TOKENIZER")
    if not location:
        raise SkipTestError("DS_COMPARE_TOKENIZER not set")
    assert make_rows.main(["--check", "--tokenizer", location]) == 0
    tokenizer, _ = make_rows.load_tokenizer(location)
    for shape, spec in ds_common.SHAPES.items():
        for row in ds_common.load_rows(ds_common.rows_path(shape)):
            for word in row["output"].replace(".", " ").split():
                for form in (" " + word, word):
                    assert len(tokenizer.encode(form, add_special_tokens=False)) == 1, word


def _tiny(tmp: str) -> tuple[str, Any]:
    from transformers import AutoTokenizer

    path = tiny_fixture.build_tiny_model_dir(Path(tmp) / "tiny")
    return path, AutoTokenizer.from_pretrained(path)


def test_render_matches_soup_loss_mask() -> None:
    """The HF script's rendering = Soup's ``build_assistant_only_labels`` on alpaca rows."""
    try:
        from soup_cli.data.formats import _convert_alpaca
        from soup_cli.data.loss_mask import build_assistant_only_labels
    except ImportError:
        raise SkipTestError("soup_cli is not importable (set PYTHONPATH to the repo src)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        _, tokenizer = _tiny(tmp)
        rows = ds_common.load_rows(ds_common.rows_path("A"))[:16]
        rows.append({"instruction": "Say hi", "input": "to Bob", "output": "Hi Bob", "system": "x"})
        for row in rows:
            ours = ds_common.render_alpaca_row(tokenizer, row, 256)
            theirs = build_assistant_only_labels(_convert_alpaca(row)["messages"], tokenizer, 256)
            assert ours["input_ids"] == theirs["input_ids"]
            assert ours["labels"] == theirs["labels"]
            assert ours["attention_mask"] == theirs["attention_mask"]
        short = ds_common.render_alpaca_row(tokenizer, rows[0], 40)
        soup_short = build_assistant_only_labels(
            _convert_alpaca(rows[0])["messages"], tokenizer, 40
        )
        assert short["input_ids"] == soup_short["input_ids"] and len(short["input_ids"]) == 40


# --------------------------------------------------------------------------------------
# recorder and metrics
# --------------------------------------------------------------------------------------


def _fake_trace(steps: int = 256, per_step: float = 0.5, tokens: int = 32) -> dict[str, Any]:
    trace = bench_trace.TraceState(None)
    raw: dict[str, Any] = {
        "global_step": steps,
        "complete": True,
        "begin_wall": 1000.0,
        "steps": [
            [s, s * per_step + (2.0 if s == 1 else 0.0), s * tokens] for s in range(1, steps + 1)
        ],
        "logs": [
            {"step": s, "loss": 1.0 / s, "num_tokens": float(s * tokens)}
            for s in range(1, steps + 1)
        ],
        "summary": {"train_runtime": steps * per_step + 2.0},
        "tokens_counted": None,
        "cuda_end": {"max_memory_reserved_bytes": 5 * 1024**3, "device_name": "fake"},
        "cuda_begin": {"max_memory_reserved_bytes": 4 * 1024**3},
    }
    assert trace.to_dict()["schema"] == "ds_compare.trace.v1"
    return raw


def test_derive_metrics() -> None:
    metrics = bench_trace.derive_metrics(_fake_trace(), expected_total_tokens=8192)
    assert metrics["tokens_trained"] == 8192 and metrics["tokens_source"] == "trainer num_tokens"
    assert abs(metrics["tok_per_s"] - 8192 / 130.0) < 1e-9
    # steady state: steps 33..256 at exactly 0.5 s per step and 32 tokens per step
    assert abs(metrics["steady_tok_per_s"] - 64.0) < 1e-9, metrics["steady_tok_per_s"]
    assert metrics["final_loss"] == 1.0 / 256
    assert set(metrics["loss_at"]) == {"64", "128", "192", "256"}
    assert metrics["max_memory_reserved_bytes"] == 5 * 1024**3
    short = bench_trace.derive_metrics(_fake_trace(steps=20), expected_total_tokens=640)
    assert short["steady_tok_per_s"] is None
    raw = _fake_trace()
    raw["tokens_counted"] = 8000
    assert bench_trace.derive_metrics(raw)["tokens_source"] == "counted by the collator"
    raw["logs"] = []
    raw["tokens_counted"] = None
    assert bench_trace.derive_metrics(raw, expected_total_tokens=8192)["tokens_source"].startswith(
        "expected"
    )


def test_trace_state_flush_and_callback_class() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / "t.json"
        state = bench_trace.TraceState(out)
        state.train_begin({"x": 1}, 3)
        state.add_tokens(10)
        state.step_end(1)
        state.log(1, {"loss": 2.0, "num_tokens": 10.0})
        state.log(3, {"train_runtime": 1.5, "train_steps_per_second": 2.0})
        state.train_end()
        raw = json.loads(out.read_text())
        assert (
            raw["complete"]
            and raw["tokens_counted"] == 10
            and raw["summary"]["train_runtime"] == 1.5
        )
        assert raw["logs"][0]["loss"] == 2.0
        cls = bench_trace.make_callback_class(state)
        assert cls().trace is state


# --------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------


def _write_run(
    root: Path,
    shape: str,
    budget: int,
    arm: str,
    rep: int,
    tok: float | None,
    status: str = "ok",
    peak: float = 1000.0,
    loss: float = 0.01,
    void: bool = False,
    util: float | None = None,
) -> None:
    folder = root / shape / str(budget) / arm
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "shape": shape,
        "budget_gb": budget,
        "rep": rep,
        "status": status,
        "void": void,
        "tok_per_s": tok,
        "final_loss": loss,
        "max_memory_reserved_bytes": int(peak * 1024**2),
        "sampler": {
            "whole_run": {"vram_peak_mib": peak},
            "train_window": {"gpu_util_mean_pct": util, "sm_clock_busy_median_mhz": 1980},
        },
    }
    (folder / f"rep{rep}.json").write_text(json.dumps(payload))


def _series(root: Path, shape: str, budget: int, arm: str, values: list[float], **kw: Any) -> None:
    for index, value in enumerate(values, start=1):
        _write_run(root, shape, budget, arm, index, value, **kw)


def test_g1_confirmed_and_lower() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "S1", [63.0, 64.0, 62.0], peak=3935)
        _series(root, "A", 80, "D1", [21.0, 22.0, 21.65], peak=38135)
        result = gates.compute_all(root)["G1"]
        assert result["verdict"] == "CONFIRMED", result
        assert abs(result["ratio"] - 63.0 / 21.65) < 1e-9 and not result["near_miss"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "S1", [50.0, 51.0, 52.0])
        _series(root, "A", 80, "D1", [21.0, 22.0, 21.65])
        result = gates.compute_all(root)["G1"]
        assert result["verdict"] == "LOWER" and 2.3 < result["ratio"] < 2.4, result
    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as tmp:  # near miss on either side of 2.64, not rounded
        root = Path(tmp)
        _series(root, "A", 80, "S1", [2.65 * 20.0] * 3)
        _series(root, "A", 80, "D1", [20.0] * 3)
        result = gates.compute_all(root)["G1"]
        assert result["verdict"] == "CONFIRMED" and result["near_miss"], result
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "S1", [2.63 * 20.0] * 3)
        _series(root, "A", 80, "D1", [20.0] * 3)
        result = gates.compute_all(root)["G1"]
        assert result["verdict"] == "LOWER" and result["near_miss"], result


def test_void_and_failed_runs_do_not_count() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "S1", [60.0, 61.0])
        _write_run(root, "A", 80, "S1", 3, 200.0, void=True)
        _series(root, "A", 80, "D1", [20.0, 20.0, 20.0])
        result = gates.compute_all(root)["G1"]
        assert result["S1"]["runs_valid"] == 2 and abs(result["ratio"] - 60.5 / 20.0) < 1e-9


def test_g2_g3_g4_ablation() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "S1", [60.0, 61.0, 60.5], peak=3900)
        _series(root, "A", 80, "SB", [70.0, 71.0, 70.5], peak=3900)
        _series(root, "A", 80, "S3", [66.0, 66.5, 66.2])
        _series(root, "A", 80, "S6", [60.6, 60.4, 60.5])
        _series(root, "A", 80, "D1", [20.0, 20.0, 20.0], peak=38000)
        _series(root, "A", 80, "D3", [35.0, 35.5, 35.2], peak=30000)
        _series(root, "A", 80, "R1", [300.0, 301.0, 299.0], peak=18000)
        _series(root, "A", 80, "R2", [250.0, 251.0, 249.0], peak=14000)
        # 24 GB: D1 fits, D3 and R1 run out of memory; 16 GB: nothing of DeepSpeed fits
        _series(root, "A", 24, "SB", [70.0, 70.0], peak=3900)
        _series(root, "A", 24, "D1", [20.0, 20.0], peak=21000)
        _series(root, "A", 24, "D3", [None, None], status="oom")
        _write_run(root, "A", 16, "SB", 1, 70.0)
        _write_run(root, "A", 16, "D1", 1, None, status="oom")
        _write_run(root, "A", 16, "D3", 1, None, status="oom")
        # Shape B: D-best (D3) is 1.5x slower than SB -> SMALLER; Shape C: reversed
        _series(root, "B", 80, "SB", [300.0, 300.0])
        _series(root, "B", 80, "D3", [200.0, 200.0])
        _series(root, "B", 80, "D1", [100.0, 100.0])
        _series(root, "B", 80, "R1", [900.0, 900.0])
        _series(root, "C", 80, "SB", [400.0, 400.0])
        _series(root, "C", 80, "D3", [500.0, 500.0])
        _series(root, "C", 80, "D1", [150.0, 150.0])
        _series(root, "C", 80, "R1", [1200.0, 1200.0])
        result = gates.compute_all(root)
        g2 = result["G2"]
        assert (
            g2["80"]["best_deepspeed"]["arm"] == "D3"
            and abs(g2["80"]["ratio"] - 70.5 / 35.2) < 1e-9
        )
        assert abs(g2["24"]["ratio"] - 70.0 / 20.0) < 1e-9
        assert g2["16"]["ratio"] is None and g2["16"]["note"] == "DeepSpeed does not fit at 16 GB"
        g3 = result["G3"]
        assert abs(g3["R1"]["ratio_over_SB"] - 300.0 / 70.5) < 1e-9
        g4 = result["G4"]
        assert g4["d_best_arm"] == "D3"
        assert g4["B"]["verdict"] == "SMALLER" and abs(g4["B"]["SB_over_Dbest"] - 1.5) < 1e-9
        assert g4["C"]["verdict"] == "REVERSED" and abs(g4["C"]["SB_over_Dbest"] - 0.8) < 1e-9
        assert abs(g4["B"]["R1_over_SB"] - 3.0) < 1e-9 and abs(g4["B"]["SB_over_D1"] - 3.0) < 1e-9
        ablation = result["ablation"]["arms"]
        assert (
            ablation["S3"]["reading"] == "faster"
            and ablation["S6"]["reading"] == "no measurable effect"
        )
        assert ablation["S2"]["reading"] == "not measured"
        summary = gates.render_summary(result)
        assert "G1" in summary and "G4" in summary and "Ablation" in summary
    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as tmp:  # HOLDS and the 2.0 near miss
        root = Path(tmp)
        _series(root, "A", 80, "D1", [20.0, 20.0])
        _series(root, "B", 80, "SB", [40.5, 40.5])
        _series(root, "B", 80, "D1", [20.0, 20.0])
        result = gates.compute_g4(gates.load_runs(root), "D1")
        assert result["B"]["verdict"] == "HOLDS" and result["B"]["near_miss"]


# --------------------------------------------------------------------------------------
# orchestrator: plan, derivations, void rules, budget
# --------------------------------------------------------------------------------------


def test_plan_shape() -> None:
    plan = orchestrate.build_plan()
    assert len(plan) == 88  # 91 before S0 left the plan (DEVIATIONS.md D-2)
    assert [s.arm for s in plan[:2]] == ["S1", "D1"] and all(s.warmup for s in plan[:2])
    # S0 is not run: soup-cli at 2c9a078 does not start on this stack (D-2)
    assert "S0" not in orchestrate.SHAPE_A_ARMS and not any(s.arm == "S0" for s in plan)
    assert orchestrate.build_plan({"S0"}) == []
    a80 = [s for s in plan if s.phase == "A"]
    assert len(a80) == 39 == 3 * len(orchestrate.SHAPE_A_ARMS)
    for rep in (1, 2, 3):
        arms = [s.arm for s in a80 if s.rep == rep]
        assert sorted(arms) == sorted(orchestrate.SHAPE_A_ARMS)
    first_reps = [s.rep for s in a80]
    assert first_reps == sorted(first_reps), "repeats must be interleaved, not grouped by arm"
    assert {s.shape for s in plan if s.phase in ("B1", "B2")} == {"B"}
    for phase in ("B1", "C1", "B2", "C2"):
        assert [s.arm for s in plan if s.phase == phase].count("DBEST") == 1
        assert {"SB", "S1", "D1", "R1"} <= {s.arm for s in plan if s.phase == phase}
    assert sum(1 for s in plan if s.phase == "cap24") == 10
    assert sum(1 for s in plan if s.phase == "cap16") == 5
    assert {s.arm for s in plan if s.phase == "cap24"} == set(orchestrate.CAP_ARMS)
    core = [s for s in plan if s.group == "core"]
    assert all(s.phase in ("warmup", "A", "SB", "B1", "C1") for s in core)
    assert [s.group for s in plan if s.phase == "variants"] == ["prefetch"] * 6
    only = orchestrate.build_plan({"S1"})
    assert {s.arm for s in only} == {"S1"}
    assert {s.phase for s in orchestrate.build_plan(None, {"cap16"})} == {"cap16"}


def _args(results: str, **extra: Any) -> argparse.Namespace:
    values = {
        "all": True,
        "only": None,
        "phases": None,
        "results": results,
        "model": "tiny",
        "dry_run": False,
        "part": None,
    }
    values.update(extra)
    return argparse.Namespace(**values)


def test_budget_drop_order() -> None:
    expected = ["S8", "prefetch", "cap16", "B_rep2", "C_rep2", "cap24"]  # protocol section 8
    assert list(orchestrate.DROP_ORDER) == expected
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session = orchestrate.Session(_args(tmp))
        total_min = session.estimate_remaining_minutes()
        session.deadline = time.time() + total_min * 60 * 3
        session.enforce_budget()
        assert session.dropped == [], "plenty of time must drop nothing"
        seen: list[str] = []
        for minutes in (
            total_min * 0.93,
            total_min * 0.8,
            total_min * 0.7,
            total_min * 0.62,
            total_min * 0.55,
            total_min * 0.45,
            total_min * 0.05,
        ):
            session.deadline = time.time() + minutes * 60
            session.enforce_budget()
            assert session.dropped == expected[: len(session.dropped)], session.dropped
            seen.append(f"{minutes:.0f} min -> {len(session.dropped)} dropped")
        assert session.dropped == expected, (session.dropped, seen)
        pending = {(s.phase, s.arm, s.rep) for s in session.pending()}
        assert ("A", "S1", 1) in pending and ("A", "D1", 3) in pending
        assert ("B1", "SB", 1) in pending and ("C1", "R1", 1) in pending
        assert all(s.group == "core" for s in session.pending())


def test_budget_cost_cap_also_drops() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session = orchestrate.Session(_args(tmp))
        session.rate, session.budget_usd = 2.0, 1.0  # a dollar buys 30 minutes
        session.enforce_budget()
        assert session.dropped == list(orchestrate.DROP_ORDER)


def _runs_for_sb(root: Path, changes: dict[str, float], spread_values=(60.0, 61.0, 60.5)) -> list:
    _series(root, "A", 80, "S1", list(spread_values))
    for arm, factor in changes.items():
        _series(root, "A", 80, arm, [60.5 * factor] * 3)
    return gates.load_runs(root)


def test_sb_derivation_rule() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        runs = _runs_for_sb(Path(tmp), {"S6": 1.20, "S7": 1.001, "S4": 0.9, "S3": 1.10, "S2": 1.15})
        out = orchestrate.derive_sb_overrides(runs)
        assert out["selected"] == ["S2", "S6"], out  # S3 loses to the faster S2; S7 inside spread
        assert out["training"] == {"stream_recompute": "auto", "stream_buffers": 3}
        assert out["recompute_conflict_dropped"] == "S3" and out["env"] == {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        runs = _runs_for_sb(Path(tmp), {"S5": 1.3, "S6": 1.0})
        out = orchestrate.derive_sb_overrides(runs)
        assert out["selected"] == ["S5"]
        assert out["env"] == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        runs = _runs_for_sb(Path(tmp), {"S6": 1.0, "S7": 0.99})
        assert orchestrate.derive_sb_overrides(runs)["selected"] == []


def test_idle_rule() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _series(root, "A", 80, "D1", [20.0] * 3, util=45.5)
        _series(root, "A", 80, "D2", [20.0] * 3, util=60.0)
        runs = gates.load_runs(root)
        assert orchestrate.idle_rule(runs, "D1") and not orchestrate.idle_rule(runs, "D2")
        assert not orchestrate.idle_rule(runs, "D3")


def test_void_rules() -> None:
    def run(**kw: Any) -> dict[str, Any]:
        base = {
            "status": "ok",
            "final_loss": 0.01,
            "trace_complete": True,
            "cap_applied": True,
            "sampler": {
                "train_window": {"sm_clock_busy_median_mhz": 1980},
                "foreign_gpu_processes": {"foreign_pids": None},
            },
        }
        base.update(kw)
        return base

    clocks = [1980.0, 1975.0, 1985.0, 1980.0]
    assert orchestrate.void_reasons(run(), 80, [0.0105], clocks, False) == []
    assert any(
        "loss" in r
        for r in orchestrate.void_reasons(run(final_loss=0.02), 80, [0.0105], clocks, False)
    )
    assert orchestrate.void_reasons(run(final_loss=0.0119), 80, [0.0105], clocks, False) == []
    slow = run(sampler={"train_window": {"sm_clock_busy_median_mhz": 1800}})
    assert any("SM clock" in r for r in orchestrate.void_reasons(slow, 80, [], clocks, False))
    assert orchestrate.void_reasons(slow, 80, [], clocks[:2], False) == []  # too few runs yet
    near = run(sampler={"train_window": {"sm_clock_busy_median_mhz": 1890}})
    assert orchestrate.void_reasons(near, 80, [], clocks, False) == []  # 4.5% below: allowed
    foreign = run(sampler={"foreign_gpu_processes": {"foreign_pids": [4242]}})
    assert any(
        "another process" in r for r in orchestrate.void_reasons(foreign, 80, [], clocks, False)
    )
    assert any(
        "another process" in r for r in orchestrate.void_reasons(run(), 80, [], clocks, True)
    )
    assert any(
        "non-zero exit" in r
        for r in orchestrate.void_reasons(run(status="failed"), 80, [], [], False)
    )
    assert any(
        "out of memory" in r for r in orchestrate.void_reasons(run(status="oom"), 80, [], [], False)
    )
    assert orchestrate.void_reasons(run(status="oom"), 24, [], [], False) == []
    assert any(
        "cap was not applied" in r
        for r in orchestrate.void_reasons(run(cap_applied=False), 24, [], [], False)
    )
    assert any(
        "train-end" in r
        for r in orchestrate.void_reasons(run(trace_complete=False), 80, [], [], False)
    )
    over = run(max_memory_reserved_bytes=int(25 * 1024**3))
    assert any("not effective" in r for r in orchestrate.void_reasons(over, 24, [], [], False))
    inside = run(max_memory_reserved_bytes=int(23.9 * 1024**3))
    assert orchestrate.void_reasons(inside, 24, [], [], False) == []
    assert orchestrate.void_reasons(over, 80, [], [], False) == []


def test_estimates_reproduce_the_protocol_total() -> None:
    plan = orchestrate.build_plan()
    conditional = sum(s.est_min for s in plan if s.conditional in ("idle_rule", "d_best"))
    total = sum(s.est_min for s in plan)
    assert 5.0 < (total - conditional) / 60 < 5.6, "unconditional plan should be about 5.2 h"
    assert total / 60 + orchestrate.SETUP_HOURS <= orchestrate.HOURS_LIMIT


# --------------------------------------------------------------------------------------
# orchestrator run loop with fake children and a fake nvidia-smi
# --------------------------------------------------------------------------------------

FAKE_CHILD = r"""
import json, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
scenario = args["--scenario"]
out = args["--out"]
if scenario == "fail":
    print("Traceback: something broke")
    sys.exit(1)
if scenario == "crash_oom_text":
    print("torch.OutOfMemoryError: CUDA out of memory")
    sys.exit(1)
loss = 0.5 if scenario == "loss_bad" else 0.01
payload = {"schema": "ds_compare.run.v1", "status": "ok", "exit_code": 0, "final_loss": loss,
           "tok_per_s": 64.0, "steady_tok_per_s": 65.0, "train_runtime_s": 1.0,
           "tokens_trained": 64, "trace_complete": True, "trace_begin_wall": time.time()}
if scenario == "oom":
    payload.update({"status": "oom", "exit_code": 3, "tok_per_s": None})
if scenario in ("cap", "ok_cap"):
    sys.stderr.write("[ds_compare] VRAM cap applied: 24 GB\n")
json.dump(payload, open(out, "w"))
time.sleep(1.2)
sys.exit(payload["exit_code"])
"""

FAKE_SMI = r"""
import sys, time
args = sys.argv[1:]
if any(a.startswith("--query-compute-apps") for a in args):
    sys.exit(0)
interval = 0.5
for a in args:
    if a.startswith("--loop-ms="):
        interval = int(a.split("=")[1]) / 1000.0
while True:
    print("1234, 87, 1980", flush=True)
    time.sleep(interval)
"""


def _fake_executable(folder: Path, name: str, body: str) -> str:
    script = folder / f"{name}.py"
    script.write_text(body)
    if os.name == "nt":
        launcher = folder / f"{name}.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = folder / name
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher)


def test_sampler_with_a_fake_nvidia_smi() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = Path(tmp)
        smi = _fake_executable(folder, "fake_smi", FAKE_SMI)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        sampler = RunSampler(str(folder / "rep1"), child.pid, 0, smi)
        sampler.start()
        time.sleep(2.6)
        sampler.stop()
        child.wait()
        info = summarize(str(folder / "rep1"), (sampler.started_at, sampler.stopped_at))
        assert info["whole_run"]["samples"] >= 3, info
        assert info["whole_run"]["vram_peak_mib"] == 1234
        assert info["whole_run"]["gpu_util_mean_pct"] == 87
        assert info["train_window"]["sm_clock_busy_median_mhz"] == 1980
        assert info["rss_peak_bytes"] and info["rss_samples"] >= 2
        assert sampler.foreign_gpu_processes()["foreign_pids"] is None  # no apps, nothing visible


def _session_with_fakes(
    tmp: Path, scenarios: list[str], part: str | None = None
) -> tuple[orchestrate.Session, list[dict]]:
    session = orchestrate.Session(_args(str(tmp / "results"), part=part))
    session.nvidia_smi = _fake_executable(tmp, "fake_smi", FAKE_SMI)
    fake = tmp / "fake_child.py"
    fake.write_text(FAKE_CHILD)
    calls: list[dict] = []

    def command_for(slot: Any, overrides_path: str | None) -> list[str]:
        scenario = scenarios.pop(0)
        overrides = json.loads(Path(overrides_path).read_text()) if overrides_path else None
        calls.append({"scenario": scenario, "overrides": overrides})
        out = f"{session.prefix(slot)}.raw.json"
        return [sys.executable, str(fake), "--scenario", scenario, "--out", out]

    session.command_for = command_for  # type: ignore[method-assign]
    return session, calls


def _slot(arm: str, shape: str, budget: int, rep: int, **kw: Any) -> orchestrate.Slot:
    return orchestrate.Slot(0, "test", "core", arm, shape, budget, rep, est_min=1.0, **kw)


def test_run_loop_ok_and_sampler_merge() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, _ = _session_with_fakes(Path(tmp), ["ok"])
        slot = _slot("R1", "A", 80, 1)
        session.run_slot(slot)
        final = json.loads((Path(tmp) / "results" / "A" / "80" / "R1" / "rep1.json").read_text())
        assert final["status"] == "ok" and final["void"] is False and final["attempt"] == 1
        assert (
            final["sampler"]["whole_run"]["vram_peak_mib"] == 1234
            and final["peak_vram_mib"] == 1234
        )
        assert final["sampler"]["train_window"]["samples"] >= 1
        assert (Path(tmp) / "results" / "A" / "80" / "R1" / "rep1.gpu.csv").exists()
        assert (Path(tmp) / "results" / "A" / "80" / "R1" / "rep1.rss.csv").exists()
        assert final["wall_seconds"] > 1.0 and final["cap_applied"] is False
        before = len(session.runs())
        session.run_slot(_slot("R1", "A", 80, 1))  # already on disk: not run again
        assert len(session.runs()) == before


def test_run_loop_failure_is_repeated_once_then_void() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["fail", "fail"])
        session.run_slot(_slot("D1", "A", 80, 1))
        folder = Path(tmp) / "results" / "A" / "80" / "D1"
        final = json.loads((folder / "rep1.json").read_text())
        assert len(calls) == 2 and final["void"] is True and final["void_attempts"] == 2
        assert (folder / "rep1.void1.json").exists()
        stats = gates.arm_stats(gates.load_runs(Path(tmp) / "results"), "A", 80, "D1")
        assert stats["status"] == "not measured"


def test_run_loop_failure_then_success() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["fail", "ok"])
        session.run_slot(_slot("D1", "A", 80, 1))
        final = json.loads((Path(tmp) / "results" / "A" / "80" / "D1" / "rep1.json").read_text())
        assert len(calls) == 2 and final["void"] is False and final["attempt"] == 2


def test_run_loop_oom_under_a_cap_is_not_repeated() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["oom"])
        session.run_slot(_slot("D3", "A", 24, 1))
        final = json.loads((Path(tmp) / "results" / "A" / "24" / "D3" / "rep1.json").read_text())
        assert len(calls) == 1 and final["status"] == "oom" and final["void"] is False
        stats = gates.arm_stats(gates.load_runs(Path(tmp) / "results"), "A", 24, "D3")
        assert stats["status"] == "does not fit"


def test_run_loop_cap_not_applied_is_void() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["ok", "ok_cap"])
        session.run_slot(_slot("R1", "A", 24, 1))
        final = json.loads((Path(tmp) / "results" / "A" / "24" / "R1" / "rep1.json").read_text())
        assert len(calls) == 2 and final["void"] is False and final["cap_applied"] is True


def test_run_loop_loss_mismatch_is_void() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["ok", "loss_bad", "ok"])
        session.run_slot(_slot("R1", "A", 80, 1))
        session.run_slot(_slot("R1", "A", 80, 2))
        final = json.loads((Path(tmp) / "results" / "A" / "80" / "R1" / "rep2.json").read_text())
        assert len(calls) == 3 and final["attempt"] == 2 and final["void"] is False
        assert (Path(tmp) / "results" / "A" / "80" / "R1" / "rep2.void1.json").exists()


def test_run_loop_shape_c_oom_reruns_with_recompute_on() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), ["oom", "ok"])
        session.sb_overrides = {"training": {"stream_recompute": "off"}, "env": {}}
        session.run_slot(_slot("SB", "C", 80, 1))
        folder = Path(tmp) / "results" / "C" / "80" / "SB"
        final = json.loads((folder / "rep1.json").read_text())
        assert [c["scenario"] for c in calls] == ["oom", "ok"]
        assert calls[0]["overrides"]["training"]["stream_recompute"] == "off"
        assert calls[1]["overrides"]["training"]["stream_recompute"] == "on"
        assert final["rerun_recompute_on"] is True and final["status"] == "ok"
        assert (folder / "rep1.oom.json").exists()
    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as tmp:  # S1 already has recompute on: no special rerun
        session, calls = _session_with_fakes(Path(tmp), ["oom", "oom"])
        session.run_slot(_slot("S1", "C", 80, 1))
        assert len(calls) == 2, "uncapped OOM of an arm that already recomputes is a void repeat"


def test_run_loop_resolves_sb_and_d_best_from_disk() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "results"
        _series(root, "A", 80, "S1", [60.0, 61.0, 60.5])
        _series(root, "A", 80, "S6", [75.0] * 3)
        _series(root, "A", 80, "D1", [20.0] * 3, util=45.0)
        _series(root, "A", 80, "D3", [35.0] * 3, util=80.0)
        _series(root, "A", 80, "D1PF", [24.0] * 3)
        session = orchestrate.Session(_args(str(root)))
        sb_slot, best_slot = (
            _slot("SB", "B", 80, 1, conditional="sb_derivation"),
            _slot("DBEST", "B", 80, 1, conditional="d_best"),
        )
        assert session.resolve(sb_slot) and session.sb_overrides["selected"] == ["S6"]
        assert (root / "sb_overrides.json").exists()
        assert session.resolve(best_slot) and best_slot.resolved_arm == "D3"
        assert json.loads((root / "d_best.json").read_text())["arm"] == "D3"
        again = orchestrate.Session(_args(str(root)))  # a restarted session reads both files
        again.ensure_sb()
        assert again.sb_overrides["selected"] == ["S6"]
        d1_slot = _slot("D1PF", "A", 80, 1, conditional="idle_rule")
        assert again.resolve(d1_slot), "D1 was idle (45% < 50%): the variant runs"
        d2_slot = _slot("D2PF", "A", 80, 1, conditional="idle_rule")
        assert not again.resolve(d2_slot) and d2_slot.skipped


# --------------------------------------------------------------------------------------
# soup arms, log parsing, sitecustomize
# --------------------------------------------------------------------------------------


def test_soup_arm_configs_load_in_soup() -> None:
    try:
        from soup_cli.config.loader import load_config
    except ImportError:
        raise SkipTestError("soup_cli is not importable (set PYTHONPATH to the repo src)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for shape in ds_common.SHAPES:
            for arm in soup_arm.STREAM_ARMS:
                config, env, notes = soup_arm.build_config(
                    arm, shape, "some/model", "rows.jsonl", "./out"
                )
                path = Path(tmp) / f"{shape}_{arm}.yaml"
                path.write_text(soup_arm.dump_yaml(config), encoding="utf-8")
                loaded = load_config(str(path))
                spec = ds_common.SHAPES[shape]
                assert (
                    loaded.training.epochs == spec["epochs"]
                    and loaded.data.max_length == spec["max_length"]
                )
                assert (
                    loaded.training.batch_size == 1
                    and loaded.training.gradient_accumulation_steps == 1
                )
                assert (
                    loaded.training.stream_layers is True and loaded.training.quantization == "none"
                )
                assert loaded.data.val_split == 0 and loaded.training.logging_steps == 1
                assert (env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}) == (
                    arm == "S5"
                )
                if arm == "S0":  # not in the plan (D-2); keys that did not exist at 2c9a078
                    assert notes["dropped_keys"] == ["training.seed", "training.stream_pin"]
                    assert loaded.training.seed is None and loaded.training.stream_pin is None
                else:  # D-1: pinning is explicit, S4 alone turns it off
                    assert loaded.training.stream_pin is (arm != "S4"), (arm, shape)


def test_soup_arm_changes_are_single_keys() -> None:
    base, _, _ = soup_arm.build_config("S1", "A", "m", "r.jsonl", "./o")
    for arm in ("S2", "S3", "S4", "S6", "S7"):
        config, env, _ = soup_arm.build_config(arm, "A", "m", "r.jsonl", "./o")
        diff = {
            k for k in config["training"] if config["training"].get(k) != base["training"].get(k)
        }
        assert len(diff) == 1 and not env, (arm, diff)
    config, env, _ = soup_arm.build_config("S5", "A", "m", "r.jsonl", "./o")
    assert config == base and env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    config, _, _ = soup_arm.build_config("S8", "A", "m", "r.jsonl", "./o")
    assert config["training"]["fast_lora"] is True and config["training"]["lora"]["dropout"] == 0.0
    config, env, notes = soup_arm.build_config(
        "SB",
        "A",
        "m",
        "r.jsonl",
        "./o",
        overrides={"training": {"stream_buffers": 3}, "env": {"X": "1"}},
    )
    assert config["training"]["stream_buffers"] == 3 and env == {"X": "1"}


def test_soup_arms_pin_explicitly() -> None:
    """D-1: every streaming arm requests the pin explicitly (S4 requests pageable memory)."""
    for shape in ds_common.SHAPES:
        for arm in soup_arm.STREAM_ARMS:
            config, _, notes = soup_arm.build_config(arm, shape, "m", "r.jsonl", "./o")
            training = config["training"]
            if arm == "S0":  # the key did not exist at 2c9a078: dropped, and listed
                assert "stream_pin" not in training
                assert "training.stream_pin" in notes["dropped_keys"]
            elif arm == "S4":
                assert training["stream_pin"] is False, shape
            else:
                assert training["stream_pin"] is True, (arm, shape)
    # the generated soup.yaml carries it literally, so the log of a run can be checked against it
    base, _, _ = soup_arm.build_config("S1", "A", "m", "r.jsonl", "./o")
    assert "  stream_pin: true\n" in soup_arm.dump_yaml(base)
    leave_one_out, _, _ = soup_arm.build_config("S4", "A", "m", "r.jsonl", "./o")
    assert "  stream_pin: false\n" in soup_arm.dump_yaml(leave_one_out)
    # SB keeps the pin unless the derivation selected S4's change
    sb, _, _ = soup_arm.build_config(
        "SB", "A", "m", "r.jsonl", "./o", overrides={"training": {"stream_buffers": 3}}
    )
    assert sb["training"]["stream_pin"] is True
    sb, _, _ = soup_arm.build_config(
        "SB", "A", "m", "r.jsonl", "./o", overrides={"training": {"stream_pin": False}}
    )
    assert sb["training"]["stream_pin"] is False
    # S4 is still a single change from S1 (true -> false)
    after, before = leave_one_out["training"], base["training"]
    assert {key for key in after if after[key] != before[key]} == {"stream_pin"}


def test_soup_arm_child_environment() -> None:
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        env = soup_arm.child_environment({}, "t.json", "db", None)
        assert "PYTORCH_CUDA_ALLOC_CONF" not in env, "S1 must see only what soup itself exports"
        env = soup_arm.child_environment(
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, "t.json", "db", 24
        )
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
        assert env["SOUP_BENCH_VRAM_CAP_GB"] == "24" and env["SOUP_BENCH_TRACE_OUT"] == "t.json"
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(HERE)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old


def test_soup_gpu_hint_respects_a_user_value() -> None:
    try:
        from soup_cli.utils import gpu
    except ImportError:
        raise SkipTestError("soup_cli is not importable")
    saved = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        gpu.configure_cuda_alloc_hint(True)
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
    finally:
        if saved is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved


def test_parse_soup_log() -> None:
    text = (
        "\x1b[36m+- training.stream_layers ---+\x1b[0m\n"
        "| Layer streaming BETA - arch llama, tier ram                |\n"
        "|   base store   16.06 GB across 32 layers (pinned)              |\n"
        "training.stream_recompute='auto': the last 32 of 32 streamed\n"
        "layers keep their activations, the rest recompute (measured\n"
        "expandable_segments allocator hint not enabled: not on linux\n"
        "[ds_compare] VRAM cap applied: 24 GB\n"
    )
    info = soup_arm.parse_soup_log(text)
    assert info["tier"] == "ram" and info["arch"] == "llama"
    assert info["store_gb"] == 16.06 and info["store_layers"] == 32 and info["store_pinned"] is True
    assert info["recompute_auto_kept_layers"] == 32 and info["vram_cap_applied_in_log"] is True
    assert info["expandable_segments_not_enabled"].startswith("not on linux")
    assert soup_arm.parse_soup_log("torch.OutOfMemoryError: CUDA out of memory")["oom_in_log"]
    assert soup_arm.parse_soup_log("pageable)")["oom_in_log"] is False


def _load_sitecustomize() -> types.ModuleType:
    saved = {k: os.environ.pop(k, None) for k in ("SOUP_BENCH_VRAM_CAP_GB", "SOUP_BENCH_TRACE_OUT")}
    try:
        spec = importlib.util.spec_from_file_location(
            "ds_sitecustomize_under_test", HERE / "sitecustomize.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_sitecustomize_cap_is_applied_after_init_not_before() -> None:
    module = _load_sitecustomize()
    events: list[str] = []
    state = {"inited": False}
    total = 80 * 1024**3

    def original_lazy_init() -> None:
        if not state["inited"]:
            events.append(f"init(alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')})")
            state["inited"] = True

    fake = types.SimpleNamespace(
        _lazy_init=original_lazy_init,
        is_initialized=lambda: state["inited"],
        device_count=lambda: 1,
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=total),
    )

    def set_fraction(fraction: float, index: int) -> None:
        fake._lazy_init()  # the real one initialises CUDA too: must not recurse
        events.append(f"fraction({fraction:.4f},{index})")

    fake.set_per_process_memory_fraction = set_fraction
    fake.memory = types.SimpleNamespace(_lazy_init=original_lazy_init)
    os.environ["SOUP_BENCH_VRAM_CAP_GB"] = "24"
    saved_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        module._install_vram_cap(fake)
        assert (
            fake._lazy_init is not original_lazy_init and fake.memory._lazy_init is fake._lazy_init
        )
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # soup exports it here
        fake._lazy_init()
        fake._lazy_init()
    finally:
        os.environ.pop("SOUP_BENCH_VRAM_CAP_GB", None)
        if saved_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved_conf
    assert events == ["init(alloc_conf=expandable_segments:True)", "fraction(0.3000,0)"], events


def test_sitecustomize_post_import_hook_and_inertness() -> None:
    code = (
        "import sys; print('torch' in sys.modules, 'transformers' in sys.modules, "
        "type(sys.meta_path[0]).__name__)"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(HERE),
        SOUP_BENCH_VRAM_CAP_GB="24",
        SOUP_BENCH_TRACE_OUT=str(HERE / "unused.json"),
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    ).stdout
    assert out.split()[:2] == ["False", "False"], f"hooks imported a heavy library: {out!r}"
    assert out.split()[2] == "_PostImportFinder"
    env = {
        k: v for k, v in env.items() if k not in ("SOUP_BENCH_VRAM_CAP_GB", "SOUP_BENCH_TRACE_OUT")
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    ).stdout
    assert out.split()[2] != "_PostImportFinder", "no environment variable set: nothing installed"


def test_sitecustomize_registers_the_trace_callback() -> None:
    code = "import transformers.trainer as t; print([c.__name__ for c in t.DEFAULT_CALLBACKS])"
    env = dict(os.environ, PYTHONPATH=str(HERE), SOUP_BENCH_TRACE_OUT=str(HERE / "unused.json"))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "BenchTraceCallback" in result.stdout, result.stderr[-500:]


# --------------------------------------------------------------------------------------
# hf_baseline on CPU
# --------------------------------------------------------------------------------------


def test_hf_baseline_cpu_tiny_model() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        model, _ = _tiny(tmp)
        out = Path(tmp) / "run.json"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(HERE))
        command = [
            sys.executable,
            str(HERE / "hf_baseline.py"),
            "--model",
            model,
            "--rows",
            str(ds_common.rows_path("A")),
            "--max-length",
            "256",
            "--epochs",
            "1",
            "--max-steps",
            "4",
            "--dtype",
            "fp32",
            "--deepspeed",
            "none",
            "--arm",
            "R1",
            "--shape",
            "A",
            "--rep",
            "1",
            "--out",
            str(out),
        ]
        done = subprocess.run(command, env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-2000:]
        run = json.loads(out.read_text())
        assert (
            run["schema"] == "ds_compare.run.v1" and run["status"] == "ok" and run["kind"] == "hf"
        )
        assert (
            run["arm"] == "R1"
            and run["steps"] == 4
            and run["tokens_source"] == "counted by the collator"
        )
        assert run["tokens_trained"] > 0 and run["tok_per_s"] > 0 and run["final_loss"] is not None
        assert run["lora"]["target_modules"] == ["q_proj", "v_proj"] and run["lora"]["r"] == 8
        assert run["versions"]["soup-cli"] is not None or run["versions"]["torch"]
        assert (
            run["trace_args"]["learning_rate"] == 2e-4
            and run["trace_args"]["optim"] == "adamw_torch"
        )
        assert run["trace_args"]["weight_decay"] == 0.0 and run["trace_args"]["seed"] == 42
        for key in (
            "train_runtime_s",
            "steady_tok_per_s",
            "loss_at",
            "max_memory_reserved_bytes",
            "argv",
        ):
            assert key in run


def test_hf_baseline_has_no_soup_import() -> None:
    import ast

    for name in ("hf_baseline.py", "ds_common.py", "bench_trace.py", "sitecustomize.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "soup_cli" for n in names), f"{name} imports soup_cli"


def test_deepspeed_configs() -> None:
    expect = {
        "ds_d1_z3_param_offload.json": (3, "cpu", "none", "auto"),
        "ds_d2_z3_param_opt_offload.json": (3, "cpu", "cpu", "auto"),
        "ds_d3_z2_opt_offload.json": (2, None, "cpu", None),
        "ds_d4_z3_no_offload.json": (3, "none", "none", "auto"),
        "ds_d1_prefetch5e8.json": (3, "cpu", "none", 500000000),
        "ds_d2_prefetch5e8.json": (3, "cpu", "cpu", 500000000),
    }
    for name, (stage, param, optim, prefetch) in expect.items():
        config = json.loads((HERE / name).read_text())
        zero = config["zero_optimization"]
        assert zero["stage"] == stage, name
        assert (zero.get("offload_param") or {}).get("device") == param, name
        assert zero["offload_optimizer"]["device"] == optim, name
        assert zero.get("stage3_prefetch_bucket_size") == prefetch, name
        assert config["bf16"] == {"enabled": True}
        assert (
            config["train_micro_batch_size_per_gpu"] == 1
            and config["gradient_accumulation_steps"] == 1
        )
        assert (
            config["optimizer"]["type"] == "AdamW" and config["optimizer"]["params"]["lr"] == 2e-4
        )
        assert config["optimizer"]["params"]["weight_decay"] == 0.0
        # HF owns the scheduler (constant lr, no warm-up); DeepSpeed owns only the optimizer.
        assert "scheduler" not in config, name
        assert zero["overlap_comm"] is True and zero["contiguous_gradients"] is True
        assert zero["reduce_bucket_size"] == "auto"
        for section in ("offload_param", "offload_optimizer"):
            if (zero.get(section) or {}).get("device") == "cpu":
                assert zero[section]["pin_memory"] is True, (name, section)
        if stage == 3:
            assert zero["stage3_param_persistence_threshold"] == "auto"
            assert zero["stage3_max_live_parameters"] == 10**9 == zero["stage3_max_reuse_distance"]
            assert zero["sub_group_size"] == 10**9


def test_text_rules() -> None:
    # Words that must not appear in the published files; the list is local (not committed).
    banned = tuple(
        word.strip().lower()
        for word in os.environ.get("DS_COMPARE_BANNED_WORDS", "").split(",")
        if word.strip()
    )
    for path in sorted(HERE.iterdir()):
        if (
            path.suffix not in (".py", ".sh", ".md", ".json", ".jsonl")
            or path.name == "selftest.py"
        ):
            continue
        if path.suffix == ".jsonl":
            continue
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} contains {word!r}"
        # "vast.ai" is the rental site, not the word.
        assert not re.search(r"\b(ai|agents?)\b", text.replace("vast.ai", "vast")), (
            f"{path.name} contains a banned word"
        )
        assert not re.search("[" + chr(0x400) + "-" + chr(0x4FF) + "]", text), (
            f"{path.name} contains Cyrillic"
        )


# --------------------------------------------------------------------------------------
# running in parts (README "Running in parts"): plan split, resume, refusal, scripts
# --------------------------------------------------------------------------------------


def _plan_key(slot: orchestrate.Slot) -> tuple:
    return (slot.phase, slot.group, slot.arm, slot.shape, slot.budget, slot.rep, slot.warmup)


def _fake_part1(root: Path) -> None:
    """Shape A at 80 GB results as part 1 leaves them: SB picks S6, D-best is D3."""
    _series(root, "A", 80, "S1", [60.0, 61.0, 60.5])
    _series(root, "A", 80, "S6", [75.0] * 3)
    _series(root, "A", 80, "D1", [20.0] * 3, util=45.0)
    _series(root, "A", 80, "D3", [35.0] * 3, util=80.0)
    _series(root, "A", 80, "D1PF", [24.0] * 3)


def _ship_run_jsons(source: Path, target: Path) -> None:
    """What make_payload.sh ships: the rep<k>.json files, nothing else."""
    for path in source.glob("*/*/*/rep*.json"):
        if re.match(r"^rep\d+\.json$", path.name):
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, destination)


def test_parts_are_disjoint_and_cover_the_plan() -> None:
    full = orchestrate.build_plan()
    full_keys = [_plan_key(slot) for slot in full]
    assert len(set(full_keys)) == len(full_keys) == 88
    by_part = {part: orchestrate.build_plan(None, None, part) for part in orchestrate.PARTS}
    assert tuple(by_part) == ("part1", "part2", "part3")
    key_sets = {part: {_plan_key(slot) for slot in slots} for part, slots in by_part.items()}
    for first, second in itertools.combinations(key_sets, 2):
        assert not key_sets[first] & key_sets[second], (first, second)
    assert set().union(*key_sets.values()) == set(full_keys)
    assert sum(len(slots) for slots in by_part.values()) == len(full)
    # every phase of the full plan is in exactly one part; the slot knows which
    phases = [phase for part_phases in orchestrate.PART_PHASES.values() for phase in part_phases]
    assert len(phases) == len(set(phases)) and set(phases) == {s.phase for s in full}
    for part, slots in by_part.items():
        assert all(slot.part == part for slot in slots)
        # the protocol's order is kept and a slot keeps its position in the full plan
        assert [slot.index for slot in slots] == sorted(slot.index for slot in slots)
        assert all(_plan_key(full[slot.index]) == _plan_key(slot) for slot in slots)

    part1, part2, part3 = (by_part[name] for name in orchestrate.PARTS)
    assert len(part1) == 53 and len(part2) == 20 and len(part3) == 15
    assert not any(s.arm == "S0" for part in (part1, part2, part3) for s in part)  # D-2
    # part 1: Shape A at 80 GB, every arm of section 2 with the warm-ups, 3 repeats
    assert {(s.shape, s.budget) for s in part1} == {("A", 80)}
    assert {s.arm for s in part1} == set(orchestrate.SHAPE_A_ARMS) | {"SB", "D1PF", "D2PF", "S8"}
    assert [s.arm for s in part1 if s.warmup] == ["S1", "D1"]
    assert {s.rep for s in part1 if not s.warmup} == {1, 2, 3}
    assert {s.phase for s in part1} == {"warmup", "A", "SB", "variants", "S8"}
    # part 2: Shapes B and C at 80 GB, SB S1 D-best D1 R1, 2 repeats
    assert {(s.shape, s.budget) for s in part2} == {("B", 80), ("C", 80)}
    assert {s.arm for s in part2} == set(orchestrate.BC_ARMS) and {s.rep for s in part2} == {1, 2}
    # part 3: the 24 and 16 GB caps on Shape A, SB D1 D2 D3 R1
    assert {(s.shape, s.budget) for s in part3} == {("A", 24), ("A", 16)}
    assert {s.arm for s in part3} == set(orchestrate.CAP_ARMS)
    assert {s.rep for s in part3 if s.budget == 24} == {1, 2}
    assert {s.rep for s in part3 if s.budget == 16} == {1}
    # --only and --part combine
    only = orchestrate.build_plan({"S1"}, None, "part2")
    assert {(s.arm, s.shape) for s in only} == {("S1", "B"), ("S1", "C")}
    try:
        orchestrate.build_plan(None, None, "part9")
    except ValueError as why:
        assert "part9" in str(why)
    else:
        raise AssertionError("an unknown part must be refused")


def test_part_estimates_add_up_and_max_hours_round_up() -> None:
    full = orchestrate.estimate_plan(orchestrate.build_plan())
    reports = [dry_run_parts.part_report(part, None, 2.23) for part in orchestrate.PARTS]
    assert sum(report.runs for report in reports) == full["runs"] == 88
    assert abs(sum(report.gpu_minutes for report in reports) - full["minutes"]) < 0.5
    for report in reports:
        raw = (
            orchestrate.SETUP_MIN_PER_PART + report.gpu_minutes + orchestrate.MARGIN_MIN_PER_PART
        ) / 60.0
        assert raw <= report.suggested_max_hours < raw + 0.25 + 1e-6
        assert (report.suggested_max_hours * 4) % 1 == 0
        assert report.cost_at_max_hours_usd == round(report.suggested_max_hours * 2.23, 2)
    # rounding: an exact quarter hour stays, a minute more goes up to the next one
    assert orchestrate.suggested_max_hours(130.0) == 3.0  # (35 + 130 + 15) / 60
    assert orchestrate.suggested_max_hours(131.0) == 3.25
    assert orchestrate.suggested_max_hours(0.0) == 1.0
    # a measured wall time replaces the estimate of that arm and shape, as in a running series
    slow = dry_run_parts.part_report("part3", {("R1", "A"): 200.0}, None)
    assert slow.gpu_minutes > reports[2].gpu_minutes + 400
    # the --max-hours written into the job scripts (less 15 minutes of boot) is this estimate
    for report in reports:
        text = (HERE / f"job_{report.part}.sh").read_text(encoding="utf-8")
        budget = int(re.search(r'JOB_BUDGET_S="\$\{JOB_BUDGET_S:-(\d+)\}"', text).group(1))
        assert budget == int(report.suggested_max_hours * 3600) - 900, (report.part, budget)


def test_part_budget_drop_is_confined_to_the_part() -> None:
    expected = {
        "part1": ["S8", "prefetch"],
        "part2": ["B_rep2", "C_rep2"],
        "part3": ["cap16", "cap24"],
    }
    for part, dropped in expected.items():
        assert dropped == [g for g in orchestrate.DROP_ORDER if g in dropped]  # protocol order
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            session = orchestrate.Session(_args(tmp, part=part))
            session.deadline = time.time() + 1
            session.enforce_budget()
            assert session.dropped == dropped, (part, session.dropped)
            # core is never dropped, whatever is left
            assert all(slot.group == "core" for slot in session.pending()), part
            assert (part == "part3") == (session.pending() == [])  # part 3 has no core group
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session = orchestrate.Session(_args(tmp, part="part1"))
        total = session.estimate_remaining_minutes()
        # room for everything except the 9.6 minutes of S8: only S8 goes
        left_s = total * 0.97 * 60 * orchestrate.SAFETY_FACTOR + orchestrate.FINISH_MARGIN_S + 5
        session.deadline = time.time() + left_s
        session.enforce_budget()
        assert session.dropped == ["S8"], session.dropped


def test_part_resume_derives_the_same_sb_and_d_best() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        part1_results = base / "part1_results"
        _fake_part1(part1_results)
        # what the whole series derives from these run jsons
        shutil.copytree(part1_results, base / "full")
        full = orchestrate.Session(_args(str(base / "full")))
        full_sb = full.ensure_sb()
        full.resolve_d_best(full.runs())
        assert full_sb["selected"] == ["S6"] and full.d_best == "D3"
        for part in ("part2", "part3"):
            shipped = base / part
            _ship_run_jsons(part1_results, shipped)
            assert not (shipped / "sb_overrides.json").exists()
            session = orchestrate.Session(_args(str(shipped), part=part))
            assert session.refusals() == []
            sb = session.ensure_sb()
            assert sb["selected"] == full_sb["selected"] and sb["training"] == full_sb["training"]
            assert sb["env"] == full_sb["env"]
            assert sb["arm_changes_pct"] == full_sb["arm_changes_pct"]
            if part == "part2":
                slot = _slot("DBEST", "B", 80, 1, conditional="d_best")
                assert session.resolve(slot) and slot.resolved_arm == full.d_best == "D3"
                assert json.loads((shipped / "d_best.json").read_text())["arm"] == "D3"
        # part 1's own files are read when they were shipped too (sb_overrides.json, d_best.json)
        shipped_files = base / "with_files"
        _ship_run_jsons(part1_results, shipped_files)
        shutil.copy(base / "part2" / "sb_overrides.json", shipped_files)
        shutil.copy(base / "part2" / "d_best.json", shipped_files)
        again = orchestrate.Session(_args(str(shipped_files), part="part3"))
        assert again.refusals() == [] and again.ensure_sb()["selected"] == ["S6"]
    # a finished run is never run again, in any part: no child process is started
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        session, calls = _session_with_fakes(Path(tmp), [], part="part1")
        _fake_part1(Path(tmp) / "results")
        slot = _slot("S1", "A", 80, 1)
        before = len(session.runs())
        session.run_slot(slot)  # an empty scenario list raises if a child would be started
        assert slot.done and calls == [] and len(session.runs()) == before


def test_part_refuses_without_part1_results() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        empty = Path(tmp) / "empty"
        for part in ("part2", "part3"):
            session = orchestrate.Session(_args(str(empty), part=part))
            problems = session.refusals()
            assert len(problems) == 1 and "found none" in problems[0] and "part1" in problems[0]
            assert "make_payload.sh" in problems[0], "the message says what to do"
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                assert session.execute() == 2
            assert "REFUSED" in captured.getvalue() and "found none" in captured.getvalue()
            assert not empty.exists(), "a refused part writes nothing"
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                argv = ["--part", part, "--preflight", "--results", str(empty)]
                assert orchestrate.main(argv) == 2
            assert "found none" in captured.getvalue()
        # part 1 needs nothing from an earlier part
        assert orchestrate.part_prerequisites("part1", empty) == []
        assert orchestrate.main(["--part", "part1", "--preflight", "--results", str(empty)]) == 0
        # S1 alone: SB can be derived (part 3 may start), D-best cannot (part 2 refuses)
        only_s1 = Path(tmp) / "only_s1"
        _series(only_s1, "A", 80, "S1", [60.0, 61.0, 60.5])
        assert orchestrate.part_prerequisites("part3", only_s1) == []
        problems = orchestrate.part_prerequisites("part2", only_s1)
        assert len(problems) == 1 and "D-best" in problems[0] and "D1PF" in problems[0]
        # DeepSpeed alone: SB cannot be derived
        only_ds = Path(tmp) / "only_ds"
        _series(only_ds, "A", 80, "D1", [20.0] * 3)
        for part in ("part2", "part3"):
            problems = orchestrate.part_prerequisites(part, only_ds)
            assert len(problems) == 1 and "S1" in problems[0] and "SB" in problems[0], part
        # runs that are void, out of memory or at the wrong budget are not part 1's results
        bad = Path(tmp) / "bad"
        _series(bad, "A", 80, "S1", [60.0] * 3, void=True)
        _series(bad, "A", 80, "D1", [20.0] * 3, status="oom")
        _series(bad, "A", 24, "S1", [60.0] * 3)
        _series(bad, "B", 80, "S1", [60.0] * 3)
        assert len(orchestrate.part_prerequisites("part2", bad)) == 2
        # a derivation file without the runs behind it is not accepted
        files_only = Path(tmp) / "files_only"
        files_only.mkdir()
        (files_only / "sb_overrides.json").write_text('{"selected": []}')
        (files_only / "d_best.json").write_text('{"arm": "D1"}')
        assert "found none" in orchestrate.part_prerequisites("part2", files_only)[0]
        # S0 is not in the plan (D-2), so a missing S0 executable refuses nothing ...
        session = orchestrate.Session(_args(str(Path(tmp) / "r"), part="part1"))
        session.s0_bin = str(Path(tmp) / "no_such_soup")
        assert session.refusals() == []
        # ... but the guard stays for a plan that does hold an S0 slot (the arm's code is kept)
        session.plan.append(_slot("S0", "A", 80, 1))
        assert any("SKIP_S0" in problem for problem in session.refusals())
        session.s0_bin = str(Path(tmp) / "empty_file")
        Path(session.s0_bin).write_text("")
        assert session.refusals() == []
        session = orchestrate.Session(_args(str(Path(tmp) / "r"), part="part2"))
        session.s0_bin = str(Path(tmp) / "no_such_soup")
        assert all("arm S0" not in problem for problem in session.refusals())


def test_part_command_line() -> None:
    args = orchestrate.parse_args(["--part", "part3"])
    assert args.part == "part3" and not args.all and not args.preflight
    assert orchestrate.parse_args(["--all"]).part is None
    for bad in (["--part", "part9"], ["--preflight"], ["--part"], []):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                orchestrate.parse_args(bad)
            except SystemExit as why:
                assert why.code == 2, bad
            else:
                raise AssertionError(f"{bad} must be refused")


def test_parts_write_their_own_records() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "results"
        _fake_part1(root)
        orchestrate.Session(_args(str(root), part="part1")).finish()
        done_one = (root / "DONE_part1").read_bytes()
        record = json.loads(done_one)
        assert record["part"] == "part1" and record["d_best_now"] == "D3"
        assert record["sb_selected"] is None, "no sb_overrides.json yet: nothing to report"
        assert not (root / "DONE").exists(), "only a whole series writes DONE"
        orchestrate.Session(_args(str(root), part="part2")).finish()
        assert (root / "DONE_part2").exists() and (root / "DONE_part1").read_bytes() == done_one
        session = orchestrate.Session(_args(str(root), part="part3"))
        session.deadline = time.time() + 1
        session.enforce_budget()
        assert (root / "dropped_part3.json").exists() and not (root / "dropped.json").exists()
        orchestrate.Session(_args(str(root))).finish()
        assert (root / "DONE").exists() and json.loads((root / "DONE").read_text())["part"] is None
        assert session.tag == "_part3" and orchestrate.Session(_args(str(root))).tag == ""


# -- shell scripts ---------------------------------------------------------------------


def _find_bash() -> str | None:
    """Git Bash on Windows (a bare ``bash`` there is WSL's), ``bash`` elsewhere."""
    if os.name != "nt":
        return shutil.which("bash")
    for candidate in (
        os.environ.get("GIT_BASH", ""),
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


_SCRIPT_ENV_KEYS = (
    "DEADLINE_EPOCH VK_DEADLINE_EPOCH VK_RESERVE_S VK_OUT VK_ROOT PAYLOAD_DIR SETUP_PART SKIP_S0 "
    "DS_COMPARE_RESULTS JOB_DRY_RUN JOB_BUDGET_S JOB_PYTHON ALLOW_DIRTY PAYLOAD_MAX_FILE_MB "
    "PAYLOAD_REPO PAYLOAD_REF PAYLOAD_PYTHON SETUP_SECONDS PY SOUP_S0_BIN"
).split()


def _run_script(script: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    bash = _find_bash()
    if bash is None:
        raise SkipTestError("no bash")
    env = {key: value for key, value in os.environ.items() if key not in _SCRIPT_ENV_KEYS}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    if os.name == "nt":  # the msys tools (find, tar, grep) before Windows' own find.exe
        env["PATH"] = str(Path(bash).parent) + os.pathsep + env.get("PATH", "")
    env.update(env_extra)
    return subprocess.run(
        [bash, script.as_posix(), *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def _copy_harness(target: Path, with_rows: bool = False) -> None:
    """The harness files as committed (LF), into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(HERE.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json", ".md", ".jsonl"):
            continue
        if path.suffix == ".jsonl" and not with_rows:
            continue
        (target / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def test_job_scripts_results_deadline_and_refusal() -> None:
    if _find_bash() is None:
        raise SkipTestError("no bash")
    python = Path(sys.executable).as_posix()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        payload, kit = base / "payload", base / "kit"
        _copy_harness(payload / "benchmarks" / "harness" / "ds_compare")
        (payload / "PAYLOAD.txt").write_text("part: part2\n")
        _fake_part1(payload / "results")
        kit.mkdir()
        now = int(time.time())
        (kit / "deadline").write_text(f"{now + 3 * 3600}\n")
        (kit / "vk.env").write_text(
            f"VK_DEADLINE_EPOCH={now + 99}\nVK_RESERVE_S=300\nVK_LAUNCH_EPOCH={now - 60}\n"
        )

        def job(part: str, out: str, **extra: str) -> subprocess.CompletedProcess:
            env = {
                "PAYLOAD_DIR": (base / "payload").as_posix(),
                "VK_OUT": (base / out).as_posix(),
                "VK_ROOT": kit.as_posix(),
                "JOB_PYTHON": python,
                "JOB_DRY_RUN": "1",
            }
            env.update(extra)
            scripts = payload / "benchmarks" / "harness" / "ds_compare"
            return _run_script(scripts / f"job_{part}.sh", **env)

        # the kit's deadline file, less its reserve (300 s) and the 10-minute finish margin
        result = job("part2", "out2")
        assert result.returncode == 0, (result.stdout, result.stderr)
        epoch = int(re.search(r"DEADLINE_EPOCH=(\d+)", result.stdout).group(1))
        assert epoch == now + 3 * 3600 - 300 - 600, (epoch, result.stdout)
        assert "deadline" in result.stdout and "SKIP_S0=1" in result.stdout, result.stdout
        out2 = base / "out2"
        assert (out2 / "job_part2.log").exists() and (out2 / "payload_part2.txt").exists()
        assert (out2 / "results" / "A" / "80" / "S1" / "rep1.json").exists(), "shipped results"
        assert "would run: bash" in result.stdout and "run_all.sh --part part2" in result.stdout
        # part 3 likewise; part 1 skips the S0 venv too (no arm of the plan uses it: D-2)
        assert job("part3", "out3").returncode == 0
        part1 = job("part1", "out1")
        assert part1.returncode == 0 and "SKIP_S0=1" in part1.stdout, part1.stdout
        assert "SKIP_S0=0" not in part1.stdout
        # a deadline passed as an environment variable wins over the files
        result = job("part2", "out2b", VK_DEADLINE_EPOCH=str(now + 7200), VK_RESERVE_S="120")
        epoch = int(re.search(r"DEADLINE_EPOCH=(\d+)", result.stdout).group(1))
        assert result.returncode == 0 and epoch == now + 7200 - 120 - 600, result.stdout
        assert "environment VK_DEADLINE_EPOCH" in result.stdout
        # no kit files: job start + JOB_BUDGET_S, less the finish margin
        (base / "empty_kit").mkdir()
        result = job("part2", "out2c", VK_ROOT=(base / "empty_kit").as_posix(), JOB_BUDGET_S="5000")
        epoch = int(re.search(r"DEADLINE_EPOCH=(\d+)", result.stdout).group(1))
        assert result.returncode == 0 and abs(epoch - (now + 5000 - 600)) < 120, result.stdout
        assert "JOB_BUDGET_S" in result.stdout
        # a deadline that is already past stops the job
        (kit / "deadline").write_text(f"{now - 10}\n")
        result = job("part2", "out2d")
        assert result.returncode != 0 and "already past" in (result.stdout + result.stderr)
        (kit / "deadline").write_text(f"{now + 3 * 3600}\n")
        # part 2 without part 1's results refuses BEFORE set-up (the dry-run line is never reached)
        shutil.move(str(payload / "results"), str(base / "results_aside"))
        result = job("part2", "out2e")
        assert result.returncode != 0, result.stdout
        merged = result.stdout + result.stderr  # the job sends its stderr through its log tee
        assert "found none" in merged and "no set-up time was spent" in merged
        assert "would run" not in merged
        # part 1 does not need them
        assert job("part1", "out1e").returncode == 0
        # a real run of setup.sh would follow: its switches are checked in test_setup_switches


def test_setup_switches_and_run_all_part_option() -> None:
    setup_text = (HERE / "setup.sh").read_text(encoding="utf-8")
    # every record of a part goes to $REC (results/setup_<part>/); only env.sh names $RESULTS
    for line in setup_text.splitlines():
        if "$RESULTS" in line and not line.lstrip().startswith("#"):
            assert any(
                allowed in line
                for allowed in (
                    'RESULTS="${DS_COMPARE_RESULTS',
                    'REC="$RESULTS"',
                    'REC="$RESULTS/setup_$SETUP_PART"',
                    'mkdir -p "$RESULTS" "$REC"',
                    'export DS_COMPARE_RESULTS="$RESULTS"',
                )
            ), line
    assert 'if [ "$SKIP_S0" = "1" ]' in setup_text and "SOUP_S0_BIN" in setup_text
    bash = _find_bash()
    if bash is None:
        raise SkipTestError("no bash")
    for script in ("setup.sh", "run_all.sh", "job_common.sh", "make_payload.sh", "job_part1.sh"):
        result = subprocess.run(
            [bash, "-n", (HERE / script).as_posix()], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (script, result.stderr)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        results = (Path(tmp) / "results").as_posix()
        for key, value, message in (
            ("SETUP_PART", "part9", "SETUP_PART must be"),
            ("SKIP_S0", "2", "SKIP_S0 must be"),
        ):
            result = _run_script(HERE / "setup.sh", DS_COMPARE_RESULTS=results, **{key: value})
            assert result.returncode == 1 and message in result.stderr, (key, result.stderr)
            assert not Path(results).exists(), "a refused set-up writes nothing"
        # run_all.sh finds the set-up time of the part it runs (results/setup_<part>/)
        stand_in = Path(tmp) / "stand_in"
        stand_in.mkdir()
        shutil.copy(HERE / "run_all.sh", stand_in / "run_all.sh")
        (stand_in / "orchestrate.py").write_text(
            "import os, sys\n"
            "print('SETUP_SECONDS=' + os.environ.get('SETUP_SECONDS', '')"
            " + ' ARGS=' + ' '.join(sys.argv[1:]))\n"
        )
        stand_results = Path(tmp) / "stand_results"
        (stand_results / "setup_part2").mkdir(parents=True)
        (stand_results / "setup_part2" / "setup_seconds.txt").write_text("321\n")
        (stand_results / "setup_seconds.txt").write_text("999\n")
        for arguments, seconds in (
            (["--part", "part2"], "321"),
            (["--part=part2", "--dry-run"], "321"),
            (["--part", "part3"], ""),  # no record of its own yet: not part 2's, not the series'
            (["--all"], "999"),
        ):
            result = _run_script(
                stand_in / "run_all.sh",
                *arguments,
                DS_COMPARE_RESULTS=stand_results.as_posix(),
                PY=Path(sys.executable).as_posix(),
            )
            assert result.returncode == 0, (arguments, result.stderr)
            assert f"SETUP_SECONDS={seconds} ARGS={' '.join(arguments)}" in result.stdout, (
                arguments,
                result.stdout,
            )
        # run_all.sh hands --part on to the real orchestrate.py
        _fake_part1(Path(tmp) / "results")
        python = Path(sys.executable).as_posix()
        for part, lines in (("part2", "plan: 20 slots"), ("part3", "plan: 15 slots")):
            for style in (["--part", part], [f"--part={part}"]):
                result = _run_script(
                    HERE / "run_all.sh", "--dry-run", *style, DS_COMPARE_RESULTS=results, PY=python
                )
                assert result.returncode == 0, (result.stdout[-400:], result.stderr[-400:])
                assert lines in result.stdout and part in result.stdout.splitlines()[0]
                assert "REFUSED" not in result.stdout
        result = _run_script(
            HERE / "run_all.sh",
            "--dry-run",
            "--part",
            "part1",
            DS_COMPARE_RESULTS=results,
            PY=python,
        )
        assert "plan: 53 slots" in result.stdout, result.stdout[:300]
        # no part-1 results: the dry run says so, a real start would refuse
        result = _run_script(
            HERE / "run_all.sh",
            "--part",
            "part2",
            DS_COMPARE_RESULTS=(Path(tmp) / "nothing").as_posix(),
            PY=python,
        )
        assert result.returncode == 2 and "found none" in result.stderr, result.stderr


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=selftest",
            "-c",
            "user.email=selftest@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _payload_repo(base: Path, extra: dict[str, bytes] | None = None) -> Path:
    """A throw-away git repository whose HEAD holds the harness (LF) plus ``extra`` files."""
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")  # make_payload.sh's own git calls read this
    _copy_harness(repo / "benchmarks" / "harness" / "ds_compare", with_rows=True)
    for name, data in (extra or {}).items():
        target = repo / "benchmarks" / "harness" / "ds_compare" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(repo, "add", "-f", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _make_payload(repo: Path, part: str, out: Path, ship: Path | None = None, **env: str):
    args = [part, out.as_posix()] + ([ship.as_posix()] if ship else [])
    return _run_script(
        HERE / "make_payload.sh",
        *args,
        PAYLOAD_REPO=repo.as_posix(),
        PAYLOAD_PYTHON=Path(sys.executable).as_posix(),
        **env,
    )


def _all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_make_payload_builds_a_clean_folder() -> None:
    if _find_bash() is None or shutil.which("git") is None:
        raise SkipTestError("no bash or git")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        repo = _payload_repo(base, {"__pycache__/stale.pyc": b"\x00bytecode\r\n"})
        out = base / "out1"
        result = _make_payload(repo, "part1", out)
        assert result.returncode == 0, (result.stdout, result.stderr)
        harness = out / "benchmarks" / "harness" / "ds_compare"
        for name in ("orchestrate.py", "setup.sh", "job_part1.sh", "job_common.sh", "PROTOCOL.md"):
            assert (harness / name).exists(), name
        files = _all_files(out)
        assert not any("__pycache__" in path.parts or path.suffix == ".pyc" for path in files)
        assert all(b"\r" not in path.read_bytes() for path in files), "LF only"
        info = (out / "PAYLOAD.txt").read_text()
        assert "harness_commit: " + _git(repo, "rev-parse", "HEAD") in info
        assert "suggested --max-hours: 4.75" in result.stdout and "job_part1.sh" in result.stdout
        # an existing non-empty folder is never built over
        again = _make_payload(repo, "part1", out)
        assert again.returncode != 0 and "not empty" in again.stderr
        # part 2 and part 3 without part 1's results: refused, no payload at the target
        for part in ("part2", "part3"):
            refused = _make_payload(repo, part, base / f"no_{part}")
            assert refused.returncode != 0 and "found none" in refused.stderr, refused.stderr
            assert not (base / f"no_{part}").exists()
        # shipped results: the run jsons and nothing else; the CSV and log files hold CR bytes
        results = base / "part1_results"
        _fake_part1(results)
        for folder in (results / "A" / "80" / "S1", results / "A" / "80" / "D1"):
            (folder / "rep1.gpu.csv").write_bytes(b"time,mem\r\n1,2\r\n")
            (folder / "rep1.log").write_bytes(b"step 1\r step 2\r")
            (folder / "rep1.raw.json").write_text("{}")
            (folder / "rep1.void1.json").write_text("{}")
        (results / "sb_overrides.json").write_text('{"selected": ["S6"]}')
        (results / "A" / "80" / "S1" / "notes.txt").write_text("stay behind")
        out2 = base / "out2"
        result = _make_payload(repo, "part2", out2, results)
        assert result.returncode == 0, (result.stdout, result.stderr)
        shipped = sorted(
            path.relative_to(out2 / "results").as_posix() for path in _all_files(out2 / "results")
        )
        wanted = sorted(
            path.relative_to(results).as_posix() for path in results.glob("*/*/*/rep[0-9].json")
        )
        assert shipped == sorted(wanted + ["sb_overrides.json"]), shipped
        assert all(b"\r" not in path.read_bytes() for path in _all_files(out2))
        # what was shipped is what the part needs: its own preflight passes on the payload
        assert orchestrate.part_prerequisites("part2", out2 / "results") == []
        # a wrong folder (no run json in it) is refused
        wrong = base / "wrong"
        wrong.mkdir()
        (wrong / "notes.txt").write_text("nothing here")
        refused = _make_payload(repo, "part3", base / "out_wrong", wrong)
        assert refused.returncode != 0 and "no run json" in refused.stderr
        assert not (base / "out_wrong").exists()


def test_make_payload_refuses_a_cr_byte_and_a_big_file() -> None:
    if _find_bash() is None or shutil.which("git") is None:
        raise SkipTestError("no bash or git")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)

        def case(name: str, extra: dict[str, bytes] | None = None) -> Path:
            (base / name).mkdir()
            return _payload_repo(base / name, extra)

        # a committed CR byte (a CRLF shell script) stops the build and leaves no payload
        repo = case("crlf_script", {"extra_job.sh": b"#!/usr/bin/env bash\r\necho hi\r\n"})
        out = base / "out_cr"
        result = _make_payload(repo, "part1", out)
        assert result.returncode != 0, (result.stdout, result.stderr)
        assert "carriage-return" in result.stderr and "extra_job.sh" in result.stderr, result.stderr
        assert not out.exists(), "a refused build leaves no payload at the target"
        # one stray CR inside a data file is enough
        repo = case("crlf_data", {"data.json": b'{"a": 1}\r\n'})
        refused = _make_payload(repo, "part1", base / "out_cr_data")
        assert refused.returncode != 0 and "data.json" in refused.stderr
        # the same repository without the CR file builds: the check is what stops it
        repo = case("lf_only", {"extra_job.sh": b"#!/usr/bin/env bash\necho hi\n"})
        assert _make_payload(repo, "part1", base / "ok").returncode == 0
        # a file over the limit (lowered to 1 MB here; the default is 50 MB)
        repo = case("big_file", {"big.bin": b"x" * (2 * 1024 * 1024)})
        result = _make_payload(repo, "part1", base / "out_big", PAYLOAD_MAX_FILE_MB="1")
        assert result.returncode != 0 and "over 1 MB" in result.stderr, result.stderr
        assert "big.bin" in result.stderr
        assert not (base / "out_big").exists()
        ok = _make_payload(repo, "part1", base / "big_ok", PAYLOAD_MAX_FILE_MB="3")
        assert ok.returncode == 0, ok.stderr
        assert "PAYLOAD_MAX_FILE_MB:-50" in (HERE / "make_payload.sh").read_text(encoding="utf-8")
        # a harness that differs from HEAD is refused: the payload must be what the owner sees
        repo = case("dirty")
        script = repo / "benchmarks" / "harness" / "ds_compare" / "orchestrate.py"
        text = script.read_text(encoding="utf-8")
        script.write_text(text + "\n# uncommitted\n", encoding="utf-8")
        result = _make_payload(repo, "part1", base / "out_dirty")
        assert result.returncode != 0 and "uncommitted" in result.stderr, result.stderr
        assert _make_payload(repo, "part1", base / "dirty_ok", ALLOW_DIRTY="1").returncode == 0


def test_harness_files_are_lf_in_the_index() -> None:
    if shutil.which("git") is None:
        raise SkipTestError("no git")
    listing = subprocess.run(
        ["git", "-C", str(HERE), "ls-files", "--eol", "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0 or not listing.stdout.strip():
        raise SkipTestError("not inside a git checkout")
    for line in listing.stdout.splitlines():
        info, _, name = line.partition("\t")
        assert info.split()[0] not in ("i/crlf", "i/mixed"), f"{name}: not LF in the index"


def test_readme_documents_running_in_parts() -> None:
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "## Running in parts" in text
    for needle in (
        "--part part1",
        "make_payload.sh",
        "job_part1.sh",
        "dry_run_parts.py",
        "SKIP_S0",
        "setup_part1",
        "refuses",
    ):
        assert needle in text, needle


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("filter", nargs="?", default="")
    args = parser.parse_args()
    tests: list[tuple[str, Callable[[], None]]] = [
        (name, func)
        for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func) and args.filter in name
    ]
    failed = 0
    for name, func in tests:
        started = time.time()
        try:
            func()
        except SkipTestError as why:
            print(f"SKIP {name}: {why}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
        else:
            print(f"PASS {name} ({time.time() - started:.1f}s)")
        sys.stdout.flush()
    print(f"\n{len(tests) - failed}/{len(tests)} passed (skips count as passed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
