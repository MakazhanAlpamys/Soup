"""The gate script decides whether --cuda-graphs ships, so its arithmetic is tested."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HARNESS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verdict = _load("cuda_graph_decode_verdict")
prompt_maker = _load("cuda_graph_decode_prompts")

_OK = {
    "completed": True,
    "same_workload": True,
    "same_token_ids": True,
    "all_requests_identical": True,
    "same_gpu": True,
}


def _report(variant, rate, reserved):
    return {
        "variant": variant,
        "summary": {
            "aggregate_tokens_per_second": rate,
            "peak_reserved_bytes_including_load_and_warmup": reserved,
        },
    }


def _comparison(checks=None):
    return {"ordered_runs": [{"checks": dict(checks or _OK)} for _ in range(4)]}


GIB = 1024**3


def test_a_clear_win_passes_every_criterion():
    reports = [
        _report("baseline", 30.0, 3 * GIB),
        _report("production_cuda_graphs", 60.0, 3 * GIB),
        _report("production_cuda_graphs", 58.0, 3 * GIB),
        _report("baseline", 31.0, 3 * GIB),
    ]
    result = verdict.harness_verdict(_comparison(), reports)
    assert result["b_slowest_graph_over_fastest_baseline"] == pytest.approx(58.0 / 31.0)
    assert result["b_pass"] and result["c_pass"]
    assert result["a_identical_ids_and_matched_conditions"]


def test_b_uses_the_slowest_graph_run_against_the_fastest_baseline():
    reports = [
        _report("baseline", 30.0, GIB),
        _report("production_cuda_graphs", 60.0, GIB),
        _report("production_cuda_graphs", 36.0, GIB),
        _report("baseline", 31.0, GIB),
    ]
    result = verdict.harness_verdict(_comparison(), reports)
    assert result["b_pass"] is False  # 36/31 = 1.16 < 1.20, although the mean ratio is 1.57


def test_c_fails_past_256_mib_of_extra_reserved():
    reports = [
        _report("baseline", 30.0, GIB),
        _report("production_cuda_graphs", 60.0, GIB + 257 * 1024**2),
    ]
    assert verdict.harness_verdict(_comparison(), reports)["c_pass"] is False


@pytest.mark.parametrize("broken", ["same_token_ids", "completed", "same_gpu"])
def test_a_fails_on_any_mismatched_run(broken):
    checks = dict(_OK, **{broken: False})
    reports = [_report("baseline", 30.0, GIB), _report("production_cuda_graphs", 60.0, GIB)]
    assert (
        verdict.harness_verdict(_comparison(checks), reports)[
            "a_identical_ids_and_matched_conditions"
        ]
        is False
    )


def test_a_strict_check_that_is_missing_counts_as_a_failure():
    checks = {name: value for name, value in _OK.items() if name != "same_token_ids"}
    reports = [_report("baseline", 30.0, GIB), _report("production_cuda_graphs", 60.0, GIB)]
    assert (
        verdict.harness_verdict(_comparison(checks), reports)[
            "a_identical_ids_and_matched_conditions"
        ]
        is False
    )


def test_a_report_set_without_both_variants_is_refused():
    with pytest.raises(ValueError, match="baseline"):
        verdict.harness_verdict(_comparison(), [_report("baseline", 30.0, GIB)])


def _cli_dir(tmp_path, wall, responses):
    (tmp_path / "cli-wall-seconds.json").write_text("﻿" + json.dumps(wall), encoding="utf-8")
    for arm, texts in responses.items():
        (tmp_path / f"out-{arm}.jsonl").write_text(
            "".join(json.dumps({"response": text}) + "\n" for text in texts), encoding="utf-8"
        )
    return tmp_path


def test_d_passes_only_when_both_graph_runs_beat_both_baselines(tmp_path):
    same = {arm: ["x", "y"] for arm in ("a1", "b1", "b2", "a2")}
    fast = _cli_dir(tmp_path, {"a1": 100.0, "b1": 70.0, "b2": 72.0, "a2": 98.0}, same)
    assert verdict.cli_verdict(fast)["d_pass"] is True
    slow = _cli_dir(tmp_path, {"a1": 100.0, "b1": 70.0, "b2": 99.0, "a2": 98.0}, same)
    assert verdict.cli_verdict(slow)["d_pass"] is False


def test_d_fails_when_any_response_differs(tmp_path):
    responses = {"a1": ["x"], "b1": ["x"], "b2": ["DIFFERENT"], "a2": ["x"]}
    cli = _cli_dir(tmp_path, {"a1": 100.0, "b1": 70.0, "b2": 70.0, "a2": 100.0}, responses)
    assert verdict.cli_verdict(cli)["d_pass"] is False


def test_the_prompt_set_is_deterministic_varied_and_not_sorted():
    first = prompt_maker.prompts(24)
    assert first == prompt_maker.prompts(24)
    lengths = [len(text) for text in first]
    assert len(set(lengths)) >= 20
    assert lengths != sorted(lengths)
    assert max(lengths) > 10 * min(lengths)
