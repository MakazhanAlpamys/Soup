"""Independent closure audit for the fixed four-comparison #2050 protocol.

Reads evidence only; never launches a model or changes a freeze. Writes the
verified result and a concise Markdown record only after all checks succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console


def bootstrap(reductions: list[float]) -> dict[str, Any]:
    if len(reductions) != 5:
        raise ValueError("exactly five complete pairs required")
    rng = random.Random(42)
    samples = sorted(
        statistics.median(rng.choice(reductions) for _ in reductions) for _ in range(10000)
    )
    return {
        "median_pct": statistics.median(reductions),
        "ci95": [samples[249], samples[9749]],
        "ci98_75": [samples[61], samples[9936]],
    }


def audit(evidence: Path, tree: Path | None = None) -> dict[str, Any]:
    def read(name: str) -> Any:
        return json.loads((evidence / name).read_text(encoding="utf-8"))

    assert read("status.json")["phase"] == "completed", "campaign incomplete"
    freeze = read("freeze.json")
    if tree is not None:
        for name, expected in freeze["files"].items():
            actual = hashlib.sha256((tree / name).read_bytes()).hexdigest()
            assert actual == expected, f"frozen file mismatch: {name}"
    assert freeze["base"] == "e05b6f6db42f9cc804fdece2b6eb812432d6aabd"
    assert freeze["model"]["revision"] == "60b99dcbc7f853b7cd18472a607e8d045b96f03e"
    expected_tags = [
        f"s{s}-{m}"
        for s in (512, 2048)
        for m in (
            "full",
            "full-repeat",
            "qkv",
            "qkvo",
            "profile-full",
            "profile-qkv",
            "profile-qkvo",
        )
    ] + [f"s{s}-{p}-pairs" for s in (512, 2048) for p in ("qkv", "qkvo")]
    ledger = read("ledger.json")
    assert [entry["tag"] for entry in ledger] == expected_tags
    assert all(entry["rc"] == 0 for entry in ledger)
    result: dict[str, Any] = {
        "verified": False,
        "full_size_gate_processes": 14,
        "updates_per_gate": 2,
        "comparisons": [],
        "operators": {},
    }
    versions = None
    for seq in (512, 2048):
        reference = read(f"s{seq}-full-audit.json")["steps"]
        assert len(reference) == 2
        for step in reference:
            assert len(step["gradients"]) == len(step["parameters"]) == 128
            assert step["gradients"].keys() == step["parameters"].keys()
            for category in ("gradients", "parameters"):
                assert all(len(value) == 64 for value in step[category].values())
        for mode in (
            "full",
            "full-repeat",
            "qkv",
            "qkvo",
            "profile-full",
            "profile-qkv",
            "profile-qkvo",
        ):
            receipt = read(f"s{seq}-{mode}-audit.json")
            assert receipt["status"] == "ok" and receipt["steps"] == reference, (seq, mode)
            assert receipt["attention_backend"] == "sdpa_math"
            assert receipt["deterministic_algorithms"] and receipt["cache_checked"]
            assert receipt["gpu_uuid"] == freeze["gpu_uuid"]
            policy = mode.removeprefix("profile-")
            assert receipt["policy"] == ("full" if policy == "full-repeat" else policy)
            assert receipt["profile"] == mode.startswith("profile-")
            width, tensors = {"qkv": (6144, 96), "qkvo": (10240, 128)}.get(policy, (0, 0))
            assert receipt["saved_bytes"] == 32 * width * seq * 2
            assert receipt["saved_tensors"] == tensors
            if mode.startswith("profile-"):
                counts: Counter = Counter()
                for key, value in receipt["operators"].items():
                    phase, name, _shapes = json.loads(key)
                    counts[f"{phase}:{name}"] += value
                assert counts, "empty operator profile"
                result["operators"][f"s{seq}-{policy}"] = dict(sorted(counts.items()))
        for policy, width in (("qkv", 6144), ("qkvo", 10240)):
            directory = f"s{seq}-{policy}-pairs"
            arm_ledger = read(f"{directory}/ledger.json")
            expected_order = [
                (pair, arm)
                for pair in range(5)
                for arm in (("A", "B") if pair % 2 == 0 else ("B", "A"))
            ]
            assert [(entry["pair"], entry["arm"]) for entry in arm_ledger] == expected_order
            assert all(entry["rc"] == 0 for entry in arm_ledger)
            pairs = []
            trace = None
            for pair in range(5):
                rows = [read(f"{directory}/p{pair}-{arm}.json") for arm in ("A", "B")]
                for arm, row in zip(("A", "B"), rows):
                    assert row["status"] == "ok" and row["measurement_kind"] == "throughput"
                    assert row["versions"]["commit"].removesuffix("-dirty") == freeze["base"]
                    if versions is None:
                        versions = row["versions"]
                    assert row["versions"] == versions, "runtime versions changed between arms"
                    assert (
                        row["seq"],
                        row["batch"],
                        row["buffers"],
                        row["lora_r"],
                        row["warmup_steps"],
                        row["measured_steps"],
                        row["quant"],
                        row["targets"],
                        row["nf4_scale_cache"],
                    ) == (seq, 1, 2, 16, 10, 50, "nf4", "q_proj,v_proj", False)
                    assert row["pin_required"] and row["store_gb_pinned"] > 4
                    assert len(row["step_times_s"]) == 50 and all(
                        value > 0 for value in row["step_times_s"]
                    )
                    assert row["gpu"]["n"] > 0 and row["gpu"]["temp_max"] <= 75
                    meta = row["selective_checkpoint"]
                    assert meta["gpu_uuid"] == freeze["gpu_uuid"]
                    assert meta["policy"] == ("full" if arm == "A" else policy)
                    assert meta["attention_backend"] == "sdpa_math" and not meta["profile"]
                    assert meta["saved_bytes"] == (0 if arm == "A" else 32 * width * seq * 2)
                    assert meta["admission"]["temperature_c"] <= 65
                    assert meta["admission"]["memory_mib"] < 1024
                    assert meta["admission"]["power_w"] == 350
                    endpoints = (row["loss_first"], row["loss_last"])
                    if trace is None:
                        trace = endpoints
                    assert endpoints == trace, (seq, policy, pair, arm, "loss endpoints differ")
                a, b = rows
                ta, tb = (sum(row["step_times_s"]) for row in rows)
                # Original harness rounds peaks to .001 decimal GB. Use the
                # worst-case rounding bound, so a marginal memory result cannot pass.
                delta_upper = (
                    b["peak_vram_gb"]["allocated"] - a["peak_vram_gb"]["allocated"] + 0.001
                ) * 1e9
                pairs.append(
                    {
                        "pair": pair,
                        "order": "AB" if pair % 2 == 0 else "BA",
                        "a_s": ta,
                        "b_s": tb,
                        "reduction_pct": 100 * (ta - tb) / ta,
                        "a_peak_gb": a["peak_vram_gb"],
                        "b_peak_gb": b["peak_vram_gb"],
                        "memory_delta_upper_bytes": delta_upper,
                    }
                )
            stats = bootstrap([pair["reduction_pct"] for pair in pairs])
            existing = read(f"{directory}/summary.json")
            assert existing["pairs"] == 5
            assert existing["reductions_pct"] == [round(pair["reduction_pct"], 3) for pair in pairs]
            assert existing["ci95"] == [round(v, 3) for v in stats["ci95"]]
            assert existing["median_pct"] == round(stats["median_pct"], 3)
            budget = 32 * width * seq * 2
            memory_pass = all(
                pair["memory_delta_upper_bytes"] <= budget + 67108864 for pair in pairs
            )
            positive = (
                stats["median_pct"] >= 2
                and stats["ci95"][0] > 0
                and stats["ci98_75"][0] > 0
                and memory_pass
            )
            verdict = (
                "CONFIRMED"
                if positive
                else "MEMORY GATE FAILED"
                if not memory_pass
                else "NO GAIN"
                if stats["ci95"][1] < 0
                else "INCONCLUSIVE"
            )
            result["comparisons"].append(
                {
                    "seq": seq,
                    "policy": policy,
                    "saved_bytes": budget,
                    "pairs": pairs,
                    **stats,
                    "memory_pass": memory_pass,
                    "verdict": verdict,
                }
            )
    result["verified"] = True
    result["timing_arms"] = 40
    result["freeze_sha256"] = hashlib.sha256((evidence / "freeze.json").read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree", type=Path, help="also verify the frozen source bytes")
    args = parser.parse_args()
    result = audit(args.evidence, args.tree)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    Console().print(f"Verified: {len(result['comparisons'])} comparisons, 40 timing arms, 14 gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
