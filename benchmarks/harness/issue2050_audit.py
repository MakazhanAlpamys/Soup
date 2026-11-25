"""Independent closure audit for the fixed four-comparison #2050 protocol.

Reads original or compact evidence; never launches a model or changes a freeze.
Writes the verified result only after all checks succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console


def canonical_digest(value: Any) -> str:
    """SHA256 of UTF-8 JSON, sorted keys, compact separators and ASCII escapes."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compact_gate(receipt: dict[str, Any]) -> dict[str, Any]:
    """Fold all tensor hashes only after validating both complete optimizer updates."""
    steps = receipt["steps"]
    assert len(steps) == 2, "two complete updates required"
    summaries = []
    for step in steps:
        assert len(step["gradients"]) == len(step["parameters"]) == 128
        assert step["gradients"].keys() == step["parameters"].keys()
        values = [step["logits"], step["loss"]]
        values += list(step["gradients"].values()) + list(step["parameters"].values())
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in values)
        summaries.append(
            {
                "logits": step["logits"],
                "loss": step["loss"],
                "gradient_count": len(step["gradients"]),
                "parameter_count": len(step["parameters"]),
                "names_sha256": canonical_digest(sorted(step["gradients"])),
                "gradients_sha256": canonical_digest(step["gradients"]),
                "parameters_sha256": canonical_digest(step["parameters"]),
            }
        )
    counts: Counter = Counter()
    for key, value in receipt["operators"].items():
        phase, name, _shapes = json.loads(key)
        counts[f"{phase}:{name}"] += value
    return {
        **{key: value for key, value in receipt.items() if key not in ("steps", "operators")},
        "tensor_hashes_sha256": canonical_digest(steps),
        "tensor_hash_count": 516,
        "step_summaries": summaries,
        "operator_counts": dict(sorted(counts.items())),
    }


def logical_digest(path: Path) -> str:
    """Pin retained data independently of a checkout's LF/CRLF conversion."""
    if path.suffix == ".json":
        return canonical_digest(json.loads(path.read_text(encoding="utf-8")))
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_source(tree: Path, frozen: dict[str, str], bindings: dict[str, Any]) -> None:
    assert frozen.keys() == bindings.keys(), "source roster changed"
    root = os.path.realpath(tree)
    for name, expected in frozen.items():
        path = os.path.realpath(tree / name)
        assert os.path.commonpath([root, path]) == root, "source outside tree"
        blob = Path(path).read_bytes().replace(b"\r\n", b"\n")
        identity = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        binding = bindings[name]
        assert identity == binding["git_blob"], f"source Git blob mismatch: {name}"
        assert binding["frozen_newlines"] in ("lf", "crlf")
        original = blob.replace(b"\n", b"\r\n") if binding["frozen_newlines"] == "crlf" else blob
        assert hashlib.sha256(original).hexdigest() == expected, f"frozen source mismatch: {name}"


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
    compact_path = evidence / "process-evidence.json"
    compact = read("process-evidence.json") if compact_path.exists() else None
    bindings = read("source-blobs.json") if compact is not None else None
    if compact is not None:
        assert compact["schema"] == "issue2050-compact-v1"
        root = os.path.realpath(evidence)
        for name, expected in compact["retained_files"].items():
            path = os.path.realpath(evidence / name)
            assert os.path.commonpath([root, path]) == root, "evidence outside directory"
            assert logical_digest(Path(path)) == expected, f"retained evidence changed: {name}"

    def gate(tag: str) -> dict[str, Any]:
        if compact is None:
            return compact_gate(read(f"{tag}-audit.json"))
        receipt = compact["processes"][tag]
        assert receipt["reference"] == tag.split("-")[0] + "-full"
        assert receipt["matches_reference"] is True
        assert receipt["tensor_hash_count"] == 516
        assert re.fullmatch(r"[0-9a-f]{64}", receipt["tensor_hashes_sha256"])
        assert len(receipt["step_summaries"]) == 2
        for step in receipt["step_summaries"]:
            assert step["gradient_count"] == step["parameter_count"] == 128
            for key in ("logits", "loss", "names_sha256", "gradients_sha256", "parameters_sha256"):
                assert re.fullmatch(r"[0-9a-f]{64}", step[key])
        return receipt

    if tree is not None:
        if bindings is not None:
            verify_source(tree, freeze["files"], bindings["files"])
        else:
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
    if compact is not None:
        assert set(compact["processes"]) == set(expected_tags[:14])
    result: dict[str, Any] = {
        "verified": False,
        "full_size_gate_processes": 14,
        "updates_per_gate": 2,
        "comparisons": [],
        "operators": {},
    }
    versions = None
    for seq in (512, 2048):
        reference = gate(f"s{seq}-full")
        for mode in (
            "full",
            "full-repeat",
            "qkv",
            "qkvo",
            "profile-full",
            "profile-qkv",
            "profile-qkvo",
        ):
            receipt = gate(f"s{seq}-{mode}")
            assert receipt["status"] == "ok"
            assert receipt["tensor_hashes_sha256"] == reference["tensor_hashes_sha256"], (seq, mode)
            assert receipt["step_summaries"] == reference["step_summaries"], (seq, mode)
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
                counts = receipt["operator_counts"]
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
    frozen_bytes = (evidence / "freeze.json").read_bytes()
    if bindings is not None:
        frozen_bytes = frozen_bytes.replace(b"\r\n", b"\n")
        if bindings["freeze_newlines"] == "crlf":
            frozen_bytes = frozen_bytes.replace(b"\n", b"\r\n")
        assert hashlib.sha256(frozen_bytes).hexdigest() == bindings["freeze_original_sha256"]
    result["freeze_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
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
