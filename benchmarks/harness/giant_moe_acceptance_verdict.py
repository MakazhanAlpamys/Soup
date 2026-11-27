"""Replay the paired quality and RAM rules for a giant MoE run."""

import argparse
import hashlib
import json
from pathlib import Path

from rich.console import Console

from benchmarks.harness.giant_moe_evaluation import load_evaluation, paired_summary
from benchmarks.harness.giant_moe_ram import ram_constraint, summarize_ram
from benchmarks.harness.giant_moe_verdict import verdict


def acceptance_verdict(
    events, index, physical_vram_mib, samples, returncode, protocol_path, data_path
):
    result = verdict(events, index, physical_vram_mib)
    failures = list(result["failures"])
    protocol, _ = load_evaluation(protocol_path, data_path)
    provenance = [r for r in events if r["event"] == "evaluation_protocol"]
    if (
        len(provenance) != 1
        or provenance[0].get("protocol_sha256")
        != hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()
        or provenance[0].get("data_sha256") != protocol["evaluation_data_sha256"]
    ):
        failures.append("evaluation protocol/data provenance mismatch")
    quality = None
    ram = None
    try:
        before = [r for r in events if r["event"] == "evaluation_before"]
        after = [r for r in events if r["event"] == "evaluation_after"]
        quality = paired_summary(before, after, expected_ids=protocol["example_ids"])
        steps = [r for r in events if r["event"] == "step"]
        if (
            len(steps) != 2
            or max(r["unix"] for r in before) >= steps[0]["unix"]
            or min(r["unix"] for r in after) <= steps[-1]["unix"]
        ):
            failures.append("evaluation arms do not surround training")
        baseline = [r for r in events if r["event"] == "unadapted_baseline"]
        if (
            len(baseline) != 1
            or baseline[0].get("zero_lora_b_tensors") != 305
            or baseline[0]["unix"] >= min(r["unix"] for r in before)
        ):
            failures.append("unadapted zero-B baseline is not established")
    except (KeyError, ValueError, TypeError) as exc:
        failures.append("invalid paired evaluation: " + str(exc))
    try:
        constraint = ram_constraint(samples[0])
        ram = summarize_ram(samples, constraint, returncode)
        if not ram["ram_32gib_constrained_pilot_passed"]:
            failures.append("RAM/swap/OOM accounting does not establish a clean constrained pilot")
        environment = next(r for r in events if r["event"] == "environment")
        if samples[0]["unix"] > environment["unix"] or samples[-1]["unix"] < events[-1]["unix"]:
            failures.append("RAM accounting does not span the full model experiment")
    except (IndexError, KeyError, StopIteration, ValueError, TypeError) as exc:
        failures.append("invalid RAM accounting: " + str(exc))
    return {
        "acceptance_measurements_complete": not failures,
        "failures": failures,
        "original_pilot_checks_passed": result["pilot_checks_passed"],
        "paired_quality": quality,
        "ram": ram,
        "owner_reviewer_acceptance": "pending external review",
        "scope": "eight short paired text fragments; recorded <=32GiB RAM constraint; "
        "no broad quality, convergence or cross-hardware speed claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("events", "index", "ram-samples", "protocol", "evaluation-data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--physical-vram-mib", type=int, required=True)
    parser.add_argument("--returncode", type=int, required=True)
    args = parser.parse_args()

    def rows(path):
        return [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
        ]

    answer = acceptance_verdict(
        rows(args.events),
        json.loads(Path(args.index).read_text(encoding="utf-8")),
        args.physical_vram_mib,
        rows(args.ram_samples),
        args.returncode,
        args.protocol,
        args.evaluation_data,
    )
    Path(args.output).write_text(json.dumps(answer, indent=2), encoding="utf-8")
    Console().print(answer)
    raise SystemExit(0 if answer["acceptance_measurements_complete"] else 1)
