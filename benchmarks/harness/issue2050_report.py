"""Render the audited #2050 results without loading models or selecting new gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rich.console import Console


def render(evidence: Path, audit_path: Path, output: Path) -> None:
    result = json.loads(audit_path.read_text(encoding="utf-8"))
    if result.get("verified") is not True or len(result.get("comparisons", [])) != 4:
        raise ValueError("a completed independent audit is required")
    freeze = json.loads((evidence / "freeze.json").read_text(encoding="utf-8"))
    prefix = "results/issue2050-rtx3090-v1"
    by_case = {(row["seq"], row["policy"]): row for row in result["comparisons"]}
    short_qkv = by_case[(512, "qkv")]
    worst = min(short_qkv["pairs"], key=lambda pair: pair["reduction_pct"])
    long_qkv = by_case[(2048, "qkv")]
    long_qkvo = by_case[(2048, "qkvo")]
    full_peak = max(pair["a_peak_gb"]["allocated"] for pair in long_qkvo["pairs"])
    saved_peak = max(pair["b_peak_gb"]["allocated"] for pair in long_qkvo["pairs"])
    observations = [
        f"QKV/seq512 includes pair {worst['pair']} with a {worst['reduction_pct']:+.3f}% "
        "reduction; that pair remains in the result. This campaign did not isolate "
        "the cause of the short-sequence timing variation. "
        f"QKV/seq2048 has a {long_qkv['median_pct']:.3f}% median against the fixed 2% threshold.",
        "",
        f"QKVO/seq2048 retained {long_qkvo['saved_bytes'] / 1e9:.9f} GB of projection outputs. "
        f"Across its five pairs, the largest allocated peaks were {saved_peak:.3f} GB "
        f"versus {full_peak:.3f} GB for full recompute. The per-pair memory gate "
        f"{'passed' if long_qkvo['memory_pass'] else 'failed'}. "
        "This measurement adds no public Soup option.",
    ]
    lines = [
        "# #2050: selective attention checkpointing on RTX 3090",
        "",
        "Four paired comparisons on one RTX 3090, with bit-exact correctness gates.",
        "",
        "## Verdict",
        "",
        "| sequence | saved outputs | paired median reduction | 95% interval | "
        "family-adjusted 98.75% interval | memory gate | verdict |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for row in result["comparisons"]:
        interval = lambda key: f"[{row[key][0]:+.3f}%, {row[key][1]:+.3f}%]"  # noqa: E731
        lines.append(
            f"| {row['seq']} | {row['policy'].upper()} | {row['median_pct']:+.3f}% | "
            f"{interval('ci95')} | {interval('ci98_75')} | "
            f"{'pass' if row['memory_pass'] else 'FAIL'} | {row['verdict']} |"
        )
    positive = [
        f"{row['policy'].upper()}/seq{row['seq']}"
        for row in result["comparisons"]
        if row["verdict"] == "CONFIRMED"
    ]
    lines += [
        "",
        (
            "The predeclared rule passed for " + ", ".join(positive) + ". A separate "
            "opt-in/predictor follow-up can evaluate these qualified cases."
            if positive
            else "No comparison passed the predeclared adoption rule. "
            "This record supports keeping the current production behavior."
        ),
        "",
        "Positive reduction means less measured-window wall time. This is a speed "
        "measurement, not a model-quality score. Each comparison has five complete "
        "fresh-process AB/BA pairs; none were excluded or repeated.",
        "",
        *observations,
        "",
        "## Correctness and scope",
        "",
        "All 14 full-size gate/profile processes passed: seq512/2048, full/full repeat, "
        "QKV, QKVO and a separate profile of each mode. Every logit, loss, all 128 LoRA "
        "gradient tensors and all 128 updated parameter tensors matched exactly after "
        "each of two PagedAdamW8bit updates. B was deliberately nonzero in these gates. "
        "The ordinary timing recipe retained its initial zero B and its original seeds.",
        "",
        "The tiny tests additionally cover four real Mistral GQA layers cycling "
        "through two buffers, a three-layer/one-slot synthetic input-gradient check, "
        "cache mutation refusal and removal of all installed forwards.",
        "",
        "The installer saves only frozen base-projection outputs. It does not save "
        "LoRA products or expanded NF4 weights. The native non-reentrant layer checkpoint, "
        "pool refill/prefetch and backward slot-ownership checks remain active.",
        "",
        "Every process verified actual unique retained storage and tensor count:",
        "",
        "| policy | tensors | seq512 bytes | seq2048 bytes |",
        "|---|---:|---:|---:|",
        "| QKV | 96 | 201326592 | 805306368 |",
        "| QKVO | 128 | 335544320 | 1342177280 |",
        "",
        "## Setup and qualifications",
        "",
        f"- Soup base: `{freeze['base']}`; frozen source SHA256s are in "
        f"[freeze.json]({prefix}/freeze.json).",
        f"- GPU: `{freeze['gpu_uuid']}`, RTX 3090, Windows, unchanged 350 W limit.",
        "- Mistral-7B-Instruct-v0.3, unsloth public mirror, pinned revision "
        "`60b99dcbc7f853b7cd18472a607e8d045b96f03e`; all source-file hashes recorded.",
        "- NF4 with double quantization, pinned RAM tier, two buffers, batch 1. LoRA "
        "q/v, r=16, alpha=32, dropout=0; PagedAdamW8bit lr=2e-4; model seed 3/input seed 17.",
        "- Every timing arm: 10 warm-ups plus 50 synchronized measured steps. The metric "
        "is the sum of the existing runner's per-step seconds (stored at 0.0001 s).",
        "- All arms explicitly used deterministic SDPA math, TF32 disabled and CUBLAS "
        "workspace `:4096:8`. These results do not establish default flash-attention "
        "performance, performance on other GPUs, or a safe production default.",
        "- Operator tracing and tensor hashing run only in separate untimed processes. "
        "Their hashes also matched the unprofiled correctness gates. Timing hooks check "
        "cache storage once during warm-up, then remove themselves.",
        "- The wrapper fixes telemetry to the selected physical UUID. The reused #361 "
        "runner's default sampler otherwise reads the first physical GPU.",
        "- Temperatures, admission records and clocks are in each row. No services, "
        "power limits or other GPUs were changed for this campaign.",
        "",
        "## All pairs",
        "",
        "Allocated/reserved peaks are decimal GB and include warm-up, following #361. "
        "The memory gate uses allocated increase <= exact saved bytes + 64 MiB in every "
        "pair. Because original peaks are rounded to 0.001 GB, the audit uses a conservative "
        "1 MB upper rounding allowance when checking this inequality.",
        "",
        "| seq/policy | pair/order | full seconds | saved seconds | reduction | "
        "full alloc/reserved GB | saved alloc/reserved GB |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in result["comparisons"]:
        for pair in row["pairs"]:
            a, b = pair["a_peak_gb"], pair["b_peak_gb"]
            lines.append(
                f"| {row['seq']}/{row['policy']} | {pair['pair']}/{pair['order']} | "
                f"{pair['a_s']:.4f} | {pair['b_s']:.4f} | "
                f"{pair['reduction_pct']:+.3f}% | {a['allocated']}/{a['reserved']} | "
                f"{b['allocated']}/{b['reserved']} |"
            )
    lines += [
        "",
        "## Executed operations in the separate profiles",
        "",
        "Counts below cover two complete updates, separated by original forward and "
        "backward (which includes checkpoint recomputation). They count dispatched "
        "operations, rather than Python projection calls or inferred FLOPs. Full "
        "shape-level counts remain in each profile receipt. An unchanged dequantization "
        "count means that saving GEMM outputs did not remove dequantization for backward.",
        "",
        "| sequence/mode | phase:operator | two-update count |",
        "|---|---|---:|",
    ]
    operation_notes = []
    for seq in (512, 2048):
        full_ops = result["operators"][f"s{seq}-full"]
        for mode in ("qkv", "qkvo"):
            selected = result["operators"][f"s{seq}-{mode}"]
            skipped = (
                full_ops["backward:aten.mm.default"] - selected["backward:aten.mm.default"]
            ) / 2
            unchanged = all(
                full_ops[key] == selected[key] for key in full_ops if "dequantize" in key
            )
            operation_notes.append(
                f"seq{seq} {mode.upper()}: {skipped:g} fewer backward/recompute matrix "
                f"multiplications per update; dequantization counts "
                f"{'unchanged' if unchanged else 'changed'}."
            )
    for key, counts in result["operators"].items():
        for operation, count in counts.items():
            lines.append(f"| {key} | `{operation}` | {count} |")
    lines += ["", " ".join(operation_notes)]
    lines += [
        "",
        "## Decision and reproducibility",
        "",
        "The frozen rule requires median reduction >= 2%, ordinary 95% and family 98.75% "
        "whole-pair bootstrap lower bounds above 0, exactness and the memory gate. "
        "10000 resamples, seed 42. Five pairs give a small, discrete bootstrap sample; "
        "the bounds describe this experiment and do not guarantee population coverage. "
        "All four planned comparisons stay in the correction family.",
        "",
        "See the [pre-timing protocol](protocol-2050-selective-checkpoint.md), "
        f"[independent audit]({prefix}/independent-audit.json), "
        f"[preparation record]({prefix}/preparation-notes.md) and raw receipts/rows beside it. "
        "The campaign froze hashes before its first full-size gate and timing arm, "
        "then rechecked them before every subprocess. The independent local audit also "
        "checks those frozen source bytes.",
        "",
        f"[source-snapshot.zip]({prefix}/source-snapshot.zip) preserves the exact 11 frozen "
        f"source files and their original line endings; [source preservation receipt]"
        f"({prefix}/source-preservation.json) verifies every entry against the freeze. "
        "The evidence directory disables Git text conversion so its byte hashes survive "
        "checkout. For an exact source replay, overlay the snapshot onto a fresh checkout "
        "of the recorded base before running the campaign. The optional audit --tree "
        "check expects these exact source bytes.",
        "",
        "```powershell",
        "# Dedicated root containing source/, model/, model-manifest.json",
        "$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'",
        "python source/benchmarks/harness/issue2050_campaign.py --root D:/experiment --gpu GPU-...",
        "python source/benchmarks/harness/issue2050_audit.py D:/experiment/evidence-v1 `",
        "  --tree source --output independent-audit.json",
        "```",
        "",
        "No #674 checkpoints or datasets were accessed, and no model weights are "
        "included in the evidence. AI-assisted implementation and measurement with OpenAI Codex.",
    ]
    validation = []
    for filename, label in (
        ("tests-lab.json", "Targeted CPU/CUDA NF4 subset"),
        ("tests-full-cpu.json", "Full suite, requested empty CUDA visibility"),
        ("tests-full-single-gpu.json", "Full suite, one GPU, default TEMP"),
        ("tests-full-single-short.json", "Full suite, one GPU, short basetemp only"),
        ("tests-full-single-temp.json", "Full suite, one GPU, short TEMP and basetemp"),
        ("tests-failure-comparison-baseline.json", "Five failures isolated on clean base"),
        ("tests-failure-comparison-source.json", "Same five isolated on candidate"),
    ):
        path = evidence / filename
        if path.exists():
            receipt = json.loads(path.read_text(encoding="utf-8"))
            summaries = [
                line.strip()
                for line in receipt["stdout"].splitlines()
                if re.search(r"\b\d+ (?:passed|failed|errors?|skipped)\b", line)
            ]
            summary = summaries[-1] if summaries else "See receipt"
            validation.append(
                f"| {label} | {receipt['rc']} | {summary} | [receipt]({prefix}/{filename}) |"
            )
    if validation:
        lines += [
            "",
            "## Submission test validation",
            "",
            "These checks ran after the scientific timing campaign. Return code 0 "
            "means success; the nonzero full-suite attempts stopped at five errors "
            "and are retained. The requested empty CUDA mask did not prevent "
            "four-device reports in simulated-CUDA tests. Later attempts exposed "
            "Windows long-path and temporary-directory admission failures. No "
            "production code or scientific result was changed for these attempts. "
            "The final broad run reached 38300 passed and 184 skipped before its "
            "five-failure stop. The same five cases also failed in isolation on "
            "an unchanged checkout of the recorded base and on the candidate; "
            f"see [baseline verification]({prefix}/baseline-verification.json). "
            "The full suite is not green, and tests beyond the early stop remain unrun.",
            "",
            "| test run | return code | pytest summary | evidence |",
            "|---|---:|---|---|",
            *validation,
        ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.audit, args.output)
    Console().print(f"Report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
