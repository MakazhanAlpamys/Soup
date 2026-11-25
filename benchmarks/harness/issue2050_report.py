"""Render the audited #2050 results without loading models or selecting new gates."""

from __future__ import annotations

import argparse
import json
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
    slow = json.loads((evidence / "s512-qkv-pairs/p4-B.json").read_text(encoding="utf-8"))
    fast = json.loads((evidence / "s512-qkv-pairs/p0-B.json").read_text(encoding="utf-8"))
    observations = [
        f"QKV/seq512 includes pair {worst['pair']} with a {worst['reduction_pct']:+.3f}% "
        "reduction; that pair remains in the result. This campaign did not isolate "
        "the cause of the short-sequence timing variation. "
        f"QKV/seq2048 has a {long_qkv['median_pct']:.3f}% median against the fixed 2% threshold.",
        f"The slow QKV/seq512 saved arm stayed at {min(slow['step_times_s']):.3f}-"
        f"{max(slow['step_times_s']):.3f} s per step with {slow['gpu']['util']}% sampled GPU "
        f"utilization, versus {min(fast['step_times_s']):.3f}-{max(fast['step_times_s']):.3f} s "
        f"and {fast['gpu']['util']}% in the fast saved arm. This was a sustained slower "
        "state across all 50 steps, rather than a few long steps.",
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
        "With five pairs, the bootstrap median equals the smallest pair in about 5.8% "
        "of resamples, above both lower-tail cutoffs (2.5% and 0.625%). Both intervals "
        "therefore span the observed minimum and maximum here; the family correction "
        "does not change them. QKVO/seq2048 passes with all five pairs above +3.0% "
        "and a median reduction of 3.141%.",
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
        "- The other three RTX 3090s were inventoried, but their activity throughout "
        "the campaign was not checked. Only the selected GPU was sampled during the "
        "timing arms. Shared host CPU or RAM-bandwidth contention cannot be ruled out; "
        "the seq512 slowdown's cause remains unisolated.",
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
        "operations, rather than Python projection calls or inferred FLOPs. Aggregated "
        "counts are retained in the compact process evidence. An unchanged dequantization "
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
        f"[compact process evidence]({prefix}/process-evidence.json) and all four raw pair "
        "directories beside it. "
        "The campaign froze hashes before its first full-size gate and timing arm, "
        "then rechecked them before every subprocess.",
        "",
        "The 14 original gate/profile receipts were validated before folding them into "
        "one file. Each process records SHA256 over its complete two-update tensor-hash "
        "record, per-update summaries, tensor counts and the full-reference comparison. "
        "The digest uses UTF-8 JSON with sorted keys, compact separators and ASCII escapes. "
        "All four comparisons, operator counts and the original independent-audit result "
        "reproduce exactly. The compact data compares digests; the individual tensor hashes "
        "and shape-level profiles remain in the "
        "[original submitted commit](https://github.com/MakazhanAlpamys/Soup/tree/"
        "95a08a21544359ed81ffd0581fc80ce642cbd9e8/benchmarks/results/issue2050-rtx3090-v1).",
        "",
        f"[source-blobs.json]({prefix}/source-blobs.json) pins all 11 frozen files by Git "
        "blob ID and records their original LF/CRLF form. Retrieve a blob with "
        "`git cat-file blob <id>` from the recorded commit; for an exact campaign replay, "
        "restore its recorded line endings and overlay the files onto the recorded base. "
        "The audit's optional `--tree` check verifies both the Git blob identity and the "
        "original frozen SHA256 after reconstructing those line endings. Retained JSON "
        "and text have logical digests that survive checkout newline conversion, so no "
        "binary source archive or special `.gitattributes` is needed.",
        "",
        "```powershell",
        "# From this PR's checkout; audits recorded evidence without a model load",
        "python benchmarks/harness/issue2050_audit.py `",
        "  benchmarks/results/issue2050-rtx3090-v1 `",
        "  --output independent-audit.json",
        "```",
        "",
        "To verify source too, add `--tree <original-source-directory>` with the files "
        "from commit `95a08a21544359ed81ffd0581fc80ce642cbd9e8`. "
        "The current checkout includes later tests and a benchmark installer guard "
        "that refuses any streamed layer with `recompute=False` before patching forwards. "
        "Those edits were not part of the measured campaign, so `--tree .` correctly "
        "rejects them. The frozen sources, evidence and decision rules remain unchanged.",
        "",
        "No #674 checkpoints or datasets were accessed, and no model weights are "
        "included in the evidence. AI-assisted implementation and measurement with OpenAI Codex.",
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
