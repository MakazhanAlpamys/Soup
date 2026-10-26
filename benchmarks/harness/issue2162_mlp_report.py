"""Render the MLP record from its independently recomputed audit and raw receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from issue2162_mlp_record import audit_record, decode_file
from rich.console import Console


def render(root: Path, result: dict) -> str:
    audit = result["budget"]
    objects = json.loads((root / "objects.json").read_text())
    references = json.loads(
        (root.parent / "issue2162-rtx3090-v1/reference-updates.json").read_text()
    )
    mapping = json.loads((root / "mlp-budget-b-v1/files.json").read_text())

    def raw(name):
        return decode_file(objects[mapping[name]], references["2048"]).decode()

    def read(name):
        return json.loads(raw(name))

    freeze, environment = read("freeze.json"), read("environment.json")
    lines = [
        "# QKVO and MLP retention against native auto",
        "",
        "Refs #2162. **The fixed B allocation passed its two-stage measurement rule:** "
        "3.625068% less time per step in the six-block primary and 3.422014% in six "
        "fresh confirmation pairs. Both family-adjusted intervals have positive lower "
        "bounds. This record covers the measurement and benchmark code; a production "
        "option still needs a separate scope and review.",
        "",
        "## Results",
        "",
        "Each process contributes its median of 50 synchronized step durations. "
        "Each block/pair contributes 100 × (baseline − candidate) / baseline; the reported "
        "gain is the median of those six reductions. Positive means less time per step.",
        "",
        "| Comparison | Median reduction | Ordinary 95% | Adjusted 99.583333% |",
        "|---|---:|---|---|",
    ]
    for label, value in [
        *audit["stage_a"].items(),
        ("independent confirmation", audit["confirmation"]),
    ]:
        low, high = value["ordinary_95"]
        alow, ahigh = value["family_adjusted_ci"]
        lines.append(
            f"| {label} | {value['median_reduction_percent']:.6f}% | "
            f"[{low:.6f}%, {high:.6f}%] | [{alow:.6f}%, {ahigh:.6f}%] |"
        )
    lines += [
        "",
        "The primary compares mixed retention with the original native auto K11. "
        "The same-K QKVO-only control isolates the added gate/up retention: mixed improves "
        "3.127229% over that control, while the control improves 0.249797% over native. "
        "This supports saving gate/up outputs in this allocation. It does not make the "
        "earlier QKVO-only or projection-scoped results pass.",
        "",
        "The frozen rule requires primary median ≥2% and positive ordinary and corrected "
        "lower bounds, plus positive mixed-versus-QKVO lower bounds and every correctness, "
        "memory and resource gate. Only then may six new AB/BA confirmation pairs run. "
        "Confirmation must independently meet median ≥2% and both positive lower bounds.",
        "",
        "Bootstrap settings: 10,000 whole-block/whole-pair resamples, NumPy default_rng, "
        "seed 42 for the primary and 73 for confirmation, linear quantiles. Family 12 "
        "includes eight earlier contrasts, these three contrasts and confirmation; adjusted "
        "quantiles are 0.05/24 and 1−0.05/24. Six observations make the bootstrap discrete. "
        "These are small experiments on one card, not a guarantee of population coverage.",
        "",
        "## Protocol and source identity",
        "",
        "The [public B protocol]"
        "(https://github.com/MakazhanAlpamys/Soup/issues/2162#issuecomment-6426785053) "
        "was posted before any B model process. The original wording is preserved in "
        "[RULE.md](results/issue2162-mlp-v1/mlp-budget-b-v1/RULE.md).",
        "",
        f"- Frozen: `{freeze['at']}`; publication: `{read('publication.json')['created_at']}`.",
        f"- Protocol SHA256: `{freeze['protocol_sha256']}`.",
        f"- Source ZIP SHA256: `{freeze['archive_sha256']}`.",
        "- Base: `094644e30fd714190562a1dcf3b38f8680a091c5`, "
        "plus the frozen benchmark helpers/tests. "
        "The helpers were uncommitted at measurement time. Their exact bytes are now bound "
        "to Git blobs in `source-blobs.json`, with explicit LF/CRLF reconstruction. "
        "The source ZIPs are retained locally and in the lab, rather than checked in.",
        "",
        "## Allocation, recipe and exactness",
        "",
        "| Arm | Whole retained layers | Cached frozen base outputs | Raw cache bytes |",
        "|---|---|---|---:|",
        "| native | 21..31 (K11) | none | 0 |",
        "| qkvo | 26..31 (K6) | Q/K/V/O in 0..25 | 1,090,519,040 |",
        "| mixed | 26..31 (K6) | same QKVO; gate/up in 15..25 | 2,382,364,672 |",
        "",
        "Mistral-7B NF4/double quantization, bf16, batch 1 × sequence 2048, pinned RAM "
        "and two stream buffers. The original model/shard identity is preserved. LoRA "
        "targets q/k/v/o, rank 8, alpha 16, dropout 0; 32 layers × four targets × A/B "
        "gives 256 trainable tensors. Model seed 42, nonzero B seed 7, input seed 1234; "
        "PagedAdamW8bit at 2e-4, deterministic algorithms, TF32 off, OMP/MKL threads 1. "
        "The observed backend is SDPA efficient attention.",
        "",
        "Six initial processes checked normal/profile execution for all three arms. "
        "Thirty timing processes each began with two exact reference updates, then finished "
        "ten warmups and 50 measured updates. Totals: **72 exact reference-update replays, "
        "1,500 measured updates, 300 warmups and 1,812 optimizer updates**. Logits, loss, "
        "all 256 gradients and all 256 updated tensors matched bit for bit. The profiler "
        "confirmed removal of 44 selected gate/up recomputation matmuls over two mixed updates.",
        "",
        "## Memory",
        "",
        "```text",
        *read("gate-native-normal.json")["messages"],
        "```",
        "",
        "Both the allocator and `training.stream_vram_override` enforce 8,589,934,592 "
        "bytes. The census reproduces charge 482,015,738 bytes per whole layer and a "
        "5,756,682,240-byte budget. B charges 2,734,318,038 cache bytes within the K6 "
        "residual 2,864,587,812 bytes, leaving 130,269,774 charged bytes.",
        "",
        "Peak allocated/reserved memory: **6.439509 / 6.759766 GiB**. All 30 allocated-increase "
        "checks passed: mixed−QKVO, mixed−native and QKVO−native in each gate variant and "
        "primary block, plus mixed−native in every confirmation pair. The allowance is "
        "saved raw bytes + 64 MiB. Mixed−native increases allocated memory by 366,390,272 "
        "bytes, net of the five whole layers no longer retained. Mixed−QKVO adds exactly "
        "1,291,845,632 bytes for the MLP outputs.",
        "",
        "## Hardware and activity",
        "",
        f"Environment snapshot recorded during the gate phase: `{environment['platform']}`; "
        f"Python `{environment['python']}`. Packages: "
        + ", ".join(f"{k} `{v}`" for k, v in environment["packages"].items())
        + ".",
        "",
        "```text",
        environment["nvidia"].strip(),
        "```",
        "",
    ]
    samples = []
    for entry in freeze["timings"] + freeze["confirmation"]:
        timing = read(entry["tag"] + "-timing.json")
        samples += [
            json.loads(line)
            for line in raw(entry["tag"] + "-resources.jsonl").splitlines()
            if timing["window_start_unix_s"]
            <= json.loads(line)["unix_s"]
            <= timing["window_end_unix_s"]
        ]
    selected = [g for s in samples for g in s["gpus"] if int(g[0]) == 0]
    others = [g for s in samples for g in s["gpus"] if int(g[0]) != 0]
    lines += [
        f"Measured-window samples: selected GPU {min(float(g[5]) for g in selected):g}–"
        f"{max(float(g[5]) for g in selected):g} °C, SM clocks "
        f"{min(float(g[7]) for g in selected):g}–{max(float(g[7]) for g in selected):g} MHz; "
        f"other GPUs at most {max(float(g[4]) for g in others):g}% utilization. "
        "Maximum sample gap: "
        f"{max(s['max_gap_s'] for s in audit['timing_sampling'].values()):.6f} s. "
        "No sampled Windows/Linux CI worker was active. Sampling does not rule out "
        "activity between probes or host contention. The 24 GiB RTX 3090 was capped at "
        "8 GiB; this does not establish performance on a physical 8 GiB card.",
        "",
        "## Every timing process",
        "",
        "Executed order, median seconds per step, and allocated/reserved bytes including warmups. "
        "All 50 durations and losses per process are reconstructible from the evidence.",
        "",
        "| Process | Median seconds | Allocated bytes | Reserved bytes |",
        "|---|---:|---:|---:|",
    ]
    for entry in freeze["timings"] + freeze["confirmation"]:
        r = read(entry["tag"] + "-timing.json")
        lines.append(
            f"| {entry['tag']} | {r['median_step_s']:.9f} | "
            f"{r['peak_allocated']:,} | {r['peak_reserved']:,} |"
        )
    lines += [
        "",
        "## Earlier diagnostic and historical comparisons",
        "",
        "Candidate A removed the intended gate/up work, but its analytical planning "
        "estimate was 2.971433%, below the fixed 3% screen. It had no timing pairs and "
        "remains ineligible under that protocol. B was a separately declared allocation "
        "with an analytical 3.990470% estimate. A's failed screen remains failed.",
        "",
        "A's first profiler process failed because ContextVar phase labels did not reach "
        "the autograd thread. Its two exact updates remain recorded. Recovery reused "
        "the three successful normal gates and retried only the failed profiler, using "
        "a thread-visible observer that saved raw events. A therefore has 12 accepted "
        "updates plus two failed-profile updates, 14 total. The corrected parser "
        "distinguishes zero-device down-projection early-stop boundaries from executed "
        "QKVO/gate/up kernels. A generated pytest metadata file absent from the source ZIP "
        "is explicitly identified in the source bindings; no missing source is silently skipped.",
        "",
        "The original QKVO outcomes and verdicts remain in "
        "[the earlier report](gate-2162-auto-qkvo.md). "
        "Below are their descriptive intervals recomputed with family 12; these do not "
        "replace the original decisions.",
        "",
        "| Prior comparison | Median reduction | Family-12 interval |",
        "|---|---:|---|",
    ]
    for row in audit["history"]:
        r = row["new_family_summary"]
        low, high = r["family_adjusted_ci"]
        label = row.get("study", "") + " / " + str(row.get("contrast", row.get("sequence", "")))
        lines.append(
            f"| {label} | {r['median_reduction_percent']:.6f}% | [{low:.6f}%, {high:.6f}%] |"
        )
    lines += [
        "",
        "## Reproduce without CUDA",
        "",
        "The compact record reconstructs 281 original files byte for byte, including "
        "raw profiler events and telemetry. Tensor hashes are shared with the earlier "
        "record; repeated profiler/GPU strings use dictionaries. Every reconstructed "
        "file is checked against its original SHA256 before the two independent auditors "
        "run. Their complete outputs must equal the original closure audits.",
        "",
        "```sh",
        "python benchmarks/harness/issue2162_mlp_record.py "
        "--evidence benchmarks/results/issue2162-mlp-v1 "
        "--verify-git-sources . --output audit-2162-mlp.json",
        "python benchmarks/harness/issue2162_mlp_report.py "
        "--evidence benchmarks/results/issue2162-mlp-v1 --output report-2162-mlp.md",
        "```",
        "",
        "The closure verified four services, two real inference requests, three containers, "
        "original power limits and 41 model/shard hashes. Full original ZIPs, closure "
        "receipts and reports remain preserved locally and in the lab. This submission "
        "does not rerun any model or change production code.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_record(args.evidence)
    args.output.write_text(render(args.evidence, result), encoding="utf-8", newline="\n")
    Console().print({"verified": True, "report": str(args.output)})


if __name__ == "__main__":
    main()
