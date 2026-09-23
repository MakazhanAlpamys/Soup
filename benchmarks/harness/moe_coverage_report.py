"""Turn `moe_expert_coverage.py` JSON into the tables a record can carry.

Separate from the measurement on purpose: the probe writes raw numbers and this
reads them, so a table can be regenerated or re-cut without re-running anything,
and a reader can check the arithmetic between the two.

Prints, per model:

* a **per-corpus x per-shape** table of mean union coverage with the uniform
  baseline beside it, the min and max layer, the skew, and the two
  one-layer-ahead prefetch hit rates;
* a **per-layer** table for one chosen shape, because a mean over layers hides
  the shape of the stack -- early and late layers are known to route
  differently and the decision rule is about what a step reads, not an average;
* the **verdict inputs** the record's rule needs, and nothing else: this script
  deliberately does not decide anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List


def fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def baseline_cell(measured: float, baseline: float) -> str:
    """Coverage against chance, in the one form that cannot be misread."""
    if baseline >= 0.9999995:
        return f"{measured:.3f} (chance 1.000)"
    return f"{measured:.3f} (chance {baseline:.3f})"


def per_corpus_table(data: Dict[str, Any]) -> List[str]:
    meta = data["meta"]
    lines = [
        f"**{meta['model']}** - `{meta['model_type']}`, {meta['hidden_layers']} layers, "
        f"{meta['n_experts']} experts, top-{meta['top_k']}, "
        f"{'NF4' if meta.get('load_4bit') else meta['dtype']} on {meta['device']}, "
        f"{meta['batches_per_shape']} steps per shape",
        "",
        "| corpus | shape | coverage (vs chance) | min layer | max layer | Gini | top-25% share | "
        "hot-25 Jaccard step-to-step | caught by prev layer's set | caught by corpus hot-25 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus in data["corpora"]:
        for shape, summary in corpus["shapes"].items():
            layers = summary["layers"]
            base = summary["uniform_baseline_coverage"]
            gini = summary["gini_mean_over_layers"]
            top25 = sum(one["top25_share"] for one in layers) / len(layers)
            jac = [
                one["hot25_jaccard_step_to_step"]
                for one in layers
                if one["hot25_jaccard_step_to_step"] is not None
            ]
            lines.append(
                f"| {corpus['label']} | {shape} | "
                f"{baseline_cell(summary['coverage_mean_over_layers'], base)} | "
                f"{fmt(summary['coverage_min_layer'])} | {fmt(summary['coverage_max_layer'])} | "
                f"{fmt(gini)} | {fmt(top25)} | {fmt(sum(jac) / len(jac)) if jac else '-'} | "
                f"{fmt(summary.get('traffic_caught_by_previous_layers_set_mean'))} | "
                f"{fmt(summary.get('traffic_caught_by_corpus_hot25_mean'))} |"
            )
    return lines


def per_layer_table(data: Dict[str, Any], corpus_label: str, shape: str) -> List[str]:
    corpus = next(
        (one for one in data["corpora"] if one["label"] == corpus_label), data["corpora"][0]
    )
    summary = corpus["shapes"].get(shape) or next(iter(corpus["shapes"].values()))
    lines = [
        "",
        f"Per layer, corpus `{corpus['label']}`, shape `{shape}` "
        f"(chance baseline {summary['uniform_baseline_coverage']:.6f}):",
        "",
        "| layer | coverage | experts touched | Gini | top-10% | top-25% | "
        "caught by prev layer | caught by corpus hot-25 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n_experts = data["meta"]["n_experts"]
    for row in summary["layers"]:
        ahead = row.get("one_layer_ahead") or {}
        lines.append(
            f"| {row['layer']} | {fmt(row['coverage_mean'])} | "
            f"{row['experts_touched_mean']:.1f} / {n_experts} | {fmt(row['gini'])} | "
            f"{fmt(row['top10_share'])} | {fmt(row['top25_share'])} | "
            f"{fmt(ahead.get('traffic_caught_by_previous_layers_set'))} | "
            f"{fmt(ahead.get('traffic_caught_by_corpus_hot25'))} |"
        )
    return lines


def verdict_inputs(data: Dict[str, Any]) -> List[str]:
    """C and S as the rule defines them, over every corpus and shape measured."""
    coverages, skews = [], []
    for corpus in data["corpora"]:
        for summary in corpus["shapes"].values():
            coverages.append(summary["coverage_mean_over_layers"])
            layers = summary["layers"]
            skews.append(sum(one["top25_share"] for one in layers) / len(layers))
    return [
        "",
        f"Verdict inputs: **C = {min(coverages):.3f} to {max(coverages):.3f}** "
        f"over {len(coverages)} corpus/shape combinations; "
        f"**S = {min(skews):.3f} to {max(skews):.3f}**.",
    ]


def compare(left_path: str, right_path: str) -> List[str]:
    """Two runs of the same configuration, differenced statistic by statistic.

    A routing count is deterministic: same weights, same tokens, same seed means
    the same top-k, so two runs of one configuration must agree EXACTLY. That
    makes a re-run a reproducibility check rather than an errand, and any
    difference a finding rather than noise. The corpus digests are compared
    first, because a difference there explains everything after it and means
    nothing else in the comparison is about the model.
    """
    left = json.loads(pathlib.Path(left_path).read_text(encoding="utf-8"))
    right = json.loads(pathlib.Path(right_path).read_text(encoding="utf-8"))
    lines = [
        f"### {pathlib.Path(left_path).name} vs {pathlib.Path(right_path).name}",
        "",
        f"harness sha256: {left['meta'].get('harness_sha256_16', 'not recorded')} vs "
        f"{right['meta'].get('harness_sha256_16', 'not recorded')}",
        f"host state recorded: {'host_before' in left['meta']} vs "
        f"{'host_before' in right['meta']}",
        "",
    ]
    keys = (
        "coverage_mean_over_layers",
        "coverage_min_layer",
        "coverage_max_layer",
        "gini_mean_over_layers",
        "traffic_caught_by_previous_layers_set_mean",
        "traffic_caught_by_corpus_hot25_mean",
    )
    worst = 0.0
    digests_differ = []
    for one, two in zip(left["corpora"], right["corpora"]):
        if one["sha256_16"] != two["sha256_16"]:
            digests_differ.append(f"{one['label']} ({one['sha256_16']} vs {two['sha256_16']})")
        for shape in one["shapes"]:
            a, b = one["shapes"][shape], two["shapes"][shape]
            for key in keys:
                if a.get(key) is None or b.get(key) is None:
                    continue
                worst = max(worst, abs(a[key] - b[key]))
            for row_a, row_b in zip(a["layers"], b["layers"]):
                if row_a["counts"] != row_b["counts"]:
                    lines.append(
                        f"- **per-expert counts differ** at {one['label']} {shape} "
                        f"layer {row_a['layer']}"
                    )
    if digests_differ:
        lines.append(
            "- **corpus digests differ**: " + ", ".join(digests_differ) + " - the two runs "
            "did not read the same text, so nothing below is about the model"
        )
    lines.append(f"- largest difference in any summary statistic: **{worst:.2e}**")
    lines.append(
        "- per-expert counts identical at every layer of every corpus and shape"
        if not any("counts differ" in line for line in lines)
        else "- per-expert counts DIFFER somewhere; see above"
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("json", nargs="+", help="result files from moe_expert_coverage.py")
    parser.add_argument("--layer-corpus", default="prose")
    parser.add_argument("--layer-shape", default="4x512")
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "difference exactly two results of the SAME configuration instead of "
            "tabulating them. Routing is deterministic, so they must agree exactly; "
            "a difference is a finding"
        ),
    )
    args = parser.parse_args()

    if args.compare:
        if len(args.json) != 2:
            parser.error("--compare wants exactly two result files")
        for line in compare(args.json[0], args.json[1]):
            print(line)
        return 0

    for path in args.json:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        print(f"### {pathlib.Path(path).name}")
        print()
        for line in per_corpus_table(data):
            print(line)
        for line in per_layer_table(data, args.layer_corpus, args.layer_shape):
            print(line)
        for line in verdict_inputs(data):
            print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
