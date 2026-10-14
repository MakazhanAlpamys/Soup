"""Replay F4's saved evidence on CPU; never execute its historical GPU scripts."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


def replay(directory: Path) -> dict:
    """Check provenance and recompute descriptive ratios, without a validity verdict."""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    for name, record in provenance["files"].items():
        for path, expected in (
            (directory / record["archive"], record["sha256"]),
            (directory / name, record["packaged_sha256"]),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"source hash mismatch: {path}")
    rule = provenance["rule"]
    if hashlib.sha256((directory / rule["path"]).read_bytes()).hexdigest() != rule["sha256"]:
        raise ValueError("pre-run rule hash mismatch")
    original = directory / "executed-source" / "original-report.md"
    if hashlib.sha256(original.read_bytes()).hexdigest() != provenance["original_report_sha256"]:
        raise ValueError("original report hash mismatch")

    rounds = [json.loads((directory / f"f4_round_{i}.json").read_text()) for i in range(3)]
    curves = {}
    for name in rounds[0]["shapes"]:
        per_round = []
        for data in rounds:
            rows = data["shapes"][name]["rows"]
            reference = next(row["point"] for row in rows if row["M"] == 8192)
            values = {}
            for row in rows:
                ratio = row["point"] / row["denom"]
                if not math.isfinite(ratio) or not math.isclose(ratio, row["ratio"], rel_tol=1e-12):
                    raise ValueError(f"invalid saved ratio: {name}, M={row['M']}")
                values[row["M"]] = (ratio, row["point"] / reference)
            per_round.append(values)
        curves[name] = {
            str(m): {
                "versus_512": statistics.median(data[m][0] for data in per_round),
                "versus_8192": statistics.median(data[m][1] for data in per_round),
            }
            for m in per_round[0]
        }
    layer = [json.loads((directory / f"f4_item4_round_{i}.json").read_text()) for i in range(3)]
    medians = {
        arm: statistics.median(data["arms_ms"][arm] for data in layer)
        for arm in layer[0]["arms_ms"]
    }
    return {
        "kind": "CPU replay of submitted records, not a new performance measurement",
        "validity_verdict": "not established; descriptive only",
        "gemm_rounds": [0, 1, 2],
        "curves": curves,
        "layer_rounds_for_published_medians": [0, 1, 2],
        "additional_preserved_layer_round": 3,
        "layer_observed_medians_ms": medians,
        "c2": "no raw timing supplied; historical attribution withdrawn",
    }


if __name__ == "__main__":
    results = Path(__file__).parent.parent / "results" / "probe-f4-rtx3060"
    (results / "replay-summary.json").write_text(
        json.dumps(replay(results), indent=2) + "\n", encoding="utf-8"
    )
