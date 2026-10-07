"""Verify #674's preserved development evidence; never opens a model or final panel."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def calculate_statistics(
    quest: list[dict[str, Any]],
    fp: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    require(bool(quest) and len(quest) == len(fp), "unpaired record lengths")
    for group in (quest, fp):
        identities = []
        for row in group:
            identity = row.get("row_sha256")
            targets = row.get("target_tokens")
            loss = row.get("nll_sum")
            require(
                isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) is not None,
                "invalid row identity",
            )
            require(type(targets) is int and targets > 0, "invalid target count")
            require(
                type(loss) in (int, float) and math.isfinite(loss) and loss >= 0,
                "invalid response loss",
            )
            identities.append(identity)
        require(len(set(identities)) == len(identities), "duplicate row identity")
    require(
        [(row["row_sha256"], row["target_tokens"]) for row in quest]
        == [(row["row_sha256"], row["target_tokens"]) for row in fp],
        "unpaired identities or targets",
    )
    require(type(resamples) is int and 1 <= resamples <= 10000, "invalid bootstrap size")
    targets = np.asarray([row["target_tokens"] for row in fp], dtype=np.float64)
    losses = np.asarray(
        [[row["nll_sum"] for row in group] for group in (quest, fp)], dtype=np.float64
    )
    means = [
        math.fsum(row["nll_sum"] for row in group) / int(targets.sum()) for group in (quest, fp)
    ]
    require(means[1] > 0, "FP NLL must be positive")
    indices = np.random.default_rng(seed).integers(0, len(fp), size=(resamples, len(fp)))
    sampled = losses[:, indices].sum(-1) / targets[indices].sum(-1)
    gap_ci = np.percentile(sampled[0] - sampled[1], [2.5, 97.5]).tolist()
    return {
        "quest_nll": means[0],
        "fp_nll": means[1],
        "nll_ratio": means[0] / means[1],
        "gap": means[0] - means[1],
        "gap_ci95": gap_ci,
        "examples": len(fp),
        "targets": int(targets.sum()),
        "zero_margin_gate": bool(means[0] <= means[1] and gap_ci[1] <= 0),
    }


def verify_evidence(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema"] == "quest-674-postfit-v1", "unknown evidence schema")
    require(data["status"] == "adaptive-development-only", "invalid development claim")
    for name in (
        "training_quality_validated",
        "confirmed_parity",
        "final256_evaluated",
        "oasst256_evaluated",
    ):
        require(data[name] is False, f"unsupported claim: {name}")
    require(data["panel"]["selection_spent"] is True, "selection scope changed")
    require(
        data["bootstrap"]
        == {
            "resamples": 2000,
            "seed": 674,
            "unit": "paired sequence",
            "weighting": "ratio of target loss sums",
        },
        "bootstrap protocol changed",
    )
    result = calculate_statistics(
        data["quest"]["rows"], data["fp"]["rows"], resamples=2000, seed=674
    )
    require(
        result["examples"] == data["panel"]["examples"] == 704
        and result["targets"] == data["panel"]["targets"] == 25017,
        "development roster changed",
    )
    stored = data["best_statistics"]
    for actual, key in (
        (result["quest_nll"], "left_nll"),
        (result["fp_nll"], "right_nll"),
        (result["nll_ratio"], "nll_ratio"),
        (result["gap"], "delta"),
    ):
        require(
            math.isclose(actual, stored[key], rel_tol=0, abs_tol=1e-12), f"summary differs: {key}"
        )
    require(
        len(stored["delta_ci95"]) == 2
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
            for actual, expected in zip(result["gap_ci95"], stored["delta_ci95"], strict=True)
        ),
        "paired interval differs",
    )
    require(not result["zero_margin_gate"], "this record does not authorize confirmation")
    return result


def contained(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "invalid artifact path")
    require(not os.path.isabs(relative), "artifact path must be relative")
    anchor = os.path.realpath(root)
    target = os.path.realpath(root / relative)
    try:
        inside = os.path.commonpath([anchor, target]) == anchor
    except ValueError:
        inside = False
    require(inside, "artifact path escapes declared root")
    return Path(target)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(data: dict[str, Any], root: Path) -> int:
    count = 0
    for arm in ("quest", "fp"):
        folder = contained(root, data[arm]["directory"])
        for relative, expected in data[arm]["file_sha256"].items():
            path = contained(folder, relative)
            require(
                path.is_file() and file_sha(path) == expected, f"artifact differs: {arm}/{relative}"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).parents[1] / "evidence/quest-674-postfit-v1.json",
    )
    parser.add_argument(
        "--artifact-root", type=Path, help="Optional preserved parity-plan-v1 directory"
    )
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = verify_evidence(data)
    if args.artifact_root is not None:
        result["checkpoint_files_verified"] = verify_artifacts(data, args.artifact_root)
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
