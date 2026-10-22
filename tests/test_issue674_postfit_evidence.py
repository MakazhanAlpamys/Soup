"""The committed research record must keep paired weighting and honest claim boundaries."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.harness.quest_parity_evidence import (
    calculate_statistics,
    verify_artifacts,
    verify_evidence,
)


def evidence():
    path = Path(__file__).parents[1] / "benchmarks/evidence/quest-674-postfit-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_losses_reproduce_the_best_retained_result():
    result = verify_evidence(evidence())
    assert result["nll_ratio"] == pytest.approx(1.0383673004974954, abs=1e-12)
    assert result["zero_margin_gate"] is False


def test_statistics_use_token_weighted_loss_sums():
    left = [
        {"row_sha256": "a" * 64, "target_tokens": 1, "nll_sum": 10.0},
        {"row_sha256": "b" * 64, "target_tokens": 9, "nll_sum": 10.0},
    ]
    right = [{**row, "nll_sum": float(row["target_tokens"])} for row in left]
    result = calculate_statistics(left, right, resamples=20, seed=674)
    assert result["quest_nll"] == 2.0
    assert result["fp_nll"] == 1.0


@pytest.mark.parametrize("change", ["order", "duplicate", "target", "nan", "summary"])
def test_changed_pairing_or_loss_is_rejected(change):
    data = copy.deepcopy(evidence())
    rows = data["quest"]["rows"]
    if change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "duplicate":
        rows[1]["row_sha256"] = rows[0]["row_sha256"]
    elif change == "target":
        rows[0]["target_tokens"] = True
    elif change == "nan":
        rows[0]["nll_sum"] = float("nan")
    else:
        data["best_statistics"]["nll_ratio"] = 1.0
    with pytest.raises(ValueError):
        verify_evidence(data)


@pytest.mark.parametrize(
    "claim", ["training_quality_validated", "confirmed_parity", "final256_evaluated"]
)
def test_development_record_cannot_be_relabelled_as_confirmation(claim):
    data = evidence()
    data[claim] = True
    with pytest.raises(ValueError, match="claim"):
        verify_evidence(data)


def test_optional_artifact_verification_refuses_paths_outside_the_root(tmp_path):
    data = evidence()
    data["quest"]["directory"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        verify_artifacts(data, tmp_path)
