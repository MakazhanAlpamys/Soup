"""The estimate against runs whose rows do not fill ``max_length``.

``benchmarks/evidence/vram-qwen3-0.6b-realised-length.json`` holds Qwen3 training peaks on an
RTX 5070 Laptop (8 GB), each with the length of its longest training row: 17 torch-reserved
peaks and a few clean device readings from a downstream application's runner, and five
``soup bench train`` runs (torch's allocated and reserved peaks beside a 200 ms whole-card
sampler) that include a run whose rows fill the length. The estimate prices every micro-batch
as ``max_length x batch`` real tokens (the rows of its calibration table,
``test_vram_estimate_calibration.py``, all fill ``max_length``), so on short data it is
several times the peak: 10.19 GB against 2.1 GB allocated. The coefficients hold for filled
rows; what is wrong for short data is the length. ``profile_from_config(..., seq_len=)`` prices
the length the data really has, and these tests hold that number to what was measured:

* priced at ``data.max_length`` (no ``seq_len``) it is still the 10.19 GB it was, so the guard
  #2146 added for filled rows is not undone;
* priced at the data's own length, its allocated terms match ``max_memory_allocated`` on the
  bf16 runs, and the estimate with the pre-flight's 10% margin is above every clean
  whole-card peak, by 0.2% on the tightest (a filled micro-batch). The estimate WITHOUT the
  margin is not: it is below the device peak for three of those runs. Outside callers must
  keep the margin.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import model_shape
from soup_cli.utils.hardware_fit import (
    decide_hardware_fit,
    estimate_peak_vram_gb,
    hardware_fit_input_or_reason,
)
from soup_cli.utils.profiler import profile_from_config
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN, fit_verdict

EVIDENCE = (
    Path(__file__).parents[1] / "benchmarks/evidence/vram-qwen3-0.6b-realised-length.json"
)
MIB = 2**20
GB = 1e9
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

#: est_total / torch-reserved at the real micro-batch, for the downstream runner's peaks.
#: Measured 1.097 to 1.238 over its 17 runs, 1.186 on Qwen3-1.7B. The bounds are as tight as
#: that allows. Reserved peaks of different runners differ by about 10% (the same recipe was
#: 3388 MiB there and 3732 MiB in ``soup bench train``), so these bounds are for that runner.
TOTAL_OVER_TORCH = (1.05, 1.30)
#: The allocated terms (everything but ``overhead``) over the downstream runner's reserved peak.
#: Measured 0.911 to 0.964.
ALLOCATED_OVER_TORCH = (0.88, 1.00)
#: est allocated terms / ``max_memory_allocated`` from ``soup bench train``. bf16 measured 0.957
#: (a filled micro-batch), 1.008 and 1.017; 4-bit 0.841 (the estimate is 16% under there: the
#: 4-bit allocated terms are a known shortfall, so the bound is where it is, not where it
#: should be).
ALLOCATED_OVER_ALLOCATED = {"none": (0.94, 1.06), "4bit": (0.80, 1.10)}
#: est_total x (1 + VRAM_SAFETY_MARGIN) / the clean whole-card peak. Measured 1.002 (a filled
#: micro-batch, the tightest) to 1.354. Below 1.0 a run that fits would be refused or a run
#: that does not fit accepted on the margin alone; above 1.45 the estimate over-prices.
VERDICT_OVER_DEVICE = (1.0, 1.45)

FILLED_WORST_CASE_GB = (10.15, 10.25)
TOKEN_SHARE_AT_MAX_LENGTH = 0.70


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _text(row: dict, base: str) -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": "sft",
            "data": {"train": "train.jsonl", "format": "alpaca", "max_length": row["max_length"]},
            "training": {
                "epochs": 1,
                "batch_size": row["batch_size"],
                "gradient_accumulation_steps": row.get("gradient_accumulation_steps", 4),
                "quantization": row["quantization"],
                "lora": {
                    "r": row["lora_r"],
                    "alpha": row["lora_alpha"],
                    "target_modules": row["target_modules"],
                },
            },
        }
    )


@pytest.fixture
def models(monkeypatch):
    """The shapes the evidence file records, with no cache or network read."""
    known = _evidence()["models"]
    monkeypatch.setattr(
        model_shape, "_hub_lookup", lambda repo, **kw: (dict(known[repo]["config"]), None)
    )
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: known[repo]["params"])
    return known


def _base(row: dict) -> str:
    return row.get("model", DEFAULT_MODEL)


def _own_length(row: dict):
    """The length a run is priced at: its longest row, or None when its rows fill max_length."""
    return None if row.get("fills_max_length") else row["longest_row_tokens"]


def _priced(row: dict, seq_len="own") -> dict:
    """The estimate for ``row`` at ``seq_len`` ("own": its longest row; None: ``max_length``).

    Every 4-bit row was measured before #2683, when peft's fp32 embeddings and head were kept
    on a bf16 card, so those are priced the way they ran (``kbit_fp32``) through
    ``hardware_fit``; ``profile_from_config`` has no such argument and prices the bf16 default,
    which is what a 4-bit run on a bf16 card does now (no such run is measured yet).
    """
    seq = _own_length(row) if seq_len == "own" else seq_len
    cfg = load_config_from_string(_text(row, _base(row)))
    if row["quantization"] == "none":
        result = profile_from_config(cfg, allow_network=False, seq_len=seq)
        assert isinstance(result, dict), result
        return result
    inp = hardware_fit_input_or_reason(
        cfg, allow_network=False, profile_defaults=True, kbit_fp32=True
    )
    if seq is not None:
        inp = dataclasses.replace(inp, seq_len=max(64, min(seq, inp.seq_len)))
    estimate = estimate_peak_vram_gb(inp).estimate
    return {"total_memory_bytes": estimate.total_bytes, "memory_terms_bytes": dict(estimate.terms)}


def _allocated_terms(result: dict) -> int:
    return result["total_memory_bytes"] - result["memory_terms_bytes"]["overhead"]


_DATA = _evidence()
_TORCH_ROWS = list(_DATA["rows"])
_ADDED_ROWS = list(_DATA["downstream_added_rows"])
_BENCH_ROWS = list(_DATA["bench_rows"])
_CLEAN_ROWS = [
    row
    for row in [*_TORCH_ROWS, *_ADDED_ROWS, *_BENCH_ROWS, *_DATA["device_only_rows"]]
    if row.get("device_clean") and row.get("device_mib")
]


def _ids(rows):
    return [row["id"] for row in rows]


def test_the_evidence_is_complete_and_says_who_measured_what():
    data = _evidence()
    assert len(data["rows"]) == 17
    assert {row["torch_reserved_mib"] for row in data["rows"]} == {
        2114, 2274, 2662, 2720, 2746, 2808, 3360, 3388,
    }
    assert [row["id"][:2] for row in data["bench_rows"]] == ["M1", "M2", "M3", "M4", "M6"]
    assert len(data["downstream_added_rows"]) == 3
    assert set(data["models"]) == {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"}
    assert "bench train" in data["description"] and "downstream" in data["description"]
    assert "reserved" in data["peak_kinds"]["torch_reserved_mib"]
    assert all(row["longest_row_tokens"] >= 1 for row in data["rows"])
    # Downstream product names stay out of a public evidence file.
    assert "zero" not in json.dumps(data).lower()


def test_the_cuda_context_and_the_device_offset_are_recorded_as_measured():
    context = _evidence()["cuda_context"]
    assert context["device_mib_with_cuda_and_cublas_initialised_and_nothing_else"] == 226
    assert context["device_minus_reserved_mib_in_every_bench_row"] == 162
    for row in _BENCH_ROWS:
        reserved_mib = round(row["max_memory_reserved_bytes"] / MIB)
        assert row["device_mib"] - reserved_mib == 162, row["id"]
        assert row["device_baseline_mib"] == 0, row["id"]


# --- 1. priced at max_length it is still the filled-rows number ---------------------------


def test_the_helpdesk_recipe_still_prices_its_filled_worst_case(models):
    row = _TORCH_ROWS[0]
    result = _priced(row, seq_len=None)
    total_gb = result["total_memory_bytes"] / GB
    assert FILLED_WORST_CASE_GB[0] <= total_gb <= FILLED_WORST_CASE_GB[1], total_gb
    terms = result["memory_terms_bytes"]
    share = (terms["activations"] + terms["logits"]) / result["total_memory_bytes"]
    assert share >= TOKEN_SHARE_AT_MAX_LENGTH, share
    assert result["seq_len"] == row["max_length"]
    # The two token terms are what make it so (1.87 and 1.82 MB per token).
    assert terms["logits"] == pytest.approx(151936 * 2 * row["max_length"] * 12, rel=1e-3)


def test_a_run_whose_rows_fill_the_length_is_still_refused_on_the_8_gb_card(models):
    row = _TORCH_ROWS[0]
    inp = hardware_fit_input_or_reason(
        load_config_from_string(_text(row, _base(row))), allow_network=False
    )
    assert not decide_hardware_fit(inp, available_vram_gb=8151 * MIB / GB).ok
    # ... and a 1024-token row through seq_len is the same run.
    full = _priced(row, seq_len=row["max_length"])
    assert full["total_memory_bytes"] == _priced(row, seq_len=None)["total_memory_bytes"]


# --- 2. the downstream runner's torch peaks, priced at the real micro-batch ---------------


@pytest.mark.parametrize("row", _TORCH_ROWS, ids=_ids(_TORCH_ROWS))
def test_each_measured_run_is_priced_above_its_torch_peak_and_not_far_above(models, row):
    result = _priced(row)
    torch_bytes = row["torch_reserved_mib"] * MIB
    total = result["total_memory_bytes"] / torch_bytes
    allocated = _allocated_terms(result) / torch_bytes
    assert TOTAL_OVER_TORCH[0] <= total <= TOTAL_OVER_TORCH[1], (row["id"], total)
    assert ALLOCATED_OVER_TORCH[0] <= allocated <= ALLOCATED_OVER_TORCH[1], (row["id"], allocated)
    assert result["seq_len"] == max(64, row["longest_row_tokens"])


def test_qwen3_1_7b_is_priced_above_its_torch_peak_with_its_tied_head_counted_twice(models):
    # Its checkpoint stores lm_head.weight although the config ties it, so the weights term
    # carries 0.62 GB too many (0.31 GB for the 0.6B): the allocated terms sit above the
    # reserved peak here (1.035), unlike the 0.6B rows (0.91 to 0.96).
    row = _ADDED_ROWS[0]
    assert row["id"].startswith("B1")
    result = _priced(row)
    torch_bytes = row["torch_reserved_mib"] * MIB
    assert TOTAL_OVER_TORCH[0] <= result["total_memory_bytes"] / torch_bytes <= TOTAL_OVER_TORCH[1]
    assert 0.98 <= _allocated_terms(result) / torch_bytes <= 1.06


@pytest.mark.parametrize("row", _TORCH_ROWS, ids=_ids(_TORCH_ROWS))
def test_priced_at_max_length_each_run_is_three_to_five_times_its_peak(models, row):
    # What the downstream application saw: 3.0x to 4.6x. Pinned so that the over-estimate
    # this change works around is on record next to the number that fixes it.
    ratio = _priced(row, seq_len=None)["total_memory_bytes"] / (row["torch_reserved_mib"] * MIB)
    assert 2.8 <= ratio <= 4.8, (row["id"], ratio)


# --- 3. soup bench train: allocated peaks, and the device peak with the margin ------------


@pytest.mark.parametrize("row", _BENCH_ROWS, ids=_ids(_BENCH_ROWS))
def test_the_allocated_terms_match_the_allocated_peak_of_soup_bench_train(models, row):
    ratio = _allocated_terms(_priced(row)) / row["max_memory_allocated_bytes"]
    low, high = ALLOCATED_OVER_ALLOCATED[row["quantization"]]
    assert low <= ratio <= high, (row["id"], ratio)


@pytest.mark.parametrize("row", _CLEAN_ROWS, ids=_ids(_CLEAN_ROWS))
def test_the_estimate_with_the_verdict_margin_is_above_every_clean_device_peak(models, row):
    total = _priced(row)["total_memory_bytes"]
    ratio = total * (1 + VRAM_SAFETY_MARGIN) / (row["device_mib"] * MIB)
    assert VERDICT_OVER_DEVICE[0] <= ratio <= VERDICT_OVER_DEVICE[1], (row["id"], ratio)


def test_the_tightest_clean_row_is_a_filled_micro_batch_and_clears_the_margin_by_a_hair(models):
    ratios = {
        row["id"]: _priced(row)["total_memory_bytes"]
        * (1 + VRAM_SAFETY_MARGIN)
        / (row["device_mib"] * MIB)
        for row in _CLEAN_ROWS
    }
    tightest = min(ratios, key=ratios.get)
    assert tightest.startswith("M6"), ratios
    assert 1.0 <= ratios[tightest] <= 1.02, ratios[tightest]
    # Every other clean row has at least 4% to spare (the next tightest is M3, 1.0497).
    assert all(value >= 1.04 for key, value in ratios.items() if key != tightest), ratios


@pytest.mark.parametrize("prefix", ["M3", "M4", "M6"])
def test_the_estimate_alone_is_not_an_upper_bound_of_the_device_peak(models, prefix):
    # Recorded so that nobody quotes est_total as "the peak": on three clean runs of this card
    # it is below the whole-card peak, and only the 10% the verdict adds covers the rest.
    row = next(row for row in _BENCH_ROWS if row["id"].startswith(prefix))
    total = _priced(row)["total_memory_bytes"]
    device = row["device_mib"] * MIB
    assert total < device <= total * (1 + VRAM_SAFETY_MARGIN), (row["id"], total / device)


def test_the_peak_does_not_depend_on_max_length_for_short_data_and_the_estimate_follows(models):
    m1, m2 = _BENCH_ROWS[0], _BENCH_ROWS[1]
    assert (m1["max_length"], m2["max_length"]) == (1024, 256)
    for key in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "device_mib"):
        assert m1[key] == m2[key], key
    card = 8151 * MIB
    # Priced at max_length the 1024 recipe does not fit the card; the 256 one does, and the
    # card ran both at the same 2462 MiB.
    assert fit_verdict(_priced(m1, seq_len=None)["total_memory_bytes"], card) == "does not fit"
    assert fit_verdict(_priced(m2, seq_len=None)["total_memory_bytes"], card) == "fits"


def test_a_4_bit_run_on_a_bf16_card_is_priced_below_the_runs_measured_before_2609(models):
    # The 4-bit rows ran with peft's fp32 embeddings and head. profile_from_config prices the
    # bf16 default of #2683: 2 bytes less per embedding-and-head parameter (155,582,464 for a
    # tied Qwen3-0.6B), and nothing else. No 4-bit run has been measured on that code.
    row = next(row for row in _BENCH_ROWS if row["quantization"] == "4bit")
    as_measured = _priced(row)
    cfg = load_config_from_string(_text(row, _base(row)))
    default = profile_from_config(cfg, allow_network=False, seq_len=_own_length(row))
    assert as_measured["memory_terms_bytes"]["weights"] - default["memory_terms_bytes"][
        "weights"
    ] == pytest.approx(2 * 151936 * 1024, abs=2)
    assert default["total_memory_bytes"] < as_measured["total_memory_bytes"]


# --- 4. what a shorter length does to the estimate ----------------------------------------


@pytest.mark.parametrize("seq_len", [64, 79, 147, 256, 512, 1023, 1024, 4096])
def test_pricing_a_shorter_length_never_raises_the_estimate(models, seq_len):
    # Monotone, so a recipe the pre-flight accepts is never refused by passing seq_len.
    row = _TORCH_ROWS[0]
    full = _priced(row, seq_len=None)["total_memory_bytes"]
    assert _priced(row, seq_len=seq_len)["total_memory_bytes"] <= full


def test_the_estimate_grows_with_the_priced_length(models):
    row = _TORCH_ROWS[0]
    totals = [
        _priced(row, seq_len=length)["total_memory_bytes"] for length in (64, 128, 512, 1024)
    ]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_the_two_token_terms_scale_with_the_length_and_the_rest_does_not(models):
    row = _TORCH_ROWS[0]
    short, long_ = _priced(row, seq_len=128), _priced(row, seq_len=1024)
    for term in ("weights", "adapters", "gradients", "optimizer"):
        assert short["memory_terms_bytes"][term] == long_["memory_terms_bytes"][term]
    for term in ("activations", "logits"):
        ratio = long_["memory_terms_bytes"][term] / short["memory_terms_bytes"][term]
        assert ratio == pytest.approx(8, rel=0.01), term


def test_the_helpdesk_run_fits_the_8_gb_card_when_priced_at_its_data(models):
    # The same recipe the pre-flight refuses at 1024 x 2 is "fits" (with its 10% margin) at
    # its longest micro-batch: 79 tokens x 2.
    row = _TORCH_ROWS[0]
    card_bytes = 8151 * MIB
    at_max_length = _priced(row, seq_len=None)["total_memory_bytes"]
    at_its_data = _priced(row)["total_memory_bytes"]
    assert fit_verdict(at_max_length, card_bytes) == "does not fit"
    assert fit_verdict(at_its_data, card_bytes) == "fits"
