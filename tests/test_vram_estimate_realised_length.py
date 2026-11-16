"""The estimate against runs whose rows do not fill ``max_length``, and against capped runs.

``benchmarks/evidence/vram-qwen3-0.6b-realised-length.json`` holds Qwen3 training peaks on an
RTX 5070 Laptop (8 GB), each with the length of its longest training row: 17 torch-reserved
peaks and a few clean device readings from a downstream application's runner, ``soup bench
train`` runs (torch's allocated and reserved peaks beside a 200 ms whole-card sampler) that
include two whose rows fill the length, and runs with torch's allocator capped. The estimate
prices every micro-batch as ``max_length x batch`` real tokens (the rows of its calibration
table, ``test_vram_estimate_calibration.py``, all fill ``max_length``), so on short data it is
several times the peak: 10,855 MiB (11.38 GB) against 2108 MiB allocated. The coefficients hold
for filled rows; what is wrong for short data is the length.
``profile_from_config(..., seq_len=)`` prices the length the data really has, and these tests
hold that number to what was measured:

* priced at ``data.max_length`` (no ``seq_len``) it is still the filled-rows number, so the
  guard #2146 added for filled rows is not undone;
* priced at the data's own length, a bf16 run's allocated terms plus the part of the overhead
  torch itself allocates (the cuBLAS workspace and the entropy metric's temporaries) match
  ``max_memory_allocated`` to 1.3% on four recipes;
* the same recipes re-run with torch's allocator capped (``capped_rows``): an uncapped peak
  includes cache the allocator gives back, so what a run needs is the smallest cap that still
  trains plus the CUDA context. The estimate WITH the pre-flight's 10% margin is above that for
  every run that trained, at every cap; the estimate alone is above it for all but one recipe
  (a 256-token filled micro-batch, 2.1% below the smallest cap that trained and above every cap
  that failed). It is below the UNCAPPED whole-card peak on seven of eleven clean runs, and with
  the margin on one (M3, 0.974): that peak is cache, not need.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.hardware_fit import decide_hardware_fit, hardware_fit_input_or_reason
from soup_cli.utils.profiler import profile_from_config
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN, fit_verdict

EVIDENCE = (
    Path(__file__).parents[1] / "benchmarks/evidence/vram-qwen3-0.6b-realised-length.json"
)
MIB = 2**20
GB = 1e9
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

#: est_total / torch-reserved at the real micro-batch, for the downstream runner's peaks.
#: Measured 1.006 to 1.059 over its 17 runs, 1.021 on Qwen3-1.7B. Reserved peaks of different
#: runners differ by about 10% (the same recipe was 3388 MiB there and 3732 MiB in ``soup bench
#: train``), so these bounds are for that runner.
TOTAL_OVER_TORCH = (1.00, 1.10)
#: The allocated terms (everything but ``overhead``) over the downstream runner's reserved peak.
#: Measured 0.826 to 0.860 for the 0.6B rows, 0.922 for Qwen3-1.7B: the overhead now holds
#: what torch allocates beyond the terms (the cuBLAS workspace, the entropy temporaries).
ALLOCATED_OVER_TORCH = (0.80, 0.95)
#: (est allocated terms + the part of the overhead torch allocates) / ``max_memory_allocated``
#: from ``soup bench train``. bf16 measured 1.001, 1.003, 1.010 and 1.013 (the four recipes the
#: overhead was fitted to and checked on). 4-bit measured 0.803 (before #2683, priced with fp32
#: embeddings and head, and the tied head counted once) and 1.101 (a filled 1,024-token
#: micro-batch): the 4-bit terms are not recalibrated and bracket their peaks.
ALLOCATED_OVER_ALLOCATED = {"none": (0.98, 1.03), "4bit": (0.75, 1.15)}
#: est_total x (1 + VRAM_SAFETY_MARGIN) / the clean whole-card peak. Measured 0.974 (M3, whose
#: uncapped peak holds 674 MiB of cache above its need) to 1.225. Below 1.0 a run that fits
#: would be refused on the margin alone; above 1.45 the estimate over-prices.
VERDICT_OVER_DEVICE = (0.95, 1.45)

FILLED_WORST_CASE_GB = (11.30, 11.45)
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
    """``profile_from_config`` for ``row`` at ``seq_len`` ("own": the longest row, None: max).

    Every 4-bit row except those in ``post_2683_rows`` was measured before #2683, when peft's
    fp32 embeddings and head stayed on a bf16 card, so for those the call runs with
    ``kbit_fp32`` set on the input it builds (``profile_from_config`` has no such argument and
    prices the bf16 default, which is what a 4-bit run on a bf16 card does now).
    """
    seq = _own_length(row) if seq_len == "own" else seq_len
    cfg = load_config_from_string(_text(row, _base(row)))
    with pytest.MonkeyPatch.context() as patch:
        if row["quantization"] != "none" and not row.get("post_2683"):
            patch.setattr(
                hardware_fit,
                "hardware_fit_input_or_reason",
                functools.partial(hardware_fit.hardware_fit_input_or_reason, kbit_fp32=True),
            )
        result = profile_from_config(cfg, allow_network=False, seq_len=seq)
    assert isinstance(result, dict), result
    return result


def _allocated_terms(result: dict) -> int:
    return result["total_memory_bytes"] - result["memory_terms_bytes"]["overhead"]


def _torch_part_of_the_overhead(result: dict) -> int:
    """What a bf16 LoRA run's overhead holds that torch allocates: the cuBLAS workspace and
    the entropy metric's two ``[min(128, tokens), vocab]`` fp32 temporaries. 0 for a run that
    keeps the old overhead (4-bit loading), which has no such part."""
    if result["quantization"] != "none":
        return 0
    tokens = result["seq_len"] * result["batch_size"]
    return 64 * MIB + 2 * min(128, tokens) * result["vocab_size"] * 4


_DATA = _evidence()
_TORCH_ROWS = list(_DATA["rows"])
_ADDED_ROWS = list(_DATA["downstream_added_rows"])
_BENCH_ROWS = list(_DATA["bench_rows"])
_POST_2683_ROWS = list(_DATA["post_2683_rows"])
_DOWNSTREAM_ALLOCATED_ROWS = list(_DATA["downstream_allocated_rows"])
_CAPPED_ROWS = list(_DATA["capped_rows"])
_FILLED_ROWS = list(_DATA["filled_micro_batch_rows"])
_UNCAPPED_ROWS = {row["id"]: row for row in [*_BENCH_ROWS, *_POST_2683_ROWS, *_FILLED_ROWS]}
_ALLOCATED_ROWS = [*_BENCH_ROWS, *_FILLED_ROWS]
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
    assert [row["id"][:2] for row in data["post_2683_rows"]] == ["M4", "M1"]
    assert len(data["downstream_allocated_rows"]) == 2
    assert [row["id"][:2] for row in data["filled_micro_batch_rows"]] == ["M7", "M8"]
    assert {row["measured"] for row in data["filled_micro_batch_rows"]} == {"2026-11-16"}
    # One model family and one card, said where the rows are described.
    assert "one model family and one card" in data["description"]
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
    # The two token terms are what make it so (2.21 and 1.82 MB per token).
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


def test_qwen3_1_7b_is_priced_above_its_torch_peak_with_its_tied_head_counted_once(models):
    # Its checkpoint stores lm_head.weight although the config ties it. The estimate counts the
    # table once, as the model loads it (0.62 GB less than the header sum), so its allocated
    # terms sit at 0.92 of the reserved peak, as the 0.6B rows' do (0.83 to 0.86).
    row = _ADDED_ROWS[0]
    assert row["id"].startswith("B1")
    result = _priced(row)
    torch_bytes = row["torch_reserved_mib"] * MIB
    assert TOTAL_OVER_TORCH[0] <= result["total_memory_bytes"] / torch_bytes <= TOTAL_OVER_TORCH[1]
    assert 0.88 <= _allocated_terms(result) / torch_bytes <= 0.97


@pytest.mark.parametrize("row", _TORCH_ROWS, ids=_ids(_TORCH_ROWS))
def test_priced_at_max_length_each_run_is_three_to_five_times_its_peak(models, row):
    # What the downstream application saw: 3.0x to 4.6x (3.3x to 5.1x with the overhead and
    # the QK-norm term of the capped runs). Pinned so that the over-estimate this change works
    # around is on record next to the number that fixes it.
    ratio = _priced(row, seq_len=None)["total_memory_bytes"] / (row["torch_reserved_mib"] * MIB)
    assert 2.8 <= ratio <= 5.4, (row["id"], ratio)


# --- 3. soup bench train: allocated peaks, and the device peak with the margin ------------


@pytest.mark.parametrize("row", _ALLOCATED_ROWS, ids=_ids(_ALLOCATED_ROWS))
def test_the_allocated_terms_match_the_allocated_peak_of_soup_bench_train(models, row):
    result = _priced(row)
    ratio = (_allocated_terms(result) + _torch_part_of_the_overhead(result)) / row[
        "max_memory_allocated_bytes"
    ]
    low, high = ALLOCATED_OVER_ALLOCATED[row["quantization"]]
    assert low <= ratio <= high, (row["id"], ratio)


def test_the_four_bf16_recipes_are_matched_to_one_and_a_third_percent(models):
    # M1, M3 and M6 are what the overhead and the QK-norm term were fitted to; M7 (a
    # 256-token filled micro-batch) was predicted before it was run (3,474 MiB against 3,441).
    rows = [
        row
        for row in _ALLOCATED_ROWS
        if row["quantization"] == "none" and not row["id"].startswith("M2")
    ]
    ratios = {}
    for row in rows:
        result = _priced(row)
        ratios[row["id"][:2]] = (
            _allocated_terms(result) + _torch_part_of_the_overhead(result)
        ) / row["max_memory_allocated_bytes"]
    assert sorted(ratios) == ["M1", "M3", "M6", "M7"], ratios
    assert all(0.99 <= value <= 1.015 for value in ratios.values()), ratios


@pytest.mark.parametrize("row", _CLEAN_ROWS, ids=_ids(_CLEAN_ROWS))
def test_the_estimate_with_the_verdict_margin_covers_every_uncapped_device_peak_but_one(
    models, row
):
    total = _priced(row)["total_memory_bytes"]
    ratio = total * (1 + VRAM_SAFETY_MARGIN) / (row["device_mib"] * MIB)
    assert VERDICT_OVER_DEVICE[0] <= ratio <= VERDICT_OVER_DEVICE[1], (row["id"], ratio)
    # Every row but M3 is above 1.0 with the margin; M3's own entry is below.
    assert (ratio < 1.0) == row["id"].startswith("M3"), (row["id"], ratio)


def test_the_one_clean_row_below_the_margin_is_m3_whose_peak_holds_cache(models):
    # M3 (r64 on rows of at most 147 tokens): estimate 3,447 MiB, 3,792 with the margin, against
    # an uncapped whole-card peak of 3,894. Its allocated peak is 3,058 MiB and it trains with
    # the allocator capped at 3,150, which is 3,312 MiB on the card: the 582 MiB between that and
    # the uncapped peak is cache the allocator gives back, and the estimate is 1.041 of the need.
    ratios = {
        row["id"]: _priced(row)["total_memory_bytes"]
        * (1 + VRAM_SAFETY_MARGIN)
        / (row["device_mib"] * MIB)
        for row in _CLEAN_ROWS
    }
    below = [key for key, value in ratios.items() if value < 1.0]
    assert [key[:2] for key in below] == ["M3"], ratios
    assert 0.97 <= ratios[below[0]] <= 0.98, ratios[below[0]]
    # Every other clean row has at least 4% to spare (the next tightest is M4, 1.046).
    assert all(value >= 1.04 for key, value in ratios.items() if key != below[0]), ratios


def test_the_stored_and_the_loaded_count_of_a_tied_model_are_priced_alike(models, monkeypatch):
    # The header of Qwen3-0.6B's checkpoint sums the tied head twice (751,632,384); the loaded
    # model holds 596,049,920. The estimate reads the header and counts the table once, so both
    # give the same bytes (before, the stored count was 311.6 MiB higher).
    row = next(row for row in _BENCH_ROWS if row["id"].startswith("M6"))
    stored = _priced(row)
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 596_049_920)
    loaded = _priced(row)
    assert stored["total_memory_bytes"] == loaded["total_memory_bytes"]
    assert stored["memory_terms_bytes"]["weights"] == math.ceil(596_049_920 * 2.0)


def test_eleven_clean_rows_are_six_downstream_and_five_bench():
    bench = [row for row in _CLEAN_ROWS if "max_memory_allocated_bytes" in row]
    downstream = [row for row in _CLEAN_ROWS if "max_memory_allocated_bytes" not in row]
    assert (len(_CLEAN_ROWS), len(downstream), len(bench)) == (11, 6, 5)
    # M2 repeats M1's numbers to the byte (a different max_length), so ten are independent.
    assert {row["device_mib"] for row in bench} == {2462, 3894, 2348, 6510}
    # Four of the six are among the 20 torch rows, two are device-only.
    assert sum(row in _TORCH_ROWS + _ADDED_ROWS for row in downstream) == 4
    assert sum(row in _DATA["device_only_rows"] for row in downstream) == 2


def test_the_estimate_alone_is_below_the_uncapped_device_peak_on_seven_clean_runs(models):
    # Recorded so that nobody quotes est_total as "the peak": on seven of the eleven clean runs
    # of this card it is below the uncapped whole-card peak. That peak holds allocator cache;
    # what a run needs is the capped runs' smallest working cap plus the context (below).
    below = {
        row["id"]: _priced(row)["total_memory_bytes"] / (row["device_mib"] * MIB)
        for row in _CLEAN_ROWS
        if _priced(row)["total_memory_bytes"] < row["device_mib"] * MIB
    }
    assert len(below) == 7, below
    assert {key[:2] for key in below if key[0] == "M"} == {"M1", "M2", "M3", "M4", "M6"}
    assert all(0.88 <= value < 1.0 for value in below.values()), below


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


def _post_2683(prefix: str) -> dict:
    return next(row for row in _POST_2683_ROWS if row["id"].startswith(prefix))


def _capped(prefix: str, outcome: str) -> list[dict]:
    """The capped runs of one recipe (``M4``, ``M1``, ...) with this outcome, smallest cap first."""
    return sorted(
        (
            row
            for row in _CAPPED_ROWS
            if row["id"].startswith(prefix + "-") and row["outcome"] == outcome
        ),
        key=lambda row: row["allocator_cap_mib"],
    )


def _context_mib() -> int:
    """What the card holds outside torch's allocator: ``device - reserved`` in every bench run."""
    return _evidence()["cuda_context"]["device_minus_reserved_mib_in_every_bench_row"]


def test_the_4_bit_run_on_main_peaks_above_the_estimate_uncapped_and_trains_inside_it_capped(
    models,
):
    # M4 re-run on origin/main a6f5e7246, after #2683 kept the embeddings and head in bf16.
    # Uncapped, the allocated peak fell 674 MiB (2081 -> 1407) but reserved / allocated rose to
    # 1.60 and the whole-card peak rose 62 MiB (2348 -> 2410): the estimate, priced at the
    # default a bf16 card now gets and with the tied head counted once, is 0.977 of the
    # allocated peak, 0.80 of that whole-card peak and 0.88 with the pre-flight's 10% margin.
    # That peak holds allocator cache the allocator gives back under pressure: with it capped at
    # 1500 MiB the same run trains, the card shows 1646 MiB, and the estimate is 1.17 of that.
    # These are the measured facts; neither number is a guarantee, and the clean-row pins above
    # stop at #2683.
    row = _post_2683("M4")
    before = next(row for row in _BENCH_ROWS if row["quantization"] == "4bit")
    result = _priced(row)
    total = result["total_memory_bytes"]
    device = row["device_mib"] * MIB
    allocated = row["max_memory_allocated_bytes"]
    reserved = row["max_memory_reserved_bytes"]
    assert 0.95 <= _allocated_terms(result) / allocated <= 1.05  # measured 0.977
    assert 0.75 <= total / device <= 0.82  # 0.797
    assert 0.85 <= total * (1 + VRAM_SAFETY_MARGIN) / device < 0.90  # 0.877
    assert 1.55 <= reserved / allocated <= 1.65  # 1.597, against 1.05 before #2683
    assert row["device_mib"] - round(reserved / MIB) == 162
    assert before["max_memory_allocated_bytes"] - allocated == pytest.approx(674 * MIB, abs=MIB)
    assert row["device_mib"] - before["device_mib"] == 62
    trains = _capped("M4", "trains")
    assert [capped["allocator_cap_mib"] for capped in trains] == [1500, 1700]
    smallest = trains[0]
    assert smallest["max_memory_allocated_bytes"] == allocated  # a cap changes no allocated peak
    assert smallest["device_mib"] == 1646
    assert row["device_mib"] - smallest["device_mib"] == 764  # MiB the allocator gave back
    assert 1.10 <= total / (smallest["device_mib"] * MIB) <= 1.25  # 1.167
    failed = _capped("M4", "oom")
    assert [capped["allocator_cap_mib"] for capped in failed] == [1450]
    assert "prepare_kbit_model" in failed[0]["failed_in"]  # before the first step


def test_the_bf16_control_on_current_main_is_the_run_before_it(models):
    control = _post_2683("M1")
    before = next(row for row in _BENCH_ROWS if row["id"].startswith("M1"))
    for key in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "device_mib"):
        assert control[key] == before[key], key
    assert _priced(control)["total_memory_bytes"] == _priced(before)["total_memory_bytes"]


# --- 3b. the same runs with torch's allocator capped ---------------------------------------
#
# An uncapped peak includes blocks the allocator only caches and gives back when a request
# would otherwise fail. Capping it (``torch.cuda.set_per_process_memory_fraction``) finds what a
# run needs: the smallest cap that still trains, plus what the card holds outside torch.


_CAPPED_OUTCOMES = {
    "M4-cap-1700": "trains",
    "M4-cap-1500": "trains",
    "M4-cap-1450": "oom",
    "M1-cap-2150": "trains",
    "M1-cap-2150-run2": "trains",
    "M1-cap-2150-run3": "trains",
    "M1-cap-2200": "oom",  # higher than a cap that trained, three times over
    "M1-cap-2200-run2": "oom",
    "M1-cap-2200-run3": "oom",
    "M3-cap-3300": "trains",
    "M3-cap-3150": "trains",
    "M6-cap-6300": "trains",
    "M6-cap-6200": "trains",
    "M6-cap-6100": "trains",
    "M6-cap-6050": "trains",
    "M6-cap-6000": "oom",
    "M6-cap-5850": "oom",
    "M6-cap-5700": "oom",
    "M7-cap-3887": "trains",
    "M7-cap-3738": "oom",
    "M7-cap-3590": "oom",
    "M8-cap-5603": "trains",
    "M8-cap-5306": "trains",
    "M8-cap-5009": "oom",
}


def test_the_capped_runs_are_recorded_as_measured():
    assert {row["id"]: row["outcome"] for row in _CAPPED_ROWS} == _CAPPED_OUTCOMES
    context = _context_mib()
    for row in _CAPPED_ROWS:
        uncapped = _UNCAPPED_ROWS[row["uncapped_row"]]
        cap = row["allocator_cap_mib"] * MIB
        assert row["tree"] == "origin/main a6f5e7246", row["id"]
        assert row["device_clean"] and row["device_baseline_mib"] == 0, row["id"]
        assert row["id"][:2] == uncapped["id"][:2], row["id"]
        assert row["device_mib"] * MIB <= cap + context * MIB, row["id"]
        if row["outcome"] == "trains":
            reserved = row["max_memory_reserved_bytes"]
            assert reserved <= cap, row["id"]
            assert row["max_memory_allocated_bytes"] == pytest.approx(
                uncapped["max_memory_allocated_bytes"], rel=0.002
            ), row["id"]
            if cap - reserved < 2 * MIB:  # the cap bound the run: the card shows cap + context
                # 146 to 162 MiB above the cap in the 2026-11-14 runs; 131 for one 2026-11-16 run
                # (M7 at 3,887), whose 200 ms sampler may have missed the peak.
                assert 120 <= row["device_mib"] - row["allocator_cap_mib"] <= context, row["id"]
        else:
            assert row["max_memory_allocated_bytes"] is None, row["id"]
            assert row["max_memory_reserved_bytes"] is None, row["id"]
            assert row["failed_in"], row["id"]
    # 2026-11-16 rows say so; the 2026-11-14 ones predate the field.
    measured = {row["id"] for row in _CAPPED_ROWS if row.get("measured") == "2026-11-16"}
    assert measured == {key for key in _CAPPED_OUTCOMES if key not in _EARLIER_CAPPED}
    # The 594 MiB request of M6 and M8 is the contiguous copy of their shifted fp32 logits (2
    # rows x 511 positions x the vocabulary x 4 bytes, in the allocator's 2 MiB blocks); M7's
    # 296 MiB is the same copy at 255 positions. The 4-bit preparation's 594 is peft's fp32
    # upcast of the embedding table (the vocabulary x 1,024 x 4 bytes), the same size by
    # coincidence.
    assert math.ceil(2 * 511 * 151936 * 4 / (2 * MIB)) * 2 == 594
    assert math.ceil(2 * 255 * 151936 * 4 / (2 * MIB)) * 2 == 296
    assert math.ceil(151936 * 1024 * 4 / (2 * MIB)) * 2 == 594
    blocks = {row["id"]: row["oom_block_mib"] for row in _CAPPED_ROWS if row["outcome"] == "oom"}
    assert blocks == {
        "M4-cap-1450": 594,
        "M1-cap-2200": 76,
        "M1-cap-2200-run2": 76,
        "M1-cap-2200-run3": 76,
        "M6-cap-6000": 594,
        "M6-cap-5850": 594,
        "M6-cap-5700": 594,
        "M7-cap-3738": 296,
        "M7-cap-3590": 296,
        "M8-cap-5009": 594,
    }


_EARLIER_CAPPED = frozenset(
    {
        "M4-cap-1700", "M4-cap-1500", "M4-cap-1450", "M1-cap-2150", "M1-cap-2200",
        "M3-cap-3300", "M3-cap-3150", "M6-cap-6000", "M6-cap-5850", "M6-cap-5700",
    }
)


def test_the_4_bit_preparation_leaves_no_cached_residue_to_give_back():
    # allocated 514 / reserved 598 MiB when prepare_kbit_model returns, and empty_cache() there
    # changes nothing, not even the run's peaks: the reserved memory above the allocated peak of
    # the M4 run builds up during training, it is not left over from the cast.
    probe = _DATA["kbit_probe"]
    assert probe["after_prepare_kbit_model_mib"] == {"allocated": 514, "reserved": 598}
    assert probe["after_empty_cache_mib"] == probe["after_prepare_kbit_model_mib"]
    peaks = probe["peaks_with_and_without_empty_cache"]
    row = _post_2683("M4")
    assert peaks["identical_to_the_unprobed_run"] is True
    assert peaks["max_memory_allocated_bytes"] == row["max_memory_allocated_bytes"]
    assert peaks["max_memory_reserved_bytes"] == row["max_memory_reserved_bytes"]
    assert peaks["device_mib"] == row["device_mib"]


def _need_mib(cap_mib: int) -> int:
    """What the card holds for a run held to ``cap_mib`` of torch's allocator."""
    return cap_mib + _context_mib()


def _estimate_mib(prefix: str) -> float:
    uncapped = _UNCAPPED_ROWS[_capped(prefix, "trains")[0]["uncapped_row"]]
    return _priced(uncapped)["total_memory_bytes"] / MIB


#: est_total / (the smallest cap that trained + the context). M4 4-bit rows of at most 79 tokens;
#: M1 the bf16 helpdesk recipe; M3 r64 on rows of at most 147 tokens; M6, M7 bf16 micro-batches
#: that fill 512 and 256 tokens; M8 a 4-bit one that fills 512. Measured 1.156, 1.027, 1.041,
#: 1.008, 0.979 and 1.083. Below 1.0 only M7, which the caps around it bracket (the next test).
ESTIMATE_OVER_NEED = {
    "M4": (1.10, 1.25),
    "M1": (1.00, 1.08),
    "M3": (1.00, 1.10),
    "M6": (1.00, 1.05),
    "M7": (0.95, 1.00),
    "M8": (1.03, 1.15),
}


@pytest.mark.parametrize("prefix", sorted(ESTIMATE_OVER_NEED))
def test_the_estimate_is_where_the_capped_runs_put_the_need(models, prefix):
    smallest = _capped(prefix, "trains")[0]
    need = _need_mib(smallest["allocator_cap_mib"])
    ratio = _estimate_mib(prefix) / need
    low, high = ESTIMATE_OVER_NEED[prefix]
    assert low <= ratio <= high, (prefix, ratio)
    # The card never showed more than the cap and the context.
    assert smallest["device_mib"] <= need


@pytest.mark.parametrize(
    "row",
    [row for row in _CAPPED_ROWS if row["outcome"] == "trains"],
    ids=[row["id"] for row in _CAPPED_ROWS if row["outcome"] == "trains"],
)
def test_the_estimate_with_the_margin_covers_every_cap_that_trained_plus_the_context(
    models, row
):
    # The requirement, at every cap and not only the smallest: est x 1.1 on the card.
    total = _estimate_mib(row["id"].split("-")[0])
    assert total * (1 + VRAM_SAFETY_MARGIN) >= _need_mib(row["allocator_cap_mib"]), row["id"]


@pytest.mark.parametrize("prefix", sorted(ESTIMATE_OVER_NEED))
def test_the_estimate_is_above_every_cap_that_failed_plus_the_context(models, prefix):
    # M1 fails at 2,200 MiB and trains at 2,150, so its failure is not a threshold; the estimate
    # is above it all the same, by 0.5% (2,374 against 2,362 MiB).
    failed = _capped(prefix, "oom")
    if not failed:
        pytest.skip("no run of this recipe failed at a cap")
    highest = failed[-1]["allocator_cap_mib"]
    assert _estimate_mib(prefix) >= _need_mib(highest), prefix


def test_the_256_token_filled_micro_batch_is_priced_inside_the_bracket_its_caps_leave(models):
    # M7 (512 tokens per micro-batch): fails at 3,590 and 3,738 MiB (the logits copy, a 296 MiB
    # block), trains at 3,887. Its need is above 3,738 + 162 and at most 3,887 + 162 = 4,049 MiB;
    # the estimate, 3,963, is 2.1% below the top of that and 1.6% above the bottom. It is the one
    # recipe whose estimate alone is below the smallest working cap's card figure.
    failed, trains = _capped("M7", "oom"), _capped("M7", "trains")
    assert [row["allocator_cap_mib"] for row in failed] == [3590, 3738]
    assert [row["allocator_cap_mib"] for row in trains] == [3887]
    total = _estimate_mib("M7")
    low, high = _need_mib(3738), _need_mib(3887)
    assert (low, high) == (3900, 4049)
    assert low < total < high, total
    assert total * (1 + VRAM_SAFETY_MARGIN) >= high


def test_the_helpdesk_recipe_trains_at_2150_three_times_and_fails_at_2200_three_times(models):
    # One run per cap: the working cap is not monotone, so a failing cap above a working one
    # is not a boundary. The same recipe reached the same peaks in every run that trained.
    trains, failed = _capped("M1", "trains"), _capped("M1", "oom")
    assert [row["allocator_cap_mib"] for row in trains] == [2150, 2150, 2150]
    assert [row["allocator_cap_mib"] for row in failed] == [2200, 2200, 2200]
    assert {row["failed_in"] for row in failed} == {"step 5 of 9, trl's entropy_from_logits"}
    assert len({row["max_memory_allocated_bytes"] for row in trains}) == 1


def test_the_filled_4_bit_micro_batch_trains_one_logits_block_above_its_allocated_peak(models):
    # M8 (1,024 tokens, 4-bit, after #2683): allocated peak 4,712 MiB, fails at 5,009 (half a
    # block above it) and trains at 5,306 = 4,712 + 594, one block of fp32 logits. The 4-bit
    # estimate, 5,923 MiB, is 1.083 of that card figure: its allocated terms are 1.10 of the
    # allocated peak, which is why the 4-bit runs keep their old overhead.
    uncapped = _UNCAPPED_ROWS["M8-main-r16-4bit-filled-512"]
    allocated = uncapped["max_memory_allocated_bytes"] / MIB
    block = 2 * 512 * 151936 * 4 / MIB
    assert [row["allocator_cap_mib"] for row in _capped("M8", "oom")] == [5009]
    assert [row["allocator_cap_mib"] for row in _capped("M8", "trains")] == [5306, 5603]
    assert _capped("M8", "trains")[0]["allocator_cap_mib"] == pytest.approx(
        allocated + block, abs=3
    )
    assert _capped("M8", "oom")[0]["allocator_cap_mib"] == pytest.approx(
        allocated + block / 2, abs=10
    )


def test_the_micro_batch_that_fills_512_tokens_needs_6162_to_6212_mib_and_is_priced_above_it(
    models,
):
    # M6, 1,024 tokens per micro-batch: fails at 5,700, 5,850 and 6,000 MiB (the logits copy, a
    # 594 MiB block), trains at 6,050, 6,100, 6,200 and 6,300. Its need is above 6,000 + 162 and
    # at most 6,050 + 162 = 6,212 MiB; the estimate is 6,260. Before the overhead and the
    # QK-norm term of these runs it was 5,930, below every cap that trained and the highest that
    # failed, so the 10% margin was all that covered it. It trains uncapped at 6,510 MiB on the
    # card: cache above the need.
    failed, trains = _capped("M6", "oom"), _capped("M6", "trains")
    assert [row["allocator_cap_mib"] for row in failed] == [5700, 5850, 6000]
    assert [row["allocator_cap_mib"] for row in trains] == [6050, 6100, 6200, 6300]
    uncapped = _UNCAPPED_ROWS["M6-r16-bf16-filled-512"]
    total = _priced(uncapped)["total_memory_bytes"]
    low, high = _need_mib(6000) * MIB, _need_mib(6050) * MIB
    assert low < high <= total
    assert total / high <= 1.02
    assert total < uncapped["device_mib"] * MIB  # 6,260 against 6,510 uncapped
    assert total * (1 + VRAM_SAFETY_MARGIN) >= _need_mib(6300) * MIB


@pytest.mark.parametrize(
    "row", _DOWNSTREAM_ALLOCATED_ROWS, ids=_ids(_DOWNSTREAM_ALLOCATED_ROWS)
)
def test_the_downstream_runner_logged_allocated_peaks_the_estimate_matches(models, row):
    # The same recipe through the downstream runner: allocated 2145 (bf16) and 1672 (4-bit, before
    # #2683) MiB, where `soup bench train` read 2108 and 2081. The 4-bit shortfall of the bench
    # run (0.84) is not in this runner's figure (1.05), which is what makes the load phase of
    # `bench train` the first suspect for it.
    result = _priced(row)
    ratio = (_allocated_terms(result) + _torch_part_of_the_overhead(result)) / (
        row["allocated_mib"] * MIB
    )
    assert 0.97 <= ratio <= 1.08, (row["id"], ratio)
    assert result["total_memory_bytes"] >= row["reserved_mib"] * MIB
    assert row["device_mib"] - row["reserved_mib"] in (162, 172)


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
