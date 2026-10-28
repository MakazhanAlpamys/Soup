"""The shared VRAM estimate against measured resident peaks (#2049).

``benchmarks/evidence/vram-2049-resident.json`` holds one-row-per-process ``soup
train`` steps on an RTX 3060 Laptop (``benchmarks/gate-2049-resident-vram.md``).
Each row's model shape is taken from the ``config.json`` and safetensors count
recorded with it, so this runs offline. The estimate's allocated terms (everything
but ``overhead``, which ``max_memory_allocated`` cannot see) must land within the
stated band of the measured peak for the row's kind of run. The bands are as tight as
each kind fits: an under-prediction lets a run through that then runs out of memory,
an over-prediction refuses a run that fits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.utils import vram_estimate as ve
from soup_cli.utils.model_shape import shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram

EVIDENCE = Path(__file__).parents[1] / "benchmarks/evidence/vram-2049-resident.json"

#: Allowed estimate / measured peak, per kind of run. The measured ranges are bf16 LoRA
#: 0.994 to 1.064, 4-bit 0.915 to 1.086 (4-bit Qwen2.5-3B with checkpointing at 512
#: tokens is the low one), 8-bit 0.969 to 1.164, full fine-tuning 0.970 to 1.127. The
#: gate's overhead term and 10% margin sit on top of these.
BANDS = {
    "bf16 LoRA": (0.98, 1.10),
    "4-bit": (0.91, 1.10),
    "8-bit": (0.95, 1.20),
    "full fine-tuning": (0.96, 1.15),
}

#: A vocabulary from which the logits term decides the row (Qwen2.5, Llama 3.x).
LARGE_VOCAB = 100_000


def _rows() -> list[dict]:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return [row for row in data["rows"] if not row["oom"] and not row["void"]]


def _kind(row: dict) -> str:
    if row["run"]["lora_r"] == 0:
        return "full fine-tuning"
    return {"none": "bf16 LoRA", "4bit": "4-bit", "8bit": "8-bit"}[row["run"]["quantization"]]


def _ident(row: dict) -> str:
    run = row["run"]
    return (
        f"{row['model'].split('/')[-1]}-{run['quantization']}-r{run['lora_r']}-"
        f"{run['seq']}x{run['batch']}-gc{run['gradient_checkpointing']}-{run['optimizer']}"
    )


def _predicted_allocated(row: dict) -> int:
    config, run = row["config"], row["run"]
    shape = shape_from_config(config, params=config["params"], params_source="safetensors")
    quant = run["quantization"]
    full = run["lora_r"] == 0
    spec = RunSpec(
        seq_len=run["seq"],
        batch_size=run["batch"],
        quant=quant,
        peft="full" if full else ("qlora" if quant == "4bit" else "lora"),
        optimizer=run["optimizer"],
        gradient_checkpointing=run["gradient_checkpointing"],
        lora_r=run["lora_r"],
        target_modules=run["target_modules"],
    )
    est = estimate_vram(shape, spec)
    return est.total_bytes - est.terms["overhead"]


def test_the_evidence_is_complete():
    rows = _rows()
    assert len(rows) >= 30
    assert all(row["realised"] == row["run"]["seq"] for row in rows)
    assert all(row["steps"] == 2 for row in rows)
    assert {_kind(row) for row in rows} == set(BANDS)
    assert any(row["run"]["lora_r"] == 0 for row in rows)
    assert any(row["config"]["vocab_size"] >= LARGE_VOCAB for row in rows)


@pytest.mark.parametrize("row", _rows(), ids=_ident)
def test_each_measured_peak_is_within_tolerance(row):
    predicted = _predicted_allocated(row)
    measured = row["peak_allocated"]
    ratio = predicted / measured
    low, high = BANDS[_kind(row)]
    assert low <= ratio <= high, (
        f"{_kind(row)}: predicted {predicted / 1e9:.3f} GB, measured {measured / 1e9:.3f} GB "
        f"({ratio:.3f}x, allowed {low} to {high})"
    )


def test_without_the_logits_term_large_vocabulary_rows_fail(monkeypatch):
    """Mutation check from the issue: the logits term is what prices these rows."""
    monkeypatch.setattr(ve, "logits_bytes", lambda shape, run: 0)
    large = [row for row in _rows() if row["config"]["vocab_size"] >= LARGE_VOCAB]
    assert large
    under = [
        _ident(row)
        for row in large
        if _predicted_allocated(row) / row["peak_allocated"] < BANDS[_kind(row)][0]
    ]
    assert under == [_ident(row) for row in large]
