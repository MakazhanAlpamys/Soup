"""Gradient checkpointing does not drop the inputs of bitsandbytes' int8 layers.

bitsandbytes' ``MatMul8bitLt`` keeps its input on ``ctx.tensors`` rather than through
``save_for_backward``, so checkpointing, which only drops what is saved through
``save_for_backward``, leaves every int8 linear layer's input on the card until that
layer's backward. The shared estimate charged an 8-bit checkpointed layer its boundary
tensor alone, so it priced every 8-bit run with full checkpointing at 0.64 to 0.92 of its
measured peak, under the 0.95 the 8-bit band allows: the pre-flight passed runs that then
ran out of memory.

``benchmarks/evidence/vram-int8-checkpointing.json`` holds the rows, measured with the
#2049 harness (``benchmarks/gate-int8-checkpointing-vram.md``): 8-bit runs with
checkpointing on Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-360M, and 8-bit without
checkpointing, 4-bit and bf16 controls, which the new term must not move. Pure arithmetic
over the recorded shapes, so this runs offline and without torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from soup_cli.utils import vram_estimate as ve
from soup_cli.utils.model_shape import shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram

EVIDENCE = Path(__file__).parents[1] / "benchmarks/evidence/vram-int8-checkpointing.json"

#: The bands of ``tests/test_vram_estimate_calibration.py``, per kind of run.
BANDS = {
    "bf16 LoRA": (0.98, 1.10),
    "4-bit": (0.91, 1.10),
    "8-bit": (0.95, 1.20),
}


def _rows() -> list[dict]:
    """The pretrained checkpoints' rows that completed.

    The file also holds rows of randomly initialised weights, which the record keeps for
    completeness: they meet no outlier column, so bitsandbytes takes another path for them
    (the record says which), and they are not evidence about a checkpoint.
    """
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return [
        row
        for row in data["rows"]
        if row["weights"] == "pretrained" and not row["oom"] and not row["void"]
    ]


def _kind(row: dict) -> str:
    return {"none": "bf16 LoRA", "4bit": "4-bit", "8bit": "8-bit"}[row["run"]["quantization"]]


def _checkpointed_8bit(row: dict) -> bool:
    run = row["run"]
    return run["quantization"] == "8bit" and bool(run["gradient_checkpointing"])


def _ident(row: dict) -> str:
    run = row["run"]
    return (
        f"{row['model'].split('/')[-1]}-{run['quantization']}-r{run['lora_r']}-"
        f"{run['seq']}x{run['batch']}-gc{run['gradient_checkpointing']}"
    )


def _spec(row: dict) -> RunSpec:
    run = row["run"]
    quant = run["quantization"]
    return RunSpec(
        seq_len=run["seq"],
        batch_size=run["batch"],
        quant=quant,
        peft="qlora" if quant == "4bit" else "lora",
        optimizer=run["optimizer"],
        gradient_checkpointing=run["gradient_checkpointing"],
        lora_r=run["lora_r"],
        target_modules=run["target_modules"],
    )


def _shape(row: dict):
    config = row["config"]
    return shape_from_config(config, params=config["params"], params_source="safetensors")


def _ratio(row: dict) -> float:
    est = estimate_vram(_shape(row), _spec(row))
    return (est.total_bytes - est.terms["overhead"]) / row["peak_allocated"]


def test_the_evidence_is_complete():
    rows = _rows()
    assert len(rows) >= 10
    assert all(row["realised"] == row["run"]["seq"] for row in rows)
    assert all(row["steps"] == 2 for row in rows)
    checkpointed = [row for row in rows if _checkpointed_8bit(row)]
    assert len({row["config"]["params"] for row in checkpointed}) >= 3
    assert any(row["run"]["gradient_checkpointing"] == "medium" for row in checkpointed)
    # The controls the term must leave alone.
    assert any(r["run"]["quantization"] == "8bit" and not r["run"]["gradient_checkpointing"]
               for r in rows)
    assert {"4-bit", "bf16 LoRA"} <= {_kind(row) for row in rows}


@pytest.mark.parametrize("row", _rows(), ids=_ident)
def test_each_measured_peak_is_within_tolerance(row):
    ratio = _ratio(row)
    low, high = BANDS[_kind(row)]
    assert low <= ratio <= high, (
        f"{_ident(row)}: estimate / measured peak {ratio:.3f}, allowed {low} to {high}"
    )


def test_without_the_int8_term_the_checkpointed_8bit_rows_are_under_priced(monkeypatch):
    """Mutation check: the term is what prices these rows, and only these. Every row with
    full checkpointing falls below the band without it; a ``medium`` row keeps half its
    layers whole, so it may not."""
    monkeypatch.setattr(ve, "INT8_CHECKPOINT_KEPT", (0.0, 0.0))
    under = [_ident(row) for row in _rows() if _ratio(row) < BANDS[_kind(row)][0]]
    full = [
        _ident(row)
        for row in _rows()
        if _checkpointed_8bit(row) and row["run"]["gradient_checkpointing"] is True
    ]
    assert set(full) <= set(under)
    assert set(under) <= {_ident(row) for row in _rows() if _checkpointed_8bit(row)}


@pytest.mark.parametrize(
    "quant,checkpointing",
    [("8bit", False), ("4bit", True), ("4bit", False), ("none", True), ("none", "medium")],
)
def test_the_term_moves_no_other_run(monkeypatch, quant, checkpointing):
    row = _rows()[0]
    spec = RunSpec(
        seq_len=1024, batch_size=2, quant=quant, peft="qlora" if quant == "4bit" else "lora",
        gradient_checkpointing=checkpointing, lora_r=8,
    )
    with_term = ve.activations_bytes(_shape(row), spec)
    monkeypatch.setattr(ve, "INT8_CHECKPOINT_KEPT", (0.0, 0.0))
    assert ve.activations_bytes(_shape(row), spec) == with_term


@pytest.mark.parametrize("checkpointing", [True, "medium"])
def test_a_checkpointed_8bit_layer_keeps_its_int8_inputs(monkeypatch, checkpointing):
    """The term on its own: every checkpointed layer keeps 5 bf16 values per hidden unit
    and 2 per intermediate unit for each token; a layer that is not checkpointed already
    keeps them in its full activations."""
    shape = _shape(_rows()[0])
    spec = RunSpec(seq_len=1024, batch_size=2, quant="8bit", peft="lora",
                   gradient_checkpointing=checkpointing, lora_r=8)
    with_term = ve.activations_bytes(shape, spec)
    monkeypatch.setattr(ve, "INT8_CHECKPOINT_KEPT", (0.0, 0.0))
    without = ve.activations_bytes(shape, spec)

    layers = shape.num_layers
    checkpointed = layers if checkpointing is True else layers - math.ceil(layers / 2)
    per_layer = 1024 * 2 * 2 * (5 * shape.hidden_size + 2 * shape.intermediate_size)
    assert with_term - without == pytest.approx(checkpointed * per_layer, abs=2)


@pytest.mark.gpu
def test_real_8bit_checkpointed_forward_keeps_what_the_term_charges(tmp_path):
    """On the card: an 8-bit forward pass with checkpointing leaves more allocated for the
    backward than the same pass in 4-bit, and the difference is what the term charges.

    A tiny randomly initialised Llama with many layers, so the per-layer inputs dominate.
    Random weights meet no outlier column, so its down_proj input is fp32, the case the term
    prices. Measured as a difference between the two loads, which cancels what both keep."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        LlamaConfig,
        LlamaForCausalLM,
    )

    config = LlamaConfig(
        hidden_size=256, intermediate_size=1024, num_hidden_layers=16, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=1024,
    )
    base = tmp_path / "tiny-llama"
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(base)
    targets = ["q_proj", "v_proj"]
    ids = torch.randint(0, config.vocab_size, (1, 512), device="cuda")

    def kept_after_forward(quantization_config) -> int:
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype="auto", device_map={"": 0}, quantization_config=quantization_config
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids)
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - before - out.logits.untyped_storage().nbytes()
        del out, model
        torch.cuda.empty_cache()
        return kept

    measured = kept_after_forward(BitsAndBytesConfig(load_in_8bit=True)) - kept_after_forward(
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    )

    params = sum(p.numel() for p in LlamaForCausalLM(config).parameters())
    shape = shape_from_config(config.to_dict(), params=params, params_source="safetensors")

    def charged(quant: str, peft: str) -> int:
        spec = RunSpec(seq_len=512, batch_size=1, quant=quant, peft=peft,
                       gradient_checkpointing=True, lora_r=8, target_modules=targets)
        return ve.activations_bytes(shape, spec)

    gap = charged("8bit", "lora") - charged("4bit", "qlora")
    assert 0.8 <= gap / measured <= 1.25, (
        f"8-bit keeps {measured / 1e6:.1f} MB more than 4-bit; the estimate charges "
        f"{gap / 1e6:.1f} MB more"
    )
