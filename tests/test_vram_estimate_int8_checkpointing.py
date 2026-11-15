"""Gradient checkpointing does not drop the inputs of bitsandbytes' int8 layers.

bitsandbytes' ``MatMul8bitLt`` keeps its input on ``ctx.tensors`` rather than through
``save_for_backward``, so checkpointing, which only drops what is saved through
``save_for_backward``, leaves every int8 linear layer's input on the card until that
layer's backward. The shared estimate charged an 8-bit checkpointed layer its boundary
tensor alone, so it priced the pretrained 8-bit runs with full checkpointing at 0.82 to
0.93 of their measured peak on main ``3f9ba5ce`` (bf16 norms, #2683; Qwen2.5-0.5B, whose
logits dominate, read 1.003) and at 0.64 to 0.92 on ``527314a4`` (fp32 norms), under the
0.95 the 8-bit band allows: the pre-flight passed runs that then ran out of memory.

``benchmarks/evidence/vram-int8-checkpointing.json`` holds the rows, measured with the
#2049 harness (``benchmarks/gate-int8-checkpointing-vram.md``): 8-bit runs with
checkpointing on Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-360M, and 8-bit without
checkpointing, 4-bit and bf16 controls, which the new term must not move. Each row records
the main commit it ran on and its ``setting``: ``bf16`` (the #2683 default on a bf16 card,
``INT8_CHECKPOINT_KEPT``) or ``kbit_fp32`` (PEFT's fp32 norms, ``INT8_CHECKPOINT_KEPT_FP32``).
Pure arithmetic over the recorded shapes, so this runs offline and without torch.
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
    """The pretrained checkpoints' rows that completed, on both settings.

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


def _kbit_fp32(row: dict) -> bool:
    return row["setting"] == "kbit_fp32"


def _banded(row: dict) -> bool:
    """The rows held to the #2049 bands.

    Every ``kbit_fp32`` row, and on the ``bf16`` setting the 8-bit runs with full
    checkpointing, which the term prices, and the bf16 LoRA control. The other ``bf16``
    rows (8-bit without checkpointing, 4-bit, ``medium``) are over the band, or close to
    it, on main before this change, 1.13 to 1.27: their per-layer activation coefficients
    were fitted on fp32-norm rows (#2049), and #2683 measured no 8-bit run. The record says
    so; the term leaves the first two alone and adds exactly its bytes to ``medium``.
    """
    if _kbit_fp32(row):
        return True
    run = row["run"]
    return run["quantization"] == "none" or (
        run["quantization"] == "8bit" and run["gradient_checkpointing"] is True
    )


def _kind(row: dict) -> str:
    return {"none": "bf16 LoRA", "4bit": "4-bit", "8bit": "8-bit"}[row["run"]["quantization"]]


def _checkpointed_8bit(row: dict) -> bool:
    run = row["run"]
    return run["quantization"] == "8bit" and bool(run["gradient_checkpointing"])


def _ident(row: dict) -> str:
    run = row["run"]
    return (
        f"{row['main']}-{row['setting']}-{row['model'].split('/')[-1]}-{run['quantization']}-"
        f"r{run['lora_r']}-{run['seq']}x{run['batch']}-gc{run['gradient_checkpointing']}"
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
        kbit_fp32=_kbit_fp32(row),
    )


def _shape(row: dict):
    config = row["config"]
    return shape_from_config(config, params=config["params"], params_source="safetensors")


def _ratio(row: dict) -> float:
    est = estimate_vram(_shape(row), _spec(row))
    return (est.total_bytes - est.terms["overhead"]) / row["peak_allocated"]


def _zero_term(monkeypatch) -> None:
    monkeypatch.setattr(ve, "INT8_CHECKPOINT_KEPT", (0.0, 0.0))
    monkeypatch.setattr(ve, "INT8_CHECKPOINT_KEPT_FP32", (0.0, 0.0))


@pytest.mark.parametrize("setting", ["bf16", "kbit_fp32"])
def test_the_evidence_is_complete(setting):
    rows = [row for row in _rows() if row["setting"] == setting]
    assert len(rows) >= 10
    assert all(row["realised"] == row["run"]["seq"] for row in rows)
    assert all(row["steps"] == 2 for row in rows)
    checkpointed = [row for row in rows if _checkpointed_8bit(row)]
    assert len({row["config"]["params"] for row in checkpointed}) >= 2
    assert any(row["run"]["gradient_checkpointing"] == "medium" for row in checkpointed)


def test_the_bf16_setting_is_measured_on_main_with_its_controls():
    """The #2683 default re-measured on main: three models, ``medium``, and the controls."""
    rows = [row for row in _rows() if row["setting"] == "bf16"]
    assert {row["main"] for row in rows} == {"3f9ba5ce"}
    checkpointed = [row for row in rows if _checkpointed_8bit(row)]
    assert len({row["config"]["params"] for row in checkpointed}) >= 3
    # The controls the term must leave alone.
    assert any(r["run"]["quantization"] == "8bit" and not r["run"]["gradient_checkpointing"]
               for r in rows)
    assert {"4-bit", "bf16 LoRA"} <= {_kind(row) for row in rows}


@pytest.mark.parametrize("row", [row for row in _rows() if _banded(row)], ids=_ident)
def test_each_measured_peak_is_within_tolerance(row):
    ratio = _ratio(row)
    low, high = BANDS[_kind(row)]
    assert low <= ratio <= high, (
        f"{_ident(row)}: estimate / measured peak {ratio:.3f}, allowed {low} to {high}"
    )


def test_without_the_int8_term_the_checkpointed_8bit_rows_are_under_priced(monkeypatch):
    """Mutation check: the term is what prices these rows, and only these. Every row with
    full checkpointing falls below the band without it, on both settings, except the
    ``bf16`` Qwen2.5-0.5B rows, whose logits dominate (1.003 without the term); a
    ``medium`` row keeps half its layers whole, so it may not."""
    _zero_term(monkeypatch)
    under = [_ident(row) for row in _rows() if _ratio(row) < BANDS[_kind(row)][0]]
    full = [
        _ident(row)
        for row in _rows()
        if _checkpointed_8bit(row) and row["run"]["gradient_checkpointing"] is True
        and not (row["setting"] == "bf16" and row["model"] == "Qwen/Qwen2.5-0.5B-Instruct")
    ]
    assert len(full) >= 12
    assert set(full) <= set(under)
    assert set(under) <= {_ident(row) for row in _rows() if _checkpointed_8bit(row)}


@pytest.mark.parametrize("kbit_fp32", [False, True], ids=["bf16", "kbit_fp32"])
@pytest.mark.parametrize(
    "quant,checkpointing",
    [("8bit", False), ("4bit", True), ("4bit", False), ("none", True), ("none", "medium")],
)
def test_the_term_moves_no_other_run(monkeypatch, quant, checkpointing, kbit_fp32):
    row = _rows()[0]
    spec = RunSpec(
        seq_len=1024, batch_size=2, quant=quant, peft="qlora" if quant == "4bit" else "lora",
        gradient_checkpointing=checkpointing, lora_r=8, kbit_fp32=kbit_fp32,
    )
    with_term = ve.activations_bytes(_shape(row), spec)
    _zero_term(monkeypatch)
    assert ve.activations_bytes(_shape(row), spec) == with_term


def test_the_controls_outside_the_band_are_not_moved_by_the_term(monkeypatch):
    """The ``bf16`` rows main already over-prices (8-bit without checkpointing, 4-bit) read
    the same with and without the term: this change neither causes nor fixes them."""
    controls = [row for row in _rows() if not _banded(row) and not _checkpointed_8bit(row)]
    assert controls
    before = {_ident(row): _ratio(row) for row in controls}
    _zero_term(monkeypatch)
    assert {_ident(row): _ratio(row) for row in controls} == before


@pytest.mark.parametrize(
    "kbit_fp32,per_hidden,per_intermediate",
    [(False, 3, 1), (True, 5, 2)],
    ids=["bf16", "kbit_fp32"],
)
@pytest.mark.parametrize("checkpointing", [True, "medium"])
def test_a_checkpointed_8bit_layer_keeps_its_int8_inputs(
    monkeypatch, checkpointing, kbit_fp32, per_hidden, per_intermediate
):
    """The term on its own: every checkpointed layer keeps, for each token, 3 bf16 values
    per hidden unit and 1 per intermediate unit with bf16 norms, and 5 and 2 with fp32
    norms (``kbit_fp32``); a layer that is not checkpointed already keeps them in its full
    activations."""
    shape = _shape(_rows()[0])
    spec = RunSpec(seq_len=1024, batch_size=2, quant="8bit", peft="lora",
                   gradient_checkpointing=checkpointing, lora_r=8, kbit_fp32=kbit_fp32)
    with_term = ve.activations_bytes(shape, spec)
    _zero_term(monkeypatch)
    without = ve.activations_bytes(shape, spec)

    layers = shape.num_layers
    checkpointed = layers if checkpointing is True else layers - math.ceil(layers / 2)
    per_layer = 1024 * 2 * 2 * (
        per_hidden * shape.hidden_size + per_intermediate * shape.intermediate_size
    )
    assert with_term - without == pytest.approx(checkpointed * per_layer, abs=2)


@pytest.mark.gpu
@pytest.mark.parametrize("kbit_fp32", [False, True], ids=["bf16", "kbit_fp32"])
def test_real_8bit_checkpointed_forward_keeps_what_the_term_charges(tmp_path, kbit_fp32):
    """On the card: an 8-bit forward pass with checkpointing leaves more allocated for the
    backward than the same pass in 4-bit, and the difference is what the term charges.

    A tiny randomly initialised Llama with many layers, so the per-layer inputs dominate,
    prepared as Soup prepares a k-bit model on a bf16 card (``prepare_kbit_model``, bf16
    norms) or with PEFT's fp32 norms (``kbit_fp32``). Random weights meet no outlier
    column, so with fp32 norms the down_proj input is fp32, the case that term prices.
    Measured as a difference between the two loads, which cancels what both keep."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        LlamaConfig,
        LlamaForCausalLM,
    )

    from soup_cli.utils.kbit_prepare import prepare_kbit_model

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("the bf16 norms of #2683 need a card with bf16 hardware")
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
        model = prepare_kbit_model(
            model, use_gradient_checkpointing=True,
            compute_dtype=torch.float32 if kbit_fp32 else torch.bfloat16,
        )
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
                       gradient_checkpointing=True, lora_r=8, target_modules=targets,
                       kbit_fp32=kbit_fp32)
        return ve.activations_bytes(shape, spec)

    gap = charged("8bit", "lora") - charged("4bit", "qlora")
    assert 0.8 <= gap / measured <= 1.25, (
        f"8-bit keeps {measured / 1e6:.1f} MB more than 4-bit; the estimate charges "
        f"{gap / 1e6:.1f} MB more"
    )
