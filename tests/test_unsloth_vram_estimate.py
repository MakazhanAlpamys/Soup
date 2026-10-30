"""Backend-aware LoRA sizing and measured Unsloth 4-bit weight precision."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.hardware_fit import (
    estimate_peak_vram_gb,
    hardware_fit_input_from_config,
    unextrapolated_floor,
)
from soup_cli.utils.model_shape import lora_targets
from soup_cli.utils.vram_estimate import RunSpec, weights_bytes

LLAMA_31_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}
SEVEN_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)


def _config(tmp_path: Path, backend: str, targets: str | list[str] = "auto") -> SoupConfig:
    model = tmp_path / "llama-3.1-8b"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(LLAMA_31_8B), encoding="utf-8")
    return SoupConfig.model_validate(
        {
            "base": str(model),
            "task": "sft",
            "backend": backend,
            "data": {"train": "train.jsonl", "max_length": 2048},
            "training": {
                "quantization": "4bit", "batch_size": 1, "optimizer": "adamw_torch",
                "lora": {"r": 16, "target_modules": targets},
            },
        }
    )


@pytest.mark.parametrize(
    ("backend", "targets", "adapter_bytes"),
    [
        ("transformers", ("q_proj", "v_proj"), 27_262_976),
        ("unsloth", SEVEN_PROJECTIONS, 167_772_160),
    ],
)
def test_config_prices_the_projections_built_by_its_backend(
    tmp_path: Path, backend: str, targets: tuple[str, ...], adapter_bytes: int
) -> None:
    inp = hardware_fit_input_from_config(_config(tmp_path, backend), allow_network=False)
    assert inp is not None
    estimate = estimate_peak_vram_gb(inp).estimate
    assert inp.backend == estimate.run.backend == backend
    assert estimate.lora_targets == targets
    assert estimate.terms["adapters"] == adapter_bytes


def test_unsloth_explicit_targets_are_not_expanded(tmp_path: Path) -> None:
    inp = hardware_fit_input_from_config(
        _config(tmp_path, "unsloth", ["q_proj"]), allow_network=False
    )
    assert inp is not None
    estimate = estimate_peak_vram_gb(inp).estimate
    assert estimate.lora_targets == ("q_proj",)
    assert estimate.terms["adapters"] == 16_777_216


@pytest.mark.parametrize("targets", ["auto", None])
def test_unsloth_auto_uses_seven_projections_for_any_architecture(targets: str | None) -> None:
    assert lora_targets(targets, "phi3", backend="unsloth") == SEVEN_PROJECTIONS
    assert lora_targets(targets, "phi3") == ("qkv_proj",)


@pytest.mark.parametrize(
    ("backend", "weight_bytes", "floor_gb"),
    [
        ("transformers", 7_901_874_299, 8.510926203),
        ("unsloth", 5_800_527_995, 6.971616635),
    ],
)
def test_llama31_8b_floor_uses_backend_targets_and_weight_precision(
    tmp_path: Path, backend: str, weight_bytes: int, floor_gb: float
) -> None:
    # Dtype provenance: benchmarks/unsloth-4bit-weight-dtypes.md. The RTX 5090
    # load + get_peft_model kept both [128256, 4096] weights in bf16 (2 bytes).
    # These are fixed expectations; the test never loads Unsloth or model weights.
    cfg = _config(tmp_path, backend)
    inp = hardware_fit_input_from_config(cfg, allow_network=False)
    assert inp is not None
    breakdown = estimate_peak_vram_gb(inp)
    assert breakdown.estimate.shape.params == 8_030_261_248
    assert breakdown.estimate.terms["weights"] == weight_bytes
    floor, terms = unextrapolated_floor(cfg, inp, breakdown)
    assert floor == pytest.approx(floor_gb, abs=1e-12, rel=0)
    assert terms == ("weights", "CUDA context", "adapters", "gradients", "optimizer state")


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
def test_tied_embeddings_are_charged_once(tmp_path: Path, backend: str) -> None:
    inp = hardware_fit_input_from_config(_config(tmp_path, backend), allow_network=False)
    assert inp is not None
    shape = inp.shape
    tied = replace(shape, params=shape.params - 525_336_576, tie_word_embeddings=True)
    run = RunSpec(seq_len=2048, batch_size=1, quant="4bit", backend=backend)
    per_embedding = 2 if backend == "unsloth" else 4
    assert weights_bytes(shape, run) - weights_bytes(tied, run) == 525_336_576 * per_embedding


@pytest.mark.parametrize("quant", ["none", "8bit", "gptq"])
def test_unsloth_dtype_measurement_only_changes_four_bit_weights(
    tmp_path: Path, quant: str
) -> None:
    inp = hardware_fit_input_from_config(_config(tmp_path, "transformers"), allow_network=False)
    assert inp is not None
    run = RunSpec(seq_len=2048, batch_size=1, quant=quant)
    assert weights_bytes(inp.shape, run) == weights_bytes(
        inp.shape, replace(run, backend="unsloth")
    )


def test_unsloth_estimate_needs_no_ml_imports_or_network(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "unsloth")
    code = """
import sys

def guard(event, args):
    if event == "import" and args[0].split(".")[0] in {
        "torch", "unsloth", "transformers", "peft"
    }:
        raise AssertionError(f"unexpected ML import: {args[0]}")
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("unexpected network request")

sys.addaudithook(guard)
from soup_cli.utils.model_shape import read_model_shape
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram

shape = read_model_shape(sys.argv[1], allow_network=False)
estimate = estimate_vram(shape, RunSpec(
    seq_len=2048, batch_size=1, quant="4bit", lora_r=16, backend="unsloth"
))
assert estimate.terms["weights"] == 5_800_527_995
assert estimate.terms["adapters"] == 167_772_160
"""
    subprocess.run([sys.executable, "-c", code, cfg.base], check=True, timeout=60)
