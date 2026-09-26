"""Regression coverage for #567: runtime-probed MPS BF16 in GRPO."""

from __future__ import annotations

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.grpo import GRPOTrainerWrapper


def _config(*, grpo_fp16: bool = False):
    return load_config_from_string(
        f"""
base: HuggingFaceTB/SmolLM2-135M-Instruct
task: grpo
backend: transformers
data:
  train: train.jsonl
  format: alpaca
training:
  reward_fn: accuracy
  num_generations: 2
  grpo_fp16: {str(grpo_fp16).lower()}
"""
    )


def test_grpo_enables_bf16_when_live_mps_probe_succeeds(monkeypatch):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "mps_supports_bf16", lambda: True)
    wrapper = GRPOTrainerWrapper(_config(), device="mps")

    assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": True}


def test_grpo_falls_back_to_fp32_when_live_mps_probe_fails(monkeypatch):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "mps_supports_bf16", lambda: False)
    wrapper = GRPOTrainerWrapper(_config(), device="mps:0")

    assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": False}


def test_cuda_only_grpo_fp16_override_does_not_force_fp16_on_mps(monkeypatch):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "mps_supports_bf16", lambda: True)
    wrapper = GRPOTrainerWrapper(_config(grpo_fp16=True), device="mps")

    assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": True}


def test_cpu_precision_policy_is_unchanged():
    wrapper = GRPOTrainerWrapper(_config(), device="cpu")

    assert wrapper._build_precision_kwargs() == {"fp16": False, "bf16": False}
