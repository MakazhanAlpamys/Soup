"""#425 / PR #429 — fp16 GradScaler crashes on bf16 LoRA adapters (pre-Ampere).

peft creates LoRA adapter weights in the base checkpoint's dtype (bf16 for many
models). On pre-Ampere cards (T4, P100, V100, GTX 16xx, RTX 20xx) training runs
in fp16, and the fp16 GradScaler raises ``_amp_foreach_non_finite_check_and_
unscale_cuda not implemented for 'BFloat16'`` on bf16 gradients. The fix is a
DECISION — cast trainable ``*lora_*`` params to fp32 when fp16 is active — so
these tests assert the decision against a plain ``nn.Module``, no GPU required.
(The failing primitive is CUDA-kernel-specific and does not raise on CPU.)

The scanner test is the guard that a hand-written call-site list would silently
leave a new trainer unprotected (the #328/#336 shape: a helper wired into one
wrapper and forgotten everywhere else).
"""

from __future__ import annotations

import pathlib

import pytest


def _make_model(*params):
    """Build a bare ``nn.Module`` from ``(name, dtype, requires_grad)`` tuples."""
    import torch
    import torch.nn as nn

    m = nn.Module()
    for name, dtype, requires_grad in params:
        setattr(m, name, nn.Parameter(torch.zeros(2, 2, dtype=dtype),
                                      requires_grad=requires_grad))
    return m


class TestAlignTrainableDtypeForFp16:
    def test_bf16_trainable_cast_to_fp32_under_fp16(self):
        import torch

        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        m = _make_model(("lora_A", torch.bfloat16, True))
        assert align_trainable_dtype_for_fp16(m, fp16=True, bf16=False) == 1
        assert m.lora_A.dtype is torch.float32

    def test_frozen_params_are_not_cast(self):
        import torch

        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        m = _make_model(("lora_A", torch.bfloat16, False))
        assert align_trainable_dtype_for_fp16(m, fp16=True, bf16=False) == 0
        assert m.lora_A.dtype is torch.bfloat16

    def test_fp16_trainable_is_left_alone(self):
        # The GradScaler handles fp16 gradients natively — no cast needed.
        import torch

        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        m = _make_model(("lora_A", torch.float16, True))
        assert align_trainable_dtype_for_fp16(m, fp16=True, bf16=False) == 0
        assert m.lora_A.dtype is torch.float16

    def test_non_lora_bf16_trainable_is_not_cast(self):
        # Bounds the blast radius: under full fine-tuning / Spectrum / LISA the
        # trainable set is a large fraction of the model, and casting it would
        # double trainable-weight memory after the VRAM pre-flight.
        import torch

        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        m = _make_model(("lm_head", torch.bfloat16, True))
        assert align_trainable_dtype_for_fp16(m, fp16=True, bf16=False) == 0
        assert m.lm_head.dtype is torch.bfloat16

    @pytest.mark.parametrize("fp16,bf16", [(False, True), (False, False)])
    def test_no_cast_off_the_fp16_path(self, fp16, bf16):
        import torch

        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        m = _make_model(("lora_A", torch.bfloat16, True))
        assert align_trainable_dtype_for_fp16(m, fp16=fp16, bf16=bf16) == 0
        assert m.lora_A.dtype is torch.bfloat16


def test_every_trainer_that_trains_also_aligns():
    """Every trainer wrapper that calls ``train()`` must align the adapter dtype.

    Scan, don't hand-write a list — a new trainer would otherwise ship without the
    guard and rot silently (the #328/#336 shape).
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
    missing = [
        p.name
        for p in root.glob("*.py")
        if "self.trainer.train(" in (s := p.read_text(encoding="utf-8"))
        and "align_trainable_dtype_for_fp16" not in s
    ]
    assert not missing, f"trainers call train() without fp16 dtype alignment: {missing}"
