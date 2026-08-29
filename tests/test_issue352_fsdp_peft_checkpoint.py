"""#352: a LoRA run under FSDP must not write the frozen base into checkpoints.

`Trainer._save_optimizer_and_scheduler` writes a separate FSDP resume checkpoint
(`pytorch_model_fsdp.bin`) via `accelerate.utils.save_fsdp_model`, independently of the
adapter-only save `Trainer.save_model` already produces. Accelerate 0.25.0/0.26.0 has no
`adapter_only` parameter on that function, so the resume checkpoint always gathers the
full (frozen base + adapter) state dict regardless of the model being a `PeftModel`.
0.27.0 added `adapter_only`, and `transformers.trainer.get_fsdp_ckpt_kwargs()`
feature-detects it and threads it through on both save and load. This pins the accelerate
floor bump in `pyproject.toml` to the real state-dict-size mechanism the issue reports,
rather than to the dependency string alone.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path


def test_pyproject_accelerate_floor_supports_adapter_only_fsdp_checkpoints():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    match = re.search(r'"accelerate>=([0-9.]+)"', pyproject.read_text(encoding="utf-8"))
    assert match, "accelerate floor pin not found in pyproject.toml"
    floor = tuple(int(part) for part in match.group(1).split("."))
    assert floor >= (0, 27, 0), (
        f"accelerate>={match.group(1)} predates save_fsdp_model's adapter_only "
        "parameter (added in 0.27.0) and will regress #352"
    )


def test_installed_accelerate_save_fsdp_model_supports_adapter_only():
    from accelerate.utils import save_fsdp_model

    assert "adapter_only" in inspect.signature(save_fsdp_model).parameters


def test_installed_accelerate_load_fsdp_model_supports_adapter_only():
    from accelerate.utils import load_fsdp_model

    assert "adapter_only" in inspect.signature(load_fsdp_model).parameters


def test_fsdp_resume_checkpoint_state_dict_is_adapter_only_for_peft_model():
    """Acceptance criterion from #352: checkpoint size is O(adapter), not O(base)."""
    import torch.nn as nn
    from accelerate.utils.fsdp_utils import _get_model_state_dict
    from peft import LoraConfig, get_peft_model

    class TinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            # A lopsided frozen-base-vs-adapter ratio, mirroring the issue's own
            # measurement (37 GB frozen base next to a 131 MB adapter).
            self.big = nn.Linear(2048, 2048, bias=False)

        def forward(self, x):
            return self.big(x)

    model = get_peft_model(TinyBase(), LoraConfig(target_modules=["big"], r=4, lora_alpha=8))

    full_numel = sum(v.numel() for v in _get_model_state_dict(model, adapter_only=False).values())
    resume_sd = _get_model_state_dict(model, adapter_only=True)
    resume_numel = sum(v.numel() for v in resume_sd.values())

    assert all("lora_" in key for key in resume_sd), (
        f"expected only adapter keys in the FSDP resume checkpoint, got {list(resume_sd)}"
    )
    # The frozen base (2048*2048 = 4_194_304 params) must not appear at all; only the
    # rank-4 LoRA A/B matrices (2 * 4 * 2048 = 16_384 params) should.
    assert resume_numel == 16_384
    assert resume_numel < full_numel / 100
