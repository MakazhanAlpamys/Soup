"""The LoRA scaling the packed kernels use is the one peft computed for the adapter.

``lora.use_rslora`` changes that scaling from alpha / r to alpha / sqrt(r); the kernels read
``proj.scaling[adapter]`` (``kernels/packed_lora/model.py:_lora``), so the rsLoRA value must
arrive there unchanged. CPU only: a peft Linear layer, no kernel launch.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")



@pytest.fixture()
def lora_reader(monkeypatch):
    """``model._lora``, imported with triton stubbed when this box has none: ``_lora`` only reads
    peft attributes, but the module's import chain reaches the kernels' ``import triton``."""
    import importlib
    import sys
    from unittest import mock

    try:
        import triton  # noqa: F401
    except ImportError:
        for name in ("triton", "triton.language", "triton.language.extra", "triton.runtime",
                     "triton.runtime.jit"):
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())
        for name in [key for key in sys.modules if key.startswith("soup_cli.kernels.packed_lora")]:
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("soup_cli.kernels.packed_lora.model")._lora


def _adapted(use_rslora):
    base = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    config = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["0"],
                             use_rslora=use_rslora)
    return peft.get_peft_model(base, config).base_model.model[0]


@pytest.mark.parametrize("use_rslora,want", [
    (False, 16 / 8),
    (True, 16 / math.sqrt(8)),
])
def test_the_scaling_peft_computed_reaches_the_kernel_path(lora_reader, use_rslora, want):
    _a, _b, scaling = lora_reader(_adapted(use_rslora))
    assert scaling == pytest.approx(want)
