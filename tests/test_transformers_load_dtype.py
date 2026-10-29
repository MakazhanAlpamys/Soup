"""Actual local model loads must not emit Transformers' deprecated dtype warning.

The check is scoped to that one warning. Turning every Python warning into an
error made the macOS cells fail on an unrelated one: torch's MPS probe on
GitHub's virtualised Apple runners warns "Skipping device Apple Paravirtual
device that does not support Metal 2.0" when Transformers imports its SDPA
integration.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _tiny_gpt2_with_valid_special_tokens(out_dir) -> str:
    """The shared tiny GPT-2 keeps GPT2Config's default bos/eos id 50256, outside
    its 300-token vocabulary, which Transformers logs a config warning for on
    every load. This copy points them at the tokenizer's own token."""
    from tests._tiny_hf_models import build_gpt2, build_tokenizer

    tokenizer = build_tokenizer()
    model = build_gpt2(tokenizer)
    for config in (model.config, model.generation_config):
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return str(out_dir)


@pytest.mark.parametrize("mode", ["from_pretrained", "from_config"])
def test_tiny_local_cpu_model_load_has_no_deprecated_dtype_warning(mode, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model_dir = _tiny_gpt2_with_valid_special_tokens(tmp_path / "tiny-gpt2")
    script = r"""
import logging
import sys
import warnings

import torch
from transformers import AutoConfig

class RejectDeprecatedDtype(logging.Handler):
    def emit(self, record):
        text = record.getMessage()
        if "torch_dtype" in text and "deprecated" in text:
            raise AssertionError(text)

logging.getLogger("transformers").addHandler(RejectDeprecatedDtype())
with warnings.catch_warnings():
    # Only a warning that names torch_dtype is fatal; a platform probe or an
    # unrelated upstream deprecation must not decide this test.
    warnings.filterwarnings("error", message=r".*torch_dtype")
    if sys.argv[2] == "from_pretrained":
        from soup_cli.utils.live_eval import load_model_and_tokenizer
        model, tokenizer, device = load_model_and_tokenizer(
            sys.argv[1], device="cpu", dtype=torch.float32,
        )
        assert model.dtype == torch.float32
        assert next(model.parameters()).device.type == "cpu"
        assert tokenizer is not None and device == "cpu"
    else:
        from soup_cli.utils.layer_stream_runtime import _skeleton_from_config
        config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
        model = _skeleton_from_config(config, torch.float32, None, False)
        assert all(parameter.is_meta for parameter in model.parameters())
        assert model.dtype == torch.float32
"""
    result = subprocess.run(
        [sys.executable, "-c", script, model_dir, mode],
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # The deprecation is logged at WARNING level; an inherited
            # TRANSFORMERS_VERBOSITY=error would hide it from the handler.
            "TRANSFORMERS_VERBOSITY": "warning",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
