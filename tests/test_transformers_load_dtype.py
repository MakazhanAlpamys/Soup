"""Actual local model loads must not emit Transformers' deprecated dtype warning."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["from_pretrained", "from_config"])
def test_tiny_local_cpu_model_load_has_no_deprecated_dtype_warning(mode):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests._tiny_hf_models import tiny_model_dir

    model_dir = tiny_model_dir("gpt2")
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
    warnings.simplefilter("error")
    if sys.argv[2] == "from_pretrained":
        from soup_cli.utils.live_eval import load_model_and_tokenizer
        model, tokenizer, device = load_model_and_tokenizer(
            sys.argv[1], device="cpu", dtype="auto",
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
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
