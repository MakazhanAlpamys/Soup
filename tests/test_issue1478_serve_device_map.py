"""Regression tests for #1478: ``soup serve --device`` is accepted but the
transformers loader (``serve._load_model``) always loads with the hard-coded
``device_map="auto"``, so ``--device cpu`` still lands on a visible GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch


def _load(device, is_adapter=False):
    from soup_cli.commands.serve import _load_model

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load, patch("peft.PeftModel.from_pretrained", return_value=model):
        _load_model(
            model_path="some-org/some-model",
            base_model="some-org/base-model" if is_adapter else None,
            is_adapter=is_adapter,
            device=device,
        )
    return load, model


def _assert_cpu(load, model):
    kwargs = load.call_args.kwargs
    device_map = kwargs.get("device_map")
    assert device_map != "auto", (
        "serve._load_model(device='cpu') still passes device_map='auto'; with "
        "a visible GPU, accelerate would place the model there"
    )
    assert device_map in ("cpu", {"": "cpu"}) or model.to.called, device_map
    assert kwargs.get("torch_dtype") == torch.float32, kwargs.get("torch_dtype")


def test_serve_load_model_cpu_plain_model_is_pinned_to_cpu():
    load, model = _load("cpu")
    _assert_cpu(load, model)


def test_serve_load_model_cpu_base_plus_adapter_is_pinned_to_cpu():
    load, model = _load("cpu", is_adapter=True)
    _assert_cpu(load, model)


def test_serve_load_model_cpu_keeps_explicit_bfloat16():
    from soup_cli.commands.serve import _load_model

    with patch("transformers.AutoTokenizer.from_pretrained"), patch(
        "transformers.AutoModelForCausalLM.from_pretrained"
    ) as load:
        _load_model(
            model_path="some-org/some-model",
            base_model=None,
            is_adapter=False,
            device="cpu",
            kv_cache_dtype="bfloat16",
        )
    assert load.call_args.kwargs["torch_dtype"] == torch.bfloat16


def test_serve_load_model_gpu_devices_keep_auto_placement():
    for device in ("cuda", "mps"):
        load, _ = _load(device)
        kwargs = load.call_args.kwargs
        assert kwargs["device_map"] == "auto", device
        assert kwargs["torch_dtype"] == torch.float16, device
