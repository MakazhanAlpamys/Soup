"""Regression tests for #1478: ``soup serve --device`` was documented but the
transformers loader ignored it — ``_load_model`` loaded the base model and the
merged model with a hard-coded ``device_map="auto"`` regardless of the flag,
so a model asked onto the CPU could still land on a GPU (and a forced CPU load
used float16). The speculative draft model's loader did honour ``--device``,
so the main model and its draft could land on different devices.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _mock_env():
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    return tokenizer, model


def _assert_cpu_placement(load) -> None:
    kwargs = load.call_args.kwargs
    assert kwargs.get("device_map") == {"": "cpu"}, (
        "serve._load_model(device='cpu') must pass device_map={'': 'cpu'} "
        f"to from_pretrained (got {kwargs.get('device_map')!r})"
    )
    assert kwargs.get("torch_dtype") == torch.float32, (
        "serve._load_model(device='cpu') must load in float32 "
        f"(got {kwargs.get('torch_dtype')})"
    )


def test_serve_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.serve import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load:
        _load_model("some-org/some-model", None, False, "cpu")
        _assert_cpu_placement(load)


def test_serve_load_model_cpu_adapter_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.serve import _load_model

    tokenizer, model = _mock_env()
    peft_model = MagicMock()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load, patch(
        "peft.PeftModel.from_pretrained", return_value=peft_model
    ):
        _load_model("runs/adapter", "some-org/base", True, "cpu")
        _assert_cpu_placement(load)


def test_serve_load_model_without_device_behaves_as_before():
    from soup_cli.commands.serve import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load:
        _load_model("some-org/some-model", None, False, "cuda")
        kwargs = load.call_args.kwargs
        assert kwargs.get("device_map") == "auto"
        assert kwargs.get("torch_dtype") == torch.float16


def test_serve_load_model_bfloat16_kv_cache_keeps_bfloat16_on_cpu():
    from soup_cli.commands.serve import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load:
        _load_model(
            "some-org/some-model", None, False, "cpu", kv_cache_dtype="bfloat16"
        )
        kwargs = load.call_args.kwargs
        assert kwargs.get("device_map") == {"": "cpu"}
        assert kwargs.get("torch_dtype") == torch.bfloat16


def test_serve_load_model_unknown_device_is_refused_with_message():
    from soup_cli.commands.serve import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ):
        with pytest.raises(ValueError) as excinfo:
            _load_model("some-org/some-model", None, False, "tpu")
        assert "cpu" in str(excinfo.value)
        assert "cuda" in str(excinfo.value)
        assert "mps" in str(excinfo.value)


def test_no_hard_coded_device_map_in_serve_loader():
    source = (
        Path(__file__).parents[1] / "src" / "soup_cli" / "commands" / "serve.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _load_model(")
    end = source.index("def _load_named_adapters(")
    body = source[start:end]
    assert 'device_map="auto"' not in body, (
        "serve._load_model must pass the --device-resolved device_map to "
        "every from_pretrained call, not a hard-coded device_map=\"auto\""
    )
