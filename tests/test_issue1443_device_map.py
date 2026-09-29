"""Regression tests for #1443: `--device` is echoed but never used by
`soup infer`, `soup chat` and `soup diff` — every `_load_model` loads with
`device_map="auto"` and the hard-coded `torch_dtype=torch.float16`
regardless of the flag. `device_map="auto"` lets accelerate place the model
on any visible accelerator, which is exactly wrong when the user asked for
the CPU on purpose (to leave VRAM free for a training job, or because the
model does not fit on the GPU).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch


def _mock_env():
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    return tokenizer, model


def _assert_cpu_placement(name: str, load, model) -> None:
    kwargs = load.call_args.kwargs
    device_map = kwargs.get("device_map")
    dtype = kwargs.get("torch_dtype")

    assert device_map != "auto", (
        f"{name}._load_model(device='cpu') still passes device_map='auto' "
        f"to from_pretrained; with a visible GPU, accelerate would still "
        f"place the model there although the user asked for the CPU"
    )
    assert device_map == {"": "cpu"} or model.to.called, (
        f"{name}._load_model(device='cpu') never resolves the model onto "
        f"the CPU (device_map={device_map!r}, model.to() called="
        f"{model.to.called})"
    )
    assert dtype != torch.float16, (
        f"{name}._load_model(device='cpu') still hard-codes float16, which "
        f"is not a CPU-appropriate dtype (got {dtype})"
    )


def test_infer_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.infer import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load, patch(
        "soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False
    ), patch(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        return_value=False,
    ):
        _load_model("some-org/some-model", None, "cpu")
        _assert_cpu_placement("infer", load, model)


def test_chat_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.chat import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load:
        _load_model("some-org/some-model", None, False, "cpu")
        _assert_cpu_placement("chat", load, model)


def test_diff_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.diff import _load_model

    tokenizer, model = _mock_env()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as load, patch(
        "soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False
    ), patch(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        return_value=False,
    ):
        _load_model("some-org/some-model", None, "cpu")
        _assert_cpu_placement("diff", load, model)
