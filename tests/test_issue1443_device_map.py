"""Regression tests for #1443: `--device` is echoed but never used by
`soup infer`, `soup chat` and `soup diff` — every `_load_model` loads with
`device_map="auto"` and the hard-coded `torch_dtype=torch.float16`
regardless of the flag. `device_map="auto"` lets accelerate place the model
on any visible accelerator, which is exactly wrong when the user asked for
the CPU on purpose (to leave VRAM free for a training job, or because the
model does not fit on the GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from typer.testing import CliRunner

from tests.conftest import strip_ansi

runner = CliRunner()


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
    assert dtype == torch.float32, (
        f"{name}._load_model(device='cpu') must load in float32 on the CPU (got {dtype})"
    )


def test_infer_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.infer import _load_model

    tokenizer, model = _mock_env()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load,
        patch("soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False),
        patch(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
            return_value=False,
        ),
    ):
        _load_model("some-org/some-model", None, "cpu")
        _assert_cpu_placement("infer", load, model)


def test_chat_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.chat import _load_model

    tokenizer, model = _mock_env()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load,
    ):
        _load_model("some-org/some-model", None, False, "cpu")
        _assert_cpu_placement("chat", load, model)


def test_diff_load_model_cpu_uses_cpu_device_map_and_dtype():
    from soup_cli.commands.diff import _load_model

    tokenizer, model = _mock_env()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load,
        patch("soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False),
        patch(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
            return_value=False,
        ),
    ):
        _load_model("some-org/some-model", None, "cpu")
        _assert_cpu_placement("diff", load, model)


def _write_adapter_config(model_dir: Path, base_model: str = "some-org/base-model") -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "r": 8})
    )


def test_infer_load_model_cpu_uses_cpu_device_map_and_dtype_on_adapter_branch(
    tmp_path: Path,
):
    """The adapter branch loads the *base* model with the same device_map/dtype (#1443).

    Both `_load_model` branches share one `device_map, torch_dtype` pair
    computed up front, but the existing cpu test above only ever exercises
    the full-model (else) branch, since its `model_path` is not a real local
    directory. Reverting only the adapter branch's `from_pretrained` call
    back to a hard-coded `device_map="auto"` must fail here even though the
    full-model test still passes.
    """
    from soup_cli.commands.infer import _load_model

    _write_adapter_config(tmp_path)
    tokenizer, model = _mock_env()
    base_obj = MagicMock()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=base_obj) as load,
        patch("peft.PeftModel.from_pretrained", return_value=model),
        patch("soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False),
        patch(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
            return_value=False,
        ),
    ):
        _load_model(str(tmp_path), None, "cpu")
        _assert_cpu_placement("infer (adapter branch)", load, base_obj)


def test_chat_load_model_cpu_uses_cpu_device_map_and_dtype_on_adapter_branch(
    tmp_path: Path,
):
    from soup_cli.commands.chat import _load_model

    _write_adapter_config(tmp_path)
    tokenizer, model = _mock_env()
    base_obj = MagicMock()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=base_obj) as load,
        patch("peft.PeftModel.from_pretrained", return_value=model),
    ):
        _load_model(str(tmp_path), "some-org/base-model", True, "cpu")
        _assert_cpu_placement("chat (adapter branch)", load, base_obj)


def test_diff_load_model_cpu_uses_cpu_device_map_and_dtype_on_adapter_branch(
    tmp_path: Path,
):
    from soup_cli.commands.diff import _load_model

    _write_adapter_config(tmp_path)
    tokenizer, model = _mock_env()
    base_obj = MagicMock()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=base_obj) as load,
        patch("peft.PeftModel.from_pretrained", return_value=model),
        patch("soup_cli.utils.trust_remote.resolve_trust_remote_code", return_value=False),
        patch(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
            return_value=False,
        ),
    ):
        _load_model(str(tmp_path), None, "cpu")
        _assert_cpu_placement("diff (adapter branch)", load, base_obj)


def test_chat_cli_without_device_flag_uses_the_auto_detected_device(tmp_path: Path):
    """Without `--device`, the auto-detected device must be what reaches the
    load, so the panel and the actual placement agree (#1443 acceptance
    criterion 3), and the CLI assertion goes through the ANSI-strip helper
    (criterion 4).
    """
    from soup_cli.cli import app

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    tokenizer, model = _mock_env()
    with (
        patch("soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU (no GPU detected)")),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load,
    ):
        result = runner.invoke(app, ["chat", "--model", str(model_dir)], input="")

    output = strip_ansi(result.output)
    assert "Device: cpu" in output, output
    _assert_cpu_placement("chat (CLI auto-detect)", load, model)


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", ({"": "cpu"}, torch.float32)),
        ("CPU", ({"": "cpu"}, torch.float32)),
        ("mlx", ({"": "cpu"}, torch.float32)),
        ("mps", ({"": "mps"}, torch.float16)),
        ("cuda:1", ({"": "cuda:1"}, torch.float16)),
        # Plain `cuda` deliberately stays "auto" (every visible GPU); pinned
        # here so the choice cannot drift silently.
        ("cuda", ("auto", torch.float16)),
        (None, ("auto", torch.float16)),
        ("", ("auto", torch.float16)),
    ],
)
def test_resolver_table(monkeypatch, device, expected):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "cuda_supports_bf16", lambda: False)
    assert gpu.resolve_inference_device_map_and_dtype(device) == expected


def test_resolver_keeps_float16_even_when_cuda_supports_bf16(monkeypatch):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "cuda_supports_bf16", lambda: True)
    assert gpu.resolve_inference_device_map_and_dtype("cuda") == ("auto", torch.float16)
    assert gpu.resolve_inference_device_map_and_dtype("cuda:0") == (
        {"": "cuda:0"},
        torch.float16,
    )
    assert gpu.resolve_inference_device_map_and_dtype("mps") == ({"": "mps"}, torch.float16)


@pytest.mark.parametrize("device", ["foo", "gpu", "tpu", "xpu", "auto", "cuda:abc", "cuda:0,1"])
def test_resolver_refuses_an_unknown_device(device):
    from soup_cli.utils import gpu

    with pytest.raises(ValueError, match="cpu, cuda"):
        gpu.resolve_inference_device_map_and_dtype(device)


@pytest.mark.parametrize(("device", "expected"), [("MPS", {"": "mps"}), ("CUDA:0", {"": "cuda:0"})])
def test_pinned_device_map_uses_the_normalised_name(monkeypatch, device, expected):
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu, "cuda_supports_bf16", lambda: False)
    device_map, dtype = gpu.resolve_inference_device_map_and_dtype(device)
    assert device_map == expected and dtype == torch.float16
    assert torch.device(device_map[""])  # what accelerate will hand to torch


def _prep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({}))
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"prompt": "hi"}) + "\n")
    return str(model_dir)


def test_infer_cli_hands_the_resolved_device_to_the_loader(tmp_path, monkeypatch):
    from soup_cli.cli import app

    model_dir = _prep(tmp_path, monkeypatch)
    with patch(
        "soup_cli.commands.infer._load_model",
        side_effect=RuntimeError("stop after the load call"),
    ) as load:
        result = runner.invoke(
            app,
            ["infer", "-m", model_dir, "-i", "prompts.jsonl", "-o", "o.jsonl", "--device", "cpu"],
        )
    assert "Device: cpu" in " ".join(strip_ansi(result.output).split()), result.output
    assert load.call_args.args[2] == "cpu", load.call_args


def test_diff_cli_hands_the_resolved_device_to_the_loader(tmp_path, monkeypatch):
    from soup_cli.cli import app

    model_dir = _prep(tmp_path, monkeypatch)
    with patch(
        "soup_cli.commands.diff._load_model",
        side_effect=RuntimeError("stop after the load call"),
    ) as load:
        result = runner.invoke(
            app,
            ["diff", "-a", model_dir, "-b", model_dir, "--prompt", "hi", "--device", "cpu"],
        )
    assert "Device: cpu" in " ".join(strip_ansi(result.output).split()), result.output
    assert load.call_args_list[0].args[2] == "cpu", load.call_args_list
