"""Tests for CUDA graph pre-flight checks before loading torch or weights (#1817)."""

import json
from unittest.mock import MagicMock

import pytest

from soup_cli.utils import cuda_graphs


@pytest.fixture
def cuda_present(monkeypatch):
    monkeypatch.setattr(cuda_graphs, "_is_cuda_available_light", lambda: True)


def _dir(tmp_path, config=None, adapter=None):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if adapter is not None:
        (tmp_path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("config", "adapter", "message"),
    [
        ({"model_type": "gpt2"}, None, "stock Qwen2 and Llama"),
        ({"model_type": "qwen2", "quantization_config": {"bits": 4}}, None, "unquantized"),
        (None, {"peft_type": "IA3"}, "LoRA adapters only, not IA3"),
        (None, {"peft_type": "LORA", "use_dora": True}, "DoRA"),
    ],
)
def test_the_model_files_refuse_before_any_load(tmp_path, cuda_present, config, adapter, message):
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.check_cuda_graphs_preflight(
            device="cuda", model_target=_dir(tmp_path, config, adapter)
        )


@pytest.mark.parametrize("model_type", ["qwen2", "llama"])
@pytest.mark.parametrize("device", ["cuda", "cuda:1", "CUDA"])
def test_a_stock_model_on_a_cuda_device_passes(tmp_path, cuda_present, model_type, device):
    cuda_graphs.check_cuda_graphs_preflight(
        device=device, model_target=_dir(tmp_path, {"model_type": model_type})
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_a_non_cuda_device_is_refused(device):
    with pytest.raises(RuntimeError, match="single CUDA device"):
        cuda_graphs.check_cuda_graphs_preflight(device=device)


@pytest.mark.parametrize("payload", [b"\xff\xfe\x00", b"{not json", b"[]"])
def test_an_unreadable_config_is_left_to_the_loader(tmp_path, cuda_present, payload):
    (tmp_path / "config.json").write_bytes(payload)
    cuda_graphs.check_cuda_graphs_preflight(device="cuda", model_target=str(tmp_path))


@pytest.mark.parametrize("visible", ["", "-1", " -1 "])
def test_is_cuda_available_light_respects_cuda_visible_devices(monkeypatch, visible):
    import ctypes
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: pytest.fail("CDLL called"))
    assert cuda_graphs._is_cuda_available_light() is False


def test_is_cuda_available_light_handles_cdll_oserror(monkeypatch):
    import ctypes
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    def fail_cdll(*args, **kwargs):
        raise OSError("libcuda.so.1 not found")

    monkeypatch.setattr(ctypes, "CDLL", fail_cdll)
    assert cuda_graphs._is_cuda_available_light() is False


def test_is_cuda_available_light_detects_device_via_driver(monkeypatch):
    import ctypes
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    fake_lib = MagicMock()
    fake_lib.cuInit.return_value = 0

    def fake_get_count(ref):
        ref._obj.value = 1
        return 0

    fake_lib.cuDeviceGetCount.side_effect = fake_get_count
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: fake_lib)
    assert cuda_graphs._is_cuda_available_light() is True
