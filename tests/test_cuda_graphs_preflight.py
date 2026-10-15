"""Tests for CUDA graph pre-flight checks before loading torch or weights (#1817)."""

import json
import sys
import types
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
@pytest.mark.parametrize("device", ["cuda", "cuda:1", "CUDA", "  cuda:0 "])
def test_a_stock_model_on_a_cuda_device_passes(tmp_path, cuda_present, model_type, device):
    cuda_graphs.check_cuda_graphs_preflight(
        device=device, model_target=_dir(tmp_path, {"model_type": model_type})
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda0", "cudax"])
def test_a_non_cuda_device_is_refused(device):
    with pytest.raises(RuntimeError, match="single CUDA device"):
        cuda_graphs.check_cuda_graphs_preflight(device=device)


@pytest.mark.parametrize("payload", [b"\xff\xfe\x00", b"{not json", b"[]"])
def test_an_unreadable_config_is_left_to_the_loader(tmp_path, cuda_present, payload):
    (tmp_path / "config.json").write_bytes(payload)
    cuda_graphs.check_cuda_graphs_preflight(device="cuda", model_target=str(tmp_path))


@pytest.mark.parametrize("payload", [b"\xff\xfe\x00", b"{not json", b"[]"])
def test_an_unreadable_adapter_config_is_left_to_the_loader(tmp_path, cuda_present, payload):
    (tmp_path / "adapter_config.json").write_bytes(payload)
    cuda_graphs.check_cuda_graphs_preflight(device="cuda", model_target=str(tmp_path))


def test_an_adapter_config_without_peft_type_is_left_to_the_loader(tmp_path, cuda_present):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    cuda_graphs.check_cuda_graphs_preflight(device="cuda", model_target=str(tmp_path))


@pytest.mark.parametrize("visible", ["", "-1", " -1 "])
def test_is_cuda_available_light_respects_cuda_visible_devices(monkeypatch, visible):
    import ctypes

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: pytest.fail("CDLL called"))
    assert cuda_graphs._is_cuda_available_light() is False


def _driver(monkeypatch, *, init_rc=0, count_rc=0, devices=1, platform="linux", symbols=True):
    import ctypes

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    opened = []
    lib = MagicMock()
    lib.cuInit.return_value = init_rc

    def get_count(ref):
        ref._obj.value = devices
        return count_rc

    lib.cuDeviceGetCount.side_effect = get_count
    if not symbols:
        del lib.cuInit

    def fake_cdll(name, *args, **kwargs):
        opened.append(name)
        return lib

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    return lib, opened


@pytest.mark.parametrize(
    ("platform", "library"), [("linux", "libcuda.so.1"), ("win32", "nvcuda.dll")]
)
def test_the_driver_library_per_platform(monkeypatch, platform, library):
    _, opened = _driver(monkeypatch, platform=platform)
    assert cuda_graphs._is_cuda_available_light() is True
    assert opened == [library]


def test_a_failing_cuinit_means_no_cuda(monkeypatch):
    lib, _ = _driver(monkeypatch, init_rc=100)
    assert cuda_graphs._is_cuda_available_light() is False
    lib.cuDeviceGetCount.assert_not_called()


def test_a_failing_device_count_call_means_no_cuda(monkeypatch):
    _driver(monkeypatch, count_rc=1, devices=1)
    assert cuda_graphs._is_cuda_available_light() is False


def test_zero_devices_means_no_cuda(monkeypatch):
    _driver(monkeypatch, devices=0)
    assert cuda_graphs._is_cuda_available_light() is False


def test_a_driver_without_the_symbols_means_no_cuda(monkeypatch):
    _driver(monkeypatch, symbols=False)
    assert cuda_graphs._is_cuda_available_light() is False


def test_macos_never_probes_the_driver(monkeypatch):
    _, opened = _driver(monkeypatch, platform="darwin")
    assert cuda_graphs._is_cuda_available_light() is False
    assert opened == []


def test_a_missing_driver_library_raises_oserror_means_no_cuda(monkeypatch):
    import ctypes

    _driver(monkeypatch)

    def raise_oserror(*args, **kwargs):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", raise_oserror)
    assert cuda_graphs._is_cuda_available_light() is False
    with pytest.raises(RuntimeError, match="single CUDA device"):
        cuda_graphs.check_cuda_graphs_preflight()


@pytest.mark.parametrize("available", [True, False])
def test_a_loaded_torch_is_trusted(monkeypatch, available):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available))
    _, opened = _driver(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert cuda_graphs._is_cuda_available_light() is available
    assert opened == []


def test_run_bench_train_refuses_before_importing_torch():
    import subprocess

    probe = (
        "import sys\n"
        "from soup_cli.bench.train_run import run_bench_train\n"
        "try:\n"
        "    run_bench_train(None, steps=2, warmup=2, device='cpu', load_dataset=lambda c: {})\n"
        "except ValueError as exc:\n"
        "    print('REFUSED', exc)\n"
        "print('TORCH', 'torch' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    # ansi-ok: subprocess probe stdout is uncoloured
    assert "REFUSED --steps 2 leaves nothing after --warmup 2" in proc.stdout
    assert "TORCH False" in proc.stdout
