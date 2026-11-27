"""Resident bitsandbytes NF4 coverage in the Triton interpreter and on CUDA."""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from soup_cli.utils import nf4_triton as nf4
from tests.conftest import strip_ansi


def test_module_import_stays_light():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from soup_cli.utils import nf4_triton; "
         "assert not {'torch', 'triton', 'bitsandbytes'} & sys.modules.keys()"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_opt_in_requires_one_without_importing_dependencies(monkeypatch, value):
    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"triton", "bitsandbytes", "torch"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    if value is None:
        monkeypatch.delenv("SOUP_NF4_TRITON", raising=False)
    else:
        monkeypatch.setenv("SOUP_NF4_TRITON", value)
    with nf4.patch_from_env() as active:
        assert not active


@pytest.mark.parametrize("missing", ["triton", "bitsandbytes"])
def test_missing_optional_dependency_leaves_opt_in_inactive(monkeypatch, missing):
    monkeypatch.setenv("SOUP_NF4_TRITON", "1")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setitem(sys.modules, missing, None)
    with nf4.patch_from_env() as active:
        assert not active


def test_unsupported_cuda_runtime_keeps_stock_hooks(monkeypatch):
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setenv("SOUP_NF4_TRITON", "1")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nf4, "_cuda_supported", lambda device: False)
    original = bnb.functional.dequantize_4bit
    with nf4.patch_from_env(device="cuda") as active:
        assert not active
        assert bnb.functional.dequantize_4bit is original


@pytest.mark.parametrize(
    ("backend", "device", "quantization", "stream", "expected"),
    [
        ("transformers", "cuda", "4bit", False, True),
        ("transformers", "cpu", "4bit", False, False),
        ("transformers", "cuda", "8bit", False, False),
        ("transformers", "cuda", "4bit", True, False),
        ("unsloth", "cuda", "4bit", False, False),
    ],
)
def test_train_scopes_opt_in_to_resident_transformers(
    monkeypatch, backend, device, quantization, stream, expected
):
    from io import StringIO

    from rich.console import Console

    from soup_cli.commands import train

    @contextmanager
    def patch(device):
        assert device == "cuda"
        try:
            calls.append("enter")
            yield True
        finally:
            calls.append("exit")

    calls = []
    screen = StringIO()
    monkeypatch.setattr(train, "console", Console(file=screen, width=100, force_terminal=True))
    monkeypatch.setattr(nf4, "patch_from_env", patch)
    cfg = SimpleNamespace(
        backend=backend, training=SimpleNamespace(quantization=quantization, stream_layers=stream)
    )
    with train._resident_nf4_dequantizer(cfg, device, announce=True):
        assert calls == (["enter"] if expected else [])
    assert calls == (["enter", "exit"] if expected else [])
    output = " ".join(strip_ansi(screen.getvalue()).split())
    assert output == ("NF4 dequantization: Triton kernel (SOUP_NF4_TRITON=1)" if expected else "")


def test_patch_restores_on_error_with_no_private_cuda_hook(monkeypatch):
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", None)
    monkeypatch.setenv("SOUP_NF4_TRITON", "1")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    original = bnb.functional.dequantize_4bit
    packed = torch.empty(10, 1, dtype=torch.uint8)
    with pytest.raises((AssertionError, ValueError)) as stock_error:
        original(packed)
    with pytest.raises(ValueError, match="invalid quant state"):
        with nf4.patch_from_env() as active:
            assert active
            with nf4.patch_from_env():
                wrapper = bnb.functional.dequantize_4bit
                assert wrapper is not original
                with pytest.raises(type(stock_error.value)) as error:
                    wrapper(packed)
                assert str(error.value) == str(stock_error.value)
            assert bnb.functional.dequantize_4bit is wrapper
            raise ValueError("invalid quant state")
    assert bnb.functional.dequantize_4bit is original


def test_cpu_uses_stock_even_with_opt_in(monkeypatch):
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes", minversion="0.48.0")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setenv("SOUP_NF4_TRITON", "1")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    packed, state = bnb.functional.quantize_4bit(torch.ones(16, 64), quant_type="nf4")
    expected = bnb.functional.dequantize_4bit(packed, state)

    def unexpected_launch(*args):
        raise AssertionError("CPU NF4 called Triton without the interpreter")

    monkeypatch.setattr(nf4, "_launch", unexpected_launch)
    with nf4.patch_from_env() as active:
        assert active
        assert torch.equal(bnb.functional.dequantize_4bit(packed, state), expected)


def test_unsafe_decode_metadata_goes_to_stock(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    shape = (16, 64)
    packed = torch.zeros(512, 1, dtype=torch.uint8)
    scale = torch.ones(16, dtype=torch.float32)
    codes = torch.zeros(16, dtype=torch.uint8)
    nested = (torch.ones(1), torch.zeros(256), torch.tensor(1.0), 256)
    full = dict(packed=packed, shape=shape, dtype=torch.float32, blocksize=64, scale=scale)

    def unexpected_launch(*args):
        raise AssertionError("unsafe NF4 metadata reached Triton")

    monkeypatch.setattr(nf4, "_launch", unexpected_launch)
    for overrides in (
        {"shape": (1024,)},
        {"shape": (0, 64)},
        {"blocksize": 7},
        {"dtype": torch.float64},
        {"packed": packed.view(torch.int16)},
        {"scale": codes},
        {"scale": scale[:-1]},
        {"scale": scale.to("meta")},
        {"scale": codes, "nested": (nested[0], nested[1][:-1], *nested[2:])},
        {"scale": codes, "nested": (*nested[:2], torch.ones(1, 1), nested[3])},
        {"out": torch.empty(64, 16).t()},
        {"out": torch.empty(shape, dtype=torch.float16)},
    ):
        assert nf4._decode(**(full | overrides)) is None, overrides.keys()

    shared = torch.zeros(shape)
    assert nf4._decode(**(full | {"out": shared, "scale": shared.flatten()[:16]})) is None
    with monkeypatch.context() as settings:
        settings.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)
        assert nf4._decode(**full) is None
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert nf4._decode(**full) is None


def _exercise_parity(device):
    import bitsandbytes.functional as bnb
    import torch
    import triton

    runs = []
    launch = nf4._launch

    def record_launch(*args, **kwargs):
        result = launch(*args, **kwargs)
        runs.append(True)
        return result

    # The spy delegates to the real kernel and proves that a stock fallback cannot pass parity.
    nf4._launch = record_launch
    torch.manual_seed(4)
    cases = [
        ((96, 128), 64, True, torch.bfloat16),
        ((300, 70), 64, True, torch.float16),  # final partial scale and launch tiles
        ((32, 256), 128, False, torch.bfloat16),
        ((48, 64), 64, False, torch.float32),
    ]
    for shape, blocksize, nested, dtype in cases:
        weight = (torch.randn(shape, device=device) * 0.05).to(dtype)
        packed, state = bnb.quantize_4bit(
            weight, blocksize=blocksize, compress_statistics=nested, quant_type="nf4",
        )
        ref = bnb.dequantize_4bit(packed, state)
        x = torch.randn(5, shape[1], device=device, dtype=dtype)
        bias = torch.randn(shape[0], device=device, dtype=dtype)
        args = (x, packed, state.shape, state.state2.absmax if nested else state.absmax,
                blocksize, "nf4", bias)
        kwargs = (dict(absmax_8bit=state.absmax, absmax_code=state.state2.code,
                       absmax_offset=state.offset) if nested else {})
        if device == "cuda":
            from bitsandbytes.backends.cuda import ops

            linear_ref = ops._dequant_linear_fallback(*args, **kwargs)
        else:
            linear_ref = torch.nn.functional.linear(x, ref, bias)

            def unexpected_fallback(*args, **kwargs):
                raise AssertionError("supported interpreter call delegated to stock linear")

            linear = nf4._wrap_linear(unexpected_fallback)

        runs.clear()
        with nf4.patch_from_env() as active:
            assert active
            out = torch.empty_like(ref)
            actual = bnb.dequantize_4bit(packed.t(), state, out=out)
            assert actual.data_ptr() == out.data_ptr()
            assert actual.shape == ref.t().shape
            integer_dtype = torch.int32 if dtype == torch.float32 else torch.int16
            assert torch.equal(out.view(integer_dtype), ref.view(integer_dtype)), shape
            if device == "cuda":
                linear = ops._dequant_linear_fallback
            assert torch.equal(linear(*args, **kwargs), linear_ref), shape
        assert len(runs) == 2, (shape, dtype, nested)

    for shape, quant_type in [((16, 64), "fp4"), ((15, 63), "nf4")]:
        weight = torch.randn(shape, device=device)
        packed, state = bnb.quantize_4bit(weight, quant_type=quant_type)
        expected = bnb.dequantize_4bit(packed, state)
        runs.clear()
        with nf4.patch_from_env():
            assert torch.equal(bnb.dequantize_4bit(packed, state), expected)
            if device == "cuda":
                x = torch.randn(5, shape[1], device=device)
                got = ops._dequant_linear_fallback(
                    x, packed, shape, state.absmax, state.blocksize, quant_type,
                )
                assert torch.equal(got, torch.nn.functional.linear(x, expected))
        assert not runs

    packed, state = bnb.quantize_4bit(torch.randn(16, 64, device=device), quant_type="nf4")
    expected = bnb.dequantize_4bit(packed, state)
    strided = torch.empty(packed.shape[0] * 2, 1, device=device, dtype=torch.uint8)[::2]
    strided.copy_(packed)
    with nf4.patch_from_env():
        assert torch.equal(bnb.dequantize_4bit(strided, state), expected)
    assert not runs

    def unavailable_kernel(*args):
        raise triton.OutOfResources(2, 1, "registers")

    nf4._launch = unavailable_kernel
    with nf4.patch_from_env():
        assert torch.equal(bnb.dequantize_4bit(packed, state), expected)


def _run_parity(device):
    env = {**os.environ, "TRITON_INTERPRET": str(int(device == "cpu")), "SOUP_NF4_TRITON": "1"}
    result = subprocess.run(
        [sys.executable, "-c",
         f"from tests.test_nf4_triton import _exercise_parity; _exercise_parity({device!r})"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def test_interpreter_parity_and_fallback():
    pytest.importorskip("bitsandbytes", minversion="0.48.0")
    pytest.importorskip("triton")
    _run_parity("cpu")


@pytest.fixture
def cuda_bitsandbytes():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    pytest.importorskip("triton")
    if not torch.version.cuda:
        pytest.skip("Triton NF4 requires NVIDIA CUDA")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("Triton NF4 requires compute capability 8.0 or newer")
    ops = pytest.importorskip("bitsandbytes.backends.cuda.ops")
    if not hasattr(ops, "_dequant_linear_fallback"):
        pytest.skip("This bitsandbytes release has no CUDA dequantized-linear hook")
    return bnb


@pytest.mark.gpu
def test_compiled_cuda_parity_and_fallback(cuda_bitsandbytes):
    _run_parity("cuda")


@pytest.mark.gpu
def test_resident_lora_step_matches_stock(cuda_bitsandbytes, monkeypatch):
    import torch

    peft = pytest.importorskip("peft")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setenv("SOUP_NF4_TRITON", "1")
    torch.manual_seed(19)
    model = torch.nn.Sequential(cuda_bitsandbytes.nn.Linear4bit(
        64, 32, compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True,
    )).to("cuda")
    model.is_loaded_in_4bit = True
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, target_modules=["0"]))
    adapters = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in adapters]
    optimizer = torch.optim.SGD(adapters, lr=0.1)
    x = torch.randn(2048, 64, device="cuda", dtype=torch.float16).requires_grad_()

    def backward():
        loss = model(x).float().square().mean()
        loss.backward()
        return loss.detach(), x.grad.clone(), [p.grad.clone() for p in adapters]

    ref_loss, ref_dx, ref_grads = backward()
    optimizer.zero_grad()
    x.grad = None
    launches = []
    original_launch = nf4._launch

    def track_launch(*args):
        result = original_launch(*args)
        launches.append(True)
        return result

    monkeypatch.setattr(nf4, "_launch", track_launch)
    with nf4.patch_from_env() as active:
        assert active
        loss, dx, grads = backward()
        optimizer.step()
    assert len(launches) == 2  # unfused forward and input-gradient dequantization
    assert torch.equal(loss, ref_loss)
    assert torch.equal(dx, ref_dx)
    assert all(torch.equal(actual, ref) for actual, ref in zip(grads, ref_grads, strict=True))
    assert any(not torch.equal(p, before) for p, before in zip(adapters, initial, strict=True))
