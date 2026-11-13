"""#1407 — budget cuBLAS workspaces retained by streamed training steps."""

import pytest

from soup_cli.utils.layer_stream import cublas_workspace_bytes, estimate_stream_peak_vram


@pytest.mark.parametrize(
    ("capability", "config", "expected"),
    [
        ((8, 6), None, 2 * 8_320 * 1024),
        ((9, 0), None, 2 * 32 * 1024 * 1024),
        ((12, 0), None, 2 * 32 * 1024 * 1024),
        ((8, 6), ":4096:8", 2 * 4096 * 8 * 1024),
        ((8, 6), ":4096:2:16:8", 2 * 8320 * 1024),
        ((8, 6), ":4096:1", 2 * 4096 * 1024),
        ((8, 6), ":0:0", 0),
        (None, None, 2 * 32 * 1024 * 1024),
    ],
)
def test_workspace_charge_tracks_config_and_device_defaults(capability, config, expected):
    assert cublas_workspace_bytes(
        compute_capability=capability, workspace_config=config
    ) == expected


@pytest.mark.parametrize("config", ["4096:8", ":4096", ":4096:x", ":4096:0", ":0:8"])
def test_malformed_or_unsupported_workspace_config_is_refused(config):
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        cublas_workspace_bytes(compute_capability=(8, 6), workspace_config=config)


@pytest.mark.parametrize("config", [":4096:0", ":0:8"])
def test_workspace_config_requires_both_size_and_count_zero_to_disable(config):
    """PyTorch documents :0:0 as the no-workspace setting, not either half-zero form."""
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        cublas_workspace_bytes(compute_capability=(8, 6), workspace_config=config)


def test_estimator_adds_workspace_bytes_without_changing_other_terms():
    shape = dict(
        layer_bytes=1_000_000,
        buffers=2,
        extras_bytes=500_000,
        adapter_params=100,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        n_layers=2,
        seq_len=16,
    )
    base = estimate_stream_peak_vram(**shape)
    charge = cublas_workspace_bytes(
        compute_capability=(8, 6), workspace_config=":4096:1"
    )
    assert estimate_stream_peak_vram(**shape, cublas_workspace_bytes=charge) == base + charge


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_estimator_rejects_invalid_workspace_bytes(value):
    with pytest.raises(ValueError, match="cublas_workspace_bytes"):
        estimate_stream_peak_vram(
            layer_bytes=1,
            buffers=1,
            extras_bytes=0,
            adapter_params=0,
            vocab_size=1,
            hidden_size=1,
            intermediate_size=1,
            n_layers=1,
            seq_len=1,
            cublas_workspace_bytes=value,
        )


def test_unreadable_capability_uses_largest_documented_default_charge():
    assert cublas_workspace_bytes(compute_capability=None) == 64 * 1024 * 1024


def test_stream_setup_wires_workspace_charge_into_estimator_and_panel(monkeypatch):
    from types import SimpleNamespace

    from soup_cli.trainer import stream_setup
    from soup_cli.utils import layer_stream

    class Setup(stream_setup.StreamingSetupMixin):
        _STREAM_ROWS_PER_EXAMPLE = 1

        @staticmethod
        def _stream_shape_config(model_config):
            return model_config

        @staticmethod
        def _stream_intermediate_size(model_config):
            return model_config.intermediate_size

        @staticmethod
        def _estimate_adapter_params(*_args, **_kwargs):
            return 0

    setup = Setup()
    setup.device = "cuda"
    config = SimpleNamespace(data=SimpleNamespace(max_length=16))
    train = SimpleNamespace(
        batch_size=1,
        stream_buffers=2,
        stream_vram_probe=False,
        stream_vram_override=4_000_000_000,
        gradient_accumulation_steps=1,
        lora=None,
    )
    model = SimpleNamespace(vocab_size=1000, hidden_size=64, intermediate_size=128)
    index = SimpleNamespace(n_layers=2, total_params=0)
    monkeypatch.setattr(stream_setup, "_stream_compute_capability_for_device", lambda *_: None)
    monkeypatch.setattr(stream_setup, "console", SimpleNamespace(print=lambda *_a, **_k: None))
    monkeypatch.setattr(
        layer_stream,
        "estimate_stream_peak_vram",
        lambda **kwargs: seen.setdefault("charge", kwargs["cublas_workspace_bytes"]) or 1,
    )
    monkeypatch.setattr(
        layer_stream,
        "decide_stream_fit",
        lambda **_kw: SimpleNamespace(fits=True, reason="fits"),
    )
    monkeypatch.setattr(
        layer_stream,
        "resolve_available_vram_bytes",
        lambda *_a, **_kw: 4_000_000_000,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_gemm_tflops",
        lambda *_a, **_kw: None,
    )
    import sys

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda *_a, **_kw: (4_000_000_000, 8_000_000_000))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seen = {}
    lines, _ = setup._stream_budget_lines(
        config,
        train,
        model_config=model,
        layer_bytes=1_000_000,
        embed_bytes=1_000_000,
        index=index,
        on_cuda=True,
        large_layer_bytes=0,
    )
    expected = cublas_workspace_bytes(compute_capability=None)
    assert seen["charge"] == expected
    assert any(
        "cuBLAS" in line
        and "64.0 MiB" in line
        and "PyTorch allocator workspace" in line
        for line in lines
    )


@pytest.mark.gpu(reason="the cuBLAS allocator reading requires CUDA")
def test_configured_workspace_is_visible_to_allocator_in_fresh_process():
    """Set the cuBLAS config before torch initializes and read its retained allocation."""
    import json
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json
        import os

        import torch

        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:1"
        torch.cuda.init()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        left = torch.randn((128, 128), device="cuda")
        right = torch.randn((128, 128), device="cuda")
        output = torch.mm(left, right)
        loss = output.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        del left, right, output, loss
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - baseline
        print(json.dumps({"retained_workspace_bytes": retained}))
        """
    )
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=env,
        text=True,
        timeout=120,
    )
    reading = json.loads(result.stdout.strip().splitlines()[-1])
    assert reading["retained_workspace_bytes"] == 8 * 1024 * 1024, result.stdout


@pytest.mark.gpu(reason="the streamed-step peak comparison needs CUDA")
def test_real_streamed_step_peak_does_not_exceed_preflight(tmp_path, monkeypatch):
    import os

    import torch

    from soup_cli.utils import layer_stream
    from tests.test_v07204 import _build_streamed_wrapper

    estimates = []
    estimate = layer_stream.estimate_stream_peak_vram

    def record_estimate(**kwargs):
        result = estimate(**kwargs)
        estimates.append((result, kwargs.get("cublas_workspace_bytes", 0)))
        return result

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", record_estimate)
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task="sft", batch_size=1
    )
    assert estimates and estimates[0][1] == cublas_workspace_bytes(
        compute_capability=torch.cuda.get_device_capability(),
        workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )

    wrapper.trainer.args.max_steps = 1
    wrapper.trainer.train()  # Warm up both the forward and autograd workspaces.
    torch.cuda.synchronize()
    predicted = max(value for value, _workspace in estimates)
    torch.cuda.reset_peak_memory_stats()
    wrapper.trainer.train()
    torch.cuda.synchronize()
    measured = int(torch.cuda.max_memory_allocated())
    assert predicted >= measured, f"preflight={predicted} measured_peak={measured}"
