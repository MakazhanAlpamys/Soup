"""#1407 — the streaming VRAM estimate must charge cuBLAS workspaces.

A streamed training step holds two cuBLAS workspaces (the forward thread's and
autograd's CUDA thread's), allocated through the caching allocator and therefore
part of ``torch.cuda.max_memory_allocated()`` — the quantity
``estimate_stream_peak_vram`` promises to bound. The fitted
``STREAM_FIXED_SLACK_BYTES`` already carries the compute-capability-8.6 baseline
(two default 8,320 KiB workspaces), so only the EXCESS over that baseline is
charged; the grid in ``test_v07203.py`` must stay pinned.

These tests pin the arithmetic on CPU; the CUDA-side check of the mirrored
device rule lives at the bottom under ``@pytest.mark.gpu``.
"""

import os
import subprocess
import sys

import pytest

from soup_cli.utils.layer_stream import (
    CUBLAS_WORKSPACE_BYTES_CC9_AND_ABOVE,
    CUBLAS_WORKSPACE_KIB_BELOW_CC9,
    CUBLAS_WORKSPACE_PAIRS_PER_STEP,
    cublas_workspace_bytes,
    estimate_stream_peak_vram,
    stream_cublas_workspaces_bytes,
)

#: The smallest legal argument set: every shape term at its floor. Anything the
#: cuBLAS term adds is visible in the difference.
_MINIMAL = dict(
    layer_bytes=0,
    buffers=0,
    extras_bytes=0,
    adapter_params=0,
    vocab_size=1,
    hidden_size=1,
    intermediate_size=1,
    n_layers=1,
    seq_len=1,
)


class TestConfigParsing:
    """``CUBLAS_WORKSPACE_CONFIG`` is :SIZE:COUNT segments in KiB."""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (":4096:8", 4096 * 8 * 1024),
            (":4096:2:16:8", (4096 * 2 + 16 * 8) * 1024),  # PyTorch's documented default
            (":4096:1", 4096 * 1024),
            (":0:0", 0),  # workspaces disabled outright
            (":128:1:256:2:512:3", (128 + 512 + 1536) * 1024),
        ],
    )
    def test_sizes_in_kib_sum_over_segments(self, config, expected):
        assert cublas_workspace_bytes(config) == expected

    @pytest.mark.parametrize("config", ["4096:8", ":4096", ":4096:2:16", "no-colons"])
    def test_malformed_values_raise(self, config):
        with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
            cublas_workspace_bytes(config)

    def test_unset_falls_back_to_the_sub_cc9_default(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        assert cublas_workspace_bytes(compute_capability=8) == (
            CUBLAS_WORKSPACE_KIB_BELOW_CC9 * 1024
        )

    def test_environment_is_read_when_config_is_not_passed(self, monkeypatch):
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:1")
        assert cublas_workspace_bytes(compute_capability=12) == 4096 * 1024


class TestDeviceRule:
    """Without the variable, PyTorch sizes the workspace by compute capability."""

    @pytest.mark.parametrize("cc", [9, 10, 11, 12])
    def test_hopper_and_blackwell_get_32_mib(self, monkeypatch, cc):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        assert cublas_workspace_bytes(compute_capability=cc) == (
            CUBLAS_WORKSPACE_BYTES_CC9_AND_ABOVE
        )

    def test_below_cc9_gets_the_documented_default(self):
        assert cublas_workspace_bytes(None, compute_capability=8) == 8_320 * 1024


class TestStreamCharge:
    """Only the excess over the fitted baseline is charged (#1407)."""

    def test_the_fitted_card_pays_nothing_new(self):
        # RTX 3050 (cc 8.6): the 8,320 KiB baseline is inside the slack
        # constant, so the v07203 grid arithmetic is untouched.
        assert stream_cublas_workspaces_bytes(None, compute_capability=8) == 0

    def test_hopper_blackwell_pays_the_excess_of_two_workspaces(self):
        baseline = CUBLAS_WORKSPACE_KIB_BELOW_CC9 * 1024
        expected = 2 * (CUBLAS_WORKSPACE_BYTES_CC9_AND_ABOVE - baseline)
        assert expected == 50_069_504  # 2 x (32 MiB - 8,320 KiB), per the issue
        assert stream_cublas_workspaces_bytes(None, compute_capability=12) == expected

    def test_a_smaller_config_pays_nothing_and_never_reduces(self):
        # :4096:1 shrinks the workspaces below the fitted baseline; charging
        # nothing (never a negative) is safe because the slack covers it.
        assert stream_cublas_workspaces_bytes(":4096:1", compute_capability=12) == 0
        assert stream_cublas_workspaces_bytes(":0:0", compute_capability=9) == 0

    def test_a_larger_config_pays_its_full_excess(self):
        assert stream_cublas_workspaces_bytes(":16384:2", compute_capability=8) == (
            CUBLAS_WORKSPACE_PAIRS_PER_STEP * (16384 * 2 * 1024 - 8_320 * 1024)
        )

    def test_estimate_adds_the_term_on_top_of_the_slack(self):
        base = estimate_stream_peak_vram(**_MINIMAL)
        charged = stream_cublas_workspaces_bytes(None, compute_capability=12)
        assert charged > 0
        assert estimate_stream_peak_vram(**_MINIMAL, cublas_workspaces_bytes=charged) == (
            base + charged
        )
        # The default stays 0: callers that know nothing about cuBLAS keep the
        # exact arithmetic the v07203 grid was fitted and pinned against.
        assert estimate_stream_peak_vram(**_MINIMAL) == base


class TestConfigIsReadPerProcess:
    """PyTorch reads CUBLAS_WORKSPACE_CONFIG once per process, so a same-process
    monkeypatch proves nothing about the env-var path a real run takes. The
    subprocess here is the smallest reproduction of that constraint."""

    def test_a_fresh_process_sees_its_own_environment(self):
        code = (
            "from soup_cli.utils.layer_stream import stream_cublas_workspaces_bytes;"
            "print(stream_cublas_workspaces_bytes(':4096:1', compute_capability=12))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "0"


@pytest.mark.gpu(reason="the cuBLAS workspace rule needs real CUDA (#1407)")
def test_a_real_step_holds_two_workspaces_of_the_assumed_size():
    """The device rule mirrored in ``cublas_workspace_bytes`` (32 MiB per pair
    on compute capability 9-12, 8,320 KiB below) is checked against the real
    allocator: after a warm-up forward+backward, one further fwd+bwd step on
    tiny tensors must push ``max_memory_allocated`` above the resident set by
    at least the two workspaces the estimate charges for (#1407 measured the
    step's own memory at ~450 KB on such a shape, i.e. noise next to them).
    """
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        pytest.skip("the operator sized the workspaces; the device rule is moot")
    cap = torch.cuda.get_device_capability()[0]
    size = cublas_workspace_bytes(compute_capability=cap)
    a = torch.randn(64, 64, device="cuda", requires_grad=True)

    def _step():
        (a @ a).sum().backward()

    _step()  # warm-up: both workspaces exist from here on
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak - resident >= CUBLAS_WORKSPACE_PAIRS_PER_STEP * size - (1 << 20), (
        f"cc {cap}: a real step added {peak - resident} bytes above the resident "
        f"set, less than the {CUBLAS_WORKSPACE_PAIRS_PER_STEP} x {size} workspaces "
        f"the estimate charges -- the mirrored PyTorch rule is stale"
    )
