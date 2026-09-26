"""Comprehensive test suite for Issue #423 & Apple Silicon hardware detection.

Covers MLX detection, MPS/MLX precedence disambiguation, MPS zero-byte memory
invariant for hardware-fit preflight, host-agnostic CPU fallbacks, and explicit
quantization preservation for MLX vs CPU downgrade guards.

The quantization tests call ``resolve_quantization()`` -- the real guard
function extracted from ``train.py`` -- so a mutation that disables the
guard causes a test failure.  The maintainer's mutation audit disabled the
guard and the previous test suite survived; these tests do not.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.utils import gpu as gpu_utils
from soup_cli.utils.gpu import resolve_quantization


class TestMLXDeviceDetection:
    """Test suite for detect_device(), get_gpu_info(), and quantization guards."""

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M2 Max"})
    def test_detect_device_pure_apple_silicon_mlx(
        self, mock_chip, mock_detect, mock_apple
    ):
        """Pure Apple Silicon with MLX returns 'mlx' device."""
        device, name = gpu_utils.detect_device(backend="mlx")
        assert device == "mlx"
        assert name == "Apple Silicon (Apple M2 Max)"

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M3 Pro"})
    def test_detect_device_dual_stack_mlx_requested(
        self, mock_chip, mock_detect, mock_apple
    ):
        """When backend='mlx', prioritizes MLX even if PyTorch MPS is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            device, name = gpu_utils.detect_device(backend="mlx")
            assert device == "mlx"
            assert "Apple Silicon (Apple M3 Pro)" in name

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    def test_detect_device_dual_stack_transformers_requested(
        self, mock_detect, mock_apple
    ):
        """When backend='transformers' on Mac, PyTorch MPS is preserved."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            device, name = gpu_utils.detect_device(backend="transformers")
            assert device == "mps"
            assert name == "Apple Silicon (MPS)"

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch(
        "soup_cli.utils.mlx.get_unified_memory_bytes", return_value=68719476736
    )  # 64 GB
    def test_get_gpu_info_apple_silicon_unified_memory(
        self, mock_mem, mock_detect, mock_apple
    ):
        """Unified memory calculation accurately formats memory string and byte counts."""
        info = gpu_utils.get_gpu_info(backend="mlx")
        assert "64.0 GB (unified)" in info["memory_total"]
        assert info["memory_total_bytes"] == 68719476736
        assert info["gpu_count"] == 1

    def test_get_gpu_info_mps_memory_zero_bytes_invariant(self):
        """PyTorch MPS returns memory_total_bytes=0 so hardware-fit preflight skips.

        This invariant is load-bearing: ``_hardware_fit_preflight`` (train.py)
        early-returns when ``total_bytes <= 0``, so every Mac MPS run skips the
        CUDA-shaped VRAM predictor.  If this test fails, previously-working Mac
        MPS runs may start being refused by the hardware-fit gate.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            info = gpu_utils.get_gpu_info(backend="transformers")
            assert info["memory_total"] == "shared (Apple Silicon)"
            assert info["memory_total_bytes"] == 0
            assert info["gpu_count"] == 1

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=False)
    def test_detect_device_cpu_fallback_on_non_apple(self, mock_apple):
        """Gracefully falls back to CPU on non-Apple machines without GPUs.

        Mocks torch so test passes deterministically on hosts with or without
        CUDA GPUs -- the maintainer caught that the original test asserted the
        *host* rather than the code.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            device, name = gpu_utils.detect_device()
            assert device == "cpu"
            assert "CPU" in name
            info = gpu_utils.get_gpu_info()
            assert info["gpu_count"] == 0
            assert info["memory_total_bytes"] == 0


class TestIssue423QuantizationDecision:
    """Pin quantization guard behavior via the real ``resolve_quantization()``.

    The maintainer's mutation audit disabled the CPU downgrade guard and the
    previous inline-simulation tests survived.  These tests call the extracted
    function directly, so removing or neutering ``resolve_quantization`` causes
    deterministic failures.
    """

    def test_cpu_downgrades_4bit_to_none_with_warning(self):
        """CPU + 4bit -> 'none' with a warning message."""
        resolved, warning = resolve_quantization(
            device="cpu", backend="transformers", quantization="4bit"
        )
        assert resolved == "none"
        assert warning is not None
        assert "not supported on CPU" in warning

    def test_cpu_downgrades_8bit_to_none_with_warning(self):
        """CPU + 8bit -> 'none' with a warning message."""
        resolved, warning = resolve_quantization(
            device="cpu", backend="transformers", quantization="8bit"
        )
        assert resolved == "none"
        assert warning is not None
        assert "not supported on CPU" in warning

    def test_cpu_none_quantization_passes_through(self):
        """CPU + none -> 'none' with no warning (nothing to downgrade)."""
        resolved, warning = resolve_quantization(
            device="cpu", backend="transformers", quantization="none"
        )
        assert resolved == "none"
        assert warning is None

    def test_mlx_preserves_4bit_without_downgrade(self):
        """MLX + 4bit -> '4bit' with no warning (pre-quantized mlx-community models).

        This is the core #423 fix: the explicit decision that MLX 4-bit is a
        different mechanism from bitsandbytes NF4 and must not be downgraded.
        """
        resolved, warning = resolve_quantization(
            device="mlx", backend="mlx", quantization="4bit"
        )
        assert resolved == "4bit"
        assert warning is None

    def test_mlx_4bit_preserved_even_on_cpu_device_label(self):
        """Even if device resolves oddly, backend='mlx' + 4bit is preserved.

        The guard checks backend first, so a hypothetical edge case where the
        device string doesn't match still preserves the MLX decision.
        """
        resolved, warning = resolve_quantization(
            device="cpu", backend="mlx", quantization="4bit"
        )
        assert resolved == "4bit"
        assert warning is None

    def test_cuda_4bit_passes_through(self):
        """CUDA + 4bit -> '4bit' with no warning (bitsandbytes handles it)."""
        resolved, warning = resolve_quantization(
            device="cuda", backend="transformers", quantization="4bit"
        )
        assert resolved == "4bit"
        assert warning is None

    def test_cuda_8bit_passes_through(self):
        """CUDA + 8bit -> '8bit' with no warning (bitsandbytes handles it)."""
        resolved, warning = resolve_quantization(
            device="cuda", backend="transformers", quantization="8bit"
        )
        assert resolved == "8bit"
        assert warning is None


class TestHardwareFitGateIsMlxAware:
    """The CUDA-shaped analytical VRAM predictor must skip on ``backend='mlx'``.

    Mirrors ``TestHardwareFitGateIsStreamingAware`` in ``test_v07200.py``.
    Apple Silicon MLX uses unified system RAM managed by Metal, not fixed CUDA
    VRAM, so the resident-memory prediction is the wrong model entirely.  On a
    non-Apple host, ``backend: mlx`` still skips the preflight harmlessly
    because ``resolve_trainer`` fails on the ``mlx_lm`` import before training
    starts — there is no silent hazard.
    """

    def _cfg(self, backend: str):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        body = {
            "base": "Qwen/Qwen2.5-3B",
            "task": "sft",
            "backend": backend,
            "modality": "text",
            "data": {"train": "train.jsonl", "max_length": 512},
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "quantization": "none",
                "lora": {"r": 8, "alpha": 16},
            },
        }
        return load_config_from_string(yaml.safe_dump(body))

    def test_resident_run_is_still_gated(self):
        """Control: without MLX the gate must still fire on a 4 GB card,
        otherwise this test proves nothing about the MLX branch."""
        import typer

        from soup_cli.commands.train import _hardware_fit_preflight

        gpu = {"memory_total_bytes": 4 * 10**9}
        with pytest.raises(typer.Exit):
            _hardware_fit_preflight(
                self._cfg("transformers"), gpu, allow_oom_attempt=False,
            )

    def test_mlx_run_is_not_blocked_by_the_resident_prediction(self):
        """``backend='mlx'`` with real unified memory bytes must not trip the
        CUDA VRAM gate — the predictor is skipped entirely."""
        from soup_cli.commands.train import _hardware_fit_preflight

        gpu = {"memory_total_bytes": 4 * 10**9}  # real bytes, not zero
        # Must not raise typer.Exit.
        _hardware_fit_preflight(
            self._cfg("mlx"), gpu, allow_oom_attempt=False,
        )
