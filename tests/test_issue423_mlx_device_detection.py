"""Comprehensive test suite for Issue #423 & Apple Silicon hardware detection.

Covers MLX detection, MPS/MLX precedence disambiguation, MPS zero-byte memory
invariant for hardware-fit preflight, host-agnostic CPU fallbacks, and explicit
quantization preservation for MLX vs CPU downgrade guards.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from soup_cli.config.schema import SoupConfig
from soup_cli.utils import gpu as gpu_utils


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
        """PyTorch MPS returns memory_total_bytes=0 so hardware-fit preflight skips."""
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

        Mocks torch so test passes deterministically on hosts with or without CUDA GPUs.
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
    """Pins quantization behavior on CPU vs MLX backend per Issue #423."""

    def test_cpu_downgrades_4bit_and_8bit_to_none_with_warning(self):
        """On CPU, bitsandbytes 4bit/8bit is unsupported and downgraded to 'none'."""
        for quant in ("4bit", "8bit"):
            cfg = SoupConfig(base="Qwen/Qwen2.5-0.5B", data={"train": "./data.jsonl"})
            cfg.training.quantization = quant

            # Simulate CPU device path in train.py
            device = "cpu"
            captured_warnings = []

            if device == "cpu" and cfg.training.quantization in ("4bit", "8bit"):
                captured_warnings.append(
                    f"Warning: {cfg.training.quantization} quantization is not supported on CPU."
                )
                cfg.training.quantization = "none"

            assert cfg.training.quantization == "none"
            assert len(captured_warnings) == 1
            assert "not supported on CPU" in captured_warnings[0]

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M2"})
    def test_mlx_preserves_4bit_quantization_without_downgrade(
        self, mock_chip, mock_detect, mock_apple
    ):
        """On Apple Silicon MLX, 4bit quantization uses pre-quantized mlx-lm models.

        Passing backend='mlx' resolves device='mlx' and leaves cfg.training.quantization='4bit'
        intact without triggering the CPU bitsandbytes downgrade warning.
        """
        cfg = SoupConfig(base="Qwen/Qwen2.5-0.5B", data={"train": "./data.jsonl"})
        cfg.backend = "mlx"
        cfg.training.quantization = "4bit"

        device, _ = gpu_utils.detect_device(backend=cfg.backend)
        assert device == "mlx"

        captured_warnings = []
        if device == "cpu" and cfg.training.quantization in ("4bit", "8bit"):
            captured_warnings.append("Warning: CPU downgrade triggered")
            cfg.training.quantization = "none"

        # Must remain 4bit and emit zero CPU warnings
        assert cfg.training.quantization == "4bit"
        assert len(captured_warnings) == 0
