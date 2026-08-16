"""
Comprehensive test suite for Issue #423 & Apple Silicon hardware detection.
Covers MLX detection, MPS/MLX precedence disambiguation, and unified memory telemetry.
"""

from unittest.mock import patch, MagicMock
import sys
from soup_cli.utils import gpu as gpu_utils


class TestMLXDeviceDetection:
    """Test suite for detect_device() and get_gpu_info()."""

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M2 Max"})
    def test_detect_device_pure_apple_silicon_mlx(self, mock_chip, mock_detect, mock_apple):
        """Pure Apple Silicon with MLX returns 'mlx' device."""
        device, name = gpu_utils.detect_device(backend="mlx")
        assert device == "mlx"
        assert name == "Apple Silicon (Apple M2 Max)"

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M3 Pro"})
    def test_detect_device_dual_stack_mlx_requested(self, mock_chip, mock_detect, mock_apple):
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
    def test_detect_device_dual_stack_transformers_requested(self, mock_detect, mock_apple):
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
    @patch("soup_cli.utils.mlx.get_unified_memory_bytes", return_value=68719476736)  # 64 GB
    def test_get_gpu_info_apple_silicon_unified_memory(self, mock_mem, mock_detect, mock_apple):
        """Unified memory calculation accurately formats memory string and byte counts."""
        info = gpu_utils.get_gpu_info(backend="mlx")
        assert "64.0 GB (unified)" in info["memory_total"]
        assert info["memory_total_bytes"] == 68719476736
        assert info["gpu_count"] == 1

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=False)
    def test_detect_device_cpu_fallback_on_non_apple(self, mock_apple):
        """Gracefully falls back to CPU on non-Apple machines without GPUs."""
        device, name = gpu_utils.detect_device()
        assert device == "cpu"
        assert "CPU" in name
        info = gpu_utils.get_gpu_info()
        assert info["gpu_count"] == 0
        assert info["memory_total_bytes"] == 0
