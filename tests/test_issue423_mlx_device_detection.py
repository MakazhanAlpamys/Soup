"""
Test suite for Issue #423: detect_device() and get_gpu_info() MLX Apple Silicon support.
"""

from unittest.mock import patch
from soup_cli.utils import gpu as gpu_utils


class TestMLXDeviceDetection:
    """Verify that detect_device() and get_gpu_info() accurately recognize Apple Silicon MLX."""

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_chip_info", return_value={"chip": "Apple M1 Max"})
    def test_detect_device_apple_silicon_mlx_live(self, mock_chip, mock_detect, mock_apple):
        device, name = gpu_utils.detect_device()
        assert device == "mlx"
        assert name == "Apple Silicon (Apple M1 Max)"

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=True)
    @patch("soup_cli.utils.mlx.detect_mlx", return_value=True)
    @patch("soup_cli.utils.mlx.get_unified_memory_bytes", return_value=34359738368)  # 32 GB
    def test_get_gpu_info_apple_silicon_unified_memory(self, mock_mem, mock_detect, mock_apple):
        info = gpu_utils.get_gpu_info()
        assert "32.0 GB (unified)" in info["memory_total"]
        assert info["gpu_count"] == 1
        assert info["memory_total_bytes"] == 34359738368

    @patch("soup_cli.utils.mlx.is_apple_silicon", return_value=False)
    def test_detect_device_cpu_fallback(self, mock_apple):
        device, name = gpu_utils.detect_device()
        assert device == "cpu"
        assert "CPU" in name
