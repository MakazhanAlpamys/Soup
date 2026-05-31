"""MLX backend utilities — detection + hardware info + batch size estimation.

MLX is Apple's ML framework for Apple Silicon (M1-M4 chips). This module
provides feature detection and helpers so the rest of Soup can opportunistically
enable MLX training paths without hard-depending on the ``mlx`` package.
"""

from __future__ import annotations

import platform
from typing import Any, Optional


def detect_mlx() -> bool:
    """Return True if the ``mlx`` package is importable on this machine.

    This does **not** check for Apple Silicon hardware — use ``get_mlx_info``
    for a full detection report.
    """
    try:
        import mlx  # noqa: F401
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def is_apple_silicon() -> bool:
    """Return True if the current machine is an Apple Silicon Mac."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def get_mlx_version() -> Optional[str]:
    """Return the MLX version string, or None if not installed."""
    try:
        import mlx
    except ImportError:
        return None
    return getattr(mlx, "__version__", "unknown")


def get_chip_info() -> dict[str, str]:
    """Best-effort chip detection for Apple Silicon Macs."""
    info: dict[str, str] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    if not is_apple_silicon():
        return info

    try:
        import subprocess  # noqa: S404 — used with list args only
        result = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return info


def get_unified_memory_bytes() -> Optional[int]:
    """Return total unified memory in bytes, or None if unavailable."""
    if not is_apple_silicon():
        return None
    try:
        import subprocess  # noqa: S404
        result = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def get_mlx_info() -> dict[str, Any]:
    """Return a full MLX detection report suitable for ``soup doctor``."""
    available = detect_mlx()
    info: dict[str, Any] = {
        "available": available,
        "version": get_mlx_version(),
        "apple_silicon": is_apple_silicon(),
        "chip": get_chip_info(),
        "unified_memory_bytes": get_unified_memory_bytes(),
    }
    return info


def estimate_mlx_batch_size(
    model_params_b: float,
    unified_memory_bytes: int,
    max_length: int,
    quantization: str = "4bit",
) -> int:
    """Rough batch-size estimator for MLX training.

    Uses the same heuristic as ``utils.gpu.estimate_batch_size`` but scaled
    for Apple Silicon unified memory. Returns at least 1.
    """
    bytes_per_param = {"4bit": 0.5, "8bit": 1.0, "none": 2.0}.get(quantization, 2.0)
    model_bytes = model_params_b * 1e9 * bytes_per_param
    activation_budget = max(0, unified_memory_bytes - model_bytes * 1.6)
    # Rough: 2 bytes per token × seq × hidden * 4 (q/k/v/o) + grads
    tokens_cost = max_length * 2.0 * 4096 * 4 * 1.5
    if tokens_cost <= 0:
        return 1
    batch = int(activation_budget / tokens_cost)
    return max(1, min(batch, 32))


def load_mlx_model(
    model_path: str, quantization: str = "4bit",
) -> tuple[Any, Any]:
    """Thin wrapper around ``mlx_lm.load`` (lazy import).

    ``quantization`` is informational: MLX models are typically already
    quantized at build time (e.g. ``mlx-community/...-4bit``), so this
    parameter is currently advisory and is not forwarded to ``mlx_lm.load``.
    """
    del quantization  # advisory only — MLX models are pre-quantized
    from mlx_lm import load

    return load(model_path)
