"""SR 11-7-style reproducibility receipt (v0.59.0 Part E).

Captures the minimum environment fingerprint needed for a regulated-org
audit: seeds (torch + numpy + python), Python interpreter version, OS +
arch, soup_cli version, kernel versions (CUDA / cuDNN / NCCL — best-effort
from torch when available), GPU model + driver (best-effort from
``torch.cuda.get_device_name`` + ``nvidia-smi`` proxy).

Pure-stdlib at module top; ``torch`` is lazy-imported so this module
loads in <50 ms on CPU-only hosts.
"""

from __future__ import annotations

import json
import logging
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional, Tuple

from soup_cli.utils.paths import atomic_write_text

_LOG = logging.getLogger(__name__)

_MAX_RUN_ID = 128
_MAX_VERSION = 64
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


@dataclass(frozen=True)
class ReproReceipt:
    """SR 11-7-style reproducibility receipt."""

    run_id: str
    soup_version: str
    python_version: str
    os: str
    arch: str
    seeds: Mapping[str, int]
    torch_version: Optional[str]
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    nccl_version: Optional[str]
    gpu_models: Tuple[str, ...]
    driver_version: Optional[str]
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.match(self.run_id):
            raise ValueError(
                f"run_id must match [A-Za-z0-9._-]+ (1-128 chars), got {self.run_id!r}"
            )
        for value, name in (
            (self.soup_version, "soup_version"),
            (self.python_version, "python_version"),
            (self.os, "os"),
            (self.arch, "arch"),
            (self.created_at, "created_at"),
        ):
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"{name} must be a null-byte-free str")
        if not isinstance(self.seeds, Mapping):
            raise ValueError("seeds must be a mapping")
        for key, val in self.seeds.items():
            if not isinstance(key, str):
                raise ValueError("seeds keys must be str")
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"seeds[{key}] must be int (got {type(val).__name__})")
        if not isinstance(self.gpu_models, tuple):
            raise ValueError("gpu_models must be a tuple")


def _detect_torch_kernel_versions() -> dict:
    """Best-effort torch / CUDA / cuDNN / NCCL detection.

    Returns a dict with optional ``torch_version`` / ``cuda_version`` /
    ``cudnn_version`` / ``nccl_version`` / ``gpu_models`` / ``driver_version``
    keys. Each is ``None`` when not detected. Lazy-imports torch.
    """
    out = {
        "torch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "nccl_version": None,
        "gpu_models": (),
        "driver_version": None,
    }
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return out
    try:
        out["torch_version"] = str(torch.__version__)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("repro_receipt torch probe failed: %s", exc)
    try:
        if torch.version.cuda is not None:
            out["cuda_version"] = str(torch.version.cuda)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("repro_receipt torch probe failed: %s", exc)
    try:
        if torch.backends.cudnn.is_available():
            out["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("repro_receipt torch probe failed: %s", exc)
    try:
        nccl_v = getattr(torch.cuda, "nccl", None)
        if nccl_v is not None and hasattr(nccl_v, "version"):
            v = nccl_v.version()
            out["nccl_version"] = ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("repro_receipt torch probe failed: %s", exc)
    try:
        if torch.cuda.is_available():
            names = []
            for i in range(torch.cuda.device_count()):
                names.append(str(torch.cuda.get_device_name(i)))
            out["gpu_models"] = tuple(names)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("repro_receipt torch probe failed: %s", exc)
    return out


def build_repro_receipt(
    seeds: Mapping[str, int],
    run_id: str,
    *,
    created_at: Optional[str] = None,
) -> ReproReceipt:
    """Build a reproducibility receipt from seeds + run_id + autodetected env.

    ``ReproReceipt.__post_init__`` performs the canonical seeds validation;
    we no longer duplicate it here (python-review HIGH fix).
    """
    if created_at is None:
        created_at = datetime.now(tz=timezone.utc).isoformat()

    from soup_cli import __version__ as _soup_version
    kernel = _detect_torch_kernel_versions()
    return ReproReceipt(
        run_id=run_id,
        soup_version=str(_soup_version)[:_MAX_VERSION],
        python_version=platform.python_version(),
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine() or "unknown",
        seeds=dict(seeds),
        torch_version=kernel.get("torch_version"),
        cuda_version=kernel.get("cuda_version"),
        cudnn_version=kernel.get("cudnn_version"),
        nccl_version=kernel.get("nccl_version"),
        gpu_models=tuple(kernel.get("gpu_models", ())),
        driver_version=kernel.get("driver_version"),
        created_at=created_at,
    )


def receipt_to_dict(r: ReproReceipt) -> dict:
    return {
        "run_id": r.run_id,
        "soup_version": r.soup_version,
        "python_version": r.python_version,
        "os": r.os,
        "arch": r.arch,
        "seeds": dict(r.seeds),
        "torch_version": r.torch_version,
        "cuda_version": r.cuda_version,
        "cudnn_version": r.cudnn_version,
        "nccl_version": r.nccl_version,
        "gpu_models": list(r.gpu_models),
        "driver_version": r.driver_version,
        "created_at": r.created_at,
    }


def write_repro_receipt(r: ReproReceipt, output_path: str) -> str:
    """Atomic write of the receipt to ``output_path`` (cwd-contained)."""
    text = json.dumps(receipt_to_dict(r), indent=2, sort_keys=True)
    return atomic_write_text(
        text, output_path, prefix=".repro.", suffix=".json.tmp",
    )
