"""Crash bundle generator (v0.34.0 Part D).

When training fails with OOM / NaN / CUDA error, package a self-contained
``.crash`` JSON file that captures: the run config, last 50 metric rows, GPU
state at crash time, environment summary, and the error trace. The file is
ready to attach to a GitHub issue without leaking secrets.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import secrets
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from soup_cli.utils.paths import is_under_cwd

CRASH_FORMAT_VERSION = 1

# Last N metric rows captured into the bundle.
TAIL_METRICS = 50
# Hard cap on serialised bundle size — keeps a runaway DataLoader stacktrace
# or a giant config from producing a >5MB file.
MAX_BUNDLE_BYTES = 1_000_000

# Patterns that classify the failure mode. Order matters: more specific first.
_CRASH_KIND_PATTERNS = (
    ("oom", re.compile(
        r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation",
        re.IGNORECASE,
    )),
    ("nan", re.compile(
        r"\b(NaN|infinit[ey])\b.*loss|loss.*\b(NaN|infinit[ey])\b",
        re.IGNORECASE,
    )),
    ("cuda", re.compile(
        r"CUDA error|cublas|cudnn|device-side assert", re.IGNORECASE,
    )),
    ("dataloader", re.compile(r"DataLoader worker|num_workers", re.IGNORECASE)),
    ("nccl", re.compile(r"NCCL|c10d", re.IGNORECASE)),
)

# Environment vars worth capturing for repro. Excludes anything secret.
_SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_CUDNN_V8_API_ENABLED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_DEEPSPEED",
    "HF_ENDPOINT",
    "HF_HOME",
    "TRANSFORMERS_OFFLINE",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
)

# Token-shaped patterns we redact from any captured string.
_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"api[_-]?key[\"' :=]+[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
)


def classify_crash(message: str) -> str:
    """Return a short tag describing the failure mode, or 'other'."""
    if not isinstance(message, str):
        return "other"
    for tag, pattern in _CRASH_KIND_PATTERNS:
        if pattern.search(message):
            return tag
    return "other"


def redact_secrets(value: str) -> str:
    """Replace anything resembling a token with '<redacted>'."""
    if not isinstance(value, str):
        return value
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("<redacted>", value)
    return value


def _capture_gpu_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            state["available"] = True
            state["device_count"] = torch.cuda.device_count()
            state["devices"] = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                # mem_get_info may fail mid-OOM; guard each call separately.
                free_bytes: Optional[int] = None
                total_bytes: Optional[int] = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                except Exception:
                    pass
                state["devices"].append({
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                    "free_memory_bytes": free_bytes,
                    "queryable_total_bytes": total_bytes,
                    "allocated_bytes": _safe_call(torch.cuda.memory_allocated, index),
                    "reserved_bytes": _safe_call(torch.cuda.memory_reserved, index),
                })
    except Exception:
        # torch unavailable or CUDA broken — that's itself information.
        state["import_error"] = True
    return state


def _safe_call(func: Callable[..., Any], *args: Any) -> Optional[Any]:
    try:
        return func(*args)
    except Exception:
        return None


def _redact_value(value: Any) -> Any:
    """Recursively redact secret-shaped strings in nested dict/list/str values."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def _safe_output_dir(value: Optional[str]) -> Optional[str]:
    """Strip leading path components so a shared `.crash` doesn't leak $HOME."""
    if not value or not isinstance(value, str):
        return value
    return os.path.basename(value.rstrip("/\\")) or value


def _capture_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }
    safe_env: Dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        if key in os.environ:
            safe_env[key] = redact_secrets(os.environ[key])
    info["env"] = safe_env

    # Library versions
    libs: Dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if version is not None:
                libs[name] = str(version)
        except ImportError:
            continue
        except Exception:
            continue
    info["libs"] = libs
    return info


def build_crash_bundle(
    *,
    error: BaseException,
    config: Optional[dict] = None,
    metrics: Optional[List[dict]] = None,
    run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the bundle dict. Always returns; never raises."""
    error_message = redact_secrets(str(error))
    bundle: Dict[str, Any] = {
        "format_version": CRASH_FORMAT_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": _safe_output_dir(output_dir),
        "error": {
            "type": type(error).__name__,
            "message": error_message,
            "kind": classify_crash(error_message),
            "traceback": [
                redact_secrets(line)
                for line in traceback.format_exception(type(error), error, error.__traceback__)
            ],
        },
        "gpu_state": _capture_gpu_state(),
        "environment": _capture_env(),
    }
    if metrics:
        tail = list(metrics[-TAIL_METRICS:])
        bundle["metrics_tail"] = _redact_value(tail)
    if config is not None:
        # Re-serialise to drop non-JSON-able values cleanly, then redact
        # secret-shaped strings recursively so the .crash is shareable.
        try:
            text = json.dumps(config, default=str)
            bundle["config"] = _redact_value(json.loads(text))
        except Exception:
            bundle["config"] = {"_unserialisable": True}
    return bundle


def write_crash_bundle(
    bundle: Dict[str, Any],
    target_dir: Optional[Path] = None,
) -> Path:
    """Serialise the bundle as ``crash_<utc>.crash`` under target_dir.

    target_dir defaults to ``./.soup-crashes`` and must stay under cwd.
    Bundle is truncated to MAX_BUNDLE_BYTES if it would exceed the cap.
    """
    raw = Path(target_dir) if target_dir is not None else Path.cwd() / ".soup-crashes"
    # Cross-platform containment: project convention requires os.path.realpath
    # (Path.resolve() can disagree on Windows 8.3 short names).
    target = Path(os.path.realpath(str(raw)))
    if not is_under_cwd(target):
        raise ValueError(f"Crash directory {target} is not under cwd")
    target.mkdir(parents=True, exist_ok=True)

    # Compute filename BEFORE serialisation so a slow json.dumps can't clash
    # with a sibling crash on the same UTC second.
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    path = target / f"crash_{stamp}_{suffix}.crash"

    text = json.dumps(bundle, default=str, indent=2)
    if len(text.encode("utf-8")) > MAX_BUNDLE_BYTES:
        truncated = {
            "format_version": bundle.get("format_version"),
            "created_at": bundle.get("created_at"),
            "error": bundle.get("error"),
            "gpu_state": bundle.get("gpu_state"),
            "_truncated": True,
            "_note": "full bundle exceeded MAX_BUNDLE_BYTES; metrics + config dropped",
        }
        text = json.dumps(truncated, default=str, indent=2)

    path.write_text(text, encoding="utf-8")
    return path
