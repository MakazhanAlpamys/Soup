"""Crash bundle generator (v0.34.0 Part D).

When training fails with OOM / NaN / CUDA error, package a self-contained
``.crash`` JSON file that captures: the run config, last 50 metric rows, GPU
state at crash time, environment summary, and the error trace. Credentials are
redacted before anything is written (``utils/redaction.py``); that is pattern-based and
best-effort, so a bundle should still be read before it is attached to a public issue.
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
from soup_cli.utils.redaction import REDACTED, RedactionReport, redact_config, redact_text

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

# The token shapes redacted before ``utils/redaction.py`` existed. They stay as a last layer
# so nothing the bundle used to redact can come back, and because the RunPod controller
# (``cloud/_runpod_controller.py``) mirrors this tuple (tests/test_runpod_remote_diagnostics.py).
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


#: Longest text, in characters, that is passed through the redaction patterns. Redaction costs
#: up to about 4.3 s per MiB, and a crash handler must not stall on a huge message: a longer
#: string is cut (the tail is dropped, never kept) and ends in a marker.
MAX_REDACT_CHARS = 256 * 1024
#: How far back a cut may move to end at whitespace, so a credential is not kept in half.
_CUT_BACKOFF_CHARS = 256
#: Stored in place of a text that could not be redacted (never the raw text).
REDACTION_FAILED = "<redaction failed: text withheld>"


def _cap_text(value: str) -> str:
    """``value`` cut to ``MAX_REDACT_CHARS``, ending at whitespace when there is one near."""
    if len(value) <= MAX_REDACT_CHARS:
        return value
    cut = MAX_REDACT_CHARS
    floor = cut - _CUT_BACKOFF_CHARS
    while cut > floor and not value[cut - 1].isspace():
        cut -= 1
    return f"{value[:cut]}<{len(value) - cut} characters dropped>"


def redact_secrets(value: str) -> str:
    """Replace credentials in ``value`` with '<redacted>'.

    Covers URL userinfo and sensitive query parameters, the common token shapes,
    ``Authorization`` headers, JWTs, PEM private keys and ``key=value`` pairs whose key
    name says it is a credential (see ``utils/redaction.py``). A text longer than
    ``MAX_REDACT_CHARS`` is cut first; one that cannot be redacted is replaced by
    ``REDACTION_FAILED``. It never raises.
    """
    return _redact_string(value)


def _redact_string(value: str, report: Optional[RedactionReport] = None) -> str:
    """The shared redaction, then the legacy token shapes; counts go to ``report``."""
    if not isinstance(value, str) or not value:
        return value
    try:
        capped = _cap_text(value)
        if capped is not value and report is not None:
            report.count += 1  # a dropped tail is a change: the walk keeps a string only if none
        value = redact_text(capped, report)
        for pattern in _SECRET_PATTERNS:
            value, count = pattern.subn(REDACTED, value)
            if report is not None:
                report.count += count
    except Exception:
        # a fault in a pattern must neither stop the crash report nor let the raw text out
        if report is not None:
            report.count += 1
        return REDACTION_FAILED
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


def _safe_output_dir(value: Optional[str]) -> Optional[str]:
    """Strip leading path components so a shared `.crash` doesn't leak $HOME.

    The value is redacted first: the last component of ``s3://key:secret@bucket`` is
    ``key:secret@bucket``.
    """
    if not value or not isinstance(value, str):
        return value
    value = _redact_string(value)
    return os.path.basename(value.rstrip("/\\")) or value


def _capture_env(report: Optional[RedactionReport] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }
    safe_env: Dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        if key in os.environ:
            safe_env[key] = _redact_string(os.environ[key], report)
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
    """Build the bundle dict. Always returns; never raises.

    The error message, the traceback, the metrics and the config are redacted before
    they are stored; ``bundle["redaction"]`` says how many values were replaced
    (counts only, never the values). A string is cut at ``MAX_REDACT_CHARS`` first. A text
    that cannot be read or redacted is stored as ``REDACTION_FAILED``, a config or a
    metrics list that cannot be serialised or redacted as a marker dict: the raw text is
    never stored. If building fails in any other way, a minimal bundle is returned.
    """
    try:
        return _build_bundle(
            error=error, config=config, metrics=metrics, run_id=run_id, output_dir=output_dir,
        )
    except Exception:
        return _minimal_bundle(error, run_id)


def _minimal_bundle(error: BaseException, run_id: Optional[str]) -> Dict[str, Any]:
    return {
        "format_version": CRASH_FORMAT_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id if isinstance(run_id, str) else None,
        "error": {
            "type": type(error).__name__, "message": REDACTION_FAILED, "kind": "other",
            "traceback": [],
        },
        "redaction": {"values_redacted": 1},
    }


def _error_text(error: BaseException, report: RedactionReport) -> str:
    try:
        text = str(error)
    except Exception:
        report.count += 1
        return REDACTION_FAILED
    return _redact_string(text, report)


def _build_bundle(
    *,
    error: BaseException,
    config: Optional[dict],
    metrics: Optional[List[dict]],
    run_id: Optional[str],
    output_dir: Optional[str],
) -> Dict[str, Any]:
    message_report = RedactionReport()
    trace_report = RedactionReport()
    env_report = RedactionReport()
    error_message = _error_text(error, message_report)
    try:
        trace_lines = traceback.format_exception(type(error), error, error.__traceback__)
    except Exception:
        trace_lines = []
    bundle: Dict[str, Any] = {
        "format_version": CRASH_FORMAT_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": _safe_output_dir(output_dir),
        "error": {
            "type": type(error).__name__,
            "message": error_message,
            "kind": classify_crash(error_message),
            "traceback": [_redact_string(line, trace_report) for line in trace_lines],
        },
        "gpu_state": _capture_gpu_state(),
        "environment": _capture_env(env_report),
    }
    counts: Dict[str, int] = {
        "error_message": message_report.count,
        "traceback": trace_report.count,
        "environment": env_report.count,
    }
    if metrics:
        bundle["metrics_tail"], counts["metrics_tail"] = _redact_serialised(
            list(metrics[-TAIL_METRICS:]), [{"_unserialisable": True}],
            [{"_redaction_failed": True}],
        )
    if config is not None:
        bundle["config"], counts["config"] = _redact_serialised(
            config, {"_unserialisable": True}, {"_redaction_failed": True},
        )
    bundle["redaction"] = {"values_redacted": sum(counts.values()), **counts}
    return bundle


def _redact_serialised(value: Any, unserialisable: Any, failed: Any) -> tuple:
    """``value`` re-serialised to plain JSON types, then redacted; plus the count.

    The round trip comes first: bytes, sets and objects become text (as they would when
    the bundle is written), so what is redacted is what is stored. A value that cannot be
    serialised gives ``unserialisable``, one whose redaction fails gives ``failed``.
    """
    try:
        plain = json.loads(json.dumps(value, default=str))
    except Exception:
        return unserialisable, 0
    try:
        redacted, report = redact_config(plain, _redact_string)
    except Exception:
        return failed, 1
    return redacted, report.count


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
            "redaction": bundle.get("redaction"),
            "_truncated": True,
            "_note": "full bundle exceeded MAX_BUNDLE_BYTES; metrics + config dropped",
        }
        text = json.dumps(truncated, default=str, indent=2)

    path.write_text(text, encoding="utf-8")
    return path
