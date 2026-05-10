"""v0.43.0 Part A — Tracker integrations + PostHog telemetry opt-out.

Closed allowlist of HF Trainer `report_to` backends Soup recognises.
Adds mlflow / swanlab / trackio to the legacy `wandb` / `tensorboard` / `none`
set. Live integrations rely on HF Trainer's built-in callbacks (mlflow,
swanlab) plus the third-party `trackio` callback when installed; Soup only
validates the name and surfaces a friendly error when the backing package
is missing.

Telemetry: opt-out via `SOUP_TELEMETRY=0` env var. Default is OFF until a
public privacy policy ships — `is_telemetry_enabled` returns False unless
the user explicitly enables it. Hardware-info-only payload schema lives in
`build_telemetry_payload` for documentation/testing; no network calls in
v0.43.0 (PostHog wire-up deferred to v0.43.1).
"""
from __future__ import annotations

import math
import os
import platform
from types import MappingProxyType
from typing import Mapping

# Closed allowlist of report_to backends.
_REPORT_TO_BACKENDS: Mapping[str, str | None] = MappingProxyType({
    "none": None,
    "wandb": "wandb",
    "tensorboard": "tensorboard",
    "mlflow": "mlflow",
    "swanlab": "swanlab",
    "trackio": "trackio",
})

SUPPORTED_TRACKERS = frozenset(_REPORT_TO_BACKENDS.keys())

# v0.43.0 additions (HF-native wandb/tensorboard already supported).
NEW_TRACKERS_V0_43 = frozenset({"mlflow", "swanlab", "trackio"})

_MAX_NAME_LEN = 32


def validate_tracker_name(name: object) -> str:
    """Validate and lowercase a `report_to` tracker name.

    Returns the canonical lower-cased name. Raises ValueError on invalid
    input. Mirrors v0.41.0 `validate_optimizer_name` policy.
    """
    if not isinstance(name, str):
        raise ValueError(f"tracker name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("tracker name must not be empty")
    if "\x00" in name:
        raise ValueError("tracker name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"tracker name length {len(name)} exceeds max {_MAX_NAME_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_TRACKERS:
        supported = ", ".join(sorted(SUPPORTED_TRACKERS))
        raise ValueError(
            f"unknown tracker '{name}'. Supported: {supported}"
        )
    return canonical


def required_tracker_package(name: str) -> str | None:
    """Return the pip-installable package name for a tracker, or None.

    Non-string input returns None (mirrors `is_new_v0_43_tracker`).
    """
    if not isinstance(name, str):
        return None
    return _REPORT_TO_BACKENDS.get(name.lower())


def is_new_v0_43_tracker(name: object) -> bool:
    """True if the name is an additive v0.43.0 tracker, False otherwise."""
    if not isinstance(name, str):
        return False
    return name.lower() in NEW_TRACKERS_V0_43


# --- Telemetry (opt-out, default OFF) ----------------------------------

_TELEMETRY_ENV_VAR = "SOUP_TELEMETRY"


def is_telemetry_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Telemetry is opt-IN until v0.43.1 ships the network code.

    The roadmap entry calls this opt-out, but until the privacy policy
    + PostHog wire-up land we keep it default-OFF so no payload is built
    or sent. Users may enable explicitly with `SOUP_TELEMETRY=1`.
    """
    source = env if env is not None else os.environ
    raw = source.get(_TELEMETRY_ENV_VAR)
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    return False


def build_telemetry_payload(
    *,
    soup_version: str,
    command: str,
    duration_seconds: float | int | None = None,
) -> dict:
    """Build the hardware-info-only telemetry payload.

    The payload contains NO user data, dataset paths, model names, or
    config contents. Documented schema:

      - `soup_version`: caller-supplied
      - `command`: top-level CLI command (e.g. `train`, `data ingest`)
      - `python`: major.minor only
      - `os`: platform.system()
      - `arch`: platform.machine()
      - `duration_seconds`: optional, finite float / int / None

    Raises ValueError for non-string `command` / `soup_version` and for
    non-finite `duration_seconds`.
    """
    if not isinstance(soup_version, str) or not soup_version:
        raise ValueError("soup_version must be a non-empty string")
    if "\x00" in soup_version:
        raise ValueError("soup_version must not contain null bytes")
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")
    if "\x00" in command:
        raise ValueError("command must not contain null bytes")
    if duration_seconds is not None:
        # bool is a subclass of int — reject explicitly (project policy)
        if isinstance(duration_seconds, bool) or not isinstance(
            duration_seconds, (int, float)
        ):
            raise ValueError("duration_seconds must be int / float / None")
        if not math.isfinite(float(duration_seconds)):
            raise ValueError("duration_seconds must be finite")
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")

    py = platform.python_version_tuple()
    py_major_minor = f"{py[0]}.{py[1]}"
    return {
        "soup_version": soup_version,
        "command": command,
        "python": py_major_minor,
        "os": platform.system(),
        "arch": platform.machine(),
        "duration_seconds": (
            float(duration_seconds) if duration_seconds is not None else None
        ),
    }


def resolve_report_to(
    *,
    wandb: bool = False,
    tensorboard: bool = False,
    tracker: str | None = None,
) -> str:
    """Resolve the HF Trainer `report_to` value from CLI flags + --tracker.

    Mutual-exclusion: only one of (wandb, tensorboard, tracker) may be set.
    Empty string / None on `tracker` is treated as unset.
    """
    set_count = sum(
        1
        for x in (
            bool(wandb),
            bool(tensorboard),
            bool(tracker) if isinstance(tracker, str) and tracker else False,
        )
        if x
    )
    if set_count > 1:
        raise ValueError(
            "--wandb, --tensorboard, and --tracker are mutually exclusive"
        )
    if wandb:
        return "wandb"
    if tensorboard:
        return "tensorboard"
    if tracker:
        return validate_tracker_name(tracker)
    return "none"
