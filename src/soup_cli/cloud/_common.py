"""Shared, dependency-light primitives for cloud training backends."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_MAX_NAME_LEN = 32
_MAX_PATH_LEN = 4096
_MAX_VERSION_LEN = 64
_MAX_CONFIG_BYTES = 1_000_000
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")


@dataclass(frozen=True)
class CloudPlan:
    """A rendered cloud-training plan (plan-only by default)."""

    cloud: str
    gpu: str
    output_dir: str
    stub_path: str
    stub_text: str
    run_command: str


def validate_choice(value: object, field: str, allowed: Collection[str]) -> str:
    """Validate and normalize a short string against a closed allowlist."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a string, got bool")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_NAME_LEN:
        raise ValueError(f"{field} exceeds {_MAX_NAME_LEN} chars")
    normalised = value.lower()
    if normalised not in allowed:
        raise ValueError(f"{field}={value!r} is not supported. Valid: {sorted(allowed)}")
    return normalised


def validate_path_shape(value: object, field: str) -> str:
    """Reject empty, overlong, or control-character-containing paths."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must not contain NUL / newline")
    if len(value) > _MAX_PATH_LEN:
        raise ValueError(f"{field} exceeds {_MAX_PATH_LEN} chars")
    return value


def write_cloud_stub(plan: CloudPlan) -> str:
    """Write a rendered controller atomically under the current directory."""
    from soup_cli.utils.paths import atomic_write_text

    return atomic_write_text(plan.stub_text, plan.stub_path, field="stub_path")
