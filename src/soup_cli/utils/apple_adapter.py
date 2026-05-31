"""``soup apple-adapter`` — HF / PEFT ↔ MLX ↔ Apple FoundationModels (v0.68.0 Part D).

Schema-only release: live conversion (HF safetensors -> MLX npz / Apple
FoundationModels adapter blob) lands in v0.68.1. Reuses v0.60 Part B
signing for the optional ``--sign`` path.

Public surface:

- ``SUPPORTED_ADAPTER_DIRECTIONS`` — closed frozenset
- ``validate_direction(name)`` — bool-first / null-byte / case-insensitive
- ``validate_source_adapter(path)`` — cwd containment + directory check + symlink reject
- ``AppleAdapterPlan`` frozen dataclass
- ``build_apple_adapter_plan(...)`` factory
- ``convert_apple_adapter(plan)`` — NotImplementedError stub w/ v0.68.1 marker
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass

from soup_cli.utils.paths import is_under_cwd

SUPPORTED_ADAPTER_DIRECTIONS: frozenset = frozenset(
    {"hf-to-mlx", "mlx-to-hf", "hf-to-apple", "mlx-to-apple"}
)

_MAX_DIRECTION_LEN = 32


def validate_direction(name: object) -> str:
    if isinstance(name, bool):
        raise TypeError("direction must not be bool")
    if not isinstance(name, str):
        raise TypeError("direction must be str")
    if not name:
        raise ValueError("direction must be non-empty")
    if "\x00" in name:
        raise ValueError("direction must not contain null bytes")
    if len(name) > _MAX_DIRECTION_LEN:
        raise ValueError(
            f"direction length {len(name)} > {_MAX_DIRECTION_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_ADAPTER_DIRECTIONS:
        raise ValueError(
            f"unknown direction {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_ADAPTER_DIRECTIONS))
        )
    return canonical


def validate_source_adapter(path: object) -> str:
    """Validate a source-adapter directory (cwd-contained, no symlink)."""
    if isinstance(path, bool):
        raise TypeError("source_dir must not be bool")
    if not isinstance(path, str):
        raise TypeError("source_dir must be str")
    if not path:
        raise ValueError("source_dir must be non-empty")
    if "\x00" in path:
        raise ValueError("source_dir must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(
            f"source_dir {os.path.basename(path)!r} must stay under cwd"
        )
    if os.path.lexists(path):
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise ValueError(
                f"source_dir unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                "source_dir must not be a symlink (TOCTOU defence)"
            )
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError("source_dir must be a directory")
    return os.path.realpath(path)


def _validate_output_dir(path: object) -> str:
    if isinstance(path, bool):
        raise TypeError("output_dir must not be bool")
    if not isinstance(path, str):
        raise TypeError("output_dir must be str")
    if not path:
        raise ValueError("output_dir must be non-empty")
    if "\x00" in path:
        raise ValueError("output_dir must not contain null bytes")
    return path


@dataclass(frozen=True)
class AppleAdapterPlan:
    source_dir: str
    output_dir: str
    direction: str
    sign: bool

    def __post_init__(self) -> None:
        validate_source_adapter(self.source_dir)
        _validate_output_dir(self.output_dir)
        object.__setattr__(
            self, "direction", validate_direction(self.direction)
        )
        if not isinstance(self.sign, bool):
            raise TypeError("sign must be bool")


def build_apple_adapter_plan(
    *,
    source_dir: str,
    output_dir: str,
    direction: str,
    sign: bool = False,
) -> AppleAdapterPlan:
    return AppleAdapterPlan(
        source_dir=source_dir,
        output_dir=output_dir,
        direction=validate_direction(direction),
        sign=sign,
    )


def convert_apple_adapter(plan: AppleAdapterPlan) -> None:
    """Live conversion. Deferred to v0.68.1."""
    if not isinstance(plan, AppleAdapterPlan):
        raise TypeError("plan must be AppleAdapterPlan")
    raise NotImplementedError(
        "apple-adapter live conversion is deferred to v0.68.1"
    )


__all__ = [
    "SUPPORTED_ADAPTER_DIRECTIONS",
    "validate_direction",
    "validate_source_adapter",
    "AppleAdapterPlan",
    "build_apple_adapter_plan",
    "convert_apple_adapter",
]
