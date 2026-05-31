"""``soup distill-prompt`` — distill prompt-heavy traces into a small FT plan (v0.68.0 Part B).

Bridge between prompt-engineering and FT worlds: take a JSONL of
large-prompt teacher calls (GPT-5 / Claude / etc.) and prepare a
distillation dataset targeting a small student model. Schema-only release:
live dataset preparation lands in v0.68.1 (composes with v0.70 Part B
cross-tokenizer KD when that ships).

Public surface:

- ``SUPPORTED_DISTILL_STRATEGIES`` — closed frozenset {sft, preference, kl}
- ``validate_distill_strategy(name)`` — bool-first / null-byte / case-insensitive
- ``validate_teacher_id`` / ``validate_student_id`` — null-byte / oversize / bool
- ``validate_traces_path(path)`` — cwd containment + symlink rejection
- ``DistillPromptPlan`` frozen dataclass
- ``build_distill_prompt_plan(...)`` factory
- ``prepare_distill_dataset(plan)`` — NotImplementedError stub w/ v0.68.1 marker
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

SUPPORTED_DISTILL_STRATEGIES: frozenset = frozenset({"sft", "preference", "kl"})

_MAX_STRATEGY_LEN = 32
_MAX_MODEL_ID_LEN = 512


def validate_distill_strategy(name: object) -> str:
    """Return canonical lowercase strategy name."""
    if isinstance(name, bool):
        raise TypeError("strategy must not be bool")
    if not isinstance(name, str):
        raise TypeError("strategy must be str")
    if not name:
        raise ValueError("strategy must be non-empty")
    if "\x00" in name:
        raise ValueError("strategy must not contain null bytes")
    if len(name) > _MAX_STRATEGY_LEN:
        raise ValueError(
            f"strategy length {len(name)} > {_MAX_STRATEGY_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_DISTILL_STRATEGIES:
        raise ValueError(
            f"unknown strategy {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_DISTILL_STRATEGIES))
        )
    return canonical


def _validate_model_id(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_MODEL_ID_LEN:
        raise ValueError(f"{field} length {len(value)} > {_MAX_MODEL_ID_LEN}")
    return value


def validate_teacher_id(value: object) -> str:
    """Validate a teacher model id (HF repo id or local path-shape)."""
    return _validate_model_id(value, field="teacher")


def validate_student_id(value: object) -> str:
    """Validate a student model id."""
    return _validate_model_id(value, field="student")


def validate_traces_path(path: object) -> str:
    """Validate a traces JSONL path (cwd-contained, no symlink)."""
    if isinstance(path, bool):
        raise TypeError("traces_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("traces_path must be str")
    enforce_under_cwd_and_no_symlink(path, field="traces_path")
    return os.path.realpath(path)


def _validate_output_path(path: object) -> str:
    if isinstance(path, bool):
        raise TypeError("output_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("output_path must be str")
    if not path:
        raise ValueError("output_path must be non-empty")
    if "\x00" in path:
        raise ValueError("output_path must not contain null bytes")
    return path


@dataclass(frozen=True)
class DistillPromptPlan:
    """A resolved distill-prompt plan."""

    traces_path: str
    teacher: str
    student: str
    strategy: str
    output_path: str

    def __post_init__(self) -> None:
        validate_traces_path(self.traces_path)
        validate_teacher_id(self.teacher)
        validate_student_id(self.student)
        object.__setattr__(
            self, "strategy", validate_distill_strategy(self.strategy)
        )
        _validate_output_path(self.output_path)


def build_distill_prompt_plan(
    *,
    traces_path: str,
    teacher: str,
    student: str,
    strategy: str,
    output_path: str,
) -> DistillPromptPlan:
    """Validate inputs and return a frozen ``DistillPromptPlan``."""
    return DistillPromptPlan(
        traces_path=traces_path,
        teacher=teacher,
        student=student,
        strategy=validate_distill_strategy(strategy),
        output_path=output_path,
    )


def prepare_distill_dataset(plan: DistillPromptPlan) -> None:
    """Live dataset preparation. Deferred to v0.68.1.

    Validates plan type at the boundary so a bare dict raises cleanly.
    """
    if not isinstance(plan, DistillPromptPlan):
        raise TypeError("plan must be DistillPromptPlan")
    raise NotImplementedError(
        "distill-prompt live dataset preparation is deferred to v0.68.1"
    )


__all__ = [
    "SUPPORTED_DISTILL_STRATEGIES",
    "validate_distill_strategy",
    "validate_teacher_id",
    "validate_student_id",
    "validate_traces_path",
    "DistillPromptPlan",
    "build_distill_prompt_plan",
    "prepare_distill_dataset",
]
