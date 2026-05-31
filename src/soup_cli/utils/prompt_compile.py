"""``soup compile`` — DSPy / GEPA prompt-program compiler (v0.68.0 Part A).

Schema-only release: live wiring (DSPy / GEPA / TextGrad orchestrator) lands
in v0.68.1. The validators + frozen dataclasses ship now so operators can
build a plan today, then re-run with the live runner when it lands.

Public surface:

- ``SUPPORTED_PROMPT_OPTIMIZERS`` — closed frozenset
- ``MAX_COMPILE_ITERS`` — hard upper bound on iteration count
- ``validate_prompt_optimizer(name)`` — bool-first / null-byte / oversize / case-insensitive
- ``validate_max_iters(n)`` — bool-first / non-int / bounds
- ``validate_program_path(path)`` — cwd containment + symlink rejection + ``.py`` only
- ``validate_eval_suite_path(path)`` — cwd containment + symlink rejection
- ``CompilePlan`` frozen dataclass
- ``CompileResult`` frozen dataclass
- ``build_compile_plan(...)`` — factory
- ``run_compile(plan)`` — NotImplementedError stub w/ v0.68.1 marker

The CLI command lives in ``soup_cli/commands/compile_cmd.py`` (named
``compile_cmd`` to avoid shadowing the Python builtin ``compile()``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# ---------------------------------------------------------------------------
# Allowlists + bounds
# ---------------------------------------------------------------------------


SUPPORTED_PROMPT_OPTIMIZERS: frozenset = frozenset(
    {
        "bootstrap_fewshot",  # DSPy classic
        "mipro",  # DSPy Multi-stage Instruction Proposal Optimizer
        "copro",  # DSPy COordinated PROmpt optimizer
        "gepa",  # Reflective Prompt Evolution (gradient-free)
        "textgrad",  # Textual-gradient optimizer
    }
)

MAX_COMPILE_ITERS = 1000
_MIN_COMPILE_ITERS = 1
_MAX_OPTIMIZER_NAME_LEN = 32


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_prompt_optimizer(name: object) -> str:
    """Return the canonical lowercase optimizer name.

    Rejects bool / non-string / empty / null-byte / >32-char / unknown.
    Mirrors v0.41.0 ``validate_optimizer_name`` policy.
    """
    if isinstance(name, bool):
        raise TypeError("optimizer must not be bool")
    if not isinstance(name, str):
        raise TypeError("optimizer must be str")
    if not name:
        raise ValueError("optimizer must be non-empty")
    if "\x00" in name:
        raise ValueError("optimizer must not contain null bytes")
    if len(name) > _MAX_OPTIMIZER_NAME_LEN:
        raise ValueError(
            f"optimizer length {len(name)} > {_MAX_OPTIMIZER_NAME_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_PROMPT_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_PROMPT_OPTIMIZERS))
        )
    return canonical


def validate_max_iters(value: object) -> int:
    """Return ``value`` when an in-bounds positive int.

    Rejects bool / non-int / <=0 / > ``MAX_COMPILE_ITERS``.
    """
    if isinstance(value, bool):
        raise TypeError("max_iters must not be bool")
    if not isinstance(value, int):
        raise TypeError("max_iters must be int")
    if value < _MIN_COMPILE_ITERS:
        raise ValueError(f"max_iters must be >= {_MIN_COMPILE_ITERS}")
    if value > MAX_COMPILE_ITERS:
        raise ValueError(f"max_iters {value} > {MAX_COMPILE_ITERS}")
    return value


def validate_program_path(path: object) -> str:
    """Validate a prompt-program path (cwd-contained, ``.py`` only, no symlink).

    Returns the realpath of the validated path.
    """
    if isinstance(path, bool):
        raise TypeError("program_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("program_path must be str")
    if not path.endswith(".py"):
        raise ValueError("program_path must end in .py")
    enforce_under_cwd_and_no_symlink(path, field="program_path")
    return os.path.realpath(path)


def validate_eval_suite_path(path: object) -> str:
    """Validate an eval-suite path (cwd-contained, no symlink)."""
    if isinstance(path, bool):
        raise TypeError("eval_suite_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("eval_suite_path must be str")
    enforce_under_cwd_and_no_symlink(path, field="eval_suite_path")
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
    # Containment + symlink rejection enforced at write time by
    # ``atomic_write_text``; we only shape-check here so a plan can be
    # rendered before the output file is materialised.
    return path


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompilePlan:
    """A resolved compilation plan, validated at construction time."""

    program_path: str
    eval_suite_path: str
    optimizer: str
    max_iters: int
    output_path: str

    def __post_init__(self) -> None:
        # Re-validate so callers bypassing ``build_compile_plan`` cannot
        # smuggle in inconsistent fields (mirrors v0.61.0 EditPlan policy).
        validate_program_path(self.program_path)
        validate_eval_suite_path(self.eval_suite_path)
        # ``optimizer`` should already be lowercase canonical from the
        # factory; re-running the validator catches direct-construction
        # bugs that bypass ``build_compile_plan``.
        object.__setattr__(
            self, "optimizer", validate_prompt_optimizer(self.optimizer)
        )
        validate_max_iters(self.max_iters)
        _validate_output_path(self.output_path)


@dataclass(frozen=True)
class CompileResult:
    """The frozen result of running a compilation."""

    program_text: str
    score: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if isinstance(self.program_text, bool) or not isinstance(self.program_text, str):
            raise TypeError("program_text must be str")
        if isinstance(self.score, bool):
            raise TypeError("score must not be bool")
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if isinstance(self.iterations, bool):
            raise TypeError("iterations must not be bool")
        if not isinstance(self.iterations, int):
            raise TypeError("iterations must be int")
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_compile_plan(
    *,
    program_path: str,
    eval_suite_path: str,
    optimizer: str,
    max_iters: int,
    output_path: str,
) -> CompilePlan:
    """Validate every input and build a frozen ``CompilePlan``."""
    canonical_opt = validate_prompt_optimizer(optimizer)
    return CompilePlan(
        program_path=program_path,
        eval_suite_path=eval_suite_path,
        optimizer=canonical_opt,
        max_iters=max_iters,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Live runner stub (v0.68.1)
# ---------------------------------------------------------------------------


def run_compile(plan: CompilePlan) -> CompileResult:
    """Run the live compilation. Deferred to v0.68.1.

    Validates the plan type at the boundary so callers passing a bare dict
    get a clean ``TypeError`` rather than a confusing AttributeError when
    the v0.68.1 runner finally lands.
    """
    if not isinstance(plan, CompilePlan):
        raise TypeError("plan must be CompilePlan")
    raise NotImplementedError(
        "soup compile live runner is deferred to v0.68.1 — re-run after upgrading"
    )


__all__ = [
    "SUPPORTED_PROMPT_OPTIMIZERS",
    "MAX_COMPILE_ITERS",
    "validate_prompt_optimizer",
    "validate_max_iters",
    "validate_program_path",
    "validate_eval_suite_path",
    "CompilePlan",
    "CompileResult",
    "build_compile_plan",
    "run_compile",
]
