"""``soup compile-tools`` — TextGrad / GEPA tool-schema optimizer (v0.68.0 Part C).

Generate tool schemas + descriptions optimized via textual gradients.
Schema-only release: live optimizer pass lands in v0.68.1.

Composes with v0.46 Agent Forge (OpenAPI / MCP / GraphQL parser) — Agent
Forge produces the spec, ``compile-tools`` optimises the descriptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

SUPPORTED_TOOL_OPTIMIZERS: frozenset = frozenset({"textgrad", "gepa"})
_SUPPORTED_SPEC_EXTENSIONS: frozenset = frozenset({".json", ".yaml", ".yml"})

_MAX_OPTIMIZER_NAME_LEN = 32


def validate_tool_optimizer(name: object) -> str:
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
    if canonical not in SUPPORTED_TOOL_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_TOOL_OPTIMIZERS))
        )
    return canonical


def validate_spec_path(path: object) -> str:
    """Validate an OpenAPI / MCP / GraphQL spec path (json / yaml / yml only)."""
    if isinstance(path, bool):
        raise TypeError("spec_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("spec_path must be str")
    lower = path.lower()
    if not any(lower.endswith(ext) for ext in _SUPPORTED_SPEC_EXTENSIONS):
        raise ValueError(
            "spec_path extension must be .json / .yaml / .yml"
        )
    enforce_under_cwd_and_no_symlink(path, field="spec_path")
    return os.path.realpath(path)


def validate_eval_suite_path(path: object) -> str:
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
    return path


@dataclass(frozen=True)
class ToolCompilePlan:
    spec_path: str
    eval_suite_path: str
    optimizer: str
    output_path: str

    def __post_init__(self) -> None:
        validate_spec_path(self.spec_path)
        validate_eval_suite_path(self.eval_suite_path)
        object.__setattr__(
            self, "optimizer", validate_tool_optimizer(self.optimizer)
        )
        _validate_output_path(self.output_path)


def build_tool_compile_plan(
    *,
    spec_path: str,
    eval_suite_path: str,
    optimizer: str,
    output_path: str,
) -> ToolCompilePlan:
    return ToolCompilePlan(
        spec_path=spec_path,
        eval_suite_path=eval_suite_path,
        optimizer=validate_tool_optimizer(optimizer),
        output_path=output_path,
    )


def run_tool_compile(plan: ToolCompilePlan) -> None:
    """Live optimiser pass. Deferred to v0.68.1."""
    if not isinstance(plan, ToolCompilePlan):
        raise TypeError("plan must be ToolCompilePlan")
    raise NotImplementedError(
        "compile-tools live optimisation is deferred to v0.68.1"
    )


__all__ = [
    "SUPPORTED_TOOL_OPTIMIZERS",
    "validate_tool_optimizer",
    "validate_spec_path",
    "validate_eval_suite_path",
    "ToolCompilePlan",
    "build_tool_compile_plan",
    "run_tool_compile",
]
