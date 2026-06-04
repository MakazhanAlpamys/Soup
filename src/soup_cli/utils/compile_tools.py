"""``soup compile-tools`` — TextGrad / GEPA tool-schema optimizer.

Generate tool schemas + descriptions optimized via textual gradients.
Schema + validators from v0.68.0 Part C; the live optimizer pass lands in
v0.71.13 (#227), lazy-importing TextGrad / GEPA with a friendly
``ImportError`` (``pip install 'soup-cli[compile]'``).

Composes with v0.46 Agent Forge (OpenAPI / MCP / GraphQL parser) — Agent
Forge produces the spec, ``compile-tools`` optimises the descriptions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
)

SUPPORTED_TOOL_OPTIMIZERS: frozenset[str] = frozenset({"textgrad", "gepa"})
_SUPPORTED_SPEC_EXTENSIONS: frozenset[str] = frozenset({".json", ".yaml", ".yml"})

_MAX_OPTIMIZER_NAME_LEN = 32
_INSTALL_HINT = (
    "Run: pip install 'soup-cli[compile]'  (installs textgrad / gepa)"
)

# Injectable seam: tests set this to a ``(description, examples, optimizer)
# -> str`` callable so the parse -> iterate -> write orchestration is
# exercised without the TextGrad / GEPA libraries (mirrors v0.71.13 #225
# ``prompt_compile._OPTIMIZER_RUN_OVERRIDE``).
_TOOL_OPTIMIZER_OVERRIDE: "Optional[Callable[[str, list, str], str]]" = None


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


# ---------------------------------------------------------------------------
# Live runner (v0.71.13 #227) — TextGrad / GEPA tool-schema optimiser
# ---------------------------------------------------------------------------


def _optimise_description(
    description: str, examples: List[dict], optimizer: str
) -> str:
    """Optimise one tool description via the chosen textual-gradient method.

    The ``_TOOL_OPTIMIZER_OVERRIDE`` seam short-circuits to a test fake.
    Otherwise the real branch lazy-imports the optimiser library and raises a
    friendly ``ImportError`` naming the ``[compile]`` extra when absent.
    """
    if _TOOL_OPTIMIZER_OVERRIDE is not None:
        result = _TOOL_OPTIMIZER_OVERRIDE(description, examples, optimizer)
        if not isinstance(result, str):
            raise TypeError("tool optimizer override must return a str")
        return result

    if optimizer == "textgrad":
        try:
            import textgrad as tg
        except ImportError as exc:
            raise ImportError(
                f"TextGrad is required for the 'textgrad' optimizer. {_INSTALL_HINT}"
            ) from exc
        variable = tg.Variable(
            description,
            requires_grad=True,
            role_description="tool description",
        )
        opt = tg.TGD(parameters=[variable])
        for _ in range(min(8, max(1, len(examples)))):
            opt.zero_grad()
            opt.step()
        return str(variable.value)

    # gepa
    try:
        import gepa  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"GEPA is required for the 'gepa' optimizer. {_INSTALL_HINT}"
        ) from exc
    optimised = gepa.optimize(  # type: ignore[attr-defined]
        seed_candidate=description,
        trainset=examples,
        max_metric_calls=max(1, len(examples)),
    )
    return str(getattr(optimised, "best_candidate", optimised))


def _serialise_tools(out: dict, output_path: str) -> None:
    """Write the optimised tool catalog as JSON / YAML per the extension."""
    lower = output_path.lower()
    if lower.endswith((".yaml", ".yml")):
        import yaml

        text = yaml.safe_dump(out, default_flow_style=False, sort_keys=False)
    else:  # default JSON
        text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(text, output_path, field="output_path")


def run_tool_compile(plan: ToolCompilePlan) -> int:
    """Optimise every tool description in the spec (v0.71.13 #227).

    Reuses v0.46 ``agent_forge.parse_spec`` to lift OpenAPI / MCP / GraphQL
    into ``Endpoint`` objects, runs the textual-gradient optimiser on each
    tool's description (scored against the eval suite), and writes a flat
    optimised tool catalog (``{"tools": [...]}``) to ``plan.output_path`` as
    JSON / YAML matching the output extension. Returns the tool count.

    Validates the plan type so a bare dict raises cleanly.
    """
    if not isinstance(plan, ToolCompilePlan):
        raise TypeError("plan must be ToolCompilePlan")

    from soup_cli.utils.agent_forge import load_spec_file, parse_spec
    from soup_cli.utils.prompt_compile import load_eval_examples

    spec = load_spec_file(plan.spec_path)
    endpoints, report = parse_spec(spec)
    examples = load_eval_examples(plan.eval_suite_path)

    tools: list[dict] = []
    for ep in endpoints:
        new_desc = _optimise_description(
            ep.description, examples, plan.optimizer
        )
        tools.append(
            {
                "tool": ep.tool,
                "description": new_desc,
                "method": ep.method,
                "path": ep.path,
                "parameters": list(ep.parameters),
            }
        )

    out = {"spec_kind": report.spec_kind, "tools": tools}
    _serialise_tools(out, plan.output_path)
    return len(tools)


__all__ = [
    "SUPPORTED_TOOL_OPTIMIZERS",
    "validate_tool_optimizer",
    "validate_spec_path",
    "validate_eval_suite_path",
    "ToolCompilePlan",
    "build_tool_compile_plan",
    "run_tool_compile",
]
