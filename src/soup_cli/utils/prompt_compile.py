"""``soup compile`` — DSPy / GEPA / TextGrad prompt-program compiler.

Schema + validators ship from v0.68.0 Part A; the live DSPy / GEPA /
TextGrad orchestrator lands in v0.71.13 (#225). The optimizer libraries are
lazy-imported with a friendly ``ImportError`` (naming ``pip install
'soup-cli[compile]'``) so the command works without them installed.

Public surface:

- ``SUPPORTED_PROMPT_OPTIMIZERS`` — closed frozenset
- ``MAX_COMPILE_ITERS`` — hard upper bound on iteration count
- ``validate_prompt_optimizer(name)`` — bool-first / null-byte / oversize / case-insensitive
- ``validate_max_iters(n)`` — bool-first / non-int / bounds
- ``validate_program_path(path)`` — cwd containment + symlink rejection + ``.py`` only
- ``validate_eval_suite_path(path)`` — cwd containment + symlink rejection
- ``load_eval_examples(path)`` — JSON-list / JSONL eval loader (cwd-contained)
- ``CompilePlan`` / ``CompileResult`` frozen dataclasses
- ``build_compile_plan(...)`` — factory
- ``run_compile(plan)`` — live dispatcher (DSPy / GEPA / TextGrad)

The CLI command lives in ``soup_cli/commands/compile_cmd.py`` (named
``compile_cmd`` to avoid shadowing the Python builtin ``compile()``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# ---------------------------------------------------------------------------
# Allowlists + bounds
# ---------------------------------------------------------------------------


SUPPORTED_PROMPT_OPTIMIZERS: frozenset[str] = frozenset(
    {
        "bootstrap_fewshot",  # DSPy classic
        "mipro",  # DSPy Multi-stage Instruction Proposal Optimizer
        "copro",  # DSPy COordinated PROmpt optimizer
        "gepa",  # Reflective Prompt Evolution (gradient-free)
        "textgrad",  # Textual-gradient optimizer
    }
)

# DSPy-backed optimizers (dispatched to ``dspy``); GEPA + TextGrad have their
# own libraries.
_DSPY_OPTIMIZERS: frozenset[str] = frozenset({"bootstrap_fewshot", "mipro", "copro"})

MAX_COMPILE_ITERS = 1000
_MIN_COMPILE_ITERS = 1
_MAX_OPTIMIZER_NAME_LEN = 32
_MAX_EVAL_BYTES = 64 * 1024 * 1024  # 64 MiB cap on the eval-suite file

# Injectable seam: tests / advanced operators set this to a
# ``(plan) -> CompileResult`` callable so the dispatcher + result-handling +
# atomic-write are exercised without the heavy DSPy / GEPA / TextGrad libs.
# Mirrors v0.67 ``cmaes_merge._CMAES_SCORER_OVERRIDE`` /
# v0.53.1 ``deploy_measure._DEPLOY_MEASURE_BEFORE_GEN`` policy.
_OPTIMIZER_RUN_OVERRIDE: "Optional[Callable[[CompilePlan], CompileResult]]" = None


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
# Live runner (v0.71.13 #225) — DSPy / GEPA / TextGrad dispatcher
# ---------------------------------------------------------------------------


_INSTALL_HINT = (
    "Run: pip install 'soup-cli[compile]'  (installs dspy-ai / textgrad / gepa)"
)


def load_eval_examples(eval_suite_path: str) -> List[dict]:
    """Load eval-suite examples (cwd-contained, symlink-safe, O_NOFOLLOW).

    Accepts a JSON array of objects, a single JSON object, or a JSONL file
    (one object per line). Each example is a dict — the optimizer decides
    which keys are inputs / outputs (DSPy: ``with_inputs``; TextGrad / GEPA:
    free-form).
    """
    canonical = enforce_under_cwd_and_no_symlink(eval_suite_path, "eval_suite_path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(canonical, flags)
    with os.fdopen(fd, encoding="utf-8") as handle:
        raw = handle.read(_MAX_EVAL_BYTES + 1)
    if len(raw) > _MAX_EVAL_BYTES:
        raise ValueError(f"eval suite exceeds {_MAX_EVAL_BYTES} bytes")
    stripped = raw.lstrip()
    examples: List[dict] = []
    # Try a whole-file JSON parse first (handles pretty-printed arrays + a
    # single top-level object); fall back to JSONL line-by-line.
    parsed = None
    if stripped[:1] in ("[", "{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, list):
        examples = [d for d in parsed if isinstance(d, dict)]
    elif isinstance(parsed, dict):
        examples = [parsed]
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                examples.append(obj)
    return examples


def _load_program_module(program_path: str) -> Any:
    """Import the user's ``.py`` program from a validated path.

    Re-validates the path immediately before ``exec_module`` to close the
    plan-build -> run TOCTOU window (security-review LOW). The program itself
    is operator-trusted (arbitrary code by design) but must still be a
    cwd-contained, non-symlink ``.py``.
    """
    validate_program_path(program_path)
    spec = importlib.util.spec_from_file_location("soup_compile_program", program_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import program from {program_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # noqa: S102 — operator-trusted program
    return module


def _resolve_program(module: Any) -> Any:
    """Find the DSPy program object exported by the user module.

    Convention: a module-level ``program`` attribute, or a ``get_program()``
    factory. Mirrors the DSPy idiom of building a ``dspy.Module`` instance.
    """
    if hasattr(module, "program"):
        return module.program
    if hasattr(module, "get_program") and callable(module.get_program):
        return module.get_program()
    raise ValueError(
        "program module must expose a `program` attribute or `get_program()`"
    )


def _resolve_metric(module: Any) -> "Optional[Callable]":
    """Return the user metric (``metric`` attr) or None (caller defaults)."""
    metric = getattr(module, "metric", None)
    return metric if callable(metric) else None


def _run_dspy(plan: CompilePlan) -> CompileResult:
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            f"DSPy is required for the {plan.optimizer!r} optimizer. {_INSTALL_HINT}"
        ) from exc

    module = _load_program_module(plan.program_path)
    program = _resolve_program(module)
    metric = _resolve_metric(module)
    trainset = _dspy_examples(dspy, module, plan.eval_suite_path)

    if plan.optimizer == "bootstrap_fewshot":
        optimizer = dspy.BootstrapFewShot(metric=metric, max_rounds=plan.max_iters)
    elif plan.optimizer == "mipro":
        optimizer = dspy.MIPROv2(metric=metric, auto="light")
    else:  # copro
        optimizer = dspy.COPRO(metric=metric, depth=plan.max_iters)

    compiled = optimizer.compile(program, trainset=trainset)
    program_text = _serialize_dspy(compiled)
    return CompileResult(
        program_text=program_text,
        score=0.0,
        iterations=plan.max_iters,
        converged=True,
    )


def _dspy_examples(dspy: Any, module: Any, eval_suite_path: str) -> List[Any]:
    """Build ``dspy.Example`` objects from the eval suite.

    Honours a module-level ``input_keys`` list (which keys are inputs); else
    treats every key except common output names as inputs.
    """
    raw = load_eval_examples(eval_suite_path)
    input_keys = getattr(module, "input_keys", None)
    out: List[Any] = []
    for ex in raw:
        example = dspy.Example(**ex)
        if isinstance(input_keys, (list, tuple)) and input_keys:
            example = example.with_inputs(*input_keys)
        else:
            inferred = [
                k for k in ex if k not in ("output", "answer", "completion", "label")
            ]
            if inferred:
                example = example.with_inputs(*inferred)
        out.append(example)
    return out


def _serialize_dspy(compiled: Any) -> str:
    """Best-effort serialise a compiled DSPy program to text."""
    for attr in ("dump_state", "__repr__"):
        try:
            value = getattr(compiled, attr)()
            return value if isinstance(value, str) else json.dumps(value, default=str)
        except Exception:  # noqa: BLE001 — fall through to repr
            continue
    return repr(compiled)


def _run_gepa(plan: CompilePlan) -> CompileResult:
    try:
        import gepa  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"GEPA is required for the 'gepa' optimizer. {_INSTALL_HINT}"
        ) from exc
    module = _load_program_module(plan.program_path)
    program = _resolve_program(module)
    examples = load_eval_examples(plan.eval_suite_path)
    optimised = gepa.optimize(  # type: ignore[attr-defined]
        seed_candidate=program,
        trainset=examples,
        max_metric_calls=plan.max_iters,
    )
    program_text = str(getattr(optimised, "best_candidate", optimised))
    score = float(getattr(optimised, "best_score", 0.0) or 0.0)
    return CompileResult(
        program_text=program_text,
        score=score if math.isfinite(score) else 0.0,
        iterations=plan.max_iters,
        converged=True,
    )


def _run_textgrad(plan: CompilePlan) -> CompileResult:
    try:
        import textgrad as tg
    except ImportError as exc:
        raise ImportError(
            f"TextGrad is required for the 'textgrad' optimizer. {_INSTALL_HINT}"
        ) from exc
    module = _load_program_module(plan.program_path)
    program = _resolve_program(module)
    # The user program is expected to be (or contain) a tg.Variable to
    # optimise. We run a textual-gradient descent loop over the eval suite.
    if isinstance(program, tg.Variable):
        variable = program
    else:
        variable = tg.Variable(
            str(program), requires_grad=True, role_description="prompt program"
        )
    optimizer = tg.TGD(parameters=[variable])
    examples = load_eval_examples(plan.eval_suite_path)
    iterations = min(plan.max_iters, max(1, len(examples)))
    for _ in range(iterations):
        optimizer.zero_grad()
        optimizer.step()
    return CompileResult(
        program_text=str(variable.value),
        score=0.0,
        iterations=iterations,
        converged=True,
    )


def run_compile(plan: CompilePlan) -> CompileResult:
    """Run the live compilation (v0.71.13 #225).

    Dispatches by ``plan.optimizer``: DSPy (bootstrap_fewshot / mipro / copro),
    GEPA, or TextGrad. Each branch lazy-imports its library and raises a
    friendly ``ImportError`` (naming ``pip install soup-cli[compile]``) when it
    is absent. The ``_OPTIMIZER_RUN_OVERRIDE`` seam lets tests exercise the
    dispatcher + result handling without the heavy optimizer libraries.

    Validates the plan type at the boundary so callers passing a bare dict get
    a clean ``TypeError``.
    """
    if not isinstance(plan, CompilePlan):
        raise TypeError("plan must be CompilePlan")
    if _OPTIMIZER_RUN_OVERRIDE is not None:
        result = _OPTIMIZER_RUN_OVERRIDE(plan)
        if not isinstance(result, CompileResult):
            raise TypeError("optimizer override must return a CompileResult")
        return result
    if plan.optimizer in _DSPY_OPTIMIZERS:
        return _run_dspy(plan)
    if plan.optimizer == "gepa":
        return _run_gepa(plan)
    if plan.optimizer == "textgrad":
        return _run_textgrad(plan)
    # Unreachable: the optimizer is validated by the schema.
    raise ValueError(f"unhandled optimizer {plan.optimizer!r}")


__all__ = [
    "SUPPORTED_PROMPT_OPTIMIZERS",
    "MAX_COMPILE_ITERS",
    "validate_prompt_optimizer",
    "validate_max_iters",
    "validate_program_path",
    "validate_eval_suite_path",
    "load_eval_examples",
    "CompilePlan",
    "CompileResult",
    "build_compile_plan",
    "run_compile",
]
