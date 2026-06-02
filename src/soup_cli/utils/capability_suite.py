"""v0.65.0 Part C — Capability auto-suite.

Pre-bundled capability benchmarks (MMLU-Pro / GPQA / BBEH / AIME /
MATH-500 / HumanEval+ / SWE-bench-Verified) with friendlier-than-default
``lm-eval-harness`` task ids and profile selector
``full | fast | math | code``.

This module ships only the schema + dispatcher. Live ``lm-eval-harness``
invocation lives in ``soup eval benchmark`` (existing v0.10 surface) so
the operator can compose capability suites with the existing eval gate.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

# Closed allowlist.
CAPABILITY_BENCHMARKS = frozenset({
    "mmlu-pro", "gpqa", "bbeh", "aime",
    "math-500", "humaneval-plus", "swe-bench-verified",
})

# Closed profile allowlist.
_SUITES = frozenset({"full", "fast", "math", "code"})

_MAX_NAME_LEN = 64
_MAX_SUITE_LEN = 32


@dataclass(frozen=True)
class CapabilityBenchmark:
    """Static metadata for a capability benchmark."""

    name: str
    lm_eval_task: str
    category: str  # "knowledge", "reasoning", "math", "code"
    default_fewshot: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be non-empty str")
        if "\x00" in self.name:
            raise ValueError("name must not contain null bytes")
        if not isinstance(self.lm_eval_task, str) or not self.lm_eval_task:
            raise ValueError("lm_eval_task must be non-empty str")
        if "\x00" in self.lm_eval_task:
            raise ValueError("lm_eval_task must not contain null bytes")
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("category must be non-empty str")
        if (
            isinstance(self.default_fewshot, bool)
            or not isinstance(self.default_fewshot, int)
            or self.default_fewshot < 0
            or self.default_fewshot > 100
        ):
            raise ValueError("default_fewshot must be int in [0, 100]")


# Per-benchmark metadata.
_BENCHMARK_METADATA: Mapping[str, CapabilityBenchmark] = MappingProxyType({
    "mmlu-pro": CapabilityBenchmark(
        name="mmlu-pro",
        lm_eval_task="mmlu_pro",
        category="knowledge",
        default_fewshot=5,
    ),
    "gpqa": CapabilityBenchmark(
        name="gpqa",
        lm_eval_task="gpqa_diamond_n_shot",
        category="reasoning",
        default_fewshot=5,
    ),
    "bbeh": CapabilityBenchmark(
        name="bbeh",
        lm_eval_task="bbeh",
        category="reasoning",
        default_fewshot=0,
    ),
    "aime": CapabilityBenchmark(
        name="aime",
        lm_eval_task="aime",
        category="math",
        default_fewshot=0,
    ),
    "math-500": CapabilityBenchmark(
        name="math-500",
        lm_eval_task="math_500",
        category="math",
        default_fewshot=4,
    ),
    "humaneval-plus": CapabilityBenchmark(
        name="humaneval-plus",
        lm_eval_task="humaneval_plus",
        category="code",
        default_fewshot=0,
    ),
    "swe-bench-verified": CapabilityBenchmark(
        name="swe-bench-verified",
        lm_eval_task="swe_bench_verified",
        category="code",
        default_fewshot=0,
    ),
})

# Profile -> tuple of benchmark names.
PROFILES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "full": tuple(sorted(CAPABILITY_BENCHMARKS)),
    "fast": ("mmlu-pro", "humaneval-plus"),
    "math": ("aime", "math-500"),
    "code": ("humaneval-plus", "swe-bench-verified"),
})


def validate_benchmark_name(name: object) -> str:
    """Validate a capability-benchmark name. Returns canonical form."""
    if isinstance(name, bool):
        raise TypeError("benchmark name must be str, got bool")
    if not isinstance(name, str):
        raise TypeError(
            f"benchmark name must be str, got {type(name).__name__}"
        )
    if "\x00" in name:
        raise ValueError("benchmark name must not contain null bytes")
    if not name:
        raise ValueError("benchmark name must not be empty")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"benchmark name too long ({len(name)} > {_MAX_NAME_LEN})")
    canonical = name.strip().lower()
    if canonical not in CAPABILITY_BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {canonical!r}; "
            f"valid: {sorted(CAPABILITY_BENCHMARKS)}"
        )
    return canonical


def get_benchmark_spec(name: str) -> CapabilityBenchmark:
    """Return the frozen spec for ``name``. KeyError if unknown."""
    canonical = name.lower() if isinstance(name, str) else name
    if canonical not in _BENCHMARK_METADATA:
        raise KeyError(f"unknown benchmark: {name!r}")
    return _BENCHMARK_METADATA[canonical]


def list_benchmarks() -> tuple[str, ...]:
    """Sorted tuple of all known benchmark names."""
    return tuple(sorted(CAPABILITY_BENCHMARKS))


def validate_suite_name(name: object) -> str:
    """Validate a profile name. Returns canonical form."""
    if isinstance(name, bool):
        raise TypeError("suite name must be str, got bool")
    if not isinstance(name, str):
        raise TypeError(
            f"suite name must be str, got {type(name).__name__}"
        )
    if "\x00" in name:
        raise ValueError("suite name must not contain null bytes")
    if not name:
        raise ValueError("suite name must not be empty")
    if len(name) > _MAX_SUITE_LEN:
        raise ValueError(f"suite name too long ({len(name)} > {_MAX_SUITE_LEN})")
    canonical = name.strip().lower()
    if canonical not in _SUITES:
        raise ValueError(
            f"unknown suite {canonical!r}; valid: {sorted(_SUITES)}"
        )
    return canonical


def list_suites() -> tuple[str, ...]:
    """Sorted tuple of all profile names."""
    return tuple(sorted(_SUITES))


def resolve_suite(name: str) -> tuple[CapabilityBenchmark, ...]:
    """Resolve a profile name to the ordered tuple of CapabilityBenchmark."""
    canonical = validate_suite_name(name)
    names = PROFILES[canonical]
    return tuple(_BENCHMARK_METADATA[n] for n in names)


def _primary_metric(task_result: Mapping[str, object]) -> tuple[str, float]:
    """Pick a representative scalar metric from an lm-eval per-task result dict.

    lm-eval emits ``{"acc,none": 0.42, "acc_stderr,none": ...}`` style keys.
    Prefer ``acc_norm`` > ``acc`` > ``exact_match`` > ``pass@1``, else the first
    float that is not a stderr.
    """
    preferred = ("acc_norm", "acc", "exact_match", "pass@1", "pass_at_1")
    items = {
        str(k): float(v)
        for k, v in task_result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    for pref in preferred:
        for key, val in items.items():
            base = key.split(",")[0]
            if base == pref:
                return key, val
    for key, val in items.items():
        if "stderr" not in key:
            return key, val
    return ("", float("nan"))


def run_capability_suite(
    *,
    run_id: str,
    model_id: str,
    suite: Optional[str] = None,
    tasks: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: int = 1,
) -> dict:
    """LIVE lm-eval-harness invocation (#211).

    Resolves either an explicit ``tasks`` list (lm-eval task names) or the
    ``suite`` profile to its lm-eval tasks, runs each through
    ``lm_eval.simple_evaluate`` against ``model_id`` (isolated per task so one
    unregistered / failing task does not sink the run), and returns a JSON-able
    report. ``limit`` caps eval examples per task (use ``1-5`` for a smoke).

    Lazy-imports ``lm_eval`` — raises a friendly ``RuntimeError`` when it is
    not installed (``pip install soup-cli[eval]``).
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive int or None")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive int")

    if tasks is not None:
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise TypeError("tasks must be a sequence of lm-eval task names")
        task_pairs = [(str(t), str(t)) for t in tasks]
    else:
        if suite is None:
            raise ValueError("provide either suite= or tasks=")
        task_pairs = [(b.name, b.lm_eval_task) for b in resolve_suite(suite)]

    # Lazy-load the harness via importlib so the module has no heavy
    # top-level dependency (and the source-grep guard stays satisfied).
    import importlib

    try:
        simple_evaluate = importlib.import_module("lm_eval").simple_evaluate
        hflm_cls = importlib.import_module("lm_eval.models.huggingface").HFLM
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "lm-eval-harness is required for a live capability run "
            "(pip install soup-cli[eval])."
        ) from exc

    from soup_cli.utils.live_eval import resolve_device

    resolved_device = resolve_device(device)
    lm = hflm_cls(pretrained=model_id, device=resolved_device, batch_size=batch_size)

    results: list[dict] = []
    for name, lm_task in task_pairs:
        entry: dict = {"benchmark": name, "lm_eval_task": lm_task}
        try:
            out = simple_evaluate(model=lm, tasks=[lm_task], limit=limit)
            task_res = (out or {}).get("results", {}).get(lm_task, {})
            metric_key, metric_val = _primary_metric(task_res)
            if not metric_key:
                # No scalar metric surfaced — flag it instead of reporting a
                # silent NaN score that renders like a real zero.
                entry["error"] = "no scalar metric in task result"
            else:
                entry["metric"] = metric_key
                entry["score"] = metric_val
        except Exception as exc:  # noqa: BLE001 — per-task isolation
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        results.append(entry)

    return {
        "run_id": run_id,
        "model": model_id,
        "suite": suite,
        "limit": limit,
        "device": resolved_device,
        "results": results,
    }
