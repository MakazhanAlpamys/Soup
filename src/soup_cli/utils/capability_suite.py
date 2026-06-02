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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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
        raise ValueError(f"benchmark name too int ({len(name)} > {_MAX_NAME_LEN})")
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
        raise ValueError(f"suite name too int ({len(name)} > {_MAX_SUITE_LEN})")
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
