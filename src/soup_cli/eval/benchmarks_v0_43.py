"""v0.43.0 Part B — ceval / cmmlu / Aider Polyglot benchmark allowlist.

ceval / cmmlu route through the existing lm-evaluation-harness wiring; this
module exposes the validated benchmark name allowlist + Aider prompt-loader
scaffold. Live Aider Polyglot eval requires the upstream `aider-chat` package
and Docker; live wiring is deferred to v0.43.1 — this release ships the
benchmark name allowlist + path containment so YAML configs lock in.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

# v0.43.0 additive benchmark names.
NEW_BENCHMARKS_V0_43 = frozenset({"ceval", "cmmlu", "aider_polyglot"})

# Lightweight per-benchmark metadata (description + lm-eval task name).
_BENCHMARK_META: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "ceval": MappingProxyType({
        "description": "Chinese evaluation suite (52 subjects)",
        "lm_eval_task": "ceval-valid",
    }),
    "cmmlu": MappingProxyType({
        "description": "Chinese MMLU (67 subjects)",
        "lm_eval_task": "cmmlu",
    }),
    "aider_polyglot": MappingProxyType({
        "description": "Aider Polyglot benchmark (multi-language code editing)",
        "lm_eval_task": "",  # No lm-eval mapping; uses upstream aider-chat
    }),
})


def is_v0_43_benchmark(name: object) -> bool:
    """True if `name` names a v0.43.0-additive benchmark."""
    if not isinstance(name, str):
        return False
    return name.lower() in NEW_BENCHMARKS_V0_43


def benchmark_metadata(name: str) -> Mapping[str, str] | None:
    """Return read-only metadata for a v0.43.0 benchmark, or None."""
    if not isinstance(name, str):
        return None
    return _BENCHMARK_META.get(name.lower())


def lm_eval_task_for(name: str) -> str | None:
    """Map a benchmark name to its lm-eval-harness task id, or None."""
    meta = benchmark_metadata(name)
    if meta is None:
        return None
    task = meta.get("lm_eval_task")
    return task if task else None
