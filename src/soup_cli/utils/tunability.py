"""`soup tunability` — probe-train 6-10 candidate bases, report Pareto frontier.

Before committing to a single base model, an operator can run a short LoRA
probe on each of N candidates against a held-out dataset slice. Reports
which candidates are on the (delta-from-base × cost × license) Pareto
frontier so the operator can pick the best fit instead of relying on
vendor catalogs / blog benchmarks.

Why blue-ocean: hosted vendors push their own catalogs (attach-rate
incentive); comparing across families is structurally costly for them.
Soup is local-first + spans every base.

Live wiring of the in-process LoRA probe is deferred to v0.64.1
(mirrors v0.27.0 MII / v0.50.0 GRPO Plus / v0.56.0 diagnose stub-then-live
pattern); v0.64.0 ships the schema, default catalogue, Pareto math,
report writer + the CLI surface so operators can plan a sweep and pipe
results through a custom probe callable today.

Public surface:
- ``CandidateBase`` frozen dataclass (name / repo_id / params_b / license_id).
- ``TunabilityResult`` frozen dataclass (per-candidate probe outcome).
- ``TunabilityReport`` frozen dataclass (results + Pareto frontier + meta).
- ``DEFAULT_CANDIDATES`` tuple of 8 sane defaults across families.
- ``validate_probe_steps(value)`` -> int in [10, 10_000].
- ``validate_holdout_size(value)`` -> int in [10, 100_000].
- ``score_candidate(*, base_loss, probe_loss)`` -> float delta.
- ``pareto_frontier(results)`` -> tuple of non-dominated TunabilityResult.
- ``run_tunability(...)`` -> TunabilityReport (orchestrator).
- ``write_report(report, path)`` / ``load_report(path)`` -> atomic JSON.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from soup_cli.utils.paths import atomic_write_text, is_under_cwd

# Bounds — mirror v0.30.0 / v0.41.0 / v0.51.0 validator policy.
_MIN_PROBE_STEPS = 10
_MAX_PROBE_STEPS = 10_000
_MIN_HOLDOUT = 10
_MAX_HOLDOUT = 100_000
_MAX_NAME_LEN = 512
_MAX_REPO_ID_LEN = 512
_MAX_LICENSE_LEN = 128
_MAX_CANDIDATES = 32


@dataclass(frozen=True)
class CandidateBase:
    """A candidate base model for a tunability probe.

    `params_b` is the parameter count in billions (e.g. ``0.6`` for 600M).
    """

    name: str
    repo_id: str
    params_b: float
    license_id: str

    def __post_init__(self) -> None:
        _check_str(self.name, "name", max_len=_MAX_NAME_LEN)
        _check_str(self.repo_id, "repo_id", max_len=_MAX_REPO_ID_LEN)
        _check_str(self.license_id, "license_id", max_len=_MAX_LICENSE_LEN)
        if isinstance(self.params_b, bool):
            raise TypeError("params_b must be a number, not bool")
        if not isinstance(self.params_b, (int, float)):
            raise TypeError(
                f"params_b must be a number, got {type(self.params_b).__name__}"
            )
        if not math.isfinite(float(self.params_b)):
            raise ValueError("params_b must be finite (no NaN / Inf)")
        if self.params_b <= 0:
            raise ValueError(f"params_b must be > 0, got {self.params_b}")


def _check_str(value: object, field: str, *, max_len: int) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} too int (> {max_len} chars)")


@dataclass(frozen=True)
class TunabilityResult:
    """Outcome of one candidate's probe run."""

    candidate: CandidateBase
    base_loss: float
    probe_loss: float
    delta: float
    wall_clock_seconds: float
    estimated_cost_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateBase):
            raise TypeError(
                f"candidate must be CandidateBase, got {type(self.candidate).__name__}"
            )
        for field, val in (
            ("base_loss", self.base_loss),
            ("probe_loss", self.probe_loss),
            ("delta", self.delta),
            ("wall_clock_seconds", self.wall_clock_seconds),
            ("estimated_cost_usd", self.estimated_cost_usd),
        ):
            if isinstance(val, bool):
                raise TypeError(f"{field} must be a number, not bool")
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"{field} must be a number, got {type(val).__name__}"
                )
            if not math.isfinite(float(val)):
                raise ValueError(f"{field} must be finite (no NaN / Inf)")
        if self.wall_clock_seconds < 0:
            raise ValueError(
                f"wall_clock_seconds must be >= 0, got {self.wall_clock_seconds}"
            )
        if self.estimated_cost_usd < 0:
            raise ValueError(
                f"estimated_cost_usd must be >= 0, got {self.estimated_cost_usd}"
            )


@dataclass(frozen=True)
class TunabilityReport:
    """End-to-end report for a tunability sweep."""

    results: Tuple[TunabilityResult, ...]
    frontier: Tuple[TunabilityResult, ...]
    probe_steps: int
    holdout_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple):
            raise TypeError("results must be a tuple of TunabilityResult")
        if not isinstance(self.frontier, tuple):
            raise TypeError("frontier must be a tuple of TunabilityResult")
        for entry in self.results:
            if not isinstance(entry, TunabilityResult):
                raise TypeError("every result must be TunabilityResult")
        for entry in self.frontier:
            if not isinstance(entry, TunabilityResult):
                raise TypeError("every frontier entry must be TunabilityResult")


# Default catalogue: cross-family small bases that fit in 12 GB VRAM with 4-bit
# LoRA. Operators can override via `--candidates <name1,name2,...>` (matched
# against `name`) or supply a custom YAML file.
DEFAULT_CANDIDATES: Tuple[CandidateBase, ...] = (
    CandidateBase(
        name="qwen3-0.6b",
        repo_id="Qwen/Qwen3-0.6B",
        params_b=0.6,
        license_id="apache-2.0",
    ),
    CandidateBase(
        name="qwen3-1.7b",
        repo_id="Qwen/Qwen3-1.7B",
        params_b=1.7,
        license_id="apache-2.0",
    ),
    CandidateBase(
        name="llama-3.2-1b",
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        params_b=1.0,
        license_id="llama-3.2",
    ),
    CandidateBase(
        name="llama-3.2-3b",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.0,
        license_id="llama-3.2",
    ),
    CandidateBase(
        name="gemma-3-e2b",
        repo_id="google/gemma-3-2b-it",
        params_b=2.0,
        license_id="gemma",
    ),
    CandidateBase(
        name="phi-4-mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        params_b=3.8,
        license_id="mit",
    ),
    CandidateBase(
        name="smollm3-1.7b",
        repo_id="HuggingFaceTB/SmolLM3-1.7B-Instruct",
        params_b=1.7,
        license_id="apache-2.0",
    ),
    CandidateBase(
        name="qwen2.5-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.5,
        license_id="apache-2.0",
    ),
)


def validate_probe_steps(value: object) -> int:
    """Validate ``probe_steps`` in [10, 10_000]; reject bool / non-int."""
    if isinstance(value, bool):
        raise TypeError("probe_steps must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"probe_steps must be int, got {type(value).__name__}")
    if not (_MIN_PROBE_STEPS <= value <= _MAX_PROBE_STEPS):
        raise ValueError(
            f"probe_steps must be in [{_MIN_PROBE_STEPS}, {_MAX_PROBE_STEPS}], "
            f"got {value}"
        )
    return value


def validate_holdout_size(value: object) -> int:
    """Validate ``holdout_size`` in [10, 100_000]; reject bool / non-int."""
    if isinstance(value, bool):
        raise TypeError("holdout_size must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"holdout_size must be int, got {type(value).__name__}")
    if not (_MIN_HOLDOUT <= value <= _MAX_HOLDOUT):
        raise ValueError(
            f"holdout_size must be in [{_MIN_HOLDOUT}, {_MAX_HOLDOUT}], "
            f"got {value}"
        )
    return value


def score_candidate(*, base_loss: float, probe_loss: float) -> float:
    """Compute the candidate's delta (higher = bigger improvement).

    delta = base_loss - probe_loss. Positive means the LoRA probe lowered
    loss from the base; negative means it made things worse.
    """
    for name, val in (("base_loss", base_loss), ("probe_loss", probe_loss)):
        if isinstance(val, bool):
            raise TypeError(f"{name} must be a number, not bool")
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
        if not math.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    return float(base_loss) - float(probe_loss)


def _dominates(a: TunabilityResult, b: TunabilityResult) -> bool:
    """Return True iff ``a`` strictly dominates ``b`` on (delta, cost).

    A dominates B iff: A.delta >= B.delta AND A.cost <= B.cost,
    and at least one of the inequalities is strict.
    """
    delta_ge = a.delta >= b.delta
    cost_le = a.estimated_cost_usd <= b.estimated_cost_usd
    strict = (a.delta > b.delta) or (a.estimated_cost_usd < b.estimated_cost_usd)
    return delta_ge and cost_le and strict


def pareto_frontier(results: Sequence[TunabilityResult]) -> Tuple[TunabilityResult, ...]:
    """Return the Pareto-optimal subset of results.

    Maximises ``delta``, minimises ``estimated_cost_usd``. A result survives
    iff no other result strictly dominates it on both axes.
    """
    if isinstance(results, str):
        raise TypeError("results must be a sequence of TunabilityResult, not str")
    try:
        materialised = list(results)
    except TypeError as exc:
        raise TypeError("results must be iterable") from exc
    for entry in materialised:
        if not isinstance(entry, TunabilityResult):
            raise TypeError(
                f"every result must be TunabilityResult, got {type(entry).__name__}"
            )
    if not materialised:
        return ()
    frontier: list[TunabilityResult] = []
    for i, candidate in enumerate(materialised):
        dominated = False
        for j, other in enumerate(materialised):
            if i == j:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


# Type alias for the probe callable. Live impl lands in v0.64.1.
ProbeFn = Callable[[CandidateBase, str], TunabilityResult]


def _default_probe(
    candidate: CandidateBase,
    dataset_path: str,
    *,
    probe_steps: int,
    holdout_size: int,
) -> TunabilityResult:
    """Heuristic stand-in. Live LoRA probe lands in v0.64.1.

    Returns deterministic, candidate-derived values so reports parse and
    Pareto math exercises without a GPU. Operators wanting a real probe
    inject a callable via ``probe_fn=...``.
    """
    # Wall-clock + cost scale with params. Delta is a tiny constant so
    # the report is honest about being a stub.
    wall_clock = 60.0 + 30.0 * float(candidate.params_b)
    cost = 0.001 * float(candidate.params_b) * float(probe_steps)
    return TunabilityResult(
        candidate=candidate,
        base_loss=2.5,
        probe_loss=2.5,  # No change — heuristic stub
        delta=0.0,
        wall_clock_seconds=wall_clock,
        estimated_cost_usd=cost,
    )


def run_tunability(
    *,
    candidates: Sequence[CandidateBase],
    dataset_path: str,
    probe_steps: int = 100,
    holdout_size: int = 64,
    probe_fn: Optional[Callable[..., TunabilityResult]] = None,
) -> TunabilityReport:
    """Orchestrate probes across candidates and assemble a report.

    The actual probing is delegated to ``probe_fn`` (signature:
    ``(cand, dataset_path, *, probe_steps, holdout_size) -> TunabilityResult``).
    If ``probe_fn`` is None, falls back to the heuristic stub.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of CandidateBase")
    if len(candidates) == 0:
        raise ValueError("candidates must contain at least one entry")
    if len(candidates) > _MAX_CANDIDATES:
        raise ValueError(
            f"too many candidates ({len(candidates)} > {_MAX_CANDIDATES})"
        )
    for entry in candidates:
        if not isinstance(entry, CandidateBase):
            raise TypeError(
                f"every candidate must be CandidateBase, "
                f"got {type(entry).__name__}"
            )

    steps = validate_probe_steps(probe_steps)
    holdout = validate_holdout_size(holdout_size)

    if not isinstance(dataset_path, str):
        raise TypeError(
            f"dataset_path must be str, got {type(dataset_path).__name__}"
        )
    if not dataset_path:
        raise ValueError("dataset_path must be non-empty")
    if "\x00" in dataset_path:
        raise ValueError("dataset_path must not contain null bytes")

    fn = probe_fn if probe_fn is not None else _default_probe
    results: list[TunabilityResult] = []
    for cand in candidates:
        result = fn(cand, dataset_path, probe_steps=steps, holdout_size=holdout)
        if not isinstance(result, TunabilityResult):
            raise TypeError(
                f"probe_fn must return TunabilityResult, "
                f"got {type(result).__name__}"
            )
        results.append(result)

    frontier = pareto_frontier(results)
    return TunabilityReport(
        results=tuple(results),
        frontier=frontier,
        probe_steps=steps,
        holdout_size=holdout,
    )


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------


def _result_to_dict(r: TunabilityResult) -> dict:
    return {
        "candidate": {
            "name": r.candidate.name,
            "repo_id": r.candidate.repo_id,
            "params_b": r.candidate.params_b,
            "license_id": r.candidate.license_id,
        },
        "base_loss": r.base_loss,
        "probe_loss": r.probe_loss,
        "delta": r.delta,
        "wall_clock_seconds": r.wall_clock_seconds,
        "estimated_cost_usd": r.estimated_cost_usd,
    }


def _result_from_dict(d: dict) -> TunabilityResult:
    if not isinstance(d, dict):
        raise ValueError("result must be a dict")
    cand_raw = d.get("candidate")
    if not isinstance(cand_raw, dict):
        raise ValueError("candidate must be a dict")
    cand = CandidateBase(
        name=cand_raw["name"],
        repo_id=cand_raw["repo_id"],
        params_b=float(cand_raw["params_b"]),
        license_id=cand_raw["license_id"],
    )
    return TunabilityResult(
        candidate=cand,
        base_loss=float(d["base_loss"]),
        probe_loss=float(d["probe_loss"]),
        delta=float(d["delta"]),
        wall_clock_seconds=float(d["wall_clock_seconds"]),
        estimated_cost_usd=float(d["estimated_cost_usd"]),
    )


def write_report(report: TunabilityReport, path: str) -> None:
    """Atomically write a TunabilityReport as JSON.

    Path is cwd-containment-checked + symlink-rejected via the shared
    ``enforce_under_cwd_and_no_symlink`` helper (mirrors v0.59.0 /
    v0.60.0 / v0.62.0 atomic-write policy).
    """
    if not isinstance(report, TunabilityReport):
        raise TypeError(
            f"report must be TunabilityReport, got {type(report).__name__}"
        )
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")

    payload = {
        "schema_version": "1",
        "probe_steps": report.probe_steps,
        "holdout_size": report.holdout_size,
        "results": [_result_to_dict(r) for r in report.results],
        "frontier": [_result_to_dict(r) for r in report.frontier],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_write_text(text, path, prefix=".tunability.", field="tunability output")


def load_report(path: str) -> TunabilityReport:
    """Load a TunabilityReport from JSON. Raises FileNotFoundError if missing.

    Path is cwd-containment-checked + symlink-rejected BEFORE the
    existence probe so a crafted path cannot leak file-existence
    distinguishing "outside cwd" from "missing" (mirrors v0.55.0 /
    v0.62.0 ordering policy).
    """
    import stat as _stat

    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"path {path!r} is outside cwd")
    if os.path.lexists(path):
        st = os.lstat(path)
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError("report path must not be a symlink")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("report root must be a dict")
    results = tuple(_result_from_dict(r) for r in payload.get("results", []))
    frontier = tuple(_result_from_dict(r) for r in payload.get("frontier", []))
    return TunabilityReport(
        results=results,
        frontier=frontier,
        probe_steps=int(payload.get("probe_steps", _MIN_PROBE_STEPS)),
        holdout_size=int(payload.get("holdout_size", _MIN_HOLDOUT)),
    )


__all__ = [
    "CandidateBase",
    "DEFAULT_CANDIDATES",
    "ProbeFn",
    "TunabilityReport",
    "TunabilityResult",
    "load_report",
    "pareto_frontier",
    "run_tunability",
    "score_candidate",
    "validate_holdout_size",
    "validate_probe_steps",
    "write_report",
]
