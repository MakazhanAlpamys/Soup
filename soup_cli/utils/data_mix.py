"""Data Mixing Optimizer (v0.48.0 Part B — BETA).

Run short proxy-training runs with different per-dataset mixture weights and
fit a Gaussian Process surrogate to recommend the optimal mixture.

This module ships the schema + budget accountant + recipe writer. The live
Bayesian optimisation loop (``scikit-optimize``) is wired through a
runtime-injected ``OptimizerProtocol`` so unit tests can drive deterministic
mock optimisers and the library import is lazy (matches the project's
``[optional-extras]`` policy — heavy deps never crash ``soup data --help``).

CLI surface:
    soup data mix --optimize --budget 1h --datasets a.jsonl,b.jsonl,c.jsonl
    soup data mix --apply <recipe.yaml>

Security:
- All input/output paths are containment-checked via ``utils.paths.is_under_cwd``.
- ``--budget`` is wall-clock capped; partial results returned on early exit.
- Dataset paths reject null bytes / oversize / non-string.
- ``scikit-optimize`` is lazy-imported; missing dep surfaces a friendly advisory.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

# --- Limits / constants ---------------------------------------------------

_MAX_DATASETS = 32  # mirrors v0.42.0 interleave cap
_MAX_PROBES = 256  # hard ceiling on N short proxy runs
_DEFAULT_PROBES = 8
_MIN_BUDGET_SECONDS = 60  # 1 minute
_MAX_BUDGET_SECONDS = 24 * 60 * 60  # 24 hours
_MAX_PATH_LEN = 4096
_MAX_RECIPE_BYTES = 256 * 1024  # mirror v0.39.0 Part E
_MAX_LOSS = 1e6
_FLOAT_TOL = 1e-6

__all__ = [
    "MixCandidate",
    "MixOptimizationReport",
    "MixOptimizationPlan",
    "BudgetTracker",
    "OptimizerProtocol",
    "validate_datasets",
    "parse_budget",
    "build_optimization_plan",
    "render_mix_recipe_yaml",
    "write_mix_recipe",
    "run_mix_optimizer",
]


# --- Dataclasses ----------------------------------------------------------


@dataclass(frozen=True)
class MixCandidate:
    """One proxy-run candidate: mixture weights + observed eval loss.

    Attributes:
        weights: Per-dataset weights summing to 1.0 ± 1e-6.
        eval_loss: Observed eval loss after the short proxy run.
        wall_clock_seconds: Time spent on this candidate.
    """

    weights: Tuple[float, ...]
    eval_loss: float
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.weights, bool) or not isinstance(
            self.weights, tuple
        ):
            raise TypeError(
                f"weights must be tuple, got {type(self.weights).__name__}"
            )
        if not self.weights:
            raise ValueError("weights must be non-empty")
        for w in self.weights:
            if isinstance(w, bool):
                raise ValueError("weight must be float, not bool")
            if not isinstance(w, (int, float)):
                raise TypeError(f"weight must be float, got {type(w).__name__}")
            fw = float(w)
            if not math.isfinite(fw):
                raise ValueError(f"weight must be finite (got {w!r})")
            if fw < 0.0 or fw > 1.0:
                raise ValueError(f"weight must be in [0, 1], got {fw}")
        total = sum(float(w) for w in self.weights)
        if abs(total - 1.0) > _FLOAT_TOL:
            raise ValueError(
                f"weights must sum to 1.0 ± {_FLOAT_TOL} (got {total})"
            )
        for name, value in (
            ("eval_loss", self.eval_loss),
            ("wall_clock_seconds", self.wall_clock_seconds),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be float, not bool")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be float, got {type(value).__name__}"
                )
            fv = float(value)
            if not math.isfinite(fv):
                raise ValueError(f"{name} must be finite (got {value!r})")
            if fv < 0.0:
                raise ValueError(f"{name} must be >= 0 (got {fv})")
        if self.eval_loss > _MAX_LOSS:
            raise ValueError(
                f"eval_loss exceeds sanity cap {_MAX_LOSS} (got {self.eval_loss})"
            )


@dataclass(frozen=True)
class MixOptimizationReport:
    """Result of a mixing-optimizer run.

    Attributes:
        datasets: The dataset paths in canonical order.
        candidates: Tuple of evaluated candidates (chronological).
        best_weights: Mixture with the lowest eval loss observed.
        best_eval_loss: The corresponding loss.
        partial: True when the budget tripped before all candidates ran.
        elapsed_seconds: Sum of per-successful-candidate wall-clock time.
            v0.53.5 #118: failed-proxy candidates are EXCLUDED so a long-failing
            proxy cannot inflate the field.
    """

    datasets: Tuple[str, ...]
    candidates: Tuple[MixCandidate, ...]
    best_weights: Tuple[float, ...]
    best_eval_loss: float
    partial: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class MixOptimizationPlan:
    """Validated plan for a mixing-optimization invocation.

    Attributes:
        datasets: Canonical dataset paths (real-paths within cwd).
        num_probes: How many proxy runs to attempt.
        budget_seconds: Hard wall-clock cap.
        seed: RNG seed for the optimizer.
    """

    datasets: Tuple[str, ...]
    num_probes: int
    budget_seconds: int
    seed: int


# --- Validation helpers ---------------------------------------------------


def _reject_bool_int(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def _check_str_path(name: str, value) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain null bytes")
    if len(value) > _MAX_PATH_LEN:
        raise ValueError(
            f"{name} length {len(value)} exceeds cap {_MAX_PATH_LEN}"
        )
    return value


def validate_datasets(raw: Sequence[str]) -> Tuple[str, ...]:
    """Validate dataset paths: containment + dedup + bounds.

    Args:
        raw: Sequence of dataset paths (relative or absolute).

    Returns:
        Tuple of real-path strings, all confined to cwd.

    Raises:
        TypeError / ValueError on bad input.
    """
    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("datasets must be a non-string Sequence")
    if len(raw) < 2:
        raise ValueError(
            f"datasets must contain at least 2 entries (got {len(raw)})"
        )
    if len(raw) > _MAX_DATASETS:
        raise ValueError(
            f"datasets has {len(raw)} entries; cap is {_MAX_DATASETS}"
        )
    seen: List[str] = []
    for item in raw:
        path = _check_str_path("dataset", item)
        # Symlink check on the ORIGINAL path BEFORE realpath (which would
        # follow the symlink, defeating the check). Matches v0.46.0 Part A
        # `_reject_symlink_target` pattern.
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise ValueError(
                f"dataset path is not stat-able: {os.path.basename(path)!r}"
            ) from exc
        if st is not None and stat.S_ISLNK(st.st_mode):
            raise ValueError(
                f"dataset path is a symlink (rejected for safety): "
                f"{os.path.basename(path)!r}"
            )
        real = os.path.realpath(path)
        if not is_under_cwd(real):
            raise ValueError(
                f"dataset path is outside cwd: {os.path.basename(real)!r}"
            )
        if real in seen:
            raise ValueError(
                f"duplicate dataset path: {os.path.basename(real)!r}"
            )
        seen.append(real)
    if len(seen) < 2:
        raise ValueError(
            "data mix requires at least 2 distinct datasets"
        )
    return tuple(seen)


def parse_budget(raw: str) -> int:
    """Parse a wall-clock budget string into seconds.

    Accepts:
        ``30s`` / ``5m`` / ``1h`` / ``600`` (bare seconds).

    Raises:
        ValueError on invalid format / out-of-bounds.
    """
    if not isinstance(raw, str):
        raise TypeError(f"budget must be str, got {type(raw).__name__}")
    s = raw.strip().lower()
    if not s:
        raise ValueError("budget must be non-empty")
    if "\x00" in s:
        raise ValueError("budget must not contain null bytes")

    multiplier = 1
    body = s
    if s.endswith("s"):
        body = s[:-1]
    elif s.endswith("m"):
        body = s[:-1]
        multiplier = 60
    elif s.endswith("h"):
        body = s[:-1]
        multiplier = 3600
    if not body or not body.isdigit():
        raise ValueError(
            f"budget must be digits + optional suffix (s/m/h), got {raw!r}"
        )
    seconds = int(body) * multiplier
    if seconds < _MIN_BUDGET_SECONDS or seconds > _MAX_BUDGET_SECONDS:
        raise ValueError(
            f"budget must resolve to [{_MIN_BUDGET_SECONDS}, "
            f"{_MAX_BUDGET_SECONDS}] seconds (got {seconds})"
        )
    return seconds


def build_optimization_plan(
    datasets: Sequence[str],
    *,
    budget: str = "1h",
    num_probes: int = _DEFAULT_PROBES,
    seed: int = 42,
) -> MixOptimizationPlan:
    """Validate args and produce a frozen plan."""
    ds = validate_datasets(datasets)
    budget_seconds = parse_budget(budget)
    nb = _reject_bool_int("num_probes", num_probes)
    if nb < 1 or nb > _MAX_PROBES:
        raise ValueError(
            f"num_probes must be in [1, {_MAX_PROBES}], got {nb}"
        )
    sd = _reject_bool_int("seed", seed)
    if sd < 0 or sd > 2**31 - 1:
        raise ValueError(f"seed must be in [0, 2**31-1], got {sd}")
    return MixOptimizationPlan(
        datasets=ds,
        num_probes=nb,
        budget_seconds=budget_seconds,
        seed=sd,
    )


# --- Budget tracker -------------------------------------------------------


class BudgetTracker:
    """Wall-clock budget accountant.

    Used by :func:`run_mix_optimizer` to terminate the BO loop when the
    cumulative time exceeds the configured budget. Partial results are
    surfaced via :class:`MixOptimizationReport` with ``partial=True``.
    """

    def __init__(
        self,
        budget_seconds: int,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        bs = _reject_bool_int("budget_seconds", budget_seconds)
        if bs < _MIN_BUDGET_SECONDS or bs > _MAX_BUDGET_SECONDS:
            raise ValueError(
                f"budget_seconds must be in "
                f"[{_MIN_BUDGET_SECONDS}, {_MAX_BUDGET_SECONDS}], got {bs}"
            )
        self._budget = bs
        self._clock = clock or time.monotonic
        self._started: Optional[float] = None

    def start(self) -> None:
        if self._started is not None:
            raise RuntimeError("BudgetTracker.start called twice")
        self._started = self._clock()

    @property
    def elapsed(self) -> float:
        if self._started is None:
            return 0.0
        return max(0.0, self._clock() - self._started)

    @property
    def remaining(self) -> float:
        return max(0.0, self._budget - self.elapsed)

    def exceeded(self) -> bool:
        return self.elapsed >= self._budget


# --- Optimizer protocol ---------------------------------------------------


class OptimizerProtocol(Protocol):
    """Duck-typed interface for the BO backend.

    Implementations must produce non-negative weights that sum to 1.0; the
    runner re-normalises to defend against floating-point drift.
    """

    def ask(self) -> Tuple[float, ...]:
        """Return the next candidate weights."""

    def tell(self, weights: Tuple[float, ...], loss: float) -> None:
        """Record an observation."""


def _build_skopt_optimizer(
    num_datasets: int, seed: int
) -> OptimizerProtocol:
    """Wrap ``skopt.Optimizer`` behind :class:`OptimizerProtocol` (v0.53.5 #117).

    Raises:
        ImportError: when ``scikit-optimize`` is not installed.
    """
    if isinstance(num_datasets, bool) or not isinstance(num_datasets, int):
        raise TypeError(
            f"num_datasets must be int, got {type(num_datasets).__name__}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if num_datasets < 2:
        raise ValueError(
            f"num_datasets must be >= 2 (got {num_datasets})"
        )
    import skopt  # noqa: PLC0415 — heavy optional dep, lazy.

    inner = skopt.Optimizer(
        dimensions=[(0.0, 1.0)] * num_datasets,
        n_initial_points=min(5, num_datasets),
        random_state=seed,
        base_estimator="GP",
    )

    class _SkoptWrapper:
        def ask(self) -> Tuple[float, ...]:
            raw = inner.ask()
            return _renormalize(raw)

        def tell(self, weights: Tuple[float, ...], loss: float) -> None:
            inner.tell(list(weights), float(loss))

    return _SkoptWrapper()


def _build_default_optimizer(
    num_datasets: int, seed: int
) -> OptimizerProtocol:
    """Return :func:`_build_skopt_optimizer` when ``scikit-optimize`` is
    installed; otherwise fall back to a deterministic Dirichlet-like sampler.
    """
    try:
        return _build_skopt_optimizer(num_datasets, seed)
    except ImportError:
        pass
    import random  # noqa: PLC0415

    rng = random.Random(seed)

    class _Dirichlet:
        def ask(self) -> Tuple[float, ...]:
            # Symmetric Dirichlet(α=1) via independent exponentials.
            raw = [rng.expovariate(1.0) for _ in range(num_datasets)]
            total = sum(raw) or 1.0
            return tuple(r / total for r in raw)

        def tell(self, weights: Tuple[float, ...], loss: float) -> None:
            return

    return _Dirichlet()


def _renormalize(weights: Sequence[float]) -> Tuple[float, ...]:
    """Clip + renormalise to a valid simplex point."""
    clipped = [max(0.0, float(w)) for w in weights]
    total = sum(clipped)
    if total <= 0.0:
        n = len(clipped)
        return tuple([1.0 / n] * n) if n else ()
    return tuple(c / total for c in clipped)


# --- Optimizer runner -----------------------------------------------------


def run_mix_optimizer(
    plan: MixOptimizationPlan,
    proxy_run: Callable[[Tuple[float, ...]], float],
    *,
    optimizer: Optional[OptimizerProtocol] = None,
    clock: Optional[Callable[[], float]] = None,
) -> MixOptimizationReport:
    """Run the BO loop over the validated plan.

    Args:
        plan: A :class:`MixOptimizationPlan` from
            :func:`build_optimization_plan`.
        proxy_run: Callable that takes weights and returns observed eval loss.
            In the live wiring this calls a short ``soup train`` invocation;
            in tests it is mocked.
        optimizer: Optional injected :class:`OptimizerProtocol`. Defaults to
            the Dirichlet sampler when absent.
        clock: Optional monotonic clock callable (testability).

    Returns:
        A :class:`MixOptimizationReport`. ``partial=True`` when budget tripped.
    """
    if not isinstance(plan, MixOptimizationPlan):
        raise TypeError(
            f"plan must be MixOptimizationPlan, got {type(plan).__name__}"
        )
    if not callable(proxy_run):
        raise TypeError("proxy_run must be callable")
    if optimizer is not None and (
        not hasattr(optimizer, "ask") or not hasattr(optimizer, "tell")
    ):
        raise TypeError(
            "optimizer must implement OptimizerProtocol (ask + tell)"
        )

    opt = optimizer or _build_default_optimizer(
        len(plan.datasets), plan.seed
    )
    tracker = BudgetTracker(plan.budget_seconds, clock=clock)
    tracker.start()

    candidates: List[MixCandidate] = []
    best_weights: Optional[Tuple[float, ...]] = None
    best_loss: float = math.inf
    partial = False

    for _ in range(plan.num_probes):
        if tracker.exceeded():
            partial = True
            break
        weights = _renormalize(opt.ask())
        if len(weights) != len(plan.datasets):
            raise ValueError(
                f"optimizer returned {len(weights)} weights; "
                f"expected {len(plan.datasets)}"
            )
        t0 = tracker.elapsed
        try:
            loss = proxy_run(weights)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Proxy failures are isolated per-candidate (matches v0.33.0
            # #47 CrossDocCollator + v0.40.3 judge_filter_pairs policy).
            # The candidate is recorded with a sentinel high loss so the
            # optimiser sees a valid observation and the run continues.
            import logging
            logging.getLogger(__name__).debug(
                "proxy_run raised for candidate %s", weights, exc_info=True
            )
            opt.tell(weights, _MAX_LOSS)
            continue
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise TypeError(
                f"proxy_run must return float, got {type(loss).__name__}"
            )
        loss_f = float(loss)
        if not math.isfinite(loss_f):
            # Skip — invalid observation should not poison best-of search.
            opt.tell(weights, _MAX_LOSS)
            continue
        opt.tell(weights, loss_f)
        cand = MixCandidate(
            weights=weights,
            eval_loss=loss_f,
            wall_clock_seconds=tracker.elapsed - t0,
        )
        candidates.append(cand)
        if loss_f < best_loss:
            best_loss = loss_f
            best_weights = weights

    if best_weights is None:
        # No valid observation — pick uniform as graceful fallback.
        n = len(plan.datasets)
        best_weights = tuple([1.0 / n] * n)
        best_loss = math.inf if not candidates else best_loss

    # v0.53.5 #118: report.elapsed_seconds reflects ONLY successful-candidate
    # time. Tracker.elapsed (which includes failed-proxy time) is no longer
    # surfaced via the public field; the caller can query the tracker directly
    # if total wall-clock is needed.
    successful_elapsed = sum(
        float(c.wall_clock_seconds) for c in candidates
    )
    return MixOptimizationReport(
        datasets=plan.datasets,
        candidates=tuple(candidates),
        best_weights=best_weights,
        best_eval_loss=best_loss if math.isfinite(best_loss) else float("nan"),
        partial=partial,
        elapsed_seconds=successful_elapsed,
    )


# --- Recipe writer --------------------------------------------------------


def render_mix_recipe_yaml(report: MixOptimizationReport) -> str:
    """Render an applied-mixture recipe snippet for human review.

    Produces a YAML fragment suitable for splicing into ``soup.yaml`` under
    ``data:``. Defends against YAML key injection by rejecting newlines and
    null bytes in dataset paths (mirrors v0.46.0 Part A
    ``render_recipe_yaml``).
    """
    if not isinstance(report, MixOptimizationReport):
        raise TypeError(
            "report must be MixOptimizationReport, "
            f"got {type(report).__name__}"
        )
    for path in report.datasets:
        if not isinstance(path, str) or "\n" in path or "\x00" in path:
            raise ValueError(
                "dataset path contains control characters — refusing to "
                "render YAML."
            )
        if len(path) > _MAX_PATH_LEN:
            raise ValueError(
                f"dataset path length {len(path)} exceeds {_MAX_PATH_LEN}"
            )
    lines = ["# Generated by `soup data mix --optimize` (v0.48.0 — BETA)"]
    lines.append(f"# Probes evaluated: {len(report.candidates)}")
    lines.append(
        f"# Best eval loss: {report.best_eval_loss:.6f}"
        if math.isfinite(report.best_eval_loss)
        else "# Best eval loss: (no valid observation)"
    )
    if report.partial:
        lines.append("# Budget exceeded — partial results.")
    lines.append("data:")
    lines.append("  interleave:")
    lines.append("    strategy: probs")
    lines.append("    probs:")
    for w in report.best_weights:
        lines.append(f"      - {w:.6f}")
    lines.append("  train:")
    for path in report.datasets:
        lines.append(f"    - {json.dumps(path)}")
    return "\n".join(lines) + "\n"


def write_mix_recipe(
    report: MixOptimizationReport,
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write the rendered recipe to ``output_path``.

    Containment + TOCTOU symlink rejection mirrors v0.46.0 Part A
    ``write_recipe`` and v0.47.0 Part A ``write_forge_dataset``.
    """
    from soup_cli.utils.paths import is_under_cwd

    _check_str_path("output_path", output_path)
    # Symlink check on the ORIGINAL path BEFORE realpath.
    try:
        st = os.lstat(output_path)
    except FileNotFoundError:
        st = None
    except OSError as exc:
        raise ValueError(
            f"output_path is not stat-able: "
            f"{os.path.basename(output_path)!r}"
        ) from exc
    if st is not None and stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"output_path is a symlink (rejected for safety): "
            f"{os.path.basename(output_path)!r}"
        )
    real = os.path.realpath(output_path)
    if not is_under_cwd(real):
        raise ValueError(
            f"output_path is outside cwd: {os.path.basename(real)!r}"
        )
    if st is not None and not overwrite:
        raise ValueError(
            f"output_path already exists (use overwrite=True): "
            f"{os.path.basename(real)!r}"
        )

    text = render_mix_recipe_yaml(report)
    if len(text.encode("utf-8")) > _MAX_RECIPE_BYTES:
        raise ValueError(
            f"rendered recipe exceeds {_MAX_RECIPE_BYTES} bytes cap"
        )
    parent = os.path.dirname(real) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".mix_recipe.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, real)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return real


def load_mix_recipe(path: str) -> Mapping[str, object]:
    """Load + validate a previously-written mix recipe.

    Used by ``soup data mix --apply <recipe.yaml>`` to splice the recommended
    mixture into a target ``soup.yaml``.
    """
    from soup_cli.utils.paths import is_under_cwd

    _check_str_path("path", path)
    # Symlink check on the ORIGINAL path BEFORE realpath.
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        st = None
    except OSError as exc:
        raise ValueError(
            f"recipe path is not stat-able: {os.path.basename(path)!r}"
        ) from exc
    if st is not None and stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"recipe path is a symlink (rejected for safety): "
            f"{os.path.basename(path)!r}"
        )
    real = os.path.realpath(path)
    if not is_under_cwd(real):
        raise ValueError(
            f"recipe path is outside cwd: {os.path.basename(real)!r}"
        )
    if not os.path.isfile(real):
        raise FileNotFoundError(
            f"recipe not found: {os.path.basename(real)!r}"
        )
    size = os.path.getsize(real)
    if size > _MAX_RECIPE_BYTES:
        raise ValueError(
            f"recipe exceeds {_MAX_RECIPE_BYTES} bytes cap (got {size})"
        )
    import yaml

    with open(real, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError("recipe must be a YAML mapping at top level")
    data_block = data.get("data")
    if not isinstance(data_block, Mapping):
        raise ValueError("recipe missing required 'data:' mapping")
    return data_block
