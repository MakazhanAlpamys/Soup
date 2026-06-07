"""Evolutionary CMA-ES merge for LoRA adapters (v0.67.0 Part A).

CMA-ES over merge weights driven by the operator's eval. Extends v0.57.0
``soup adapters merge`` with a ``cmaes`` strategy that searches the
N-dimensional simplex of mixing weights to maximise an operator-supplied
eval score.

Pure Python; no `cma` dependency. The implementation is a small simplex-
projecting CMA-ES (rank-mu + diagonal covariance) sufficient for ≤16
adapters and 1–100 generations. Operators wanting full CMA-ES (BIPOP,
restart strategies) can plug their own optimiser via the ``eval_fn``
hook; this module's contract is "given eval_fn, run a budgeted search,
return the best simplex weights".

Public surface:

- ``CmaesPlan`` + ``CmaesResult`` frozen dataclasses
- ``validate_population_size`` / ``validate_generations``
- ``build_cmaes_plan(...)`` returns frozen ``CmaesPlan``
- ``run_cmaes_merge(plan, *, eval_fn, ...)`` returns ``CmaesResult``

Reuses ``parse_budget`` from v0.57.0 ``blame.py`` (60s..24h bounds).

Design notes:

- Output weights live on the simplex (sum=1, each ≥0); the optimiser
  parameterises N-1 logits and softmaxes them so any candidate is valid.
- Failures inside ``eval_fn`` are swallowed as a sentinel low score so
  one broken adapter doesn't crash the run (mirrors v0.40.3 #33 / v0.48
  proxy-failure isolation).
- Live ``soup eval`` wiring is operator-supplied — `cmaes_merge` does
  NOT auto-load models. Callers wrap their eval suite as a closure.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from soup_cli.utils.blame import parse_budget
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# Test / smoke escape hatch (mirrors v0.53.1 #109 deploy_measure pattern):
# when set, build_cmaes_eval_fn uses this scorer instead of loading a model.
# Production callers leave it None and either pass scorer= explicitly or let
# the default lazy model-loading scorer run.
_CMAES_SCORER_OVERRIDE: Optional[Callable[[str, str], float]] = None

# ---------------------------------------------------------------------------
# Bounds (closed, locked at module load)
# ---------------------------------------------------------------------------

MIN_POPULATION = 2
MAX_POPULATION = 256
MIN_GENERATIONS = 1
MAX_GENERATIONS = 10_000

_MIN_ADAPTERS = 2
_MAX_ADAPTERS = 16  # mirrors v0.57.0 adapter_merge cap
_SIMPLEX_TOL = 1e-6
_FAILED_EVAL_SENTINEL = -1.0e9  # very negative so failed candidates never win
_MAX_ADAPTER_CONFIG_BYTES = 256 * 1024  # mirrors adapter_merge config-read cap


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_population_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("population_size must be int")
    if value < MIN_POPULATION:
        raise ValueError(
            f"population_size {value} below floor {MIN_POPULATION}"
        )
    if value > MAX_POPULATION:
        raise ValueError(
            f"population_size {value} above cap {MAX_POPULATION}"
        )
    return value


def validate_generations(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_generations must be int")
    if value < MIN_GENERATIONS:
        raise ValueError(
            f"max_generations {value} below floor {MIN_GENERATIONS}"
        )
    if value > MAX_GENERATIONS:
        raise ValueError(
            f"max_generations {value} above cap {MAX_GENERATIONS}"
        )
    return value


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if seed > 2**31 - 1:
        raise ValueError("seed too large")
    return seed


def _validate_finite_score(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    return val


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmaesPlan:
    """Plan for an evolutionary merge run.

    Reuses v0.57 ``parse_budget`` so the budget bounds (60s..24h) are
    consistent across blame / cmaes.
    """

    adapters: Tuple[str, ...]
    eval_suite: str
    budget_seconds: int
    population_size: int
    max_generations: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.adapters, tuple):
            raise TypeError("adapters must be tuple")
        if len(self.adapters) < _MIN_ADAPTERS:
            raise ValueError(
                f"need at least {_MIN_ADAPTERS} adapters"
            )
        if len(self.adapters) > _MAX_ADAPTERS:
            raise ValueError(f"at most {_MAX_ADAPTERS} adapters")
        for path in self.adapters:
            if not isinstance(path, str) or not path:
                raise ValueError("adapters entries must be non-empty str")
        if not isinstance(self.eval_suite, str) or not self.eval_suite:
            raise ValueError("eval_suite must be non-empty str")
        validate_population_size(self.population_size)
        validate_generations(self.max_generations)
        _validate_seed(self.seed)


@dataclass(frozen=True)
class CmaesResult:
    """Result of an evolutionary merge run.

    ``best_weights`` live on the simplex (sum=1, each ≥0).
    """

    best_weights: Tuple[float, ...]
    best_score: float
    generations_run: int
    evaluations: int
    wall_clock_seconds: float
    converged: bool
    history: Tuple[float, ...]

    def __post_init__(self) -> None:
        # Score finite + bool-rejected
        _validate_finite_score(self.best_score, "best_score")
        if not isinstance(self.generations_run, int) or isinstance(
            self.generations_run, bool
        ):
            raise TypeError("generations_run must be int")
        if self.generations_run < 0:
            raise ValueError("generations_run must be non-negative")
        if not isinstance(self.evaluations, int) or isinstance(
            self.evaluations, bool
        ):
            raise TypeError("evaluations must be int")
        if self.evaluations < 0:
            raise ValueError("evaluations must be non-negative")
        _validate_finite_score(self.wall_clock_seconds, "wall_clock_seconds")
        if self.wall_clock_seconds < 0:
            raise ValueError("wall_clock_seconds must be non-negative")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        if not isinstance(self.best_weights, tuple):
            raise TypeError("best_weights must be tuple")
        if not self.best_weights:
            raise ValueError("best_weights must be non-empty")
        for w in self.best_weights:
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise TypeError("best_weights entries must be numeric")
            wf = float(w)
            if not math.isfinite(wf):
                raise ValueError("best_weights entries must be finite")
            if wf < 0:
                raise ValueError("best_weights entries must be non-negative")
        total = sum(float(w) for w in self.best_weights)
        if not math.isclose(total, 1.0, abs_tol=_SIMPLEX_TOL):
            raise ValueError(
                f"best_weights must sum to 1.0 (got {total:.6f})"
            )
        if not isinstance(self.history, tuple):
            raise TypeError("history must be tuple")
        for h in self.history:
            if isinstance(h, bool) or not isinstance(h, (int, float)):
                raise TypeError("history entries must be numeric")
            if not math.isfinite(float(h)):
                raise ValueError("history entries must be finite")


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_cmaes_plan(
    *,
    adapters: Sequence[str],
    eval_suite: str,
    budget_spec: str,
    population_size: int = 8,
    max_generations: int = 20,
    seed: int = 0,
) -> CmaesPlan:
    """Validate inputs and return a frozen ``CmaesPlan``.

    - ``adapters`` must contain ≥2 unique paths, all under cwd.
    - ``eval_suite`` must be a real path under cwd (no symlinks).
    - ``budget_spec`` is parsed via v0.57 ``parse_budget`` (60s..24h).
    """
    if not isinstance(adapters, Sequence) or isinstance(adapters, str):
        raise TypeError("adapters must be a sequence")
    if len(adapters) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    validated_adapters: list[str] = []
    for ad in adapters:
        if not isinstance(ad, str) or not ad:
            raise ValueError("each adapter must be a non-empty str")
        enforce_under_cwd_and_no_symlink(ad, field="adapter")
        validated_adapters.append(os.path.realpath(ad))

    enforce_under_cwd_and_no_symlink(eval_suite, field="eval_suite")
    if not os.path.exists(eval_suite):
        raise FileNotFoundError(f"eval_suite not found: {eval_suite!r}")

    budget_seconds = parse_budget(budget_spec)

    return CmaesPlan(
        adapters=tuple(validated_adapters),
        eval_suite=os.path.realpath(eval_suite),
        budget_seconds=budget_seconds,
        population_size=validate_population_size(population_size),
        max_generations=validate_generations(max_generations),
        seed=_validate_seed(seed),
    )


# ---------------------------------------------------------------------------
# Optimiser (minimal CMA-ES on the N-1 logit space, softmaxed onto simplex)
# ---------------------------------------------------------------------------


def _softmax(logits: Sequence[float]) -> Tuple[float, ...]:
    """Numerically-stable softmax onto the simplex."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    if z <= 0:
        # Degenerate fallback: uniform
        return tuple(1.0 / len(logits) for _ in logits)
    return tuple(e / z for e in exps)


def _eval_safely(
    eval_fn: Callable[[Tuple[float, ...]], float],
    weights: Tuple[float, ...],
) -> float:
    """Call ``eval_fn`` with sentinel-low score on exception.

    Failure isolation mirrors v0.40.3 #33 / v0.48 / v0.53.7 #106 policy:
    one bad eval must not crash the whole run.
    """
    try:
        score = eval_fn(weights)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001 — eval_fn surface is operator-controlled
        return _FAILED_EVAL_SENTINEL
    if isinstance(score, bool):
        return _FAILED_EVAL_SENTINEL
    if not isinstance(score, (int, float)):
        return _FAILED_EVAL_SENTINEL
    s = float(score)
    if not math.isfinite(s):
        return _FAILED_EVAL_SENTINEL
    return s


def run_cmaes_merge(
    plan: CmaesPlan,
    *,
    eval_fn: Callable[[Tuple[float, ...]], float],
    sigma_init: float = 0.5,
    elite_frac: float = 0.5,
    convergence_tol: float = 1e-4,
) -> CmaesResult:
    """Run a small CMA-ES-style search over simplex merge weights.

    Operator supplies ``eval_fn(weights) -> score`` (higher is better);
    we softmax N-1 logits onto the simplex, sample a population, keep
    the elite half, re-fit a diagonal Gaussian, repeat until either:

    - ``max_generations`` reached, or
    - elapsed wall-clock ≥ ``budget_seconds``, or
    - score plateau < ``convergence_tol`` for 3 generations in a row.
    """
    if not isinstance(plan, CmaesPlan):
        raise TypeError("plan must be CmaesPlan")
    if eval_fn is None or not callable(eval_fn):
        raise TypeError("eval_fn must be callable")
    if isinstance(sigma_init, bool) or not isinstance(sigma_init, (int, float)):
        raise TypeError("sigma_init must be numeric")
    if not math.isfinite(float(sigma_init)) or float(sigma_init) <= 0:
        raise ValueError("sigma_init must be positive and finite")
    if isinstance(elite_frac, bool) or not isinstance(elite_frac, (int, float)):
        raise TypeError("elite_frac must be numeric")
    elite = float(elite_frac)
    if not (0.0 < elite < 1.0) or not math.isfinite(elite):
        raise ValueError("elite_frac must be in (0, 1)")

    n_dim = len(plan.adapters) - 1  # softmax over N-1 free logits
    rng = _LcgRng(plan.seed)

    mean = [0.0] * n_dim
    sigma = [float(sigma_init)] * n_dim
    elite_count = max(1, int(plan.population_size * elite))

    history: list[float] = []
    best_score = -math.inf
    best_weights: Tuple[float, ...] = tuple(
        1.0 / len(plan.adapters) for _ in plan.adapters
    )

    start = time.monotonic()
    generations_run = 0
    evaluations = 0
    converged = False
    plateau_run = 0

    for gen in range(plan.max_generations):
        if (time.monotonic() - start) >= plan.budget_seconds:
            break

        # Sample population
        candidates: list[Tuple[list[float], Tuple[float, ...], float]] = []
        for _ in range(plan.population_size):
            sample = [
                mean[i] + sigma[i] * rng.normal() for i in range(n_dim)
            ]
            # Append 0.0 reference logit (softmax is shift-invariant) so we
            # span the full simplex.
            logits = sample + [0.0]
            weights = _softmax(logits)
            score = _eval_safely(eval_fn, weights)
            evaluations += 1
            candidates.append((sample, weights, score))
            if score > best_score:
                best_score = score
                best_weights = weights
            if (time.monotonic() - start) >= plan.budget_seconds:
                break

        generations_run += 1
        # Pick elite by score (descending)
        candidates.sort(key=lambda c: c[2], reverse=True)
        elite_samples = [c[0] for c in candidates[:elite_count]]
        # Recompute mean + sigma per dim from elite
        new_mean = [
            sum(s[i] for s in elite_samples) / len(elite_samples)
            for i in range(n_dim)
        ]
        new_sigma = []
        for i in range(n_dim):
            var = sum(
                (s[i] - new_mean[i]) ** 2 for s in elite_samples
            ) / len(elite_samples)
            # Clamp to a sensible floor; CMA-ES typically blends with prior
            # but for ≤16-adapter case the simple rank-mu update converges.
            new_sigma.append(max(math.sqrt(var), 1e-4))
        # Plateau detection
        gen_best = candidates[0][2]
        history.append(gen_best)
        if len(history) >= 2:
            delta = abs(history[-1] - history[-2])
            if delta < convergence_tol:
                plateau_run += 1
            else:
                plateau_run = 0
        if plateau_run >= 3:
            converged = True
            break
        mean = new_mean
        sigma = new_sigma

    wall_clock = time.monotonic() - start
    if best_score == -math.inf:
        # Defensive — should not happen because we always sample at least once
        best_score = 0.0

    return CmaesResult(
        best_weights=tuple(best_weights),
        best_score=float(best_score),
        generations_run=generations_run,
        evaluations=evaluations,
        wall_clock_seconds=float(wall_clock),
        converged=converged,
        history=tuple(history),
    )


# ---------------------------------------------------------------------------
# Live eval-suite auto-wiring (v0.71.4 #220)
# ---------------------------------------------------------------------------


def _read_merged_base_name(merged_dir: str) -> str:
    """Read + validate ``base_model_name_or_path`` from a merged adapter dir.

    On the wired cmaes path ``merged_dir`` is always a Soup-written mkdtemp,
    but this is a public helper — guard the config read with a symlink
    rejection + size cap so a caller-supplied dir can't smuggle a symlink or
    a multi-GB JSON (defence-in-depth, parity with every other config read).
    """
    import json
    import os
    import stat
    from pathlib import Path

    cfg_path = Path(merged_dir) / "adapter_config.json"
    try:
        cst = os.lstat(cfg_path)
    except OSError as exc:
        raise ValueError(
            f"merged adapter_config.json unreadable: {type(exc).__name__}"
        ) from exc
    if stat.S_ISLNK(cst.st_mode):
        raise ValueError("merged adapter_config.json must not be a symlink")
    if cst.st_size > _MAX_ADAPTER_CONFIG_BYTES:
        raise ValueError("merged adapter_config.json exceeds size cap")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            "merged adapter_config.json has no base_model_name_or_path"
        )
    return base


def _generate(model, tokenizer, prompt: str) -> str:
    """Greedy single-prompt generation (continuation only)."""
    if not prompt:
        return ""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _load_adapter_generator(merged_dir: str) -> Callable[[str], str]:
    """Build a ``generate_fn(prompt) -> str`` from a merged LoRA adapter.

    Loads the base model named in the adapter's ``adapter_config.json`` and
    applies the merged adapter via PEFT. Heavy imports stay inside the
    function (project lazy-import policy + the cmaes_merge no-top-level-torch
    grep guard). One-shot: reloads the base on every call. The CMA-ES loop
    uses :class:`_CachedBaseScorer` instead so the base loads only once.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = _read_merged_base_name(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, merged_dir)
    model.eval()

    def _gen(prompt: str) -> str:
        return _generate(model, tokenizer, prompt)

    return _gen


class _CachedBaseScorer:
    """Live CMA-ES default scorer that loads the base model ONCE and reuses it.

    Every candidate in the CMA-ES population is a *linear merge of the same
    source LoRAs*, so they all share one base model. The base (multi-GB for a
    7B model) is loaded lazily on the first candidate and then reused for the
    whole ``population × max_generations`` loop — each candidate only loads its
    small merged LoRA, applies it, generates, and unloads it. This amortises
    the base load across the whole run (v0.71.15 #246; before, the stateless
    default scorer reloaded the base on every candidate).

    A fresh instance is built per :func:`build_cmaes_eval_fn` call so each
    merge run gets its own cache (no cross-run leakage).
    """

    def __init__(self) -> None:
        self._base = None
        self._tokenizer = None
        self._base_name: Optional[str] = None

    def _ensure_base(self, base_name: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._base is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_name, trust_remote_code=False
            )
            self._base = AutoModelForCausalLM.from_pretrained(
                base_name, trust_remote_code=False
            )
            self._base_name = base_name
        elif base_name != self._base_name:
            # All source adapters in one merge share a base; a mismatch means
            # a malformed merged dir. Fail loud rather than silently scoring
            # against the wrong base.
            raise ValueError(
                "cmaes candidate base_model mismatch: "
                f"{base_name!r} != cached {self._base_name!r}"
            )

    def __call__(self, merged_dir: str, eval_suite: str) -> float:
        from peft import PeftModel

        from soup_cli.eval.custom import load_eval_tasks, score_task

        base_name = _read_merged_base_name(merged_dir)
        self._ensure_base(base_name)
        peft_model = PeftModel.from_pretrained(self._base, merged_dir)
        peft_model.eval()
        try:
            tasks = load_eval_tasks(eval_suite)
            if not tasks:
                return 0.0
            total = 0.0
            for task in tasks:
                output = _generate(peft_model, self._tokenizer, task.prompt)
                total += float(score_task(task, output).score)
            return total / len(tasks)
        finally:
            # Strip the candidate LoRA so the next candidate re-wraps the same
            # base — restores a clean base model for reuse (no second adapter
            # accumulating onto the cached base). ``unload()`` removes the LoRA
            # layers but can leave a ``peft_config`` attribute behind, which
            # makes the next ``from_pretrained`` warn about "multiple adapters";
            # clear it so each candidate re-wraps a genuinely clean base.
            self._base = peft_model.unload()
            if hasattr(self._base, "peft_config"):
                try:
                    delattr(self._base, "peft_config")
                except (AttributeError, TypeError):
                    pass


def _default_cmaes_scorer(merged_dir: str, eval_suite: str) -> float:
    """One-shot live scorer (loads base + merged adapter, scores, frees).

    Stateless — reloads the base model on every call. The CMA-ES loop reuses
    the base via :class:`_CachedBaseScorer` (see :func:`build_cmaes_eval_fn`);
    this remains for direct / back-compat callers.
    """
    from soup_cli.eval.custom import load_eval_tasks, score_task

    generate_fn = _load_adapter_generator(merged_dir)
    tasks = load_eval_tasks(eval_suite)
    if not tasks:
        return 0.0
    total = 0.0
    for task in tasks:
        output = generate_fn(task.prompt)
        total += float(score_task(task, output).score)
    return total / len(tasks)


def _resolve_cmaes_scorer(
    scorer: Optional[Callable[[str, str], float]],
) -> Callable[[str, str], float]:
    if scorer is not None:
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        return scorer
    if _CMAES_SCORER_OVERRIDE is not None:
        return _CMAES_SCORER_OVERRIDE
    # Fresh cached-base scorer per build → base loads once, reused across the
    # whole population × generations loop (v0.71.15 #246).
    return _CachedBaseScorer()


def build_cmaes_eval_fn(
    plan: CmaesPlan,
    *,
    scorer: Optional[Callable[[str, str], float]] = None,
) -> Callable[[Tuple[float, ...]], float]:
    """Return an ``eval_fn(weights) -> float`` for :func:`run_cmaes_merge`.

    The closure linearly merges ``plan.adapters`` with the candidate weights,
    materialises the merged LoRA to a temp dir, scores it via ``scorer``
    (default: live model-loading scorer), and cleans up. The *adapter* weights
    are loaded from disk once up front so per-generation cost is merge + score.

    Perf note: the default scorer is :class:`_CachedBaseScorer`, which loads
    the *base model* exactly once and reuses it across every candidate (each
    candidate only loads its small merged LoRA — v0.71.15 #246). Pass a custom
    ``scorer=`` (or set ``_CMAES_SCORER_OVERRIDE``) to avoid any model load for
    tests / smokes.

    ``scorer(merged_dir, eval_suite) -> float`` returns a mean score in
    ``[0, 1]``.
    """
    if not isinstance(plan, CmaesPlan):
        raise TypeError("plan must be CmaesPlan")
    resolved = _resolve_cmaes_scorer(scorer)

    from soup_cli.utils.adapter_diff import load_adapter_weights
    from soup_cli.utils.adapter_merge import merge_linear, write_merged_adapter

    weights_list = [load_adapter_weights(p) for p in plan.adapters]
    template_source = plan.adapters[0]

    def _eval_fn(weights: Tuple[float, ...]) -> float:
        merged, _skipped = merge_linear(weights_list, list(weights))
        tmp_dir = tempfile.mkdtemp(prefix=".soup_cmaes_")
        try:
            write_merged_adapter(tmp_dir, template_source, merged)
            return float(resolved(tmp_dir, plan.eval_suite))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return _eval_fn


# ---------------------------------------------------------------------------
# Deterministic RNG (no numpy dependency at module top)
# ---------------------------------------------------------------------------


class _LcgRng:
    """Tiny deterministic linear-congruential RNG + Box-Muller normals.

    Avoids a numpy import at module top (matches v0.66 review-grep policy
    — heavy numpy stays inside live callsites only). Seeded so two runs
    with the same ``seed`` produce identical trajectories.
    """

    def __init__(self, seed: int) -> None:
        self._state = (seed * 2654435761 + 1) & 0xFFFFFFFFFFFFFFFF
        self._pending: Optional[float] = None

    def _uniform(self) -> float:
        # 64-bit LCG (Numerical Recipes-style)
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        # Take the top 53 bits to fit a double's mantissa
        return ((self._state >> 11) / float(1 << 53))

    def normal(self) -> float:
        """Box-Muller standard normal."""
        if self._pending is not None:
            val = self._pending
            self._pending = None
            return val
        u1 = max(self._uniform(), 1e-12)
        u2 = self._uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        a = 2.0 * math.pi * u2
        self._pending = r * math.sin(a)
        return r * math.cos(a)
