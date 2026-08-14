"""SHIP / DON'T-SHIP verdict engine — `soup ship` (v0.71.25).

After fine-tuning, answer ONE question: did the model get better, or did I
break it? The output is a binary **SHIP / DON'T SHIP** plus a one-screen
reason — not a dashboard to guess from.

The decision fuses two legs into a single verdict::

    SHIP  <=>  (leg 1: task_tuned  >  task_base   — STRICT inequality)
          AND  (leg 2: for every benchmark:  base - tuned  <=  forgetting_threshold)
    else DON'T SHIP — even if the task metric looks great.

The moat is **leg 2 (catastrophic-forgetting / regression gate) as a
first-class co-equal of leg 1**, fused into one decision. The regression delta
math reuses the project's existing semantics (``eval/gate.py::run_gate`` and
``eval/leaderboard.compare_runs``): ``delta = tuned - base``; a benchmark
regresses when its *drop* (``base - tuned``) exceeds ``forgetting_threshold``
(absolute points, same meaning as ``EvalGateConfig.regression_threshold``).

This module is **pure-python (NO top-level torch)** so the whole truth table is
CPU-testable. Model loading + live evaluation live in ``commands/ship.py``.

Public surface
--------------
- Frozen dataclasses: ``TaskWin``, ``BenchmarkDelta``, ``ShipVerdict``.
- Constants: ``TASK_MODES``, ``SUPPORTED_TASK_MODES``, ``DECISION_SHIP`` /
  ``DECISION_DONT_SHIP``, the ``FAILED_*`` rule codes, ``DEFAULT_FORGETTING_THRESHOLD``.
- Pure functions: ``build_task_win``, ``compute_benchmark_deltas``,
  ``decide_ship`` (the moat), ``render_ship_panel``, ``format_ship_rubric``,
  ``verdict_to_dict``, ``verdict_to_evidence`` (the inverse of the ``--evidence``
  reader — makes ``soup ship`` output replayable as input, #312).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from soup_cli import __version__

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Leg-1 task-win modes. ``pairwise`` (true judge win-rate) landed in v0.71.31:
# a ``TaskWin(base=0.5 coin-flip, tuned=win-rate)`` where ``won = tuned > 0.5``.
TASK_MODES: Tuple[str, ...] = ("metric", "judge_score", "pairwise")
SUPPORTED_TASK_MODES: Tuple[str, ...] = ("metric", "judge_score", "pairwise")

DECISION_SHIP = "SHIP"
DECISION_DONT_SHIP = "DON'T SHIP"

# Failed-rule codes — which rule of the decision turned a SHIP into a DON'T.
FAILED_MISSING_BASELINE = "missing_baseline"  # leg 2 could not be measured
FAILED_TASK_WIN = "task_win"  # leg 1: task did not strictly improve
FAILED_REGRESSION = "regression"  # leg 2: a general benchmark regressed

# Default forgetting threshold — 0.05 ABSOLUTE points, mirroring
# ``EvalGateConfig.regression_threshold`` and ``run_gate`` semantics.
DEFAULT_FORGETTING_THRESHOLD = 0.05

# Float-noise tolerance so an exactly-at-threshold drop reads as OK (a -5.00%
# drop must not flip to "regressed" just because 0.80 - 0.75 == 0.05000000004).
_REGRESSION_EPS = 1e-9

# Round stored deltas so JSON / display are clean; the regression test uses the
# RAW base/tuned (not the rounded delta), so this never changes a verdict.
_DELTA_ROUND = 6

# CommonMark's minimum code-fence length (used by render_ship_pr_markdown).
_MIN_MD_FENCE_LEN = 3


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskWin:
    """Leg 1 — did the tuned model beat the base model on the target task?"""

    mode: str  # one of TASK_MODES
    base: float
    tuned: float
    won: bool  # tuned > base (STRICT)


@dataclass(frozen=True)
class BenchmarkDelta:
    """Leg 2 — one general-suite benchmark, base vs tuned."""

    name: str
    base: float
    tuned: float
    delta: float  # tuned - base (signed; negative = drop)
    regressed: bool  # drop exceeds the forgetting threshold


@dataclass(frozen=True)
class ShipVerdict:
    """The binary decision plus the evidence that produced it."""

    decision: str  # DECISION_SHIP | DECISION_DONT_SHIP
    task_win: TaskWin
    benchmark_deltas: Tuple[BenchmarkDelta, ...]
    failed_rule: Optional[str]  # None on SHIP; a FAILED_* code on DON'T SHIP
    forgetting_threshold: float
    soup_version: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_score(value: object, name: str) -> float:
    """Coerce ``value`` to a finite float, rejecting bool / non-numeric / NaN."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _validate_threshold(value: object) -> float:
    """Forgetting threshold: a finite float in ``[0.0, 1.0]`` (absolute points)."""
    out = _validate_score(value, "forgetting_threshold")
    if not (0.0 <= out <= 1.0):
        raise ValueError("forgetting_threshold must be in [0.0, 1.0]")
    return out


def _is_regressed(base: float, tuned: float, threshold: float) -> bool:
    """True when the drop (``base - tuned``) exceeds ``threshold`` (with eps).

    Mirrors ``run_gate``'s ``delta < -abs(threshold)`` but stated as a drop so
    the epsilon cleanly absorbs float noise at the exact boundary.
    """
    return (base - tuned) > threshold + _REGRESSION_EPS


# ---------------------------------------------------------------------------
# Builders (pure)
# ---------------------------------------------------------------------------

def build_task_win(mode: str, base: object, tuned: object) -> TaskWin:
    """Build a leg-1 ``TaskWin``. ``won`` is the STRICT inequality tuned > base."""
    if mode not in TASK_MODES:
        raise ValueError(f"mode must be one of {TASK_MODES}, got {mode!r}")
    base_f = _validate_score(base, "task base")
    tuned_f = _validate_score(tuned, "task tuned")
    return TaskWin(mode=mode, base=base_f, tuned=tuned_f, won=tuned_f > base_f)


def compute_benchmark_deltas(
    base_scores: Mapping[str, object],
    tuned_scores: Mapping[str, object],
    *,
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD,
) -> List[BenchmarkDelta]:
    """Turn two ``{benchmark: score}`` maps into sorted ``BenchmarkDelta``s.

    Only benchmarks present in BOTH maps are compared (a one-sided benchmark
    cannot produce a delta). Pure — no tracker / model coupling.
    """
    if not isinstance(base_scores, Mapping):
        raise TypeError("base_scores must be a mapping")
    if not isinstance(tuned_scores, Mapping):
        raise TypeError("tuned_scores must be a mapping")
    threshold = _validate_threshold(forgetting_threshold)

    common = sorted(set(base_scores) & set(tuned_scores))
    deltas: List[BenchmarkDelta] = []
    for name in common:
        base_f = _validate_score(base_scores[name], f"base[{name!r}]")
        tuned_f = _validate_score(tuned_scores[name], f"tuned[{name!r}]")
        deltas.append(
            BenchmarkDelta(
                name=str(name),
                base=base_f,
                tuned=tuned_f,
                delta=round(tuned_f - base_f, _DELTA_ROUND),
                regressed=_is_regressed(base_f, tuned_f, threshold),
            )
        )
    return deltas


# ---------------------------------------------------------------------------
# decide_ship — the moat (pure fn)
# ---------------------------------------------------------------------------

def decide_ship(
    task_win: TaskWin,
    benchmark_deltas: Sequence[BenchmarkDelta],
    *,
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD,
    soup_version: str = __version__,
) -> ShipVerdict:
    """Fuse leg 1 + leg 2 into a single SHIP / DON'T-SHIP verdict.

    ``decide_ship`` is the SINGLE source of truth for the threshold: it
    recomputes each benchmark's ``regressed`` flag against
    ``forgetting_threshold`` (so a delta built with a stale threshold is
    corrected) and re-applies the STRICT leg-1 inequality from the raw scores.

    Decision rule::

        SHIP  <=>  task_win.tuned > task_win.base
              AND  benchmark_deltas is non-empty
              AND  no benchmark regressed

    Precedence when more than one rule fails (most decisive first):
      1. ``missing_baseline`` — leg 2 had nothing to compare (refuse, never
         silently SHIP),
      2. ``task_win`` — the task did not strictly improve (incl. a tie),
      3. ``regression`` — a general benchmark dropped past the threshold.
    """
    if not isinstance(task_win, TaskWin):
        raise TypeError("task_win must be a TaskWin instance")
    deltas_in = list(benchmark_deltas)
    for item in deltas_in:
        if not isinstance(item, BenchmarkDelta):
            raise TypeError("benchmark_deltas must contain BenchmarkDelta items")
    threshold = _validate_threshold(forgetting_threshold)

    # Canonical deltas: recompute at THIS threshold so the verdict is internally
    # consistent regardless of how the inputs were built.
    canonical: Tuple[BenchmarkDelta, ...] = tuple(
        BenchmarkDelta(
            name=item.name,
            base=item.base,
            tuned=item.tuned,
            delta=round(item.tuned - item.base, _DELTA_ROUND),
            regressed=_is_regressed(item.base, item.tuned, threshold),
        )
        for item in deltas_in
    )

    leg1_pass = task_win.tuned > task_win.base
    any_regressed = any(item.regressed for item in canonical)

    if not canonical:
        decision, failed = DECISION_DONT_SHIP, FAILED_MISSING_BASELINE
    elif not leg1_pass:
        decision, failed = DECISION_DONT_SHIP, FAILED_TASK_WIN
    elif any_regressed:
        decision, failed = DECISION_DONT_SHIP, FAILED_REGRESSION
    else:
        decision, failed = DECISION_SHIP, None

    return ShipVerdict(
        decision=decision,
        task_win=task_win,
        benchmark_deltas=canonical,
        failed_rule=failed,
        forgetting_threshold=threshold,
        soup_version=soup_version,
    )


# ---------------------------------------------------------------------------
# Reason strings (shared by rubric + panel)
# ---------------------------------------------------------------------------

def _regressed_names(verdict: ShipVerdict) -> List[str]:
    return [item.name for item in verdict.benchmark_deltas if item.regressed]


def _failed_rule_explanation(verdict: ShipVerdict) -> str:
    """One-line, human-readable reason for the failed rule (or a SHIP note)."""
    if verdict.failed_rule is None:
        return (
            "Task improved AND no general benchmark regressed past "
            f"{verdict.forgetting_threshold:.2%}."
        )
    if verdict.failed_rule == FAILED_MISSING_BASELINE:
        return (
            "No general-suite benchmarks were measured — cannot verify the "
            "model didn't regress. Supply a general suite or --baseline."
        )
    if verdict.failed_rule == FAILED_TASK_WIN:
        win = verdict.task_win
        verb = "tied" if win.tuned == win.base else "got worse"
        return (
            f"Task did not improve ({win.mode}: {win.base:.4f} -> "
            f"{win.tuned:.4f}, {verb}). A strict win is required to ship."
        )
    if verdict.failed_rule == FAILED_REGRESSION:
        names = _regressed_names(verdict)
        joined = ", ".join(names) if names else "(unknown)"
        return (
            f"General benchmark(s) regressed past "
            f"{verdict.forgetting_threshold:.2%}: {joined}. "
            "Catastrophic forgetting — DON'T SHIP even though the task improved."
        )
    return f"Unrecognised failed rule: {verdict.failed_rule!r}"


# ---------------------------------------------------------------------------
# Rendering — plain text (for --output / clipboard) and Rich (for the terminal)
# ---------------------------------------------------------------------------

def format_ship_rubric(verdict: ShipVerdict) -> str:
    """Plain-text, one-screen verdict (stable output; NO Rich markup applied).

    The output may contain user-controlled benchmark names verbatim. Callers
    that render this through a Rich ``Console`` MUST ``rich.markup.escape`` it
    first — the terminal path uses :func:`render_ship_panel` (which escapes);
    this function is for files / clipboards / plain stdout.
    """
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    win = verdict.task_win
    won_str = "won" if win.won else "no win"
    parts: List[str] = []
    parts.append(f"Decision:    {verdict.decision}")
    parts.append("")
    parts.append(
        f"Leg 1 task win ({win.mode}): "
        f"{win.base:.4f} -> {win.tuned:.4f}  [{won_str}]"
    )
    parts.append(
        f"Leg 2 general suite (forgetting_threshold {verdict.forgetting_threshold:.2%}):"
    )
    if verdict.benchmark_deltas:
        for item in verdict.benchmark_deltas:
            flag = "REGRESSED" if item.regressed else "ok"
            parts.append(
                f"  {item.name:<24} {item.base:.4f} -> {item.tuned:.4f}  "
                f"{item.delta:+.4f}  [{flag}]"
            )
    else:
        parts.append("  (no benchmarks measured)")
    parts.append("")
    if verdict.failed_rule is not None:
        parts.append(f"Failed rule: {verdict.failed_rule}")
    parts.append(f"Reason:      {_failed_rule_explanation(verdict)}")
    return "\n".join(parts)


def _decision_style(decision: str) -> str:
    return "green" if decision == DECISION_SHIP else "red"


def render_ship_panel(verdict: ShipVerdict) -> Panel:
    """Rich ``Panel`` for the terminal — a one-screen verdict card."""
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    from rich.markup import escape

    style = _decision_style(verdict.decision)
    win = verdict.task_win
    won_str = "won" if win.won else "no win"

    header = (
        f"[bold {style}]{verdict.decision}[/]\n"
        f"Leg 1 task win ([cyan]{escape(win.mode)}[/]): "
        f"{win.base:.4f} -> {win.tuned:.4f}  [{won_str}]"
    )

    table = Table(
        title=f"Leg 2 general suite (threshold {verdict.forgetting_threshold:.2%})",
        title_justify="left",
        expand=False,
    )
    table.add_column("Benchmark", style="bold")
    table.add_column("Base", justify="right")
    table.add_column("Tuned", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Verdict")
    if verdict.benchmark_deltas:
        for item in verdict.benchmark_deltas:
            flag = "[red]REGRESSED[/]" if item.regressed else "[green]ok[/]"
            table.add_row(
                escape(item.name),
                f"{item.base:.4f}",
                f"{item.tuned:.4f}",
                f"{item.delta:+.4f}",
                flag,
            )
    else:
        table.add_row("[dim](none measured)[/]", "-", "-", "-", "[red]missing[/]")

    footer = f"[dim]{escape(_failed_rule_explanation(verdict))}[/]"
    body = Group(header, "", table, "", footer)
    return Panel(body, title="soup ship", border_style=style)


# ---------------------------------------------------------------------------
# GitHub PR comment (--push) — v0.71.39
# ---------------------------------------------------------------------------

def _longest_backtick_run(text: str) -> int:
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        if run > longest:
            longest = run
    return longest


def render_ship_pr_markdown(verdict: ShipVerdict) -> str:
    """Render the verdict as a GitHub-PR-comment Markdown body (v0.71.39).

    The verdict rubric (which contains user-controlled benchmark names) is
    wrapped in a fenced code block whose fence is chosen ONE backtick longer
    than the longest backtick run in the body — per CommonMark a code fence can
    only be closed by a fence of >= its own length, so a hostile benchmark name
    containing ``` can neither break out of the block nor inject Markdown.
    """
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    body = format_ship_rubric(verdict)
    fence = "`" * max(_MIN_MD_FENCE_LEN, _longest_backtick_run(body) + 1)
    emoji = "✅" if verdict.decision == DECISION_SHIP else "❌"
    return (
        f"## soup ship: {verdict.decision} {emoji}\n\n"
        f"{fence}\n{body}\n{fence}\n\n"
        f"<sub>Generated by <code>soup ship</code> v{verdict.soup_version}.</sub>\n"
    )


# ---------------------------------------------------------------------------
# Serialization (--output)
# ---------------------------------------------------------------------------

def verdict_to_evidence(
    verdict: ShipVerdict,
    *,
    provenance: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Project a verdict into the ``--evidence`` INPUT schema (the inverse read).

    ``commands/ship.py`` reads pre-computed evidence as
    ``{"task": {"mode", "base", "tuned"}, "benchmarks": {name: {"base", "tuned"}}}``.
    This is the missing serialiser that makes ``soup ship`` *output* replayable
    as *input* (#312): feeding the result back through ``--evidence`` with the
    same ``forgetting_threshold`` reproduces an identical verdict (decision +
    both legs + failed_rule). The threshold / decision / failed_rule are NOT
    stored — they are the verdict, recomputed by ``decide_ship`` on read — so a
    stale threshold on disk can never desync from the scores.

    ``provenance`` (optional) is attached verbatim under a ``"provenance"`` key.
    It is informational to the verdict (the reader ignores it), but the CI
    staleness gate uses ``provenance.config_sha`` to bind evidence to the exact
    config that produced it (v0.71.39).
    """
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    win = verdict.task_win
    evidence: Dict[str, object] = {
        "task": {"mode": win.mode, "base": win.base, "tuned": win.tuned},
        "benchmarks": {
            item.name: {"base": item.base, "tuned": item.tuned}
            for item in verdict.benchmark_deltas
        },
    }
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        evidence["provenance"] = dict(provenance)
    return evidence


def verdict_to_dict(verdict: ShipVerdict) -> Dict[str, object]:
    """JSON-serializable dict for ``--output`` / programmatic consumers."""
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    win = verdict.task_win
    return {
        "decision": verdict.decision,
        "task_win": {
            "mode": win.mode,
            "base": win.base,
            "tuned": win.tuned,
            "won": win.won,
        },
        "benchmark_deltas": [
            {
                "name": item.name,
                "base": item.base,
                "tuned": item.tuned,
                "delta": item.delta,
                "regressed": item.regressed,
            }
            for item in verdict.benchmark_deltas
        ],
        "failed_rule": verdict.failed_rule,
        "forgetting_threshold": verdict.forgetting_threshold,
        "soup_version": verdict.soup_version,
    }
