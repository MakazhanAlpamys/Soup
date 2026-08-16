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
from rich.markup import escape
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

# Strip C0 controls + DEL before any UNTRUSTED name reaches the terminal
# (mirrors ``commands/data_doctor._for_terminal``, and the reasoning is the
# same): ``rich.markup.escape`` neutralises Rich's own ``[...]`` tag syntax and
# nothing else, so a raw ESC byte survives it. Benchmark and noise-floor axis
# names come from an ``--evidence`` JSON file, which is untrusted input — a
# crafted name can spoof the terminal title, emit OSC-8 links, or use cursor
# tricks to obscure a DON'T-SHIP verdict. Tab / LF / CR are kept because Rich
# lays them out safely. ``--output`` JSON is unaffected: ``json.dumps`` already
# ``\\u00XX``-escapes control characters.
_CONTROL_STRIP_TABLE: Dict[int, None] = {
    i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)
}
_CONTROL_STRIP_TABLE[0x7F] = None


def for_terminal(text: str) -> str:
    """Strip control bytes ``rich.markup.escape`` does not handle."""
    return str(text).translate(_CONTROL_STRIP_TABLE)

# ---------------------------------------------------------------------------
# Noise floor (v0.73.2) — what the instrument can actually resolve
# ---------------------------------------------------------------------------
#
# Greedy decoding is not deterministic on GPU. Measured on an H100: the same
# model, no adapter, five runs, spread **0.015 strict / 0.020 format-blind** —
# and `soup ship` compared against a 0.05 threshold without ever telling the
# operator what its own instrument could resolve. Four of six paired deltas in
# that session sat inside the floor.
#
# CAVEAT, carried because it bounds the claim: n=3, one model, one dataset. It
# SIZES the effect; it does not calibrate a threshold.

#: Axis key under which the leg-1 task metric's floor is stored, so one
#: ``NoiseFloor`` covers both legs. Double-underscored to keep it out of the
#: benchmark namespace by convention — note this is NOT enforced anywhere:
#: ``_parse_suite`` accepts any name, and ``--general-suite __task__`` is
#: harmless only because it then fails downstream as an unscoreable benchmark.
TASK_AXIS = "__task__"

#: A floor needs a spread, and a spread needs at least two samples.
MIN_NOISE_FLOOR_RUNS = 2
#: Each run is a full pass over the base model; ten is already expensive.
MAX_NOISE_FLOOR_RUNS = 10

# Input hygiene on an evidence-supplied floors mapping. These mirror
# ``commands/ship._MAX_SUITE_BENCHMARKS`` / ``_MAX_BENCHMARK_NAME_CHARS`` so the
# file-read path is bounded the same way the CLI flag path is.
_MAX_FLOOR_AXES = 50
_MAX_FLOOR_AXIS_CHARS = 256


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
class NoiseFloor:
    """What the instrument can resolve, measured by repeating the BASE run.

    ``floors`` is ``(axis, spread)`` pairs sorted by axis, where the spread is
    ``max - min`` of that axis's score across the repeats. A tuple of pairs
    rather than a dict so the dataclass stays genuinely immutable, mirroring
    ``ShipVerdict.benchmark_deltas``.

    ``judge_inclusive`` marks the leg-1 task axis (``TASK_AXIS``) floor as
    combined decode + judge noise rather than decode-only. It is set for
    ``--task-mode judge_score`` / ``pairwise``, where the repeats fold the
    judge's own sampling noise into the spread (#403). The leg-2 axes are always
    decode-only, so this flag scopes to the one axis that can carry judge noise;
    a reader must not mistake a judge-inclusive floor for a decode-only one.
    Defaults ``False`` so the metric-mode path stays byte-identical.
    """

    runs: int
    floors: Tuple[Tuple[str, float], ...]
    judge_inclusive: bool = False

    def of(self, axis: str) -> float:
        """This axis's measured floor, or ``0.0`` if it was never measured.

        An unmeasured axis MUST read 0.0 rather than inherit another axis's
        spread — inheriting would silently suppress a real regression on a
        benchmark the repeat never covered.
        """
        for name, value in self.floors:
            if name == axis:
                return value
        return 0.0

    def as_dict(self) -> Dict[str, float]:
        """The floors as a plain mapping (for JSON / display)."""
        return {name: value for name, value in self.floors}


@dataclass(frozen=True)
class ShipVerdict:
    """The binary decision plus the evidence that produced it."""

    decision: str  # DECISION_SHIP | DECISION_DONT_SHIP
    task_win: TaskWin
    benchmark_deltas: Tuple[BenchmarkDelta, ...]
    failed_rule: Optional[str]  # None on SHIP; a FAILED_* code on DON'T SHIP
    forgetting_threshold: float
    soup_version: str
    #: Set only when ``--noise-floor`` measured one; ``None`` keeps the v0.73.1
    #: path byte-identical.
    noise_floor: Optional[NoiseFloor] = None


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

def compute_noise_floor(
    runs: Sequence[Mapping[str, object]], *, judge_inclusive: bool = False
) -> NoiseFloor:
    """Measure the instrument's resolution from repeated BASE-model runs.

    Each element of ``runs`` is one ``{axis: score}`` map from an independent
    repeat of the same unchanged model. The floor of an axis is the spread
    (``max - min``) across those repeats — anything smaller than that is
    something this instrument cannot distinguish from itself.

    ``judge_inclusive`` stamps the result as combined decode + judge noise (the
    leg-1 task axis was measured through a judge); see ``NoiseFloor``.

    Refuses fewer than two runs (one sample has no spread, and reporting 0.0
    would present an unmeasured floor as a measured one) and refuses a ragged
    set (an axis missing from any run would be averaged over fewer samples
    than the caller believes).
    """
    runs_list = list(runs)
    if len(runs_list) < MIN_NOISE_FLOOR_RUNS:
        raise ValueError(
            f"a noise floor needs at least {MIN_NOISE_FLOOR_RUNS} runs, "
            f"got {len(runs_list)}"
        )
    for run in runs_list:
        if not isinstance(run, Mapping):
            raise TypeError("each noise-floor run must be a mapping")

    axes = set(runs_list[0])
    for index, run in enumerate(runs_list[1:], start=1):
        if set(run) != axes:
            raise ValueError(
                "every run must score the same axes; "
                f"run {index} differs from run 0"
            )

    floors: List[Tuple[str, float]] = []
    for axis in sorted(axes):
        values = [
            _validate_score(run[axis], f"noise-floor run[{axis!r}]")
            for run in runs_list
        ]
        floors.append((str(axis), round(max(values) - min(values), _DELTA_ROUND)))
    return NoiseFloor(
        runs=len(runs_list),
        floors=tuple(floors),
        judge_inclusive=judge_inclusive,
    )


def _floor_of(noise_floor: Optional[NoiseFloor], axis: str) -> float:
    """This axis's floor, or 0.0 when no floor was measured at all."""
    if noise_floor is None:
        return 0.0
    if not isinstance(noise_floor, NoiseFloor):
        raise TypeError("noise_floor must be a NoiseFloor instance")
    return noise_floor.of(axis)


def build_task_win(
    mode: str,
    base: object,
    tuned: object,
    *,
    noise_floor: Optional[NoiseFloor] = None,
) -> TaskWin:
    """Build a leg-1 ``TaskWin``. ``won`` is the STRICT inequality tuned > base.

    With a ``noise_floor``, the margin must also clear what the instrument can
    resolve on ``TASK_AXIS``: a win of 0.004 measured on an instrument with a
    0.020 floor is not a win, it is the instrument. Without one the behaviour
    is byte-identical to v0.73.1.
    """
    if mode not in TASK_MODES:
        raise ValueError(f"mode must be one of {TASK_MODES}, got {mode!r}")
    base_f = _validate_score(base, "task base")
    tuned_f = _validate_score(tuned, "task tuned")
    floor = _floor_of(noise_floor, TASK_AXIS)
    return TaskWin(
        mode=mode, base=base_f, tuned=tuned_f, won=tuned_f > base_f + floor
    )


def compute_benchmark_deltas(
    base_scores: Mapping[str, object],
    tuned_scores: Mapping[str, object],
    *,
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD,
    noise_floor: Optional[NoiseFloor] = None,
) -> List[BenchmarkDelta]:
    """Turn two ``{benchmark: score}`` maps into sorted ``BenchmarkDelta``s.

    Only benchmarks present in BOTH maps are compared (a one-sided benchmark
    cannot produce a delta). Pure — no tracker / model coupling.

    With a ``noise_floor``, each axis is tested against
    ``max(forgetting_threshold, floor_for_that_axis)``. The ``max`` is
    load-bearing in both directions: a measured floor ABOVE the threshold
    widens the gate to what the instrument can actually resolve, and a floor
    BELOW it must never tighten the gate behind the operator's back.
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
        effective = max(threshold, _floor_of(noise_floor, str(name)))
        deltas.append(
            BenchmarkDelta(
                name=str(name),
                base=base_f,
                tuned=tuned_f,
                delta=round(tuned_f - base_f, _DELTA_ROUND),
                regressed=_is_regressed(base_f, tuned_f, effective),
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
    noise_floor: Optional[NoiseFloor] = None,
) -> ShipVerdict:
    """Fuse leg 1 + leg 2 into a single SHIP / DON'T-SHIP verdict.

    ``decide_ship`` is the SINGLE source of truth for the threshold: it
    recomputes each benchmark's ``regressed`` flag against
    ``forgetting_threshold`` (so a delta built with a stale threshold is
    corrected) and re-applies the STRICT leg-1 inequality from the raw scores.

    That recomputation is why ``noise_floor`` has to be passed HERE as well as
    to the builders: a floor applied only in ``compute_benchmark_deltas`` /
    ``build_task_win`` would be silently undone one line later.

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
            regressed=_is_regressed(
                item.base,
                item.tuned,
                max(threshold, _floor_of(noise_floor, item.name)),
            ),
        )
        for item in deltas_in
    )

    leg1_pass = task_win.tuned > task_win.base + _floor_of(noise_floor, TASK_AXIS)
    # Canonicalise the task win for the SAME reason the deltas are canonicalised
    # above: the panel renders ``verdict.task_win.won``, so a TaskWin built
    # without the floor would print "won" beside a DON'T SHIP decided with it.
    canonical_win = (
        task_win
        if task_win.won == leg1_pass
        else TaskWin(
            mode=task_win.mode,
            base=task_win.base,
            tuned=task_win.tuned,
            won=leg1_pass,
        )
    )
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
        task_win=canonical_win,
        benchmark_deltas=canonical,
        failed_rule=failed,
        forgetting_threshold=threshold,
        soup_version=soup_version,
        noise_floor=noise_floor,
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
        floor = _floor_of(verdict.noise_floor, TASK_AXIS)
        if floor > 0.0 and win.tuned > win.base:
            # The task DID go up — just not by more than the instrument can
            # resolve. Saying "got worse" here would be plainly false, and this
            # is a gate whose whole point is not to state more than it measured.
            return (
                f"Task improved ({win.mode}: {win.base:.4f} -> {win.tuned:.4f}, "
                f"+{win.tuned - win.base:.4f}) but by less than the measured "
                f"noise floor of {floor:.4f}, so the gain is not distinguishable "
                "from run-to-run variation."
            )
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
    style = _decision_style(verdict.decision)
    win = verdict.task_win
    won_str = "won" if win.won else "no win"

    header = (
        f"[bold {style}]{verdict.decision}[/]\n"
        f"Leg 1 task win ([cyan]{escape(win.mode)}[/]): "
        # escape(): a bare "[no win]" is valid Rich markup for an unknown tag,
        # so the panel silently ATE its own leg-1 marker on every run up to and
        # including v0.73.1. The plain-text rubric, which has no markup parser,
        # printed it correctly the whole time — which is why it went unnoticed.
        f"{win.base:.4f} -> {win.tuned:.4f}  {escape('[' + won_str + ']')}"
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
                escape(for_terminal(item.name)),
                f"{item.base:.4f}",
                f"{item.tuned:.4f}",
                f"{item.delta:+.4f}",
                flag,
            )
    else:
        table.add_row("[dim](none measured)[/]", "-", "-", "-", "[red]missing[/]")

    footer = f"[dim]{escape(_failed_rule_explanation(verdict))}[/]"
    parts = [header, "", table, ""]
    if verdict.noise_floor is not None:
        parts.extend([
            _render_noise_floor(verdict.noise_floor),
            f"[dim]Measured over {verdict.noise_floor.runs} base repeats. Each "
            "axis is gated at max(threshold, its floor). A floor sizes this "
            "instrument's resolution; it does not calibrate a threshold.[/]",
            "",
        ])
    parts.append(footer)
    body = Group(*parts)
    return Panel(body, title="soup ship", border_style=style)


def _render_noise_floor(floor: NoiseFloor) -> Table:
    """The measured instrument resolution, printed beside the verdict.

    The point of showing it is that a threshold the operator chose can be
    smaller than what the run can resolve; the caveat line says so out loud
    rather than letting a printed number read as a calibrated one.
    """
    # The title is kept SHORT because Rich wraps it to the table's own width,
    # and this table is two narrow columns — "Noise floor (2 base repeats)"
    # rendered across two lines. The run count moves to the caption line below.
    table = Table(title="Noise floor", title_justify="left", expand=False)
    table.add_column("Axis", style="bold")
    table.add_column("Floor", justify="right")
    if floor.floors:
        for name, value in floor.floors:
            if name == TASK_AXIS:
                # A judge-inclusive floor mixes decode noise with the judge's
                # own sampling noise; say so, so it is never read as the
                # decode-only number the leg-2 axes report (#403).
                label = (
                    "leg 1 task (decode + judge)"
                    if floor.judge_inclusive
                    else "leg 1 task"
                )
            else:
                label = name
            table.add_row(escape(for_terminal(label)), f"{value:.4f}")
    else:
        table.add_row("[dim](none measured)[/]", "-")
    return table


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
    # A verdict decided against a measured floor does NOT replay without it —
    # the offline reader would recompute both legs at the bare threshold and
    # could return the opposite decision. #312's property is that the output is
    # replayable as input, so the floor is part of the evidence, not decoration.
    if verdict.noise_floor is not None:
        floor_block: Dict[str, object] = {
            "runs": verdict.noise_floor.runs,
            "floors": verdict.noise_floor.as_dict(),
        }
        # Emitted only when True so a decode-only floor round-trips byte-for-byte
        # (every pre-#403 evidence file). A judge-inclusive floor must carry the
        # marker or the reader would replay it as decode-only.
        if verdict.noise_floor.judge_inclusive:
            floor_block["judge_inclusive"] = True
        evidence["noise_floor"] = floor_block
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        evidence["provenance"] = dict(provenance)
    return evidence


def noise_floor_from_evidence(payload: object) -> Optional[NoiseFloor]:
    """Parse an evidence ``noise_floor`` block back into a ``NoiseFloor``.

    Returns ``None`` when the key is absent (every pre-v0.73.2 evidence file).
    Raises ``ValueError`` on a malformed block rather than dropping it: a floor
    silently discarded on read would replay as a DIFFERENT verdict, which is
    the exact failure the round-trip exists to prevent.

    SECURITY. An evidence file is untrusted input, and a floor WIDENS the gate:
    ``"floors": {"mini_mmlu": 1.0}`` masks any possible drop on that axis. This
    does not cross a new trust boundary — anyone who can edit the file can
    already write ``{"base": 0.9, "tuned": 0.9}`` and force a SHIP outright —
    but it is a far quieter edit to miss in review, so two things are done
    about it. Values are bounded to ``[0, 1]`` here (a score delta cannot
    exceed the metric's own range), and ``floor_exceeds_threshold`` lets every
    reader announce a widening floor exactly as the live measurement path does.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("evidence.noise_floor must be an object")
    runs = payload.get("runs")
    if isinstance(runs, bool) or not isinstance(runs, int):
        raise ValueError("evidence.noise_floor.runs must be an integer")
    if not (MIN_NOISE_FLOOR_RUNS <= runs <= MAX_NOISE_FLOOR_RUNS):
        raise ValueError(
            f"evidence.noise_floor.runs must be in "
            f"[{MIN_NOISE_FLOOR_RUNS}, {MAX_NOISE_FLOOR_RUNS}]"
        )
    raw = payload.get("floors")
    if not isinstance(raw, Mapping):
        raise ValueError("evidence.noise_floor.floors must be an object")
    if len(raw) > _MAX_FLOOR_AXES:
        raise ValueError(
            f"evidence.noise_floor.floors has too many axes (max {_MAX_FLOOR_AXES})"
        )
    floors: List[Tuple[str, float]] = []
    for axis in sorted(raw):
        name = str(axis)
        if len(name) > _MAX_FLOOR_AXIS_CHARS:
            raise ValueError(
                f"evidence.noise_floor.floors axis names must be "
                f"< {_MAX_FLOOR_AXIS_CHARS} chars"
            )
        value = _validate_score(raw[axis], f"evidence.noise_floor.floors[{name!r}]")
        if not (0.0 <= value <= 1.0):
            raise ValueError("a noise floor must be in [0.0, 1.0]")
        floors.append((name, value))
    judge_inclusive = payload.get("judge_inclusive", False)
    if not isinstance(judge_inclusive, bool):
        raise ValueError("evidence.noise_floor.judge_inclusive must be a boolean")
    return NoiseFloor(
        runs=runs, floors=tuple(floors), judge_inclusive=judge_inclusive
    )


def floor_exceeds_threshold(
    noise_floor: Optional[NoiseFloor], threshold: float
) -> List[Tuple[str, float]]:
    """Axes whose floor is WIDER than ``threshold``, sorted by name.

    A floor above the operator's threshold loosens the gate on that axis. That
    is intended — a threshold finer than the instrument is not a threshold —
    but it must never be silent, whether the floor was measured live or read
    out of an evidence file. Both readers call this so neither can be the quiet
    one.
    """
    if noise_floor is None:
        return []
    return sorted(
        (name, value) for name, value in noise_floor.floors if value > threshold
    )


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
        "noise_floor": (
            None
            if verdict.noise_floor is None
            else {
                "runs": verdict.noise_floor.runs,
                "floors": verdict.noise_floor.as_dict(),
                # Always present so a programmatic reader can tell a judge-
                # inclusive floor from a decode-only one without inference (#403).
                "judge_inclusive": verdict.noise_floor.judge_inclusive,
            }
        ),
    }
