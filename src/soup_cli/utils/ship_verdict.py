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
- Frozen dataclasses: ``TaskWin``, ``BenchmarkDelta``, ``AdapterChange``,
  ``ShipVerdict``, and the detail of the task leg: ``TaskDetail`` (``TaskRows``,
  ``Truncation``, ``GenerationFailures``) and ``Judged``.
- ``LegScore``: a float that also carries the per-row scores and flags it was built from.
- Constants: ``EVIDENCE_SCHEMA_FIELDS``, ``SCHEMA_VERSION``, ``TASK_MODES``,
  ``SUPPORTED_TASK_MODES``, ``DECISION_SHIP`` / ``DECISION_DONT_SHIP``, the
  ``FAILED_*`` rule codes, ``DEFAULT_FORGETTING_THRESHOLD``,
  ``ADAPTER_CHANGE_WARN_BELOW``.
- Pure functions: ``build_task_win``, ``compute_benchmark_deltas``,
  ``decide_ship`` (the moat), ``render_ship_panel``, ``format_ship_rubric``,
  ``verdict_from_evidence`` (the canonical evidence reader),
  ``verdict_to_dict``, ``verdict_to_evidence`` (the inverse of the ``--evidence``
  reader — makes ``soup ship`` output replayable as input, #312),
  ``numerics_family`` / ``numerics_from_evidence`` (the #367 judge-numerics stamp).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli import __version__

# Declared in the dependency-free leaf so the schema can import the bound
# without this module (#780); re-exported so there is only one object.
from soup_cli.utils.config_bounds import MAX_NOISE_FLOOR_RUNS, MIN_NOISE_FLOOR_RUNS
from soup_cli.utils.terminal import strip_control as for_terminal

if TYPE_CHECKING:  # typing only: the row module imports this one
    from soup_cli.utils.ship_task_rows import TaskRow

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# One registry for the evidence fields understood by the canonical reader and
# emitted by ``verdict_to_evidence``. Parity tests must exercise every field so
# a future optional addition cannot be silently ignored by both public readers.
# ``provenance`` is intentionally opaque metadata, so its nested keys are not
# part of the verdict schema.
#
# Schema version 2 (soup-cli 0.78) added ``schema_version``, ``judged`` and ``task_rows`` (the
# name of the per-row file, see ``utils/ship_task_rows.py``) at the root and
# ``rows`` / ``truncated`` / ``generation_failures`` in the task; their own keys are the
# ``rows`` ... ``judged`` sections. ``V2_ONLY_FIELDS`` lists the paths a reader of version 1 does
# not know: the reader refuses them under version 1.
EVIDENCE_SCHEMA_FIELDS: Mapping[str, FrozenSet[str]] = MappingProxyType({
    "root": frozenset({
        "task", "benchmarks", "noise_floor", "provenance", "numerics", "schema_version", "judged",
        "task_rows",
    }),
    "task": frozenset({"mode", "base", "tuned", "rows", "truncated", "generation_failures"}),
    "benchmark": frozenset({"base", "tuned"}),
    "noise_floor": frozenset({"runs", "floors", "judge_inclusive"}),
    "rows": frozenset({"better", "worse", "same", "n"}),
    "truncated": frozenset({"base", "tuned", "n", "max_new_tokens"}),
    "generation_failures": frozenset({"base", "tuned", "n"}),
    "judged": frozenset({"base", "adapter", "tuned", "base_adapter", "checkpoint_step"}),
})

#: ``--output`` and ``--emit-evidence`` carry ``schema_version``. Absent means 1 (soup-cli 0.77.x).
#: Version 2 adds ``task_win.rows`` / ``truncated`` / ``generation_failures`` and ``judged``.
#: A reader of version 1 must treat a file with a ``schema_version`` it does not know as
#: unreadable (soup-cli 0.77.x does: ``unsupported field(s): 'schema_version'``, exit 3).
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS: Tuple[int, ...] = (1, 2)

#: Evidence paths introduced by version 2: (section, key).
V2_ONLY_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("root", "judged"),
    ("root", "task_rows"),
    ("task", "rows"),
    ("task", "truncated"),
    ("task", "generation_failures"),
)

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
FAILED_ADAPTER_NOOP = "adapter_noop"  # every lora_B is zero: tuned == base, nothing ran

# Default forgetting threshold — 0.05 ABSOLUTE points, mirroring
# ``EvalGateConfig.regression_threshold`` and ``run_gate`` semantics.
DEFAULT_FORGETTING_THRESHOLD = 0.05

# Below this changed fraction ``soup ship --adapter`` adds a warning line (#2762).
# A display line, not a rule: the calibration table in docs/evaluation.md has
# near-zero-learning-rate controls up to 6.5e-3 and early checkpoints of real runs
# from 2.5e-3, so no fraction separates them and the verdict never depends on it.
ADAPTER_CHANGE_WARN_BELOW = 1e-2

# Task-leg detail: what a SHIP / DON'T SHIP is made of, shown beside the verdict and never part
# of the decision.

#: Name of the one warning the detail can raise. It is information: the verdict, ``failed_rule``
#: and the exit code are the same with and without it.
WARNING_TRUNCATION_GROWTH = "truncation_growth"

#: ``truncation_growth`` fires when the tuned model's cut-off answers exceed the base's by at
#: least ``max(TRUNCATION_GROWTH_MIN_ROWS, ceil(TRUNCATION_GROWTH_MIN_SHARE * n))`` rows. One or
#: two rows never fire (greedy decoding on a GPU is not bit-stable, #376); a fall never fires.
#: A display constant like the changed-fraction bound, NOT calibrated: no table of real runs
#: exists yet, and it stays a warning until one does. A reported run 5 -> 14 of 64: threshold 7,
#: rise 9, fires.
TRUNCATION_GROWTH_MIN_ROWS = 3
TRUNCATION_GROWTH_MIN_SHARE = 0.10

#: Two per-row scores closer than this are the same row (float noise is not a change).
ROW_SCORE_EPS = 1e-9

#: What a row is, tuned against base (``row_change``); the per-row file stores one per row.
CHANGE_BETTER = "better"
CHANGE_WORSE = "worse"
CHANGE_SAME = "same"
ROW_CHANGES: Tuple[str, ...] = (CHANGE_BETTER, CHANGE_WORSE, CHANGE_SAME)

#: The file ``soup ship`` writes next to ``--output`` is ``<stem>`` + this suffix, and the verdict
#: JSON and the evidence name it (``task_rows``) by that bare file name.
TASK_ROWS_SUFFIX = ".task-rows.jsonl"
_MAX_TASK_ROWS_NAME_CHARS = 255

# Bounds on an evidence-supplied detail block (a task file holds at most 10 000 rows, so a count
# or an ``n`` above that cannot describe a real run). ``_MAX_EVIDENCE_ROWS`` equals
# ``eval.custom.MAX_EVAL_TASKS``; it is not imported (this module stays light) and a test pins
# the equality.
_MAX_EVIDENCE_ROWS = 10_000
#: The most rows one task leg (and so one per-row file) can have: the task-file bound.
MAX_TASK_ROWS = _MAX_EVIDENCE_ROWS
_MAX_EVIDENCE_NEW_TOKENS = 1 << 20
_MAX_JUDGED_CHARS = 4096  # a path
_MAX_CHECKPOINT_STEP = 10**15

#: ``checkpoint-<digits>`` exactly: ``\Z`` (``$`` also accepts a trailing newline), ASCII digits
#: only, and a bounded digit run so ``int()`` never sees an unbounded string.
_CHECKPOINT_RE = re.compile(r"checkpoint-([0-9]{1,15})\Z")
_PATH_SEPARATORS = re.compile(r"[\\/]")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")

# Canonical live-eval load stamps (#367). Quantized loads use the BitsAndBytes
# format name; full-precision loads use the torch dtype that actually went to
# ``from_pretrained``. The staleness gate compares *family* (``4bit`` / ``8bit``
# / ``full``), so bf16, fp16 and fp32 runs of the same recipe are not stale
# relative to each other — the gate distinguishes quantized vs full loads.
KNOWN_NUMERICS: Tuple[str, ...] = ("4bit", "8bit", "bfloat16", "float16", "float32")
_QUANTIZED_NUMERICS = frozenset({"4bit", "8bit"})
_FULL_PRECISION_NUMERICS = frozenset({"bfloat16", "float16", "float32"})
NUMERICS_FAMILY_FULL = "full"

# Float-noise tolerance so an exactly-at-threshold drop reads as OK (a -5.00%
# drop must not flip to "regressed" just because 0.80 - 0.75 == 0.05000000004).
_REGRESSION_EPS = 1e-9

# Round stored deltas so JSON / display are clean; the regression test uses the
# RAW base/tuned (not the rounded delta), so this never changes a verdict.
_DELTA_ROUND = 6

# CommonMark's minimum code-fence length (used by render_ship_pr_markdown).
_MIN_MD_FENCE_LEN = 3

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

# Input hygiene on an evidence-supplied floors mapping. These mirror
# ``commands/ship._MAX_SUITE_BENCHMARKS`` / ``_MAX_BENCHMARK_NAME_CHARS`` so the
# file-read path is bounded the same way the CLI flag path is.
_MAX_FLOOR_AXES = 50
_MAX_FLOOR_AXIS_CHARS = 256


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

def truncation_growth_threshold(n: int) -> int:
    """Rows by which the tuned truncation count must exceed the base's to raise the warning.

    ``max(TRUNCATION_GROWTH_MIN_ROWS, ceil(TRUNCATION_GROWTH_MIN_SHARE * n))`` in exact
    arithmetic (a ``Fraction`` of the constant's decimal text), so no float rounding can move a
    boundary if the share is ever changed: ``0.07 * 100`` is ``7.000000000000001``, whose float
    ceiling is 8, not 7.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    share = Fraction(str(TRUNCATION_GROWTH_MIN_SHARE))
    return max(TRUNCATION_GROWTH_MIN_ROWS, math.ceil(share * n))


@dataclass(frozen=True)
class TaskRows:
    """Per-row comparison of the task leg: how many rows the tuned model got better / worse."""

    better: int
    worse: int
    same: int

    @property
    def n(self) -> int:
        return self.better + self.worse + self.same


@dataclass(frozen=True)
class Truncation:
    """Answers that stopped at the new-token cap, for the base and the tuned model.

    A count is ``None`` (never 0) when it was not measured: the generator returned a plain
    ``str`` for some row, or a row's generation failed. ``max_new_tokens`` is the cap the task
    leg generated with (``None`` when unknown).
    """

    base: Optional[int]
    tuned: Optional[int]
    n: int
    max_new_tokens: Optional[int] = None

    @property
    def warning(self) -> Optional[str]:
        """``truncation_growth`` when the tuned count exceeds the base's by the threshold.

        Computed, never stored: it needs both counts measured. Information only.
        """
        if self.base is None or self.tuned is None:
            return None
        if self.tuned - self.base >= truncation_growth_threshold(self.n):
            return WARNING_TRUNCATION_GROWTH
        return None


@dataclass(frozen=True)
class GenerationFailures:
    """Rows whose generation raised and was swallowed (scored as wrong), per side."""

    base: int
    tuned: int
    n: int


@dataclass(frozen=True)
class TaskDetail:
    """What the task leg's number is made of. Every part is ``None`` when it was not measured."""

    rows: Optional[TaskRows] = None
    truncated: Optional[Truncation] = None
    generation_failures: Optional[GenerationFailures] = None
    #: One record per task (prompts, answers, scores) for the per-row file next to ``--output``.
    #: User data, kept in memory only until that file is written: it is not compared, not shown
    #: and not part of ``verdict_to_dict`` / ``verdict_to_evidence`` / the PR comment, and a
    #: replay of evidence has none.
    records: "Optional[Tuple[TaskRow, ...]]" = field(default=None, compare=False, repr=False)

    def is_empty(self) -> bool:
        return (
            self.rows is None and self.truncated is None and self.generation_failures is None
        )


@dataclass(frozen=True)
class Judged:
    """What was judged: the strings of the command line, and the checkpoint step they name."""

    base: str
    adapter: Optional[str] = None
    tuned: Optional[str] = None
    base_adapter: Optional[str] = None
    checkpoint_step: Optional[int] = None


class LegScore(float):
    """A leg-1 score (a ``float`` in every other respect) that remembers what it came from.

    The scorers hand back a plain-looking float, so every caller (the noise floor, a test that
    stubs a scorer with a bare float) keeps working; ``build_task_detail`` reads the extra
    attributes when they are there. ``row_scores`` is one score per task, in task order.
    ``truncated`` is one flag per generated answer (``True`` / ``False`` / ``None`` for "the
    generator could not tell", see ``GeneratedText``); a pairwise leg generates for two sides,
    so ``truncated`` holds side A's flags and ``truncated_other`` side B's. ``failures`` counts
    the generations that raised and were swallowed. For the per-row file a leg also keeps the
    ``answers`` it was scored on, which of them ``failed_rows``, and the ``tasks`` behind them
    (all in task order); ``value`` is then the ``row_mean`` of ``row_scores``, so the file's
    mean is the reported number.
    """

    row_scores: Optional[Tuple[float, ...]]
    truncated: Optional[Tuple[Optional[bool], ...]]
    truncated_other: Optional[Tuple[Optional[bool], ...]]
    failures: Optional[int]
    answers: Optional[Tuple[str, ...]]
    failed_rows: Optional[Tuple[bool, ...]]
    tasks: Optional[Tuple[object, ...]]

    def __new__(
        cls,
        value: float,
        *,
        row_scores: Optional[Sequence[float]] = None,
        truncated: Optional[Sequence[Optional[bool]]] = None,
        truncated_other: Optional[Sequence[Optional[bool]]] = None,
        failures: Optional[int] = None,
        answers: Optional[Sequence[str]] = None,
        failed_rows: Optional[Sequence[bool]] = None,
        tasks: Optional[Sequence[object]] = None,
    ) -> "LegScore":
        instance = super().__new__(cls, value)
        instance.row_scores = None if row_scores is None else tuple(row_scores)
        instance.truncated = None if truncated is None else tuple(truncated)
        instance.truncated_other = None if truncated_other is None else tuple(truncated_other)
        instance.failures = failures
        instance.answers = None if answers is None else tuple(answers)
        instance.failed_rows = None if failed_rows is None else tuple(failed_rows)
        instance.tasks = None if tasks is None else tuple(tasks)
        return instance


@dataclass(frozen=True)
class TaskWin:
    """Leg 1 — did the tuned model beat the base model on the target task?"""

    mode: str  # one of TASK_MODES
    base: float
    tuned: float
    won: bool  # tuned > base (STRICT)
    #: What the two numbers are made of (rows better / worse, cut-off answers, failed
    #: generations). ``None`` when the leg could not say (an evidence file of version 1, an
    #: adapter that changes nothing). Never part of the decision.
    detail: Optional[TaskDetail] = None


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
class AdapterChange:
    """How much of the base an adapter's merged update changes (#2762).

    ``changed`` of ``elements`` (the elements of the ``modules`` target modules)
    differ in value after the update is added in the base dtype. ``reason`` is set
    instead when the adapter or base could not be measured; the counts are then 0.
    """

    changed: int = 0
    elements: int = 0
    modules: int = 0
    base_dtypes: Tuple[str, ...] = ()
    reason: Optional[str] = None

    @classmethod
    def not_measured(cls, reason: str) -> "AdapterChange":
        return cls(reason=reason or "unknown reason")

    @property
    def measured(self) -> bool:
        return self.reason is None

    @property
    def fraction(self) -> Optional[float]:
        if not self.measured or self.elements <= 0:
            return None
        return self.changed / self.elements

    @property
    def warning(self) -> bool:
        fraction = self.fraction
        return fraction is not None and fraction < ADAPTER_CHANGE_WARN_BELOW


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
    #: Actual live-eval load precision (one of ``KNOWN_NUMERICS``). ``None`` on
    #: pre-#367 evidence that never stamped it — the panel then says
    #: ``unstamped`` rather than implying bf16.
    numerics: Optional[str] = None
    #: Set only by a live ``--adapter`` run that reached the evaluation (#2762).
    #: Display only: no rule reads it.
    adapter_change: Optional[AdapterChange] = None
    #: What was judged (the strings of the command line and the checkpoint step they name).
    #: ``None`` for a verdict that was not produced by a live run and whose evidence did not
    #: record it (a file of version 1). Never part of the decision.
    judged: Optional[Judged] = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def parse_numerics(value: object) -> str:
    """Validate a present numerics stamp. Does not echo unknown values."""
    if isinstance(value, str) and value in KNOWN_NUMERICS:
        return value
    if not isinstance(value, str):
        raise ValueError(f"numerics must be a string, got {type(value).__name__}")
    raise ValueError(
        "numerics must be one of " + ", ".join(KNOWN_NUMERICS)
    )


def numerics_family(numerics: str) -> str:
    """Map a stamp to the family the staleness gate compares.

    ``4bit`` and ``8bit`` stay themselves; ``bfloat16`` and ``float32`` collapse
    to ``full``. Unknown values raise — callers must ``parse_numerics`` first.
    """
    if numerics in _QUANTIZED_NUMERICS:
        return numerics
    if numerics in _FULL_PRECISION_NUMERICS:
        return NUMERICS_FAMILY_FULL
    raise ValueError(
        "numerics must be one of " + ", ".join(KNOWN_NUMERICS)
    )


def numerics_from_evidence(payload: object) -> Optional[str]:
    """Parse an evidence ``numerics`` stamp.

    Returns ``None`` when the key is absent (every pre-#367 evidence file).
    Raises ``ValueError`` on a present-but-malformed value rather than dropping
    it: a stamp silently discarded on read would let mismatched numerics replay
    as if they were never recorded.
    """
    if payload is None:
        return None
    return parse_numerics(payload)


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


def _validate_evidence_fields(
    payload: Mapping[object, object], section: str, path: str
) -> None:
    """Refuse unregistered verdict fields instead of silently ignoring them."""
    allowed = EVIDENCE_SCHEMA_FIELDS[section]
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        joined = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{path} has unsupported field(s): {joined}")


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


def _score_tuple(values: Sequence[object], name: str) -> Tuple[float, ...]:
    return tuple(_validate_score(value, f"{name}[{index}]") for index, value in enumerate(values))


def _flag_tuple(flags: Sequence[object], name: str) -> Tuple[Optional[bool], ...]:
    out: List[Optional[bool]] = []
    for index, flag in enumerate(flags):
        if flag is not None and not isinstance(flag, bool):
            raise TypeError(f"{name}[{index}] must be True, False or None")
        out.append(flag)
    return tuple(out)


def row_change(base_score: float, tuned_score: float) -> str:
    """One row, tuned against base: better iff ``tuned > base + ROW_SCORE_EPS``, worse iff
    ``tuned < base - ROW_SCORE_EPS``, the same otherwise. The one definition of the three
    classes: ``compare_rows`` counts them and the per-row file stores them."""
    if tuned_score > base_score + ROW_SCORE_EPS:
        return CHANGE_BETTER
    if tuned_score < base_score - ROW_SCORE_EPS:
        return CHANGE_WORSE
    return CHANGE_SAME


def compare_rows(
    base_scores: Sequence[object], tuned_scores: Sequence[object]
) -> TaskRows:
    """How many rows the tuned model got better / worse / the same, from per-row scores.

    The classes are those of ``row_change``, so ``better + worse + same`` is the row count.
    """
    base = _score_tuple(base_scores, "base row score")
    tuned = _score_tuple(tuned_scores, "tuned row score")
    if len(base) != len(tuned):
        raise ValueError("base and tuned need the same number of rows to be compared")
    changes = [row_change(base_score, tuned_score) for base_score, tuned_score in zip(base, tuned)]
    better = changes.count(CHANGE_BETTER)
    worse = changes.count(CHANGE_WORSE)
    return TaskRows(better=better, worse=worse, same=len(base) - better - worse)


def count_truncated(flags: Sequence[Optional[bool]]) -> Optional[int]:
    """Answers that hit the cap, from one flag per answer; ``None`` when any flag is unknown.

    One unmeasured answer makes the whole count unmeasured: a count that skipped it would read
    as measured over every row. ``None`` is never read as ``False``.
    """
    total = 0
    for flag in _flag_tuple(flags, "truncated flag"):
        if flag is None:
            return None
        total += flag
    return total


def build_truncation(
    base_flags: Sequence[Optional[bool]],
    tuned_flags: Sequence[Optional[bool]],
    *,
    max_new_tokens: Optional[int] = None,
) -> Truncation:
    """The two truncation counts of a task leg (one flag per answer and side)."""
    base = _flag_tuple(base_flags, "base truncated flag")
    tuned = _flag_tuple(tuned_flags, "tuned truncated flag")
    if len(base) != len(tuned):
        raise ValueError("base and tuned need the same number of rows to be compared")
    return Truncation(
        base=count_truncated(base),
        tuned=count_truncated(tuned),
        n=len(base),
        max_new_tokens=max_new_tokens,
    )


def build_task_detail(
    base_leg: object, tuned_leg: object, *, max_new_tokens: Optional[int] = None
) -> Optional[TaskDetail]:
    """The task leg's detail from the two ``LegScore`` s, or ``None`` when they carry none.

    A plain float on either side (a stubbed scorer) means the leg could not say, and gives
    ``None``: the verdict is built exactly as without the detail.
    """
    if not isinstance(base_leg, LegScore) or not isinstance(tuned_leg, LegScore):
        return None
    rows = None
    if base_leg.row_scores is not None and tuned_leg.row_scores is not None:
        rows = compare_rows(base_leg.row_scores, tuned_leg.row_scores)
    truncated = None
    if base_leg.truncated is not None and tuned_leg.truncated is not None:
        truncated = build_truncation(
            base_leg.truncated, tuned_leg.truncated, max_new_tokens=max_new_tokens
        )
    failures = None
    if base_leg.failures is not None and tuned_leg.failures is not None:
        size = next(
            (
                len(seq)
                for seq in (base_leg.row_scores, base_leg.truncated)
                if seq is not None
            ),
            None,
        )
        if size is not None:
            failures = GenerationFailures(
                base=base_leg.failures, tuned=tuned_leg.failures, n=size
            )
    detail = TaskDetail(rows=rows, truncated=truncated, generation_failures=failures)
    return None if detail.is_empty() else detail


def checkpoint_step_of(adapter: Optional[str]) -> Optional[int]:
    """N when the LAST path component of ``adapter`` is exactly ``checkpoint-N``, else ``None``.

    Never guessed: ``checkpoint-12-old``, ``ckpt-12``, ``checkpoint-`` and a checkpoint that is
    not the last component all give ``None``. Both separators count (a Windows path) and a
    trailing one is ignored.
    """
    if not isinstance(adapter, str):
        return None
    last = _PATH_SEPARATORS.split(adapter.rstrip("\\/"))[-1]
    match = _CHECKPOINT_RE.match(last)
    return int(match.group(1)) if match else None


def task_rows_pointer_error(value: object) -> Optional[str]:
    """Why ``value`` cannot be the ``task_rows`` pointer, or ``None`` when it can.

    The pointer is a bare file name ending in ``TASK_ROWS_SUFFIX``: no directory part (either
    separator), no drive, no control character, at most 255 characters. It is resolved next to
    the file that holds it, so it can never name a place outside that directory. The writer
    refuses any anchor whose derived name this would refuse, so a reader never meets a pointer
    the command wrote and cannot take.
    """
    reason = (
        f"must be a bare file name ending in {TASK_ROWS_SUFFIX} (no directory part, no "
        f"control character, at most {_MAX_TASK_ROWS_NAME_CHARS} characters)"
    )
    if not isinstance(value, str):
        return reason
    if not value.endswith(TASK_ROWS_SUFFIX) or len(value) == len(TASK_ROWS_SUFFIX):
        return reason
    if len(value) > _MAX_TASK_ROWS_NAME_CHARS:
        return reason
    if _PATH_SEPARATORS.search(value) or _DRIVE_PREFIX.match(value):
        return reason
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in value):
        return reason
    return None


def build_judged(
    *,
    base: str,
    adapter: Optional[str] = None,
    tuned: Optional[str] = None,
    base_adapter: Optional[str] = None,
) -> Judged:
    """What a live run judged, from the command-line strings exactly as given."""
    return Judged(
        base=base,
        adapter=adapter,
        tuned=tuned,
        base_adapter=base_adapter,
        checkpoint_step=checkpoint_step_of(adapter),
    )


def build_task_win(
    mode: str,
    base: object,
    tuned: object,
    *,
    noise_floor: Optional[NoiseFloor] = None,
    detail: Optional[TaskDetail] = None,
) -> TaskWin:
    """Build a leg-1 ``TaskWin``. ``won`` is the STRICT inequality tuned > base.

    With a ``noise_floor``, the margin must also clear what the instrument can
    resolve on ``TASK_AXIS``: a win of 0.004 measured on an instrument with a
    0.020 floor is not a win, it is the instrument. Without one the behaviour
    is byte-identical to v0.73.1. ``detail`` rides along and decides nothing.
    """
    if mode not in TASK_MODES:
        raise ValueError(f"mode must be one of {TASK_MODES}, got {mode!r}")
    if detail is not None and not isinstance(detail, TaskDetail):
        raise TypeError("detail must be a TaskDetail instance")
    base_f = _validate_score(base, "task base")
    tuned_f = _validate_score(tuned, "task tuned")
    floor = _floor_of(noise_floor, TASK_AXIS)
    return TaskWin(
        mode=mode, base=base_f, tuned=tuned_f, won=tuned_f > base_f + floor, detail=detail
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
    # ``replace`` and not a rebuilt ``TaskWin``: every field the caller set (the task detail
    # included) survives; only ``won`` is recomputed.
    canonical_win = (
        task_win if task_win.won == leg1_pass else replace(task_win, won=leg1_pass)
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
    if verdict.failed_rule == FAILED_ADAPTER_NOOP:
        return (
            "Every lora_B tensor in the adapter is zero, so it adds nothing to the "
            "base model (an untrained adapter, or a DPO ref/ folder). No model was "
            "loaded or evaluated."
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


def adapter_change_lines(change: AdapterChange) -> List[str]:
    """Plain-text lines for the adapter-update number (and its warning)."""
    if not change.measured:
        return [
            f"Adapter update: not measured ({change.reason}). Only plain LoRA "
            "adapters on safetensors bases are measured; nothing was checked here."
        ]
    fraction = change.fraction
    dtypes = ", ".join(change.base_dtypes) or "unknown dtype"
    if fraction is None:
        return [f"Adapter update: its {change.modules} target modules have no elements."]
    lines = [
        f"Adapter update: changes {change.changed:,} of {change.elements:,} elements "
        f"of its {change.modules} target modules in the base dtype ({dtypes}), "
        f"fraction {fraction:.3e}."
    ]
    if change.warning:
        lines.append(
            f"Warning: the adapter changes fewer than {ADAPTER_CHANGE_WARN_BELOW:.0%} of "
            f"the elements of its target modules ({fraction:.3e} < "
            f"{ADAPTER_CHANGE_WARN_BELOW:.0e}). Adapters trained with a near-zero "
            "learning rate sit in this range, and so can an early checkpoint of a real "
            "run. This line does not change the verdict."
        )
    return lines


# ---------------------------------------------------------------------------
# Task-leg detail lines (shared by the rubric, the panel and a replay of evidence)
# ---------------------------------------------------------------------------

def _count_text(count: Optional[int]) -> str:
    return "not measured" if count is None else str(count)


def detail_lines(win: TaskWin) -> List[Tuple[str, bool]]:
    """The task leg's detail as ``(text, is_warning)`` lines; empty when there is none.

    Plain text with integers and constants only (no user string), so a caller escapes it for
    its own medium. A live run and a replay of its evidence print the same lines.
    """
    detail = win.detail
    if detail is None:
        return []
    lines: List[Tuple[str, bool]] = []
    if detail.rows is not None:
        rows = detail.rows
        lines.append((
            f"Rows (tuned vs base): better {rows.better}, worse {rows.worse}, "
            f"same {rows.same} of {rows.n}",
            False,
        ))
    if detail.truncated is not None:
        cut = detail.truncated
        text = (
            f"Truncated answers: base {_count_text(cut.base)}, "
            f"tuned {_count_text(cut.tuned)} of {cut.n}"
        )
        if cut.max_new_tokens is not None:
            text += f" (cap {cut.max_new_tokens} new tokens)"
        lines.append((text, False))
        warning = cut.warning
        if warning is not None and cut.base is not None and cut.tuned is not None:
            lines.append((
                f"Warning {warning}: truncated {cut.base} -> {cut.tuned} of {cut.n} "
                f"(rise {cut.tuned - cut.base}, threshold {truncation_growth_threshold(cut.n)})",
                True,
            ))
    if detail.generation_failures is not None:
        failed = detail.generation_failures
        any_failed = failed.base > 0 or failed.tuned > 0
        text = f"Generation failures: base {failed.base}, tuned {failed.tuned} of {failed.n}"
        if any_failed:
            text += " (scored as wrong)"
        lines.append((text, any_failed))
    return lines


def _public_name(value: str) -> str:
    """What a judged value may show in text that leaves this machine (the ``--push`` comment).

    A local path shows only its last component, so the directories above it (a user name, a
    project layout) are not posted. A value is a local path when it looks absolute (a leading
    separator, a drive letter, ``~``) or names something that exists here; anything else, a Hub
    id such as ``org/name``, is shown whole. The terminal, the JSON and the evidence keep the
    value exactly as given.
    """
    looks_absolute = value.startswith(("/", "\\", "~")) or _DRIVE_PREFIX.match(value) is not None
    try:
        exists = os.path.exists(value)
    except (OSError, ValueError):
        exists = False
    if not (looks_absolute or exists):
        return value
    parts = [part for part in _PATH_SEPARATORS.split(value) if part]
    return parts[-1] if parts else value


def judged_line(verdict: ShipVerdict, *, public: bool = False) -> Optional[str]:
    """``Judged: ...`` for the terminal, or ``None`` when the verdict did not record it.

    The strings come from the command line (or an evidence file): control bytes are stripped.
    With ``public`` a local path is cut to its last component (see ``_public_name``): that is
    the text posted to GitHub by ``--push``.
    """
    judged = verdict.judged
    if judged is None:
        return None

    def shown(value: str) -> str:
        return for_terminal(_public_name(value) if public else value)

    parts = [f"base {shown(judged.base)}"]
    if judged.base_adapter is not None:
        parts.append(f"base-adapter {shown(judged.base_adapter)}")
    if judged.adapter is not None:
        text = f"adapter {shown(judged.adapter)}"
        if judged.checkpoint_step is not None:
            text += f" (checkpoint step {judged.checkpoint_step})"
        parts.append(text)
    if judged.tuned is not None:
        parts.append(f"tuned {shown(judged.tuned)}")
    return "Judged: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Rendering — plain text (for --output / clipboard) and Rich (for the terminal)
# ---------------------------------------------------------------------------

def format_ship_rubric(verdict: ShipVerdict, *, public: bool = False) -> str:
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
    if verdict.numerics is not None:
        parts.append(f"Judge numerics: {verdict.numerics}")
    else:
        parts.append("Judge numerics: unstamped")
    judged = judged_line(verdict, public=public)
    if judged is not None:
        parts.append(judged)
    parts.append("")
    parts.append(
        f"Leg 1 task win ({win.mode}): "
        f"{win.base:.4f} -> {win.tuned:.4f}  [{won_str}]"
    )
    parts.extend(f"  {text}" for text, _warning in detail_lines(win))
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
    if verdict.adapter_change is not None:
        parts.append("")
        parts.extend(adapter_change_lines(verdict.adapter_change))
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
        f"{win.base:.4f} -> {win.tuned:.4f}  {escape('[' + won_str + ']')}\n"
        "[dim]Judge numerics: "
        f"{escape(verdict.numerics) if verdict.numerics is not None else 'unstamped'}"
        "[/]"
    )
    for text, is_warning in detail_lines(win):
        header += f"\n[{'yellow' if is_warning else 'dim'}]{escape(text)}[/]"
    judged = judged_line(verdict)
    if judged is not None:
        # judged_line stripped the control bytes (the strings are user input); escape the markup.
        header += f"\n[dim]{escape(judged)}[/]"

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

    # The explanation names regressed benchmarks, which come from evidence
    # files: strip control characters as well as escaping markup.
    footer = f"[dim]{escape(for_terminal(_failed_rule_explanation(verdict)))}[/]"
    parts = [header, "", table, ""]
    if verdict.noise_floor is not None:
        parts.extend([
            _render_noise_floor(verdict.noise_floor),
            f"[dim]Measured over {verdict.noise_floor.runs} base repeats. Each "
            "axis is gated at max(threshold, its floor). A floor sizes this "
            "instrument's resolution; it does not calibrate a threshold.[/]",
            "",
        ])
    if verdict.adapter_change is not None:
        # The reason can quote tensor names from the adapter file.
        for line in adapter_change_lines(verdict.adapter_change):
            text = escape(for_terminal(line))
            if line.startswith("Warning:"):
                text = "[yellow]" + text + "[/]"
            parts.append(text)
        parts.append("")
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
    # Posted to GitHub: a local path in the "Judged:" line is cut to its last component.
    body = format_ship_rubric(verdict, public=True)
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

def _rows_dict(rows: TaskRows) -> Dict[str, object]:
    return {"better": rows.better, "worse": rows.worse, "same": rows.same, "n": rows.n}


def _truncated_dict(cut: Truncation, *, with_warning: bool) -> Dict[str, object]:
    block: Dict[str, object] = {
        "base": cut.base,
        "tuned": cut.tuned,
        "n": cut.n,
        "max_new_tokens": cut.max_new_tokens,
    }
    if with_warning:
        block["warning"] = cut.warning
    return block


def _failures_dict(failed: GenerationFailures) -> Dict[str, object]:
    return {"base": failed.base, "tuned": failed.tuned, "n": failed.n}


def _judged_dict(judged: Judged) -> Dict[str, object]:
    return {
        "base": judged.base,
        "adapter": judged.adapter,
        "tuned": judged.tuned,
        "base_adapter": judged.base_adapter,
        "checkpoint_step": judged.checkpoint_step,
    }


def verdict_to_evidence(
    verdict: ShipVerdict,
    *,
    provenance: Optional[Mapping[str, object]] = None,
    task_rows: Optional[str] = None,
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
    config that produced it (v0.71.39). ``numerics`` (optional) is the actual
    live-eval load precision; the gate compares its family against ``--config``
    (#367). ``task_rows`` (optional) is the bare name of the per-row file next to this one
    (``task_rows_pointer_error``); like ``provenance`` it is metadata of the file, not part of
    the verdict, and the reader discards it after checking it.
    """
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    win = verdict.task_win
    task_block: Dict[str, object] = {"mode": win.mode, "base": win.base, "tuned": win.tuned}
    evidence: Dict[str, object] = {
        "task": task_block,
        "benchmarks": {
            item.name: {"base": item.base, "tuned": item.tuned}
            for item in verdict.benchmark_deltas
        },
    }
    # Schema version 2: the task detail and what was judged. Each block is written only when it
    # was measured, and the version is stamped only when something version 2 is written, so a
    # verdict that carries none of it (a replayed version-1 file) is still a version-1 file that
    # soup-cli 0.77.x can read. The truncation warning is NOT stored: it is computed from the
    # counts on every read, so a stored warning can never contradict its own counts.
    detail = win.detail
    carries_v2 = False
    if detail is not None:
        if detail.rows is not None:
            task_block["rows"] = _rows_dict(detail.rows)
            carries_v2 = True
        if detail.truncated is not None:
            task_block["truncated"] = _truncated_dict(detail.truncated, with_warning=False)
            carries_v2 = True
        if detail.generation_failures is not None:
            task_block["generation_failures"] = _failures_dict(detail.generation_failures)
            carries_v2 = True
    if verdict.judged is not None:
        evidence["judged"] = _judged_dict(verdict.judged)
        carries_v2 = True
    if task_rows is not None:
        pointer_error = task_rows_pointer_error(task_rows)
        if pointer_error is not None:
            raise ValueError(f"task_rows {pointer_error}")
        evidence["task_rows"] = task_rows
        carries_v2 = True
    if carries_v2:
        evidence["schema_version"] = SCHEMA_VERSION
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
    if verdict.numerics is not None:
        evidence["numerics"] = verdict.numerics
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
    _validate_evidence_fields(payload, "noise_floor", "evidence.noise_floor")
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


def _evidence_schema_version(payload: Mapping[object, object]) -> int:
    """The evidence's ``schema_version``: absent is 1 (soup-cli 0.77.x), 1 and 2 are read.

    Anything else, a bool and a float included, is refused with the key named. A reader never
    guesses what a version it does not know means.
    """
    if "schema_version" not in payload:
        return 1
    value = payload["schema_version"]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in SUPPORTED_SCHEMA_VERSIONS
    ):
        shown = (
            str(value)
            if isinstance(value, int) and not isinstance(value, bool) and abs(value) < 10**6
            else type(value).__name__
        )
        raise ValueError(f"evidence.schema_version must be 1 or 2, got {shown}")
    return value


def _refuse_v2_fields_under_v1(
    payload: Mapping[object, object], task: Mapping[object, object]
) -> None:
    """Version 1 (or an absent version) does not know the version-2 fields: refuse, naming one."""
    for section, key in V2_ONLY_FIELDS:
        block = payload if section == "root" else task
        if key in block:
            path = f"evidence.{key}" if section == "root" else f"evidence.{section}.{key}"
            raise ValueError(
                f"{path} is a schema_version 2 field, but this evidence is schema_version 1 "
                "(an absent schema_version means 1); set schema_version to 2 to carry it"
            )


def _evidence_int(
    value: object, path: str, *, low: int, high: int, nullable: bool = False
) -> Optional[int]:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer" + (" or null" if nullable else ""))
    if not (low <= value <= high):
        raise ValueError(f"{path} must be in [{low}, {high}]")
    return value


def _evidence_block(block: object, section: str, path: str) -> Mapping[object, object]:
    """A detail block: an object holding exactly the registered keys (null says not measured)."""
    if not isinstance(block, Mapping):
        raise ValueError(f"{path} must be an object")
    _validate_evidence_fields(block, section, path)
    missing = sorted(EVIDENCE_SCHEMA_FIELDS[section] - set(block))
    if missing:
        raise ValueError(f"{path} needs {', '.join(missing)} (use null for not measured)")
    return block


def _rows_from_evidence(payload: object) -> TaskRows:
    path = "evidence.task.rows"
    block = _evidence_block(payload, "rows", path)
    counts = {
        key: _evidence_int(block[key], f"{path}.{key}", low=0, high=_MAX_EVIDENCE_ROWS)
        for key in ("better", "worse", "same")
    }
    rows = TaskRows(better=counts["better"], worse=counts["worse"], same=counts["same"])
    n = _evidence_int(block["n"], f"{path}.n", low=1, high=_MAX_EVIDENCE_ROWS)
    if rows.n != n:
        raise ValueError(f"{path}: better + worse + same must equal n ({rows.n} != {n})")
    return rows


def _truncation_from_evidence(payload: object) -> Truncation:
    path = "evidence.task.truncated"
    block = _evidence_block(payload, "truncated", path)
    n = _evidence_int(block["n"], f"{path}.n", low=1, high=_MAX_EVIDENCE_ROWS)
    return Truncation(
        base=_evidence_int(block["base"], f"{path}.base", low=0, high=n, nullable=True),
        tuned=_evidence_int(block["tuned"], f"{path}.tuned", low=0, high=n, nullable=True),
        n=n,
        max_new_tokens=_evidence_int(
            block["max_new_tokens"], f"{path}.max_new_tokens",
            low=1, high=_MAX_EVIDENCE_NEW_TOKENS, nullable=True,
        ),
    )


def _failures_from_evidence(payload: object) -> GenerationFailures:
    path = "evidence.task.generation_failures"
    block = _evidence_block(payload, "generation_failures", path)
    n = _evidence_int(block["n"], f"{path}.n", low=1, high=_MAX_EVIDENCE_ROWS)
    return GenerationFailures(
        base=_evidence_int(block["base"], f"{path}.base", low=0, high=n),
        tuned=_evidence_int(block["tuned"], f"{path}.tuned", low=0, high=n),
        n=n,
    )


def _task_detail_from_evidence(task: Mapping[object, object]) -> Optional[TaskDetail]:
    """Decode the version-2 detail blocks of ``evidence.task`` (each is optional)."""
    rows = _rows_from_evidence(task["rows"]) if "rows" in task else None
    truncated = _truncation_from_evidence(task["truncated"]) if "truncated" in task else None
    failures = (
        _failures_from_evidence(task["generation_failures"])
        if "generation_failures" in task else None
    )
    # One task file, one row count: the blocks describe the same rows.
    sizes = [
        (name, block.n)
        for name, block in (
            ("rows", rows), ("truncated", truncated), ("generation_failures", failures),
        )
        if block is not None
    ]
    for name, size in sizes[1:]:
        if size != sizes[0][1]:
            raise ValueError(
                f"evidence.task.{name}.n ({size}) differs from "
                f"evidence.task.{sizes[0][0]}.n ({sizes[0][1]})"
            )
    detail = TaskDetail(rows=rows, truncated=truncated, generation_failures=failures)
    return None if detail.is_empty() else detail


def _judged_string(value: object, path: str, *, nullable: bool) -> Optional[str]:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_JUDGED_CHARS
        or "\x00" in value
    ):
        raise ValueError(
            f"{path} must be a non-empty string of at most {_MAX_JUDGED_CHARS} characters "
            "without NUL" + (" or null" if nullable else "")
        )
    return value


def _judged_from_evidence(payload: object) -> Judged:
    path = "evidence.judged"
    block = _evidence_block(payload, "judged", path)
    adapter = _judged_string(block["adapter"], f"{path}.adapter", nullable=True)
    step = _evidence_int(
        block["checkpoint_step"], f"{path}.checkpoint_step",
        low=0, high=_MAX_CHECKPOINT_STEP, nullable=True,
    )
    if step != checkpoint_step_of(adapter):
        # "Never guessed" holds on read too: the step is what the path says, or null.
        raise ValueError(
            f"{path}.checkpoint_step must be {checkpoint_step_of(adapter)} for the adapter "
            "path it names (null when the last path component is not checkpoint-<digits>)"
        )
    return Judged(
        base=_judged_string(block["base"], f"{path}.base", nullable=False),
        adapter=adapter,
        tuned=_judged_string(block["tuned"], f"{path}.tuned", nullable=True),
        base_adapter=_judged_string(block["base_adapter"], f"{path}.base_adapter", nullable=True),
        checkpoint_step=step,
    )


def verdict_from_evidence(
    payload: Mapping[str, object],
    *,
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD,
) -> ShipVerdict:
    """Decode the complete evidence schema into a verdict.

    This is the single reader shared by the CLI and MCP surfaces. Schema
    validation belongs here so adding an evidence key cannot update one public
    surface while leaving the other to silently compute a different verdict.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("evidence must be an object")
    _validate_evidence_fields(payload, "root", "evidence")
    version = _evidence_schema_version(payload)
    threshold = _validate_threshold(forgetting_threshold)

    task = payload.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("evidence.task must be an object with 'mode', 'base', 'tuned'")
    _validate_evidence_fields(task, "task", "evidence.task")
    if version == 1:  # not "< SCHEMA_VERSION": a reader that learns version 3 still reads 2
        _refuse_v2_fields_under_v1(payload, task)
    mode = task.get("mode", "metric")
    if mode not in SUPPORTED_TASK_MODES:
        supported = ", ".join(SUPPORTED_TASK_MODES)
        raise ValueError(
            f"evidence.task.mode must be one of {supported}; got {mode!r}"
        )
    if "base" not in task or "tuned" not in task:
        raise ValueError("evidence.task needs both 'base' and 'tuned' scores")

    try:
        stored_floor = noise_floor_from_evidence(payload.get("noise_floor"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid evidence.noise_floor: {exc}") from exc
    detail = _task_detail_from_evidence(task)
    try:
        task_win = build_task_win(
            str(mode), task["base"], task["tuned"], noise_floor=stored_floor, detail=detail
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid evidence.task: {exc}") from exc

    raw_benchmarks = payload.get("benchmarks", {})
    if not isinstance(raw_benchmarks, Mapping):
        raise ValueError(
            "evidence.benchmarks must be an object of {name: {base, tuned}}"
        )
    base_scores: Dict[str, object] = {}
    tuned_scores: Dict[str, object] = {}
    for name, entry in raw_benchmarks.items():
        if not isinstance(entry, Mapping) or "base" not in entry or "tuned" not in entry:
            raise ValueError(
                f"evidence.benchmarks[{name!r}] needs 'base' and 'tuned'"
            )
        _validate_evidence_fields(
            entry, "benchmark", f"evidence.benchmarks[{name!r}]"
        )
        key = str(name)
        base_scores[key] = entry["base"]
        tuned_scores[key] = entry["tuned"]

    try:
        deltas = compute_benchmark_deltas(
            base_scores,
            tuned_scores,
            forgetting_threshold=threshold,
            noise_floor=stored_floor,
        )
        verdict = decide_ship(
            task_win,
            deltas,
            forgetting_threshold=threshold,
            noise_floor=stored_floor,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid evidence.benchmarks: {exc}") from exc
    try:
        stored_numerics = numerics_from_evidence(payload.get("numerics"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid evidence.numerics: {exc}") from exc
    if "judged" in payload:
        verdict = replace(verdict, judged=_judged_from_evidence(payload["judged"]))
    if "task_rows" in payload:
        # Checked and then dropped: the pointer names a file next to the evidence, it is not part
        # of the verdict (a replay writes no row file, so it has no pointer to give).
        pointer_error = task_rows_pointer_error(payload["task_rows"])
        if pointer_error is not None:
            raise ValueError(f"evidence.task_rows {pointer_error}")
    if stored_numerics is not None:
        return replace(verdict, numerics=stored_numerics)
    return verdict


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


def verdict_to_dict(
    verdict: ShipVerdict, *, task_rows: Optional[str] = None
) -> Dict[str, object]:
    """JSON-serializable dict for ``--output`` / programmatic consumers.

    ``task_rows`` is the bare name of the per-row file next to the output, or ``None`` when
    this run wrote none (the key is always present).
    """
    if not isinstance(verdict, ShipVerdict):
        raise TypeError("verdict must be a ShipVerdict instance")
    if task_rows is not None:
        pointer_error = task_rows_pointer_error(task_rows)
        if pointer_error is not None:
            raise ValueError(f"task_rows {pointer_error}")
    win = verdict.task_win
    detail = win.detail
    data: Dict[str, object] = {
        # Absent in soup-cli 0.77.x, which reads as version 1. Version 2 adds ``judged`` and
        # ``task_win.rows`` / ``truncated`` / ``generation_failures`` (each null when not
        # measured); no key of version 1 is removed or renamed.
        "schema_version": SCHEMA_VERSION,
        "decision": verdict.decision,
        "task_win": {
            "mode": win.mode,
            "base": win.base,
            "tuned": win.tuned,
            "won": win.won,
            "rows": (
                None if detail is None or detail.rows is None else _rows_dict(detail.rows)
            ),
            "truncated": (
                None
                if detail is None or detail.truncated is None
                else _truncated_dict(detail.truncated, with_warning=True)
            ),
            "generation_failures": (
                None
                if detail is None or detail.generation_failures is None
                else _failures_dict(detail.generation_failures)
            ),
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
        "numerics": verdict.numerics,
        "adapter_change": _adapter_change_dict(verdict.adapter_change),
        "judged": None if verdict.judged is None else _judged_dict(verdict.judged),
        "task_rows": task_rows,
    }
    return data


def _adapter_change_dict(change: Optional[AdapterChange]) -> Optional[Dict[str, object]]:
    if change is None:
        return None
    return {
        "measured": change.measured,
        "fraction": change.fraction,
        "changed": change.changed,
        "elements": change.elements,
        "modules": change.modules,
        "base_dtypes": list(change.base_dtypes),
        "warn_below": ADAPTER_CHANGE_WARN_BELOW,
        "warning": change.warning,
        "reason": change.reason,
    }
