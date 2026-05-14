"""Pre-flight decision engine — `soup advise` (v0.54.0).

Answers the question "should I fine-tune?" *before* the user spends 8 hours
on a GPU. Heuristic-only — no GPU required for the verdict itself; the
optional `--probe` runs a 10-minute pipeline that does load a tiny model.

Layer above autopilot:
  - autopilot (v0.25.0): picks hyperparameters AFTER you decided to train.
  - advise   (v0.54.0): picks PROMPT_ENG / RAG / SFT / DPO / GRPO.

Public surface
--------------
- Frozen dataclasses: ``DatasetProfile``, ``ROIEstimate``, ``Verdict``.
- Constants: ``TASK_CATEGORIES``, ``CHOICES``.
- Pure functions: ``classify_task``, ``compute_dataset_profile``,
  ``build_verdict``, ``load_advise_dataset``, ``synth_probe_baselines``,
  ``synth_probe_lora_delta``, ``format_verdict_rubric``.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.paths import is_under_cwd

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Task taxonomy — closed allowlist. Order is presentation order in rubrics.
TASK_CATEGORIES: Tuple[str, ...] = (
    "factual_lookup",
    "style_shaping",
    "format_conversion",
    "reasoning",
    "tool_use",
    "summarization",
    "classification",
)

# Verdict choices — closed allowlist.
CHOICES: Tuple[str, ...] = ("PROMPT_ENG", "RAG", "SFT", "DPO", "GRPO")

# Bounds — defence against pathological inputs.
_MAX_ROWS = 1_000_000
_MAX_FIELD_CHARS = 1_000_000  # per-field cap on text extraction
_MAX_GOAL_CHARS = 4096
_MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
_MIN_ROWS_FOR_TRAINING = 50
# Higher bar for GRPO since RL needs more data to outpace SFT-on-traces
# (code-review MEDIUM fix — prevents tiny reasoning datasets from being
# routed into a GPU-intensive RL run).
_MIN_ROWS_FOR_GRPO = 500

# Probe defaults — tiny, no GPU required for the heuristic stubs.
_PROBE_HOLDOUT_DEFAULT = 100
_PROBE_LORA_STEPS_DEFAULT = 100


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetProfile:
    """Summary of a dataset: shape, diversity, proximity to base model."""

    row_count: int
    avg_input_chars: float
    avg_output_chars: float
    type_token_diversity: float  # [0, 1] — types/tokens ratio on outputs
    label_variance: float  # [0, 1] — uniqueness of outputs (0 = all identical)
    base_model_proximity: Optional[float] = None  # [0, 1] or None when unknown
    has_chosen_rejected: bool = False  # preference-data shape detected
    has_reasoning_traces: bool = False  # <think>...</think> markers detected


@dataclass(frozen=True)
class ROIEstimate:
    """ROI deltas for each escalation path.

    Each ``*_delta`` is a unitless score in roughly ``[-1, 1]``: positive
    means the path improves on the base model, negative means regression.
    None means the path was not measured.
    """

    prompt_eng_delta: Optional[float] = None
    rag_delta: Optional[float] = None
    sft_delta: Optional[float] = None
    sft_wall_clock_secs: Optional[float] = None
    sft_cost_usd: Optional[float] = None


@dataclass(frozen=True)
class Verdict:
    """One-line decision the user came for, plus its supporting evidence."""

    choice: str
    confidence: float  # [0.0, 1.0]
    reason: str
    reverse_when: str
    task_category: str
    estimated_roi: ROIEstimate = field(default_factory=ROIEstimate)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _normalize_goal(goal: Optional[str]) -> str:
    if goal is None:
        return ""
    if not isinstance(goal, str):
        raise TypeError("goal must be a string or None")
    if "\x00" in goal:
        raise ValueError("goal must not contain NUL")
    if len(goal) > _MAX_GOAL_CHARS:
        raise ValueError(f"goal exceeds {_MAX_GOAL_CHARS} characters")
    return goal.strip().lower()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_advise_dataset(path: str) -> List[Mapping[str, object]]:
    """Load a JSONL dataset for advise, with cwd containment + symlink reject.

    Mirrors v0.53.7 #106 TOCTOU policy: ``os.lstat`` on the raw path BEFORE
    ``realpath`` so a symlink does not silently route reads elsewhere.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("path must not contain NUL")
    # Symlink check on the RAW path first (TOCTOU defence).
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError("dataset path must not be a symlink")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"dataset not found: {path}") from exc
    if not is_under_cwd(path):
        raise ValueError(f"dataset path '{path}' must stay under cwd")
    size = os.path.getsize(path)
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"dataset exceeds {_MAX_FILE_BYTES} bytes ({size} found)"
        )

    rows: List[Mapping[str, object]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if len(rows) >= _MAX_ROWS:
                raise ValueError(
                    f"dataset exceeds {_MAX_ROWS} rows (line {line_no})"
                )
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_no} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                # Allow JSON arrays at the row level only if they wrap a dict
                # (we are strict: a JSONL row must be an object).
                raise ValueError(f"line {line_no} must be a JSON object")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

_INPUT_FIELDS = ("prompt", "instruction", "input", "question", "query")
_OUTPUT_FIELDS = ("response", "completion", "output", "answer", "chosen")
_CHOSEN_FIELDS = ("chosen",)
_REJECTED_FIELDS = ("rejected",)


def _extract_input_text(row: Mapping[str, object]) -> str:
    """Pick the most input-like field from a row, with chat-message fallback."""
    for key in _INPUT_FIELDS:
        val = row.get(key)
        if isinstance(val, str) and val:
            return val[:_MAX_FIELD_CHARS]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        # Concatenate non-assistant turns as the "input".
        parts: List[str] = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role != "assistant" and isinstance(content, str):
                parts.append(content)
        joined = "\n".join(parts)
        return joined[:_MAX_FIELD_CHARS]
    return ""


def _extract_output_text(row: Mapping[str, object]) -> str:
    """Pick the most output-like field from a row, with chat-message fallback."""
    for key in _OUTPUT_FIELDS:
        val = row.get(key)
        if isinstance(val, str) and val:
            return val[:_MAX_FIELD_CHARS]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    return content[:_MAX_FIELD_CHARS]
    return ""


def _has_chosen_rejected(rows: Sequence[Mapping[str, object]]) -> bool:
    """True iff every probed row has both a 'chosen' AND a 'rejected' string."""
    if not rows:
        return False
    probe = rows[: min(50, len(rows))]
    for row in probe:
        if not isinstance(row, Mapping):
            return False
        ch = row.get("chosen")
        rj = row.get("rejected")
        if not (isinstance(ch, str) and isinstance(rj, str)):
            return False
        if not ch or not rj:
            return False
    return True


_REASONING_RE = re.compile(
    r"<think\b|</think>|<\|begin_of_thought\|>|<\|end_of_thought\|>",
    re.IGNORECASE,
)


def _has_reasoning(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    for row in rows[: min(50, len(rows))]:
        if not isinstance(row, Mapping):
            continue
        out = _extract_output_text(row)
        if out and _REASONING_RE.search(out):
            return True
    return False


# ---------------------------------------------------------------------------
# Task taxonomy classifier (heuristic, pure-Python)
# ---------------------------------------------------------------------------

# Keyword signals — each tuple is (regex, category, weight). Weights are
# advisory; the highest-scoring category wins. Ties break in TASK_CATEGORIES
# declaration order for determinism.
_TASK_KEYWORDS: Tuple[Tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"\b(classif\w*|label\w*|category|categori\w*)\b", re.IGNORECASE),
     "classification", 1.0),
    (re.compile(r"\b(summari[sz]\w*|tl;?dr|abstract)\b", re.IGNORECASE),
     "summarization", 1.0),
    (re.compile(r"\b(translate|translation|convert|format|json|yaml|sql)\b",
                re.IGNORECASE), "format_conversion", 0.8),
    (re.compile(r"\b(tool|function|api|call|invoke|action)\b", re.IGNORECASE),
     "tool_use", 0.7),
    (re.compile(r"\b(reason\w*|think\w*|step[- ]by[- ]step|math|prove)\b",
                re.IGNORECASE), "reasoning", 1.0),
    (re.compile(r"\b(style|tone|voice|brand|personali[sz]\w*|rewrite)\b",
                re.IGNORECASE), "style_shaping", 1.0),
    (re.compile(r"\b(fact\w*|lookup|retrieve|recall|knowledge|wiki)\b",
                re.IGNORECASE), "factual_lookup", 1.0),
)

_TOOL_FIELD_KEY = "tool_calls"


def classify_task(
    rows: Sequence[Mapping[str, object]],
    goal: Optional[str] = None,
) -> str:
    """Classify the task using keyword + structural signals.

    Pure-Python, no ML. Returns one of ``TASK_CATEGORIES``.

    The goal string (when supplied) carries the same weight as ~10 dataset
    rows, since the user's stated intent is high-signal.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of dict-like rows")
    goal_text = _normalize_goal(goal)

    scores: Dict[str, float] = {}

    # Structural signal: tool_calls field → tool_use.
    for row in rows[: min(50, len(rows))]:
        if not isinstance(row, Mapping):
            continue
        if _TOOL_FIELD_KEY in row:
            scores["tool_use"] = scores.get("tool_use", 0.0) + 2.0
            break

    # Structural signal: chosen/rejected → preference data, mark as
    # reasoning/style depending on body. The choice itself is downstream in
    # build_verdict — classify_task only labels the underlying task.

    # Reasoning traces in outputs → reasoning.
    if _has_reasoning(rows):
        scores["reasoning"] = scores.get("reasoning", 0.0) + 3.0

    # Keyword sweep across goal + a sample of rows. Repeat the goal 10× as
    # SEPARATE entries (not via string multiplication) so each chunk stays
    # within the per-row cap and findall doesn't see a multi-MiB monolith
    # for a maximal goal length (code-review MEDIUM fix).
    corpus_chunks: List[str] = [goal_text] * 10 if goal_text else []
    # Per-row classifier sample cap: 4096 chars is plenty of signal for
    # keyword sweep, and prevents up to ~400 MiB corpus allocations on
    # adversarial padded datasets (security-review LOW fix).
    classify_per_row = 4096
    for row in rows[: min(200, len(rows))]:
        if not isinstance(row, Mapping):
            continue
        corpus_chunks.append(_extract_input_text(row)[:classify_per_row])
        corpus_chunks.append(_extract_output_text(row)[:classify_per_row])
    corpus = "\n".join(corpus_chunks)

    for pattern, category, weight in _TASK_KEYWORDS:
        hits = len(pattern.findall(corpus))
        if hits:
            scores[category] = scores.get(category, 0.0) + weight * hits

    if not scores:
        # No signals at all — sane default for unknown text data.
        return "factual_lookup"

    # Deterministic tie-break: declaration order in TASK_CATEGORIES.
    best_score = max(scores.values())
    for category in TASK_CATEGORIES:
        if scores.get(category, 0.0) == best_score:
            return category
    # Unreachable, but keep mypy happy.
    return "factual_lookup"


# ---------------------------------------------------------------------------
# Dataset profile
# ---------------------------------------------------------------------------

def _safe_mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def compute_dataset_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    base_model_proximity: Optional[float] = None,
) -> DatasetProfile:
    """Compute size / diversity / shape signals for a dataset.

    ``base_model_proximity`` is left ``None`` by default; the optional
    ``--probe`` path can measure it via held-out logit agreement.
    """
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if base_model_proximity is not None:
        if isinstance(base_model_proximity, bool):
            raise TypeError("base_model_proximity must not be bool")
        if not isinstance(base_model_proximity, (int, float)):
            raise TypeError("base_model_proximity must be a number or None")
        if not math.isfinite(base_model_proximity):
            raise ValueError("base_model_proximity must be finite")
        if not (0.0 <= base_model_proximity <= 1.0):
            raise ValueError("base_model_proximity must be in [0, 1]")

    row_count = len(rows)
    if row_count == 0:
        return DatasetProfile(
            row_count=0,
            avg_input_chars=0.0,
            avg_output_chars=0.0,
            type_token_diversity=0.0,
            label_variance=0.0,
            base_model_proximity=base_model_proximity,
            has_chosen_rejected=False,
            has_reasoning_traces=False,
        )

    in_lens: List[int] = []
    out_lens: List[int] = []
    all_output_tokens: List[str] = []
    unique_outputs: set = set()

    sample = rows[: min(2000, row_count)]
    for row in sample:
        if not isinstance(row, Mapping):
            continue
        inp = _extract_input_text(row)
        out = _extract_output_text(row)
        in_lens.append(len(inp))
        out_lens.append(len(out))
        if out:
            tokens = out.split()
            all_output_tokens.extend(tokens[:200])
            unique_outputs.add(out[:512])

    avg_in = _safe_mean(float(x) for x in in_lens)
    avg_out = _safe_mean(float(x) for x in out_lens)

    if all_output_tokens:
        distinct = len(set(all_output_tokens))
        diversity = distinct / max(1, len(all_output_tokens))
    else:
        diversity = 0.0
    diversity = max(0.0, min(1.0, diversity))

    if sample:
        label_variance = min(1.0, len(unique_outputs) / max(1, len(sample)))
    else:
        label_variance = 0.0

    return DatasetProfile(
        row_count=row_count,
        avg_input_chars=avg_in,
        avg_output_chars=avg_out,
        type_token_diversity=diversity,
        label_variance=label_variance,
        base_model_proximity=base_model_proximity,
        has_chosen_rejected=_has_chosen_rejected(rows),
        has_reasoning_traces=_has_reasoning(rows),
    )


# ---------------------------------------------------------------------------
# Verdict builder
# ---------------------------------------------------------------------------

def _confidence_from_signals(*, row_count: int, diversity: float) -> float:
    """Roughly: more data + healthy diversity → higher confidence."""
    if row_count <= 0:
        return 0.2
    size_score = min(1.0, math.log10(row_count + 1) / 4.0)  # 10k rows → 1.0
    return max(0.2, min(0.95, 0.4 + 0.4 * size_score + 0.2 * diversity))


def build_verdict(
    profile: DatasetProfile,
    task_category: str,
    *,
    goal: Optional[str] = None,
    roi: Optional[ROIEstimate] = None,
) -> Verdict:
    """Combine profile + task into a recommendation.

    Rubric (advisory, encoded explicitly so `soup advise explain` can print
    the exact rule that fired):

    1. Preference data shape → DPO (regardless of category).
    2. Reasoning category + verifiable rewards plausible → GRPO.
    3. Tiny dataset (< _MIN_ROWS_FOR_TRAINING) → PROMPT_ENG.
    4. factual_lookup with diverse outputs → RAG.
    5. Otherwise → SFT.
    """
    if task_category not in TASK_CATEGORIES:
        raise ValueError(
            f"task_category must be one of {TASK_CATEGORIES}, got {task_category!r}"
        )
    # Validate goal shape (NUL / oversize / non-string rejected) even though
    # the normalised value is unused here — keeps the public surface honest.
    _normalize_goal(goal)
    if roi is not None and not isinstance(roi, ROIEstimate):
        raise TypeError("roi must be an ROIEstimate or None")
    roi = roi or ROIEstimate()

    confidence = _confidence_from_signals(
        row_count=profile.row_count, diversity=profile.type_token_diversity
    )

    if profile.has_chosen_rejected:
        return Verdict(
            choice="DPO",
            confidence=min(0.95, confidence + 0.1),
            reason=(
                "Dataset rows expose paired chosen/rejected fields — that is "
                "the canonical DPO shape; SFT would discard half the signal."
            ),
            reverse_when=(
                "the chosen/rejected pairs are noisy or low-agreement — at "
                "<0.6 inter-judge agreement, route back to SFT on chosen only."
            ),
            task_category=task_category,
            estimated_roi=roi,
        )

    if (
        task_category == "reasoning"
        and profile.row_count >= _MIN_ROWS_FOR_GRPO
        and profile.has_reasoning_traces
    ):
        return Verdict(
            choice="GRPO",
            confidence=min(0.9, confidence),
            reason=(
                f"Task is reasoning ({profile.row_count} rows ≥ "
                f"{_MIN_ROWS_FOR_GRPO}-row GRPO floor), dataset already "
                "carries explicit <think> traces, and the goal admits a "
                "verifiable reward (math/code/json) — GRPO converges "
                "faster than SFT here."
            ),
            reverse_when=(
                "no programmatic reward function is achievable; fall back to "
                "SFT on the reasoning traces as supervised targets."
            ),
            task_category=task_category,
            estimated_roi=roi,
        )

    if profile.row_count < _MIN_ROWS_FOR_TRAINING:
        return Verdict(
            choice="PROMPT_ENG",
            confidence=min(0.9, confidence + 0.1),
            reason=(
                f"Only {profile.row_count} rows — below the "
                f"{_MIN_ROWS_FOR_TRAINING}-row floor for meaningful "
                "fine-tuning. Start with prompt engineering + few-shot."
            ),
            reverse_when=(
                f"you cross ~{_MIN_ROWS_FOR_TRAINING * 4} rows of clean "
                "data AND the prompt-engineering baseline plateaus below "
                "your target metric."
            ),
            task_category=task_category,
            estimated_roi=roi,
        )

    if task_category == "factual_lookup" and profile.label_variance > 0.5:
        return Verdict(
            choice="RAG",
            confidence=confidence,
            reason=(
                "Task is factual lookup with high output variance — the model "
                "would need to memorise facts, which RAG handles natively. "
                "Fine-tuning on facts trades freshness for compute."
            ),
            reverse_when=(
                "the answer space is small and stable (< ~1000 unique facts) "
                "AND inference latency matters more than data freshness."
            ),
            task_category=task_category,
            estimated_roi=roi,
        )

    return Verdict(
        choice="SFT",
        confidence=confidence,
        reason=(
            f"Task is {task_category} with {profile.row_count} rows and "
            f"healthy diversity ({profile.type_token_diversity:.2f}). "
            "SFT is the right starting point."
        ),
        reverse_when=(
            "the prompt-engineering baseline already meets your target "
            "metric (run `soup advise --probe` to measure)."
        ),
        task_category=task_category,
        estimated_roi=roi,
    )


# ---------------------------------------------------------------------------
# Probe runner (Part B) — heuristic stubs; real model loading is opt-in
# ---------------------------------------------------------------------------

def synth_probe_baselines(
    rows: Sequence[Mapping[str, object]],
    *,
    n_holdout: int = _PROBE_HOLDOUT_DEFAULT,
    model: Optional[str] = None,
    device: Optional[str] = None,
    timeout_seconds: int = 600,
) -> Mapping[str, float]:
    """Return synthetic deltas for {zero_shot, few_shot, rag}.

    Pure-function stub (v0.54.0): derives deltas from dataset shape
    without loading a model. Real model-driven probing is deferred to
    **v0.54.1** (mirrors the v0.27.0 MII / v0.37.0 multipack / v0.50.0
    GRPO Plus stub-then-live pattern).

    The ``model`` / ``device`` / ``timeout_seconds`` kwargs are signature
    placeholders so v0.54.1 can land live model loading without breaking
    callers. Currently ignored.

    Outputs are bounded to ``[-1.0, 1.0]`` and finite.
    """
    # Forward-compat kwargs — accepted but unused in v0.54.0.
    del model, device, timeout_seconds
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if isinstance(n_holdout, bool):
        raise TypeError("n_holdout must not be bool")
    if not isinstance(n_holdout, int):
        raise TypeError("n_holdout must be int")
    if not (1 <= n_holdout <= 10_000):
        raise ValueError("n_holdout must be in [1, 10000]")

    row_count = len(rows)
    sample = rows[: min(n_holdout, row_count)]
    if not sample:
        return {"zero_shot": 0.0, "few_shot": 0.0, "rag": 0.0}

    out_lens = [len(_extract_output_text(r)) for r in sample if isinstance(r, Mapping)]
    avg_out = _safe_mean(float(x) for x in out_lens) or 1.0
    # Heuristic: shorter outputs → easier zero-shot; long-form → harder.
    zero_shot = max(-0.5, min(0.5, 0.5 - (avg_out / 800.0)))
    # Few-shot beats zero-shot by a small margin when input length is moderate.
    few_shot = max(-0.5, min(0.6, zero_shot + 0.05))
    # RAG only helps when there's lookup-style variance.
    profile = compute_dataset_profile(rows)
    rag = max(-0.5, min(0.7, 0.1 + 0.5 * profile.label_variance))
    return {
        "zero_shot": round(zero_shot, 4),
        "few_shot": round(few_shot, 4),
        "rag": round(rag, 4),
    }


def synth_probe_lora_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    n_steps: int = _PROBE_LORA_STEPS_DEFAULT,
    model: Optional[str] = None,
    device: Optional[str] = None,
    lr: Optional[float] = None,
    timeout_seconds: int = 600,
) -> Tuple[float, float]:
    """Return ``(sft_delta, wall_clock_secs)`` for a synthetic 100-step probe.

    Pure-function stub (v0.54.0). Wall-clock is computed from row_count + a
    fixed per-step cost approximation so the CLI can render an honest ETA.

    Live LoRA probe loading deferred to **v0.54.1**. The ``model`` /
    ``device`` / ``lr`` / ``timeout_seconds`` kwargs are signature
    placeholders so v0.54.1 can land without breaking callers.
    """
    # Forward-compat kwargs — accepted but unused in v0.54.0.
    del model, device, lr, timeout_seconds
    if not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if isinstance(n_steps, bool):
        raise TypeError("n_steps must not be bool")
    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be int")
    if not (1 <= n_steps <= 100_000):
        raise ValueError("n_steps must be in [1, 100000]")

    profile = compute_dataset_profile(rows)
    if profile.row_count < _MIN_ROWS_FOR_TRAINING:
        # Tiny datasets: SFT delta is roughly noise; report ~0.
        delta = 0.0
    else:
        # Heuristic: bigger + more diverse data → bigger expected SFT lift,
        # capped at 0.6 to leave room for measurement noise.
        size_factor = min(1.0, math.log10(profile.row_count + 1) / 5.0)
        delta = round(0.6 * size_factor + 0.2 * profile.type_token_diversity, 4)
        delta = max(-0.2, min(0.7, delta))
    # Wall-clock model: ~0.5s per step on a small LoRA, capped at 10 minutes.
    wall_clock = min(600.0, max(5.0, 0.5 * n_steps))
    return float(delta), float(wall_clock)


# ---------------------------------------------------------------------------
# Rubric rendering (`soup advise explain`)
# ---------------------------------------------------------------------------

def format_verdict_rubric(verdict: Verdict) -> str:
    """Plain-text rubric: which rule fired, evidence, what flips it.

    Stable text output (not Rich markup) so callers can pipe to a file or
    a clipboard. CLI layer wraps with markup if desired.
    """
    if not isinstance(verdict, Verdict):
        raise TypeError("verdict must be a Verdict instance")

    roi = verdict.estimated_roi
    parts: List[str] = []
    parts.append(f"Choice:           {verdict.choice}")
    parts.append(f"Confidence:       {verdict.confidence:.2f}")
    parts.append(f"Task category:    {verdict.task_category}")
    parts.append("")
    parts.append("Reason:")
    parts.append(f"  {verdict.reason}")
    parts.append("")
    parts.append("Reverses when:")
    parts.append(f"  {verdict.reverse_when}")
    parts.append("")
    parts.append("ROI deltas:")
    parts.append(f"  prompt_eng: {_fmt_delta(roi.prompt_eng_delta)}")
    parts.append(f"  rag:        {_fmt_delta(roi.rag_delta)}")
    parts.append(f"  sft:        {_fmt_delta(roi.sft_delta)}")
    if roi.sft_wall_clock_secs is not None:
        parts.append(f"  sft ETA:    {roi.sft_wall_clock_secs:.0f}s")
    if roi.sft_cost_usd is not None:
        parts.append(f"  sft cost:   ${roi.sft_cost_usd:.2f}")
    parts.append("")
    parts.append(f"Next command:  {next_command_for(verdict)}")
    return "\n".join(parts)


def next_command_for(verdict: Verdict) -> str:
    """Render the literal next CLI command that follows from a verdict.

    Mirrors the operator-handoff convention used in v0.20–v0.21 issue
    close-comments ("Try: ``soup <command> <args>``"). Architect-review
    follow-up: surface a concrete next step instead of leaving the user
    to derive the autopilot/RAG/eval invocation from the choice string.
    """
    if not isinstance(verdict, Verdict):
        raise TypeError("verdict must be a Verdict instance")
    if verdict.choice == "PROMPT_ENG":
        return "soup chat --model <base>   # iterate on prompts first"
    if verdict.choice == "RAG":
        return (
            "# RAG is outside Soup core — wire your data into a vector store "
            "(pgvector / weaviate / faiss) and prompt the base model."
        )
    if verdict.choice == "DPO":
        return "soup autopilot --data <data.jsonl> --task dpo"
    if verdict.choice == "GRPO":
        return "soup autopilot --data <data.jsonl> --task grpo"
    if verdict.choice == "SFT":
        return "soup autopilot --data <data.jsonl> --task sft"
    return f"# Unknown choice {verdict.choice!r} — no recommended command"


def _fmt_delta(value: Optional[float]) -> str:
    if value is None:
        return "(not measured)"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "(invalid)"
    if not math.isfinite(value):
        return "(non-finite)"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"
