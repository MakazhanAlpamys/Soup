"""v0.69.0 Part E — Brain-rot detector (arXiv 2510.13928).

Scores dataset rows on two orthogonal "low-quality slop" axes:

- **Triviality** — short / repetitive / exclamation-heavy text dominates → high.
- **Popularity signal** — clickbait phrases ("you won't believe", "top 10") +
  excessive punctuation / emoji density → high.

Per-row brain-rot score is in [0, 1] where ``1.0`` = healthy substantive content
and ``0.0`` = pure slop (so we can reuse the OK/MINOR/MAJOR taxonomy from
v0.26 / v0.56 / v0.65 with the same threshold band — high score = good).

Composes with v0.47.0 ``score_educational_value`` (educational-value scorer),
which the live runner can mix in for a composite score; this module ships the
slop-detection kernel that the paper measures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional

from soup_cli.utils.brain_rot_lang import (
    SUPPORTED_LANGS,
    BrainRotLangBundle,
    get_lang_bundle,
)

BRAIN_ROT_VERDICTS = ("OK", "MINOR", "MAJOR")

_OK_THRESHOLD = 0.85
_MINOR_THRESHOLD = 0.60

_MAX_TEXT_LEN = 65_536
_PUNCT_PATTERN = re.compile(r"[!]{2,}|[?]{2,}")
_TEXT_FIELDS = ("text", "content", "output", "prompt", "instruction", "response")


@dataclass(frozen=True)
class BrainRotReport:
    """Outcome of scoring a dataset for brain-rot."""

    num_rows: int
    mean_score: float
    num_major: int
    num_minor: int
    num_ok: int
    overall_verdict: str

    def __post_init__(self) -> None:
        for field_name in ("num_rows", "num_major", "num_minor", "num_ok"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"BrainRotReport.{field_name} must be int"
                )
            if value < 0:
                raise ValueError(
                    f"BrainRotReport.{field_name} must be non-negative"
                )
        if isinstance(self.mean_score, bool):
            raise TypeError("BrainRotReport.mean_score must be float")
        if not isinstance(self.mean_score, (int, float)):
            raise TypeError("BrainRotReport.mean_score must be a number")
        if not math.isfinite(float(self.mean_score)):
            raise ValueError("BrainRotReport.mean_score must be finite")
        if not (0.0 <= float(self.mean_score) <= 1.0):
            raise ValueError(
                "BrainRotReport.mean_score must be in [0.0, 1.0]"
            )
        if self.overall_verdict not in BRAIN_ROT_VERDICTS:
            raise ValueError(
                f"overall_verdict must be one of {BRAIN_ROT_VERDICTS}"
            )


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------


def classify_brain_rot(score: object) -> str:
    """Classify a score in [0, 1] (1.0 = healthy) into OK / MINOR / MAJOR.

    Mirrors v0.26.0 Quant-Lobotomy / v0.56.0 diagnose / v0.65.0 behavior taxonomy:
    ``>= 0.85 → OK``, ``>= 0.60 → MINOR``, else ``MAJOR``.
    """
    if isinstance(score, bool):
        raise TypeError("score must be float, not bool")
    if not isinstance(score, (int, float)):
        raise TypeError(
            f"score must be a number, got {type(score).__name__}"
        )
    fscore = float(score)
    if not math.isfinite(fscore):
        raise ValueError("score must be finite")
    if not (0.0 <= fscore <= 1.0):
        raise ValueError("score must be in [0.0, 1.0]")
    if fscore >= _OK_THRESHOLD:
        return "OK"
    if fscore >= _MINOR_THRESHOLD:
        return "MINOR"
    return "MAJOR"


# -----------------------------------------------------------------------------
# Heuristic scorers
# -----------------------------------------------------------------------------


def _require_str(text: object, *, field: str = "text") -> str:
    if isinstance(text, bool):
        raise TypeError(f"{field} must be str, not bool")
    if not isinstance(text, str):
        raise TypeError(
            f"{field} must be str, got {type(text).__name__}"
        )
    if len(text) > _MAX_TEXT_LEN:
        return text[:_MAX_TEXT_LEN]
    return text


def _validate_lang_arg(lang: object) -> None:
    """Eager boundary check for ``lang`` kwarg on dataset-level scorers.

    Closes a gap surfaced in python-review: per-row resolution rejects a
    bool / null-byte / oversize lang, but only on the first iteration —
    an empty ``rows`` list would silently fall through. Validating upfront
    also catches operator typos before any scoring work happens.
    """
    if lang is None:
        return
    if isinstance(lang, bool):
        raise TypeError("lang must be str, not bool")
    if not isinstance(lang, str):
        raise TypeError(f"lang must be str, got {type(lang).__name__}")
    if "\x00" in lang:
        raise ValueError("lang must not contain null bytes")
    if len(lang) > 64:
        raise ValueError("lang must be <= 64 chars")


def _resolve_bundle(lang: Optional[str], *, text: str = "") -> BrainRotLangBundle:
    """Resolve a ``lang`` kwarg (incl. the ``"auto"`` sentinel) to a bundle.

    - ``None`` -> English bundle (backward-compat with the v0.69.0 surface).
    - ``"auto"`` -> probabilistic detection via :func:`data_score._langdetect_fast`;
      silent fallback to English when the optional ``[data-pro]`` ``langdetect``
      package is missing OR the detector returns ``unknown`` / a code not in
      :data:`SUPPORTED_LANGS`. This matches the issue acceptance criterion
      "``--lang auto`` falls back to English when language detection returns
      ``unknown`` or ``[data-pro]`` not installed".
    - Any other string -> :func:`get_lang_bundle` lookup (silent fallback to
      English on unknown codes; keeps the detector working on under-resourced
      corpora rather than crashing).
    """
    if lang is None:
        return get_lang_bundle(None)
    # Shape-check via get_lang_bundle's _check_lang_arg_shape for the
    # non-"auto" path; for "auto" we just inspect the canonical form.
    if not isinstance(lang, str) or isinstance(lang, bool):
        # get_lang_bundle would raise TypeError — preserve that.
        return get_lang_bundle(lang)  # raises TypeError
    canonical = lang.lower()
    if canonical != "auto":
        return get_lang_bundle(lang)
    # auto: probe via langdetect; lazy-import so a bare install still works.
    detected: Optional[str] = None
    if text:
        try:
            from soup_cli.utils.data_score import _langdetect_fast  # noqa: PLC0415

            detected = _langdetect_fast(text)
        except Exception:  # noqa: BLE001 — silent fallback per issue spec
            detected = None
    if detected and detected in SUPPORTED_LANGS:
        return get_lang_bundle(detected)
    return get_lang_bundle("en")


def score_triviality(text: object, *, lang: Optional[str] = None) -> float:
    """Higher = more trivial / repetitive / exclamation-heavy.

    Heuristic: punctuation-runs density + short-text penalty + token diversity
    inversion. Returns 1.0 for empty/unparseable input (worst case).

    ``lang`` selects a per-language token bundle (en / es / fr / de / ru) or
    ``"auto"`` for langdetect-driven detection. Default (``None``) preserves
    v0.69.0 English behaviour for backward-compat.
    """
    s = _require_str(text)
    if not s.strip():
        return 1.0
    bundle = _resolve_bundle(lang, text=s)
    tokens = s.lower().split()
    n = len(tokens)
    if n == 0:
        return 1.0
    unique = len(set(tokens))
    diversity = unique / n  # 1.0 = all unique, 0.0 = pathological
    # Length penalty: very short text is suspect for SFT.
    length_penalty = 1.0 if n >= 30 else (1.0 - n / 30.0)
    # Punctuation: int !!!! / ???? runs are slop markers.
    punct_hits = len(_PUNCT_PATTERN.findall(s))
    punct_density = min(1.0, punct_hits / max(1, n / 10))
    # Low-effort token density (per-language bundle).
    low_effort_set = set(bundle.low_effort_tokens)
    low_effort = sum(
        1 for tok in tokens if tok.strip("!?.,") in low_effort_set
    )
    low_effort_density = min(1.0, low_effort / max(1, n / 5))
    triviality = (
        0.2 * (1.0 - diversity)
        + 0.1 * length_penalty
        + 0.3 * punct_density
        + 0.4 * low_effort_density
    )
    return max(0.0, min(1.0, triviality))


def score_popularity_signal(
    text: object, *, lang: Optional[str] = None
) -> float:
    """Higher = clickbait / engagement-bait / popularity-optimised slop.

    Heuristic: substring scan against the per-language clickbait phrase
    bundle + emoji density. Returns 0.0 for empty input.

    ``lang`` selects a per-language phrase bundle (en / es / fr / de / ru)
    or ``"auto"`` for langdetect-driven detection. Default (``None``)
    preserves v0.69.0 English behaviour for backward-compat.
    """
    s = _require_str(text)
    if not s.strip():
        return 0.0
    bundle = _resolve_bundle(lang, text=s)
    lower = s.lower()
    hits = sum(1 for phrase in bundle.clickbait_phrases if phrase in lower)
    # Emoji density: count non-ASCII chars in [U+1F300, U+1FAFF] range
    # (covers most pictographs without importing emoji libs).
    emoji_hits = sum(1 for c in s if 0x1F300 <= ord(c) <= 0x1FAFF)
    n_tokens = len(s.split()) or 1
    phrase_density = min(1.0, hits / 2.0)  # 2+ clickbait phrases → max
    emoji_density = min(1.0, emoji_hits / max(1, n_tokens / 5))
    return max(0.0, min(1.0, 0.7 * phrase_density + 0.3 * emoji_density))


def _row_text(row: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in _TEXT_FIELDS:
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
    return "\n".join(parts)


def score_row_brain_rot(row: Any, *, lang: Optional[str] = None) -> float:
    """Return a per-row score in [0, 1]; 1.0 = healthy, 0.0 = pure slop.

    Composite: ``1 - max(triviality, popularity_signal)`` (worst-signal
    wins). Rows with no text fields return ``0.0`` (unjudgeable = worst).

    ``lang`` selects a per-language token + phrase bundle (en / es / fr /
    de / ru) or ``"auto"`` for langdetect-driven detection. Default
    (``None``) preserves v0.69.0 English behaviour for backward-compat.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            f"row must be a Mapping, got {type(row).__name__}"
        )
    text = _row_text(row)
    if not text:
        return 0.0
    triviality = score_triviality(text, lang=lang)
    popularity = score_popularity_signal(text, lang=lang)
    # Worst-signal composite: a single strong slop signal drives the score
    # down hard (mirrors v0.56.0 ``overall_verdict`` worst-case policy).
    score = 1.0 - max(triviality, popularity)
    return max(0.0, min(1.0, score))


def score_dataset_brain_rot(
    rows: Any, *, lang: Optional[str] = None
) -> BrainRotReport:
    """Score a dataset and return a frozen ``BrainRotReport``.

    Empty inputs return ``MAJOR`` (no signal = treat as broken).
    ``lang`` is threaded through to :func:`score_row_brain_rot` so the
    per-language token / phrase bundle is applied to every row.
    """
    if isinstance(rows, (str, bytes)) or not hasattr(rows, "__iter__"):
        raise TypeError("rows must be iterable")
    _validate_lang_arg(lang)
    scores: List[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scores.append(score_row_brain_rot(row, lang=lang))
    if not scores:
        return BrainRotReport(
            num_rows=0,
            mean_score=0.0,
            num_major=0,
            num_minor=0,
            num_ok=0,
            overall_verdict="MAJOR",
        )
    verdict_counts: Counter = Counter(classify_brain_rot(s) for s in scores)
    mean = sum(scores) / len(scores)
    overall = classify_brain_rot(mean)
    return BrainRotReport(
        num_rows=len(scores),
        mean_score=mean,
        num_major=verdict_counts.get("MAJOR", 0),
        num_minor=verdict_counts.get("MINOR", 0),
        num_ok=verdict_counts.get("OK", 0),
        overall_verdict=overall,
    )


def refuse_if_rotten(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_major_fraction: float = 0.25,
    lang: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` when too many rows score MAJOR brain-rot.

    Composes with v0.69.0 Part A's build pipeline so a transform can refuse to
    produce a tokenised dataset that's mostly slop. ``lang`` is threaded
    through to :func:`score_dataset_brain_rot` (per-language bundle picker).
    """
    if isinstance(max_major_fraction, bool):
        raise TypeError("max_major_fraction must be float, not bool")
    if not isinstance(max_major_fraction, (int, float)):
        raise TypeError("max_major_fraction must be a number")
    if not math.isfinite(float(max_major_fraction)):
        raise ValueError("max_major_fraction must be finite")
    if not (0.0 <= float(max_major_fraction) <= 1.0):
        raise ValueError("max_major_fraction must be in [0.0, 1.0]")
    _validate_lang_arg(lang)
    report = score_dataset_brain_rot(rows, lang=lang)
    if report.num_rows == 0:
        return  # no data → nothing to refuse
    fraction = report.num_major / report.num_rows
    if fraction > float(max_major_fraction):
        raise ValueError(
            f"brain-rot threshold exceeded: {fraction:.1%} of rows are MAJOR "
            f"(limit {float(max_major_fraction):.1%})"
        )


__all__ = [
    "BRAIN_ROT_VERDICTS",
    "BrainRotReport",
    "classify_brain_rot",
    "refuse_if_rotten",
    "score_dataset_brain_rot",
    "score_popularity_signal",
    "score_row_brain_rot",
    "score_triviality",
]
