"""v0.62.0 Part D — Citation-faithful fine-tuning.

When enabled, the model is trained to cite document IDs verbatim from
the training corpus. Composes with v0.62.0 Part A RAFT — the RAFT row
already names the golden_doc + distractor_docs.

Schema-only release:

* The schema flag ``training.citation_faithful: bool`` opts INTO the
  citation-precision / recall scorer + a loss-mask rule that emphasises
  citation spans.
* The live span-mask training kernel and the eval-suite hook land in
  v0.62.1, mirroring v0.50.0 / v0.52.0 / v0.61.0 stub-then-live policy.

This module ships the pure ``score_citations`` kernel so callers (and
the eval gate) can compute precision/recall/F1 today, plus the closed
allowlist for ``citation_style``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

SUPPORTED_CITATION_STYLES: frozenset[str] = frozenset(
    {"bracket", "inline", "footnote"}
)

_MAX_STYLE_LEN: int = 32
_MAX_PREDICTED_LEN: int = 2_000_000  # 2 MB cap on per-row predicted text.
_MAX_EXPECTED_IDS: int = 10_000  # Per-row expected-citation cap.

# Default extraction regex — matches both ``[doc-id]`` brackets and bare
# ``doc-id`` tokens. The bracket form is the canonical RAFT default;
# ``inline`` and ``footnote`` use the same characters today (live
# per-style extractors ship in v0.62.1 once we benchmark variations).
_CITATION_RE: re.Pattern[str] = re.compile(
    r"\[(?P<bracketed>[A-Za-z0-9][A-Za-z0-9._\-]{0,127})\]"
)


@dataclass(frozen=True)
class CitationScore:
    """Precision / recall / F1 over predicted vs expected document IDs."""

    precision: float
    recall: float
    f1: float
    predicted_count: int
    expected_count: int


def validate_citation_style(value: object) -> str:
    """Normalise + validate a citation-style name.

    Mirrors v0.41.0 / v0.51.0 / v0.61.0 validator policy.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"citation_style must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"citation_style must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("citation_style must be non-empty")
    if "\x00" in value:
        raise ValueError("citation_style must not contain null bytes")
    if len(value) > _MAX_STYLE_LEN:
        raise ValueError(
            f"citation_style must be <= {_MAX_STYLE_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_CITATION_STYLES:
        supported = ", ".join(sorted(SUPPORTED_CITATION_STYLES))
        raise ValueError(
            f"unknown citation_style {value!r}; supported: {supported}"
        )
    return canonical


def validate_citation_threshold(value: object) -> float:
    """Validate the ``citation_recall_threshold`` in ``[0.0, 1.0]``.

    Bool-rejected, NaN/Inf-rejected via ``math.isfinite``.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"citation_recall_threshold must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"citation_recall_threshold must be a number, "
            f"got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(
            "citation_recall_threshold must be finite (no NaN / Inf)"
        )
    if fval < 0.0 or fval > 1.0:
        raise ValueError(
            f"citation_recall_threshold must be in [0.0, 1.0]; got {fval}"
        )
    return fval


def extract_citation_ids(text: str) -> tuple[str, ...]:
    """Extract every ``[doc-id]`` citation from ``text``.

    Returns a tuple of IDs in encounter order. Duplicates are preserved
    so the caller can compute precision honestly (a model that cites
    the same doc three times should not silently dedupe).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be str, got {type(text).__name__}"
        )
    if len(text) > _MAX_PREDICTED_LEN:
        raise ValueError(
            f"text must be <= {_MAX_PREDICTED_LEN} chars for citation extract"
        )
    return tuple(m.group("bracketed") for m in _CITATION_RE.finditer(text))


def score_citations(
    *,
    predicted: object,
    expected_ids: object,
) -> CitationScore:
    """Compute citation precision / recall / F1.

    Precision = |predicted ∩ expected| / |predicted|.
    Recall = |predicted ∩ expected| / |expected|.

    Undefined denominators (empty predicted / expected) return 0.0 by
    convention — same policy as v0.43.0 BLEU on zero-precision n-grams.
    """
    if isinstance(predicted, bool):
        raise TypeError(
            f"predicted must not be bool, got {predicted!r}"
        )
    if not isinstance(predicted, str):
        raise TypeError(
            f"predicted must be str, got {type(predicted).__name__}"
        )
    if len(predicted) > _MAX_PREDICTED_LEN:
        raise ValueError(
            f"predicted must be <= {_MAX_PREDICTED_LEN} chars"
        )
    if not isinstance(expected_ids, Iterable) or isinstance(expected_ids, str):
        raise TypeError(
            "expected_ids must be an iterable of strings (not a single str)"
        )
    expected_tuple = tuple(expected_ids)
    if len(expected_tuple) > _MAX_EXPECTED_IDS:
        raise ValueError(
            f"expected_ids must have <= {_MAX_EXPECTED_IDS} entries"
        )
    expected_set: set[str] = set()
    for index, eid in enumerate(expected_tuple):
        if isinstance(eid, bool) or not isinstance(eid, str):
            raise TypeError(
                f"expected_ids[{index}] must be str, got {type(eid).__name__}"
            )
        if not eid:
            raise ValueError(f"expected_ids[{index}] must be non-empty")
        if "\x00" in eid:
            raise ValueError(
                f"expected_ids[{index}] must not contain null bytes"
            )
        expected_set.add(eid)

    predicted_ids = extract_citation_ids(predicted)
    predicted_count = len(predicted_ids)
    expected_count = len(expected_set)

    if predicted_count == 0 or expected_count == 0:
        return CitationScore(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            predicted_count=predicted_count,
            expected_count=expected_count,
        )

    # Precision: predicted IDs that hit the expected set.
    hits = sum(1 for pid in predicted_ids if pid in expected_set)
    precision = hits / predicted_count

    # Recall: how many expected IDs the model actually cited.
    predicted_set = set(predicted_ids)
    recalled = sum(1 for eid in expected_set if eid in predicted_set)
    recall = recalled / expected_count

    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return CitationScore(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=predicted_count,
        expected_count=expected_count,
    )
