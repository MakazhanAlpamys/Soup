"""Training-data contamination probe (v0.56.0).

Reuses the v0.47 ``data_score.ngram_overlap_ratio`` containment policy:
counts the fraction of training rows whose n-grams cover a public
benchmark row above a threshold. Score = 1 - contamination_rate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from soup_cli.utils.diagnose._common import (
    extract_row_text,
    merge_evidence,
    reject_bool,
    require_finite_unit,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score

_row_text = extract_row_text  # back-compat alias for direct callers


def score_contamination(
    training_rows: Sequence[Mapping[str, object]],
    benchmark_corpus: Sequence,
    *,
    n: int = 8,
    threshold: float = 0.5,
) -> FailureScore:
    """Score 1 - fraction of training rows overlapping any benchmark row.

    ``benchmark_corpus`` may be a sequence of strings OR a sequence of
    Mappings carrying a ``text``/``content``/``prompt``/``instruction``
    field.
    """
    if not isinstance(training_rows, Sequence):
        raise TypeError("training_rows must be a sequence")
    if not isinstance(benchmark_corpus, Sequence):
        raise TypeError("benchmark_corpus must be a sequence")
    reject_bool(n, "n")
    if not isinstance(n, int):
        raise TypeError("n must be int")
    if n < 1 or n > 32:
        raise ValueError("n must be in [1, 32]")
    require_finite_unit(threshold, "threshold")
    if len(training_rows) > 100_000:
        raise ValueError("too many training rows (max 100_000)")
    if len(benchmark_corpus) > 100_000:
        raise ValueError("too many benchmark rows (max 100_000)")
    # Combined-complexity cap (python-review MEDIUM fix) — worst-case
    # nested scan is O(N×M) n-gram set ops; reject when the product
    # would exceed 1 e9 to prevent operator-side DoS.
    if len(training_rows) * len(benchmark_corpus) > 1_000_000_000:
        raise ValueError(
            "training_rows × benchmark_corpus exceeds 1e9 (combined-complexity cap)"
        )
    # Lazy import — keeps utils/diagnose import-cheap.
    from soup_cli.utils.data_score import ngram_overlap_ratio

    benchmark_texts = []
    for entry in benchmark_corpus:
        text = entry if isinstance(entry, str) else _row_text(entry)
        if isinstance(text, str) and text.strip():
            benchmark_texts.append(text)
    if not benchmark_texts:
        return FailureScore(
            mode="contamination",
            score=1.0,
            verdict="OK",
            evidence="no benchmark corpus; nothing to compare",
        )
    contaminated = 0
    scanned = 0
    for row in training_rows:
        text = _row_text(row)
        if not text:
            continue
        scanned += 1
        for bench in benchmark_texts:
            if ngram_overlap_ratio(text, bench, n=n) >= threshold:
                contaminated += 1
                break
    if scanned == 0:
        return FailureScore(
            mode="contamination",
            score=1.0,
            verdict="OK",
            evidence="no scannable training rows",
        )
    rate = contaminated / scanned
    score = max(0.0, min(1.0, 1.0 - rate))
    verdict = classify_score(score)
    evidence = merge_evidence(
        {
            "scanned": scanned,
            "contaminated": contaminated,
            "rate": rate,
            "ngram_n": n,
            "threshold": threshold,
        }
    )
    return FailureScore(
        mode="contamination",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
