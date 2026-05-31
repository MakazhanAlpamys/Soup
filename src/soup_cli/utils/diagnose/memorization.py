"""Training-prefix echo probe (v0.56.0).

Given a training row's first ``prefix_fraction`` of tokens, the probe
asks the adapter to continue. If the adapter's continuation Jaccard-
overlaps the held-out suffix above a threshold, that's a memorization
signal. Score = 1 - mean(echo_rate) so 1.0 = healthy / no echo.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from soup_cli.utils.diagnose._common import (
    GeneratorFn,
    call_generator,
    extract_row_text,
    jaccard,
    merge_evidence,
    require_finite_unit,
    require_str,
    tokenize,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score


def split_prefix(text: str, *, fraction: float = 0.25) -> tuple:
    """Split text into (prefix, suffix) by word-count fraction."""
    require_str(text, "text", max_len=64 * 1024)
    require_finite_unit(fraction, "fraction")
    tokens = text.split()
    if not tokens:
        return ("", "")
    cut = max(1, int(len(tokens) * fraction))
    prefix = " ".join(tokens[:cut])
    suffix = " ".join(tokens[cut:])
    return (prefix, suffix)


def score_memorization(
    training_rows: Sequence[Mapping[str, object]],
    adapter_gen: GeneratorFn,
    *,
    prefix_fraction: float = 0.25,
    echo_threshold: float = 0.5,
) -> FailureScore:
    """Score memorization on a sample of training rows.

    Each row must contain a ``text`` (or ``content`` / ``prompt``) string
    field. Rows lacking text are skipped.
    """
    if not isinstance(training_rows, Sequence):
        raise TypeError("training_rows must be a sequence of dicts")
    if len(training_rows) > 5_000:
        raise ValueError("too many training rows (max 5_000)")
    require_finite_unit(prefix_fraction, "prefix_fraction")
    require_finite_unit(echo_threshold, "echo_threshold")
    echoes = []
    scanned = 0
    for _index, row in enumerate(training_rows):
        text = extract_row_text(row)
        if not text:
            continue
        prefix, suffix = split_prefix(text, fraction=prefix_fraction)
        if not suffix:
            continue
        scanned += 1
        completion = call_generator(adapter_gen, prefix)
        overlap = jaccard(tokenize(completion), tokenize(suffix))
        echoes.append(1.0 if overlap >= echo_threshold else 0.0)
        if scanned >= 1000:
            break
    if not echoes:
        return FailureScore(
            mode="memorization",
            score=1.0,
            verdict="OK",
            evidence="no rows with text+suffix; nothing to check",
        )
    echo_rate = sum(echoes) / len(echoes)
    score = max(0.0, min(1.0, 1.0 - echo_rate))
    verdict = classify_score(score)
    evidence = merge_evidence(
        {
            "scanned": scanned,
            "echo_rate": echo_rate,
            "threshold": echo_threshold,
            "prefix_fraction": prefix_fraction,
        }
    )
    return FailureScore(
        mode="memorization",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
