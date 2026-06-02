"""Training-prefix echo probe (v0.56.0).

Given a training row's first ``prefix_fraction`` of tokens, the probe
asks the adapter to continue. If the adapter's continuation Jaccard-
overlaps the held-out suffix above a threshold, that's a memorization
signal. Score = 1 - mean(echo_rate) so 1.0 = healthy / no echo.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from soup_cli.utils.diagnose._common import (
    GeneratorFn,
    call_generator,
    decode_ids,
    encode_ids,
    extract_row_text,
    jaccard,
    merge_evidence,
    require_finite_unit,
    require_str,
    resolve_tokenizer,
    subword_tokens,
    tokenize,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score


def split_prefix(
    text: str,
    *,
    fraction: float = 0.25,
    tokenizer: Optional[object] = None,
) -> tuple[str, str]:
    """Split text into (prefix, suffix).

    Default: split by whitespace word-count fraction. When ``tokenizer`` is
    supplied (v0.71.6 #167 — an HF model id / path string or a duck-typed
    tokenizer object), split on the actual token-id boundary so the prefix
    never ends mid-sub-word — the honest sub-word/BPE memorization probe.
    """
    require_str(text, "text", max_len=64 * 1024)
    require_finite_unit(fraction, "fraction")
    tok = resolve_tokenizer(tokenizer) if tokenizer is not None else None
    return _split_with_resolved(text, fraction, tok)


def _split_with_resolved(
    text: str, fraction: float, tok: Optional[object]
) -> tuple[str, str]:
    """Core split given an ALREADY-resolved tokenizer (or None for whitespace).

    Separated so :func:`score_memorization` can resolve the tokenizer ONCE and
    reuse it across rows rather than re-running resolution per row.
    """
    if tok is None:
        tokens = text.split()
        if not tokens:
            return ("", "")
        cut = max(1, int(len(tokens) * fraction))
        return (" ".join(tokens[:cut]), " ".join(tokens[cut:]))
    ids = encode_ids(tok, text)
    if not ids:
        return ("", "")
    cut = max(1, int(len(ids) * fraction))
    return (decode_ids(tok, ids[:cut]), decode_ids(tok, ids[cut:]))


def score_memorization(
    training_rows: Sequence[Mapping[str, object]],
    adapter_gen: GeneratorFn,
    *,
    prefix_fraction: float = 0.25,
    echo_threshold: float = 0.5,
    tokenizer: Optional[object] = None,
) -> FailureScore:
    """Score memorization on a sample of training rows.

    Each row must contain a ``text`` (or ``content`` / ``prompt``) string
    field. Rows lacking text are skipped.

    When ``tokenizer`` is supplied (v0.71.6 #167) the prefix/suffix split AND
    the echo-overlap are computed over sub-word tokens (resolved ONCE up front,
    not per row) instead of whitespace words — catching BPE-level memorization
    that whitespace tokenisation misses. The live ``soup diagnose`` wiring of
    ``--tokenizer`` lands with the live probe runner (#165).
    """
    if not isinstance(training_rows, Sequence):
        raise TypeError("training_rows must be a sequence of dicts")
    if len(training_rows) > 5_000:
        raise ValueError("too many training rows (max 5_000)")
    require_finite_unit(prefix_fraction, "prefix_fraction")
    require_finite_unit(echo_threshold, "echo_threshold")
    tok = resolve_tokenizer(tokenizer) if tokenizer is not None else None

    def _overlap_tokens(value: str) -> list:
        return subword_tokens(tok, value) if tok is not None else tokenize(value)

    echoes = []
    scanned = 0
    for row in training_rows:
        text = extract_row_text(row)
        if not text:
            continue
        # Reuse the once-resolved tokenizer (no per-row re-resolution).
        prefix, suffix = _split_with_resolved(text, prefix_fraction, tok)
        if not suffix:
            continue
        scanned += 1
        completion = call_generator(adapter_gen, prefix)
        overlap = jaccard(_overlap_tokens(completion), _overlap_tokens(suffix))
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
            "tokenizer_aware": tok is not None,
        }
    )
    return FailureScore(
        mode="memorization",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
