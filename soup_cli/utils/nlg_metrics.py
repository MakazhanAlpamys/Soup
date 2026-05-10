"""v0.43.0 Part B — BLEU + ROUGE NLG metrics + effective_tokens_per_second.

Pure-Python implementations sufficient for unit-level eval. For research-grade
scoring users should still wire in `sacrebleu` / `rouge_score` via lm-eval; this
module provides a self-contained baseline that does not require those packages
so `soup eval custom --metric bleu` / `--metric rouge_l` works on a vanilla
install. The closed metric allowlist `NLG_METRICS` is shared with the schema
field validator on `EvalConfig.nlg_metrics`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

NLG_METRICS = frozenset({"bleu", "rouge_1", "rouge_2", "rouge_l"})

# Bounds matching v0.19.0 custom-eval policy.
_MAX_INPUT_CHARS = 1_000_000
_MAX_NGRAM = 4


def _tokenize(text: str) -> list[str]:
    """Word-level whitespace + punctuation-stripped tokenizer.

    Rejects null bytes / non-string / oversize input.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if "\x00" in text:
        raise ValueError("text must not contain null bytes")
    if len(text) > _MAX_INPUT_CHARS:
        raise ValueError(
            f"text length {len(text)} exceeds max {_MAX_INPUT_CHARS}"
        )
    # Word-piece-style tokenizer: lowercase + non-alphanumeric split.
    return [tok for tok in re.findall(r"[A-Za-z0-9]+", text.lower()) if tok]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or isinstance(n, bool):
        raise ValueError("n must be a positive int")
    if n > _MAX_NGRAM:
        raise ValueError(f"n must be <= {_MAX_NGRAM}")
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Corpus-level BLEU (single reference per prediction).

    Returns score in [0.0, 1.0]. Empty corpus returns 0.0.

    `smooth=True` (default) uses Chen & Cherry epsilon smoothing for
    *zero-correct* buckets where `total[n] > 0`. It does NOT cover
    *empty* buckets where `total[n] == 0` (e.g. predictions shorter than
    `max_n` tokens) — those force the score to 0.0 even with smoothing.
    """
    if isinstance(max_n, bool) or not isinstance(max_n, int):
        raise ValueError("max_n must be an int")
    if max_n < 1 or max_n > _MAX_NGRAM:
        raise ValueError(f"max_n must be in [1, {_MAX_NGRAM}]")

    pred_list = list(predictions)
    ref_list = list(references)
    if len(pred_list) != len(ref_list):
        raise ValueError(
            f"predictions ({len(pred_list)}) and references "
            f"({len(ref_list)}) must have the same length"
        )
    if not pred_list:
        return 0.0

    pred_lengths = 0
    ref_lengths = 0
    correct = [0] * max_n
    total = [0] * max_n

    for pred, ref in zip(pred_list, ref_list):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        pred_lengths += len(pred_tokens)
        ref_lengths += len(ref_tokens)
        for n in range(1, max_n + 1):
            if len(pred_tokens) < n:
                continue
            pred_ng = _ngrams(pred_tokens, n)
            ref_ng = _ngrams(ref_tokens, n)
            overlap = sum(min(c, ref_ng[ng]) for ng, c in pred_ng.items())
            correct[n - 1] += overlap
            total[n - 1] += sum(pred_ng.values())

    # Modified n-gram precision with Chen & Cherry smoothing for empty buckets.
    precisions: list[float] = []
    for n in range(max_n):
        if total[n] == 0:
            precisions.append(0.0)
            continue
        if correct[n] == 0 and smooth:
            precisions.append(1.0 / (2.0 ** (n + 1) * total[n]))
        else:
            precisions.append(correct[n] / total[n])

    # Standard BLEU: geometric mean over all max_n precisions; any zero
    # collapses the score to 0.0 unless smoothing is on.
    if any(p == 0.0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # Brevity penalty.
    if pred_lengths == 0:
        bp = 0.0
    elif pred_lengths > ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_lengths / pred_lengths)
    return float(bp * geo_mean)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of longest common subsequence — DP, O(len(a) * len(b))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_n_score(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    n: int = 1,
) -> float:
    """ROUGE-N F1 (corpus average over total pair count).

    Pairs where either side has fewer than `n` tokens contribute 0.0 to
    the average — short-sentence corpora are penalised, matching the
    `rouge-score` default.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be an int")
    if n < 1 or n > _MAX_NGRAM:
        raise ValueError(f"n must be in [1, {_MAX_NGRAM}]")
    pred_list = list(predictions)
    ref_list = list(references)
    if len(pred_list) != len(ref_list):
        raise ValueError(
            "predictions and references must have the same length"
        )
    if not pred_list:
        return 0.0

    f1_sum = 0.0
    for pred, ref in zip(pred_list, ref_list):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if len(pred_tokens) < n or len(ref_tokens) < n:
            continue
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        overlap = sum(min(c, ref_ng[ng]) for ng, c in pred_ng.items())
        if overlap == 0:
            continue
        precision = overlap / sum(pred_ng.values())
        recall = overlap / sum(ref_ng.values())
        if precision + recall > 0:
            f1_sum += 2 * precision * recall / (precision + recall)
    return f1_sum / len(pred_list)


def rouge_l_score(
    predictions: Iterable[str],
    references: Iterable[str],
) -> float:
    """ROUGE-L F1 — corpus-average sentence-level LCS."""
    pred_list = list(predictions)
    ref_list = list(references)
    if len(pred_list) != len(ref_list):
        raise ValueError(
            "predictions and references must have the same length"
        )
    if not pred_list:
        return 0.0
    f1_sum = 0.0
    for pred, ref in zip(pred_list, ref_list):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if not pred_tokens or not ref_tokens:
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall > 0:
            f1_sum += 2 * precision * recall / (precision + recall)
    return f1_sum / len(pred_list)


def compute_nlg_metric(
    metric: str,
    predictions: Iterable[str],
    references: Iterable[str],
) -> float:
    """Dispatch by canonical metric name."""
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    name = metric.lower()
    if name not in NLG_METRICS:
        supported = ", ".join(sorted(NLG_METRICS))
        raise ValueError(f"unknown nlg metric '{metric}'. Supported: {supported}")
    if name == "bleu":
        return bleu_score(predictions, references)
    if name == "rouge_1":
        return rouge_n_score(predictions, references, n=1)
    if name == "rouge_2":
        return rouge_n_score(predictions, references, n=2)
    if name == "rouge_l":
        return rouge_l_score(predictions, references)
    raise AssertionError("unreachable")  # pragma: no cover


def effective_tokens_per_second(
    *,
    unmasked_tokens: int,
    wall_clock_seconds: float,
) -> float | None:
    """Effective tokens-per-second (LF metric).

    `unmasked_tokens` = total non-padding labels seen during training.
    Returns None when wall_clock <= 0 (avoid div-by-zero rather than fabricate).
    """
    if isinstance(unmasked_tokens, bool) or not isinstance(unmasked_tokens, int):
        raise ValueError("unmasked_tokens must be an int")
    if unmasked_tokens < 0:
        raise ValueError("unmasked_tokens must be >= 0")
    if isinstance(wall_clock_seconds, bool) or not isinstance(
        wall_clock_seconds, (int, float)
    ):
        raise ValueError("wall_clock_seconds must be a number")
    if not math.isfinite(float(wall_clock_seconds)):
        raise ValueError("wall_clock_seconds must be finite")
    if wall_clock_seconds <= 0:
        return None
    return unmasked_tokens / float(wall_clock_seconds)
