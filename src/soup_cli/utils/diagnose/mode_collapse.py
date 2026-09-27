"""Mode-collapse probe (v0.56.0).

Diversity collapse signal: generate K completions per prompt at
temperature=0 and temperature=1, then score 1 - self-BLEU-style overlap
(actually averaged pairwise n-gram Jaccard distance — pure-python, no
nltk).
"""

from __future__ import annotations

from typing import Callable, Sequence

from soup_cli.utils.diagnose._common import (
    jaccard,
    merge_evidence,
    ngrams,
    reject_bool,
    require_prompts,
    tokenize,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score

# Generator that emits K completions per prompt; (prompt, k) -> list[str].
MultiGen = Callable[[str, int], Sequence[str]]


def _pairwise_diversity(samples: Sequence[str], *, n: int = 3) -> float:
    """1 - average pairwise n-gram-set Jaccard; 1.0 = fully diverse."""
    cleaned = [tokenize(sample) for sample in samples if isinstance(sample, str)]
    if len(cleaned) < 2:
        return 1.0
    pairs = 0
    overlap = 0.0
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            a = ngrams(cleaned[i], n) or [tuple(cleaned[i])]
            b = ngrams(cleaned[j], n) or [tuple(cleaned[j])]
            overlap += jaccard(a, b)
            pairs += 1
    if pairs == 0:
        return 1.0
    return max(0.0, 1.0 - overlap / pairs)


def score_mode_collapse(
    prompts: Sequence[str],
    adapter_multi_gen: MultiGen,
    *,
    k: int = 4,
    ngram_n: int = 3,
) -> FailureScore:
    """Score 1.0 - mean pairwise overlap across K completions per prompt."""
    reject_bool(k, "k")
    if not isinstance(k, int):
        raise TypeError("k must be int")
    if k < 2 or k > 32:
        raise ValueError("k must be in [2, 32]")
    reject_bool(ngram_n, "ngram_n")
    if not isinstance(ngram_n, int):
        raise TypeError("ngram_n must be int")
    if ngram_n < 1 or ngram_n > 8:
        raise ValueError("ngram_n must be in [1, 8]")
    if not callable(adapter_multi_gen):
        raise TypeError("adapter_multi_gen must be callable")
    prompts_list = require_prompts(prompts, max_count=2_000)
    if not prompts_list:
        return FailureScore(
            mode="mode_collapse",
            score=1.0,
            verdict="OK",
            evidence="no prompts; nothing to check",
        )
    diversities = []
    for prompt in prompts_list:
        samples = adapter_multi_gen(prompt, k)
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("adapter_multi_gen must return a sequence of str")
        diversities.append(_pairwise_diversity(samples, n=ngram_n))
    score = sum(diversities) / len(diversities)
    score = max(0.0, min(1.0, score))
    verdict = classify_score(score)
    evidence = merge_evidence(
        {
            "prompts": len(prompts_list),
            "k": k,
            "ngram_n": ngram_n,
            "mean_diversity": score,
        }
    )
    return FailureScore(
        mode="mode_collapse",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
