"""v0.71.32 — Pure-python ASR metrics (WER / CER).

Word- and character-level error rates via Levenshtein edit distance, with a
light Whisper-style text normalizer. No heavy dependency (no torch /
transformers / datasets) — usable from ``soup infer --task asr`` and reused as
``soup ship``'s task-win metric via :func:`word_accuracy` (= ``1 - WER``, so
ship's "higher is better" leg works unchanged).

Semantics:
- ``wer`` / ``cer`` are true error *rates* — edits divided by the reference
  length. They are unbounded above (many insertions can push WER past 1.0).
- Empty reference: ``0.0`` when the hypothesis is also empty, else ``1.0``.
- ``corpus_wer`` aggregates ``sum(edits) / sum(ref_len)`` across a set — the
  standard corpus-level WER, NOT the mean of per-example WERs.

The default normalizer lowercases, removes punctuation (including apostrophes —
``don't`` -> ``dont``, adequate for before/after deltas), and collapses
whitespace. It is intentionally lighter than the full Whisper English text
normalizer (numbers / abbreviations); see the module known-limitations note.
"""

from __future__ import annotations

import string
from typing import List, Sequence

# DoS guard: cap the token / char sequence length fed to the DP. A malformed
# or adversarial row must not allocate an O(n*m) table without bound.
_MAX_SEQ: int = 100_000

# Punctuation stripped by the default normalizer (removed, not spaced —
# mirrors jiwer's RemovePunctuation default).
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(
    text: str,
    *,
    lower: bool = True,
    strip_punct: bool = True,
    collapse_ws: bool = True,
) -> str:
    """Light text normalization for WER/CER comparison.

    Args:
        text: input string.
        lower: lowercase the text.
        strip_punct: remove ASCII punctuation characters.
        collapse_ws: collapse runs of whitespace to a single space and strip.

    Returns:
        The normalized string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    out = text
    if lower:
        out = out.lower()
    if strip_punct:
        out = out.translate(_PUNCT_TABLE)
    if collapse_ws:
        out = " ".join(out.split())
    return out


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Edit distance between two sequences via a bounded two-row DP."""
    if len(ref) > _MAX_SEQ or len(hyp) > _MAX_SEQ:
        raise ValueError(
            f"sequence too long for edit distance (> {_MAX_SEQ} units); "
            "split the input"
        )
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev: List[int] = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ref_i = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_i == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[m]


def _error_rate(ref_units: Sequence, hyp_units: Sequence) -> float:
    """edits / len(ref); empty-ref handled (0.0 both empty else 1.0)."""
    if len(ref_units) == 0:
        return 0.0 if len(hyp_units) == 0 else 1.0
    return _levenshtein(ref_units, hyp_units) / len(ref_units)


def wer(ref: str, hyp: str, *, normalize: bool = True) -> float:
    """Word error rate — word-level edit distance / reference word count."""
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    return _error_rate(ref.split(), hyp.split())


def cer(ref: str, hyp: str, *, normalize: bool = True) -> float:
    """Character error rate — char-level edit distance / reference char count."""
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    return _error_rate(list(ref), list(hyp))


def word_accuracy(ref: str, hyp: str, *, normalize: bool = True) -> float:
    """``max(0.0, 1 - WER)`` — the higher-is-better score ship consumes."""
    return max(0.0, 1.0 - wer(ref, hyp, normalize=normalize))


def corpus_wer(refs: Sequence[str], hyps: Sequence[str], *, normalize: bool = True) -> float:
    """Corpus-level WER = ``sum(edits) / sum(ref_words)`` over paired lists."""
    if len(refs) != len(hyps):
        raise ValueError(
            f"refs and hyps must be the same length ({len(refs)} != {len(hyps)})"
        )
    total_edits = 0
    total_ref = 0
    empty_ref_errors = 0
    for ref, hyp in zip(refs, hyps):
        if normalize:
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)
        ref_words = ref.split()
        hyp_words = hyp.split()
        if len(ref_words) == 0:
            # No reference words to divide by — count a full error when the
            # hypothesis is non-empty so the corpus rate is not silently 0.
            empty_ref_errors += 1 if len(hyp_words) else 0
            continue
        total_edits += _levenshtein(ref_words, hyp_words)
        total_ref += len(ref_words)
    if total_ref == 0:
        return 0.0 if empty_ref_errors == 0 else 1.0
    return total_edits / total_ref
