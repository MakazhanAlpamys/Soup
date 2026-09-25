"""Shared text-extraction helpers for v0.55.0 eval-design + canary-discovery.

Extracted out of `eval_design.py` so `canary_discovery.py` no longer
imports private symbols across modules (code-review LOW fix — removes
hidden coupling).

Pure functions — no I/O, no torch.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import List

# Small stop-words list used by the TF-IDF salience clustering; just
# enough to filter the dominant function-word tokens.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
        "from", "have", "i", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "will", "with", "you",
    }
)


def _is_no_space(c: str) -> bool:
    cp = ord(c)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        or (0x0E00 <= cp <= 0x0E7F)
    )


def row_text(row: Mapping[str, object]) -> str:
    """Best-effort text extraction from the *output* side of a dataset row.

    Soup datasets normalise to ``{"messages": [...]}``; this helper
    pulls the assistant turn(s) when present, falling back to common
    SFT fields. Returns an empty string on missing data.
    """
    if not isinstance(row, Mapping):
        return ""
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        chunks: List[str] = []
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    chunks.append(content)
        if chunks:
            return "\n".join(chunks)
    for key in ("output", "completion", "chosen", "response", "answer", "text"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def tokenize(text: str, *, filter_stopwords: bool = True) -> List[str]:
    """Tokenise to lowercase word/subword runs across Latin and non-Latin scripts.

    Normalises Unicode (NFC, lowercase, dotless/dotted I handling), supports
    non-Latin alphabetic scripts (Cyrillic, Greek, Arabic, Hebrew, Devanagari,
    etc.), non-spaced scripts (CJK, Kana, Thai via character bigrams), and
    digits. When ``filter_stopwords=True``, filters English stopwords and short
    ASCII words <= 2 characters.

    Returns an empty list for empty / non-string input.
    """
    if not text or not isinstance(text, str):
        return []
    norm = (
        unicodedata.normalize("NFC", text)
        .replace("İ", "i")
        .lower()
        .replace("\u0307", "")
    )
    raw_tokens: List[str] = []
    current: List[str] = []

    for ch in norm:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N") or ch in ("_", "-"):
            current.append(ch)
        else:
            if current:
                raw_tokens.append("".join(current))
                current = []
    if current:
        raw_tokens.append("".join(current))

    result: List[str] = []
    for tok in raw_tokens:
        clean = tok.strip("-_")
        if not clean:
            continue
        if any(_is_no_space(c) for c in clean):
            if len(clean) == 1:
                result.append(clean)
            else:
                for i in range(len(clean) - 1):
                    result.append(clean[i : i + 2])
        else:
            if filter_stopwords:
                if clean in STOPWORDS:
                    continue
                if clean.isascii() and clean.isalpha() and len(clean) <= 2:
                    continue
            result.append(clean)
    return result
