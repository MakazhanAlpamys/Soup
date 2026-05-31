"""Shared text-extraction helpers for v0.55.0 eval-design + canary-discovery.

Extracted out of `eval_design.py` so `canary_discovery.py` no longer
imports private symbols across modules (code-review LOW fix — removes
hidden coupling).

Pure functions — no I/O, no torch.
"""

from __future__ import annotations

import re
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


def tokenize(text: str) -> List[str]:
    """Tokenise to lowercase alphanumeric runs, filtering stop-words.

    Returns an empty list for empty / non-string input.
    """
    if not text:
        return []
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,30}", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]
