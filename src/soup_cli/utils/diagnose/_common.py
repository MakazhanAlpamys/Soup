"""Shared helpers for the 6 v0.56.0 diagnose probes.

Pure functions only; no torch / transformers imports. Each public probe
takes either a list of pre-recorded ``(prompt, completion)`` pairs OR a
callable generator so tests can run without GPU and the live wiring lands
in v0.56.1 alongside the same forward-compat kwarg policy used by
v0.54.0 ``synth_probe_*``.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

# Generator protocol: (prompt: str) -> str. Live model loaders implement
# this in v0.56.1; tests inject closures.
GeneratorFn = Callable[[str], str]


def reject_bool(value: object, field: str) -> None:
    """Reject bool-as-int / float at the public boundary."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be int/float, not bool")


def require_finite_unit(value: object, field: str) -> float:
    """Validate ``value`` is a finite float in ``[0, 1]``; return the float."""
    reject_bool(value, field)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be float, got {type(value).__name__}")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(f"{field} must be finite")
    if not 0.0 <= fvalue <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {fvalue}")
    return fvalue


def require_str(value: object, field: str, *, max_len: int = 4096) -> str:
    """Validate string field with null-byte rejection + length cap."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} too long (max {max_len} chars)")
    return value


def require_prompts(prompts: object, *, max_count: int = 10_000) -> list:
    """Validate a sequence of prompt strings; return a list copy."""
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
        raise TypeError("prompts must be a sequence of str")
    if len(prompts) > max_count:
        raise ValueError(f"too many prompts (max {max_count})")
    out: list = []
    for index, value in enumerate(prompts):
        out.append(require_str(value, f"prompts[{index}]", max_len=8192))
    return out


def call_generator(gen: object, prompt: str) -> str:
    """Call a generator with one prompt; surface a typed TypeError on misuse."""
    if not callable(gen):
        raise TypeError("generator must be callable")
    result = gen(prompt)
    if not isinstance(result, str):
        raise TypeError("generator must return str")
    return result


def tokenize(text: str) -> list:
    """Cheap whitespace tokeniser — delegates to ``utils/_eval_text``.

    Kept as a thin shim so the canonical implementation lives in one
    place (code-review MEDIUM fix — prevents drift if `_eval_text`
    changes its tokenisation policy in v0.56.x).
    """
    from soup_cli.utils._eval_text import tokenize as _shared

    return list(_shared(text))


def extract_row_text(row: object) -> str:
    """Canonical text-extraction for dataset rows.

    Mirrors ``utils/_eval_text.extract_row_text`` but adds the
    ``instruction`` fallback used by alpaca-style training data so
    contamination + memorization probes can score rows that lack a
    ``text``/``content``/``prompt`` field. Returns "" when no
    candidate field is present.
    """
    if not isinstance(row, Mapping):
        return ""
    for key in ("text", "content", "prompt", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages") if isinstance(row, Mapping) else None
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "\n".join(parts)
    return ""


_MAX_TOKENIZE_IDS = 1_000_000


def resolve_tokenizer(tokenizer: object) -> object:
    """Return a tokenizer object from a name (lazy AutoTokenizer) or object.

    Mirrors v0.71.5 ``prune_prompt._resolve_tokenizer``: a duck-typed object
    with ``encode`` + ``decode`` is returned as-is (the injectable test seam);
    a string is treated as an HF model id / local path and lazy-loaded via
    ``transformers.AutoTokenizer`` so importing this module never pulls
    transformers.
    """
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        return tokenizer
    if not isinstance(tokenizer, str):
        raise TypeError(
            "tokenizer must be a model id / path string or a tokenizer object"
        )
    if not tokenizer:
        raise ValueError("tokenizer name must be non-empty")
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(
            "tokenizer-aware diagnose needs transformers — "
            "install with: pip install 'soup-cli[train]'"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(tokenizer)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message.
        raise ValueError(
            f"could not load tokenizer {tokenizer!r}: {type(exc).__name__}: {exc}"
        ) from exc


def encode_ids(tok: object, text: str) -> list[int]:
    """Encode ``text`` to token IDs (no special tokens), capped per call."""
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
    return list(ids)[:_MAX_TOKENIZE_IDS]


def decode_ids(tok: object, ids: Sequence[int]) -> str:
    """Decode token IDs back to text (skip special tokens when supported)."""
    try:
        return tok.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tok.decode(ids)


def subword_tokens(tok: object, text: str) -> list[str]:
    """Return sub-word token strings for ``text`` (for set-overlap scoring)."""
    ids = encode_ids(tok, text)
    convert = getattr(tok, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            return list(convert(ids))
        except Exception:  # noqa: BLE001 — fall back to id strings
            pass
    return [str(i) for i in ids]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Set-Jaccard similarity; empty sets → 1.0 (identical absence)."""
    set_left = set(left)
    set_right = set(right)
    if not set_left and not set_right:
        return 1.0
    union = set_left | set_right
    if not union:
        return 1.0
    return len(set_left & set_right) / len(union)


def ngrams(tokens: Sequence[str], n: int) -> list:
    """Materialise ``n``-grams over the token sequence."""
    reject_bool(n, "n")
    if not isinstance(n, int):
        raise TypeError("n must be int")
    if n < 1 or n > 32:
        raise ValueError("n must be in [1, 32]")
    if len(tokens) < n:
        return []
    return [tuple(tokens[start : start + n]) for start in range(len(tokens) - n + 1)]


def average(values: Iterable[float]) -> float:
    """Arithmetic mean; empty → 0.0."""
    total = 0.0
    count = 0
    for value in values:
        if not math.isfinite(float(value)):
            continue
        total += float(value)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def merge_evidence(parts: Mapping[str, object]) -> str:
    """Render a compact ``k=v`` evidence string (≤4096 chars)."""
    chunks = []
    for key, value in parts.items():
        if isinstance(value, float):
            chunks.append(f"{key}={value:.3f}")
        else:
            chunks.append(f"{key}={value}")
    evidence = " ".join(chunks)
    if len(evidence) > 4096:
        evidence = evidence[:4093] + "..."
    return evidence
