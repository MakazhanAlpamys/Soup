"""`soup prune-prompt` — detect + strip a shared system-prompt prefix.

Mines a JSONL of prompts (typically the output of `soup ingest`) for a
static prefix that appears in >= `min_frequency` of rows, then strips it
from the training data so the fine-tuned model internalises the prefix
instead of needing it pinned at inference time. OpenPipe's signature
trick, OSS.

Why this matters: production LLM apps often pin a multi-paragraph system
prompt to every request. Fine-tuning with that prefix wastes tokens (the
model learns to copy what's already in context). Stripping it teaches the
model the behaviour directly so deployments save tokens + latency.

Algorithm:
1. Sample up to ``_MAX_SCAN_ROWS`` rows.
2. Find the longest character prefix that appears in >= ``min_frequency``
   fraction of rows. We use a streaming two-pass approach:
   pass 1 collects candidate prefixes of growing length;
   pass 2 picks the longest one that clears the threshold.
3. Cap any individual row scan at ``_MAX_ROW_CHARS`` so a pathological
   row never blocks the pipeline.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from soup_cli.utils.paths import is_under_cwd

# DoS caps
_MAX_SCAN_ROWS = 100_000
_MAX_ROW_CHARS = 1_000_000  # 1 MB / row
_MAX_PREFIX_LEN = 100_000  # hard cap on returned prefix length
_MAX_TOKENS_PER_ROW = 50_000  # token-aware mode (v0.71.5 #205) per-row cap

# Tunable: a frequency below this is meaningless (we want a *near-universal*
# prefix). Operator can pick anything in [0, 1] via --min-frequency.
_DEFAULT_MIN_FREQUENCY = 0.95


@dataclass(frozen=True)
class PrunePromptReport:
    """Result of a prune-prompt pass."""

    prefix: str
    prefix_chars: int
    rows_total: int
    rows_pruned: int
    min_frequency: float

    def __post_init__(self) -> None:
        if self.rows_total < 0:
            raise ValueError("rows_total must be >= 0")
        if self.rows_pruned < 0:
            raise ValueError("rows_pruned must be >= 0")
        if self.rows_pruned > self.rows_total:
            raise ValueError(
                f"rows_pruned ({self.rows_pruned}) cannot exceed rows_total "
                f"({self.rows_total})"
            )


def validate_min_frequency(value: object) -> float:
    """Validate ``min_frequency`` is a finite float in [0.0, 1.0].

    Mirrors v0.41.0 Part B / v0.50.0 / v0.62.0 numeric validator policy:
    explicit bool-first rejection, NaN/Inf rejection via ``math.isfinite``.
    """
    if isinstance(value, bool):
        raise TypeError("min_frequency must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"min_frequency must be a number, got {type(value).__name__}"
        )
    f_value = float(value)
    if not math.isfinite(f_value):
        raise ValueError("min_frequency must be finite (no NaN / Inf)")
    if not (0.0 <= f_value <= 1.0):
        raise ValueError(
            f"min_frequency must be in [0.0, 1.0], got {f_value}"
        )
    return f_value


def detect_common_prefix(
    rows: Sequence[str],
    *,
    min_frequency: float,
) -> str:
    """Return the longest prefix shared by >= min_frequency of rows.

    Empty / single-row / no-overlap fall through to "" except the trivial
    single-row case at ``min_frequency=1.0`` where the entire row IS the
    common prefix by definition.
    """
    threshold = validate_min_frequency(min_frequency)

    # Sequence input check — strings ARE sequences, reject them explicitly
    # otherwise iteration yields characters and not rows.
    if isinstance(rows, str) or not hasattr(rows, "__iter__"):
        raise TypeError(
            f"rows must be an iterable of strings, got {type(rows).__name__}"
        )

    materialised: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, str):
            raise TypeError(
                f"rows[{idx}] must be str, got {type(row).__name__}"
            )
        # Per-row length cap (DoS defence).
        if len(row) > _MAX_ROW_CHARS:
            materialised.append(row[:_MAX_ROW_CHARS])
        else:
            materialised.append(row)
        if len(materialised) >= _MAX_SCAN_ROWS:
            break

    if not materialised:
        return ""

    if len(materialised) == 1:
        # Single-row sentinel — only the trivial 100% case yields a prefix.
        if threshold >= 1.0:
            return materialised[0][:_MAX_PREFIX_LEN]
        return ""

    # Find the longest threshold-meeting prefix by binary-searching over
    # candidate templates (up to 32 of them for cost). Even when 100% of
    # rows share a short prefix, a longer prefix MAY be shared by a
    # threshold-meeting majority — so we never early-exit on the 100%
    # match (code-review HIGH fix v0.63.0: returning the universal prefix
    # before the binary search ran was returning the *shortest* qualifying
    # prefix instead of the *longest*).
    need = max(1, int(math.ceil(threshold * len(materialised))))
    best_prefix = ""
    # Try each row as a template, cap candidates to first N for cost
    # (templates beyond the 32nd add no information in practice).
    sample_templates = materialised[: min(32, len(materialised))]
    for template in sample_templates:
        # Binary-search the longest length L for which >= need rows share
        # the first L chars of `template`.
        lo, hi = 0, min(len(template), _MAX_PREFIX_LEN)
        best_len = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                best_len = max(best_len, 0)
                lo = mid + 1
                continue
            pfx = template[:mid]
            count = sum(1 for r in materialised if r.startswith(pfx))
            if count >= need:
                best_len = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_len > len(best_prefix):
            best_prefix = template[:best_len]
    return best_prefix


def detect_common_prefix_tokens(
    token_rows: Sequence[Sequence[int]],
    *,
    min_frequency: float,
) -> List[int]:
    """Token-level analogue of :func:`detect_common_prefix` (v0.71.5 #205).

    Returns the longest token-id prefix shared by >= ``min_frequency`` of
    rows. Operating on token IDs (not characters) guarantees the prefix
    always ends on a token boundary — a multi-byte UTF-8 sequence can never
    be split mid-code-point.
    """
    threshold = validate_min_frequency(min_frequency)
    if isinstance(token_rows, (str, bytes)) or not hasattr(token_rows, "__iter__"):
        raise TypeError(
            f"token_rows must be an iterable of int sequences, "
            f"got {type(token_rows).__name__}"
        )

    materialised: List[List[int]] = []
    for idx, row in enumerate(token_rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise TypeError(
                f"token_rows[{idx}] must be a sequence of ints, "
                f"got {type(row).__name__}"
            )
        materialised.append(list(row)[:_MAX_TOKENS_PER_ROW])
        if len(materialised) >= _MAX_SCAN_ROWS:
            break

    if not materialised:
        return []
    if len(materialised) == 1:
        return list(materialised[0]) if threshold >= 1.0 else []

    need = max(1, int(math.ceil(threshold * len(materialised))))
    best_prefix: List[int] = []
    sample_templates = materialised[: min(32, len(materialised))]
    for template in sample_templates:
        lo, hi = 0, len(template)
        best_len = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                lo = mid + 1
                continue
            pfx = template[:mid]
            count = sum(1 for r in materialised if r[:mid] == pfx)
            if count >= need:
                best_len = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_len > len(best_prefix):
            best_prefix = template[:best_len]
    return best_prefix


def _resolve_tokenizer(tokenizer: Union[str, Any]) -> Any:
    """Return a tokenizer object from a name (lazy AutoTokenizer) or object.

    A pre-built tokenizer-like object (duck-typed ``encode`` / ``decode``)
    is returned as-is — this is the injectable test seam + lets advanced
    callers pass an already-loaded tokenizer. A string is treated as an HF
    model id / local path and lazy-loaded via ``transformers.AutoTokenizer``
    (so importing this module never pulls transformers).
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
            "tokenizer-aware prune-prompt needs transformers — "
            "install with: pip install 'soup-cli[train]'"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(tokenizer)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message.
        raise ValueError(
            f"could not load tokenizer {tokenizer!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _encode(tok: Any, text: str) -> List[int]:
    """Encode ``text`` to token IDs (no special tokens), capped per-row."""
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        # Tokenizers / fakes without the kwarg.
        ids = tok.encode(text)
    return list(ids)[:_MAX_TOKENS_PER_ROW]


def prune_traces(
    input_path: str,
    *,
    output_path: str,
    min_frequency: float = _DEFAULT_MIN_FREQUENCY,
    tokenizer: Optional[Union[str, Any]] = None,
) -> PrunePromptReport:
    """Read a JSONL of {prompt, output} rows, strip shared prefix, write.

    Returns a :class:`PrunePromptReport` summarising the pass. Output JSONL
    contains every input row with the shared prefix stripped from the
    ``prompt`` field (other fields untouched). When no prefix clears the
    threshold, the output is byte-identical to the input plus a
    ``rows_pruned=0`` report.

    v0.71.5 #205: pass ``tokenizer`` (an HF model id / local path string, or
    a pre-built tokenizer object) to detect + strip the prefix on token
    boundaries instead of characters — the prefix can then never end
    mid-UTF-8-code-point. Default (``None``) keeps the whitespace-character
    behaviour.
    """
    threshold = validate_min_frequency(min_frequency)

    if not isinstance(input_path, str):
        raise TypeError(
            f"input_path must be str, got {type(input_path).__name__}"
        )
    if not isinstance(output_path, str):
        raise TypeError(
            f"output_path must be str, got {type(output_path).__name__}"
        )
    if not input_path or not output_path:
        raise ValueError("input_path and output_path must be non-empty")
    if "\x00" in input_path or "\x00" in output_path:
        raise ValueError("paths must not contain null bytes")
    if not is_under_cwd(input_path):
        raise ValueError(f"input_path {input_path!r} is outside cwd")
    if not is_under_cwd(output_path):
        raise ValueError(f"output_path {output_path!r} is outside cwd")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    # Resolve the tokenizer up front (fails fast on a bad name even on an
    # empty input file) — None keeps the legacy character path.
    tok = _resolve_tokenizer(tokenizer) if tokenizer is not None else None

    # First pass: collect prompts (capped).
    prompts: list[str] = []
    rows_total = 0
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rows_total += 1
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                prompts.append(prompt)
            if len(prompts) >= _MAX_SCAN_ROWS:
                # Stop reading once we've sampled enough rows to identify
                # the prefix; the second pass below re-streams the file and
                # strips even rows we didn't scan (code-review HIGH fix
                # v0.63.0: previous draft had `pass` not `break`, leaving
                # the DoS cap unenforced).
                break

    if rows_total == 0:
        return PrunePromptReport(
            prefix="",
            prefix_chars=0,
            rows_total=0,
            rows_pruned=0,
            min_frequency=threshold,
        )

    if tok is None:
        return _prune_char_level(
            input_path, output_path, prompts, rows_total, threshold
        )
    return _prune_token_level(
        tok, input_path, output_path, prompts, rows_total, threshold
    )


def _prune_char_level(
    input_path: str,
    output_path: str,
    prompts: List[str],
    rows_total: int,
    threshold: float,
) -> PrunePromptReport:
    """Character-level prefix strip (the v0.63.0 default behaviour)."""
    prefix = detect_common_prefix(prompts, min_frequency=threshold)
    rows_pruned = 0
    with open(input_path, encoding="utf-8") as fh_in, \
            open(output_path, "w", encoding="utf-8") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if prefix and isinstance(row.get("prompt"), str) and row["prompt"].startswith(prefix):
                row["prompt"] = row["prompt"][len(prefix):]
                rows_pruned += 1
            fh_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return PrunePromptReport(
        prefix=prefix,
        prefix_chars=len(prefix),
        rows_total=rows_total,
        rows_pruned=rows_pruned,
        min_frequency=threshold,
    )


def _prune_token_level(
    tok: Any,
    input_path: str,
    output_path: str,
    prompts: List[str],
    rows_total: int,
    threshold: float,
) -> PrunePromptReport:
    """Token-aware prefix strip (v0.71.5 #205).

    The detected prefix is a list of token IDs; stripping a row decodes the
    REMAINING token IDs so the boundary is always a clean token break.
    """
    token_rows = [_encode(tok, p) for p in prompts]
    prefix_ids = detect_common_prefix_tokens(token_rows, min_frequency=threshold)
    prefix_text = tok.decode(prefix_ids) if prefix_ids else ""
    plen = len(prefix_ids)

    rows_pruned = 0
    with open(input_path, encoding="utf-8") as fh_in, \
            open(output_path, "w", encoding="utf-8") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            if prefix_ids and isinstance(prompt, str):
                ids = _encode(tok, prompt)
                if ids[:plen] == prefix_ids:
                    row["prompt"] = tok.decode(ids[plen:])
                    rows_pruned += 1
            fh_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return PrunePromptReport(
        prefix=prefix_text,
        prefix_chars=len(prefix_text),
        rows_total=rows_total,
        rows_pruned=rows_pruned,
        min_frequency=threshold,
    )


__all__ = [
    "PrunePromptReport",
    "detect_common_prefix",
    "detect_common_prefix_tokens",
    "prune_traces",
    "validate_min_frequency",
]
