"""Dataset validation and statistics."""

from __future__ import annotations

from typing import Any, Optional

from soup_cli.data.formats import (
    VALID_FORMATS,
    _DROP_EXCEPTIONS,
    _dispatch_conversion,
)

# How many per-row drop reasons to surface in `issues`. Enough to make
# `validate` actionable ("which rows and why") without flooding the output on a
# file where every row is bad.
_MAX_REASON_SAMPLES = 3


def _to_hashable(val: Any) -> Any:
    """Recursively convert values into type-tagged hashable nested tuples."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return (type(val).__name__, val)
    if isinstance(val, dict):
        return (
            "dict",
            tuple(
                (str(k), _to_hashable(v))
                for k, v in sorted(val.items(), key=lambda item: str(item[0]))
            ),
        )
    if isinstance(val, list):
        return ("list", tuple(_to_hashable(v) for v in val))
    if isinstance(val, tuple):
        return ("tuple", tuple(_to_hashable(v) for v in val))
    if isinstance(val, set):
        try:
            return ("set", tuple(_to_hashable(v) for v in sorted(val, key=repr)))
        except TypeError:
            return ("set", tuple(_to_hashable(v) for v in val))
    return (type(val).__name__, str(val))


def _row_signature(row: dict) -> tuple:
    """Return a type-tagged hashable canonical representation of a row dict."""
    return tuple(
        (str(k), _to_hashable(v))
        for k, v in sorted(row.items(), key=lambda item: str(item[0]))
    )


def _compute_row_text_length(row: dict) -> tuple[int, int]:
    """Compute text length and count empty/None fields without intermediate string joins."""
    parts_len = 0
    parts_count = 0
    empty_count = 0
    for v in row.values():
        if v is None:
            empty_count += 1
        elif v:
            v_str = v if isinstance(v, str) else str(v)
            parts_len += len(v_str)
            parts_count += 1

    char_len = parts_len + (parts_count - 1 if parts_count > 0 else 0)
    return char_len, empty_count


def validate_and_stats(data: list[dict], expected_format: Optional[str] = None) -> dict:
    """Compute stats and validate dataset."""
    if not data:
        return {
            "total": 0,
            "columns": [],
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "empty_fields": 0,
            "duplicates": 0,
            "issues": ["Dataset is empty"],
            "valid_rows": 0,
        }

    columns = list(data[0].keys())

    empty_count = 0
    short_count = 0
    seen_rows: set[tuple] = set()
    dup_count = 0
    total_length = 0
    min_length = float("inf")
    max_length = 0

    invalid_count = 0
    sample_reasons: list[str] = []
    check_format = bool(expected_format and expected_format in VALID_FORMATS)

    # Probe whether the dataset values are all strings/None (flat rows).
    # Real JSONL datasets are homogeneous — if row 0 is flat, all rows are.
    # The fast path avoids the _to_hashable type-tagging overhead entirely.
    # Guard with try/except for heterogeneous datasets (mixed formats).
    _flat_values = all(
        isinstance(v, str) or v is None for v in data[0].values()
    )

    for idx, row in enumerate(data):
        # 1. Duplicate detection — fast path for flat rows (3x faster),
        # type-tagged fallback for rows with non-string values (nested
        # dicts/lists, ints, bools) where Python equality conflates types
        # (e.g. 1 == True, 1 == 1.0).
        if _flat_values:
            try:
                sig = tuple(sorted(row.items()))
                if sig in seen_rows:
                    dup_count += 1
                else:
                    seen_rows.add(sig)
            except TypeError:
                # Heterogeneous dataset: this row has unhashable values
                # even though row 0 didn't. Fall back for this row.
                sig = _row_signature(row)
                if sig in seen_rows:
                    dup_count += 1
                else:
                    seen_rows.add(sig)
        else:
            sig = _row_signature(row)
            if sig in seen_rows:
                dup_count += 1
            else:
                seen_rows.add(sig)

        # 2. Format validation using real converter path (#712).
        # Inlined from format_to_messages_with_reason: we already validated
        # expected_format ∈ VALID_FORMATS above, so skip the per-row check.
        if check_format:
            try:
                _dispatch_conversion(row, expected_format)
                reason = None
            except _DROP_EXCEPTIONS as exc:
                reason = str(exc)
            if reason is not None:
                invalid_count += 1
                if len(sample_reasons) < _MAX_REASON_SAMPLES:
                    sample_reasons.append(f"row {idx}: {reason}")

        # 3. Inlined text length & empty field calculation (avoids per-row
        # function call overhead for 20k+ rows).
        parts_len = 0
        parts_count = 0
        for v in row.values():
            if v is None:
                empty_count += 1
            elif v:
                v_str = v if isinstance(v, str) else str(v)
                parts_len += len(v_str)
                parts_count += 1
        char_len = parts_len + (parts_count - 1 if parts_count > 0 else 0)
        total_length += char_len
        if char_len < min_length:
            min_length = char_len
        if char_len > max_length:
            max_length = char_len
        if char_len < 10:
            short_count += 1

    valid_rows = len(data) - invalid_count

    issues: list[str] = []
    if check_format and invalid_count > 0:
        issues.append(
            f"{invalid_count} rows fail to convert for '{expected_format}' format "
            f"(load_dataset would drop them)"
        )
        issues.extend(sample_reasons)
        if invalid_count > len(sample_reasons):
            issues.append(f"... and {invalid_count - len(sample_reasons)} more")

    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")
    if empty_count > 0:
        issues.append(f"{empty_count} empty fields found")
    if short_count > 0:
        issues.append(f"{short_count} samples are very short (<10 chars)")

    return {
        "total": len(data),
        "columns": columns,
        "avg_length": round(total_length / len(data)),
        "min_length": int(min_length),
        "max_length": int(max_length),
        "empty_fields": empty_count,
        "duplicates": dup_count,
        "issues": issues,
        "valid_rows": valid_rows,
    }


def _percentile(sorted_vals: list, pct: int) -> int:
    """Compute a percentile from a sorted list."""
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def extended_stats(data: list[dict]) -> dict:
    """Compute extended statistics: length distribution, token counts, languages."""
    if not data:
        return {
            "total": 0,
            "lengths": [],
            "token_counts": [],
            "length_p10": 0,
            "length_p25": 0,
            "length_p50": 0,
            "length_p75": 0,
            "length_p90": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "languages": {},
        }

    lengths = []
    token_counts = []

    for row in data:
        char_len, _ = _compute_row_text_length(row)
        lengths.append(char_len)
        # Approximate token count: ~4 chars per token for English
        token_counts.append(max(1, char_len // 4))

    sorted_lengths = sorted(lengths)

    # Language detection (optional, lazy import)
    languages: dict[str, int] = {}
    try:
        from langdetect import detect

        sample_size = min(100, len(data))
        for row in data[:sample_size]:
            text = " ".join(str(v) for v in row.values() if v)
            if len(text) > 20:
                try:
                    lang = detect(text)
                    languages[lang] = languages.get(lang, 0) + 1
                except Exception:
                    pass
    except ImportError:
        pass  # langdetect not installed, skip

    return {
        "total": len(data),
        "lengths": lengths,
        "token_counts": token_counts,
        "length_p10": _percentile(sorted_lengths, 10),
        "length_p25": _percentile(sorted_lengths, 25),
        "length_p50": _percentile(sorted_lengths, 50),
        "length_p75": _percentile(sorted_lengths, 75),
        "length_p90": _percentile(sorted_lengths, 90),
        "avg_tokens": round(sum(token_counts) / len(token_counts)),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "languages": languages,
    }
