"""Dataset validation and statistics."""

from __future__ import annotations

from typing import Any, Optional

from soup_cli.data.formats import FORMAT_SIGNATURES


def _to_hashable(val: Any) -> Any:
    """Recursively convert dicts and lists into hashable nested tuples."""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_to_hashable(v) for v in val)
    return str(val)


def _row_signature(row: dict) -> tuple:
    """Return a hashable canonical representation of a row dict."""
    try:
        return tuple(sorted(row.items()))
    except TypeError:
        return tuple((k, _to_hashable(v)) for k, v in sorted(row.items()))


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
    invalid_rows = 0
    seen_rows: set[tuple] = set()
    dup_count = 0
    total_length = 0
    row_count = 0
    min_length = float("inf")
    max_length = 0

    check_format = expected_format is not None and expected_format in FORMAT_SIGNATURES
    required = FORMAT_SIGNATURES[expected_format] if check_format else set()

    for row in data:
        # Detect duplicates via canonical hashable tuple — fast path for flat rows,
        # recursive fallback for rows containing nested dicts/lists.
        try:
            sig = tuple(sorted(row.items()))
            if sig in seen_rows:
                dup_count += 1
            else:
                seen_rows.add(sig)
        except TypeError:
            sig = tuple((k, _to_hashable(v)) for k, v in sorted(row.items()))
            if sig in seen_rows:
                dup_count += 1
            else:
                seen_rows.add(sig)

        # Validate format
        if check_format and not required.issubset(row.keys()):
            invalid_rows += 1

        # Compute text length and count empty/None fields without intermediate
        # list or joined-string allocations.
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
        row_count += 1
        if char_len < min_length:
            min_length = char_len
        if char_len > max_length:
            max_length = char_len
        if char_len < 10:
            short_count += 1

    valid_rows = len(data) - invalid_rows

    issues: list[str] = []
    if check_format and invalid_rows > 0:
        issues.append(
            f"{invalid_rows} rows missing required keys for '{expected_format}' format: {required}"
        )
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")
    if empty_count > 0:
        issues.append(f"{empty_count} empty fields found")
    if short_count > 0:
        issues.append(f"{short_count} samples are very short (<10 chars)")

    return {
        "total": len(data),
        "columns": columns,
        "avg_length": round(total_length / row_count),
        "min_length": int(min_length) if row_count > 0 else 0,
        "max_length": int(max_length) if row_count > 0 else 0,
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
        parts_len = 0
        parts_count = 0
        for v in row.values():
            if v:
                v_str = v if isinstance(v, str) else str(v)
                parts_len += len(v_str)
                parts_count += 1
        char_len = parts_len + (parts_count - 1 if parts_count > 0 else 0)
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
