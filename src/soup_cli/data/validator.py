"""Dataset validation and statistics."""

from __future__ import annotations

from typing import Optional

from soup_cli.data.formats import FORMAT_SIGNATURES


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

    # Compute text lengths (join all string values)
    lengths = []
    empty_count = 0
    for row in data:
        text = " ".join(str(v) for v in row.values() if v)
        lengths.append(len(text))
        for v in row.values():
            if v is None:
                empty_count += 1

    # Detect duplicates by stringifying rows
    row_strs = [str(sorted(row.items())) for row in data]
    dup_count = len(row_strs) - len(set(row_strs))

    # Validate format
    issues = []
    valid_rows = len(data)
    if expected_format and expected_format in FORMAT_SIGNATURES:
        required = FORMAT_SIGNATURES[expected_format]
        invalid = 0
        for row in data:
            if not required.issubset(row.keys()):
                invalid += 1
        valid_rows = len(data) - invalid
        if invalid > 0:
            issues.append(
                f"{invalid} rows missing required keys for '{expected_format}' format: {required}"
            )

    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")
    if empty_count > 0:
        issues.append(f"{empty_count} empty fields found")

    # Check for very short samples
    short = sum(1 for length in lengths if length < 10)
    if short > 0:
        issues.append(f"{short} samples are very short (<10 chars)")

    return {
        "total": len(data),
        "columns": columns,
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
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
        text = " ".join(str(v) for v in row.values() if v)
        char_len = len(text)
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
