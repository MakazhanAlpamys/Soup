"""Dataset validation and statistics."""

from soup_cli.data.formats import FORMAT_SIGNATURES


def validate_and_stats(data: list[dict], expected_format: str | None = None) -> dict:
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
