"""Tests for issue #771: fast-path dataset validator optimizations and edge-case guards."""

from __future__ import annotations

from soup_cli.data.formats import VALID_FORMATS, format_to_messages_with_reason
from soup_cli.data.validator import _to_hashable, extended_stats, validate_and_stats


def _reference_validate_and_stats(data: list[dict], expected_format: str | None = None) -> dict:
    """Reference implementation matching original behavior for differential testing."""
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

    lengths = []
    empty_count = 0
    for row in data:
        text = " ".join(str(v) for v in row.values() if v)
        lengths.append(len(text))
        for v in row.values():
            if v is None:
                empty_count += 1

    # Type-tagged signature to prevent type collision false duplicates
    row_strs = [str(tuple((k, _to_hashable(v)) for k, v in sorted(row.items()))) for row in data]
    dup_count = len(row_strs) - len(set(row_strs))

    issues = []
    valid_rows = len(data)
    if expected_format and expected_format in VALID_FORMATS:
        invalid = 0
        sample_reasons: list[str] = []
        for idx, row in enumerate(data):
            _, reason = format_to_messages_with_reason(row, expected_format)
            if reason is None:
                continue
            invalid += 1
            if len(sample_reasons) < 3:
                sample_reasons.append(f"row {idx}: {reason}")
        valid_rows = len(data) - invalid
        if invalid > 0:
            issues.append(
                f"{invalid} rows fail to convert for '{expected_format}' format "
                f"(load_dataset would drop them)"
            )
            issues.extend(sample_reasons)
            if invalid > len(sample_reasons):
                issues.append(f"... and {invalid - len(sample_reasons)} more")

    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")
    if empty_count > 0:
        issues.append(f"{empty_count} empty fields found")

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


def test_text_length_calculation_join_separator():
    """Verify text length matches joined string for validate_and_stats and extended_stats."""
    row = {"instruction": "hello", "input": "world", "output": "test"}
    expected_text = "hello world test"
    res = validate_and_stats([row])
    assert res["min_length"] == len(expected_text)
    assert res["max_length"] == len(expected_text)
    assert res["avg_length"] == len(expected_text)

    # Multi-field exact length test for extended_stats (guards against line 176 separator bug)
    multi_row = {"a": "abc", "b": "de"}
    ext_res = extended_stats([multi_row])
    assert ext_res["lengths"] == [6]
    assert ext_res["avg_tokens"] == 1


def test_type_tagged_duplicates_no_collisions():
    """Verify distinct types do not collapse into false duplicates."""
    # 1 vs True
    res1 = validate_and_stats([{"a": 1}, {"a": True}])
    assert res1["duplicates"] == 0

    # 1 vs 1.0
    res2 = validate_and_stats([{"a": 1}, {"a": 1.0}])
    assert res2["duplicates"] == 0

    # 0 vs False
    res2b = validate_and_stats([{"a": 0}, {"a": False}])
    assert res2b["duplicates"] == 0

    # "1" vs 1
    res2c = validate_and_stats([{"a": "1"}, {"a": 1}])
    assert res2c["duplicates"] == 0

    # list vs tuple
    res3 = validate_and_stats([{"a": [1, 2]}, {"a": (1, 2)}])
    assert res3["duplicates"] == 0

    # dict vs list of tuples
    res4 = validate_and_stats([{"a": {"x": 1}}, {"a": [("x", 1)]}])
    assert res4["duplicates"] == 0

    # empty list vs empty dict
    res5 = validate_and_stats([{"messages": []}, {"messages": {}}])
    assert res5["duplicates"] == 0


def test_nested_dict_key_order_invariance():
    """Verify nested dicts with identical content in different key order count as duplicates."""
    data = [{"m": {"b": 1, "a": 2}}, {"m": {"a": 2, "b": 1}}]
    res = validate_and_stats(data)
    assert res["duplicates"] == 1


def test_mixed_key_types_nested_dict_no_typeerror():
    """Verify nested dicts with mixed key types (int and str) do not raise TypeError."""
    data = [{"a": {1: "x", "b": "y"}}]
    res = validate_and_stats(data)
    assert res["total"] == 1


def test_format_validation_reasons_preserved():
    """Verify #712 format validation reasons are returned correctly."""
    bad_chatml = [{"messages": "not a list"}]
    res = validate_and_stats(bad_chatml, expected_format="chatml")
    assert res["valid_rows"] == 0
    assert any("chatml" in issue for issue in res["issues"])


def test_differential_equivalence():
    """Verify optimized validate_and_stats matches reference on diverse samples."""
    samples = [
        [{"instruction": "A", "input": "B", "output": "C"}],
        [{"a": None, "b": "text"}, {"a": None, "b": "text"}],
        [{"messages": [{"role": "user", "content": "hi"}]}] * 3,
        [{"x": 0, "y": False, "z": "content string"}],
        [{"instruction": "short"}],
        [{"a": "hello", "b": "world"}, {"b": "world", "a": "hello"}],
    ]
    for sample in samples:
        expected = _reference_validate_and_stats(sample, expected_format="alpaca")
        actual = validate_and_stats(sample, expected_format="alpaca")
        assert actual == expected
