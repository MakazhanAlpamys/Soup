"""Tests for data/validator.py — validate_and_stats + extended_stats."""

from soup_cli.data.validator import (
    _percentile,
    extended_stats,
    validate_and_stats,
)

# --- validate_and_stats ---

def test_validate_empty_data():
    """Empty dataset should return zero stats with an issue."""
    stats = validate_and_stats([])
    assert stats["total"] == 0
    assert stats["avg_length"] == 0
    assert "empty" in stats["issues"][0].lower()


def test_validate_basic_stats():
    """Basic stats should be computed correctly."""
    data = [
        {"instruction": "What is Python?", "output": "A programming language."},
        {"instruction": "What is AI?", "output": "Artificial Intelligence."},
    ]
    stats = validate_and_stats(data)
    assert stats["total"] == 2
    assert stats["avg_length"] > 0
    assert stats["min_length"] > 0
    assert stats["max_length"] >= stats["min_length"]
    assert "instruction" in stats["columns"]
    assert "output" in stats["columns"]


def test_validate_detects_duplicates():
    """Duplicated rows should be counted."""
    row = {"instruction": "Hello", "output": "Hi"}
    data = [row, row, {"instruction": "Bye", "output": "See ya"}]
    stats = validate_and_stats(data)
    assert stats["duplicates"] == 1
    assert any("duplicate" in issue.lower() for issue in stats["issues"])


def test_validate_detects_empty_fields():
    """None fields should be counted as empty."""
    data = [
        {"instruction": "Test", "output": None},
        {"instruction": "Test2", "output": "Ok"},
    ]
    stats = validate_and_stats(data)
    assert stats["empty_fields"] == 1
    assert any("empty" in issue.lower() for issue in stats["issues"])


def test_validate_format_alpaca_valid():
    """Valid alpaca data should have no format issues."""
    data = [
        {"instruction": "Q", "input": "", "output": "A"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]
    stats = validate_and_stats(data, expected_format="alpaca")
    assert stats["valid_rows"] == 2
    # No format-related issues (may have short-sample issues)
    format_issues = [iss for iss in stats["issues"] if "missing required keys" in iss]
    assert len(format_issues) == 0


def test_validate_format_alpaca_invalid():
    """Rows the converter would drop should be flagged, with the row and reason."""
    data = [
        {"instruction": "Q", "output": "A"},
        {"wrong_key": "Q2", "output": "A2"},  # missing 'instruction'
    ]
    stats = validate_and_stats(data, expected_format="alpaca")
    assert stats["valid_rows"] == 1
    assert any("fail to convert" in iss for iss in stats["issues"])
    # The specific offending row is surfaced so `validate` is actionable.
    assert any("row 1" in iss for iss in stats["issues"])


def test_validate_short_samples():
    """Very short samples (<10 chars) should be flagged."""
    data = [
        {"text": "Hi"},
        {"text": "This is a longer sentence that should not trigger the warning."},
    ]
    stats = validate_and_stats(data)
    assert any("short" in iss.lower() for iss in stats["issues"])


# --- extended_stats ---

def test_extended_stats_empty():
    """Extended stats on empty data should return defaults."""
    stats = extended_stats([])
    assert stats["total"] == 0
    assert stats["avg_tokens"] == 0
    assert stats["lengths"] == []


def test_extended_stats_basic():
    """Extended stats should compute percentiles and token counts."""
    data = [
        {"text": "Short text"},
        {"text": "A medium length piece of example text for testing"},
        {"text": "An even longer sentence that should be significantly more characters than rest"},
    ]
    stats = extended_stats(data)
    assert stats["total"] == 3
    assert len(stats["lengths"]) == 3
    assert len(stats["token_counts"]) == 3
    assert stats["avg_tokens"] > 0
    assert stats["min_tokens"] >= 1
    assert stats["max_tokens"] >= stats["min_tokens"]
    assert stats["length_p50"] > 0


def test_extended_stats_percentiles():
    """Percentiles should be ordered correctly."""
    data = [{"text": "x" * i} for i in range(10, 110, 10)]
    stats = extended_stats(data)
    assert stats["length_p10"] <= stats["length_p25"]
    assert stats["length_p25"] <= stats["length_p50"]
    assert stats["length_p50"] <= stats["length_p75"]
    assert stats["length_p75"] <= stats["length_p90"]


# --- _percentile ---

def test_percentile_empty():
    assert _percentile([], 50) == 0


def test_percentile_single():
    assert _percentile([42], 50) == 42


def test_percentile_sorted():
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(vals, 50) == 60  # index 5
    assert _percentile(vals, 0) == 10   # index 0
    assert _percentile(vals, 100) == 100  # last element
