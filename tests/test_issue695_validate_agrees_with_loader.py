"""Regression tests for issue #695.

`soup data validate` used to disagree with `load_dataset`: `validate_and_stats`
validated a row by top-level key presence (`FORMAT_SIGNATURES`), while the loader
runs `format_to_messages` and drops any row that returns `None`. Those are two
different notions of "valid", so they drifted apart in two ways:

1. Six formats are absent from `FORMAT_SIGNATURES` (`prm`, `pre_tokenized`,
   `input_output`, `video`, `multimodal`, `raft`), so validation was skipped
   entirely and every row was reported valid regardless of content.
2. For the formats that were checked, key presence passed rows the converter
   drops on a value (a null field, a non-dict message, ...).

The fix computes `valid_rows` by running the real conversion path per row, so
agreement is structural. These tests pin that: they assert `validate` and
`load_dataset` agree on a *specific count*, not merely that some rows are
flagged — a validator wrong in the other direction would pass the weaker form.
"""

import pytest

from soup_cli.data.loader import _format_rows
from soup_cli.data.validator import validate_and_stats

# Each entry: (good_row, bad_row) where bad_row is a row the loader drops on
# current `main`. The spread deliberately mixes:
#   - value-level failures on FORMAT_SIGNATURES formats (alpaca/sharegpt null
#     field) — key presence passes these, so they discriminate the fix,
#   - a missing-key failure (chatml),
#   - formats NOT in FORMAT_SIGNATURES (input_output/multimodal/video) — key
#     presence skipped these entirely.
FORMAT_CASES = {
    "alpaca": (
        {"instruction": "say hi", "output": "hello there"},
        {"instruction": "x", "output": None},
    ),
    "sharegpt": (
        {"conversations": [{"from": "human", "value": "hi there"}]},
        {"conversations": [{"from": "human", "value": None}]},
    ),
    "chatml": (
        {"messages": [{"role": "user", "content": "hi there"}]},
        {"nope": "x"},
    ),
    "input_output": (
        {"segments": [{"text": "hello there", "label": True}]},
        {"segments": ["x"]},
    ),
    "multimodal": (
        {"messages": [{"role": "u", "content": [{"type": "text", "text": "hi there"}]}]},
        {"messages": ["hello"]},
    ),
    "video": (
        {"video": "v.mp4", "messages": [{"role": "u", "content": "hi there"}]},
        {"video": 123},
    ),
}

# The formats that carried no FORMAT_SIGNATURES entry, so validation was skipped
# for them entirely before this fix.
SKIP_VALIDATION_FORMATS = [
    "prm", "pre_tokenized", "input_output", "video", "multimodal", "raft",
]


class TestValidateAgreesWithLoader:
    @pytest.mark.parametrize("fmt", list(FORMAT_CASES))
    def test_valid_rows_equals_what_the_loader_keeps(self, fmt):
        good, bad = FORMAT_CASES[fmt]
        rows = [good] * 4 + [bad]

        reported = validate_and_stats(rows, expected_format=fmt)["valid_rows"]
        kept = len(_format_rows(rows, fmt))

        # Structural agreement, on a specific count. A revert to key-presence
        # validation reports 5 here (the bad row has its keys / is unchecked),
        # while the loader keeps 4 — so both this equality and the == 4 fail.
        assert reported == kept == 4

    @pytest.mark.parametrize("fmt", list(FORMAT_CASES))
    def test_all_good_file_is_fully_valid(self, fmt):
        good, _ = FORMAT_CASES[fmt]
        rows = [good] * 3

        stats = validate_and_stats(rows, expected_format=fmt)
        assert stats["valid_rows"] == 3
        assert not any("fail to convert" in iss for iss in stats["issues"])


class TestSkippedFormatsAreNowValidated:
    """Defect 1: the six formats absent from FORMAT_SIGNATURES were never
    validated, so a bad row was silently reported valid."""

    def test_input_output_bad_row_is_counted_invalid(self):
        rows = [{"segments": [{"text": "hello there", "label": True}]}] * 4
        rows.append({"segments": ["not a dict"]})

        stats = validate_and_stats(rows, expected_format="input_output")
        # Before the fix this returned 5 (validation skipped for input_output).
        assert stats["valid_rows"] == 4

    @pytest.mark.parametrize("fmt", SKIP_VALIDATION_FORMATS)
    def test_previously_skipped_format_now_matches_loader(self, fmt):
        # A row that is missing everything is dropped by every converter, so
        # this holds for all six without needing a per-format valid fixture.
        rows = [{"definitely": "not valid for any format"}]
        reported = validate_and_stats(rows, expected_format=fmt)["valid_rows"]
        kept = len(_format_rows(rows, fmt))
        assert reported == kept == 0


class TestDropReasonIsSurfaced:
    """The maintainer asked for 'which rows and why', not just a count."""

    def test_reason_and_row_index_appear_in_issues(self):
        rows = [
            {"messages": [{"role": "u", "content": [{"type": "text", "text": "hi there"}]}]},
            {"messages": ["hello"]},  # non-dict message -> dropped
        ]
        stats = validate_and_stats(rows, expected_format="multimodal")

        assert stats["valid_rows"] == 1
        # The converter's own message is shown, tied to the offending row.
        assert any("multimodal message must be a dict" in iss for iss in stats["issues"])
        assert any("row 1" in iss for iss in stats["issues"])

    def test_reason_sample_is_capped_but_total_is_reported(self):
        # 10 bad rows; the sample is capped but the full count is still stated.
        rows = [{"messages": ["bad"]} for _ in range(10)]
        stats = validate_and_stats(rows, expected_format="multimodal")

        assert stats["valid_rows"] == 0
        assert any("10 rows fail to convert" in iss for iss in stats["issues"])
        # Capped sample + an "... and N more" line rather than 10 row lines.
        row_lines = [iss for iss in stats["issues"] if iss.startswith("row ")]
        assert len(row_lines) == 3
        assert any("more" in iss for iss in stats["issues"])
