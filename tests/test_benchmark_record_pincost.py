"""#2224: finding 9 of the H100 record said the 6.56x was "measured with
correctness asserted", which reads as both arms having passed. The record's own
STEP 8 table shows the opposite: the fast ``pin=True`` arm was **WRONG 8/256**
on all three repetitions and the slow ``pin=False`` arm was exact 256/256.

The record is a published measurement, so the sentence and every number stay
standing and a dated ``CLARIFICATION`` note sits under finding 9 instead. This
test reads the markdown on CPU and pins four things:

* the STEP 8 table still carries 8/256 + WRONG for ``pin=True`` and 256/256 +
  correct for ``pin=False`` (changing a row fails the first check);
* those two medians still divide to 6.56;
* the README harness index still says the pinned arm was WRONG 8/256;
* finding 9's sentence is byte-identical and a ``CLARIFICATION`` note with a
  date and the WRONG arm sits between finding 9 and finding 10 (deleting the
  note fails this check).
"""

from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
RECORD = BENCHMARKS / "gate-h100-validation.md"
HARNESS_INDEX = BENCHMARKS / "README.md"

FINDING_9 = (
    "9. **Turning pinning off costs 6.56x throughput** at 32B NF4, measured with\n"
    "   correctness asserted in the same process. That is too expensive to be a\n"
    "   silent automatic fallback.\n"
)
FINDING_10 = "10. **The defect is repaired, and the repair is cheap**"


def _read(path):
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def _step8_section(text):
    start = text.find("## STEP 8 —")
    assert start != -1, "the record no longer has a STEP 8 section"
    end = text.find("\n## STEP 9", start)
    assert end != -1, "the record no longer ends STEP 8 at STEP 9"
    return text[start:end]


def _table_rows(section):
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `pin="):
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    assert rows, "STEP 8 no longer has per-repetition table rows"
    return rows


def _row(cells):
    assert len(cells) == 6, f"unexpected STEP 8 row shape: {cells}"
    return {
        "arm": cells[0].strip("`"),
        "tok_s": float(cells[2]),
        "gradients": cells[4],
        "verdict": cells[5],
    }


def _step8_rows():
    return [_row(cells) for cells in _table_rows(_step8_section(_read(RECORD)))]


def test_step8_pinned_arm_is_recorded_as_wrong():
    rows = [r for r in _step8_rows() if r["arm"] == "pin=True"]
    assert len(rows) == 3, rows
    for row in rows:
        assert row["gradients"] == "8/256", row
        assert row["verdict"] == "**WRONG**", row


def test_step8_pageable_arm_is_recorded_as_exact():
    rows = [r for r in _step8_rows() if r["arm"] == "pin=False"]
    assert len(rows) == 3, rows
    for row in rows:
        assert row["gradients"] == "256/256", row
        assert row["verdict"] == "correct", row


def test_step8_ratio_is_the_two_medians():
    def median(arm):
        values = sorted(r["tok_s"] for r in _step8_rows() if r["arm"] == arm)
        assert len(values) % 2 == 1, values
        return values[len(values) // 2]

    pinned, pageable = median("pin=True"), median("pin=False")
    assert (pinned, pageable) == (425.07, 64.79), (pinned, pageable)
    assert round(pinned / pageable, 2) == 6.56


def test_harness_index_states_the_pinned_arm_was_wrong():
    rows = [
        line for line in _read(HARNESS_INDEX).splitlines() if line.startswith("| [`pincost.py`]")
    ]
    assert len(rows) == 1, rows
    assert "WRONG 8/256" in rows[0]
    assert "6.56x" in rows[0]


def test_finding_9_is_unchanged_and_carries_a_clarification():
    text = _read(RECORD)
    assert FINDING_9 in text, "finding 9's wording or numbers were edited"

    start = text.index(FINDING_9) + len(FINDING_9)
    end = text.index(FINDING_10, start)
    note = text[start:end]

    assert "CLARIFICATION" in note
    assert "WRONG" in note
    assert "6.56x" in note
    date = note.split("CLARIFICATION", 1)[1].split()[0]
    assert date.count("-") == 2 and len(date) == 10, (
        f"the note carries no YYYY-MM-DD date: {date!r}"
    )
    assert note.lstrip().startswith(">"), "the note is not in the record's blockquote form"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
