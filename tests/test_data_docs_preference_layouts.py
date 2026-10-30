"""Validate preference row layouts documented in docs/data.md.

Issue #2086: docs/data.md documents the accepted and refused layouts for DPO and
KTO rows. This test extracts the example rows and documented outcomes directly from
the markdown tables in docs/data.md, executes them against
``soup_cli.data.formats.format_to_messages_with_reason``, and verifies that:
1. Every accepted row converts cleanly (and satisfies the documented layout).
2. Every refused row returns the exact error string shown in the documentation.
3. The documentation maintains at least the baseline count of accepted and refused
   examples, preventing accidental row deletion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.data.formats import format_to_messages_with_reason, unify_preference_layout

_ROOT = Path(__file__).resolve().parent.parent
_DATA_MD = _ROOT / "docs" / "data.md"


def _clean_cell(cell: str) -> str:
    """Strip surrounding markdown backticks and whitespace."""
    s = cell.strip()
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def _extract_preference_section() -> str:
    """Extract the ### Preference row layouts (dpo, kto) section from docs/data.md."""
    text = _DATA_MD.read_text(encoding="utf-8")
    start = text.find("### Preference row layouts (dpo, kto)")
    assert start != -1, "### Preference row layouts (dpo, kto) section not found in docs/data.md"
    end = text.find("**LLaVA (vision):**", start)
    assert end != -1, "End boundary **LLaVA (vision):** not found after preference section"
    return text[start:end]


def _parse_table_rows(section_text: str, heading: str) -> list[list[str]]:
    """Parse rows from a table following a given markdown heading."""
    idx = section_text.find(heading)
    assert idx != -1, f"Heading '{heading}' not found in preference section"
    sub = section_text[idx + len(heading) :]
    lines = sub.splitlines()

    rows: list[list[str]] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        if stripped.startswith("|---") or stripped.startswith("|:---") or stripped.startswith("|-"):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        # Skip header row
        if cells and cells[0].lower() in ("layout", "row"):
            continue
        rows.append(cells)
    return rows


def _extract_accepted_rows(heading: str) -> list[tuple[str, dict, str]]:
    section = _extract_preference_section()
    table = _parse_table_rows(section, heading)
    results = []
    for cells in table:
        layout_name = cells[0]
        row_cell = _clean_cell(cells[1])
        outcome = cells[2]
        if row_cell.startswith("{") and row_cell.endswith("}"):
            row_data = json.loads(row_cell)
            results.append((layout_name, row_data, outcome))
    return results


def _extract_refused_rows(heading: str) -> list[tuple[dict, str]]:
    section = _extract_preference_section()
    table = _parse_table_rows(section, heading)
    results = []
    for cells in table:
        row_cell = _clean_cell(cells[0])
        reason_cell = _clean_cell(cells[1])
        if row_cell.startswith("{") and row_cell.endswith("}"):
            row_data = json.loads(row_cell)
            results.append((row_data, reason_cell))
    return results


_DPO_ACCEPTED = _extract_accepted_rows("#### Accepted preference pair layouts (`data.format: dpo`)")
_DPO_REFUSED = _extract_refused_rows("#### Refused preference pair layouts (`data.format: dpo`)")
_KTO_ACCEPTED = _extract_accepted_rows("##### Accepted KTO rows")
_KTO_REFUSED = _extract_refused_rows("##### Refused KTO rows")


REQUIRED_DPO_REFUSAL_MESSAGES = (
    (
        "dpo.chosen is a string and dpo.rejected is a message list: "
        "give both as strings or both as message lists"
    ),
    "dpo.chosen has no turn after the 1 it shares with dpo.rejected: there is no chosen answer",
    (
        "dpo.prompt is missing and dpo.chosen and dpo.rejected share no leading turn to use as "
        "the prompt: add 'prompt' (a string or a message list)"
    ),
    "dpo.chosen is an empty message list",
    "dpo.prompt must be a string, a message or a list of messages, got int",
    "dpo.chosen[1] must be a message (an object with 'role' and 'content'), got str",
    "dpo.chosen[1].role must be a non-empty string",
    "dpo.rejected[1].content must be a string or a list of content blocks, got int",
    (
        "dpo.chosen[2].content must be a string or a list of content blocks, got dict "
        "(write a JSON tool result as a JSON string)"
    ),
    "dpo.chosen must not be null",
    "dpo.prompt must not be null",
)

REQUIRED_KTO_REFUSAL_MESSAGES = (
    "kto.completion has no turn after the 2 that lead it: there is no assistant answer",
    "kto.prompt must not be null",
    "kto.completion is an empty message list",
    "KTO label must be true/false, got string: 'maybe'",
)

U = {"role": "user", "content": "Q"}
G = {"role": "assistant", "content": "good"}
B = {"role": "assistant", "content": "bad"}
G_NULL = {"role": "assistant", "content": None}


def check_documented_row_counts(
    dpo_accepted: list,
    dpo_refused: list,
    kto_accepted: list,
    kto_refused: list,
    *,
    min_dpo_accepted: int = 9,
    min_dpo_refused: int = 11,
    min_kto_accepted: int = 4,
    min_kto_refused: int = 4,
) -> None:
    """Validate that documented tables maintain at least the required row minimums."""
    assert len(dpo_accepted) >= min_dpo_accepted, (
        f"dpo_accepted has {len(dpo_accepted)} rows, expected >= {min_dpo_accepted}"
    )
    assert len(dpo_refused) >= min_dpo_refused, (
        f"dpo_refused has {len(dpo_refused)} rows, expected >= {min_dpo_refused}"
    )
    assert len(kto_accepted) >= min_kto_accepted, (
        f"kto_accepted has {len(kto_accepted)} rows, expected >= {min_kto_accepted}"
    )
    assert len(kto_refused) >= min_kto_refused, (
        f"kto_refused has {len(kto_refused)} rows, expected >= {min_kto_refused}"
    )


def test_preference_layouts_table_row_counts_and_coverage():
    """Assert minimum count and presence of all required documented examples."""
    check_documented_row_counts(_DPO_ACCEPTED, _DPO_REFUSED, _KTO_ACCEPTED, _KTO_REFUSED)

    doc_dpo_reasons = {reason for _, reason in _DPO_REFUSED}
    for expected in REQUIRED_DPO_REFUSAL_MESSAGES:
        assert expected in doc_dpo_reasons, f"Required DPO refusal missing from docs: {expected}"

    doc_kto_reasons = {reason for _, reason in _KTO_REFUSED}
    for expected in REQUIRED_KTO_REFUSAL_MESSAGES:
        assert expected in doc_kto_reasons, f"Required KTO refusal missing from docs: {expected}"


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "dpo_accepted",
            {
                "dpo_accepted": _DPO_ACCEPTED[:-1],
                "dpo_refused": _DPO_REFUSED,
                "kto_accepted": _KTO_ACCEPTED,
                "kto_refused": _KTO_REFUSED,
                "min_dpo_accepted": len(_DPO_ACCEPTED),
            },
        ),
        (
            "dpo_refused",
            {
                "dpo_accepted": _DPO_ACCEPTED,
                "dpo_refused": _DPO_REFUSED[:-1],
                "kto_accepted": _KTO_ACCEPTED,
                "kto_refused": _KTO_REFUSED,
                "min_dpo_refused": len(_DPO_REFUSED),
            },
        ),
        (
            "kto_accepted",
            {
                "dpo_accepted": _DPO_ACCEPTED,
                "dpo_refused": _DPO_REFUSED,
                "kto_accepted": _KTO_ACCEPTED[:-1],
                "kto_refused": _KTO_REFUSED,
                "min_kto_accepted": len(_KTO_ACCEPTED),
            },
        ),
        (
            "kto_refused",
            {
                "dpo_accepted": _DPO_ACCEPTED,
                "dpo_refused": _DPO_REFUSED,
                "kto_accepted": _KTO_ACCEPTED,
                "kto_refused": _KTO_REFUSED[:-1],
                "min_kto_refused": len(_KTO_REFUSED),
            },
        ),
    ],
)
def test_deleting_documented_row_fails_count_assertion(name: str, kwargs: dict):
    """Mutation check: removing a row from any table fails count check naming the list."""
    with pytest.raises(AssertionError, match=name):
        check_documented_row_counts(**kwargs)


@pytest.mark.parametrize(
    ("layout", "row", "outcome"),
    _DPO_ACCEPTED,
    ids=[layout for layout, _, _ in _DPO_ACCEPTED],
)
def test_documented_dpo_accepted_rows_convert(layout: str, row: dict, outcome: str):
    """Every accepted DPO row in docs/data.md must convert and match documented layout."""
    converted, reason = format_to_messages_with_reason(row, "dpo")
    assert reason is None, f"Expected {layout} row {row} to convert cleanly, got reason: {reason}"
    assert converted is not None
    assert set(converted.keys()) == {"prompt", "chosen", "rejected"}

    if layout == "all strings":
        assert converted == {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    elif layout == "turn with no text":
        assert converted == {"prompt": [U], "chosen": [G_NULL], "rejected": [B]}
    else:
        assert converted == {"prompt": [U], "chosen": [G], "rejected": [B]}


@pytest.mark.parametrize(
    ("row", "expected_reason"),
    _DPO_REFUSED,
    ids=[f"dpo_refused_{i}" for i in range(len(_DPO_REFUSED))],
)
def test_documented_dpo_refused_rows_match_exact_message(row: dict, expected_reason: str):
    """Every refused DPO row in docs/data.md must fail with the exact documented reason."""
    converted, reason = format_to_messages_with_reason(row, "dpo")
    assert converted is None, f"Expected row {row} to be refused, but converted to {converted}"
    assert reason == expected_reason


@pytest.mark.parametrize(
    ("layout", "row", "outcome"),
    _KTO_ACCEPTED,
    ids=[layout for layout, _, _ in _KTO_ACCEPTED],
)
def test_documented_kto_accepted_rows_convert(layout: str, row: dict, outcome: str):
    """Every accepted KTO row in docs/data.md must convert and match documented layout."""
    converted, reason = format_to_messages_with_reason(row, "kto")
    assert reason is None, f"Expected {layout} row {row} to convert cleanly, got reason: {reason}"
    assert converted is not None
    assert set(converted.keys()) == {"prompt", "completion", "label"}
    assert isinstance(converted["label"], bool)

    if layout == "all strings":
        assert converted == {"prompt": "Q", "completion": "good", "label": True}
    else:
        assert converted == {"prompt": [U], "completion": [G], "label": True}


@pytest.mark.parametrize(
    ("row", "expected_reason"),
    _KTO_REFUSED,
    ids=[f"kto_refused_{i}" for i in range(len(_KTO_REFUSED))],
)
def test_documented_kto_refused_rows_match_exact_message(row: dict, expected_reason: str):
    """Every refused KTO row in docs/data.md must fail with the exact documented reason."""
    converted, reason = format_to_messages_with_reason(row, "kto")
    assert converted is None, f"Expected row {row} to be refused, but converted to {converted}"
    assert reason == expected_reason


def test_mixed_text_and_conversational_unification():
    """Verify unification of text rows alongside conversational rows as documented."""
    text_row = {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    conv_row = {
        "prompt": [{"role": "user", "content": "Q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    conv_text, _ = format_to_messages_with_reason(text_row, "dpo")
    conv_conv, _ = format_to_messages_with_reason(conv_row, "dpo")
    assert conv_text is not None and conv_conv is not None

    unified, rewritten = unify_preference_layout([conv_text, conv_conv])
    assert rewritten == 1
    assert isinstance(unified[0]["prompt"], list)
    assert isinstance(unified[0]["chosen"], list)
    assert isinstance(unified[0]["rejected"], list)


def test_docs_preference_section_prose_and_anchors():
    """Verify exact prose strings, error messages, and cross-reference links in docs/data.md."""
    text = _DATA_MD.read_text(encoding="utf-8")
    assert "[Preference row layouts (dpo, kto)](#preference-row-layouts-dpo-kto)" in text
    section = _extract_preference_section()
    assert "data.format: dpo" in section
    assert "data.format: kto" in section
    assert "reward_model" in section
    assert "Cannot detect format at row 0" in section
    assert "... and N more" in section
    assert "Warning: N of M rows dropped: they do not convert as 'dpo'" in section
    assert "List them all with: soup data validate <source> --format dpo" in section
    assert "soup data validate" in section
    assert "stops with exit code 1 before loading the model, including under `--dry-run`" in section
    assert "system" in section
    assert (
        "Note: N of M train rows are plain-text preference rows in a dataset whose other rows "
        "are conversations. Each was rewritten"
    ) in section
