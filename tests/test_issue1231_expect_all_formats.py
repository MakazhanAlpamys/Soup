"""Tests for Issue #1231: soup expect PII and refusal checks across all dataset formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.formats import FORMAT_SIGNATURES
from soup_cli.utils.expectations import (
    FORMAT_EXTRACTION_RULES,
    expect_no_pii,
    expect_no_refusal_pattern,
    expect_token_length_between,
)
from tests.conftest import strip_ansi

PII_SAMPLE = "Contact jane.doe@example.com, SSN 123-45-6789."
REFUSAL_SAMPLE = "I'm sorry, but I can't help with that."


# -----------------------------------------------------------------------------
# 1. PII extraction across all format signatures and text-bearing fields
# -----------------------------------------------------------------------------

ALL_FORMAT_PII_CASES = [
    ("plaintext_text", {"text": PII_SAMPLE}),
    ("chatml_user", {"messages": [{"role": "user", "content": PII_SAMPLE}]}),
    ("chatml_assistant", {"messages": [{"role": "assistant", "content": PII_SAMPLE}]}),
    ("chatml_system", {"messages": [{"role": "system", "content": PII_SAMPLE}]}),
    (
        "chatml_content_parts",
        {"messages": [{"role": "user", "content": [{"type": "text", "text": PII_SAMPLE}]}]},
    ),
    ("alpaca_instruction", {"instruction": PII_SAMPLE, "output": "ok"}),
    ("alpaca_input", {"instruction": "Summarize", "input": PII_SAMPLE, "output": "ok"}),
    ("alpaca_output", {"instruction": "Summarize", "output": PII_SAMPLE}),
    ("alpaca_system", {"instruction": "Summarize", "system": PII_SAMPLE, "output": "ok"}),
    (
        "sharegpt_human",
        {"conversations": [{"from": "human", "value": PII_SAMPLE}, {"from": "gpt", "value": "ok"}]},
    ),
    (
        "sharegpt_gpt",
        {"conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": PII_SAMPLE}]},
    ),
    (
        "llava",
        {
            "image": "img.png",
            "conversations": [
                {"from": "human", "value": PII_SAMPLE},
                {"from": "gpt", "value": "ok"},
            ],
        },
    ),
    (
        "sharegpt4v",
        {
            "image": "img.png",
            "conversations": [
                {"from": "human", "value": PII_SAMPLE},
                {"from": "gpt", "value": "ok"},
            ],
        },
    ),
    ("dpo_prompt", {"prompt": PII_SAMPLE, "chosen": "c", "rejected": "r"}),
    ("dpo_chosen_str", {"prompt": "p", "chosen": PII_SAMPLE, "rejected": "r"}),
    ("dpo_rejected_str", {"prompt": "p", "chosen": "c", "rejected": PII_SAMPLE}),
    (
        "dpo_chosen_conversational",
        {
            "prompt": "p",
            "chosen": [{"role": "assistant", "content": PII_SAMPLE}],
            "rejected": [{"role": "assistant", "content": "r"}],
        },
    ),
    (
        "dpo_rejected_conversational",
        {
            "prompt": "p",
            "chosen": [{"role": "assistant", "content": "c"}],
            "rejected": [{"role": "assistant", "content": PII_SAMPLE}],
        },
    ),
    ("kto_prompt", {"prompt": PII_SAMPLE, "completion": "ok", "label": True}),
    ("kto_completion", {"prompt": "p", "completion": PII_SAMPLE, "label": True}),
    ("embedding_anchor_str", {"anchor": PII_SAMPLE, "positive": "pos"}),
    ("embedding_positive_str", {"anchor": "anc", "positive": PII_SAMPLE}),
    ("embedding_negative_str", {"anchor": "anc", "positive": "pos", "negative": PII_SAMPLE}),
    ("embedding_anchor_list", {"anchor": [PII_SAMPLE], "positive": "pos"}),
    ("embedding_positive_list", {"anchor": "anc", "positive": [PII_SAMPLE]}),
    ("embedding_negative_list", {"anchor": "anc", "positive": "pos", "negative": [PII_SAMPLE]}),
    ("audio", {"audio": "a.wav", "messages": [{"role": "user", "content": PII_SAMPLE}]}),
    ("asr", {"audio": "a.wav", "text": PII_SAMPLE}),
    (
        "tool_calling_messages",
        {"messages": [{"role": "user", "content": PII_SAMPLE}], "tools": [], "tool_calls": []},
    ),
    (
        "tool_calling_tools_desc",
        {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f", "description": PII_SAMPLE}}],
            "tool_calls": [],
        },
    ),
    (
        "tool_calling_tool_calls_str",
        {
            "messages": [{"role": "assistant", "content": "run"}],
            "tools": [],
            "tool_calls": [
                {"function": {"name": "f", "arguments": f'{{"email": "{PII_SAMPLE}"}}'}}
            ],
        },
    ),
    (
        "tool_calling_tool_calls_dict",
        {
            "messages": [{"role": "assistant", "content": "run"}],
            "tools": [],
            "tool_calls": [{"function": {"name": "f", "arguments": {"contact": PII_SAMPLE}}}],
        },
    ),
    (
        "tool_calling_message_tool_calls",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "run",
                    "tool_calls": [{"function": {"name": "f", "arguments": PII_SAMPLE}}],
                }
            ],
            "tools": [],
            "tool_calls": [],
        },
    ),
]


@pytest.mark.parametrize(
    "case_name,row", ALL_FORMAT_PII_CASES, ids=[c[0] for c in ALL_FORMAT_PII_CASES]
)
def test_expect_no_pii_all_formats(case_name: str, row: Mapping[str, Any]) -> None:
    res = expect_no_pii([row])
    assert res.passed is False, f"Expected PII failure for {case_name}, but passed: {res}"
    assert res.num_violations == 1
    assert any("PII detected" in d for d in res.details)


# -----------------------------------------------------------------------------
# 2. Assistant refusal detection across formats
# -----------------------------------------------------------------------------


def test_expect_no_refusal_pattern_sharegpt_gpt() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "How do I do X?"},
            {"from": "gpt", "value": REFUSAL_SAMPLE},
        ]
    }
    res = expect_no_refusal_pattern([row])
    assert res.passed is False
    assert res.num_violations == 1
    assert "refusal pattern matched" in res.details[0]


def test_expect_no_refusal_pattern_sharegpt_assistant() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "How do I do X?"},
            {"from": "assistant", "value": REFUSAL_SAMPLE},
        ]
    }
    res = expect_no_refusal_pattern([row])
    assert res.passed is False
    assert res.num_violations == 1
    assert "refusal pattern matched" in res.details[0]


def test_expect_no_refusal_pattern_dpo_chosen_string() -> None:
    row = {
        "prompt": "How do I bake bread?",
        "chosen": REFUSAL_SAMPLE,
        "rejected": "Mix flour and water...",
    }
    res = expect_no_refusal_pattern([row])
    assert res.passed is False
    assert res.num_violations == 1
    assert "refusal pattern matched" in res.details[0]


def test_expect_no_refusal_pattern_dpo_chosen_messages() -> None:
    row = {
        "prompt": "How do I bake bread?",
        "chosen": [{"role": "assistant", "content": REFUSAL_SAMPLE}],
        "rejected": [{"role": "assistant", "content": "Mix flour and water..."}],
    }
    res = expect_no_refusal_pattern([row])
    assert res.passed is False
    assert res.num_violations == 1
    assert "refusal pattern matched" in res.details[0]


def test_expect_no_refusal_pattern_kto_completion() -> None:
    row = {
        "prompt": "How do I bake bread?",
        "completion": REFUSAL_SAMPLE,
        "label": True,
    }
    res = expect_no_refusal_pattern([row])
    assert res.passed is False
    assert res.num_violations == 1
    assert "refusal pattern matched" in res.details[0]


def test_expect_no_refusal_pattern_controls() -> None:
    # User message contains refusal words -> should NOT fail assistant refusal check.
    user_refusal = {
        "conversations": [
            {"from": "human", "value": REFUSAL_SAMPLE},
            {"from": "gpt", "value": "Sure, here is how you do it."},
        ]
    }
    res = expect_no_refusal_pattern([user_refusal])
    assert res.passed is True
    assert res.num_violations == 0

    # DPO rejected contains refusal -> should NOT fail (only chosen is assistant target).
    rejected_refusal = {
        "prompt": "How do I bake bread?",
        "chosen": "Mix flour and water...",
        "rejected": REFUSAL_SAMPLE,
    }
    res = expect_no_refusal_pattern([rejected_refusal])
    assert res.passed is True
    assert res.num_violations == 0


# -----------------------------------------------------------------------------
# 3. Token length counting across formats
# -----------------------------------------------------------------------------


def test_expect_token_length_between_across_formats() -> None:
    text_9_words = "one two three four five six seven eight nine"
    rows = [
        {"text": text_9_words},
        {"conversations": [{"from": "human", "value": text_9_words}]},
        {"instruction": text_9_words, "output": ""},
        {"messages": [{"role": "user", "content": text_9_words}]},
        {"prompt": text_9_words, "chosen": "", "rejected": ""},
        {"prompt": text_9_words, "completion": "", "label": True},
        {"anchor": text_9_words, "positive": ""},
        {"audio": "audio.wav", "text": text_9_words},
    ]

    for i, row in enumerate(rows):
        res = expect_token_length_between([row], min_tokens=9, max_tokens=9)
        assert res.passed is True, f"Row {i} ({row}) failed 9-word length check: {res}"

        res_fail = expect_token_length_between([row], min_tokens=10, max_tokens=20)
        assert res_fail.passed is False, f"Row {i} should fail [10, 20]: {res_fail}"
        assert res_fail.num_violations == 1
        assert "9 tokens" in res_fail.details[0]


# -----------------------------------------------------------------------------
# 4. Fail-closed on rows with no extractable text
# -----------------------------------------------------------------------------


def test_no_extractable_text_fails_closed() -> None:
    unextractable_rows = [
        {},
        {"image": "photo.png"},
        {"count": 42},
    ]

    pii_res = expect_no_pii(unextractable_rows)
    assert pii_res.passed is False
    assert pii_res.num_violations == 3
    assert all("no extractable text" in d for d in pii_res.details)

    length_res = expect_token_length_between(unextractable_rows, min_tokens=1, max_tokens=100)
    assert length_res.passed is False
    assert length_res.num_violations == 3
    assert all("no extractable text" in d for d in length_res.details)

    refusal_res = expect_no_refusal_pattern(unextractable_rows)
    assert refusal_res.passed is False
    assert refusal_res.num_violations == 3
    assert all("no extractable text" in d for d in refusal_res.details)


# -----------------------------------------------------------------------------
# 5. Ratchet test: FORMAT_SIGNATURES vs FORMAT_EXTRACTION_RULES
# -----------------------------------------------------------------------------


def test_ratchet_all_format_signatures_have_extraction_rules() -> None:
    for fmt_name in FORMAT_SIGNATURES:
        assert fmt_name in FORMAT_EXTRACTION_RULES, (
            f"Format {fmt_name!r} in FORMAT_SIGNATURES lacks rules in FORMAT_EXTRACTION_RULES"
        )
        rule = FORMAT_EXTRACTION_RULES[fmt_name]
        assert "text_fields" in rule
        assert "assistant_fields" in rule
        assert isinstance(rule["text_fields"], tuple)
        assert isinstance(rule["assistant_fields"], tuple)


# -----------------------------------------------------------------------------
# 6. CLI reproduction test from Issue #1231 exits 2
# -----------------------------------------------------------------------------


def test_cli_expect_flags_pii_and_refusal_and_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # 4 rows from the issue: ShareGPT (PII), DPO (PII), Alpaca input (PII), DPO (refusal).
    data_file = tmp_path / "data.jsonl"
    data_rows = [
        {"conversations": [{"from": "human", "value": PII_SAMPLE}, {"from": "gpt", "value": "ok"}]},
        {"prompt": "hi", "chosen": PII_SAMPLE, "rejected": "no"},
        {"instruction": "Summarize", "input": PII_SAMPLE, "output": "ok"},
        {"prompt": "How do I bake bread?", "chosen": REFUSAL_SAMPLE, "rejected": "Mix flour..."},
    ]
    with data_file.open("w", encoding="utf-8") as f:
        for r in data_rows:
            f.write(json.dumps(r) + "\n")

    suite_file = tmp_path / "suite.yaml"
    suite_yaml = """
name: test_all_formats_suite
expectations:
  - name: expect_no_pii
  - name: expect_no_refusal_pattern
"""
    suite_file.write_text(suite_yaml, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["expect", str(data_file), str(suite_file)])

    assert result.exit_code == 2, (
        f"Expected exit code 2, got {result.exit_code}. Output:\n{result.output}"
    )

    clean_out = " ".join(strip_ansi(result.output).split())
    assert "expect_no_pii" in clean_out
    assert "expect_no_refusal_pattern" in clean_out
    assert "FAIL" in clean_out
    assert "Violations: expect_no_pii" in clean_out
    assert "Violations: expect_no_refusal_pattern" in clean_out


# -----------------------------------------------------------------------------
# 7. Comprehensive matrix: every field of every format scanned
# -----------------------------------------------------------------------------

NON_TEXT_FIELDS = frozenset({"label", "image", "audio"})

_SHAPES = {
    "conversations": lambda t: [{"from": "human", "value": "q"}, {"from": "gpt", "value": t}],
    "messages": lambda t: [{"role": "user", "content": "q"}, {"role": "assistant", "content": t}],
    "tools": lambda t: [{"type": "function", "function": {"name": "f", "description": t}}],
    "tool_calls": lambda t: [{"function": {"name": "f", "arguments": t}}],
    "label": lambda t: True,
    "image": lambda t: "img.png",
    "audio": lambda t: "a.wav",
}


def _row(fmt: str, field: str, payload: str) -> dict:
    keys = (
        set(FORMAT_SIGNATURES[fmt])
        | set(FORMAT_EXTRACTION_RULES[fmt]["text_fields"])
        | {field}
    )
    return {
        key: _SHAPES.get(key, lambda t: t)(payload if key == field else "plain words")
        for key in sorted(keys)
    }


def _text_fields(fmt: str) -> list:
    declared = set(FORMAT_EXTRACTION_RULES[fmt]["text_fields"])
    return sorted(declared | (set(FORMAT_SIGNATURES[fmt]) - NON_TEXT_FIELDS))


PII_CASES = [(fmt, field) for fmt in sorted(FORMAT_SIGNATURES) for field in _text_fields(fmt)]
REFUSAL_CASES = [
    (fmt, field)
    for fmt in sorted(FORMAT_SIGNATURES)
    for field in FORMAT_EXTRACTION_RULES[fmt]["assistant_fields"]
]


@pytest.mark.parametrize("fmt,field", PII_CASES, ids=[f"{f}.{k}" for f, k in PII_CASES])
def test_every_text_field_of_every_format_is_scanned(fmt: str, field: str) -> None:
    res = expect_no_pii([_row(fmt, field, PII_SAMPLE)])
    assert any("PII detected" in d for d in res.details), (fmt, field, res)


@pytest.mark.parametrize("fmt,field", REFUSAL_CASES, ids=[f"{f}.{k}" for f, k in REFUSAL_CASES])
def test_every_assistant_field_of_every_format_is_scanned(fmt: str, field: str) -> None:
    res = expect_no_refusal_pattern([_row(fmt, field, REFUSAL_SAMPLE)])
    assert any("refusal pattern matched" in d for d in res.details), (fmt, field, res)


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "What is 2+2?"},
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        {"conversations": [{"from": "human", "value": "What is 2+2?"}]},
        {"anchor": "a cat", "positive": "a feline"},
    ],
    ids=["prompt_only", "chatml_user_only", "sharegpt_human_only", "embedding"],
)
def test_rows_without_an_assistant_turn_are_not_refusal_violations(row: dict) -> None:
    assert expect_no_refusal_pattern([row]).passed is True


def test_refusal_in_a_user_message_is_not_an_assistant_refusal() -> None:
    row = {
        "messages": [
            {"role": "user", "content": REFUSAL_SAMPLE},
            {"role": "assistant", "content": "Sure, here is how."},
        ]
    }
    assert expect_no_refusal_pattern([row]).passed is True


def test_pii_after_a_long_field_is_still_found() -> None:
    row = {"instruction": "Summarise", "input": "lorem ipsum " * 5000, "output": PII_SAMPLE}
    res = expect_no_pii([row])
    assert any("PII detected" in d for d in res.details), res


def test_plaintext_assistant_refusal_flags_and_ok_passes() -> None:
    assert expect_no_refusal_pattern([{"text": REFUSAL_SAMPLE}]).passed is False
    assert expect_no_refusal_pattern([{"text": "ok"}]).passed is True


def test_raft_and_prm_formats_are_scanned_properly() -> None:
    raft_clean = {
        "query": "What is Python?",
        "golden_doc": "Python is a language.",
        "distractor_docs": ["Cats are pets."],
        "answer": "A programming language.",
    }
    assert expect_no_pii([raft_clean]).passed is True
    assert expect_no_refusal_pattern([raft_clean]).passed is True

    raft_refusal = dict(raft_clean, answer=REFUSAL_SAMPLE)
    res_refusal = expect_no_refusal_pattern([raft_refusal])
    assert res_refusal.passed is False
    assert "refusal pattern matched" in res_refusal.details[0]

    prm_row_pii = {
        "prompt": "What is 2+2?",
        "completions": [f"Step 1: {PII_SAMPLE}"],
        "labels": [True],
    }
    res_prm_pii = expect_no_pii([prm_row_pii])
    assert res_prm_pii.passed is False
    assert any("PII detected" in d for d in res_prm_pii.details)


def test_chosen_user_message_refusal_does_not_flag_assistant_refusal() -> None:
    row = {
        "prompt": "p",
        "chosen": [{"role": "user", "content": REFUSAL_SAMPLE}],
        "rejected": "r",
    }
    assert expect_no_refusal_pattern([row]).passed is True
