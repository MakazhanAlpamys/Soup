"""#810: ``soup data lint`` must judge a completion, not the prompt inside it.

A conversational DPO row carries its prompt as the leading user turn of
``chosen`` / ``rejected``. Flattening every turn put the prompt inside the text
``prompt_leak`` searches, so every row with a 40+ character prompt was flagged,
and ``length_bias`` measured prompt + answer. ``length_bias`` also had no
magnitude floor: Cohen's d is scale-free, so a one-word gap between lengths that
barely vary read as a large effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.utils.data_lint import (
    check_length_bias,
    check_prompt_leak,
    extract_completion_text,
    run_lint,
)

_REPO = Path(__file__).resolve().parents[1]
_PROMPT = "Explain, in a couple of sentences, why the sky looks blue at noon."


def _pairs(n: int = 10):
    return [
        (f"{_PROMPT} (variant {i})", f"Rayleigh scattering favours short wavelengths {i}",
         f"It reflects the colour of the ocean below it {i} too")
        for i in range(n)
    ]


def _conversational(prompt: str, answer: str):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def _as_rows(shape: str):
    rows = []
    for prompt, chosen, rejected in _pairs():
        if shape == "conversational":
            chosen, rejected = _conversational(prompt, chosen), _conversational(prompt, rejected)
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def _verdicts(report, *names):
    return {c.name: c.verdict for c in report.checks if c.name in names}


def test_conversational_and_plain_forms_of_the_same_pairs_get_the_same_verdicts():
    conversational = run_lint(_as_rows("conversational"), "dpo")
    plain = run_lint(_as_rows("plain"), "dpo")

    assert _verdicts(conversational, "prompt_leak", "length_bias") == _verdicts(
        plain, "prompt_leak", "length_bias"
    )
    assert _verdicts(conversational, "prompt_leak") == {"prompt_leak": "OK"}


def test_the_shipped_conversational_example_has_no_prompt_leak():
    path = _REPO / "examples" / "data" / "dpo_sample.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    assert _verdicts(run_lint(rows, "auto"), "prompt_leak") == {"prompt_leak": "OK"}


def test_a_prompt_echoed_inside_the_assistant_turn_is_still_flagged():
    rows = [
        {
            "prompt": prompt,
            "chosen": _conversational(prompt, f"You asked: {prompt} The answer is scattering."),
            "rejected": _conversational(prompt, rejected),
        }
        for prompt, _, rejected in _pairs()
    ]

    check = check_prompt_leak(rows, fmt="dpo")

    assert check.verdict == "MAJOR"
    assert check.message.startswith("10/10 rows")


def test_a_one_word_gap_with_zero_variance_is_not_major():
    rows = [{"chosen": "w " * 11, "rejected": "w " * 10} for _ in range(10)]

    check = check_length_bias(rows, length_fn=lambda text: float(len(text.split())))

    # Cohen's d falls back to 1.0 here; the 9.1% mean difference keeps it below MAJOR.
    assert check.verdict != "MAJOR", check.message


def test_a_near_identical_length_difference_is_ok():
    rows = [
        {"chosen": "w " * (20 + i % 3), "rejected": "w " * (21 + i % 3)} for i in range(10)
    ]

    check = check_length_bias(rows, length_fn=lambda text: float(len(text.split())))

    assert check.verdict == "OK", check.message


def test_a_large_length_bias_is_still_major():
    rows = [{"chosen": "w " * (40 + i), "rejected": "w " * (10 + i)} for i in range(10)]

    check = check_length_bias(rows, length_fn=lambda text: float(len(text.split())))

    assert check.verdict == "MAJOR", check.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain answer", "plain answer"),
        ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "a"),
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
            "a1\na2",
        ),
        ([{"content": "no roles"}, {"content": "at all"}], "no roles\nat all"),
    ],
)
def test_extract_completion_text(value, expected):
    assert extract_completion_text(value) == expected
