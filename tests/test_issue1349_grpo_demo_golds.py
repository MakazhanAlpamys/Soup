"""#1349: the shipped GRPO demo data and the ``reasoning`` template give GRPO numeric golds.

``grpo_demo`` ships ``reasoning_math.jsonl`` twice: the package copy that ``soup data demo``
installs and the ``examples/`` copy the example config trains on. Every row ends with an
``Answer:`` line that the shared final-answer parser reads to its value — including the two
rows whose answer names a variable (``c = 5``, ``x = 7``, read as ``5`` and ``7`` since
#1371) — so ``\\boxed{5}`` pays under ``accuracy`` and ``verifiable/math``.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from soup_cli.data.templates.reasoning import build_prompt
from soup_cli.trainer.grpo import _prepare_grpo_dataset
from soup_cli.trainer.rewards import load_reward_fn
from soup_cli.utils.final_answer import parse_reference

_REPO = Path(__file__).resolve().parents[1]
_PACKAGE_COPY = _REPO / "src" / "soup_cli" / "data" / "_fixtures" / "reasoning_math.jsonl"
_EXAMPLES_COPY = _REPO / "examples" / "data" / "reasoning_math.jsonl"
_GOLDS = [Decimal(17), Decimal(5), Decimal(6), Decimal(7), Decimal(30)]


def _prepared(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return _prepare_grpo_dataset(
        [{"prompt": row["instruction"], "answer": row["output"]} for row in rows]
    )


def _msg(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def test_the_two_fixture_copies_are_byte_identical():
    assert _PACKAGE_COPY.read_bytes() == _EXAMPLES_COPY.read_bytes()


@pytest.mark.parametrize("path", [_PACKAGE_COPY, _EXAMPLES_COPY], ids=["package", "examples"])
def test_every_row_parses_to_a_numeric_gold(path):
    parsed = [parse_reference(row["answer"]) for row in _prepared(path)]
    assert [p.number for p in parsed] == _GOLDS


@pytest.mark.parametrize(
    ("reward_fn", "domain"), [("accuracy", None), ("verifiable", "math")]
)
@pytest.mark.parametrize("completion", [r"\boxed{5}", "The answer is 5.", "#### 5"])
def test_row_one_pays_the_value_however_it_is_written(reward_fn, domain, completion):
    gold = _prepared(_PACKAGE_COPY)[1]["answer"]
    reward = load_reward_fn(reward_fn, verifiable_domain=domain)
    assert reward([_msg(completion)], answer=[gold]) == [1.0]
    assert reward([_msg(r"\boxed{6}")], answer=[gold]) == [0.0]


def test_the_math_domain_asks_for_the_hash_marker():
    assert "'####' on its own line" in build_prompt(5, "alpaca", "spec", domain="math")


@pytest.mark.parametrize("domain", ["logic", "code"])
def test_the_other_domains_ask_for_an_answer_line(domain):
    prompt = build_prompt(5, "alpaca", "spec", domain=domain)
    assert "'Answer: <answer>'" in prompt
    assert "####" not in prompt
