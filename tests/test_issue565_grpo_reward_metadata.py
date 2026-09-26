"""Regression tests for #565: preserve GRPO reward metadata end to end."""

from __future__ import annotations

import json

import pytest

from soup_cli.config.schema import DataConfig, TrainingConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.grpo import (
    _prepare_grpo_dataset,
    _validate_grpo_reward_metadata,
)
from soup_cli.trainer.rewards import accuracy_reward, validate_reward_funcs


def _write_jsonl(tmp_path, rows: list[dict]):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_grpo_local_alpaca_preserves_metadata_and_derives_answer(tmp_path):
    schema = {"type": "object", "required": ["result"]}
    path = _write_jsonl(
        tmp_path,
        [
            {
                "instruction": "Return 2.",
                "input": "",
                "output": "2",
                "expected": "2",
                "schema": schema,
                "difficulty": "easy",
            }
        ],
    )
    cfg = DataConfig(train=str(path), format="alpaca", val_split=0)

    loaded = load_dataset(cfg, preserve_source_columns=True)
    prepared = _prepare_grpo_dataset(loaded["train"])

    assert prepared == [
        {
            "prompt": [{"role": "user", "content": "Return 2."}],
            "instruction": "Return 2.",
            "input": "",
            "output": "2",
            "expected": "2",
            "schema": schema,
            "difficulty": "easy",
            "answer": "2",
        }
    ]


def test_default_loader_contract_still_drops_unrelated_source_columns(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            {
                "instruction": "Return 2.",
                "output": "2",
                "private_reward_metadata": "must not leak into SFT",
            }
        ],
    )
    cfg = DataConfig(train=str(path), format="alpaca", val_split=0)

    loaded = load_dataset(cfg)

    assert loaded["train"] == [
        {
            "messages": [
                {"role": "user", "content": "Return 2."},
                {"role": "assistant", "content": "2"},
            ]
        }
    ]


def test_explicit_answer_wins_over_assistant_reference():
    prepared = _prepare_grpo_dataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "assistant gold"},
                ],
                "answer": "explicit gold",
                "source_id": "row-1",
            }
        ]
    )

    assert prepared[0]["answer"] == "explicit gold"
    assert prepared[0]["source_id"] == "row-1"


def test_multi_turn_history_reaches_grpo_with_only_final_reference_removed():
    messages = [
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Answer one"},
        {"role": "user", "content": "Question two"},
        {"role": "assistant", "content": "Answer two"},
    ]

    prepared = _prepare_grpo_dataset([{"messages": messages}])

    assert prepared[0]["prompt"] == messages[:-1]
    assert prepared[0]["answer"] == "Answer two"


@pytest.mark.parametrize(
    ("reward_fn", "domain", "row"),
    [
        ("accuracy", None, {"prompt": "question"}),
        ("verifiable", "math", {"prompt": "question"}),
        ("verifiable", "code", {"prompt": "question"}),
        ("verifiable", "json_schema", {"prompt": "question"}),
    ],
)
def test_gold_dependent_rewards_fail_before_generation(reward_fn, domain, row):
    config = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)

    with pytest.raises(ValueError, match=r"GRPO train row 0 is missing"):
        _validate_grpo_reward_metadata([row], config, split="train")


def test_code_reward_accepts_expected_without_answer():
    config = TrainingConfig(reward_fn="verifiable", verifiable_domain="code")

    _validate_grpo_reward_metadata(
        [{"prompt": "question", "expected": "stdout"}],
        config,
        split="train",
    )


def test_empty_gold_fails_normal_loader_to_grpo_setup_path(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [{"instruction": "Return 2.", "input": "", "output": "   "}],
    )
    cfg = DataConfig(train=str(path), format="alpaca", val_split=0)
    loaded = load_dataset(cfg, preserve_source_columns=True)
    prepared = _prepare_grpo_dataset(loaded["train"])

    with pytest.raises(ValueError, match=r"GRPO train row 0 is missing or empty 'answer'"):
        _validate_grpo_reward_metadata(
            prepared,
            TrainingConfig(reward_fn="accuracy"),
            split="train",
        )


@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_accuracy_reward_never_awards_empty_gold(answer):
    completions = [[{"role": "assistant", "content": "unrelated completion"}]]

    assert accuracy_reward(completions, answer=[answer]) == [0.0]


def test_reward_contract_rejects_wrong_score_count():
    def bad_reward(completions, **kwargs):
        return []

    checked = validate_reward_funcs(bad_reward)

    with pytest.raises(ValueError, match=r"returned 0 scores for 2 completions"):
        checked(completions=[[{"content": "a"}], [{"content": "b"}]])


def test_reward_contract_rejects_non_finite_score():
    def bad_reward(completions, **kwargs):
        return [float("nan") for _ in completions]

    checked = validate_reward_funcs(bad_reward)

    with pytest.raises(ValueError, match="non-finite"):
        checked(completions=[[{"content": "a"}]])


def test_reward_contract_preserves_ensemble_shape_and_valid_scores():
    def reward_a(completions, **kwargs):
        return [0.0 for _ in completions]

    def reward_b(completions, **kwargs):
        return [1.0 for _ in completions]

    checked = validate_reward_funcs([reward_a, reward_b])

    assert isinstance(checked, list)
    assert [reward(completions=[[{"content": "a"}]]) for reward in checked] == [
        [0.0],
        [1.0],
    ]
