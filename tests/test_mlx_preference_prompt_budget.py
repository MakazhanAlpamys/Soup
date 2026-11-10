"""MLX preference rows keep fitting prompts, including special-token overhead."""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.mlx_dpo import build_preference_example, collate_pairs
from soup_cli.trainer.mlx_orpo import build_orpo_example, collate_orpo
from tests.test_mlx_dpo import FakeTokenizer
from tests.test_mlx_orpo import BOS, EOS, CharTokenizer, MergingTokenizer


@pytest.mark.parametrize("max_length", [11, 12])
def test_dpo_fitting_prompt_longer_than_half_stays_whole(max_length: int) -> None:
    tokenizer = FakeTokenizer()
    row = {"prompt": "ABCDEFG", "chosen": "xy", "rejected": "z"}
    example = build_preference_example(
        row, tokenizer, max_length=max_length, max_prompt_length=max_length // 2,
    )
    assert example.prompt_ids == tokenizer.encode(row["prompt"])
    assert example.chosen_ids == [ord(c) for c in "xy\x03"]
    assert example.rejected_ids == [ord(c) for c in "z\x03"]
    tokens, masks = collate_pairs([example])
    assert tokens[0] == tokenizer.encode("ABCDEFGxy\x03")
    assert masks[0] == [0] * 8 + [1] * 3
    assert masks[1] == [0] * 8 + [1] * 2 + [0]


@pytest.mark.parametrize("long_side", ["chosen", "rejected"])
def test_dpo_one_token_overflow_caps_shared_prompt_and_preserves_completions(
    long_side: str,
) -> None:
    row = {"prompt": "ABCDEFG", "chosen": "x", "rejected": "y"}
    row[long_side] *= 2
    example = build_preference_example(
        row, FakeTokenizer(), max_length=10, max_prompt_length=5,
    )
    assert example.prompt_ids == FakeTokenizer().encode("ABCD")
    assert example.chosen_ids == [ord(c) for c in row["chosen"] + "\x03"]
    assert example.rejected_ids == [ord(c) for c in row["rejected"] + "\x03"]


@pytest.mark.parametrize("max_length", [11, 12])
def test_orpo_fitting_prompt_counts_bos_and_eos_and_keeps_masks(max_length: int) -> None:
    tokenizer = CharTokenizer()
    row = {"prompt": "ABCDEFG", "chosen": "xy", "rejected": "z"}
    example = build_orpo_example(
        row, tokenizer, max_length=max_length, max_prompt_length=max_length // 2,
    )
    prompt = [BOS] + tokenizer.encode("ABCDEFG", add_special_tokens=False)
    chosen = tokenizer.encode("xy", add_special_tokens=False) + [EOS]
    rejected = tokenizer.encode("z", add_special_tokens=False) + [EOS]
    assert example.chosen_ids == prompt + chosen
    assert example.rejected_ids == prompt + rejected
    assert example.chosen_prompt_len == example.rejected_prompt_len == 8
    tokens, completion, sequence = collate_orpo([example])
    assert tokens == [prompt + chosen, prompt + rejected + [0]]
    assert completion == [[0] * 8 + [1] * 3, [0] * 8 + [1] * 2 + [0]]
    assert sequence == [[1] * 11, [1] * 10 + [0]]


@pytest.mark.parametrize("long_side", ["chosen", "rejected"])
def test_orpo_overflow_in_either_side_caps_both_prompts_with_bos(long_side: str) -> None:
    tokenizer = CharTokenizer()
    row = {"prompt": "ABCDEFG", "chosen": "x", "rejected": "y"}
    row[long_side] *= 2
    example = build_orpo_example(row, tokenizer, max_length=10, max_prompt_length=5)
    prompt = [BOS] + tokenizer.encode("DEFG", add_special_tokens=False)
    assert example.chosen_prompt_len == example.rejected_prompt_len == 5
    assert example.chosen_ids == prompt + tokenizer.encode(row["chosen"], False) + [EOS]
    assert example.rejected_ids == prompt + tokenizer.encode(row["rejected"], False) + [EOS]


def test_orpo_fitting_pair_preserves_distinct_merged_prompt_spans() -> None:
    tokenizer = MergingTokenizer()
    row = {"prompt": "abcdefx", "chosen": "y1", "rejected": "q2"}
    example = build_orpo_example(row, tokenizer, max_length=12, max_prompt_length=6)
    assert example.chosen_prompt_len == 7
    assert example.rejected_prompt_len == 8
    assert example.chosen_ids == [BOS] + tokenizer.encode("abcdefxy1", False) + [EOS]
    assert example.rejected_ids == [BOS] + tokenizer.encode("abcdefxq2", False) + [EOS]
    _, completion, _ = collate_orpo([example])
    assert completion[0] == [0] * 7 + [1] * 3 + [0]
    assert completion[1] == [0] * 8 + [1] * 3


def test_orpo_pair_budget_uses_each_actual_span_not_unrelated_maxima() -> None:
    tokenizer = MergingTokenizer()
    row = {"prompt": "abcdefx", "chosen": "y12", "rejected": "q2"}
    example = build_orpo_example(row, tokenizer, max_length=11, max_prompt_length=5)
    # The longer answer belongs to the shorter prompt: both complete inputs fit.
    assert example.chosen_prompt_len == 7
    assert example.rejected_prompt_len == 8
    assert example.chosen_ids == [BOS] + tokenizer.encode("abcdefxy12", False) + [EOS]
    assert example.rejected_ids == [BOS] + tokenizer.encode("abcdefxq2", False) + [EOS]
    _, completion, sequence = collate_orpo([example])
    assert completion == [[0] * 7 + [1] * 4, [0] * 8 + [1] * 3]
    assert sequence == [[1] * 11, [1] * 11]


@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo"])
@pytest.mark.parametrize("conversational", [False, True])
def test_actual_trainer_builder_keeps_fitting_prompt(
    task: str, conversational: bool,
) -> None:
    from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper
    from soup_cli.trainer.mlx_orpo import MLXORPOTrainerWrapper
    from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper

    wrapper_class = {
        "dpo": MLXDPOTrainerWrapper, "orpo": MLXORPOTrainerWrapper,
        "simpo": MLXSimPOTrainerWrapper,
    }[task]
    cfg = load_config_from_string(
        f"base: test/model\nbackend: mlx\ntask: {task}\n"
        "data:\n  train: ./train.jsonl\n  format: dpo\n  max_length: 64\n"
    )
    wrapper = wrapper_class(cfg)
    wrapper.tokenizer = FakeTokenizer() if task == "dpo" else CharTokenizer()
    row = {"prompt": "P" * 40, "chosen": "a", "rejected": "bb"}
    if conversational:
        # Template overhead matters too: this still fits with the longer answer.
        row = {
            "prompt": [{"role": "user", "content": "P" * 18}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "bb"}],
        }
    if task == "dpo":
        expected = build_preference_example(
            row, wrapper.tokenizer, max_length=64, max_prompt_length=64,
        )
        actual = wrapper._build_example(row, 64)
        assert len(actual.prompt_ids) > 32
        assert actual == expected
    else:
        expected = build_orpo_example(
            row, wrapper.tokenizer, max_length=64, max_prompt_length=64,
        )
        actual = wrapper._build_example(row, 64)
        assert actual.chosen_prompt_len > 32
        assert actual == expected


@pytest.mark.parametrize("builder,tokenizer", [
    (build_preference_example, FakeTokenizer()), (build_orpo_example, CharTokenizer()),
])
def test_long_answers_still_stop_at_the_remaining_budget(builder, tokenizer) -> None:
    example = builder(
        {"prompt": "ABCDEFG", "chosen": "x" * 20, "rejected": "y"},
        tokenizer, max_length=10, max_prompt_length=5,
    )
    if builder is build_preference_example:
        assert len(example.prompt_ids) == 5
        assert example.chosen_ids == [ord("x")] * 5
        assert example.rejected_ids == [ord("y"), 3]
    else:
        assert example.chosen_prompt_len == example.rejected_prompt_len == 5
        assert example.chosen_ids[-5:] == tokenizer.encode("x" * 5, False)
        assert len(example.chosen_ids) == 10
        assert example.rejected_ids[-2:] == tokenizer.encode("y", False) + [EOS]


@pytest.mark.parametrize("builder,tokenizer", [
    (build_preference_example, FakeTokenizer()), (build_orpo_example, CharTokenizer()),
])
def test_answer_overflow_leaves_a_short_prompt_unchanged(builder, tokenizer) -> None:
    row = {"prompt": "AB", "chosen": "x" * 20, "rejected": "y"}
    example = builder(row, tokenizer, max_length=10, max_prompt_length=5)
    prompt = tokenizer.encode("AB")
    if builder is build_preference_example:
        assert example.prompt_ids == prompt
        assert example.chosen_ids == [ord("x")] * 7
        assert example.rejected_ids == [ord("y"), 3]
    else:
        assert example.chosen_prompt_len == example.rejected_prompt_len == 3
        assert example.chosen_ids == prompt + tokenizer.encode("x" * 7, False)
        assert example.rejected_ids == prompt + tokenizer.encode("y", False) + [EOS]


@pytest.mark.parametrize("has_bos", [False, True])
def test_orpo_overflow_with_one_token_prompt_budget(has_bos: bool) -> None:
    from tests.test_mlx_prompt_cap_bos import MockTokenizer

    tokenizer = MockTokenizer(has_bos=has_bos)
    example = build_orpo_example(
        {"prompt": "ABCDEFG", "chosen": "xy", "rejected": "z"},
        tokenizer, max_length=6, max_prompt_length=1,
    )
    prompt = [BOS] if has_bos else tokenizer.encode("G", False)
    assert example.chosen_prompt_len == example.rejected_prompt_len == 1
    assert example.chosen_ids == prompt + tokenizer.encode("xy", False) + [EOS]
    assert example.rejected_ids == prompt + tokenizer.encode("z", False) + [EOS]
