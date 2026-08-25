"""Regression coverage for issue #532's silent all-NaN SFT adapter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


class _BoundaryTokenizer:
    """Put one assistant token either inside or just beyond max_length=3."""

    chat_template = "{% generation %}"

    def apply_chat_template(self, messages, **kwargs):
        kept = messages[-1]["content"] == "kept"
        if kept:
            input_ids = [10, 11, 12]
            assistant_mask = [0, 0, 1]
        else:
            input_ids = [10, 11, 12, 13]
            assistant_mask = [0, 0, 0, 1]
        if kwargs.get("return_assistant_tokens_mask"):
            return {
                "input_ids": input_ids,
                "assistant_masks": assistant_mask,
            }
        return input_ids


def _data_cfg(max_length: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        train_on_responses_only=True,
        train_on_messages_with_train_field=False,
        max_length=max_length,
        chat_template=None,
        prompt_strategy=None,
    )


def _row(answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ]
    }


def test_one_surviving_assistant_target_is_accepted() -> None:
    from soup_cli.data.loss_mask import IGNORE_INDEX
    from soup_cli.data.sft_format import build_format_row

    formatted = build_format_row(_BoundaryTokenizer(), _data_cfg())(_row("kept"))

    assert formatted["labels"] == [IGNORE_INDEX, IGNORE_INDEX, 12]


def test_fully_truncated_target_names_row_and_max_length() -> None:
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.sft import _map_text_sft_rows

    format_row = build_format_row(_BoundaryTokenizer(), _data_cfg())

    with pytest.raises(
        ValueError,
        match=r"train row 2.*no causal-loss target.*data\.max_length=3",
    ):
        _map_text_sft_rows(
            [_row("kept"), _row("truncated")],
            format_row=format_row,
            split="train",
            max_length=3,
        )


def test_position_zero_alone_is_not_a_causal_loss_target() -> None:
    from soup_cli.data.loss_mask import ensure_causal_loss_target

    with pytest.raises(ValueError, match="no causal-loss target"):
        ensure_causal_loss_target([7, -100, -100], max_length=3)


def test_non_finite_training_metric_is_rejected() -> None:
    from soup_cli.trainer.sft import _assert_finite_training_state

    with pytest.raises(
        RuntimeError,
        match=r"non-finite training metric grad_norm=nan.*step 2.*refusing to save",
    ):
        _assert_finite_training_state(
            [{"loss": 0.0, "grad_norm": float("nan"), "step": 2}]
        )


def test_finite_training_metrics_are_accepted() -> None:
    from soup_cli.trainer.sft import _assert_finite_training_state

    _assert_finite_training_state(
        [{"loss": 2.2, "grad_norm": 1.5, "entropy": 0.9, "step": 1}]
    )


def test_non_finite_trainable_parameter_is_rejected() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.sft import _assert_finite_training_state

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    with pytest.raises(
        RuntimeError,
        match=r"non-finite trainable parameter 'weight'.*refusing to save",
    ):
        _assert_finite_training_state([], model=model)


def test_pretokenized_zero_target_row_is_rejected() -> None:
    from soup_cli.trainer.sft import _validate_pretokenized_targets

    class _Pretokenized:
        column_names = ["input_ids", "labels"]

        def __init__(self) -> None:
            self.rows = [
                {"input_ids": [1, 2], "labels": [-100, 2]},
                {"input_ids": [3, 4], "labels": [-100, -100]},
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    with pytest.raises(
        ValueError,
        match=r"validation row 2.*data\.max_length=8",
    ):
        _validate_pretokenized_targets(
            _Pretokenized(), split="validation", max_length=8
        )


def test_training_state_is_checked_before_final_save() -> None:
    from soup_cli.trainer.sft import SFTTrainerWrapper

    source = inspect.getsource(SFTTrainerWrapper.train)
    assert source.index("_assert_finite_training_state") < source.index(
        "self.trainer.save_model"
    )
