"""#1208 — SimPO must not train to NaN behind a logged loss of 0.0.

On trl 0.29.x a SimPO row whose `chosen` is longer than `data.max_length` loses
its whole `rejected`: trl's CPO truncation slices each answer to
`answer[: max_length - longer_response_length]`
(`trl/experimental/cpo/cpo_trainer.py:513-515`, identical in 0.29.0 and 0.29.1),
so the shorter answer ends with zero tokens and SimPO's length-normalised
log-probability becomes 0/0. Every LoRA tensor goes NaN, and because
transformers' `logging_nan_inf_filter` defaults to true the loss reads 0.0.

These tests pin the predicate that detects such a row and the refusal that stops
the run, against the same arithmetic trl applies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

MAX_LENGTH = 64


class _FakeDataset:
    def __init__(self, rows, column_names):
        self.rows = rows
        self.column_names = column_names

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


class _FakeTrainer:
    def __init__(self, train_rows, columns, eval_rows=None):
        self.train_dataset = _FakeDataset(train_rows, columns)
        self.eval_dataset = (
            _FakeDataset(eval_rows, columns) if eval_rows is not None else None
        )


class _Cfg:
    def __init__(self, max_length=MAX_LENGTH):
        class _Data:
            pass

        self.data = _Data()
        self.data.max_length = max_length


class _Wrapper:
    """Just the guard, bound to a real SimPO wrapper's method."""

    def __init__(self, max_length=MAX_LENGTH):
        self.config = _Cfg(max_length)


def _guard(wrapper, trainer, split="train"):
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    return SimPOTrainerWrapper._refuse_empty_completion_rows(
        wrapper, trainer, split=split,
    )


def _combined(chosen_len: int, rejected_len: int, prompt: int = 4) -> dict:
    """A prepared combined-layout row: -100 over the prompt, ids after it."""
    marker = [-100] * prompt
    return {
        "chosen_labels": marker + list(range(chosen_len)),
        "rejected_labels": marker + list(range(rejected_len)),
    }


COMBINED_COLUMNS = {"chosen_labels", "rejected_labels"}


# --------------------------------------------------------------------------- #
# The predicate, on both layouts trl prepares
# --------------------------------------------------------------------------- #


class TestThePredicateMatchesTrl:
    def test_a_lopsided_pair_is_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        # The issue's shape: an 80-token chosen against " bad bad world".
        rows = [{"chosen_ids": list(range(80)), "rejected_ids": list(range(4))}]
        dataset = _FakeDataset(rows, {"chosen_ids", "rejected_ids"})
        assert preference_rows_with_empty_completion(
            dataset, max_length=MAX_LENGTH, layout="dpo",
        ) == [0]

    def test_a_balanced_pair_is_not_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(20, 18)]
        dataset = _FakeDataset(rows, COMBINED_COLUMNS)
        assert preference_rows_with_empty_completion(
            dataset, max_length=MAX_LENGTH, layout="combined",
        ) == []

    def test_a_longer_that_keeps_fourteen_tokens_is_not_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        # The issue's own control: a long `chosen` whose `rejected` keeps 14
        # trainable tokens trains normally.
        rows = [_combined(60, 14)]
        dataset = _FakeDataset(rows, COMBINED_COLUMNS)
        assert preference_rows_with_empty_completion(
            dataset, max_length=MAX_LENGTH, layout="combined",
        ) == []

    def test_only_the_affected_row_is_reported(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(20, 18), _combined(80, 4), _combined(19, 19)]
        dataset = _FakeDataset(rows, COMBINED_COLUMNS)
        assert preference_rows_with_empty_completion(
            dataset, max_length=MAX_LENGTH, layout="combined",
        ) == [1]

    def test_an_unknown_layout_is_left_to_the_caller(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        dataset = _FakeDataset([{"text": "x"}], {"text"})
        assert preference_rows_with_empty_completion(
            dataset, max_length=MAX_LENGTH, layout="combined",
        ) == []

    def test_the_installed_trl_still_truncates_the_way_this_assumes(self):
        """The predicate rests on trl's own slice. Pin that slice.

        Read from trl's source file rather than through its lazy module, so this
        works on the trl that has moved CPO to `trl.experimental.cpo` as well as
        one that has not. If trl changes the rule, this fails here rather than
        leaving the guard and the trainer disagreeing about what a zero-token
        row is.
        """
        import trl

        candidates = [
            Path(trl.__file__).parent / "experimental" / "cpo" / "cpo_trainer.py",
            Path(trl.__file__).parent / "trainer" / "cpo_trainer.py",
        ]
        sources = [path.read_text(encoding="utf-8") for path in candidates if path.is_file()]
        assert sources, "trl ships no cpo_trainer.py to pin the truncation against"
        source = "\n".join(sources)
        assert "longer_response_length" in source, (
            "trl's CPO no longer truncates by the longer response length; "
            "re-derive #1208's guard against the new rule"
        )
        assert "max_length - longer_response_length" in source, (
            "trl's CPO truncation no longer subtracts the longer length from "
            "max_length; re-derive #1208's guard"
        )


# --------------------------------------------------------------------------- #
# The refusal: it names the count and the limit, and stays quiet on a safe set.
# --------------------------------------------------------------------------- #


class TestTheRefusal:
    def test_a_zero_token_row_is_refused_with_count_and_limit(self):
        trainer = _FakeTrainer([_combined(80, 4)], COMBINED_COLUMNS)
        with pytest.raises(ValueError) as excinfo:
            _guard(_Wrapper(), trainer)
        message = str(excinfo.value)
        assert "1 train row(s)" in message, message
        assert f"max_length={MAX_LENGTH}" in message, message
        # The message has to say why a zero-looking 0.0 is what the run would do.
        assert "NaN" in message and "0.0" in message, message

    def test_the_refusal_names_rows_past_the_fifth(self):
        rows = [_combined(80, 2) for _ in range(7)]
        trainer = _FakeTrainer(rows, COMBINED_COLUMNS)
        with pytest.raises(ValueError) as excinfo:
            _guard(_Wrapper(), trainer)
        message = str(excinfo.value)
        assert "7 train row(s)" in message, message
        assert "+2 more" in message, message

    def test_a_safe_dataset_does_not_raise(self):
        trainer = _FakeTrainer([_combined(20, 18)], COMBINED_COLUMNS)
        _guard(_Wrapper(), trainer)  # must not raise

    def test_the_eval_split_is_checked_too(self):
        trainer = _FakeTrainer(
            [_combined(20, 18)], COMBINED_COLUMNS, eval_rows=[_combined(80, 4)],
        )
        with pytest.raises(ValueError) as excinfo:
            _guard(_Wrapper(), trainer, split="eval")
        assert "1 eval row(s)" in str(excinfo.value), str(excinfo.value)

    def test_a_larger_max_length_clears_the_same_pair(self):
        """The refusal is about this limit, not the data being unusable."""
        trainer = _FakeTrainer([_combined(80, 4)], COMBINED_COLUMNS)
        _guard(_Wrapper(max_length=128), trainer)  # must not raise

    def test_an_unrecognised_layout_is_skipped_rather_than_guessed(self):
        trainer = _FakeTrainer([{"text": "x"}], {"text"})
        _guard(_Wrapper(), trainer)  # must not raise
