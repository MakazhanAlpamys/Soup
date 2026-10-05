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
        self.eval_dataset = _FakeDataset(eval_rows, columns) if eval_rows is not None else None


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
        wrapper,
        trainer,
        split=split,
    )


def _combined(chosen_len: int, rejected_len: int, prompt: int = 4) -> dict:
    """A row in the shape ``CPOTrainer`` hands over: already truncated.

    trl tokenises and truncates in ``__init__``, so these are POST-truncation
    shapes. The issue's original 80/4 pair arrives here as 64/0 — the longer side
    is exactly ``max_length`` trainable tokens and the short side is empty.
    """
    marker = [-100] * prompt
    return {
        "chosen_labels": marker + list(range(chosen_len)),
        "rejected_labels": marker + list(range(rejected_len)),
    }


COMBINED_COLUMNS = {"chosen_labels", "rejected_labels"}


# --------------------------------------------------------------------------- #
# The predicate, on the layout a CPOTrainer actually prepares
# --------------------------------------------------------------------------- #


class TestThePredicateMatchesTrl:
    def test_a_pair_emptied_by_truncation_is_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        # The issue's shape as trl hands it over: 80-token chosen against
        # " bad bad world" arrives already cut to 64 / 0.
        rows = [_combined(MAX_LENGTH, 0)]
        dataset = _FakeDataset(rows, COMBINED_COLUMNS)
        assert preference_rows_with_empty_completion(dataset) == [0]

    def test_an_empty_chosen_side_is_also_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(0, MAX_LENGTH)]
        assert preference_rows_with_empty_completion(
            _FakeDataset(rows, COMBINED_COLUMNS),
        ) == [0]

    def test_a_balanced_pair_is_not_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(20, 18)]
        assert (
            preference_rows_with_empty_completion(
                _FakeDataset(rows, COMBINED_COLUMNS),
            )
            == []
        )

    def test_a_truncated_pair_that_keeps_fourteen_tokens_is_not_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        # The issue's own control, in post-truncation shape: 64 / 14 trains
        # normally on `main`, and the first version of this guard refused it by
        # re-applying trl's slice to already-truncated rows (#1208 review).
        rows = [_combined(MAX_LENGTH, 14)]
        assert (
            preference_rows_with_empty_completion(
                _FakeDataset(rows, COMBINED_COLUMNS),
            )
            == []
        )

    def test_two_long_balanced_answers_are_not_flagged(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(MAX_LENGTH, MAX_LENGTH - 2)]
        assert (
            preference_rows_with_empty_completion(
                _FakeDataset(rows, COMBINED_COLUMNS),
            )
            == []
        )

    def test_only_the_affected_row_is_reported(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [_combined(20, 18), _combined(MAX_LENGTH, 0), _combined(19, 19)]
        assert preference_rows_with_empty_completion(
            _FakeDataset(rows, COMBINED_COLUMNS),
        ) == [1]

    def test_an_unknown_layout_is_left_to_the_caller(self):
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        dataset = _FakeDataset([{"text": "x"}], {"text"})
        assert preference_rows_with_empty_completion(dataset) == []

    def test_the_dpo_collator_shape_is_not_read(self):
        """SimPO only builds a CPOTrainer; `chosen_ids`/`rejected_ids` is
        DPOTrainer's collator input, so it is not a layout to flag on."""
        from soup_cli.trainer._trl_compat import (
            preference_rows_with_empty_completion,
        )

        rows = [{"chosen_ids": list(range(80)), "rejected_ids": []}]
        assert (
            preference_rows_with_empty_completion(
                _FakeDataset(rows, {"chosen_ids", "rejected_ids"}),
            )
            == []
        )

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
        trainer = _FakeTrainer([_combined(MAX_LENGTH, 0)], COMBINED_COLUMNS)
        with pytest.raises(ValueError) as excinfo:
            _guard(_Wrapper(), trainer)
        message = str(excinfo.value)
        assert "1 train row(s)" in message, message
        assert f"max_length={MAX_LENGTH}" in message, message
        # The message has to say why a zero-looking 0.0 is what the run would do.
        assert "NaN" in message and "0.0" in message, message

    def test_the_refusal_names_rows_past_the_fifth(self):
        rows = [_combined(MAX_LENGTH, 0) for _ in range(7)]
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
            [_combined(20, 18)],
            COMBINED_COLUMNS,
            eval_rows=[_combined(MAX_LENGTH, 0)],
        )
        with pytest.raises(ValueError) as excinfo:
            _guard(_Wrapper(), trainer, split="eval")
        assert "1 eval row(s)" in str(excinfo.value), str(excinfo.value)

    def test_a_truncated_pair_that_keeps_some_tokens_is_not_refused(self):
        """#1208 review: this pair trains fine on `main`, so the guard must not
        stop it just because both answers were cut."""
        trainer = _FakeTrainer([_combined(MAX_LENGTH, 14)], COMBINED_COLUMNS)
        _guard(_Wrapper(), trainer)  # must not raise

    def test_an_unrecognised_layout_is_skipped_rather_than_guessed(self):
        trainer = _FakeTrainer([{"text": "x"}], {"text"})
        _guard(_Wrapper(), trainer)  # must not raise


# --------------------------------------------------------------------------- #
# The call site and the data, through a real trl CPOTrainer (#1208 review)
# --------------------------------------------------------------------------- #


WORDS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "what",
    "is",
    "one",
    "plus",
    "?",
    "two",
    "three",
    "hello",
    "there",
    "hi",
    "no",
    "bad",
    "world",
]


def _tiny_model_dir(tmp_path):
    """A random-init 2-layer Llama plus a WordLevel tokenizer."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    raw = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"),
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    torch.manual_seed(0)
    path = tmp_path / "tiny"
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(WORDS),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            max_position_embeddings=512,
        )
    ).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _setup_simpo(tmp_path, monkeypatch, rows):
    """Run a real SimPO `setup()` and return the wrapper.

    `importorskip("trl")` is not enough: trl is a lazy module, so that passes
    while `trl.experimental.cpo` still fails to import. Resolving the concrete
    class is what actually tells us the test can run.
    """
    from soup_cli.trainer._trl_compat import resolve_trl_symbol

    try:
        resolve_trl_symbol("CPOConfig", "trl.experimental.cpo")
    except Exception as exc:  # noqa: BLE001 — a broken trl is not a failure
        pytest.skip(f"this environment cannot import a trl CPO config: {exc}")

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_model_dir(tmp_path)
    cfg = load_config_from_string(
        f"base: {base.as_posix()}\ntask: simpo\n"
        "data: {train: ./unused.jsonl, format: dpo, max_length: 64}\n"
        "training:\n  epochs: 1\n  batch_size: 2\n  quantization: none\n"
        "  lora: {r: 4, alpha: 8, dropout: 0.0}\noutput: ./out\n"
    )
    wrapper = SimPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows})
    return wrapper


def _rows(chosen_words: int, rejected: str):
    return [
        {
            "prompt": "what is one plus one ?",
            "chosen": " ".join(["hello"] * chosen_words),
            "rejected": rejected,
        }
    ] * 4


def _trainable(row, side: str) -> int:
    return sum(1 for x in row[f"{side}_labels"] if x != -100)


class TestTheGuardIsWiredIntoSetup:
    """Deleting the two calls in `setup()` must fail here.

    Every other test in the refusal classes calls the guard on a fake trainer, so
    with the calls removed they all still passed — the run was silently inert
    again (#1208 review). `TestOnARealCpoTrainer` is the real evidence; this one
    runs without trl, so the wiring cannot go unpinned on a machine where the CPO
    trainer cannot import.
    """

    def test_setup_invokes_the_guard_for_both_splits(self):
        import inspect

        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        source = inspect.getsource(SimPOTrainerWrapper.setup)
        assert '_refuse_empty_completion_rows(self.trainer, split="train")' in source, (
            "SimPOTrainerWrapper.setup no longer runs the empty-completion guard "
            "on the train split; a PPO-path SimPO run with a lopsided pair would "
            "train to NaN behind a logged loss of 0.0"
        )
        assert '_refuse_empty_completion_rows(self.trainer, split="eval")' in source, (
            "the eval split is no longer guarded"
        )


class TestOnARealCpoTrainer:
    """The guard as trl's own prepared dataset presents it (#1208 review)."""

    def test_the_issue_shape_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="zero completion tokens"):
            _setup_simpo(tmp_path, monkeypatch, _rows(80, " bad bad world"))

    def test_the_issues_own_control_still_loads(self, tmp_path, monkeypatch):
        wrapper = _setup_simpo(
            tmp_path,
            monkeypatch,
            _rows(80, " ".join(["bad"] * 30)),
        )
        assert _trainable(wrapper.trainer.train_dataset[0], "rejected") == 14

    def test_two_long_balanced_answers_still_load(self, tmp_path, monkeypatch):
        rows = [
            {
                "prompt": "what is one plus one ?",
                "chosen": " ".join(["hello"] * 70),
                "rejected": " ".join(["bad"] * 68),
            }
        ] * 4
        wrapper = _setup_simpo(tmp_path, monkeypatch, rows)
        assert _trainable(wrapper.trainer.train_dataset[0], "rejected") > 0
