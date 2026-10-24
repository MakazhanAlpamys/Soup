"""#2112 — the transformers SimPO path caps the prompt, as DPO and ORPO do.

TRL 0.29's ``CPOConfig`` has no ``max_prompt_length``, and its CPO
``tokenize_row`` cuts only the answers (``answer[: max_length -
longer_response_length]``), never the prompt. A prompt longer than
``data.max_length`` therefore reached the model whole. ``simpo.py`` now runs
``enforce_preference_sequence_limit`` on TRL's prepared rows, with the same
``max_length // 2`` prompt budget and ``keep_end`` mode ``orpo.py`` uses, before
the #1208 empty-completion refusal.

"Before" in these tests is TRL's own ``tokenize_row`` on the same raw row, so
the comparison is against exactly what the trainer would have kept without the
cap, not a re-derivation of it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

MAX_LENGTH = 64
PROMPT_BUDGET = MAX_LENGTH // 2

WORDS = [
    "<pad>", "<s>", "</s>", "<unk>", "what", "is", "one", "plus", "?",
    "two", "three", "hello", "there", "hi", "no", "bad", "world",
]

SEQUENCE_KEYS = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")


def _tiny_model_dir(tmp_path):
    """A random-init 2-layer Llama plus a WordLevel tokenizer, all offline."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    raw = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"),
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", unk_token="<unk>",
    )
    torch.manual_seed(0)
    path = tmp_path / "tiny"
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(WORDS), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, max_position_embeddings=512,
    )).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _setup_simpo(tmp_path, monkeypatch, train_rows, val_rows=None):
    """A real ``SimPOTrainerWrapper.setup()`` on the transformers backend."""
    from soup_cli.trainer._trl_compat import config_accepts, resolve_trl_symbol

    try:
        resolve_trl_symbol("CPOConfig", "trl.experimental.cpo")
    except Exception as exc:  # noqa: BLE001 — a broken trl is not a failure
        pytest.skip(f"this environment cannot import a trl CPO config: {exc}")

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_model_dir(tmp_path)
    cfg = load_config_from_string(
        f"base: {base.as_posix()}\ntask: simpo\nbackend: transformers\n"
        f"data: {{train: ./unused.jsonl, format: dpo, max_length: {MAX_LENGTH}}}\n"
        "training:\n  epochs: 1\n  batch_size: 2\n  quantization: none\n"
        "  lora: {r: 4, alpha: 8, dropout: 0.0}\noutput: ./out\n"
    )
    wrapper = SimPOTrainerWrapper(cfg, device="cpu")
    dataset = {"train": train_rows}
    if val_rows is not None:
        dataset["val"] = val_rows
    wrapper.setup(dataset)
    if config_accepts(type(wrapper.trainer.args), "max_prompt_length"):
        pytest.skip("installed TRL still enforces its own prompt cap")
    return wrapper


def _row(prompt_words: int, chosen_words: int, rejected_words: int) -> dict:
    return {
        "prompt": " ".join(["what"] * prompt_words),
        "chosen": " ".join(["hello"] * chosen_words),
        "rejected": " ".join(["bad"] * rejected_words),
    }


#: Prompt alone is 161 tokens against data.max_length=64; both answers are short.
LONG = _row(161, 2, 1)
#: Fits whole: 3 + 3 + EOS, nothing to cut.
FITS = _row(3, 2, 1)


def _trainable(row: dict, side: str) -> list[int]:
    return [label for label in row[f"{side}_labels"] if label != -100]


def _before(wrapper, raw: dict) -> dict:
    """What TRL's own CPO tokenizer keeps for ``raw``, with no cap applied."""
    return wrapper.trainer.tokenize_row(raw)


class TestAPromptLongerThanMaxLength:
    def test_every_sequence_fits_and_the_prompt_keeps_its_end(self, tmp_path, monkeypatch):
        wrapper = _setup_simpo(tmp_path, monkeypatch, [LONG] * 4, val_rows=[LONG] * 2)
        before = _before(wrapper, LONG)
        assert len(before["prompt_input_ids"]) > MAX_LENGTH, "fixture no longer over-long"

        for split in ("train", "eval"):
            dataset = getattr(wrapper.trainer, f"{split}_dataset")
            for row in dataset:
                lengths = {key: len(row[key]) for key in SEQUENCE_KEYS}
                assert all(n <= MAX_LENGTH for n in lengths.values()), (split, lengths)

                kept = before["prompt_input_ids"][-PROMPT_BUDGET:]
                assert row["prompt_input_ids"] == kept, split
                assert row["prompt_attention_mask"] == [1] * PROMPT_BUDGET, split
                for side in ("chosen", "rejected"):
                    assert row[f"{side}_input_ids"][:PROMPT_BUDGET] == kept, (split, side)
                    assert row[f"{side}_labels"][:PROMPT_BUDGET] == [-100] * PROMPT_BUDGET
                    assert len(row[f"{side}_attention_mask"]) == len(row[f"{side}_input_ids"])
                    # Both answers survive whole: they fit the room the cap leaves.
                    assert _trainable(row, side) == _trainable(before, side), (split, side)
                    assert _trainable(row, side), (split, side)

    def test_the_batch_the_model_receives_obeys_max_length(self, tmp_path, monkeypatch):
        wrapper = _setup_simpo(tmp_path, monkeypatch, [LONG] * 4)
        batch = next(iter(wrapper.trainer.get_train_dataloader()))
        keys = [key for key in batch if key.endswith("input_ids")]
        assert keys, batch.keys()
        assert all(batch[key].shape[-1] <= MAX_LENGTH for key in keys), {
            key: tuple(batch[key].shape) for key in keys
        }


class TestARowThatFits:
    def test_is_identical_before_and_after(self, tmp_path, monkeypatch):
        wrapper = _setup_simpo(tmp_path, monkeypatch, [FITS, LONG, FITS, LONG])
        before = _before(wrapper, FITS)
        for index in (0, 2):
            after = wrapper.trainer.train_dataset[index]
            for key, value in before.items():
                assert after[key] == value, (index, key)


def _prompt_words_for(tmp_path, monkeypatch, n_tokens: int) -> int:
    """How many prompt words TRL's CPO tokenizer turns into ``n_tokens`` prompt ids.

    Asked of TRL itself (it may add a BOS), on a probe trainer, so the boundary
    rows below hold exactly the token count they claim.
    """
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = _setup_simpo(probe_dir, monkeypatch, [FITS] * 4)
    for words in range(1, n_tokens + 1):
        if len(_before(probe, _row(words, 2, 1))["prompt_input_ids"]) == n_tokens:
            return words
    raise AssertionError(f"no word count gives a {n_tokens}-token prompt")


class TestThePromptBudgetBoundary:
    """The cap cuts by the prompt alone: any prompt over ``max_length // 2``
    tokens loses its start, even when prompt plus answers fit in
    ``max_length``; one of exactly ``max_length // 2`` tokens is untouched."""

    def test_a_prompt_of_exactly_the_budget_is_unchanged(self, tmp_path, monkeypatch):
        raw = _row(_prompt_words_for(tmp_path, monkeypatch, PROMPT_BUDGET), 2, 1)
        wrapper = _setup_simpo(tmp_path, monkeypatch, [raw] * 4)
        before = _before(wrapper, raw)
        assert len(before["prompt_input_ids"]) == PROMPT_BUDGET
        after = wrapper.trainer.train_dataset[0]
        for key, value in before.items():
            assert after[key] == value, key

    def test_one_token_over_the_budget_is_cut_although_the_row_fits(
        self, tmp_path, monkeypatch
    ):
        raw = _row(_prompt_words_for(tmp_path, monkeypatch, PROMPT_BUDGET + 1), 2, 1)
        wrapper = _setup_simpo(tmp_path, monkeypatch, [raw] * 4)
        before = _before(wrapper, raw)
        assert len(before["prompt_input_ids"]) == PROMPT_BUDGET + 1
        # The whole row already fits: the cut comes from the prompt budget alone.
        assert all(len(before[key]) <= MAX_LENGTH for key in SEQUENCE_KEYS), before

        after = wrapper.trainer.train_dataset[0]
        kept = before["prompt_input_ids"][-PROMPT_BUDGET:]
        assert after["prompt_input_ids"] == kept
        assert after["prompt_attention_mask"] == [1] * PROMPT_BUDGET
        for side in ("chosen", "rejected"):
            assert after[f"{side}_input_ids"][:PROMPT_BUDGET] == kept, side
            assert after[f"{side}_labels"][:PROMPT_BUDGET] == [-100] * PROMPT_BUDGET, side
            assert len(after[f"{side}_input_ids"]) == len(before[f"{side}_input_ids"]) - 1
            assert _trainable(after, side) == _trainable(before, side), side


class TestTheCapAndThe1208Refusal:
    """The cap runs before #1208's guard, so the guard sees capped rows.

    TRL's answer cut is ``max_length - longer_response_length``: it does not
    depend on the prompt's length, so no row is emptied by its prompt and the
    cap cannot refill an answer TRL already emptied. What the cap guarantees is
    that it never empties one itself — it leaves ``max_length - max_length // 2``
    tokens of room — and a pair TRL empties is still refused.
    """

    def test_a_long_prompt_with_long_answers_trains_with_both_answers(
        self, tmp_path, monkeypatch
    ):
        raw = _row(200, 40, 20)
        wrapper = _setup_simpo(tmp_path, monkeypatch, [raw] * 4)
        before = _before(wrapper, raw)
        row = wrapper.trainer.train_dataset[0]
        room = MAX_LENGTH - PROMPT_BUDGET
        for side in ("chosen", "rejected"):
            assert _trainable(row, side) == _trainable(before, side)[:room], side
            assert _trainable(row, side), side
            assert len(row[f"{side}_input_ids"]) <= MAX_LENGTH, side

    def test_a_pair_trl_empties_is_still_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="zero completion tokens"):
            _setup_simpo(tmp_path, monkeypatch, [_row(161, 80, 1)] * 4)


class TestTheHelperReadsCposLayout:
    def test_cpo_rows_take_the_orpo_branch(self, tmp_path, monkeypatch):
        """CPO prepares ORPO's columns, not DPO's, so the helper's ORPO branch
        is the one that runs; pin that rather than assume it."""
        from datasets import Dataset

        from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

        wrapper = _setup_simpo(tmp_path, monkeypatch, [FITS] * 4)
        columns = set(wrapper.trainer.train_dataset.column_names)
        assert {
            "prompt_input_ids", "prompt_attention_mask",
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        } <= columns, sorted(columns)
        assert not {"prompt_ids", "chosen_ids", "rejected_ids"} <= columns, sorted(columns)

        raw = Dataset.from_list([_before(wrapper, LONG)])
        capped = enforce_preference_sequence_limit(
            raw, max_length=MAX_LENGTH, max_prompt_length=PROMPT_BUDGET,
            truncation_mode=wrapper.trainer.args.truncation_mode,
        )[0]
        assert wrapper.trainer.args.truncation_mode == "keep_end"
        assert len(capped["prompt_input_ids"]) == PROMPT_BUDGET
        assert all(len(capped[key]) <= MAX_LENGTH for key in SEQUENCE_KEYS)


class TestTheCapIsWiredIntoSetup:
    """Runs without trl: removing the calls fails here even where CPO cannot
    import, and the order against #1208's guard is pinned."""

    def test_setup_caps_both_splits_before_the_1208_guard(self):
        import inspect

        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        source = inspect.getsource(SimPOTrainerWrapper.setup)
        train_cap = "self.trainer.train_dataset = enforce_preference_sequence_limit("
        eval_cap = "self.trainer.eval_dataset = enforce_preference_sequence_limit("
        assert train_cap in source, "SimPO setup no longer caps the train split (#2112)"
        assert eval_cap in source, "SimPO setup no longer caps the eval split (#2112)"
        assert '"max_prompt_length": cfg.data.max_length // 2' in source
        assert '"truncation_mode": cpo_config.truncation_mode' in source
        guard = '_refuse_empty_completion_rows(self.trainer, split="train")'
        assert source.index(eval_cap) < source.index(guard), (
            "the #1208 guard must see the capped rows"
        )
