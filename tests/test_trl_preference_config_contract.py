"""The canary for the blind spot that let a broken `trl` bound ship twice.

Six trainers — bco, dpo, ipo, kto, orpo, simpo — build a `trl` config inside
`setup()`. `trl` removed `max_prompt_length` from those configs in stages, so
`soup train --task orpo` failed at import for anyone who pip-installed fresh.
Nothing caught it, and the reason is exactly one sentence: **the `trl` imports
and the config construction live inside `setup()`, which no test had ever
called on those wrappers.** Constructing a wrapper does not touch `trl` at all
— `__init__` only stores the config — so a test that instantiates the wrapper
and mocks `setup` proves nothing about whether the package can train.

v0.72.4 closed that for four of the six through the layer-streaming suite
(`tests/test_v07204.py`), which drives the real `setup()`. It left `bco` and
`ipo` open: `tests/test_bco.py` and `tests/test_ipo.py` both patch `.setup`
out. This file closes it for all six on the ORDINARY (non-streaming) path,
which is what users actually run and what raising the cap (#326) has to keep
working.

What it is NOT: a training test. It asserts that the wrapper reaches a live
`trl` trainer with the arguments it means to pass. That is the whole failure
class — a `TypeError` on an unexpected keyword argument, or an `ImportError` on
a config class that moved out of the `trl` namespace — and it is cheap.
"""

import pytest


# --------------------------------------------------------------------------
# fixtures -- a real, tiny checkpoint on disk (mirrors tests/test_v07204.py;
# duplicated deliberately, as the four v0.72.x suites already do, so this file
# stays runnable on its own)
# --------------------------------------------------------------------------
def _tiny_llama_dir(tmp_path, vocab=64, hidden=64):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(directory))


def _pref_rows(n=8):
    return [{"prompt": "hi", "chosen": " good answer", "rejected": " bad"} for _ in range(n)]


_MAX_SEQUENCE_LENGTH = 64
_MAX_PROMPT_LENGTH = _MAX_SEQUENCE_LENGTH // 2
_OVERLONG_PROMPT_WORDS = 161


def _long_pref_rows(n=8):
    prompt = " ".join(["hello"] * _OVERLONG_PROMPT_WORDS)
    return [{"prompt": prompt, "chosen": " good answer", "rejected": " bad"} for _ in range(n)]


def _kto_rows(n=8):
    return [{"prompt": "hi", "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


def _long_kto_rows(n=8):
    prompt = " ".join(["hello"] * _OVERLONG_PROMPT_WORDS)
    return [{"prompt": prompt, "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


#: Every word here is a single token in the tiny tokenizer, so the completion is
#: this many tokens long — comfortably over ``_MAX_SEQUENCE_LENGTH`` on its own.
_LONG_COMPLETION_WORDS = 100
_LONG_COMPLETION = " " + " ".join(["answer"] * _LONG_COMPLETION_WORDS)

#: The label used to mask prompt tokens out of the completion loss.
_IGNORE_LABEL = -100

_FALLBACK_MAX_SEQUENCE_LENGTH = 4
_FALLBACK_MAX_PROMPT_LENGTH = 2
_FALLBACK_PROMPT_TOKENS = [101, 102, 103, 104]
_FALLBACK_COMPLETION_TOKENS = [201, 202, 203]
_FALLBACK_KL_COMPLETION_TOKENS = [301, 302, 303]
_FALLBACK_ATTENTION_MASK = [1]


def _mixed_kto_rows(n=8):
    """Half the rows overflow on the prompt, half on the completion.

    The prompt-heavy rows are the regression: TRL slices the answer to fit
    ``max_length`` before Soup's cap runs, so a wrapper that re-derives the
    completion from TRL's own truncated column keeps only the trailing EOS. The
    completion-heavy rows exercise the ordinary answer-truncation path.
    """
    long_prompt = " ".join(["hello"] * _OVERLONG_PROMPT_WORDS)
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"prompt": long_prompt, "completion": " good answer", "label": True})
        else:
            rows.append({"prompt": "hi", "completion": _LONG_COMPLETION, "label": False})
    return rows


def _mixed_pref_rows(n=8):
    """BCO's preference-shaped analogue of :func:`_mixed_kto_rows`."""
    long_prompt = " ".join(["hello"] * _OVERLONG_PROMPT_WORDS)
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({"prompt": long_prompt, "chosen": " good answer", "rejected": " bad"})
        else:
            rows.append({"prompt": "hi", "chosen": _LONG_COMPLETION, "rejected": " bad"})
    return rows


#: KTO refuses `per_device_train_batch_size == 1` outright — its KL term is
#: degenerate at 1 — so the floor is per task, not global.
_MIN_BATCH = {"kto": 2}

_ROWS = {
    "bco": _pref_rows,
    "dpo": _pref_rows,
    "ipo": _pref_rows,
    "kto": _kto_rows,
    "orpo": _pref_rows,
    "simpo": _pref_rows,
}
_PREPARED_ROW_MULTIPLIER = {"bco": 2}

#: task -> (wrapper import path, the trl config class its `setup()` builds).
#: The config class is named so a failure says *which* trl API moved, rather
#: than only that some import failed.
_WRAPPERS = {
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", "BCOConfig"),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "DPOConfig"),
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", "DPOConfig"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", "KTOConfig"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "ORPOConfig"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", "CPOConfig"),
}

_ALL_SIX = tuple(_WRAPPERS)

#: The two this file exists for. Kept as its own name so the regression guard
#: below can assert they are covered without restating the list.
_PREVIOUSLY_UNCOVERED = ("bco", "ipo")


def _cfg(weights, out_dir, task):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": _MAX_SEQUENCE_LENGTH,
                    "chat_template": "chatml",
                },
                "training": {
                    "batch_size": _MIN_BATCH.get(task, 1),
                    "quantization": "none",
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(out_dir),
            }
        )
    )


def _build(tmp_path, monkeypatch, task):
    module, cls_name, _ = _WRAPPERS[task]
    import importlib

    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(weights, tmp_path / "out", task)
    wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
    wrapper.setup({"train": _ROWS[task](8)})
    return wrapper


class TestEveryPreferenceTrainerReachesALiveTrlTrainer:
    """The load-bearing test. `setup()` is CALLED, not mocked."""

    @pytest.mark.parametrize("task", _ALL_SIX)
    def test_setup_builds_a_trainer(self, tmp_path, monkeypatch, task):
        wrapper = _build(tmp_path, monkeypatch, task)
        assert wrapper.trainer is not None, f"{task}: setup() left no trainer"
        assert wrapper.model is not None and wrapper.tokenizer is not None

    @pytest.mark.parametrize("task", _ALL_SIX)
    def test_the_config_carries_the_arguments_the_wrapper_passes(
        self, tmp_path, monkeypatch, task
    ):
        """`max_prompt_length` is the field that actually broke, so it is named
        here rather than left implicit in "setup() didn't raise". A trl release
        that keeps accepting the keyword but stops storing it would pass the
        test above and fail this one.

        #326 made the assertion conditional, because trl removed the field in
        stages (kto 0.27, bco/orpo/simpo 0.28, dpo/ipo 0.29) and the wrappers
        now pass it only to a config that still accepts it. The contract being
        pinned is therefore two-sided, and BOTH halves have teeth: where the
        field exists it must carry the value the wrapper computed, and where it
        does not it must be genuinely absent — not silently defaulted, which is
        how a wrapper that had quietly stopped passing it would look.
        """
        from soup_cli.trainer._trl_compat import config_accepts

        wrapper = _build(tmp_path, monkeypatch, task)
        args = wrapper.trainer.args
        if config_accepts(type(args), "max_prompt_length"):
            assert args.max_prompt_length == _MAX_PROMPT_LENGTH, (task, args.max_prompt_length)
        else:
            assert not hasattr(args, "max_prompt_length"), (
                f"{task}: installed trl does not accept `max_prompt_length` as a "
                f"keyword, yet the built config has the attribute — the capability "
                f"probe and the object disagree"
            )
        assert args.max_length == _MAX_SEQUENCE_LENGTH, (task, args.max_length)

    @pytest.mark.parametrize(
        ("task", "rows_factory"),
        (
            ("bco", _long_pref_rows),
            ("dpo", _long_pref_rows),
            ("ipo", _long_pref_rows),
            ("kto", _long_kto_rows),
            ("orpo", _long_pref_rows),
            ("simpo", _long_pref_rows),
        ),
    )
    def test_removed_prompt_cap_is_enforced_on_the_effective_batch(
        self, tmp_path, monkeypatch, task, rows_factory
    ):
        """TRL 0.29 must not turn ``data.max_length`` into a cosmetic field.

        The assertion is on the tensors the model receives, not the config or
        an intermediate column: 0.29 accepted ``max_length=64`` while emitting
        over-length batches from this exact 161-token prompt.
        """
        from soup_cli.trainer._trl_compat import config_accepts

        module, cls_name, _ = _WRAPPERS[task]
        import importlib

        weights = _tiny_llama_dir(tmp_path)
        _write_tiny_tokenizer(weights)
        monkeypatch.chdir(tmp_path)
        cfg = _cfg(weights, tmp_path / "out", task)
        wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
        wrapper.setup({"train": rows_factory(8)})

        if config_accepts(type(wrapper.trainer.args), "max_prompt_length"):
            pytest.skip("installed TRL still enforces its own prompt cap")

        expected_rows = 8 * _PREPARED_ROW_MULTIPLIER.get(task, 1)
        assert len(wrapper.trainer.train_dataset) == expected_rows
        batch = next(iter(wrapper.trainer.get_train_dataloader()))
        sequence_keys = [key for key in batch if key.endswith("input_ids")]
        assert sequence_keys, batch.keys()
        assert all(batch[key].shape[-1] <= _MAX_SEQUENCE_LENGTH for key in sequence_keys), {
            key: tuple(batch[key].shape) for key in sequence_keys
        }

    @pytest.mark.parametrize(
        ("task", "rows_factory"),
        (
            ("kto", _mixed_kto_rows),
            ("bco", _mixed_pref_rows),
        ),
    )
    def test_unpaired_completion_content_survives_the_cap(
        self, tmp_path, monkeypatch, task, rows_factory
    ):
        """The cap must keep the *content* of the completion, not just its shape.

        The shape check above passes even when the completion is empty — an
        over-length prompt makes TRL slice the answer away inside
        ``completion_input_ids`` before Soup's cap runs, so a row can be exactly
        ``max_length`` tokens yet train on nothing but a trailing EOS. This
        asserts, per row, that the unmasked completion tokens are the real
        leading answer tokens (``answer_input_ids``), including at least one
        non-EOS token.

        The assertion reads content off ``completion_labels``, never off
        ``answer_input_ids`` itself: that column is TRL scratch the loss ignores
        and Soup does not cap, so a generic ``*input_ids <= max_length`` sweep
        would wrongly flag it on the long-completion rows.
        """
        from soup_cli.trainer._trl_compat import config_accepts

        module, cls_name, _ = _WRAPPERS[task]
        import importlib

        from transformers import AutoTokenizer

        weights = _tiny_llama_dir(tmp_path)
        _write_tiny_tokenizer(weights)
        monkeypatch.chdir(tmp_path)
        eos_id = AutoTokenizer.from_pretrained(weights).eos_token_id
        cfg = _cfg(weights, tmp_path / "out", task)
        wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
        wrapper.setup({"train": rows_factory(8)})

        if config_accepts(type(wrapper.trainer.args), "max_prompt_length"):
            pytest.skip("installed TRL still enforces its own prompt cap")

        dataset = wrapper.trainer.train_dataset
        assert "answer_input_ids" in dataset.column_names, dataset.column_names
        for index, row in enumerate(dataset):
            completion_ids = list(row["completion_input_ids"])
            labels = list(row["completion_labels"])
            assert len(completion_ids) <= _MAX_SEQUENCE_LENGTH, (task, index, len(completion_ids))
            unmasked = [tok for tok, lab in zip(completion_ids, labels) if lab != _IGNORE_LABEL]
            non_eos = [tok for tok in unmasked if tok != eos_id]
            assert non_eos, (
                f"{task} row {index}: the capped completion is EOS-only — the answer "
                f"was truncated away before the cap, so the row trains on nothing"
            )
            answer = list(row["answer_input_ids"])
            assert non_eos == answer[: len(non_eos)], (task, index, non_eos, answer)


class TestTheCanaryCoversWhatItClaims:
    """Guards the *coverage* property, not the code. The bug shipped because a
    blind spot was invisible; a shrinking parametrize list would make it
    invisible again."""

    def test_the_two_trainers_that_had_no_setup_test_are_covered(self):
        for task in _PREVIOUSLY_UNCOVERED:
            assert task in _ALL_SIX, task

    def test_every_wrapper_that_passes_max_prompt_length_is_listed(self):
        """Derived from the source, so a seventh trainer adopting the same trl
        argument joins this file automatically instead of silently reopening
        the gap.

        #326 moved the marker. The canary fired exactly as designed when the
        wrappers migrated — it asserted "did the path move?" and the path had
        moved: the literal `max_prompt_length=` was replaced by a call to
        `prompt_length_kwargs(...)`, which passes the keyword only to a trl that
        still accepts it. Both spellings are searched, so a trainer using either
        the old direct keyword or the new helper is covered.
        """
        import pathlib

        trainer_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
        markers = ("max_prompt_length=", "prompt_length_kwargs(")
        users = {
            path.stem
            for path in trainer_dir.glob("*.py")
            if any(marker in path.read_text(encoding="utf-8") for marker in markers)
            and path.stem != "_trl_compat"  # the helper defines it, does not pass it
        }
        assert users, "found no trainer passing a prompt-length cap — did the path move?"
        uncovered = sorted(users - set(_ALL_SIX))
        assert not uncovered, f"passes a prompt-length cap with no setup() test: {uncovered}"


class TestTheTrlBoundsAreConsistentWithTheCode:
    """The installed TRL must provide every symbol through Soup's real resolver."""

    def test_the_installed_trl_provides_every_symbol_the_trainers_import(self):
        """Exercise the same public-first, experimental-second imports as setup()."""
        from soup_cli.trainer._trl_compat import resolve_trl_symbol

        broken = {}
        symbols = {
            "DPOConfig": None,
            "DPOTrainer": None,
            "KTOConfig": None,
            "KTOTrainer": None,
            "ORPOConfig": "trl.experimental.orpo",
            "ORPOTrainer": "trl.experimental.orpo",
            "CPOConfig": "trl.experimental.cpo",
            "CPOTrainer": "trl.experimental.cpo",
            "BCOConfig": "trl.experimental.bco",
            "BCOTrainer": "trl.experimental.bco",
            "GRPOTrainer": None,
            "GRPOConfig": None,
        }
        for name, experimental_module in symbols.items():
            try:
                resolve_trl_symbol(name, experimental_module)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                broken[name] = f"{type(exc).__name__}: {exc}"
        assert not broken, (
            f"installed trl cannot supply {sorted(broken)} through Soup's "
            f"public/experimental resolver — the [train] bounds and the code "
            f"disagree: {broken}"
        )


class TestKtoTruncatesThePromptFromTheEnd:
    """KTOConfig exposes no ``truncation_mode`` field, so KTO caps its prompt
    with ``DEFAULT_PROMPT_TRUNCATION_MODE``. That must be ``keep_end`` — TRL's
    own historical direction for KTO/BCO/CPO/ORPO — because a chat template puts
    the generation header (the assistant turn the completion continues) at the
    END of the prompt; ``keep_start`` would cut it off.
    """

    def test_the_kto_cap_keeps_the_prompt_tail_not_its_head(self):
        from datasets import Dataset

        from soup_cli.trainer._trl_compat import (
            DEFAULT_PROMPT_TRUNCATION_MODE,
            enforce_preference_sequence_limit,
        )

        assert DEFAULT_PROMPT_TRUNCATION_MODE == "keep_end"

        eos_id = 999
        prompt = list(range(100, 116))  # 16 distinct, order-revealing prompt tokens
        answer = [200, 201]
        max_prompt_length = 4
        max_length = 8
        dataset = Dataset.from_dict(
            {
                "prompt_input_ids": [prompt],
                "prompt_attention_mask": [[1] * len(prompt)],
                "answer_input_ids": [answer],
                "answer_attention_mask": [[1] * len(answer)],
                "completion_input_ids": [prompt + answer + [eos_id]],
                "completion_attention_mask": [[1] * (len(prompt) + len(answer) + 1)],
                "completion_labels": [[_IGNORE_LABEL] * len(prompt) + answer + [eos_id]],
            }
        )

        capped = enforce_preference_sequence_limit(
            dataset,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            truncation_mode=DEFAULT_PROMPT_TRUNCATION_MODE,
        )[0]

        tail = prompt[-max_prompt_length:]
        head = prompt[:max_prompt_length]
        assert list(capped["prompt_input_ids"]) == tail
        assert list(capped["prompt_input_ids"]) != head
        # the completion's masked prompt span must keep the same tail, so the
        # generation header survives on the sequence the model is trained on
        masked = [
            tok
            for tok, label in zip(capped["completion_input_ids"], capped["completion_labels"])
            if label == _IGNORE_LABEL
        ]
        assert masked == tail


class TestUnpairedKlFallbackWithoutAnswerColumns:
    def test_kl_completion_caps_without_answer_columns(self):
        from datasets import Dataset

        from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

        prompt = _FALLBACK_PROMPT_TOKENS
        completion = _FALLBACK_COMPLETION_TOKENS
        kl_completion = _FALLBACK_KL_COMPLETION_TOKENS
        prompt_mask = _FALLBACK_ATTENTION_MASK * len(prompt)
        completion_mask = _FALLBACK_ATTENTION_MASK * len(completion)
        kl_completion_mask = _FALLBACK_ATTENTION_MASK * len(kl_completion)
        dataset = Dataset.from_dict(
            {
                "prompt_input_ids": [prompt],
                "prompt_attention_mask": [prompt_mask],
                "completion_input_ids": [prompt + completion],
                "completion_attention_mask": [prompt_mask + completion_mask],
                "completion_labels": [[_IGNORE_LABEL] * len(prompt) + completion],
                "KL_prompt_input_ids": [prompt],
                "KL_prompt_attention_mask": [prompt_mask],
                "KL_completion_input_ids": [prompt + kl_completion],
                "KL_completion_attention_mask": [prompt_mask + kl_completion_mask],
                "KL_completion_labels": [[_IGNORE_LABEL] * len(prompt) + kl_completion],
            }
        )

        capped = enforce_preference_sequence_limit(
            dataset,
            max_length=_FALLBACK_MAX_SEQUENCE_LENGTH,
            max_prompt_length=_FALLBACK_MAX_PROMPT_LENGTH,
            truncation_mode="keep_end",
        )[0]

        expected_prompt = prompt[-_FALLBACK_MAX_PROMPT_LENGTH:]
        expected_kl_completion = kl_completion[
            : _FALLBACK_MAX_SEQUENCE_LENGTH - len(expected_prompt)
        ]
        assert list(capped["KL_completion_input_ids"]) == (
            expected_prompt + expected_kl_completion
        )
        assert list(capped["KL_completion_labels"]) == (
            [_IGNORE_LABEL] * len(expected_prompt) + expected_kl_completion
        )


#: 41 words: longer than max_length // 2, yet prompt + completion is 45 of 64 tokens.
_FITTING_PROMPT = " ".join(["hi"] * 40 + ["world"])
#: An over-long prompt whose end is visible, so the kept side of the cut can be asserted.
_OVERLONG_PROMPT_WITH_TAIL = " ".join(["hello"] * (_OVERLONG_PROMPT_WORDS - 1) + ["world"])


def _setup_rows(tmp_path, monkeypatch, task, rows, val_rows=None):
    import importlib

    module, cls_name, _ = _WRAPPERS[task]
    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(weights, tmp_path / "out", task)
    wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
    dataset = {"train": rows}
    if val_rows is not None:
        dataset["val"] = val_rows
    wrapper.setup(dataset)
    return wrapper


def _rows_with(task, prompt, completion, n=8):
    if task == "kto":
        return [
            {"prompt": prompt, "completion": completion, "label": i % 2 == 0}
            for i in range(n)
        ]
    return [{"prompt": prompt, "chosen": completion, "rejected": " bad"} for _ in range(n)]


class TestTheCapTouchesOnlyRowsThatOverflow:
    @pytest.mark.parametrize("task", ("bco", "ipo", "kto", "simpo"))
    def test_a_row_that_fits_reaches_the_trainer_uncut(self, tmp_path, monkeypatch, task):
        rows = _rows_with(task, _FITTING_PROMPT, " good answer")
        wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
        row = wrapper.trainer.train_dataset[0]
        prompt = list(row["prompt_ids"] if "prompt_ids" in row else row["prompt_input_ids"])
        words = wrapper.tokenizer.convert_ids_to_tokens(prompt)
        assert len(prompt) > _MAX_PROMPT_LENGTH, (task, len(prompt), words[:2], words[-2:])
        assert words[-1] == "world", (task, words[-3:])

    @pytest.mark.parametrize("task", ("bco", "ipo", "kto", "simpo"))
    def test_a_validation_row_that_fits_reaches_the_trainer_uncut(
        self, tmp_path, monkeypatch, task
    ):
        rows = _rows_with(task, "hi", " good answer")
        val_rows = _rows_with(task, _FITTING_PROMPT, " good answer")
        wrapper = _setup_rows(tmp_path, monkeypatch, task, rows, val_rows=val_rows)
        row = wrapper.trainer.eval_dataset[0]
        prompt = list(row["prompt_ids"] if "prompt_ids" in row else row["prompt_input_ids"])
        words = wrapper.tokenizer.convert_ids_to_tokens(prompt)
        assert len(prompt) > _MAX_PROMPT_LENGTH, (task, len(prompt), words[:2], words[-2:])
        assert words[-1] == "world", (task, words[-3:])

    @pytest.mark.parametrize("task", ("bco", "kto"))
    def test_a_row_trl_already_fit_keeps_its_eos(self, tmp_path, monkeypatch, task):
        rows = _rows_with(task, "hi", _LONG_COMPLETION)
        wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
        eos_id = wrapper.tokenizer.eos_token_id
        for index, row in enumerate(wrapper.trainer.train_dataset):
            if task == "bco" and len(row["answer_input_ids"]) < 10:
                continue  # BCO's rejected half (" bad") is not the long row
            ids = list(row["completion_input_ids"])
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (task, index, len(ids))
            assert ids[-1] == eos_id, (task, index, "EOS dropped from a row trl already fit")

    @pytest.mark.parametrize("task", ("bco", "kto"))
    def test_an_overlong_prompt_keeps_its_end_and_the_completion_its_eos(
        self, tmp_path, monkeypatch, task
    ):
        rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good answer")
        wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
        tok = wrapper.tokenizer
        for index, row in enumerate(wrapper.trainer.train_dataset):
            ids = list(row["completion_input_ids"])
            labels = list(row["completion_labels"])
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (task, index, len(ids))
            masked = [tok for tok, label in zip(ids, labels) if label == _IGNORE_LABEL]
            assert tok.convert_ids_to_tokens(masked[-1:]) == ["world"], (task, index)
            assert ids[-1] == tok.eos_token_id, (task, index, "the completion lost its EOS")

    def test_the_kto_kl_completion_keeps_its_answer(self, tmp_path, monkeypatch):
        wrapper = _setup_rows(tmp_path, monkeypatch, "kto", _long_kto_rows(8))
        eos_id = wrapper.tokenizer.eos_token_id
        for index, row in enumerate(wrapper.trainer.train_dataset):
            ids = list(row["KL_completion_input_ids"])
            labels = list(row["KL_completion_labels"])
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (index, len(ids))
            non_eos = [
                token
                for token, label in zip(ids, labels)
                if label != _IGNORE_LABEL and token != eos_id
            ]
            assert non_eos, f"kto row {index}: the KL completion is EOS-only"

    def test_the_kto_kl_rebuild_matches_trl_on_rows_that_fit(self, tmp_path, monkeypatch):
        """The KL reconstruction reproduces TRL's own rotated KL rows exactly.

        Rows that fit reach the trainer untouched, so they are TRL's own output;
        rebuilding the KL completion from answer_input_ids with the same
        per-device-batch rotation must give the identical ids, mask and labels.
        """
        from soup_cli.trainer._trl_compat import (
            _rebuild_unpaired_sequence,
            _rotated_answer_index,
        )

        completions = [" good answer", " fine", " a much longer answer with words", " no"]
        rows = [
            {"prompt": "hi there", "completion": completions[index % 4], "label": index % 2 == 0}
            for index in range(9)
        ]
        wrapper = _setup_rows(tmp_path, monkeypatch, "kto", rows)
        dataset = wrapper.trainer.train_dataset
        answers = list(dataset["answer_input_ids"])
        masks = list(dataset["answer_attention_mask"])
        chunk = wrapper.trainer.args.per_device_train_batch_size
        eos_id = wrapper.tokenizer.eos_token_id
        for index, row in enumerate(dataset):
            lender = _rotated_answer_index(index, len(answers), chunk)
            rebuilt = _rebuild_unpaired_sequence(
                row["KL_prompt_input_ids"],
                row["KL_prompt_attention_mask"],
                answers[lender],
                masks[lender],
                eos_id,
                max_length=_MAX_SEQUENCE_LENGTH,
                max_prompt_length=_MAX_SEQUENCE_LENGTH,
                truncation_mode="keep_end",
            )
            assert rebuilt["input_ids"] == list(row["KL_completion_input_ids"]), index
            assert rebuilt["labels"] == list(row["KL_completion_labels"]), index
            assert rebuilt["attention_mask"] == list(row["KL_completion_attention_mask"]), index

    def test_the_kto_kl_completion_of_an_overlong_prompt_borrows_the_rotated_answer(
        self, tmp_path, monkeypatch
    ):
        """An over-long row's KL completion is its prompt plus the neighbour's answer.

        TRL pairs each row with another row's answer inside every
        per_device_train_batch_size chunk; the rebuild has to follow the same
        rotation, or the KL estimate compares the wrong text.
        """
        from soup_cli.trainer._trl_compat import _rotated_answer_index

        completions = [" alpha beta", " gamma", " delta epsilon zeta", " eta"]
        rows = [
            {
                "prompt": _OVERLONG_PROMPT_WITH_TAIL,
                "completion": completions[index % 4],
                "label": index % 2 == 0,
            }
            for index in range(8)
        ]
        wrapper = _setup_rows(tmp_path, monkeypatch, "kto", rows)
        eos_id = wrapper.tokenizer.eos_token_id
        dataset = wrapper.trainer.train_dataset
        answers = list(dataset["answer_input_ids"])
        chunk = wrapper.trainer.args.per_device_train_batch_size
        for index, row in enumerate(dataset):
            lender = _rotated_answer_index(index, len(answers), chunk)
            ids = list(row["KL_completion_input_ids"])
            labels = list(row["KL_completion_labels"])
            trained = [token for token, label in zip(ids, labels) if label != _IGNORE_LABEL]
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (index, len(ids))
            assert trained == list(answers[lender]) + [eos_id], index
