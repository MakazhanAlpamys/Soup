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
# At batch 2 both KL rotation directions coincide; 4 tells them apart.
_KL_ROTATION_BATCH = 4

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
        over-length batches from this exact 161-token prompt (164/165-token
        DPO/ORPO batches, and SimPO did the same until #2112 gave it the cap).
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
            # The native MLX loops build no trl config: their `max_prompt_length=`
            # goes to their own tokenizer (tests/test_mlx_dpo.py and
            # tests/test_mlx_orpo.py pin the cap).
            and path.stem not in ("mlx_dpo", "mlx_orpo")
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


class TestUnpairedRebuildPreservesAnExistingTerminalEos:
    """An already-terminal EOS must survive the rebuild, not be doubled.

    When the prompt overflows, ``cap_unpaired`` rebuilds the KTO/BCO completion
    (and KTO's KL completion) from the untruncated ``answer_input_ids``. TRL
    stores the answer *with* its stop token — a valid short completion is
    ``[good, </s>]`` — and the original TRL path appends EOS only when it is
    absent. Appending unconditionally trains the matched and KL targets on
    ``[good, </s>, </s>]``, changing the target for inputs that already carry
    their stop token. The oracle is the exact answer the row was built with.
    """

    def test_a_terminal_eos_is_kept_single_on_matched_and_kl_targets(self):
        from datasets import Dataset

        from soup_cli.trainer._trl_compat import (
            DEFAULT_PROMPT_TRUNCATION_MODE,
            enforce_preference_sequence_limit,
        )

        eos_id = 999
        prompt = list(range(100, 116))  # 16 tokens, overflows the prompt cap
        answer = [200, eos_id]  # a short, valid completion that already ends in EOS
        max_prompt_length = 4
        max_length = 8
        # KTO batches run the KL rotation; pin a chunk so the branch is exercised.
        kl_chunk_size = 2
        dataset = Dataset.from_dict(
            {
                "prompt_input_ids": [prompt],
                "prompt_attention_mask": [[1] * len(prompt)],
                "answer_input_ids": [answer],
                "answer_attention_mask": [[1] * len(answer)],
                "completion_input_ids": [prompt + answer],
                "completion_attention_mask": [[1] * (len(prompt) + len(answer))],
                "completion_labels": [[_IGNORE_LABEL] * len(prompt) + answer],
                "KL_prompt_input_ids": [prompt],
                "KL_prompt_attention_mask": [[1] * len(prompt)],
                "KL_completion_input_ids": [prompt + answer],
                "KL_completion_attention_mask": [[1] * (len(prompt) + len(answer))],
                "KL_completion_labels": [[_IGNORE_LABEL] * len(prompt) + answer],
            }
        )

        capped = enforce_preference_sequence_limit(
            dataset,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            truncation_mode=DEFAULT_PROMPT_TRUNCATION_MODE,
            # The production call from bco/ipo/kto/simpo caps only overflowing
            # rows; this row overflows on the prompt, so the rebuild still runs.
            only_when_overflow=True,
            kl_chunk_size=kl_chunk_size,
        )[0]

        for prefix in ("completion", "KL_completion"):
            ids = list(capped[f"{prefix}_input_ids"])
            labels = list(capped[f"{prefix}_labels"])
            unmasked = [tok for tok, label in zip(ids, labels) if label != _IGNORE_LABEL]
            assert unmasked == answer, (prefix, unmasked, answer)
            assert unmasked.count(eos_id) == 1, (prefix, "EOS was doubled", unmasked)
            assert len(ids) <= max_length, (prefix, len(ids))


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
        from soup_cli.trainer._trl_compat import _rebuild_unpaired_sequence

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
            lender = _trl_kl_lender(index, len(answers), chunk)
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
            lender = _trl_kl_lender(index, len(answers), chunk)
            ids = list(row["KL_completion_input_ids"])
            labels = list(row["KL_completion_labels"])
            trained = [token for token, label in zip(ids, labels) if label != _IGNORE_LABEL]
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (index, len(ids))
            assert trained == list(answers[lender]) + [eos_id], index


def _trl_kl_lender(index, size, chunk):
    """trl's ``_get_kl_dataset`` rotates ``[a[-1]] + a[:-1]`` inside each chunk.

    So row ``i`` borrows row ``i - 1``'s answer, and the first row of a chunk
    borrows the last one's. Written out here, not imported, so the test does
    not share an oracle with the code under test.
    """
    start = (index // chunk) * chunk
    end = min(start + chunk, size)
    return index - 1 if index > start else end - 1


@pytest.mark.parametrize(
    "prompt", ("hi there", _OVERLONG_PROMPT_WITH_TAIL), ids=("fits", "overlong")
)
def test_kto_kl_rows_borrow_the_predecessor_answer_at_batch_four(tmp_path, monkeypatch, prompt):
    monkeypatch.setitem(_MIN_BATCH, "kto", _KL_ROTATION_BATCH)
    # Distinct ids in the fixture vocabulary (unknown words all become <unk>).
    completions = [" good", " answer answer", " bad bad bad", " good bad good bad"]
    rows = [
        {"prompt": prompt, "completion": completions[i % len(completions)], "label": i % 2 == 0}
        for i in range(8)
    ]
    wrapper = _setup_rows(tmp_path, monkeypatch, "kto", rows)
    assert wrapper.trainer.args.per_device_train_batch_size == _KL_ROTATION_BATCH
    dataset = wrapper.trainer.train_dataset
    answers = list(dataset["answer_input_ids"])
    eos_id = wrapper.tokenizer.eos_token_id
    for index, row in enumerate(dataset):
        ids = list(row["KL_completion_input_ids"])
        labels = list(row["KL_completion_labels"])
        trained = [tok for tok, lab in zip(ids, labels) if lab != _IGNORE_LABEL]
        lender = _trl_kl_lender(index, len(answers), _KL_ROTATION_BATCH)
        assert trained == list(answers[lender]) + [eos_id], index


@pytest.mark.parametrize("task", ("bco", "kto"))
def test_an_overlong_prompt_with_a_long_completion_fits_and_ends_in_eos(
    tmp_path, monkeypatch, task
):
    rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, _LONG_COMPLETION)
    wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
    eos_id = wrapper.tokenizer.eos_token_id
    prefixes = ["completion"] + (["KL_completion"] if task == "kto" else [])
    for index, row in enumerate(wrapper.trainer.train_dataset):
        for prefix in prefixes:
            ids = list(row[f"{prefix}_input_ids"])
            labels = list(row[f"{prefix}_labels"])
            trained = [tok for tok, lab in zip(ids, labels) if lab != _IGNORE_LABEL]
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (task, index, prefix, len(ids))
            assert ids[-1] == eos_id, (task, index, prefix, "EOS cut off by the cap")
            assert len(trained) > 1, (task, index, prefix, "EOS-only completion")


def _world_id(wrapper):
    return wrapper.tokenizer("world", add_special_tokens=False)["input_ids"][0]


@pytest.mark.parametrize("task", ("bco", "ipo", "kto", "simpo"))
def test_overlong_validation_rows_are_capped_too(tmp_path, monkeypatch, task):
    rows = _rows_with(task, "hi", " good answer")
    val_rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good answer", n=4)
    wrapper = _setup_rows(tmp_path, monkeypatch, task, rows, val_rows=val_rows)
    eval_dataset = wrapper.trainer.eval_dataset
    assert eval_dataset is not None and len(eval_dataset) > 0
    for row in eval_dataset:
        if task == "ipo":
            assert len(row["prompt_ids"]) + len(row["chosen_ids"]) <= _MAX_SEQUENCE_LENGTH
            assert len(row["prompt_ids"]) + len(row["rejected_ids"]) <= _MAX_SEQUENCE_LENGTH
        elif task == "simpo":
            assert len(row["chosen_input_ids"]) <= _MAX_SEQUENCE_LENGTH
            assert len(row["rejected_input_ids"]) <= _MAX_SEQUENCE_LENGTH
        else:
            assert len(row["completion_input_ids"]) <= _MAX_SEQUENCE_LENGTH
            if task == "kto":
                assert len(row["KL_completion_input_ids"]) <= _MAX_SEQUENCE_LENGTH


@pytest.mark.parametrize(("size", "chunk"), ((8, 4), (10, 4), (7, 3), (5, 2), (9, 2), (3, 8)))
def test_the_kl_rotation_stays_inside_its_chunk_and_handles_a_short_last_chunk(size, chunk):
    from soup_cli.trainer._trl_compat import _rotated_answer_index

    for index in range(size):
        assert _rotated_answer_index(index, size, chunk) == _trl_kl_lender(index, size, chunk)


def test_ipo_keeps_the_prompt_head_within_half_the_budget(tmp_path, monkeypatch):
    rows = _rows_with("ipo", _OVERLONG_PROMPT_WITH_TAIL, " good answer")
    wrapper = _setup_rows(tmp_path, monkeypatch, "ipo", rows)
    world = _world_id(wrapper)
    for row in wrapper.trainer.train_dataset:
        assert len(row["prompt_ids"]) == _MAX_PROMPT_LENGTH
        assert world not in list(row["prompt_ids"])  # keep_start drops the tail


def test_simpo_keeps_the_prompt_tail_within_half_the_budget(tmp_path, monkeypatch):
    rows = _rows_with("simpo", _OVERLONG_PROMPT_WITH_TAIL, " good answer")
    wrapper = _setup_rows(tmp_path, monkeypatch, "simpo", rows)
    world = _world_id(wrapper)
    for row in wrapper.trainer.train_dataset:
        assert len(row["prompt_input_ids"]) == _MAX_PROMPT_LENGTH
        assert list(row["prompt_input_ids"])[-1] == world  # keep_end keeps the tail


def test_the_kto_prompt_masks_stay_as_long_as_their_ids(tmp_path, monkeypatch):
    rows = _rows_with("kto", _OVERLONG_PROMPT_WITH_TAIL, " good answer")
    wrapper = _setup_rows(tmp_path, monkeypatch, "kto", rows)
    for row in wrapper.trainer.train_dataset:
        assert len(row["prompt_attention_mask"]) == len(row["prompt_input_ids"])
        assert len(row["KL_prompt_attention_mask"]) == len(row["KL_prompt_input_ids"])


def test_an_unpaired_row_that_exactly_fits_is_not_cut():
    from datasets import Dataset

    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    eos_id = 999
    prompt = list(range(100, 140))  # 40 tokens: longer than max_prompt_length 32
    answer = list(range(200, 223))  # 23 tokens: 40 + 23 + EOS == 64 == max_length
    dataset = Dataset.from_dict(
        {
            "prompt_input_ids": [prompt],
            "prompt_attention_mask": [[1] * len(prompt)],
            "answer_input_ids": [answer],
            "answer_attention_mask": [[1] * len(answer)],
            "completion_input_ids": [prompt + answer + [eos_id]],
            "completion_attention_mask": [[1] * 64],
            "completion_labels": [[_IGNORE_LABEL] * len(prompt) + answer + [eos_id]],
        }
    )
    capped = enforce_preference_sequence_limit(
        dataset,
        max_length=64,
        max_prompt_length=32,
        truncation_mode="keep_end",
        only_when_overflow=True,
    )[0]
    assert list(capped["completion_input_ids"]) == prompt + answer + [eos_id]


def test_a_dpo_layout_row_that_exactly_fits_is_not_cut():
    from datasets import Dataset

    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    prompt = list(range(100, 140))  # 40 tokens: longer than max_prompt_length 32
    chosen = list(range(200, 224))  # 40 + 24 == 64 == max_length
    dataset = Dataset.from_dict(
        {"prompt_ids": [prompt], "chosen_ids": [chosen], "rejected_ids": [list(range(300, 310))]}
    )
    capped = enforce_preference_sequence_limit(
        dataset,
        max_length=64,
        max_prompt_length=32,
        truncation_mode="keep_start",
        only_when_overflow=True,
    )[0]
    assert list(capped["prompt_ids"]) == prompt
    assert list(capped["chosen_ids"]) == chosen


@pytest.mark.parametrize("task", ("bco", "kto"))
@pytest.mark.parametrize("prompt_kind", ("fits", "overlong"))
def test_an_answer_that_already_ends_in_eos_keeps_exactly_one(
    tmp_path, monkeypatch, task, prompt_kind
):
    prompt = "hi" if prompt_kind == "fits" else _OVERLONG_PROMPT_WITH_TAIL
    rows = _rows_with(task, prompt, " good</s>")
    wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
    eos_id = wrapper.tokenizer.eos_token_id
    prefixes = ["completion"] + (["KL_completion"] if task == "kto" else [])
    for row in wrapper.trainer.train_dataset:
        for prefix in prefixes:
            ids = list(row[f"{prefix}_input_ids"])
            labels = list(row[f"{prefix}_labels"])
            trained = [tok for tok, lab in zip(ids, labels) if lab != _IGNORE_LABEL]
            assert trained[-1] == eos_id, (task, prompt_kind, prefix, trained)
            assert trained.count(eos_id) == 1, (task, prompt_kind, prefix, trained)


@pytest.mark.parametrize("task", ("bco", "kto"))
def test_a_prompt_that_consumes_the_budget_still_keeps_a_short_answer(
    tmp_path, monkeypatch, task
):
    # The prompt alone is longer than ``max_length``, so TRL slices the short
    # answer out of the completion. The conditional cap must still rebuild it
    # (the sliced row overflows ``max_length``) rather than treat the row as
    # fitting and skip it, which would train on an answer-less completion.
    rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good")
    wrapper = _setup_rows(tmp_path, monkeypatch, task, rows)
    eos_id = wrapper.tokenizer.eos_token_id
    prefixes = ["completion"] + (["KL_completion"] if task == "kto" else [])
    for row in wrapper.trainer.train_dataset:
        for prefix in prefixes:
            ids = list(row[f"{prefix}_input_ids"])
            labels = list(row[f"{prefix}_labels"])
            trained = [tok for tok, lab in zip(ids, labels) if lab != _IGNORE_LABEL]
            assert len(ids) <= _MAX_SEQUENCE_LENGTH, (task, prefix, len(ids))
            assert trained[-1] == eos_id, (task, prefix, trained)
            assert [t for t in trained if t != eos_id], (task, prefix, "answer discarded", trained)


# --------------------------------------------------------------------------
# #2486 -- the unpaired cap reads the KL lender from its own batch, never a
# whole answer column, so nothing per-row is pickled into the map fingerprint
# --------------------------------------------------------------------------
_UNPAIRED_EOS = 2
_UNPAIRED_BOS = 1
_KL_COLUMNS = (
    "KL_prompt_input_ids",
    "KL_prompt_attention_mask",
    "KL_completion_input_ids",
    "KL_completion_attention_mask",
    "KL_completion_labels",
)
_ANSWER_COLUMNS = ("answer_input_ids", "answer_attention_mask")


def _fitting_kto_layout(rows):
    """The issue's KTO-layout row, repeated: every row fits, so none is cut."""
    from datasets import Dataset

    row = {
        "prompt_input_ids": [1, 5],
        "prompt_attention_mask": [1, 1],
        "completion_input_ids": [1, 5, 6, 2],
        "completion_attention_mask": [1, 1, 1, 1],
        "completion_labels": [_IGNORE_LABEL, _IGNORE_LABEL, 6, 2],
        "answer_input_ids": [6],
        "answer_attention_mask": [1],
        "KL_prompt_input_ids": [1, 5],
        "KL_prompt_attention_mask": [1, 1],
        "KL_completion_input_ids": [1, 5, 6, 2],
        "KL_completion_attention_mask": [1, 1, 1, 1],
        "KL_completion_labels": [_IGNORE_LABEL, _IGNORE_LABEL, 6, 2],
    }
    return Dataset.from_list([row] * rows)


def _trl_like_unpaired_rows(size, seed, *, layout, kl_chunk, max_length):
    """Random unpaired rows shaped the way trl 0.29 prepares them.

    Prompts and answers of random length, so some rows fit, some overflow on
    the answer and some on a prompt longer than ``max_length`` on its own; a
    fifth of the answers already end in EOS.
    ``layout`` is ``kto`` (answer + KL columns), ``bco`` (answer, no KL) or
    ``kl_without_answer`` (KL, no answer columns: the fallback path).
    """
    import random

    from datasets import Dataset

    rng = random.Random(seed)
    prompts, answers = [], []
    for _ in range(size):
        prompt = [rng.randrange(10, 99) for _ in range(rng.randrange(max_length + 16))]
        prompts.append([_UNPAIRED_BOS] + prompt)
        answer = [rng.randrange(100, 199) for _ in range(rng.randrange(60))]
        if rng.random() < 0.2:
            answer.append(_UNPAIRED_EOS)
        answers.append(answer)

    def sequence(prompt, answer):
        body = answer[:-1] if answer[-1:] == [_UNPAIRED_EOS] else answer
        ids = prompt + body + [_UNPAIRED_EOS]
        return ids, [1] * len(ids), [_IGNORE_LABEL] * len(prompt) + ids[len(prompt) :]

    columns = {}

    def put(name, value):
        columns.setdefault(name, []).append(value)

    for index, (prompt, answer) in enumerate(zip(prompts, answers)):
        ids, mask, labels = sequence(prompt, answer)
        put("prompt_input_ids", prompt)
        put("prompt_attention_mask", [1] * len(prompt))
        put("completion_input_ids", ids)
        put("completion_attention_mask", mask)
        put("completion_labels", labels)
        if layout != "kl_without_answer":
            put("answer_input_ids", answer)
            put("answer_attention_mask", [1] * len(answer))
        if layout != "bco":
            lender = _trl_kl_lender(index, size, kl_chunk)
            kl_ids, kl_mask, kl_labels = sequence(prompt, answers[lender])
            put("KL_prompt_input_ids", prompt)
            put("KL_prompt_attention_mask", [1] * len(prompt))
            put("KL_completion_input_ids", kl_ids)
            put("KL_completion_attention_mask", kl_mask)
            put("KL_completion_labels", kl_labels)
    return Dataset.from_dict(columns)


def _reference_unpaired_cap(
    dataset, *, max_length, max_prompt_length, truncation_mode, only_when_overflow, kl_chunk
):
    """The cap written out one row at a time, from ``dataset[i]``.

    It reuses only the row helpers the fix leaves untouched, and picks the KL
    lender with this file's own ``_trl_kl_lender``, so it does not share the
    batching or the lender lookup with the code under test.
    """
    from soup_cli.trainer._trl_compat import (
        _cap_combined_preference_sequence,
        _rebuild_unpaired_sequence,
        _truncate_tokens,
    )

    names = set(dataset.column_names)
    has_answer = set(_ANSWER_COLUMNS) <= names
    has_kl = set(_KL_COLUMNS) <= names
    limits = {
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "truncation_mode": truncation_mode,
    }
    rows = [dataset[index] for index in range(len(dataset))]
    expected = []
    for index, row in enumerate(rows):
        fits = len(row["completion_input_ids"]) <= max_length and (
            not has_kl or len(row["KL_completion_input_ids"]) <= max_length
        )
        if only_when_overflow and fits:
            expected.append(dict(row))
            continue
        out = dict(row)
        for prefix in ("", "KL_") if has_kl else ("",):
            for part in ("input_ids", "attention_mask"):
                out[f"{prefix}prompt_{part}"] = _truncate_tokens(
                    list(row[f"{prefix}prompt_{part}"]), max_prompt_length, truncation_mode
                )
        if has_answer:
            rebuilt = _rebuild_unpaired_sequence(
                row["prompt_input_ids"],
                row["prompt_attention_mask"],
                row["answer_input_ids"],
                row["answer_attention_mask"],
                row["completion_input_ids"][-1],
                **limits,
            )
            source = {f"completion_{key}": value for key, value in rebuilt.items()}
        else:
            source = row
        out.update(_cap_combined_preference_sequence(source, "completion", **limits))
        if has_kl and has_answer:
            lender = rows[_trl_kl_lender(index, len(rows), kl_chunk)]
            rebuilt = _rebuild_unpaired_sequence(
                row["KL_prompt_input_ids"],
                row["KL_prompt_attention_mask"],
                lender["answer_input_ids"],
                lender["answer_attention_mask"],
                row["KL_completion_input_ids"][-1],
                **limits,
            )
            out.update({f"KL_completion_{key}": value for key, value in rebuilt.items()})
        elif has_kl:
            out.update(_cap_combined_preference_sequence(row, "KL_completion", **limits))
        expected.append(out)
    return expected


def test_the_kto_cap_reads_no_whole_column(monkeypatch):
    """The issue's spy: a KTO-layout dataset where no row overflows."""
    from datasets import Dataset

    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    dataset = _fitting_kto_layout(1000)
    column_reads = []
    original = Dataset.__getitem__

    def spy(self, key):
        if isinstance(key, str):
            column_reads.append(key)
        return original(self, key)

    monkeypatch.setattr(Dataset, "__getitem__", spy)
    capped = enforce_preference_sequence_limit(
        dataset,
        max_length=64,
        max_prompt_length=32,
        truncation_mode="keep_end",
        only_when_overflow=True,
        kl_chunk_size=4,
    )
    monkeypatch.undo()
    assert column_reads == [], column_reads
    assert capped["KL_completion_input_ids"] == dataset["KL_completion_input_ids"]


def test_the_function_the_cap_maps_does_not_grow_with_the_dataset(monkeypatch):
    """``datasets`` fingerprints a ``map`` by pickling its function, closure included.

    The function must pickle to the same size for 4 rows and for 4,000, so the
    fingerprint never serialises per-row data.
    """
    from datasets import Dataset
    from datasets.utils._dill import dumps

    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    mapped = []
    original = Dataset.map

    def spy(self, function=None, *args, **kwargs):
        mapped.append(function)
        return original(self, function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", spy)
    sizes = {}
    for rows in (4, 4000):
        mapped.clear()
        enforce_preference_sequence_limit(
            _fitting_kto_layout(rows),
            max_length=64,
            max_prompt_length=32,
            truncation_mode="keep_end",
            only_when_overflow=True,
            kl_chunk_size=4,
        )
        assert len(mapped) == 1, mapped
        sizes[rows] = len(dumps(mapped[0]))
    # A few bytes of slack: the row count itself is one of the closed-over scalars.
    assert abs(sizes[4000] - sizes[4]) <= 16, sizes


@pytest.mark.parametrize("layout", ("kto", "bco", "kl_without_answer"))
@pytest.mark.parametrize("kl_chunk_size", (None, 1, 2, 3, 4, 7, 8))
def test_the_batched_unpaired_cap_matches_the_per_row_cap(layout, kl_chunk_size):
    """Every capped row equals the cap applied one row at a time.

    1,003 rows is not a multiple of any chunk here and spans two ``map``
    batches, so a batch that split a KL chunk would show up; 3 rows is shorter
    than most chunks.
    """
    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    max_length, max_prompt_length = 64, 32
    kl_chunk = kl_chunk_size or 1
    for size in (3, 1003):
        dataset = _trl_like_unpaired_rows(
            size, seed=size, layout=layout, kl_chunk=kl_chunk, max_length=max_length
        )
        for truncation_mode in ("keep_end", "keep_start"):
            for only_when_overflow in (True, False):
                case = (layout, kl_chunk_size, size, truncation_mode, only_when_overflow)
                capped = enforce_preference_sequence_limit(
                    dataset,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                    only_when_overflow=only_when_overflow,
                    kl_chunk_size=kl_chunk_size,
                )
                expected = _reference_unpaired_cap(
                    dataset,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                    only_when_overflow=only_when_overflow,
                    kl_chunk=kl_chunk,
                )
                assert capped.column_names == dataset.column_names, case
                assert capped.features == dataset.features, case
                for name in dataset.column_names:
                    assert capped[name] == [row[name] for row in expected], (case, name)


@pytest.mark.parametrize(
    ("kl_chunk", "batch_rows"),
    ((1, 1000), (2, 1000), (3, 999), (4, 1000), (7, 994), (8, 1000), (1000, 1000), (1500, 1500)),
)
def test_the_unpaired_cap_batch_is_whole_kl_chunks(kl_chunk, batch_rows):
    """1,000 rows rounded down to a multiple of the KL chunk, never below one chunk."""
    from soup_cli.trainer._trl_compat import _unpaired_cap_batch_size

    assert _unpaired_cap_batch_size(kl_chunk) == batch_rows


def test_a_kl_lender_outside_its_batch_raises(monkeypatch):
    """A batch that splits a KL chunk must fail loudly, not borrow the wrong answer."""
    from soup_cli.trainer import _trl_compat

    monkeypatch.setattr(_trl_compat, "_unpaired_cap_batch_size", lambda kl_chunk: kl_chunk + 1)
    with pytest.raises(RuntimeError, match="lender"):
        _trl_compat.enforce_preference_sequence_limit(
            _fitting_kto_layout(12),
            max_length=64,
            max_prompt_length=32,
            truncation_mode="keep_end",
            only_when_overflow=True,
            kl_chunk_size=4,
        )


# --------------------------------------------------------------------------
# #2487 -- a keep_end prompt cut keeps the tokenizer's BOS at index 0 for KTO
# and BCO: BOS, then the last ``max_prompt_length - 1`` prompt tokens
# --------------------------------------------------------------------------
_CAP_PROMPT = [_UNPAIRED_BOS] + list(range(100, 116))  # 17 tokens, BOS first
_CAP_ANSWER = [200, 201]
_CAP_MAX_PROMPT_LENGTH = 4
_CAP_MAX_LENGTH = 8
#: BOS plus the last ``_CAP_MAX_PROMPT_LENGTH - 1`` prompt tokens.
_CAP_PROMPT_WITH_BOS = [_UNPAIRED_BOS, 113, 114, 115]


def _overlong_unpaired_layout(prompt, *, with_answer=True):
    """One KTO-layout row whose prompt overflows ``_CAP_MAX_LENGTH`` on its own."""
    from datasets import Dataset

    ids = prompt + _CAP_ANSWER + [_UNPAIRED_EOS]
    labels = [_IGNORE_LABEL] * len(prompt) + _CAP_ANSWER + [_UNPAIRED_EOS]
    row = {
        "prompt_input_ids": prompt,
        "prompt_attention_mask": [1] * len(prompt),
        "completion_input_ids": ids,
        "completion_attention_mask": [1] * len(ids),
        "completion_labels": labels,
        "KL_prompt_input_ids": prompt,
        "KL_prompt_attention_mask": [1] * len(prompt),
        "KL_completion_input_ids": ids,
        "KL_completion_attention_mask": [1] * len(ids),
        "KL_completion_labels": labels,
    }
    if with_answer:
        row["answer_input_ids"] = _CAP_ANSWER
        row["answer_attention_mask"] = [1] * len(_CAP_ANSWER)
    return Dataset.from_list([row])


def _cap_one_row(dataset, **overrides):
    from soup_cli.trainer._trl_compat import enforce_preference_sequence_limit

    kwargs = {
        "max_length": _CAP_MAX_LENGTH,
        "max_prompt_length": _CAP_MAX_PROMPT_LENGTH,
        "truncation_mode": "keep_end",
        "only_when_overflow": True,
        "kl_chunk_size": 1,
        "bos_token_id": _UNPAIRED_BOS,
    }
    kwargs.update(overrides)
    return enforce_preference_sequence_limit(dataset, **kwargs)[0]


@pytest.mark.parametrize(
    ("column", "expected"),
    (
        ("prompt_input_ids", _CAP_PROMPT_WITH_BOS),
        ("KL_prompt_input_ids", _CAP_PROMPT_WITH_BOS),
        ("completion_input_ids", _CAP_PROMPT_WITH_BOS + _CAP_ANSWER + [_UNPAIRED_EOS]),
        ("KL_completion_input_ids", _CAP_PROMPT_WITH_BOS + _CAP_ANSWER + [_UNPAIRED_EOS]),
    ),
)
def test_a_keep_end_cut_keeps_the_leading_bos(column, expected):
    capped = _cap_one_row(_overlong_unpaired_layout(_CAP_PROMPT))
    assert list(capped[column]) == expected, column
    mask = column.replace("input_ids", "attention_mask")
    assert len(capped[mask]) == len(capped[column]), mask


@pytest.mark.parametrize("prefix", ("completion", "KL_completion"))
def test_a_keep_end_cut_keeps_the_leading_bos_without_answer_columns(prefix):
    """No ``answer_*`` columns: the combined sequence is cut as it stands."""
    capped = _cap_one_row(_overlong_unpaired_layout(_CAP_PROMPT, with_answer=False))
    ids = list(capped[f"{prefix}_input_ids"])
    assert ids == _CAP_PROMPT_WITH_BOS + _CAP_ANSWER + [_UNPAIRED_EOS], prefix
    assert list(capped[f"{prefix}_labels"]) == (
        [_IGNORE_LABEL] * _CAP_MAX_PROMPT_LENGTH + _CAP_ANSWER + [_UNPAIRED_EOS]
    ), prefix
    assert len(capped[f"{prefix}_attention_mask"]) == len(ids), prefix


_ALL_CAPPED_COLUMNS = (
    "prompt_input_ids",
    "KL_prompt_input_ids",
    "completion_input_ids",
    "KL_completion_input_ids",
)


def test_a_prompt_that_does_not_start_with_bos_gets_the_plain_keep_end_cut():
    prompt = [100] + _CAP_PROMPT  # BOS is present, but not at index 0
    capped = _cap_one_row(_overlong_unpaired_layout(prompt))
    tail = prompt[-_CAP_MAX_PROMPT_LENGTH:]
    for column in _ALL_CAPPED_COLUMNS:
        assert list(capped[column])[:_CAP_MAX_PROMPT_LENGTH] == tail, column


def test_without_a_bos_token_id_the_cut_is_unchanged():
    tail = _CAP_PROMPT[-_CAP_MAX_PROMPT_LENGTH:]
    capped = _cap_one_row(_overlong_unpaired_layout(_CAP_PROMPT), bos_token_id=None)
    for column in _ALL_CAPPED_COLUMNS:
        assert list(capped[column])[:_CAP_MAX_PROMPT_LENGTH] == tail, column


def test_keep_start_is_unchanged_by_the_bos_token_id():
    head = _CAP_PROMPT[:_CAP_MAX_PROMPT_LENGTH]
    capped = _cap_one_row(_overlong_unpaired_layout(_CAP_PROMPT), truncation_mode="keep_start")
    for column in _ALL_CAPPED_COLUMNS:
        assert list(capped[column])[:_CAP_MAX_PROMPT_LENGTH] == head, column


def test_a_row_that_fits_is_unchanged_by_the_bos_token_id():
    dataset = _overlong_unpaired_layout(_CAP_PROMPT)
    capped = _cap_one_row(dataset, max_length=64, max_prompt_length=2)
    assert capped == dataset[0]


def test_a_one_token_prompt_cap_keeps_only_the_bos():
    capped = _cap_one_row(_overlong_unpaired_layout(_CAP_PROMPT), max_prompt_length=1)
    for column in _ALL_CAPPED_COLUMNS:
        assert list(capped[column])[:1] == [_UNPAIRED_BOS], column
    assert list(capped["prompt_input_ids"]) == [_UNPAIRED_BOS]
    assert list(capped["KL_prompt_input_ids"]) == [_UNPAIRED_BOS]


def test_truncate_tokens_keeps_its_default_for_the_mlx_callers():
    from soup_cli.trainer._trl_compat import _truncate_tokens

    tokens = [_UNPAIRED_BOS, 10, 11, 12, 13]
    assert _truncate_tokens(tokens, 3, "keep_end") == [11, 12, 13]
    assert _truncate_tokens(tokens, 3, "keep_end", keep_first=True) == [_UNPAIRED_BOS, 12, 13]
    assert _truncate_tokens(tokens, 3, "keep_start", keep_first=True) == [_UNPAIRED_BOS, 10, 11]


def test_truncate_prompt_leaves_an_empty_prompt_empty():
    from soup_cli.trainer._trl_compat import _truncate_prompt

    assert _truncate_prompt([], [], 4, "keep_end", _UNPAIRED_BOS) == ([], [])


def _prompt_prefixes(task):
    return ("", "KL_") if task == "kto" else ("",)


@pytest.mark.parametrize("task", ("bco", "kto"))
def test_an_overlong_prompt_keeps_its_bos_on_train_and_validation(tmp_path, monkeypatch, task):
    rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good answer")
    val_rows = _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good answer", n=4)
    wrapper = _setup_rows(tmp_path, monkeypatch, task, rows, val_rows=val_rows)
    tok = wrapper.tokenizer
    assert tok.bos_token_id is not None
    for split, dataset in (
        ("train", wrapper.trainer.train_dataset),
        ("validation", wrapper.trainer.eval_dataset),
    ):
        assert len(dataset) > 0, split
        for index, row in enumerate(dataset):
            for prefix in _prompt_prefixes(task):
                case = (task, split, index, prefix)
                prompt = list(row[f"{prefix}prompt_input_ids"])
                ids = list(row[f"{prefix}completion_input_ids"])
                labels = list(row[f"{prefix}completion_labels"])
                assert prompt[0] == tok.bos_token_id, (*case, "prompt lost its BOS")
                assert ids[0] == tok.bos_token_id, (*case, "completion lost its BOS")
                assert len(prompt) <= _MAX_PROMPT_LENGTH, (*case, len(prompt))
                assert len(ids) <= _MAX_SEQUENCE_LENGTH, (*case, len(ids))
                assert len(row[f"{prefix}prompt_attention_mask"]) == len(prompt), case
                assert len(row[f"{prefix}completion_attention_mask"]) == len(ids), case
                masked = [token for token, label in zip(ids, labels) if label == _IGNORE_LABEL]
                assert tok.convert_ids_to_tokens(masked[-1:]) == ["world"], case
                assert ids[-1] == tok.eos_token_id, (*case, "the completion lost its EOS")


def _write_tiny_tokenizer_without_bos(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
    )
    fast.save_pretrained(str(directory))


@pytest.mark.parametrize("task", ("bco", "kto"))
def test_a_tokenizer_without_bos_gets_todays_rows(tmp_path, monkeypatch, task):
    import importlib

    from soup_cli.trainer import _trl_compat

    module, cls_name, _ = _WRAPPERS[task]
    trainer_module = importlib.import_module(module)
    calls = []
    original = _trl_compat.enforce_preference_sequence_limit

    def spy(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return original(dataset, **kwargs)

    monkeypatch.setattr(_trl_compat, "enforce_preference_sequence_limit", spy)
    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer_without_bos(weights)
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(weights, tmp_path / "out", task)
    wrapper = getattr(trainer_module, cls_name)(cfg, device="cpu")
    wrapper.setup({"train": _rows_with(task, _OVERLONG_PROMPT_WITH_TAIL, " good answer")})
    assert wrapper.tokenizer.bos_token_id is None
    assert len(calls) == 1, calls
    prepared, kwargs = calls[0]
    assert kwargs.get("bos_token_id") is None, kwargs
    today = original(prepared, **{k: v for k, v in kwargs.items() if k != "bos_token_id"})
    capped = wrapper.trainer.train_dataset
    for name in today.column_names:
        assert capped[name] == today[name], (task, name)
