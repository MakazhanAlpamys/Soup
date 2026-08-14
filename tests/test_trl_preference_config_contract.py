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


def _kto_rows(n=8):
    return [{"prompt": "hi", "completion": " good answer", "label": i % 2 == 0} for i in range(n)]


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
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
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
            assert args.max_prompt_length == 32, (task, args.max_prompt_length)
        else:
            assert not hasattr(args, "max_prompt_length"), (
                f"{task}: installed trl does not accept `max_prompt_length` as a "
                f"keyword, yet the built config has the attribute — the capability "
                f"probe and the object disagree"
            )
        assert args.max_length == 64, (task, args.max_length)


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
    """The floor shipped as `>=0.7.0` while `setup()` imported `GRPOTrainer`,
    which trl first exports at 0.14.0 — a declared floor the code could never
    have run on. Cheap to state as a property: whatever is installed must
    satisfy the declared bound AND provide the symbols the trainers import."""

    def test_the_installed_trl_provides_every_symbol_the_trainers_import(self):
        """`getattr`, not `hasattr`: trl exposes these through a lazy module, so
        a submodule that fails to import raises something other than
        AttributeError, and `hasattr` would report a clean False without saying
        why. From 0.29 `ORPOConfig` / `CPOConfig` / `BCOConfig` leave the `trl`
        namespace altogether — that is the shape of the next break, so it is
        worth naming the symbol rather than only the version."""
        import trl

        broken = {}
        for name in (
            "DPOConfig",
            "DPOTrainer",
            "KTOConfig",
            "KTOTrainer",
            "ORPOConfig",
            "ORPOTrainer",
            "CPOConfig",
            "CPOTrainer",
            "BCOConfig",
            "BCOTrainer",
            "GRPOTrainer",
            "GRPOConfig",
        ):
            try:
                getattr(trl, name)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                broken[name] = f"{type(exc).__name__}: {exc}"
        assert not broken, (
            f"installed trl {trl.__version__} cannot supply "
            f"{sorted(broken)} — the [train] extra's bounds and the code "
            f"disagree: {broken}"
        )
