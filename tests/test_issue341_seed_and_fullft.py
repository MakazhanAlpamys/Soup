"""#341 (training seed) + #340 (first-class full fine-tuning).

Two schema gaps that share `config/schema.py` + `trainer/sft.py`.

**#341** — there was no general training seed. `TrainingArguments` took HF's
default on every run (`seed=42, data_seed=None`), so "run it again with a
different seed" was impossible and `sft.py`'s `getattr(tcfg, "seed", 0) or 0`
was a defensive lookup for an attribute that did not exist.

**#340** — `SFTTrainerWrapper._setup_transformers` had no path that trains
without adapters; the `else` branch applied LoRA unconditionally. The only
shipped spelling for full fine-tuning was `unfrozen_parameters: ['.*']`, the
Spectrum regex feature used as a workaround. `lora.r: 0` is now the
first-class switch — the spelling `trainer/classifier.py` already uses
(`_lora_active = classifier_lora and tcfg.lora.r > 0`) and the one
`commands/card.py::_is_adapter` already reads correctly.

The controls are the point:

* seed — two runs at the SAME seed must reproduce each other. Without that,
  "different seeds give different weights" also passes for a seed that is
  ignored entirely (any two runs differ for unrelated reasons).
* full-FT — a normal LoRA config must STILL produce a PEFT-wrapped model with
  the base frozen. Without that, "no PEFT wrapper" also passes for a build
  that silently stopped applying LoRA to everyone.
* backwards compatibility — a config setting neither new field must reach
  `TrainingArguments` with `seed=42, data_seed=None`, i.e. exactly what HF's
  own defaults produced before this change.

The schema half is import-light and runs without any `[train]` extra; the
behavioural half skips (loudly) when torch / peft / trl are absent.
"""

import json
import os

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string

# ==========================================================================
# Shared config builders (pure YAML — no torch)
# ==========================================================================
_BASE_CFG = {
    # A Llama-family name on purpose: several unrelated validators
    # (`use_longlora`, multipack, MoD) carry architecture allowlists keyed off
    # the base string, and a gpt2 name trips those first — masking the gate
    # under test with someone else's rejection.
    "base": "meta-llama/Llama-3.2-1B",
    "task": "sft",
    "backend": "transformers",
    "modality": "text",
    "data": {"train": "train.jsonl"},
    "training": {"quantization": "none"},
}


def _cfg_yaml(**overrides):
    """Deep-ish merge of `training` / `data` overrides onto the base config."""
    import copy

    cfg = copy.deepcopy(_BASE_CFG)
    for key, val in overrides.items():
        if key in ("training", "data", "lora") and isinstance(val, dict):
            if key == "lora":
                cfg["training"].setdefault("lora", {}).update(val)
            else:
                cfg[key].update(val)
        else:
            cfg[key] = val
    return yaml.safe_dump(cfg)


def _load(**overrides):
    return load_config_from_string(_cfg_yaml(**overrides))


# ==========================================================================
# #341 — schema (light: no torch)
# ==========================================================================
class TestSeedSchema:
    def test_seed_and_data_seed_exist_and_default_to_none(self):
        """`None` (not 42) is the schema default so the trainer can tell
        "unset" from "explicitly 42" — the multipack sampler's historical
        default of 0 has to survive an unset seed, while `TrainingArguments`
        keeps HF's 42."""
        cfg = _load()
        assert cfg.training.seed is None
        assert cfg.training.data_seed is None

    def test_seed_accepts_an_int(self):
        assert _load(training={"seed": 1234}).training.seed == 1234

    def test_data_seed_accepts_an_int(self):
        assert _load(training={"data_seed": 99}).training.data_seed == 99

    def test_zero_is_a_legitimate_seed(self):
        """0 is a real seed, not a sentinel — a `ge=1` bound would silently
        make `seed: 0` unusable."""
        assert _load(training={"seed": 0}).training.seed == 0

    @pytest.mark.parametrize("field", ["seed", "data_seed"])
    def test_bool_is_rejected(self, field):
        """`bool` subclasses `int`, so `seed: true` would silently become
        seed 1. Mirrors the v0.71.34 LISA / v0.41.0 expand_layers policy."""
        with pytest.raises(ValueError, match="(?i)bool"):
            _load(training={field: True})

    @pytest.mark.parametrize("field", ["seed", "data_seed"])
    def test_negative_is_rejected(self, field):
        with pytest.raises(ValueError):
            _load(training={field: -1})

    @pytest.mark.parametrize("field", ["seed", "data_seed"])
    def test_absurdly_large_is_rejected(self, field):
        """torch's RNG takes a 64-bit seed; numpy's `default_rng` and HF's
        `set_seed` path go through 32-bit consumers, so an out-of-range value
        raises deep inside numpy at step 0 instead of at config load."""
        with pytest.raises(ValueError):
            _load(training={field: 2**32})

    def test_seed_survives_a_yaml_round_trip(self):
        cfg = _load(training={"seed": 7, "data_seed": 8})
        dumped = yaml.safe_dump(json.loads(cfg.model_dump_json()))
        again = load_config_from_string(dumped)
        assert (again.training.seed, again.training.data_seed) == (7, 8)


# ==========================================================================
# #340 — schema (light: no torch)
# ==========================================================================
class TestFullFinetuneSchema:
    def test_lora_r_zero_is_accepted_on_sft(self):
        cfg = _load(lora={"r": 0})
        assert cfg.training.lora.r == 0

    def test_control_the_default_lora_config_is_still_valid(self):
        """CONTROL — the new gates must not reject an ordinary LoRA run."""
        cfg = _load()
        assert cfg.training.lora.r == 64

    def test_negative_rank_is_rejected(self):
        """Pre-existing hole: `r` had no lower bound, so `r: -5` parsed and
        died inside peft. 0 now means full-FT; below 0 means nothing."""
        with pytest.raises(ValueError):
            _load(lora={"r": -5})

    def test_full_ft_requires_transformers_backend(self):
        with pytest.raises(ValueError, match="backend"):
            _load(backend="unsloth", lora={"r": 0})

    def test_full_ft_requires_text_modality(self):
        with pytest.raises(ValueError, match="modality"):
            _load(modality="vision", lora={"r": 0})

    def test_full_ft_requires_quantization_none(self):
        """Quantized weights cannot be trained directly — the same reason
        Spectrum and LISA carry this gate."""
        with pytest.raises(ValueError, match="quantization"):
            _load(training={"quantization": "4bit"}, lora={"r": 0})

    @pytest.mark.parametrize(
        "training_over,lora_over,needle",
        [
            ({}, {"use_dora": True}, "use_dora"),
            ({}, {"use_vera": True}, "use_vera"),
            ({}, {"use_olora": True}, "use_olora"),
            ({}, {"use_rslora": True}, "use_rslora"),
            ({}, {"init_strategy": "pissa"}, "init_strategy"),
            ({}, {"rank_pattern": {"q_proj": 8}}, "rank_pattern"),
            ({}, {"alpha_pattern": {"q_proj": 8}}, "alpha_pattern"),
            ({"moe_lora": True}, {}, "moe_lora"),
            ({"relora_steps": 100}, {}, "relora_steps"),
            ({"loraplus_lr_ratio": 4.0}, {}, "loraplus_lr_ratio"),
            ({"use_longlora": True}, {}, "use_longlora"),
        ],
    )
    def test_full_ft_rejects_lora_features(self, training_over, lora_over, needle):
        """A LoRA knob alongside `r: 0` is a contradiction the user must see —
        silently ignoring it is how someone ends up believing DoRA ran."""
        over = dict(training_over)
        with pytest.raises(ValueError, match=needle):
            _load(training=over, lora={"r": 0, **lora_over})

    def test_full_ft_rejects_unfrozen_parameters(self):
        """Both select what trains; together there is no defined answer."""
        with pytest.raises(ValueError, match="unfrozen_parameters"):
            _load(
                training={"unfrozen_parameters": ["model.layers.0.mlp.down_proj"]},
                lora={"r": 0},
            )

    def test_full_ft_rejects_lisa(self):
        with pytest.raises(ValueError, match="lisa_enabled"):
            _load(training={"lisa_enabled": True}, lora={"r": 0})

    def test_full_ft_rejects_layer_streaming(self):
        """Streaming keeps the decoder on the meta device and trains only the
        adapter, so full-FT is not merely unwise there, it is impossible. The
        message must say so rather than the pre-#340 "nothing trainable"."""
        with pytest.raises(ValueError, match="(?i)full fine-tun"):
            _load(
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                },
                lora={"r": 0},
            )

    def test_the_gates_are_scoped_to_sft_and_do_not_fire_elsewhere(self):
        """DELIBERATE SCOPING, pinned so it cannot be "tidied up" later.

        `r: 0` gains full-FT semantics for `task='sft'` only. Widening the
        gate to every task would reject configs that are legal TODAY — see
        the two controls below — so a non-sft task keeps whatever it does
        now (peft's "`r` should be a positive integer value" where the rank
        is passed through unconditionally).
        """
        cfg = _load(task="dpo", data={"format": "dpo"}, lora={"r": 0})
        assert cfg.training.lora.r == 0
        # ... and the sft-only gates did NOT fire: this combination would be
        # refused under task='sft' (4bit), but is untouched here.
        assert _load(
            task="dpo",
            data={"format": "dpo"},
            training={"quantization": "4bit"},
            lora={"r": 0},
        ).training.quantization == "4bit"

    def test_control_asr_may_still_set_rank_zero(self):
        """CONTROL — `trainer/asr.py` gates LoRA behind `asr_lora` (v0.71.32,
        default False = full fine-tune), so `r: 0` is already inert there."""
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": "openai/whisper-tiny",
                    "task": "asr",
                    "backend": "transformers",
                    "data": {"train": "train.jsonl", "format": "asr"},
                    "training": {"quantization": "none", "lora": {"r": 0}},
                }
            )
        )
        assert cfg.training.lora.r == 0

    def test_control_classifier_may_still_set_rank_zero(self):
        """CONTROL — `trainer/classifier.py` has read `lora.r > 0` as "no
        adapter" since v0.71.12, and full fine-tuning is the classifier's
        DEFAULT. A gate that refused `r: 0` outside sft would break a config
        that is legal today."""
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": "bert-base-uncased",
                    "task": "classifier",
                    "data": {"train": "train.jsonl", "format": "auto"},
                    "training": {"num_labels": 2, "lora": {"r": 0}},
                }
            )
        )
        assert cfg.training.lora.r == 0


# ==========================================================================
# Behavioural half — needs the [train] extra
# ==========================================================================
def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _tiny_llama_dir(tmp_path, n_layers=2):
    """A real (tiny) Llama checkpoint + offline tokenizer on disk.

    Complete on purpose: `from_pretrained` must initialise NOTHING at random,
    or two runs in one process would differ for a reason that is not the seed.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        # A downloaded checkpoint always carries this; a from-scratch config
        # does not, and `multipack_trainer.detect_arch_name` reads it FIRST,
        # falling back to `type(model).__name__` — which under LoRA is
        # `PeftModelForCausalLM` and matches no allowlist. Without this the
        # multipack tests would fail on a fixture artifact no user can hit.
        architectures=["LlamaForCausalLM"],
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)  # tied
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    _write_tiny_tokenizer(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    """A real, offline PreTrainedTokenizerFast — no HF cache, no network."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in (
        "hello", "world", "hi", "yo", "the", "cat", "sat", "on", "mat",
        "dog", "ran", "fast", "slow", "red", "blue", "green", "one", "two",
    ):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            fh,
        )


_ROWS = [
    ("hi", "hello world"),
    ("yo", "the cat sat"),
    ("hello", "on the mat"),
    ("the dog", "ran fast"),
    ("red", "blue green"),
    ("one", "two red"),
    ("cat", "sat slow"),
    ("world", "hi yo"),
]


def _dataset():
    return {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            for user, assistant in _ROWS
        ]
    }


def _wrapper(tmp_path, monkeypatch, base=None, **training_over):
    """A real SFTTrainerWrapper over a real tiny checkpoint."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    base = base or _tiny_llama_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    training = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "epochs": 1,
        "lr": 1e-3,
        "logging_steps": 100,
        "save_steps": 10_000,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    lora_over = training_over.pop("lora", None)
    training.update(training_over)
    if lora_over is not None:
        training["lora"] = {**training["lora"], **lora_over}
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    return SFTTrainerWrapper(cfg, device="cpu"), _dataset()


# ==========================================================================
# #341 — the seed reaches TrainingArguments
# ==========================================================================
class TestSeedReachesTrainingArguments:
    def test_default_is_still_42_and_data_seed_none(self, tmp_path, monkeypatch):
        """BACKWARDS COMPATIBILITY. A config that sets neither field must
        produce byte-identical TrainingArguments to the pre-#341 build, where
        HF's own defaults applied."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)
        assert wrapper.trainer.args.seed == 42
        assert wrapper.trainer.args.data_seed is None

    def test_configured_seed_reaches_the_trainer(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, seed=1234)
        wrapper.setup(dataset)
        assert wrapper.trainer.args.seed == 1234

    def test_configured_data_seed_reaches_the_trainer(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, seed=5, data_seed=99)
        wrapper.setup(dataset)
        assert wrapper.trainer.args.seed == 5
        assert wrapper.trainer.args.data_seed == 99


class TestSeedActuallyChangesTraining:
    """The acceptance pair from #341. Neither test means anything alone."""

    @staticmethod
    def _train(tmp_path, monkeypatch, base, seed):
        import torch

        # Normalise the pre-Trainer RNG state: `get_peft_model` draws lora_A
        # from the GLOBAL generator, and these runs share one process, so
        # without this the second run's adapter starts somewhere else and
        # every comparison below measures init, not the seed.
        torch.manual_seed(0)
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, base=base, seed=seed)
        wrapper.setup(dataset)
        wrapper.trainer.train()
        # `.cpu()` because HF's Trainer moves the model to `args.device`, which
        # is CUDA on a GPU box even though the wrapper was built with
        # device="cpu" — comparing a cpu clone against a cuda tensor raises.
        return {
            name: param.detach().cpu().clone()
            for name, param in wrapper.model.named_parameters()
            if "lora_" in name and param.requires_grad
        }

    def test_same_seed_reproduces(self, tmp_path, monkeypatch):
        """THE CONTROL. Without it, `test_different_seed_diverges` passes for
        a seed that is thrown away — two runs of anything differ."""
        _requires_train_extra()
        import torch

        base = _tiny_llama_dir(tmp_path)
        first = self._train(tmp_path, monkeypatch, base, seed=1234)
        second = self._train(tmp_path, monkeypatch, base, seed=1234)
        assert first and second, "no trainable LoRA parameters were captured"
        for name in first:
            assert torch.equal(first[name], second[name]), (
                f"{name} differs across two runs at the SAME seed"
            )

    def test_different_seed_diverges(self, tmp_path, monkeypatch):
        _requires_train_extra()
        import torch

        base = _tiny_llama_dir(tmp_path)
        first = self._train(tmp_path, monkeypatch, base, seed=1234)
        other = self._train(tmp_path, monkeypatch, base, seed=4321)
        assert first and other
        assert any(
            not torch.equal(first[name], other[name]) for name in first
        ), "changing training.seed changed nothing — the value is being ignored"


class TestMultipackSamplerSeed:
    """`sft.py`'s `getattr(tcfg, "seed", 0) or 0` was a lookup for an
    attribute that did not exist, so the FFD sampler was always seeded 0."""

    @staticmethod
    def _seed_passed_to_multipack(tmp_path, monkeypatch, **over):
        import soup_cli.utils.multipack_trainer as mt

        seen = {}
        real = mt.attach_multipack_state

        def spy(trainer, **kwargs):
            seen.update(kwargs)
            return real(trainer, **kwargs)

        monkeypatch.setattr(mt, "attach_multipack_state", spy)
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, multipack=True, **over)
        wrapper.setup(dataset)
        return seen["seed"]

    def test_configured_seed_reaches_the_sampler(self, tmp_path, monkeypatch):
        _requires_train_extra()
        assert self._seed_passed_to_multipack(tmp_path, monkeypatch, seed=77) == 77

    def test_control_unset_seed_still_seeds_the_sampler_zero(self, tmp_path, monkeypatch):
        """CONTROL + backwards compatibility. An unset seed must keep the
        historical 0, or every existing multipack run silently re-orders."""
        _requires_train_extra()
        assert self._seed_passed_to_multipack(tmp_path, monkeypatch) == 0


# ==========================================================================
# #340 — full fine-tuning in the trainer
# ==========================================================================
class TestFullFinetuneTrainer:
    def test_rank_zero_leaves_the_model_unwrapped_and_fully_trainable(
        self, tmp_path, monkeypatch
    ):
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, lora={"r": 0})
        wrapper.setup(dataset)

        from peft import PeftModel

        assert not isinstance(wrapper.model, PeftModel), (
            "full fine-tuning must not wrap the model in PEFT"
        )
        assert not any("lora_" in name for name, _ in wrapper.model.named_parameters())
        untrainable = [
            name
            for name, param in wrapper.model.named_parameters()
            if not param.requires_grad
        ]
        assert not untrainable, f"full-FT left parameters frozen: {untrainable[:5]}"

    def test_control_lora_still_wraps_in_peft_with_the_base_frozen(
        self, tmp_path, monkeypatch
    ):
        """CONTROL. Without it, the test above also passes for a build that
        stopped applying LoRA to everybody."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)

        from peft import PeftModel

        assert isinstance(wrapper.model, PeftModel)
        trainable = [
            name
            for name, param in wrapper.model.named_parameters()
            if param.requires_grad
        ]
        assert trainable, "the LoRA control trains nothing"
        assert all("lora_" in name for name in trainable), (
            f"base weights are trainable under LoRA: {trainable[:5]}"
        )

    def test_a_full_ft_step_actually_updates_base_weights(self, tmp_path, monkeypatch):
        """The end-to-end claim: forward + backward + optimizer step move the
        BASE weights, which is the whole point of full fine-tuning."""
        _requires_train_extra()
        import torch

        wrapper, dataset = _wrapper(tmp_path, monkeypatch, lora={"r": 0}, seed=3)
        wrapper.setup(dataset)
        target = "model.layers.0.self_attn.q_proj.weight"
        before = dict(wrapper.model.named_parameters())[target].detach().cpu().clone()
        wrapper.trainer.args.max_steps = 2
        wrapper.trainer.train()
        after = dict(wrapper.model.named_parameters())[target].detach().cpu()
        assert not torch.equal(before, after), (
            "a full fine-tuning step left the base weights untouched"
        )

    def test_the_summary_line_says_full_fine_tuning_not_lora_applied(
        self, tmp_path, monkeypatch, capsys
    ):
        """`setup()` prints "LoRA applied: N trainable" — a false statement on
        a run with no adapter, and the line an operator screenshots."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, lora={"r": 0})
        wrapper.setup(dataset)
        out = capsys.readouterr().out
        assert "LoRA applied" not in out
        assert "Full fine-tuning" in out

    def test_full_ft_trains_under_gradient_checkpointing(self, tmp_path, monkeypatch):
        """Gradient checkpointing is how full-FT fits at all, so the two are
        the common pairing, not an exotic one. A frozen input embedding breaks
        the backward pass under checkpointing ("none of the inputs have
        requires_grad"), which is why the branch calls
        `enable_input_require_grads` exactly as `get_peft_model` does."""
        _requires_train_extra()
        import torch

        wrapper, dataset = _wrapper(
            tmp_path,
            monkeypatch,
            lora={"r": 0},
            gradient_checkpointing=True,
            seed=11,
        )
        wrapper.setup(dataset)
        target = "model.layers.0.mlp.down_proj.weight"
        before = dict(wrapper.model.named_parameters())[target].detach().cpu().clone()
        wrapper.trainer.args.max_steps = 1
        wrapper.trainer.train()
        after = dict(wrapper.model.named_parameters())[target].detach().cpu()
        assert not torch.equal(before, after)

    def test_full_ft_respects_freeze_layers_instead_of_undoing_it(
        self, tmp_path, monkeypatch
    ):
        """`freeze_layers` stays legal with `r: 0` and the branch must NOT
        `requires_grad_(True)` over the top of it — "train everything above
        layer N" is a real technique, and silently unfreezing would train a
        model the config said not to."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, lora={"r": 0}, freeze_layers=1
        )
        wrapper.setup(dataset)
        by_name = dict(wrapper.model.named_parameters())
        assert not by_name["model.layers.0.self_attn.q_proj.weight"].requires_grad
        assert by_name["model.layers.1.self_attn.q_proj.weight"].requires_grad

    def test_full_ft_refuses_a_model_with_nothing_trainable(self, tmp_path, monkeypatch):
        """`freeze_layers` stays legal with `r: 0` (train everything above
        layer N is a real technique), so a config CAN freeze the whole model.
        That must fail loudly, not run for hours as a no-op."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, lora={"r": 0})

        from soup_cli.trainer.sft import SFTTrainerWrapper

        original = SFTTrainerWrapper._setup_transformers

        def freeze_everything(self, cfg, tcfg):
            import transformers

            real_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

            def loader(*args, **kwargs):
                model = real_from_pretrained(*args, **kwargs)
                model.requires_grad_(False)
                return model

            monkeypatch.setattr(
                transformers.AutoModelForCausalLM, "from_pretrained", loader
            )
            return original(self, cfg, tcfg)

        monkeypatch.setattr(
            SFTTrainerWrapper, "_setup_transformers", freeze_everything
        )
        with pytest.raises(ValueError, match="(?i)no parameter is trainable"):
            wrapper.setup(dataset)


# ==========================================================================
# Import hygiene — the schema half must stay on the light CLI path
# ==========================================================================
class TestSchemaHalfStaysLight:
    def test_config_schema_still_imports_without_torch(self):
        """`config/schema.py` is imported by every light command; a stray
        top-level torch import there would cost `soup --help` ~5 s."""
        import ast
        import pathlib

        import soup_cli.config.schema as schema_mod

        tree = ast.parse(pathlib.Path(schema_mod.__file__).read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and node.col_offset == 0:
                offenders += [
                    a.name for a in node.names if a.name.split(".")[0] == "torch"
                ]
            elif isinstance(node, ast.ImportFrom) and node.col_offset == 0:
                if (node.module or "").split(".")[0] == "torch":
                    offenders.append(node.module)
        assert not offenders, f"top-level torch import in schema.py: {offenders}"
