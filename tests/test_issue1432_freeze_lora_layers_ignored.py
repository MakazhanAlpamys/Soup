"""#1432: freeze_layers / freeze_ratio did nothing under LoRA.

``_setup_transformers`` called ``freeze_model_layers()`` (sets
``requires_grad=False`` on the bottom layers) BEFORE ``get_peft_model()``
wraps the model, and the ``LoraConfig`` it built carried no
``layers_to_transform``, so ``get_peft_model`` re-attached a fresh, trainable
LoRA adapter to every targeted module in EVERY layer, including the ones just
frozen. The freeze plan was silently discarded: a "frozen" layer still
trained, through its adapter, exactly as if freeze_layers/freeze_ratio had
never been set.

The fix threads the same ``(cutoff, total_layers)`` pair
``freeze_model_layers`` already computes (now exposed as
``freeze.freeze_layer_cutoff``) into ``build_lora_config`` as
``layers_to_transform``, so ``get_peft_model`` only attaches an adapter to
the layers the freeze plan actually left trainable.

The controls are the point:

* the MECHANISM control (``TestLayersToTransformIsWhatFixesIt``) directly
  reproduces the reported bug by calling ``build_lora_config`` the OLD way
  (no ``layers_to_transform``) against the same frozen model, and shows the
  frozen layer's adapter comes back trainable. That proves the harness would
  have caught #1432 before the fix, not just that the new code happens to
  pass.
* the real-trainer half (``TestFreezeLayersRestrictsLoraAdapter``) drives
  the actual ``SFTTrainerWrapper._setup_transformers`` path end to end.

Import-light: the schema/helper half runs without the ``[train]`` extra; the
behavioural half skips (loudly) when torch / transformers / peft are absent,
matching ``tests/test_issue341_seed_and_fullft.py``'s own convention.
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _tiny_llama_dir(tmp_path, n_layers=2):
    """A real (tiny) 2-layer Llama checkpoint + offline tokenizer on disk.

    Mirrors test_issue341_seed_and_fullft.py's helper: a real from-scratch
    model, not a mock, so get_peft_model's actual module-name matching (the
    thing #1432 is about) runs for real.
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
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "yo", "the", "cat", "sat", "on", "mat"):
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


def _dataset():
    rows = [("hi", "hello world"), ("yo", "the cat sat"), ("hello", "on the mat")]
    return {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            for user, assistant in rows
        ]
    }


def _wrapper(tmp_path, monkeypatch, n_layers=2, **training_over):
    """A real SFTTrainerWrapper over a real tiny checkpoint, LoRA on."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    base = _tiny_llama_dir(tmp_path, n_layers=n_layers)
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


def _lora_param_names(model):
    return [name for name, _ in model.named_parameters() if "lora_" in name]


class TestFreezeLayersRestrictsLoraAdapter:
    """The real _setup_transformers path, real tiny Llama, real get_peft_model."""

    def test_freeze_layers_excludes_frozen_layer_from_adapter(self, tmp_path, monkeypatch):
        """AmirF194's exact repro shape: freeze_layers=1 on a 2-layer model
        must leave layer 0 with no LoRA adapter at all, and layer 1 with a
        trainable one."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, freeze_layers=1)
        wrapper.setup(dataset)

        lora_names = _lora_param_names(wrapper.model)
        assert lora_names, "expected a LoRA adapter on the unfrozen layer"
        assert not any(".layers.0." in name for name in lora_names), (
            f"layer 0 is frozen but has a LoRA adapter: {lora_names}"
        )
        assert any(".layers.1." in name for name in lora_names)

        by_name = dict(wrapper.model.named_parameters())
        layer1_lora_a = next(n for n in by_name if ".layers.1." in n and "lora_A" in n)
        assert by_name[layer1_lora_a].requires_grad is True

        # The base weight itself is still frozen (freeze_model_layers' own
        # job, unaffected by this fix). Confirms the fix restricts the
        # ADAPTER without undoing the base freeze.
        base_q_proj = next(
            n
            for n in by_name
            if n.endswith("layers.0.self_attn.q_proj.base_layer.weight")
            or n.endswith("layers.0.self_attn.q_proj.weight")
        )
        assert by_name[base_q_proj].requires_grad is False

    def test_freeze_ratio_excludes_frozen_layers_from_adapter(self, tmp_path, monkeypatch):
        """freeze_ratio=0.5 on a 4-layer model freezes layers 0-1; only
        layers 2-3 may get a LoRA adapter."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, n_layers=4, freeze_ratio=0.5)
        wrapper.setup(dataset)

        lora_names = _lora_param_names(wrapper.model)
        frozen_layers_with_adapter = {
            idx for idx in (0, 1) if any(f".layers.{idx}." in n for n in lora_names)
        }
        assert not frozen_layers_with_adapter, (
            f"frozen layers still got a LoRA adapter: {frozen_layers_with_adapter}"
        )
        assert any(".layers.2." in n for n in lora_names)
        assert any(".layers.3." in n for n in lora_names)

    def test_no_freeze_still_adapts_every_layer(self, tmp_path, monkeypatch):
        """CONTROL: with no freeze_layers/freeze_ratio set, every layer
        still gets an adapter (this fix must not narrow the ordinary case)."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)

        lora_names = _lora_param_names(wrapper.model)
        assert any(".layers.0." in n for n in lora_names)
        assert any(".layers.1." in n for n in lora_names)

    def test_freeze_layers_with_default_auto_target_modules(self, tmp_path, monkeypatch):
        """training.lora.target_modules defaults to 'auto'. For a plain
        Llama-family model (no Soup table entry), resolve_lora_target_modules
        returns None and peft resolves its OWN default target list before
        matching. Confirm layers_to_transform still restricts the adapter
        on this, the path most configs actually take, not only the explicit
        target_modules list the other tests pin."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, freeze_layers=1, lora={"target_modules": "auto"}
        )
        wrapper.setup(dataset)

        lora_names = _lora_param_names(wrapper.model)
        assert lora_names
        assert not any(".layers.0." in n for n in lora_names)
        assert any(".layers.1." in n for n in lora_names)

    def test_freeze_everything_refuses_rather_than_silently_no_op(self, tmp_path, monkeypatch):
        """freeze_layers >= total layers leaves nothing for LoRA to attach
        to; must raise, not silently build a useless adapter."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, freeze_layers=99)
        with pytest.raises(ValueError, match="leaving no layer"):
            wrapper.setup(dataset)

    def test_freeze_layers_with_regex_target_modules_refuses(self, tmp_path, monkeypatch):
        """peft's layers_to_transform only filters a LIST of target module
        suffixes, never a regex target_modules string, so refuse loudly
        instead of silently shipping the same #1432 bug for this shape."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path,
            monkeypatch,
            freeze_layers=1,
            lora={"target_modules": r".*self_attn\.(q_proj|v_proj)"},
        )
        with pytest.raises(ValueError, match="regex"):
            wrapper.setup(dataset)

    def test_freeze_layers_with_target_parameters_refuses(self, tmp_path, monkeypatch):
        """peft's target_parameters (per-parameter LoRA) matching has no
        layers_to_transform check at all, so refuse loudly rather than
        silently ignore the freeze plan for this shape too. An explicit
        target_parameters list is returned unchanged by
        resolve_lora_target_parameters regardless of architecture, so this
        does not need a real MoE model to reach the guard."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path,
            monkeypatch,
            freeze_layers=1,
            lora={"target_parameters": ["mlp.gate_up_proj"]},
        )
        with pytest.raises(ValueError, match="target_parameters"):
            wrapper.setup(dataset)

    def test_freeze_layers_with_use_vera_refuses(self, tmp_path, monkeypatch):
        """VeRA is a separate peft tuner built by build_peft_config_spec's
        own branch, not the LoraConfig path layers_to_transform is wired
        into, so refuse loudly rather than silently reproduce #1432 for it
        too (VeraConfig does accept its own layers_to_transform, but nothing
        here passes it through)."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, freeze_layers=1, lora={"use_vera": True}
        )
        with pytest.raises(ValueError, match="use_vera"):
            wrapper.setup(dataset)

    def test_freeze_with_expand_layers_keeps_appended_blocks_adapted(self, tmp_path, monkeypatch):
        """#1494 review: lora_layers_to_transform's range was measured from
        the layer count BEFORE apply_block_expansion_if_configured (LLaMA
        Pro) runs, so blocks it appends land outside the range and get no
        adapter even though freeze_model_layers never froze them."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, n_layers=4,
            freeze_layers=1, expand_layers=2, freeze_trainable_layers=-1,
        )
        wrapper.setup(dataset)
        names = _lora_param_names(wrapper.model)
        assert not any(".layers.0." in name for name in names)
        for idx in (1, 2, 3, 4, 5):
            assert any(f".layers.{idx}." in name for name in names), (
                f"layer {idx} lost its adapter"
            )

    def test_freeze_with_embed_tokens_target_is_not_silently_dropped(self, tmp_path, monkeypatch):
        """#1494 review: peft only restricts layers_to_transform to a list
        target that lives inside a numbered decoder layer, so embed_tokens
        (module path model.embed_tokens, no layer index) silently lost its
        adapter as soon as any freeze option was set."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, n_layers=4, freeze_layers=2,
            lora={"target_modules": ["q_proj", "v_proj", "embed_tokens"]},
        )
        try:
            wrapper.setup(dataset)
        except ValueError as exc:
            assert "embed_tokens" in str(exc)
            return
        names = _lora_param_names(wrapper.model)
        assert any("embed_tokens" in name for name in names), (
            "embed_tokens adapter silently dropped"
        )

    def test_freeze_ratio_cutoff_zero_does_not_restrict_layers(self, tmp_path, monkeypatch):
        """#1494 review: freeze_ratio=0.1 on a 4-layer model resolves to
        cutoff=0 (nothing frozen), but the pre-fix code still built a
        non-None layers_to_transform=[0..4), which silently drops a target
        outside a numbered layer even though nothing was frozen."""
        _requires_train_extra()
        wrapper, dataset = _wrapper(
            tmp_path, monkeypatch, n_layers=4, freeze_ratio=0.1,
            lora={"target_modules": ["q_proj", "v_proj", "embed_tokens"]},
        )
        wrapper.setup(dataset)
        names = _lora_param_names(wrapper.model)
        assert any(".layers.0." in name for name in names)
        assert any("embed_tokens" in name for name in names)


class TestLayersToTransformIsWhatFixesIt:
    """MECHANISM control: reproduces #1432 directly against peft, without
    going through sft.py at all, and shows layers_to_transform is what
    closes it. If this control could not be made to fail on the pre-fix
    call shape, the behavioural tests above would not be trustworthy."""

    def test_without_layers_to_transform_frozen_layer_gets_a_trainable_adapter(
        self, tmp_path
    ):
        """This is #1432 itself: the exact pre-fix call (no
        layers_to_transform) against a model with layer 0 already frozen."""
        _requires_train_extra()
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.freeze import freeze_model_layers

        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            )
        )
        freeze_model_layers(model, freeze_layers=1)

        old_config = LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, old_config)
        by_name = dict(peft_model.named_parameters())
        layer0_lora_a = next(n for n in by_name if ".layers.0." in n and "lora_A" in n)
        # The bug: a fresh, trainable adapter on a layer that was frozen.
        assert by_name[layer0_lora_a].requires_grad is True

    def test_with_layers_to_transform_frozen_layer_gets_no_adapter(self, tmp_path):
        """Same setup, layers_to_transform=[1] (build_lora_config's #1432
        fix) added: layer 0 gets no adapter at all."""
        _requires_train_extra()
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.freeze import freeze_model_layers

        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            )
        )
        freeze_model_layers(model, freeze_layers=1)

        new_config = LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM, layers_to_transform=[1],
        )
        peft_model = get_peft_model(model, new_config)
        lora_names = [n for n, _ in peft_model.named_parameters() if "lora_" in n]
        assert not any(".layers.0." in n for n in lora_names)
        assert any(".layers.1." in n for n in lora_names)
