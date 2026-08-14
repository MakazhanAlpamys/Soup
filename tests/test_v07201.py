"""v0.72.1 — a streamed adapter must be a NORMAL adapter on disk.

v0.72.0's ``install_streaming`` replaces ``layers[i]`` with a wrapper that holds
the real layer as a child named ``inner``. Every adapter parameter therefore
serialised as ``...layers.0.inner.self_attn.q_proj.lora_A.weight``, and loading
that file into any normal model dropped **every** tensor — PEFT warns about
missing keys and hands back the untuned base. Training was correct; the artifact
it wrote was inert in ``soup merge`` / ``serve`` / ``chat`` and in
``PeftModel.from_pretrained``.

The assertion that matters is NOT "no exception raised": 0-of-N tensors loading
raises nothing at all. These tests assert by count, by name, and by value.
"""

import json
import os

import pytest

# --------------------------------------------------------------------------
# fixtures (standalone, mirroring tests/test_v07200.py)
# --------------------------------------------------------------------------


def _tiny_llama_dir(tmp_path, n_layers=2, tie=True):
    """A real (tiny) Llama checkpoint on disk: config.json + model.safetensors."""
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
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed_cpu(tmp_path, n_layers=2):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path, n_layers=n_layers)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index,
        lora_config=_tiny_lora(), device="cpu", dtype="float32",
        buffers=2, pin=False, seed=3,
    )
    return model, runtime, weights


def _build_plain_peft(weights_dir):
    """The reference: an ordinary, non-streaming LoRA model on the same base."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.float32)
    return get_peft_model(base, _tiny_lora())


def _make_adapters_detectable(model, value=0.05):
    """PEFT initialises lora_B to ZERO.

    An adapter that fails to load is byte-identical to a freshly-initialised one
    while B is zero, so every "did it load?" assertion would pass vacuously.
    Writing a non-zero B is what makes a dropped tensor detectable at all.
    """
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.full_like(param, value))
            elif "lora_A" in name:
                param.copy_(torch.full_like(param, value / 2))


def _adapter_keys(directory):
    from safetensors.torch import load_file

    return load_file(os.path.join(directory, "adapter_model.safetensors"))


# --------------------------------------------------------------------------
# the saved artifact
# --------------------------------------------------------------------------
class TestSavedAdapterKeysAreCanonical:
    def test_saved_keys_carry_no_inner_prefix(self, tmp_path):
        model, _, _ = _build_streamed_cpu(tmp_path)
        _make_adapters_detectable(model)
        out = tmp_path / "adapter"
        model.save_pretrained(str(out))

        saved = _adapter_keys(str(out))
        leaked = [key for key in saved if ".inner." in key]
        assert saved, "no adapter tensors were written at all"
        assert leaked == [], (
            f"{len(leaked)} of {len(saved)} saved keys carry the streaming "
            f"wrapper's '.inner.' segment, e.g. {leaked[:1]}"
        )

    def test_saved_key_set_is_identical_to_a_normal_lora_run(self, tmp_path):
        """The artifact must be indistinguishable from a non-streamed run.

        Portability is the point: tooling that has never heard of layer
        streaming has to load this file.
        """
        model, _, weights = _build_streamed_cpu(tmp_path)
        streamed_out = tmp_path / "streamed_adapter"
        model.save_pretrained(str(streamed_out))

        reference = _build_plain_peft(weights)
        plain_out = tmp_path / "plain_adapter"
        reference.save_pretrained(str(plain_out))

        assert set(_adapter_keys(str(streamed_out))) == set(_adapter_keys(str(plain_out)))

    def test_in_memory_state_dict_is_canonical(self, tmp_path):
        """The checkpoint path (Trainer._save) serialises via state_dict()."""
        from peft import get_peft_model_state_dict

        model, _, _ = _build_streamed_cpu(tmp_path)
        keys = list(get_peft_model_state_dict(model))
        assert keys
        assert [k for k in keys if ".inner." in k] == []

    def test_adapter_config_is_unchanged(self, tmp_path):
        """The fix must not perturb adapter_config.json."""
        model, _, weights = _build_streamed_cpu(tmp_path)
        out = tmp_path / "adapter"
        model.save_pretrained(str(out))
        with open(out / "adapter_config.json", encoding="utf-8") as handle:
            cfg = json.load(handle)
        assert cfg["r"] == 4
        assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]


# --------------------------------------------------------------------------
# THE regression test
# --------------------------------------------------------------------------
class TestAdapterReloadsIntoNormalModel:
    def test_every_tensor_lands_by_count_and_by_name_and_by_value(self, tmp_path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model, _, weights = _build_streamed_cpu(tmp_path)
        _make_adapters_detectable(model)
        out = tmp_path / "adapter"
        model.save_pretrained(str(out))
        saved = _adapter_keys(str(out))

        base = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(base, str(out))
        landed = {
            name: param
            for name, param in reloaded.named_parameters()
            if "lora_" in name
        }

        # by count
        assert len(landed) == len(saved), (
            f"saved {len(saved)} adapter tensors but the reloaded model exposes "
            f"{len(landed)}"
        )
        # by name + by value
        for key, tensor in saved.items():
            target = key.replace(".weight", ".default.weight")
            assert target in landed, f"saved key {key!r} did not land as {target!r}"
            assert torch.equal(landed[target].detach().cpu(), tensor.cpu()), (
                f"{target} loaded with different values"
            )
        # and it is not the untuned base: B must be non-zero somewhere
        assert any(
            param.abs().sum().item() > 0
            for name, param in landed.items()
            if "lora_B" in name
        ), "every lora_B is zero — the adapter loaded as a no-op"

    def test_the_assertion_would_catch_a_dropped_adapter(self, tmp_path):
        """Control: prove the check above is not vacuous.

        Rewriting the saved keys back to the broken v0.72.0 form must make the
        reload fail the same assertions — otherwise a green test says nothing.
        """
        import shutil

        import torch
        from peft import PeftModel
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM

        model, _, weights = _build_streamed_cpu(tmp_path)
        _make_adapters_detectable(model)
        out = tmp_path / "adapter"
        model.save_pretrained(str(out))

        # Write the v0.72.0-shaped copy into its OWN directory: safetensors
        # mmaps the file it just read, and overwriting it in place fails on
        # Windows with error 1224 (the same trap utils/adapter_fuse.py documents).
        broken_dir = tmp_path / "adapter_v0720_shaped"
        broken_dir.mkdir()
        shutil.copy(out / "adapter_config.json", broken_dir / "adapter_config.json")
        mangled = {
            key.replace(".self_attn.", ".inner.self_attn."): value
            for key, value in _adapter_keys(str(out)).items()
        }
        save_file(mangled, str(broken_dir / "adapter_model.safetensors"))

        base = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(base, str(broken_dir))
        b_sum = sum(
            param.abs().sum().item()
            for name, param in reloaded.named_parameters()
            if "lora_B" in name
        )
        assert b_sum == 0.0, (
            "the v0.72.0-shaped keys were expected to load as nothing; if this "
            "fails the round-trip assertion above proves nothing"
        )


# --------------------------------------------------------------------------
# the fix must not touch numerics
# --------------------------------------------------------------------------
class TestFixIsNumericsNeutral:
    def test_streamed_forward_still_matches_resident_bit_exactly(self, tmp_path):
        """v0.72.0's correctness gates must remain valid after this change.

        The fix is serialisation-only precisely so that the bit-exactness
        result does not have to be re-earned; this pins that claim.
        """
        import torch
        from transformers import AutoModelForCausalLM

        model, _, weights = _build_streamed_cpu(tmp_path)
        resident = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32).eval()
        model.eval()

        ids = torch.arange(8, dtype=torch.long).unsqueeze(0) % 64
        with torch.no_grad():
            streamed_logits = model(input_ids=ids).logits
            resident_logits = resident(input_ids=ids).logits
        # lora_B is still zero here, so the adapter is a no-op and the streamed
        # base must reproduce the resident base exactly
        assert torch.equal(streamed_logits, resident_logits)

    def test_layer_zero_adapter_still_receives_gradient(self, tmp_path):
        """plan P2: a severed graph still lowers the loss, so check the grad."""
        import torch

        model, _, _ = _build_streamed_cpu(tmp_path)
        _make_adapters_detectable(model)
        ids = torch.arange(8, dtype=torch.long).unsqueeze(0) % 64
        model(input_ids=ids, labels=ids.clone()).loss.backward()

        grads = {
            name: param.grad
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name and ".layers.0." in name
        }
        assert grads, "no layer-0 adapter parameters found"
        assert any(g is not None and g.abs().max().item() > 0 for g in grads.values())


# --------------------------------------------------------------------------
# the known limitation, pinned so it cannot silently half-work
# --------------------------------------------------------------------------
class TestLoadingIntoAStreamedModelStaysUnsupported:
    def test_named_parameters_still_carry_the_wrapper_segment(self, tmp_path):
        """Fix A is serialisation-only, by design.

        In memory the wrapper is still a real module, so ``named_parameters()``
        and ``state_dict()`` disagree. v0.72.1 handled that by refusing to load
        INTO a streamed model at all; v0.72.3 instead redirects canonical keys at
        load time, which closes ``--resume`` while leaving this asymmetry — and
        therefore v0.72.0's bit-exactness gates — untouched.
        """
        model, _, _ = _build_streamed_cpu(tmp_path)
        live = [n for n, _ in model.named_parameters() if "lora_" in n]
        assert live
        assert any(".inner." in n for n in live)

    def test_hf_resume_with_streaming_is_no_longer_refused(self, tmp_path, monkeypatch):
        """v0.72.1 refused BOTH resume flags; v0.72.3 lifts both.

        The refusal existed because a streamed model's ``named_parameters()``
        carry an ``.inner.`` segment that ``load_state_dict`` narrows away, so a
        canonical checkpoint matched NOTHING and training silently continued
        with a freshly initialised adapter (measured: 0 of 12 tensors, and a
        resumed loss curve byte-identical to a from-scratch one).
        ``StreamedDecoderLayer`` now redirects canonical keys at load time,
        mirroring the save-side delegation this release added.

        Kept as the ``--hf-resume`` case specifically, because that flag reaches
        ``resume_from`` through a different branch and was the one a guard on
        ``--resume`` alone let slip through. Asserted behaviourally — a source
        grep would break on a harmless refactor and pass on a guard moved into
        dead code.
        """
        from typer.testing import CliRunner

        from soup_cli.cli import app

        weights = _tiny_llama_dir(tmp_path)
        data = tmp_path / "data.jsonl"
        data.write_text('{"text": "hello world"}\n', encoding="utf-8")
        config = tmp_path / "soup.yaml"
        config.write_text(
            f"base: {weights}\n"
            "task: sft\n"
            f"data:\n  train: {json.dumps(str(data))}\n  format: plaintext\n"
            "training:\n"
            "  stream_layers: true\n  batch_size: 1\n  quantization: none\n"
            # both default to values streaming refuses, and those gates fire
            # before the resume guard — without them this asserts the wrong refusal
            "  gradient_accumulation_steps: 1\n  epochs: 1\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app,
            ["train", "--config", str(config), "--yes",
             "--push-as", "someone/somewhere", "--hf-resume"],
        )
        assert "are not supported with" not in result.output, result.output
        assert "lands in v0.72.3" not in result.output, result.output


# --------------------------------------------------------------------------
# the PRODUCTION save path, not just the helper
# --------------------------------------------------------------------------
def _write_tiny_tokenizer(directory):
    """A real, offline PreTrainedTokenizerFast so `setup()` can tokenize."""
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


class TestProductionSavePathWritesCanonicalKeys:
    """The tests above save via `model.save_pretrained()` directly.

    Real runs save through `SFTTrainerWrapper` -> TRL `SFTTrainer` ->
    `Trainer.save_model()` -> `PeftModel.save_pretrained()`, which is also the
    `save_steps` checkpoint path. Asserting only on the helper leaves exactly
    the integration gap that hid the original v0.72.0 defect, so this drives
    the production path and then inspects the bytes on disk.
    """

    def test_trainer_save_model_writes_a_loadable_adapter(self, tmp_path, monkeypatch):
        import torch
        import yaml
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        weights = _tiny_llama_dir(tmp_path, n_layers=2)
        _write_tiny_tokenizer(weights)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": weights,
                    "task": "sft",
                    "backend": "transformers",
                    "modality": "text",
                    "data": {"train": "train.jsonl", "max_length": 64,
                             "chat_template": "chatml"},
                    "training": {
                        "batch_size": 1, "gradient_accumulation_steps": 1,
                        "quantization": "none", "stream_layers": True,
                        "epochs": 1, "logging_steps": 1, "save_steps": 1000,
                        "gradient_checkpointing": True,
                        "lora": {"r": 4, "alpha": 8,
                                 "target_modules": ["q_proj", "v_proj"]},
                    },
                    "output": str(tmp_path / "out"),
                }
            )
        )
        dataset = {
            "train": [
                {"messages": [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello world"}]}
                for _ in range(4)
            ]
        }
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper.setup(dataset)

        # make a dropped adapter detectable before the production save
        _make_adapters_detectable(wrapper.model)

        saved_dir = tmp_path / "saved"
        wrapper.trainer.save_model(str(saved_dir))

        saved = _adapter_keys(str(saved_dir))
        assert saved, "the production save path wrote no adapter tensors"
        assert [k for k in saved if ".inner." in k] == []

        base = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(base, str(saved_dir))
        b_tensors = [p for n, p in reloaded.named_parameters() if "lora_B" in n]
        assert b_tensors
        assert all(p.abs().sum().item() > 0 for p in b_tensors), (
            "adapter saved by the production path reloaded as a no-op"
        )


# --------------------------------------------------------------------------
# the renumber: every refusal must name the slot that actually lifts it
# --------------------------------------------------------------------------
def _stream_yaml(**training):
    # quantization defaults to 4bit repo-wide, which trips the NF4 gate before
    # any other streaming gate is reached — every case below would otherwise
    # assert the wrong refusal.
    fields = {"stream_layers": "true", "batch_size": 1, "quantization": "none"}
    fields.update(training)
    lines = [
        "base: hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task: sft",
        "data:",
        "  train: data.jsonl",
        "training:",
    ]
    lines += [f"  {key}: {value}" for key, value in fields.items()]
    return "\n".join(lines) + "\n"


class TestRefusalsNameThePostRenumberSlot:
    def test_shipped_slots_no_longer_refuse(self):
        """Refusals are deleted as their slot ships, so this shrinks rather than
        grows. NF4 went in v0.72.2; larger batches, gradient accumulation,
        resume, the architecture allowlist and the disk tier all went in
        v0.72.3. What remains is the pair that has NOT shipped, below."""
        from soup_cli.config.loader import load_config_from_string

        for field, value in (
            ("stream_source", "disk"),
            ("batch_size", "2"),
            ("gradient_accumulation_steps", "4"),
            ("quantization", "4bit"),
        ):
            load_config_from_string(_stream_yaml(**{field: value}))

    def test_preference_losses_shipped_in_v0724(self):
        """This refusal is GONE, per the shrink-not-grow rule above. What
        replaced it is a PERMANENT exclusion for the rollout tasks, which must
        NOT name a release — a version number there would read as "coming soon"
        to the next maintainer and invite them to wire up something that cannot
        work."""
        from soup_cli.config.loader import load_config_from_string

        def _yaml(task):
            return (
                "base: hf-internal-testing/tiny-random-LlamaForCausalLM\n"
                f"task: {task}\n"
                "data:\n  train: data.jsonl\n"
                "training:\n  stream_layers: true\n  batch_size: 1\n"
            )

        assert load_config_from_string(_yaml("dpo")).task == "dpo"
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(_yaml("grpo"))
        assert "v0.72" not in str(excinfo.value)

    def test_arch_allowlist_names_v0723(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        class _Cfg:
            model_type = "gpt2"

        with pytest.raises(ValueError, match="v0.72.3"):
            stream_arch_of(_Cfg())
