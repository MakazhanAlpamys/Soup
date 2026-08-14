"""#369: a streamed model's ``named_parameters()`` still carries the wrapper's
``.inner.`` segment, and a name-keyed comparison against a resident model of
the same checkpoint saw an empty intersection.

v0.72.1 made ``state_dict()`` canonical (serialisation only, deliberately) but
left ``named_parameters()``/``named_modules()`` untouched. The gap looked
harmless until the #331 repair gate at 32B compared streamed and resident
gradients by name and read ``grads exact 0/0`` as success
(``benchmarks/gate-h100-validation.md`` L2627-2643): the two key sets did not
intersect, so ``n_exact == n_compared`` was vacuously true on nothing
compared.

Ships fix shape (B) from the issue: ``canonical_named_parameters()`` strips
the ``.inner.`` segment the same way ``state_dict()`` already does, and
``assert_canonical_parameters_intersect()`` gives a future comparison a
primitive that raises on an empty intersection instead of reporting it as a
pass. (A), making ``named_parameters()`` itself canonical, would also
require re-running the v0.72.0/.2/.3 bit-exactness suites on real hardware,
which this sandbox cannot do; (B) touches neither the forward nor the
``state_dict`` path, so those gates stay valid unexercised, the same way they
did for the v0.72.1 fix.
"""

import json
import os

import pytest


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "safetensors", "tokenizers", "yaml"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


# --------------------------------------------------------------------------
# fixtures: duplicated from tests/test_v07204.py, trimmed to the dpo task
# --------------------------------------------------------------------------
def _tiny_llama_dir(tmp_path, n_layers=2, hidden=64, vocab=64):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
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
    return str(weights), model


def _write_tiny_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
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


def _lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed_dpo_wrapper(tmp_path, monkeypatch):
    """The real ``setup()`` path, streaming, task=dpo, device=cpu."""
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.dpo import DPOTrainerWrapper

    weights, resident = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "dpo",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "stream_layers": True,
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = DPOTrainerWrapper(cfg, device="cpu")
    rows = [{"prompt": "hi", "chosen": " good answer", "rejected": " bad"} for _ in range(4)]
    wrapper.setup({"train": rows})
    return wrapper, resident


# --------------------------------------------------------------------------
# the bug and the fix
# --------------------------------------------------------------------------
class TestCanonicalNamedParameters:
    def test_control_raw_named_parameters_do_not_intersect_the_resident_peft_model(
        self, tmp_path, monkeypatch
    ):
        """CONTROL: proves the bug this issue reports, and that the fix below
        is not a no-op. Same checkpoint, same LoRA config, one streamed and
        one not: every trainable (LoRA) name lives inside a wrapped decoder
        layer, so the trainable key sets share nothing, reproducing the
        #331 gate's "grads exact 0/0", since gradients only exist for the
        trainable parameters. (Non-layer weights like the embeddings already
        match without canonicalising; only per-layer names carry `.inner.`,
        so the assertion is scoped to the LoRA names the gate actually
        compared.)"""
        _requires_train_extra()
        from peft import get_peft_model

        wrapper, resident = _build_streamed_dpo_wrapper(tmp_path, monkeypatch)
        resident_peft = get_peft_model(resident, _lora_config())

        streamed_lora = {name for name, _ in wrapper.model.named_parameters() if "lora_" in name}
        resident_lora = {name for name, _ in resident_peft.named_parameters() if "lora_" in name}
        assert streamed_lora and resident_lora
        assert any(".inner." in name for name in streamed_lora)
        assert not any(".inner." in name for name in resident_lora)
        assert not (streamed_lora & resident_lora)

    def test_canonical_names_strip_the_inner_segment(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

        wrapper, _ = _build_streamed_dpo_wrapper(tmp_path, monkeypatch)
        raw_names = [name for name, _ in wrapper.model.named_parameters()]
        canonical_names = [name for name, _ in canonical_named_parameters(wrapper.model)]
        assert any(".inner." in name for name in raw_names)
        assert not any(".inner." in name for name in canonical_names)
        assert len(canonical_names) == len(raw_names)

    def test_streamed_and_resident_canonical_names_intersect(self, tmp_path, monkeypatch):
        """Acceptance criterion: a streamed and a resident model built from
        the same checkpoint (same LoRA config) produce intersecting
        parameter-name sets once both sides are read canonically."""
        _requires_train_extra()
        from peft import get_peft_model

        from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

        wrapper, resident = _build_streamed_dpo_wrapper(tmp_path, monkeypatch)
        resident_peft = get_peft_model(resident, _lora_config())

        streamed_names = {name for name, _ in canonical_named_parameters(wrapper.model)}
        resident_names = {name for name, _ in resident_peft.named_parameters()}
        assert streamed_names == resident_names


class TestAssertCanonicalParametersIntersect:
    def test_raises_on_empty_intersection(self):
        """Acceptance criterion: the comparison helper raises rather than
        reporting an empty (0/0) comparison as a pass."""
        _requires_train_extra()
        import torch
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import assert_canonical_parameters_intersect

        a = nn.Module()
        a.x = nn.Parameter(torch.zeros(1))
        b = nn.Module()
        b.y = nn.Parameter(torch.zeros(1))
        with pytest.raises(ValueError, match="no canonical parameter name"):
            assert_canonical_parameters_intersect(a, b)

    def test_returns_the_shared_canonical_names_when_they_intersect(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from peft import get_peft_model

        from soup_cli.utils.layer_stream_runtime import assert_canonical_parameters_intersect

        wrapper, resident = _build_streamed_dpo_wrapper(tmp_path, monkeypatch)
        resident_peft = get_peft_model(resident, _lora_config())

        shared = assert_canonical_parameters_intersect(wrapper.model, resident_peft)
        assert shared
        assert all(".inner." not in name for name in shared)
        assert any("lora_" in name for name in shared)
