"""Regression coverage for #793: RoPE must be configured before model construction."""

import json
from copy import deepcopy

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _requires_train_extra() -> None:
    for module in ("torch", "transformers", "peft"):
        pytest.importorskip(module, reason=f"{module} is only in the [train] extra")


def _write_tiny_tokenizer(directory: str) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "hello": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(f"{directory}/tokenizer.json")
    with open(f"{directory}/tokenizer_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 64,
            },
            handle,
        )


@pytest.fixture
def tiny_llama(tmp_path) -> str:
    _requires_train_extra()
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=500_000.0,
    )
    model_dir = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(model_dir)
    _write_tiny_tokenizer(str(model_dir))
    return str(model_dir)


def _config(base: str, *, task: str = "sft", rope_type: str = "linear", **training):
    training_config = {
        "batch_size": 1,
        "quantization": "none",
        "rope_scaling_type": rope_type,
        "lora": {"r": 0 if task == "sft" else 2, "target_modules": ["q_proj"]},
    }
    training_config.update(training)
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "backend": "transformers",
                "data": {"train": "train.jsonl", "max_length": 256},
                "training": training_config,
            }
        )
    )


def _rotary_embedding(model):
    matches = [module.rotary_emb for module in model.modules() if hasattr(module, "rotary_emb")]
    assert matches
    return matches[0]


def test_sft_constructs_the_model_with_scaled_rope(tiny_llama) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.trainer.sft import SFTTrainerWrapper

    baseline = AutoModelForCausalLM.from_pretrained(tiny_llama)
    baseline_inv_freq = _rotary_embedding(baseline).inv_freq.clone()

    config = _config(tiny_llama)
    wrapper = SFTTrainerWrapper(config, device="cpu")
    wrapper._setup_transformers(config, config.training)
    rotary = _rotary_embedding(wrapper.model)

    assert rotary.rope_type == "linear"
    assert not torch.equal(rotary.inv_freq, baseline_inv_freq)
    assert wrapper.model.config.rope_parameters["rope_type"] == "linear"


def test_sft_forwards_yarn_tunables_before_construction(tiny_llama) -> None:
    from soup_cli.trainer.sft import SFTTrainerWrapper

    config = _config(
        tiny_llama,
        rope_type="yarn",
        yarn_factor=4.0,
        yarn_attn_factor=1.5,
        yarn_beta_fast=24,
        yarn_beta_slow=2,
    )
    wrapper = SFTTrainerWrapper(config, device="cpu")
    wrapper._setup_transformers(config, config.training)

    params = wrapper.model.config.rope_parameters
    assert params["factor"] == 4.0
    assert params["attention_factor"] == 1.5
    assert params["beta_fast"] == 24
    assert params["beta_slow"] == 2
    assert _rotary_embedding(wrapper.model).attention_scaling == 1.5


def test_scaled_model_save_reload_preserves_rope_theta(tiny_llama, tmp_path) -> None:
    from transformers import AutoConfig

    from soup_cli.trainer.sft import SFTTrainerWrapper

    config = _config(tiny_llama)
    wrapper = SFTTrainerWrapper(config, device="cpu")
    wrapper._setup_transformers(config, config.training)
    output = tmp_path / "saved"
    wrapper.model.save_pretrained(output)

    reloaded = AutoConfig.from_pretrained(output)
    assert reloaded.rope_parameters["rope_theta"] == 500_000.0
    assert reloaded.rope_parameters["rope_type"] == "linear"


def test_pretrain_constructs_the_model_with_scaled_rope(tiny_llama) -> None:
    from soup_cli.trainer.pretrain import PretrainTrainerWrapper

    config = _config(tiny_llama, task="pretrain")
    wrapper = PretrainTrainerWrapper(config, device="cpu")
    wrapper._setup_transformers(config, config.training)

    assert wrapper.model.config.rope_parameters["rope_type"] == "linear"
    assert _rotary_embedding(wrapper.model).rope_type == "linear"


def test_longrope_is_refused_on_a_checkpoint_without_factor_vectors() -> None:
    from transformers import LlamaConfig

    from soup_cli.utils.long_context import apply_long_context_config

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    # #1239: the vectors exist only on already-LongRoPE-scaled checkpoints, which
    # may not be extended either, so the refusal no longer suggests finding one.
    with pytest.raises(ValueError, match="exist only on checkpoints already scaled with LongRoPE"):
        apply_long_context_config(config, 256, "longrope")


def test_nested_gemma3_rope_parameters_are_refused_before_any_mutation() -> None:
    _requires_train_extra()
    from transformers import Gemma3TextConfig

    from soup_cli.utils.long_context import apply_long_context_config

    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    before = deepcopy(config.rope_parameters)

    with pytest.raises(ValueError, match="nested rope_parameters are not supported safely"):
        apply_long_context_config(config, 256, "linear")

    assert config.max_position_embeddings == 64
    assert config.rope_parameters == before


def test_switching_rope_type_drops_foreign_algorithm_tunables() -> None:
    from types import SimpleNamespace

    from soup_cli.utils.long_context import apply_long_context_config

    # #1239: a native llama3 block is no longer replaced (linear on top of it is
    # refused, see test_issue1239_llama3_rope_compose.py), so a type switch now
    # starts from an unscaled block. It must still shed another algorithm's keys.
    config = SimpleNamespace(
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
            "rope_theta": 500_000.0,
            "partial_rotary_factor": 0.5,
        },
    )

    applied = apply_long_context_config(config, 256, "linear")

    assert applied == {
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.5,
        "rope_type": "linear",
        "factor": 4.0,
    }
    assert config.rope_parameters == applied
    assert config.max_position_embeddings == 256
