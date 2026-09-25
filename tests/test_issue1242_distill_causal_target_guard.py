"""Regression tests for Issue #1242:
`task: distill` refuses rows with no loss target at setup with human-facing row number,
matching SFT behavior, and compute_loss returns finite zero when a microbatch has no targets.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.distill import DistillTrainerWrapper


def _tiny_llama_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "user": 4,
        "assistant": 5,
        "hi": 6,
        "good": 7,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        chat_template=(
            "{% for message in messages %}"
            "{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}"
        ),
    )
    fast.save_pretrained(str(weights))
    return weights


def _make_cfg(
    weights: pathlib.Path,
    tmp_path: pathlib.Path,
    *,
    divergence: str = "forward_kl",
    max_length: int = 64,
    gradient_accumulation_steps: int = 1,
    train_on_messages_with_train_field: bool = False,
) -> str:
    train_field_opt = (
        "  train_on_responses_only: false\n  train_on_messages_with_train_field: true\n"
        if train_on_messages_with_train_field
        else ""
    )
    return f"""
base: {weights.as_posix()}
task: distill
backend: transformers
data:
  train: train.jsonl
  max_length: {max_length}
{train_field_opt}training:
  teacher_model: {weights.as_posix()}
  distill_divergence: {divergence}
  batch_size: 1
  epochs: 1
  gradient_accumulation_steps: {gradient_accumulation_steps}
  lora:
    r: 4
    alpha: 8
    target_modules: ['q_proj', 'v_proj']
output: {tmp_path.as_posix()}/out
"""


SHORT_ROW = {
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "good"},
    ]
}

# 96 tokens when tokenized by whitespace;
# prompt truncation at max_length=64 drops the assistant response
LONG_ROW = {
    "messages": [
        {"role": "user", "content": " ".join(["hi"] * 96)},
        {"role": "assistant", "content": "good"},
    ]
}


class TestDistillCausalTargetGuard:
    def test_distill_setup_refuses_train_row_with_no_loss_target(
        self, tmp_path: pathlib.Path
    ) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(_make_cfg(weights, tmp_path, max_length=64))
        wrapper = DistillTrainerWrapper(cfg, device="cpu")

        with pytest.raises(
            ValueError,
            match=r"train row 2.*no causal-loss target.*data\.max_length=64",
        ):
            wrapper.setup({"train": [SHORT_ROW, LONG_ROW]})

    def test_distill_setup_refuses_val_row_with_no_loss_target(
        self, tmp_path: pathlib.Path
    ) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(_make_cfg(weights, tmp_path, max_length=64))
        wrapper = DistillTrainerWrapper(cfg, device="cpu")

        with pytest.raises(
            ValueError,
            match=r"val row 2.*no causal-loss target.*data\.max_length=64",
        ):
            wrapper.setup({"train": [SHORT_ROW], "val": [SHORT_ROW, LONG_ROW]})

    def test_control_surviving_target_is_accepted(self, tmp_path: pathlib.Path) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(_make_cfg(weights, tmp_path, max_length=64))
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [SHORT_ROW]})
        assert wrapper.trainer is not None

    @pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
    @pytest.mark.parametrize("ga", [1, 4])
    def test_divergences_and_gradient_accumulation(
        self, tmp_path: pathlib.Path, divergence: str, ga: int
    ) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(
            _make_cfg(
                weights,
                tmp_path,
                divergence=divergence,
                max_length=64,
                gradient_accumulation_steps=ga,
            )
        )
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        with pytest.raises(
            ValueError,
            match=r"train row 2.*no causal-loss target.*data\.max_length=64",
        ):
            wrapper.setup({"train": [SHORT_ROW, LONG_ROW]})

    def test_train_on_messages_with_train_field_all_false(self, tmp_path: pathlib.Path) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(
            _make_cfg(
                weights,
                tmp_path,
                max_length=64,
                train_on_messages_with_train_field=True,
            )
        )
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        row_all_false = {
            "messages": [
                {"role": "user", "content": "hi", "train": False},
                {"role": "assistant", "content": "good", "train": False},
            ]
        }
        with pytest.raises(
            ValueError,
            match=r"train row 1.*no causal-loss target",
        ):
            wrapper.setup({"train": [row_all_false]})


class TestComputeLossDefenseInDepth:
    """Defense in depth: compute_loss evaluates to finite 0.0 even if a microbatch has 0 targets."""

    def test_all_masked_labels_return_finite_loss(self, tmp_path: pathlib.Path) -> None:
        weights = _tiny_llama_dir(tmp_path)
        cfg = load_config_from_string(_make_cfg(weights, tmp_path, max_length=64))
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [SHORT_ROW]})

        # Manually invoke compute_loss with all -100 labels
        inputs = {
            "input_ids": torch.tensor([[4, 6, 5, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, -100]], dtype=torch.long),
        }
        loss = wrapper.trainer.compute_loss(wrapper.model, inputs)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss.item() == 0.0
