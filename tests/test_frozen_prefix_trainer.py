"""Training with the frozen-prefix cache matches training without it (CPU, tiny model)."""

import copy
import re
from unittest.mock import patch

import torch
from datasets import Dataset
from peft import get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_prefix_configuration,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner
from soup_cli.utils.peft_wiring import build_lora_config

CUTOFF = 2


def make_model():
    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    base = MistralForCausalLM(config)
    lora = LoraConfig(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
    # The layers the freeze plan leaves trainable: everything from the cutoff up.
    peft_config = build_lora_config(
        lora,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(CUTOFF, config.num_hidden_layers)),
    )
    return get_peft_model(base, peft_config)


def make_tokenizer():
    backend = Tokenizer(
        WordLevel(
            vocab={
                "[PAD]": 0,
                "[UNK]": 1,
                "[EOS]": 2,
                **{f"token_{index}": index for index in range(3, 100)},
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )


def make_trainer(model, output_dir, tokenizer):
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "labels": [[1, 2, 3, 4, 5]],
        }
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        packing=False,
        use_cache=False,
        use_cpu=True,
        save_strategy="no",
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=16,
        logging_steps=1,
        learning_rate=2e-5,
        seed=42,
        data_seed=42,
        optim="adamw_torch",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    return SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer
    )


def trainable_weights(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_adapter_sits_only_above_the_cutoff():
    names = list(trainable_weights(make_model()))

    assert names
    for name in names:
        match = re.search(r"\.layers\.(\d+)\.", name)
        assert match is not None and int(match.group(1)) >= CUTOFF, name


def test_cached_gradients_equal_plain_gradients(tmp_path):
    torch.manual_seed(42)
    plain = make_model()
    cached = copy.deepcopy(plain)
    plain.train()
    cached.train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    decoder = cached.base_model.model.model

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny",
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=fingerprint_prefix_configuration(decoder, CUTOFF),
            cutoff=CUTOFF,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=CUTOFF,
        metadata_factory=metadata_factory,
    )

    plain_loss = plain(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    plain_loss.backward()

    with patch.object(decoder, "forward", side_effect=runner.forward):
        for _ in range(2):  # first a miss, then a hit
            cached.zero_grad(set_to_none=True)
            cached_loss = cached(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            cached_loss.backward()

    assert (runner.misses, runner.hits) == (1, 1)
    torch.testing.assert_close(plain_loss, cached_loss, rtol=1e-5, atol=1e-6)
    plain_grads = {n: p.grad for n, p in plain.named_parameters() if p.requires_grad}
    cached_grads = {n: p.grad for n, p in cached.named_parameters() if p.requires_grad}
    assert plain_grads.keys() == cached_grads.keys() and plain_grads
    for name, grad in plain_grads.items():
        assert grad is not None and cached_grads[name] is not None, name
        torch.testing.assert_close(grad, cached_grads[name], rtol=1e-4, atol=1e-6)


def test_normal_and_cached_sft_trainer_agree(tmp_path):
    torch.manual_seed(42)
    plain_model = make_model()
    cached_model = copy.deepcopy(plain_model)
    tokenizer = make_tokenizer()
    initial = trainable_weights(plain_model)
    assert initial

    plain_trainer = make_trainer(plain_model, tmp_path / "plain", tokenizer)
    plain_result = plain_trainer.train()

    decoder = cached_model.base_model.model.model
    configuration = fingerprint_prefix_configuration(decoder, CUTOFF)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny",
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=configuration,
            cutoff=CUTOFF,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=CUTOFF,
        metadata_factory=metadata_factory,
    )
    cached_trainer = make_trainer(cached_model, tmp_path / "cached", tokenizer)

    frozen_calls = [0]
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args: frozen_calls.__setitem__(0, frozen_calls[0] + 1)
        )
        for layer in decoder.layers[:CUTOFF]
    ]
    try:
        with patch.object(decoder, "forward", side_effect=runner.forward):
            cached_result = cached_trainer.train()
    finally:
        for handle in handles:
            handle.remove()

    assert plain_trainer.state.global_step == cached_trainer.state.global_step == 2
    assert (runner.misses, runner.hits) == (1, 1)
    assert frozen_calls[0] == CUTOFF, "each frozen layer runs once, in the epoch that fills it"

    plain_weights = trainable_weights(plain_model)
    cached_weights = trainable_weights(cached_model)
    assert plain_weights.keys() == cached_weights.keys()
    changed = 0
    for name, weight in plain_weights.items():
        torch.testing.assert_close(weight, cached_weights[name], rtol=1e-4, atol=1e-6)
        if not torch.equal(initial[name], weight):
            changed += 1
    assert changed > 0, "no adapter tensor was updated"
    torch.testing.assert_close(
        torch.tensor(plain_result.training_loss),
        torch.tensor(cached_result.training_loss),
        rtol=1e-5,
        atol=1e-6,
    )
