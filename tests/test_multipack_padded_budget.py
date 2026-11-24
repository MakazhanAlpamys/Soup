"""Bound the real padded text batch, not just its non-padding token sum (#3218)."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import pytest

from soup_cli.utils.multipack_sampler import MultipackBatchSampler, ffd_bin_pack
from soup_cli.utils.multipack_trainer import (
    attach_multipack_state,
    make_multipack_trainer_class,
)


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory = tmp_path_factory.mktemp("multipack-base")
    vocabulary = {"<unk>": 0, "<pad>": 1, "</s>": 2, "x": 3}
    vocabulary.update({f"word{index}": index for index in range(4, 32)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>",
    )
    tokenizer.save_pretrained(directory)
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
        pad_token_id=1, eos_token_id=2, attention_dropout=0.0,
    ))
    model.save_pretrained(directory)
    return directory


def _rows(lengths):
    return [
        {"input_ids": [4 + index % 28] * length,
         "attention_mask": [1] * length,
         "labels": [-100] * (length // 2) + [4 + index % 28] * (length - length // 2)}
        for index, length in enumerate(lengths)
    ]


def _build_wrapper(tmp_path, monkeypatch, tiny_base, *, task="sft", lengths=None,
                   max_length=65, padding_multiple=None, accumulation=1, raft_shuffle=None,
                   multipack=True):
    from datasets import Dataset
    from trl import SFTConfig

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer import pretrain, sft

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.chdir(tmp_path)
    rows = _rows(lengths if lengths is not None else [65] + [2] * 32)
    if raft_shuffle is None:
        cache = tmp_path / "tokens"
        Dataset.from_list(rows).save_to_disk(str(cache))
        data_options = {"format": "pre_tokenized", "tokenized_path": "tokens"}
    else:
        data_options = {"format": "raft", "raft_epoch_shuffle": raft_shuffle}
        rows = [
            {"query": "x", "golden_doc": "x x", "distractor_docs": ["x"],
             "answer": " ".join(["x"] * count)}
            for count in (1, 5)
        ]
    original = sft.SFTTrainerWrapper._as_sft_config

    def cpu_config(arguments, max_length, packing=False):
        configured = original(arguments, max_length, packing=packing)
        values = configured.to_dict()
        values.update(use_cpu=True, bf16=False, fp16=False, pad_to_multiple_of=padding_multiple)
        return SFTConfig(**values)

    monkeypatch.setattr(sft.SFTTrainerWrapper, "_as_sft_config", staticmethod(cpu_config))
    config = SoupConfig(
        base=str(tiny_base), task=task,
        data={"train": "unused.jsonl", "max_length": max_length, "val_split": 0.0,
              **data_options},
        training={"batch_size": 2, "gradient_accumulation_steps": accumulation,
                  "quantization": "none", "multipack": multipack,
                  "gradient_checkpointing": False, "warmup_ratio": 0.0,
                  "seed": 7, "lora": {"r": 2, "alpha": 4, "dropout": 0.0,
                                        "target_modules": ["q_proj", "v_proj"]}},
        output=str(tmp_path / "output"),
    )
    wrapper_class = sft.SFTTrainerWrapper if task == "sft" else pretrain.PretrainTrainerWrapper
    wrapper = wrapper_class(config, device="cpu")
    wrapper.setup({"train": rows})
    return wrapper


@pytest.mark.parametrize("task", ["sft", "pretrain"])
@pytest.mark.parametrize("padding_multiple", [None, 8])
def test_real_wrappers_bound_collated_positions(tmp_path, monkeypatch, tiny_base,
                                              task, padding_multiple):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    wrapper = _build_wrapper(
        tmp_path, monkeypatch, tiny_base, task=task, padding_multiple=padding_multiple,
    )
    trainer = wrapper.trainer
    assert isinstance(trainer.data_collator, DataCollatorForLanguageModeling)
    assert trainer.data_collator.pad_to_multiple_of == padding_multiple
    assert not trainer.padding_free
    loader = trainer.get_train_dataloader()
    batches = list(loader)
    assert max(batch["input_ids"].numel() for batch in batches) <= 130
    assert len(loader) == len(batches) == (2 if padding_multiple is None else 3)
    assert sum(batch["attention_mask"].sum().item() for batch in batches) == 129
    assert sorted(index for batch in loader.batch_sampler for index in batch) == list(range(33))
    for batch, indices in zip(batches, loader.batch_sampler, strict=True):
        assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
        assert (batch["labels"][:, 0] == -100).all()
        for row_position, row_index in enumerate(indices):
            original = trainer.train_dataset[row_index]
            labels = batch["labels"][row_position, :len(original["labels"])].tolist()
            assert labels == original["labels"]


@pytest.mark.parametrize("task", ["sft", "pretrain"])
@pytest.mark.parametrize("padding_multiple", [None, 8])
@pytest.mark.parametrize("lengths", [[66, 64], [66, 65, 64]], ids=["truncated", "exact-boundary"])
def test_real_wrappers_price_prepared_rows(tmp_path, monkeypatch, tiny_base, task,
                                         padding_multiple, lengths):
    wrapper = _build_wrapper(
        tmp_path, monkeypatch, tiny_base, task=task, lengths=lengths,
        padding_multiple=padding_multiple,
    )
    trainer = wrapper.trainer
    prepared_lengths = [len(row["input_ids"]) for row in trainer.train_dataset]
    assert prepared_lengths == [min(length, 65) for length in lengths]
    loader = trainer.get_train_dataloader()
    batches = list(loader)
    expected_count = len(lengths) if padding_multiple == 8 else len(lengths) - 1
    assert len(loader) == len(batches) == expected_count
    assert loader.batch_sampler._lengths == prepared_lengths
    assert sorted(index for indices in loader.batch_sampler for index in indices) == list(
        range(len(lengths)),
    )
    assert sum(batch["attention_mask"].sum().item() for batch in batches) == sum(prepared_lengths)
    assert max(batch["input_ids"].numel() for batch in batches) <= 130
    if padding_multiple == 8:
        assert any(batch["input_ids"].shape[-1] == 72 for batch in batches)
        assert all(batch["input_ids"].shape[0] == 1 for batch in batches)
    else:
        assert any(batch["input_ids"].shape == (2, 65) for batch in batches)


def test_reported_512_token_fixture(tmp_path, tiny_base):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    trainer = make_multipack_trainer_class(SFTTrainer)(
        model=AutoModelForCausalLM.from_pretrained(tiny_base),
        processing_class=AutoTokenizer.from_pretrained(tiny_base),
        train_dataset=Dataset.from_list(_rows([512] + [2] * 256)),
        args=SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
                       report_to="none", packing=False, max_length=512,
                       per_device_train_batch_size=2),
    )
    attach_multipack_state(trainer, lengths=[512] + [2] * 256, max_seq_len=512, batch_size=2)
    batches = list(trainer.get_train_dataloader())
    assert max(batch["input_ids"].numel() for batch in batches) == 1024
    assert len(batches) == 2
    assert sum(batch["attention_mask"].sum().item() for batch in batches) == 1024


@pytest.mark.parametrize("multiple", [1, 8])
@pytest.mark.parametrize("drop_last", [False, True])
def test_sampler_caches_bounded_batches_with_exact_coverage(multiple, drop_last):
    lengths = [17] + [2] * 8
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=34, batch_size=2, real_batches=False, seed=7,
        drop_last=drop_last, padding_multiple=multiple,
    )
    batches = list(sampler)
    expected = [[0, 1], list(range(2, 9))] if multiple == 1 else [[0], [1, 2, 3, 4], [5, 6, 7, 8]]
    random.Random(7).shuffle(expected)
    assert batches == expected
    assert len(sampler) == len(batches)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert list(sampler) == batches
    batches[0].clear()
    assert list(sampler) == expected


@pytest.mark.parametrize("multiple", [1, 3, 8])
def test_sampler_random_rectangles_and_legacy_ffd_unchanged(multiple):
    lengths = [random.Random(seed).randint(2, 31) for seed in range(200)]
    legacy = ffd_bin_pack(lengths, 64)
    sampler = MultipackBatchSampler(
        lengths, 64, 2, real_batches=False, seed=13, padding_multiple=multiple,
    )
    for batch in sampler:
        width = ((max(lengths[index] for index in batch) + multiple - 1) // multiple) * multiple
        assert len(batch) * width <= 64
    assert sorted(index for batch in sampler for index in batch) == list(range(len(lengths)))
    random.Random(13).shuffle(legacy)
    assert list(MultipackBatchSampler(lengths, 64, 2, real_batches=False, seed=13)) == legacy


@pytest.mark.parametrize("multiple,error", [(True, TypeError), (1.5, TypeError),
                                          (0, ValueError), (-1, ValueError)])
def test_invalid_padding_multiple_is_rejected(multiple, error):
    with pytest.raises(error, match="padding_multiple"):
        MultipackBatchSampler([2], 8, 1, real_batches=False, padding_multiple=multiple)


def test_single_row_rounded_past_budget_is_rejected():
    with pytest.raises(ValueError, match=r"padded lengths\[0\]=24 exceeds batch_max_len=17"):
        MultipackBatchSampler([17], 17, 1, real_batches=False, padding_multiple=8)


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_unequal_microbatch_accumulation_matches_reference(tmp_path, monkeypatch, tiny_base, task):
    import torch

    wrapper = _build_wrapper(tmp_path, monkeypatch, tiny_base, task=task, accumulation=2)
    trainer = wrapper.trainer
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 2
    assert batches[0]["input_ids"].shape[0] != batches[1]["input_ids"].shape[0]
    trainer.model.eval()
    reference = copy.deepcopy(trainer.model)
    all_rows = [trainer.train_dataset[index] for index in range(len(trainer.train_dataset))]
    reference_batch = trainer.data_collator(all_rows)
    total_tokens = (reference_batch["labels"][:, 1:] != -100).sum().item()
    expected = trainer.compute_loss(
        reference, dict(reference_batch), num_items_in_batch=total_tokens,
    )
    expected.backward()
    actual = torch.zeros(())
    for batch in batches:
        loss = trainer.compute_loss(trainer.model, dict(batch), num_items_in_batch=total_tokens)
        actual += loss.detach()
        loss.backward()
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    reference_gradients = dict(reference.named_parameters())
    nonzero_gradients = 0
    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_gradients[name].grad,
                                       rtol=1e-4, atol=1e-6)
            nonzero_gradients += bool(parameter.grad.count_nonzero())
    assert nonzero_gradients > 0
    reference_optimizer = torch.optim.SGD(
        [parameter for parameter in reference.parameters() if parameter.requires_grad], lr=0.1,
    )
    reference_optimizer.step()
    trainer.model.zero_grad()
    trainer.optimizer = torch.optim.SGD(
        [parameter for parameter in trainer.model.parameters() if parameter.requires_grad], lr=0.1,
    )
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.0)
    trainer.args.max_steps = 1
    trainer.args.max_grad_norm = 0.0
    trainer.args.save_strategy = "no"
    trainer.args.disable_tqdm = True
    result = trainer.train()
    assert result.global_step == 1
    assert result.training_loss == pytest.approx(expected.item(), rel=1e-6)
    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, reference_gradients[name], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("epoch_shuffle", [False, True])
def test_raft_keeps_its_existing_trainer_and_collation(tmp_path, monkeypatch, tiny_base,
                                                    epoch_shuffle):
    from soup_cli.trainer.raft import RaftDataCollator, RaftEpochShuffleCollator

    wrapper = _build_wrapper(tmp_path, monkeypatch, tiny_base, max_length=128,
                             raft_shuffle=epoch_shuffle)
    trainer = wrapper.trainer
    expected_type = RaftEpochShuffleCollator if epoch_shuffle else RaftDataCollator
    assert isinstance(trainer.data_collator, expected_type)
    assert not getattr(trainer, "soup_multipack", False)
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 2
    batch = batches[0]
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["loss_weights"][batch["attention_mask"] == 0] == 0).all()


def _plain_loader(collator, lengths, features=None):
    trainer = make_multipack_trainer_class(SimpleNamespace)(
        train_dataset=_rows(lengths) if features is None else features,
        data_collator=collator,
        args=SimpleNamespace(dataloader_drop_last=False, dataloader_num_workers=0,
                             dataloader_pin_memory=False),
    )
    attach_multipack_state(trainer, lengths=lengths, max_seq_len=65, batch_size=2, seed=7)
    return trainer.get_train_dataloader()


def test_padding_free_collator_keeps_sum_based_batches():
    pytest.importorskip("trl")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    loader = _plain_loader(
        DataCollatorForLanguageModeling(pad_token_id=1, padding_free=True), [65] + [2] * 32,
    )
    assert len(loader) == 1
    assert next(iter(loader))["input_ids"].shape == (1, 129)


def test_custom_collator_with_similar_attributes_is_not_reinterpreted():
    pytest.importorskip("trl")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    class CustomCollator:
        padding_free = False
        pad_to_multiple_of = 8

        def __call__(self, features):
            return DataCollatorForLanguageModeling(pad_token_id=1)(features)

    loader = _plain_loader(CustomCollator(), [65] + [2] * 32)
    assert len(loader) == 1
    assert next(iter(loader))["input_ids"].shape == (33, 65)


def test_custom_collator_does_not_require_optional_trl(monkeypatch):
    import sys

    pytest.importorskip("torch")
    collator = SimpleNamespace(padding_free=False)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", None)
    loader = _plain_loader(collator, [65] + [2] * 32)
    assert list(loader.batch_sampler) == [list(range(33))]


def test_vision_collator_keeps_its_existing_sum_based_admission(tiny_base):
    import torch
    from transformers import AutoTokenizer

    from soup_cli.trainer.sft import VisionLanguageDataCollator

    class Processor:
        tokenizer = AutoTokenizer.from_pretrained(tiny_base)

        def apply_chat_template(self, messages, **kwargs):
            return " ".join(part["text"] for message in messages for part in message["content"])

        def __call__(self, *, text, images, **kwargs):
            batch = self.tokenizer(text, **kwargs)
            batch["pixel_values"] = torch.zeros(len(images), 3, 1, 1)
            return batch

    lengths = [65] + [2] * 32
    features = _rows(lengths)
    for feature, length in zip(features, lengths, strict=True):
        feature["messages"] = [{"role": "user", "content": " ".join(["x"] * length)}]
    collator = VisionLanguageDataCollator(Processor(), max_length=65)
    loader = _plain_loader(collator, lengths, features)
    assert loader.collate_fn is collator
    assert len(loader) == 1
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (33, 65)
    assert batch["pixel_values"].shape[0] == 33
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
