"""The KTO window loss ships as code; ``training.l2l`` does not reach it yet.

KTO did not pass the readiness rule (its run is not verified to match gradient accumulation),
so the schema refuses ``task: kto`` under ``training.l2l``. The window reader and the trainer
checks stay importable and tested at the unit level.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _kl_args(**overrides):
    values = {"dataset_num_proc": None, "world_size": 1}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("num_proc", [None, 1])
def test_kl_blocks_that_match_the_micro_batches_are_accepted(num_proc):
    from torch.utils.data import SequentialSampler

    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    refuse_unaligned_kl_blocks(_kl_args(dataset_num_proc=num_proc))
    refuse_unaligned_kl_blocks(
        _kl_args(dataset_num_proc=num_proc), SequentialSampler(range(4))
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dataset_num_proc": 2}, "dataset_num_proc"),
        ({"dataset_num_proc": 8}, "dataset_num_proc"),
        ({"world_size": 2}, "process"),
    ],
)
def test_kl_blocks_that_may_not_match_the_micro_batches_are_refused(overrides, match):
    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    with pytest.raises(ValueError, match=match):
        refuse_unaligned_kl_blocks(_kl_args(**overrides))


def test_a_sampler_that_does_not_keep_the_row_order_is_refused():
    from torch.utils.data import RandomSampler

    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    with pytest.raises(ValueError, match="SequentialSampler"):
        refuse_unaligned_kl_blocks(_kl_args(), RandomSampler(range(4)))


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("loss_type", "apo_zero_unpaired", "loss_type"),
        ("calculate_KL", False, "calculate_KL"),
        ("ref_model", object(), "ref_model"),
        ("precompute_ref_log_probs", True, "precompute_ref_log_probs"),
        ("ref_adapter_name", "ref", "ref_adapter_name"),
        ("aux_loss_enabled", True, "aux_loss_enabled"),
    ],
)
def test_a_trainer_that_departs_from_plain_kto_is_refused(attribute, value, match):
    from soup_cli.trainer.l2l_kto import refuse_unsupported_kto

    trainer = SimpleNamespace(
        loss_type="kto",
        calculate_KL=True,
        ref_model=None,
        precompute_ref_log_probs=False,
        ref_adapter_name=None,
        aux_loss_enabled=False,
    )
    refuse_unsupported_kto(trainer)
    setattr(trainer, attribute, value)
    with pytest.raises(ValueError, match=match):
        refuse_unsupported_kto(trainer)


def _example(prompt, completion, kl_completion, label):
    """One tokenized KTO row as TRL's ``_process_tokens`` leaves it (ids are made up)."""
    def side(answer):
        ids = list(prompt) + list(answer)
        return ids, [1] * len(ids), [-100] * len(prompt) + list(answer)

    ids, mask, labels = side(completion)
    kl_ids, kl_mask, kl_labels = side(kl_completion)
    return {
        "completion_input_ids": ids,
        "completion_attention_mask": mask,
        "completion_labels": labels,
        "KL_completion_input_ids": kl_ids,
        "KL_completion_attention_mask": kl_mask,
        "KL_completion_labels": kl_labels,
        "label": label,
    }


def _collated(*examples):
    """What TRL's ``DPODataCollatorWithPadding`` hands the trainer, built by that class."""
    from trl.experimental.utils import DPODataCollatorWithPadding

    return DPODataCollatorWithPadding(pad_token_id=3)(list(examples))


def _args(beta=0.3, loss_type="kto", desirable_weight=1.0, undesirable_weight=1.0):
    return SimpleNamespace(
        beta=beta,
        loss_type=loss_type,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )


def _batch(labels=(True, False)):
    return _collated(
        _example((1, 2), (5, 6, 7), (8,), labels[0]),
        _example((1, 2), (9,), (5, 6, 7, 4), labels[1]),
    )


def test_the_window_reader_walks_the_completions_and_the_kl_completions_at_their_own_widths():
    import torch

    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    batch = _batch()
    ids_list, extras = strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert len(ids_list) == 1
    assert torch.equal(ids_list[0], batch["completion_input_ids"])
    assert tuple(ids_list[0].shape) == (2, 5)

    policy_kl, reference, reference_kl = strategy.make_reference(extras)
    assert torch.equal(policy_kl.micro_batches[0], batch["KL_completion_input_ids"])
    assert tuple(policy_kl.micro_batches[0].shape) == (2, 6)
    assert reference_kl.micro_batches[0] is policy_kl.micro_batches[0]
    assert reference.micro_batches is None  # the completions are the policy's ids


def test_the_window_reader_sets_the_adapters_as_kto_does():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    _ids_list, extras = strategy.read_window(
        [_batch()], "cpu", neftune=False, has_dropout=False
    )
    policy_kl, reference, reference_kl = strategy.make_reference(extras)
    assert not policy_kl.adapters_off and policy_kl.adapter is None  # policy, no grad
    assert reference.adapters_off and reference_kl.adapters_off  # TRL's null_ref_context


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    short = _collated(_example((1,), (5,), (6,), True), _example((1,), (7,), (8,), False))
    long = _collated(
        _example((1,), (5, 6, 7, 8), (6,), True), _example((1,), (7,), (8, 9, 4), False)
    )
    ids_list, _ = strategy.read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


@pytest.mark.parametrize(
    "key,row", [("completion_attention_mask", 1), ("KL_completion_attention_mask", 0)]
)
def test_the_window_reader_refuses_left_padding(key, row):
    """``row`` is the row of ``_batch`` that has padding in that mask."""
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    batch = _batch()
    batch[key][row] = batch[key][row].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    window = [
        _collated(_example((1,), (5,), (6,), True), _example((1,), (7,), (8,), False)),
        _collated(_example((1,), (5, 6, 7), (6,), True), _example((1,), (7,), (8,), False)),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        strategy.read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        strategy.read_window(window, "cpu", neftune=False, has_dropout=True)


@pytest.mark.parametrize("labels", [(True, True), (False, False), (True, False)])
def test_the_window_reader_takes_micro_batches_of_one_label_or_two(labels):
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    ids_list, _ = strategy.read_window(
        [_batch(labels)], "cpu", neftune=False, has_dropout=False
    )
    assert len(ids_list) == 1


def test_kto_divides_the_window_by_k_and_needs_no_item_count():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    assert strategy.divide_by == "micro_batches"
    assert strategy.divisor(3, None) == 3


def _l2l_yaml(task):
    import yaml

    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl", "format": "dpo"},
            "training": {
                "batch_size": 2,
                "quantization": "none",
                "stream_layers": True,
                "l2l": True,
                "gradient_accumulation_steps": 2,
                "lora": {"r": 4, "alpha": 8},
            },
        }
    )


def test_the_schema_refuses_kto_with_l2l_until_it_is_ready():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "kto" not in L2L_SUPPORTED_TASKS
    pattern = r"training\.l2l supports task in \['sft', 'dpo'\]; got task='kto'"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_l2l_yaml("kto"))


def test_the_kto_trainer_wrapper_without_l2l_builds_the_unchanged_trainer_source():
    """The wrapper has no L2L dispatch: nothing in it can reach the kto window loss."""
    import inspect

    from soup_cli.trainer.kto import KTOTrainerWrapper

    assert "l2l" not in inspect.getsource(KTOTrainerWrapper)
