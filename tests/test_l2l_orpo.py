"""The ORPO window loss ships as code; ``training.l2l`` does not reach it yet.

ORPO did not pass the readiness rule (its run is not verified to match gradient accumulation),
so the schema refuses ``task: orpo`` under ``training.l2l``. The window reader and the loss
stay importable and tested at the unit level.
"""

from __future__ import annotations

import pytest


def _collated(chosen, rejected):
    """What TRL's ``DPODataCollatorWithPadding`` hands the trainer: no ``input_ids``."""
    import torch

    def rows(seqs, pad):
        width = max(len(s) for s in seqs)
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
        labels = torch.tensor([[-100] + s[1:] + [-100] * (width - len(s)) for s in seqs])
        return ids, mask, labels

    c_ids, c_mask, c_labels = rows(chosen, 3)
    r_ids, r_mask, r_labels = rows(rejected, 3)
    return {
        "prompt": ["a", "b"][: len(chosen)],
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


def test_the_window_reader_stacks_chosen_over_rejected_padded_on_the_right():
    import torch

    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    batch = _collated(chosen=[[1, 5, 6, 7], [1, 8]], rejected=[[1, 9], [1, 4, 5]])
    assert "input_ids" not in batch
    ids_list, _extras = strategy.read_window(
        [batch], "cpu", neftune=False, has_dropout=False
    )
    assert len(ids_list) == 1
    expected = torch.tensor(
        [[1, 5, 6, 7], [1, 8, 3, 3], [1, 9, 3, 3], [1, 4, 5, 3]]  # chosen rows, rejected rows
    )
    assert torch.equal(ids_list[0], expected)


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    short = _collated(chosen=[[1, 5]], rejected=[[1, 9]])
    long = _collated(chosen=[[1, 5, 6, 7, 8]], rejected=[[1, 9]])
    ids_list, _ = strategy.read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


def test_the_window_reader_refuses_left_padding():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    batch = _collated(chosen=[[1, 5, 6], [1, 8]], rejected=[[1, 9], [1, 4]])
    batch["chosen_attention_mask"][1] = batch["chosen_attention_mask"][1].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    window = [
        _collated(chosen=[[1, 5]], rejected=[[1, 9]]),
        _collated(chosen=[[1, 5, 6]], rejected=[[1, 9]]),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        strategy.read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        strategy.read_window(window, "cpu", neftune=False, has_dropout=True)


def test_orpo_divides_the_window_by_k_and_needs_no_item_count():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
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


def test_the_schema_refuses_orpo_with_l2l_until_it_is_ready():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "orpo" not in L2L_SUPPORTED_TASKS
    pattern = r"training\.l2l supports task in \['sft', 'dpo'\]; got task='orpo'"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_l2l_yaml("orpo"))


def test_the_orpo_trainer_wrapper_without_l2l_builds_the_unchanged_trainer_source():
    """The wrapper has no L2L dispatch: nothing in it can reach the orpo window loss."""
    import inspect

    from soup_cli.trainer.orpo import ORPOTrainerWrapper

    assert "l2l" not in inspect.getsource(ORPOTrainerWrapper)
