"""The SimPO window loss ships as code; ``training.l2l`` does not reach it yet.

SimPO did not pass the readiness rule (its run is not verified to match gradient accumulation),
so the schema refuses ``task: simpo`` under ``training.l2l``. The loss and the window reader
stay importable and tested at the unit level.
"""

from __future__ import annotations

import pytest

GAMMA = 0.7
CPO_ALPHA = 0.6


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


def _hand_loss(logits, labels, len_chosen, beta, gamma, cpo_alpha):
    """``-logsigmoid(beta*(p_c - p_r) - gamma)`` + ``cpo_alpha * NLL`` from the spec, in plain
    torch: p is the mean log-probability of the response tokens of a row; the NLL is the mean
    over the response tokens of the chosen rows."""
    import torch

    shifted = labels[:, 1:]
    logp = torch.log_softmax(logits[:, :-1].double(), dim=-1)
    mask = shifted != -100
    token_logp = logp.gather(-1, shifted.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    row_logp = token_logp.sum(-1) / mask.sum(-1)
    margin = beta * (row_logp[:len_chosen] - row_logp[len_chosen:]) - gamma
    nll = -token_logp[:len_chosen].sum() / mask[:len_chosen].sum()
    return -torch.nn.functional.logsigmoid(margin).mean() + cpo_alpha * nll


def test_the_simpo_loss_matches_the_formula_on_random_logits():
    import torch

    from soup_cli.trainer._trl_compat import resolve_trl_symbol
    from soup_cli.trainer.l2l_simpo import simpo_loss

    torch.manual_seed(0)
    trainer_cls = resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")
    logits = torch.randn(4, 6, 11)
    # two pairs; rows differ in the length of the response, prompts are masked with -100
    labels = torch.tensor(
        [
            [-100, -100, 4, 5, 6, 7],
            [-100, 3, 4, 5, -100, -100],
            [-100, -100, 8, 9, -100, -100],
            [-100, -100, -100, 2, 3, 4],
        ]
    )
    got = simpo_loss(
        logits, labels, 2, beta=0.1, gamma=GAMMA, cpo_alpha=CPO_ALPHA, trainer_cls=trainer_cls
    )
    want = _hand_loss(logits, labels, 2, 0.1, GAMMA, CPO_ALPHA)
    assert torch.isfinite(got)
    assert got.item() == pytest.approx(want.item(), rel=1e-5)


def test_the_nll_term_ignores_the_prompt_unlike_orpo():
    import torch

    from soup_cli.trainer._trl_compat import resolve_trl_symbol
    from soup_cli.trainer.l2l_simpo import simpo_loss

    trainer_cls = resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")
    logits = torch.randn(2, 5, 9)
    labels = torch.tensor([[-100, -100, 4, 5, 6], [-100, -100, 7, 8, 2]])
    with_nll = simpo_loss(
        logits, labels, 1, beta=0.1, gamma=0.0, cpo_alpha=1.0, trainer_cls=trainer_cls
    )
    without = simpo_loss(
        logits, labels, 1, beta=0.1, gamma=0.0, cpo_alpha=0.0, trainer_cls=trainer_cls
    )
    hand_nll = _hand_loss(logits, labels, 1, 0.1, 0.0, 1.0) - _hand_loss(
        logits, labels, 1, 0.1, 0.0, 0.0
    )
    assert (with_nll - without).item() == pytest.approx(hand_nll.item(), rel=1e-5)


def _strategy():
    from soup_cli.trainer.l2l_simpo import simpo_window_loss

    return simpo_window_loss(beta=0.1, gamma=GAMMA, cpo_alpha=CPO_ALPHA, padding_value=3)


def test_the_window_reader_stacks_chosen_over_rejected_padded_on_the_right():
    import torch

    batch = _collated(chosen=[[1, 5, 6, 7], [1, 8]], rejected=[[1, 9], [1, 4, 5]])
    assert "input_ids" not in batch
    ids_list, _extras = _strategy().read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert len(ids_list) == 1
    expected = torch.tensor(
        [[1, 5, 6, 7], [1, 8, 3, 3], [1, 9, 3, 3], [1, 4, 5, 3]]  # chosen rows, rejected rows
    )
    assert torch.equal(ids_list[0], expected)


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    short = _collated(chosen=[[1, 5]], rejected=[[1, 9]])
    long = _collated(chosen=[[1, 5, 6, 7, 8]], rejected=[[1, 9]])
    ids_list, _ = _strategy().read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


def test_the_window_reader_refuses_left_padding():
    batch = _collated(chosen=[[1, 5, 6], [1, 8]], rejected=[[1, 9], [1, 4]])
    batch["chosen_attention_mask"][1] = batch["chosen_attention_mask"][1].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        _strategy().read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    window = [
        _collated(chosen=[[1, 5]], rejected=[[1, 9]]),
        _collated(chosen=[[1, 5, 6]], rejected=[[1, 9]]),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        _strategy().read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        _strategy().read_window(window, "cpu", neftune=False, has_dropout=True)


def test_simpo_divides_the_window_by_k_and_needs_no_item_count():
    assert _strategy().divisor(3, None) == 3


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


def test_the_schema_refuses_simpo_with_l2l_until_it_is_ready():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "simpo" not in L2L_SUPPORTED_TASKS
    pattern = r"training\.l2l supports task in \['sft', 'dpo'\]; got task='simpo'"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_l2l_yaml("simpo"))


def test_the_simpo_trainer_wrapper_without_l2l_builds_the_unchanged_trainer_source():
    """The wrapper has no L2L dispatch: nothing in it can reach the simpo window loss."""
    import inspect

    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    assert "l2l" not in inspect.getsource(SimPOTrainerWrapper)
