"""``training.stream_loss_chunk`` under ``training.l2l``.

The L2L window's head computes the chunked causal-LM loss (the logits of one chunk of
positions at a time) instead of the model's full-logits loss, so a large-vocabulary model
never holds one micro-batch's logits. An L2L step with the chunked head must equal gradient
accumulation with the same chunked loss, bit for bit, and stay within rounding of the
full-logits head.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.test_l2l_probe_harness import _load, _micro_batches, streamed  # noqa: E402,F401
from tests.test_l2l_step import _store  # noqa: E402

CHUNK = 3  # SEQ is 8, so each row is split into three chunks


def _ga_chunked(model, micro_batches, chunk):
    """Gradient accumulation through the streamed forward with the chunked loss."""
    from soup_cli.utils.l2l_step import label_tokens, model_parts
    from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

    causal, owner = model_parts(model)
    total = label_tokens(micro_batches)
    losses = []
    for ids in micro_batches:
        hidden = owner(input_ids=ids).last_hidden_state
        loss, _step = chunked_causal_lm_loss(
            hidden, causal.lm_head, ids, chunk, num_items_in_batch=total
        )
        loss.backward()
        losses.append(loss.detach())
    return losses


def _l2l(model, runtime, micro_batches, head):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = L2LStep(model, runtime, capture_layer_call(model, micro_batches[0]))
    return step.run(micro_batches, _store(len(micro_batches)), head=head)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_l2l_chunked_equals_accumulation_with_the_chunked_loss(streamed, k):  # noqa: F811
    from soup_cli.utils.l2l_step import label_tokens, sft_chunked_head

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    ga_losses = _ga_chunked(model, mbs, CHUNK)
    ga = harness.collect_grads(model)

    harness.zero_grads(model)
    l2l_losses = _l2l(model, runtime, mbs, sft_chunked_head(mbs, label_tokens(mbs), CHUNK))
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_the_chunked_head_matches_the_full_logits_head(streamed):  # noqa: F811
    import torch

    from soup_cli.utils.l2l_step import label_tokens, sft_chunked_head, sft_head

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    n_items = label_tokens(mbs)

    harness.zero_grads(model)
    full_losses = _l2l(model, runtime, mbs, sft_head(mbs, n_items))
    full = harness.collect_grads(model)
    harness.zero_grads(model)
    chunked_losses = _l2l(model, runtime, mbs, sft_chunked_head(mbs, n_items, CHUNK))
    chunked = harness.collect_grads(model)

    for want, got in zip(full_losses, chunked_losses):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert chunked.keys() == full.keys() and full
    for name in full:
        torch.testing.assert_close(chunked[name], full[name], rtol=1e-4, atol=1e-6)


def test_the_chunked_head_refuses_a_trainable_output_head(streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import label_tokens, model_parts, sft_chunked_head

    model, _runtime = streamed
    causal, owner = model_parts(model)
    mbs = _micro_batches(1)
    head = sft_chunked_head(mbs, label_tokens(mbs), CHUNK)
    param = next(causal.lm_head.parameters())
    was = param.requires_grad
    param.requires_grad_(True)
    try:
        hidden = owner(input_ids=mbs[0]).last_hidden_state.detach()
        with pytest.raises(ValueError, match="the output head must be frozen; trainable:"):
            head(causal, owner, hidden, 0)
    finally:
        param.requires_grad_(was)


def test_the_window_loss_for_a_chunk_keeps_the_sft_reading_and_scale():
    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS, sft_chunked_window_loss
    from soup_cli.utils.l2l_step import sft_head

    chunked = sft_chunked_window_loss(16)
    assert chunked.make_head is not sft_head and SFT_WINDOW_LOSS.make_head is sft_head
    assert chunked.divide_by == SFT_WINDOW_LOSS.divide_by == "items"
    assert chunked.read_window is SFT_WINDOW_LOSS.read_window
    assert callable(chunked.make_head([], 1))


# --- the schema --------------------------------------------------------------------


def _yaml(**training):
    import yaml

    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "l2l": True,
        "gradient_accumulation_steps": 2,
    }
    fields.update(training)
    return yaml.safe_dump(
        {"base": "local/model", "task": "sft", "data": {"train": "t.jsonl"}, "training": fields}
    )


def test_l2l_and_stream_loss_chunk_load_together():
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(_yaml(stream_loss_chunk=256))
    assert cfg.training.l2l is True and cfg.training.stream_loss_chunk == 256


def test_l2l_and_cut_ce_refuse_each_other_naming_both():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=r"training\.use_cut_ce.*cannot be combined with.*l2l"):
        load_config_from_string(_yaml(use_cut_ce=True))


def test_the_other_refusals_of_l2l_stand():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=r"training\.l2l cannot be combined with .*prefix_tree"):
        load_config_from_string(_yaml(stream_loss_chunk=256, prefix_tree=True))


def test_l2l_without_the_chunk_is_unchanged():
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(_yaml())
    assert cfg.training.stream_loss_chunk is None and cfg.training.use_cut_ce is False
