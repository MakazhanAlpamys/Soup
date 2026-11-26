"""Masked labels and mixed lengths through the real SFT trainer (ticket 01, CPU).

HF counts ``num_items_in_batch`` over the whole window and the losses divide by it. The
L2L trainer must give the loss of the window and the LoRA gradients that gradient
accumulation gives at the same k.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, streamed  # noqa: F401


class _MaskedRows:
    """Rows of random ids; ``masked_spans[i]`` lists the (start, stop) label slices set to -100."""

    def __init__(self, lengths, masked_spans, vocab=64, seed=5):
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.rows = []
        for length, spans in zip(lengths, masked_spans):
            ids = torch.randint(0, vocab, (length,), generator=gen)
            labels = ids.clone()
            for start, stop in spans:
                labels[start:stop] = -100
            self.rows.append(
                {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _train_epoch(base_cls, model, runtime, tmp_path, data, *, l2l, k):
    """Train one epoch; return the logged loss and the LoRA gradient of every window."""
    import torch
    from transformers import TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads = []

    class GradientRecorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

    args = TrainingArguments(
        output_dir=str(tmp_path / ("l2l" if l2l else "ga")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=k,
        learning_rate=0.5,
        optim="sgd",
        lr_scheduler_type="constant",
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=1,
        data_seed=1,
        max_grad_norm=0.0,
        disable_tqdm=True,
    )
    cls = make_l2l_trainer_class(base_cls) if l2l else base_cls
    trainer = cls(model=model, args=args, train_dataset=data, callbacks=[GradientRecorder()])
    if l2l:
        attach_l2l_state(trainer, runtime, dtype=torch.float32)
    trainer.train()
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    return losses, grads


def _assert_matches_accumulation(streamed, tmp_path, k, lengths, masked_spans, *, exact):  # noqa: F811
    import math

    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)

    data = _MaskedRows(lengths, masked_spans)
    ga_losses, ga_grads = _train_epoch(Trainer, model, runtime, tmp_path, data, l2l=False, k=k)
    harness.restore_adapters(model, start)
    l2l_losses, l2l_grads = _train_epoch(Trainer, model, runtime, tmp_path, data, l2l=True, k=k)

    assert len(ga_grads) == len(l2l_grads) == len(lengths) // k
    assert all(math.isfinite(loss) for loss in l2l_losses)
    for ga, l2l in zip(ga_grads, l2l_grads):
        assert ga.keys() == l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in l2l.values())
        for name in ga:
            assert torch.isfinite(l2l[name]).all(), name
            if exact:
                assert torch.equal(ga[name], l2l[name]), name
            else:
                diff = (ga[name] - l2l[name]).abs().max().item()
                assert torch.allclose(ga[name], l2l[name], atol=2e-5, rtol=1e-4), (name, diff)
    if exact:
        assert ga_losses == l2l_losses
    else:
        assert l2l_losses == pytest.approx(ga_losses, abs=2e-5, rel=1e-4)


@pytest.mark.parametrize("k", [2, 4])
def test_masked_labels_in_different_places_match_accumulation(streamed, tmp_path, k):  # noqa: F811
    masked_spans = [
        [(0, 3)], [(SEQ - 2, SEQ)], [], [(2, 5)], [(0, 1), (6, 8)], [(3, 4)], [], [(1, 4)]
    ]
    _assert_matches_accumulation(streamed, tmp_path, k, [SEQ] * 8, masked_spans, exact=True)


@pytest.mark.parametrize("k", [2, 4])
def test_a_micro_batch_of_only_masked_labels_gives_no_nan(streamed, tmp_path, k):  # noqa: F811
    masked_spans = [[(0, SEQ)], [(2, 4)], [], [(5, 7)], [], [(0, SEQ)], [(1, 3)], []]
    _assert_matches_accumulation(streamed, tmp_path, k, [SEQ] * 8, masked_spans, exact=True)


@pytest.mark.parametrize("k", [2, 4])
def test_rows_of_different_lengths_with_masked_labels_match_accumulation(
    streamed, tmp_path, k  # noqa: F811
):
    lengths = [5, 8, 3, 6, 70, 9, 4, 66]
    masked_spans = [[(0, 2)], [], [(1, 3)], [(4, 6)], [(0, 30)], [(7, 9)], [], [(60, 66)]]
    _assert_matches_accumulation(streamed, tmp_path, k, lengths, masked_spans, exact=False)


def test_k1_with_masked_labels_matches_accumulation(streamed, tmp_path):  # noqa: F811
    masked_spans = [[(0, 3)], [], [(SEQ - 2, SEQ)], [(2, 5)]]
    _assert_matches_accumulation(streamed, tmp_path, 1, [SEQ] * 4, masked_spans, exact=True)
