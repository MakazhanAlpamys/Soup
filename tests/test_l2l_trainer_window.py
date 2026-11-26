"""A last window shorter than k through the real SFT trainer (ticket 02, CPU).

A dataset whose size k does not divide ends in a shorter window. HF builds that window
with its own ``num_items_in_batch``, takes one optimizer step on it and moves the lr
schedule by one. The L2L trainer must match gradient accumulation on the gradients, the
number of optimizer steps, the lr of every step and the logged loss.
"""

from __future__ import annotations

import math
from typing import Any, List, NamedTuple

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, streamed  # noqa: F401
from tests.test_l2l_trainer_labels import _MaskedRows


class Run(NamedTuple):
    """What one training run leaves behind, for comparison with its twin."""

    logs: List[dict]
    rates: List[float]
    grads: List[dict]
    stores: List[tuple]
    state: Any
    trainer: Any


def _train(base_cls, model, runtime, tmp_path, data, *, l2l, k, resume=None, **overrides):
    """Train; record the lr, the LoRA gradient and the activation store of every optimizer step."""
    import torch
    from transformers import TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads, rates, stores = [], [], []
    holder = {}

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

        def on_step_end(self, args, state, control, optimizer=None, **kw):
            rates.append(optimizer.param_groups[0]["lr"])
            store = holder["state"].store if l2l else None
            if store is not None:
                stores.append(
                    (id(store), store.k, store.stats.resident_chunks, tuple(store.mb_shapes))
                )

    options = dict(
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
    options.update(overrides)
    cls = make_l2l_trainer_class(base_cls) if l2l else base_cls
    trainer = cls(
        model=model, args=TrainingArguments(**options), train_dataset=data, callbacks=[Recorder()]
    )
    state = attach_l2l_state(trainer, runtime, dtype=torch.float32) if l2l else None
    holder["state"] = state
    trainer.train(resume_from_checkpoint=resume)
    logs = [row for row in trainer.state.log_history if "loss" in row]
    return Run(logs, rates, grads, stores, state, trainer)


def _both(streamed, tmp_path, rows, k, **overrides):  # noqa: F811
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _MaskedRows([SEQ] * rows, [[(0, i % 3)] if i % 3 else [] for i in range(rows)])
    ga = _train(Trainer, model, runtime, tmp_path, data, l2l=False, k=k, **overrides)
    harness.restore_adapters(model, start)
    l2l = _train(Trainer, model, runtime, tmp_path, data, l2l=True, k=k, **overrides)
    return ga, l2l


def _assert_same_run(ga, l2l, windows):
    import torch

    assert len(ga.grads) == len(l2l.grads) == windows
    assert ga.trainer.state.global_step == l2l.trainer.state.global_step == windows
    assert l2l.state.steps == windows
    for gradient_ga, gradient_l2l in zip(ga.grads, l2l.grads):
        assert gradient_ga.keys() == gradient_l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in gradient_l2l.values())
        for name in gradient_ga:
            assert torch.isfinite(gradient_l2l[name]).all(), name
            assert torch.equal(gradient_ga[name], gradient_l2l[name]), name
    assert l2l.rates == ga.rates
    assert all(math.isfinite(row["loss"]) for row in l2l.logs)
    assert [row["loss"] for row in l2l.logs] == [row["loss"] for row in ga.logs]
    assert [row["learning_rate"] for row in l2l.logs] == [row["learning_rate"] for row in ga.logs]
    assert [row["l2l_steps"] for row in l2l.logs] == list(range(1, windows + 1))
    for earlier, later in zip(l2l.logs, l2l.logs[1:]):
        for key in ("l2l_layer_loads", "l2l_large_loads"):
            assert later[key] > earlier[key], key
        assert later["l2l_weight_restores"] >= earlier["l2l_weight_restores"]
    assert l2l.logs[0]["l2l_layer_loads"] > 0


@pytest.mark.parametrize(
    "rows,k,windows", [(7, 3, 3), (8, 3, 3), (5, 4, 2), (3, 4, 1)]
)
def test_a_short_last_window_matches_accumulation(streamed, tmp_path, rows, k, windows):  # noqa: F811
    ga, l2l = _both(streamed, tmp_path, rows, k)
    _assert_same_run(ga, l2l, windows)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("budget", [{"num_train_epochs": 2}, {"max_steps": 5}])
def test_the_lr_of_every_step_matches_accumulation(streamed, tmp_path, schedule, budget):  # noqa: F811
    ga, l2l = _both(
        streamed, tmp_path, 7, 3, lr_scheduler_type=schedule, warmup_steps=1, **budget
    )
    _assert_same_run(ga, l2l, budget.get("max_steps", 6))
    assert len(set(l2l.rates)) > 2  # the schedule really moves


@pytest.mark.parametrize("max_steps", [3, 5])
def test_max_steps_counts_windows_across_epochs(streamed, tmp_path, max_steps):  # noqa: F811
    # 7 rows and k=3 give windows 3, 3, 1 per epoch: 3 ends on the short window, 5 inside epoch 2
    ga, l2l = _both(streamed, tmp_path, 7, 3, max_steps=max_steps)
    _assert_same_run(ga, l2l, max_steps)


def test_the_store_survives_short_windows_without_growing(streamed, tmp_path):  # noqa: F811
    _, l2l = _both(streamed, tmp_path, 7, 3, num_train_epochs=2)
    # windows 3, 3, 1, 3, 3, 1: the same full-size store serves all six steps
    assert len(l2l.stores) == 6
    assert len(set(l2l.stores)) == 1
    _, k, resident, _ = l2l.stores[0]
    assert k == 3
    assert resident == 3 * k  # three layers x k chunks, nothing accumulated
    assert l2l.state.window is None
    l2l.state.close()
    assert l2l.state.store is None


def test_a_short_window_before_a_full_one_replaces_the_store(streamed, tmp_path, monkeypatch):  # noqa: F811
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _l2l_step_for, _store_for

    model, runtime = streamed
    state = L2LState(runtime=runtime, dtype=torch.float32)
    ids = torch.zeros((1, SEQ), dtype=torch.long)
    stepper, _ = _l2l_step_for(state, model, [ids])

    small = _store_for(state, stepper, [ids], "cpu")
    closed = []
    real_close = small.close

    def close(*args, **kwargs):
        closed.append(small)
        real_close(*args, **kwargs)

    monkeypatch.setattr(small, "close", close)
    full = _store_for(state, stepper, [ids] * 3, "cpu")
    assert full is not small and full.k == 3
    assert closed == [small]  # the short-window store is released, not leaked
    assert _store_for(state, stepper, [ids], "cpu") is full  # a later short window reuses it
    state.close()


def test_resume_does_not_shift_the_windows(streamed, tmp_path, monkeypatch):  # noqa: F811
    import torch
    from transformers import Trainer

    # peft loads a resumed adapter onto the best device (mps on a Mac); the runtime is CPU
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _MaskedRows([SEQ] * 7, [[(0, i % 3)] if i % 3 else [] for i in range(7)])
    options = dict(
        l2l=True, k=3, num_train_epochs=2, lr_scheduler_type="linear", warmup_steps=1,
    )

    full = _train(
        Trainer, model, runtime, tmp_path, data, save_strategy="steps", save_steps=1, **options
    )
    checkpoints = {4: tmp_path / "l2l" / "checkpoint-4", 3: tmp_path / "l2l" / "checkpoint-3"}
    assert all(path.is_dir() for path in checkpoints.values())
    # 3 is the end of epoch 1 (its last window was short); 4 opens epoch 2 with a full window
    for step, checkpoint in checkpoints.items():
        harness.restore_adapters(model, start)
        resumed = _train(
            Trainer, model, runtime, tmp_path, data, resume=str(checkpoint), **options
        )
        left = 6 - step
        assert resumed.trainer.state.global_step == 6
        assert resumed.state.steps == left
        assert len(resumed.grads) == left
        for ours, theirs in zip(resumed.grads, full.grads[step:]):
            for name in theirs:
                assert torch.allclose(ours[name], theirs[name], atol=1e-6), (step, name)
        assert [row["loss"] for row in resumed.logs[-left:]] == [
            row["loss"] for row in full.logs[-left:]
        ]
        assert [row["learning_rate"] for row in resumed.logs[-left:]] == [
            row["learning_rate"] for row in full.logs[-left:]
        ]


@pytest.mark.parametrize("k,windows", [(1, 7), (2, 4), (4, 2)])
def test_rows_of_different_lengths_match_accumulation_bit_for_bit(
    streamed, tmp_path, k, windows  # noqa: F811
):
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    # 7 rows, so the last window is incomplete for k=2 and k=4
    lengths = [5, 8, 3, 6, 4, 7, 2]
    data = _MaskedRows(lengths, [[(0, i % 3)] if i % 3 else [] for i in range(7)])
    ga = _train(Trainer, model, runtime, tmp_path, data, l2l=False, k=k)
    harness.restore_adapters(model, start)
    l2l = _train(Trainer, model, runtime, tmp_path, data, l2l=True, k=k)
    _assert_same_run(ga, l2l, windows)
    # one captured call per distinct shape, and the store holds each row at its own size
    assert len(l2l.state.calls_by_shape) == len({n for n in lengths})
    assert l2l.state.store is None  # train closes the store on every exit
    if k > 1:
        assert len(set(l2l.stores[-1][3])) > 1
