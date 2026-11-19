"""Native cache numerical contracts without the optional Arrow reader."""

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402


def test_native_explicit_cached_targets_and_rewind_share_the_numerical_objective():
    from types import SimpleNamespace

    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset, labeled_loss
    from soup_cli.trainer.rewind_mlx import ROW_LOSS_KEY, ROW_TOKENS_KEY, make_rewind_loss

    class FixedTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.arange(64, dtype=mx.float32).reshape(8, 8) / 10

        def __call__(self, tokens):
            return self.weight[tokens]

    row = {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 6, 2]}
    data = PretokenizedDataset([row], vocab_size=8, max_length=4)
    tokens, targets = data.process(data[0])
    batch, labels = mx.array([tokens]), mx.array([targets])
    model = FixedTable()
    model.train()
    state = SimpleNamespace(failed=False, optimizer_state={}, mark_pending=lambda: None)
    recorded = make_rewind_loss(state, kind="labels")
    logits = model(batch[:, :-1])
    expected = ((mx.logsumexp(logits, axis=-1)[0, 1] - logits[0, 1, 6]) +
                (mx.logsumexp(logits, axis=-1)[0, 2] - logits[0, 2, 2])) / 2
    loss, count = labeled_loss(model, batch, labels)
    tracked, tracked_count = recorded(model, batch, labels)
    assert count.item() == tracked_count.item() == 2
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert tracked.item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_LOSS_KEY].item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_TOKENS_KEY].item() == 2
    _, actual_grad = nn.value_and_grad(model, labeled_loss)(model, batch, labels)
    _, rewind_grad = nn.value_and_grad(model, recorded)(model, batch, labels)
    assert mx.allclose(actual_grad["weight"], rewind_grad["weight"], atol=1e-6).item()
    assert actual_grad["weight"][3, 6].item() < 0
    assert actual_grad["weight"][3, 4].item() > 0
