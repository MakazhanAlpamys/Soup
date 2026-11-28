"""Ticket 09: capturing the layer call of a new shape must not walk the streamed layers.

The capture used to be a full forward, so every micro-batch of a new length cost one load of
each layer on top of the step's own. Layer 0's arguments are known the moment layer 0 is
called, so the capture stops there.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import build_streamed

N_LAYERS = 8


@pytest.fixture()
def deep(tmp_path):
    model, runtime = build_streamed(tmp_path, n_layers=N_LAYERS)
    yield model, runtime
    runtime.close()


def _rows(lengths, vocab=64, seed=5):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (1, n), generator=gen) for n in lengths]


def _window_loads(deep, lengths, *, warm):
    """Layer loads of one ``run_window`` over rows of ``lengths``, capture included.

    ``warm`` runs the same window once first, so its shapes are already captured and the
    count is the step's own.
    """
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, run_window
    from soup_cli.utils.l2l_step import count_loads

    model, runtime = deep
    model.train()
    state = L2LState(runtime=runtime, dtype=torch.float32, has_dropout=False)
    rows = _rows(lengths)
    if warm:
        run_window(state, model, rows, rows, sum(lengths), "cpu")
    with count_loads(runtime) as counters:
        run_window(state, model, rows, rows, sum(lengths), "cpu")
    state.close()
    return counters.layer_loads


def test_a_window_of_new_shapes_loads_what_a_window_of_known_shapes_loads(deep):
    lengths = [3, 4, 5, 8]
    known = _window_loads(deep, lengths, warm=True)
    new = _window_loads(deep, lengths, warm=False)
    # a capture may touch the first layer or two, never the whole stack of N_LAYERS
    assert new <= known + len(lengths) * 2


def test_the_captured_call_is_the_one_a_full_forward_records(deep):
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call

    model, _runtime = deep
    model.train()
    ids = _rows([9])[0]
    full = capture_layer_call(model, ids)
    early = capture_layer_call(model, ids, draws=False)
    assert full.args == early.args
    assert full.kwargs.keys() == early.kwargs.keys()
    for name, value in full.kwargs.items():
        other = early.kwargs[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), name
        elif isinstance(value, tuple):
            assert all(torch.equal(a, b) for a, b in zip(value, other)), name
        elif name != "past_key_values":
            assert value == other, name
    assert early.draws == ()


def test_the_state_counts_each_capture_its_loads_and_its_time(deep):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, run_window

    model, runtime = deep
    model.train()
    state = L2LState(runtime=runtime, dtype=torch.float32, has_dropout=False)
    rows = _rows([3, 4, 4, 5])
    run_window(state, model, rows, rows, 12, "cpu")
    assert state.captures == 3
    assert state.capture_loads <= 3 * 2
    assert state.capture_seconds > 0
    run_window(state, model, rows, rows, 12, "cpu")
    assert state.captures == 3
    assert state.log_fields()["l2l_captures"] == 3
    state.close()


def test_a_model_with_dropout_keeps_the_full_forward_capture(tmp_path):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, run_window

    model, runtime = build_streamed(tmp_path, lora_dropout=0.1, n_layers=N_LAYERS)
    try:
        model.train()
        state = L2LState(runtime=runtime, dtype=torch.float32, has_dropout=True)
        rows = _rows([6, 6])
        run_window(state, model, rows, rows, 10, "cpu")
        call = next(iter(state.calls_by_shape.values()))
        assert any(call.draws)
        assert state.capture_loads >= N_LAYERS - 1
        state.close()
    finally:
        runtime.close()


def test_a_state_that_does_not_know_its_dropout_finds_it_out(deep):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, run_window

    model, runtime = deep
    model.train()
    state = L2LState(runtime=runtime, dtype=torch.float32)
    rows = _rows([3, 4])
    run_window(state, model, rows, rows, 5, "cpu")
    assert state.has_dropout is False
    assert state.capture_loads <= 2 * 2
    state.close()
