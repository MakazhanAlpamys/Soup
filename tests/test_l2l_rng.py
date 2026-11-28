"""l2l_rng: the replay of the random stream that plain gradient accumulation draws.

Seams: ``record_draws`` on two dropout layers, ``replay`` against the masks a plain
micro-batch-major accumulation draws from one generator, ``state_at`` and ``leave_stream_at``.
The L2L step that consumes these lands in a later change.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

SEED = 1234
K = 3
SHAPE = (2, 5, 8)


class _Layer(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        return self.drop(x)


def _layers():
    layers = [_Layer(0.3), _Layer(0.0), _Layer(0.5)]
    for layer in layers:
        layer.train()
    return layers


def _accumulation_masks(layers):
    """Draw as accumulation does: micro-batch after micro-batch, layer after layer."""
    torch.manual_seed(SEED)
    masks = {}
    for mb in range(K):
        for index, layer in enumerate(layers):
            masks[(mb, index)] = layer(torch.ones(SHAPE))
    return masks, torch.get_rng_state()


def _plan(layers):
    from soup_cli.utils import l2l_rng

    torch.manual_seed(SEED)
    with l2l_rng.record_draws(layers) as plan:
        for index, layer in enumerate(layers):
            layer(torch.ones(SHAPE))
    return plan


def test_record_draws_notes_each_dropout_with_its_metadata():
    layers = _layers()
    plan = _plan(layers)
    assert [len(layer_plan) for layer_plan in plan] == [1, 0, 1]
    first = plan[0][0]
    assert (first.shape, first.p, first.dtype) == (SHAPE, 0.3, torch.float32)
    assert plan[2][0].p == 0.5


def test_record_draws_restores_functional_dropout_on_exit():
    from soup_cli.utils import l2l_rng

    real = torch.nn.functional.dropout
    with l2l_rng.record_draws(_layers()):
        assert torch.nn.functional.dropout is not real
    assert torch.nn.functional.dropout is real


def test_has_draws():
    from soup_cli.utils import l2l_rng

    assert l2l_rng.has_draws(_plan(_layers()))
    assert not l2l_rng.has_draws([[], []])


def test_replayed_states_give_layer_calls_the_masks_accumulation_draws():
    from soup_cli.utils import l2l_rng

    layers = _layers()
    expected, expected_end = _accumulation_masks(layers)
    plan = _plan(layers)
    torch.manual_seed(SEED)
    states, end = l2l_rng.replay(plan, K)
    # layer-major visiting order, each call at the state of its (micro-batch, layer) pair
    for index, layer in enumerate(layers):
        for mb in range(K):
            with l2l_rng.state_at(states[mb][index]):
                got = layer(torch.ones(SHAPE))
            assert torch.equal(got, expected[(mb, index)]), (mb, index)
    assert torch.equal(end.cpu, expected_end)


def test_replay_leaves_the_outer_stream_where_it_found_it():
    from soup_cli.utils import l2l_rng

    plan = _plan(_layers())
    torch.manual_seed(SEED)
    before = torch.get_rng_state()
    l2l_rng.replay(plan, K)
    assert torch.equal(torch.get_rng_state(), before)


def test_state_at_puts_the_surrounding_stream_back():
    from soup_cli.utils import l2l_rng

    torch.manual_seed(SEED)
    pinned = l2l_rng.snapshot()
    torch.manual_seed(SEED + 1)
    before = torch.get_rng_state()
    with l2l_rng.state_at(pinned):
        assert torch.equal(torch.get_rng_state(), pinned.cpu)
        torch.rand(4)
    assert torch.equal(torch.get_rng_state(), before)


def test_leave_stream_at_sets_the_outer_stream():
    from soup_cli.utils import l2l_rng

    torch.manual_seed(SEED)
    target = l2l_rng.snapshot()
    torch.manual_seed(SEED + 1)
    l2l_rng.leave_stream_at(target)
    assert torch.equal(torch.get_rng_state(), target.cpu)


def test_a_replay_in_the_wrong_order_disagrees_with_accumulation():
    """Control: the states are keyed by (micro-batch, layer); a layer-major replay differs."""
    from soup_cli.utils import l2l_rng

    layers = _layers()
    expected, _end = _accumulation_masks(layers)
    plan = _plan(layers)
    torch.manual_seed(SEED)
    states, _ = l2l_rng.replay(plan, K)
    with l2l_rng.state_at(states[1][0]):
        got = layers[0](torch.ones(SHAPE))
    assert not torch.equal(got, expected[(0, 0)])


@pytest.mark.parametrize("dropout_p,refused", [(0.1, True), (0.0, False)])
def test_sdpa_attention_dropout_is_refused_at_capture(dropout_p, refused):
    from soup_cli.utils import l2l_rng

    class Attention(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.scaled_dot_product_attention(x, x, x, dropout_p=dropout_p)

    layer = Attention()
    x = torch.ones(1, 1, 4, 8)
    with l2l_rng.record_draws([layer]):
        if refused:
            with pytest.raises(ValueError, match="cannot replay attention dropout"):
                layer(x)
        else:
            layer(x)
