"""CPU comparisons with autograd for one native expert segment."""

from contextlib import nullcontext

import pytest

from soup_cli.utils.direct_expert_backward import expert_vjp


@pytest.mark.parametrize("precision", ["fp32", "bf16", "autocast"])
@pytest.mark.parametrize("need_hidden,need_weights", [(True, True), (True, False), (False, True)])
def test_direct_expert_vjp_matches_autograd(precision, need_hidden, need_weights):
    torch = pytest.importorskip("torch")
    from torch.nn import functional

    gen = torch.Generator().manual_seed(81)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    hidden = torch.randn(7, 16, generator=gen, dtype=dtype).requires_grad_(need_hidden)
    weights = torch.rand(7, 1, generator=gen).requires_grad_(need_weights)
    gate_up = torch.randn(24, 16, generator=gen, dtype=dtype)
    down = torch.randn(16, 12, generator=gen, dtype=dtype)
    # Replay can send a bf16 gradient to a product that recomputes as fp32.
    grad = torch.randn(7, 16, generator=gen, dtype=torch.bfloat16)
    cast = torch.bfloat16 if precision == "autocast" else None
    with torch.autocast("cpu", dtype=cast) if cast is not None else nullcontext():
        gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
        expected = functional.linear(functional.silu(gate) * up, down) * weights
    inputs = [tensor for tensor in (hidden, weights) if tensor.requires_grad]
    reference = torch.autograd.grad(expected, inputs, grad)

    with torch.no_grad():
        actual = expert_vjp(
            hidden, weights, gate_up, down, grad, need_hidden=need_hidden,
            need_weights=need_weights, autocast_dtype=cast,
        )
    assert (actual[0] is not None, actual[1] is not None) == (need_hidden, need_weights)
    for result, want in zip((value for value in actual if value is not None), reference):
        assert result.dtype == want.dtype
        assert torch.equal(result, want)
