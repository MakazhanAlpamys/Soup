"""The streamed-expert backward inside an L2L step: autocast state and the outer RNG stream.

The L2L step runs the whole window under autocast, but accumulation's backward runs after the
autocast block has exited. A streamed expert's backward must reproduce the resident module in
that state: fp32 experts under a bf16 forward, backward outside autocast, equal ``torch.equal``
to transformers' eager experts loop. The expert backward also replays only the validated
stateless SiLU, so neither the forward nor the backward may draw from, or move, the global
generator that dropout replay depends on.

Fixtures reuse ``tests/test_streamed_experts_pool.py`` (integer-built weights and routing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests.test_streamed_experts_pool import (  # noqa: E402
    _assert_exact,
    _experts,
    _inputs,
    _store,
    _streamed,
)

LEVERS = {
    "default": ({}, False),
    "two-reads": (dict(two_reads=True), False),
    "all-levers": (dict(two_reads=True, grouped_expert_routes=True), True),
}


def _run_bf16_forward(module: Any, args: Any) -> Any:
    """Forward under CPU bf16 autocast, backward after the block (HF's order)."""
    hidden, index, weights, grad_out = args
    hidden = hidden.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(hidden, index, weights)
    out.backward(grad_out)
    return {"out": out.detach(), "hidden": hidden.grad, "weights": weights.grad}


@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_fp32_experts_under_a_bf16_forward_equal_eager_with_backward_outside_autocast(
    tmp_path: Path, lever: str
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(tokens=24, top_k=2)
        want = _run_bf16_forward(module, args)
        kwargs, fast = LEVERS[lever]
        streamed = _streamed(module, reader, pool, **kwargs)
        streamed.fast_row_gather = fast
        _assert_exact(_run_bf16_forward(streamed, args), want)
    finally:
        reader.close()
        pool.close()


def test_a_backward_inside_autocast_still_matches_the_resident_module_in_that_state(
    tmp_path: Path,
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)

    def run(target: Any) -> Any:
        hidden, index, weights, grad_out = _inputs(tokens=24, top_k=2)
        hidden = hidden.clone().requires_grad_(True)
        weights = weights.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = target(hidden, index, weights)
            out.backward(grad_out)
        return {"out": out.detach(), "hidden": hidden.grad, "weights": weights.grad}

    try:
        _assert_exact(run(_streamed(module, reader, pool)), run(module))
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_neither_the_forward_nor_the_backward_moves_the_global_generator(
    tmp_path: Path, lever: str
) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        kwargs, fast = LEVERS[lever]
        streamed = _streamed(module, reader, pool, **kwargs)
        streamed.fast_row_gather = fast
        hidden, index, weights, grad_out = _inputs(tokens=24, top_k=2)
        hidden = hidden.clone().requires_grad_(True)
        weights = weights.clone().requires_grad_(True)
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        out = streamed(hidden, index, weights)
        assert torch.equal(torch.get_rng_state(), before)
        out.backward(grad_out)
        assert torch.equal(torch.get_rng_state(), before)
        assert float(hidden.grad.abs().sum()) > 0.0
    finally:
        reader.close()
        pool.close()


def test_a_stateful_activation_is_refused_before_any_expert_is_read(tmp_path: Path) -> None:
    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        streamed = _streamed(module, reader, pool)

        class Stochastic(torch.nn.Module):
            def forward(self, value: Any) -> Any:
                return torch.nn.functional.dropout(value, p=0.5, training=True)

        streamed.act_fn = Stochastic()
        hidden, index, weights, _ = _inputs(tokens=4, top_k=2)
        reads = pool.fetches
        with pytest.raises(ValueError, match="stateless SiLU"):
            streamed(hidden.clone().requires_grad_(True), index, weights)
        assert pool.fetches == reads
    finally:
        reader.close()
        pool.close()
