"""Packed NF4 expert projections and first-order input gradients."""

from __future__ import annotations

from typing import Any


class PackedGemm:
    def __init__(self, *, backend: str, call_budget_bytes: int) -> None:
        self.backend = backend
        self.call_budget_bytes = call_budget_bytes

    def linear(self, inputs: list, weights: list, *, transpose: bool = False) -> list:
        from soup_cli.utils.expert_group_fused import fused_group_linear

        return fused_group_linear(
            [tensor.detach().contiguous() for tensor in inputs], weights,
            transpose=transpose, backend=self.backend, call_budget_bytes=self.call_budget_bytes,
        )

    def vjp(
        self, act_fn: Any, hidden: Any, weights: tuple, grad: Any, routing: Any = None,
        *, need_hidden: bool = True, need_routing: bool = False,
    ) -> tuple:
        """Recompute the packed projections and differentiate the native pointwise operations."""
        import torch

        gate_up, down = weights
        first = self.linear([hidden], [gate_up])[0]
        # The only differentiable graph here is over activations, never arena views.
        with torch.enable_grad():
            first.requires_grad_(need_hidden)
            gate, up = first.chunk(2, dim=-1)
            middle = act_fn(gate) * up
        grad_routing = None
        if routing is not None:
            output = self.linear([middle], [down])[0]
            with torch.enable_grad():
                output.requires_grad_(need_hidden)
                local_routing = routing.detach().requires_grad_(need_routing)
                scaled = output * local_routing
                wanted = [
                    tensor for tensor in (output, local_routing) if tensor.requires_grad
                ]
                grads = torch.autograd.grad(scaled, wanted, grad)
            if need_routing:
                grad_routing = grads[-1]
            if need_hidden:
                grad = grads[0]
        if not need_hidden:
            return None, grad_routing
        grad_mid = self.linear([grad], [down], transpose=True)[0]
        (grad_first,) = torch.autograd.grad(middle, first, grad_mid)
        return self.linear([grad_first], [gate_up], transpose=True)[0], grad_routing
