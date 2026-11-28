"""First-order VJP for a native frozen SwiGLU expert segment."""

from typing import Any


def expert_vjp(
    hidden: Any,
    weights: Any,
    gate_up: Any,
    down: Any,
    grad_out: Any,
    *,
    need_hidden: bool,
    need_weights: bool,
    autocast_dtype: Any = None,
) -> tuple[Any, Any]:
    """Differentiate one segment's expert output times its routing weights.

    The caller holds the expert lease, has validated the stateless SiLU and has disabled
    autocast. Replay the forward's linear casts explicitly so the backward GEMMs run in the
    compute dtype, and return gradients in the original input dtypes.
    """
    import torch
    from torch.nn import functional

    hidden_dtype = hidden.dtype
    if autocast_dtype is not None:
        hidden, gate_up, down = (
            tensor.to(autocast_dtype) for tensor in (hidden, gate_up, down)
        )
    gate, up = torch.mm(hidden, gate_up.t()).chunk(2, dim=-1)
    activated = functional.silu(gate)
    # A replay node can return a low-precision zero in place of the promoted product.
    grad_out = grad_out.to(torch.promote_types(activated.dtype, weights.dtype))

    grad_weights = None
    if need_weights:
        output = torch.mm(activated * up, down.t())
        grad_weights = (grad_out * output).to(weights.dtype).sum(dim=1, keepdim=True)

    grad_hidden = None
    if need_hidden:
        grad_middle = torch.mm((grad_out * weights).to(activated.dtype), down)
        grad_gate = torch.ops.aten.silu_backward(grad_middle * up, gate)
        grad_gate_up = torch.cat((grad_gate, grad_middle * activated), dim=-1)
        grad_hidden = torch.mm(grad_gate_up, gate_up).to(hidden_dtype)
    return grad_hidden, grad_weights
