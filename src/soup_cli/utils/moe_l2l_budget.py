"""Memory estimates for MoE L2L windows, including FP8 boundary storage."""

from dataclasses import dataclass
from typing import Any, Sequence

from soup_cli.utils.budgeted_activations import activation_allocation_bytes
from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.layer_stream import SUPPORTED_STREAM_DTYPES, dtype_bytes


@dataclass(frozen=True)
class MoEL2LMemoryPlan:
    """Accounted host tensors and saved expert inputs for the selected window."""

    requested_k: int
    effective_k: int
    available_host_bytes: int
    activation_bytes: int
    routing_record_bytes: int
    routing_comparison_bytes: int
    n_layers: int
    n_moe_layers: int
    experts_per_token: int
    max_shape: tuple[int, int, int]
    dtype: str
    pin: bool
    codec: str
    device_type: str
    saved_expert_input_bytes: int | None = None
    window_host_bytes: int = 0

    @property
    def host_bytes(self) -> int:
        return (
            self.activation_bytes + self.routing_record_bytes
            + self.routing_comparison_bytes + self.window_host_bytes
        )

    def validate_microbatches(
        self, *, n_layers: int, shapes: Sequence[tuple[int, ...]],
        dtype: str, pin: bool, codec: str,
    ) -> None:
        """Refuse changes to the resolved window before allocating or collecting work."""
        if (
            not shapes or len(shapes) > self.effective_k or n_layers != self.n_layers
            or dtype != self.dtype or pin != self.pin or codec != self.codec
        ):
            raise ValueError("MoE L2L window differs from its resolved memory plan")
        for shape in shapes:
            if (
                len(shape) != 3
                or any(type(value) is not int or value < 1 for value in shape)
                or shape[-1] != self.max_shape[-1]
                or any(value > cap for value, cap in zip(shape, self.max_shape))
            ):
                raise ValueError("MoE L2L shape exceeds its resolved memory plan")


def plan_moe_l2l(
    host_budget: HostMemoryBudget,
    *,
    n_layers: int,
    n_moe_layers: int,
    max_shape: tuple[int, int, int],
    experts_per_token: int,
    requested_k: int,
    dtype: str,
    pin: bool,
    expert_input_dtype: str | None = None,
    codec: str = "none",
    device_type: str = "cuda",
    host_workspace_bytes_per_batch: int = 0,
) -> MoEL2LMemoryPlan:
    """Select the largest RAM-fitting ``k`` for a fixed upper bound on micro-batch shape.

    ``max_shape`` is ``(batch, sequence, hidden)``. Charge every position at that
    capacity using the store's allocation calculation. FP8 includes per-token scales,
    pinned rounding and codec host scratch. Charge pageable int64 routing records for
    streamed MoE layers, plus one temporary record for the routing comparison of the recompute.

    If supplied, expert_input_dtype is the dtype of the gathered hidden states
    entering the streamed experts. Backward runs for one recomputed layer and
    micro-batch at a time, so the saved input estimate has no layer or k factor.
    The host k limit uses host allocations. The plan reports the saved expert inputs
    separately as compute-device bytes.

    The caller can charge additional host workspace that grows with k, and accounts
    for fixed host consumers and safety headroom in ``host_budget``.
    Other device workspace and process memory need separate bounds. Planning reads
    the available budget once; allocations still acquire leases through ``HostMemoryBudget``.
    """
    dimensions = (n_layers, n_moe_layers, *max_shape, experts_per_token, requested_k)
    if len(max_shape) != 3 or any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("MoE L2L needs a positive (batch, sequence, hidden) shape and counts")
    if n_moe_layers > n_layers:
        raise ValueError("MoE layer count must not exceed the total layer count")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("MoE L2L host planning supports float32 or bfloat16 activations")
    if expert_input_dtype is not None and expert_input_dtype not in SUPPORTED_STREAM_DTYPES:
        raise ValueError(f"unsupported saved expert input dtype {expert_input_dtype!r}")
    if codec not in ("none", "fp8_e4m3"):
        raise ValueError("MoE L2L codec must be none or fp8_e4m3")
    if device_type not in ("cpu", "cuda") or type(pin) is not bool:
        raise ValueError("MoE L2L needs a cpu/cuda device type and boolean pin")
    if type(host_workspace_bytes_per_batch) is not int or host_workspace_bytes_per_batch < 0:
        raise ValueError("MoE L2L window workspace must be non-negative integer bytes")

    batch_size, sequence_length, hidden_size = max_shape
    tokens = batch_size * sequence_length
    comparison_bytes = tokens * experts_per_token * 8
    routing_bytes_per_batch = n_moe_layers * comparison_bytes
    # Routing records and caller workspace both grow linearly with k.
    window_bytes_per_batch = routing_bytes_per_batch + host_workspace_bytes_per_batch
    available = host_budget.available_bytes
    if codec == "none":
        activation_bytes_per_batch = activation_allocation_bytes(
            n_layers=n_layers, k=1,
            chunk_bytes=tokens * hidden_size * dtype_bytes(dtype),
            spill_layers=0, ring=1, pin=pin,
        )
        host_bytes_per_batch = activation_bytes_per_batch + window_bytes_per_batch
        effective_k = min(requested_k, (available - comparison_bytes) // host_bytes_per_batch)
        activation_bytes = max(0, effective_k) * activation_bytes_per_batch
        minimum_bytes = host_bytes_per_batch + comparison_bytes
    else:
        from soup_cli.utils.fp8_activations import fp8_activation_allocation

        def allocation(k: int) -> int:
            return fp8_activation_allocation(
                n_layers=n_layers, mb_shapes=[max_shape] * k, pin=pin,
                compute_dtype=dtype, device_type=device_type,
            ).host_bytes

        minimum_bytes = allocation(1) + window_bytes_per_batch + comparison_bytes
        # Scale slabs and shared codec scratch need not scale linearly with k.
        # Search the shared estimator rather than multiplying a one-batch estimate.
        low, high = 0, requested_k
        while low < high:
            middle = (low + high + 1) // 2
            needed = allocation(middle) + middle * window_bytes_per_batch + comparison_bytes
            if needed <= available:
                low = middle
            else:
                high = middle - 1
        effective_k = low
        activation_bytes = allocation(effective_k) if effective_k else 0
    if effective_k < 1:
        raise MemoryError(
            f"MoE L2L k=1 needs {minimum_bytes} host bytes "
            f"for activations, routing and window workspace, but only {available} available"
        )

    saved_expert_input_bytes = None
    if expert_input_dtype is not None:
        saved_expert_input_bytes = (
            tokens * experts_per_token * hidden_size * dtype_bytes(expert_input_dtype)
        )

    return MoEL2LMemoryPlan(
        requested_k=requested_k,
        effective_k=effective_k,
        available_host_bytes=available,
        activation_bytes=activation_bytes,
        routing_record_bytes=effective_k * routing_bytes_per_batch,
        routing_comparison_bytes=comparison_bytes,
        n_layers=n_layers,
        n_moe_layers=n_moe_layers,
        experts_per_token=experts_per_token,
        max_shape=max_shape,
        dtype=dtype,
        pin=pin,
        codec=codec,
        device_type=device_type,
        saved_expert_input_bytes=saved_expert_input_bytes,
        window_host_bytes=effective_k * host_workspace_bytes_per_batch,
    )


def estimate_moe_l2l_optimizer_host_bytes(model: Any, tcfg: Any) -> int:
    """Budget gradients and optimizer tensors from the live trainable parameters."""
    from soup_cli.utils.vram_estimate import (
        OPTIMIZER_BYTES_PER_PARAM,
        OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM,
    )

    optimizer = tcfg.optimizer
    if (
        optimizer not in ("adamw_torch", "sgd", "adagrad", "rmsprop", "adafactor")
        or getattr(tcfg, "use_galore", False)
        or getattr(tcfg, "use_lorafa", False)
        or getattr(tcfg, "loraplus_lr_ratio", None) is not None
    ):
        raise ValueError("training.l2l_auto_k requires a known optimizer host budget")

    state_bytes = int(OPTIMIZER_BYTES_PER_PARAM.get(optimizer, 4))
    scratch_bytes = int(OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM.get(optimizer, 0))
    host_bytes = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        on_cpu = param.device.type == "cpu"
        if optimizer in ("adamw_torch", "adagrad", "rmsprop") or (
            optimizer == "adafactor" and on_cpu
        ):
            host_bytes += 4  # CPU scalar step or RMS state.
        if on_cpu:
            # Optimizer states follow the parameter dtype or use FP32.
            size, elements = param.element_size(), param.numel()
            scale = max(4, size) // 4
            state_elements = elements
            if optimizer == "adafactor" and param.ndim >= 2:
                state_elements = elements // param.shape[-1] + elements // param.shape[-2]
            host_bytes += elements * (size + scale * scratch_bytes)
            host_bytes += state_elements * scale * state_bytes
    return host_bytes
