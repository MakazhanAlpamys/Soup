"""Reserve the public L2L RAM activation window before streaming weights."""

from dataclasses import dataclass
from typing import Sequence

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.partial_ram_source import PartialRamBudget

ACTIVATION_LENGTH_BUCKET = 64


@dataclass(frozen=True)
class L2LHostBudgetPlan:
    """A fixed charge in the weight envelope and a reusable activation partition."""

    weights: PartialRamBudget
    activations: HostMemoryBudget
    n_layers: int
    max_shape: tuple[int, int, int]
    max_microbatches: int
    dtype: str
    pinned: bool
    routing_bytes: int = 0

    def validate_microbatches(
        self,
        *,
        n_layers: int,
        shapes: Sequence[tuple[int, int, int]],
        dtype: str,
        pin: bool,
    ) -> None:
        if (
            not shapes
            or type(n_layers) is not int
            or n_layers < 1
            or any(
                len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape)
                for shape in shapes
            )
        ):
            raise ValueError("training.l2l activation dimensions must be positive integers")
        max_batch, max_sequence, hidden = self.max_shape
        if (
            n_layers != self.n_layers
            or len(shapes) > self.max_microbatches
            or dtype != self.dtype
            or pin is not self.pinned
            or any(
                batch > max_batch or sequence > max_sequence or width != hidden
                for batch, sequence, width in shapes
            )
        ):
            option = "training.l2l_auto_k" if self.routing_bytes else "training.stream_partial_ram"
            raise ValueError(
                "training.l2l activation window exceeds or differs from the window "
                f"reserved by {option}"
            )


def plan_l2l_host_budget(
    weights: PartialRamBudget,
    *,
    n_layers: int,
    batch_size: int,
    max_length: int,
    hidden_size: int,
    microbatches: int,
    dtype: str,
    pin: bool,
    shape_round: int = ACTIVATION_LENGTH_BUCKET,
    routing_bytes: int = 0,
) -> L2LHostBudgetPlan:
    """Reserve activations and any resolved MoE routing in one host partition."""
    from soup_cli.utils.budgeted_activations import activation_allocation_bytes
    from soup_cli.utils.layer_stream import dtype_bytes

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(
            "training.stream_partial_ram with training.l2l requires a fixed "
            "positive integer training.batch_size"
        )
    dimensions = (n_layers, max_length, hidden_size, microbatches, shape_round)
    if any(type(size) is not int or size < 1 for size in dimensions) or type(pin) is not bool:
        raise ValueError("training.l2l requires positive activation dimensions and boolean pin")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("training.l2l requires float32 or bfloat16 activations")
    if type(routing_bytes) is not int or routing_bytes < 0:
        raise ValueError("training.l2l routing_bytes must be a non-negative integer")
    sequence = -(-max_length // shape_round) * shape_round
    allocation = activation_allocation_bytes(
        n_layers=n_layers,
        k=microbatches,
        chunk_bytes=batch_size * sequence * hidden_size * dtype_bytes(dtype),
        spill_layers=0,
        ring=1,
        pin=pin,
    ) + routing_bytes
    reserved = weights.reserved_bytes + allocation
    if reserved > weights.limit_bytes:
        raise ValueError(
            f"training.stream_partial_ram.limit_bytes={weights.limit_bytes} cannot cover "
            f"{weights.reserved_bytes} fixed bytes and {allocation} bytes for training.l2l"
        )
    return L2LHostBudgetPlan(
        weights=PartialRamBudget(weights.limit_bytes, reserved, weights.cache_bytes),
        activations=HostMemoryBudget(allocation),
        n_layers=n_layers,
        max_shape=(batch_size, sequence, hidden_size),
        max_microbatches=microbatches,
        dtype=dtype,
        pinned=pin,
        routing_bytes=routing_bytes,
    )
