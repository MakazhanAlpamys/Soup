"""Account for persistent adapter tensors using resolved shapes and storage dtypes.

Pass each physical tensor once after target resolution, PEFT conversion and
freezing. Include frozen adapter tensors. Optimizer profiles describe effective
settings that the caller has verified against the training path.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import Literal

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
_PROFILES = frozenset(
    {
        "torch_adamw_fp32",  # non-fused, non-capturable, amsgrad=False
        "torch_sgd_fp32",  # momentum=0
        "torch_adagrad_fp32",  # non-fused
        "torch_rmsprop_fp32",  # momentum=0, centered=False, non-capturable
        "hf_adafactor_fp32",  # beta1=None
        "peft_lorafa_fp32",  # A frozen, B trainable
        "bnb_0.50.2_adamw8bit_fp32",  # non-paged, blockwise, min_8bit_size=4096
    }
)


@dataclass(frozen=True)
class AdapterTensor:
    """Metadata for a tensor with a known materialized storage dtype."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"adapter tensor name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError(
                f"adapter tensor {self.name!r} shape must be a non-empty tuple; got {self.shape!r}"
            )
        if any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError(
                f"adapter tensor {self.name!r} shape dimensions must be positive integers; "
                f"got {self.shape!r}"
            )
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"adapter tensor {self.name!r} has unsupported storage dtype: {self.dtype!r}"
            )
        if type(self.trainable) is not bool:
            raise ValueError(
                f"adapter tensor {self.name!r} trainable must be boolean; got {self.trainable!r}"
            )

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def storage_bytes(self) -> int:
        return self.numel * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class MemoryPart:
    """Bytes charged separately to host RAM and CUDA memory."""

    cpu_bytes: int = 0
    cuda_bytes: int = 0

    def __post_init__(self) -> None:
        for field, value in (("cpu_bytes", self.cpu_bytes), ("cuda_bytes", self.cuda_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"adapter memory {field} must be a non-negative integer; got {value!r}"
                )

    def __add__(self, other: "MemoryPart") -> "MemoryPart":
        return MemoryPart(self.cpu_bytes + other.cpu_bytes, self.cuda_bytes + other.cuda_bytes)


@dataclass(frozen=True)
class AdapterBudget:
    profile: str
    stored_parameters: int
    trainable_parameters: int
    weights: MemoryPart
    gradients: MemoryPart
    optimizer: MemoryPart | None
    unsupported_reason: str | None = None

    @property
    def total(self) -> MemoryPart | None:
        """Return None if the persistent optimizer state size is unknown."""
        if self.optimizer is None:
            return None
        return self.weights + self.gradients + self.optimizer


def _has_lorafa_pairs(tensors: tuple[AdapterTensor, ...]) -> bool:
    # Validate rank and freezing before charging state for B only.
    a_ranks = {}
    b_ranks = {}
    for tensor in tensors:
        is_a = ".lora_A." in tensor.name
        is_b = ".lora_B." in tensor.name
        if is_a == is_b or len(tensor.shape) != 2 or tensor.trainable != is_b:
            return False
        if is_a:
            a_ranks[tensor.name.replace(".lora_A.", ".lora_B.", 1)] = tensor.shape[0]
        else:
            b_ranks[tensor.name] = tensor.shape[1]
    return bool(a_ranks) and a_ranks == b_ranks


def estimate_adapter_budget(
    tensors: Iterable[AdapterTensor],
    *,
    profile: str,
    device: Literal["cpu", "cuda"],
) -> AdapterBudget:
    """Budget persistent memory assuming every trainable tensor gets a gradient.

    The estimate excludes base weights, activations, optimizer workspaces,
    Python objects and allocator rounding. Non-capturable Torch profiles keep
    scalar steps on CPU. The bitsandbytes profile includes shared maps once and
    charges block scales per tensor.
    """
    if not isinstance(device, str) or device not in ("cpu", "cuda"):
        raise ValueError(f"adapter budget device must be cpu or cuda; got {device!r}")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"adapter budget profile must be a non-empty string; got {profile!r}")
    try:
        tensors = tuple(tensors)
    except TypeError as exc:
        raise ValueError(
            "adapter budget tensors must be an iterable of AdapterTensor records"
        ) from exc
    if not tensors:
        raise ValueError("empty adapter profile: check target resolution")
    names: set[str] = set()
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, AdapterTensor):
            raise ValueError(
                f"adapter budget tensors[{index}] must be an AdapterTensor; "
                f"got {type(tensor).__name__}"
            )
        if tensor.name in names:
            raise ValueError(
                f"duplicate adapter tensor name {tensor.name!r}; deduplicate physical parameters"
            )
        names.add(tensor.name)
    trainable = tuple(tensor for tensor in tensors if tensor.trainable)

    def on_device(size: int) -> MemoryPart:
        return MemoryPart(cpu_bytes=size) if device == "cpu" else MemoryPart(cuda_bytes=size)

    weights = on_device(sum(tensor.storage_bytes for tensor in tensors))
    gradients = on_device(sum(tensor.storage_bytes for tensor in trainable))
    reason = None
    if profile not in _PROFILES:
        reason = f"unverified optimizer profile: {profile!r}"
    elif any(tensor.dtype != "float32" for tensor in tensors):
        reason = "optimizer profiles currently require materialized float32 adapters"
    elif profile == "bnb_0.50.2_adamw8bit_fp32" and device != "cuda":
        reason = "this bitsandbytes profile was verified on CUDA only"
    elif profile == "peft_lorafa_fp32" and not _has_lorafa_pairs(tensors):
        reason = (
            "LoRA-FA needs complete 2D A/B pairs with matching rank, A frozen and B trainable"
        )

    optimizer = None
    if reason is None:
        state_bytes = 0
        host_bytes = 0
        quantized_state = False
        for tensor in trainable:
            count = tensor.numel
            if profile == "torch_adamw_fp32":
                state_bytes += 8 * count
                host_bytes += 4
            elif profile in ("torch_adagrad_fp32", "torch_rmsprop_fp32"):
                state_bytes += 4 * count
                host_bytes += 4
            elif profile == "hf_adafactor_fp32":
                # HF factors the final two axes; a vector uses a full accumulator.
                shape = tensor.shape
                elements = (
                    prod(shape[:-1]) + prod(shape[:-2]) * shape[-1] if len(shape) >= 2 else count
                )
                state_bytes += 4 * elements + 4  # RMS scalar; step is a Python int
            elif profile == "peft_lorafa_fp32":
                state_bytes += 8 * count  # Two B moments; step is a Python int.
            elif profile == "bnb_0.50.2_adamw8bit_fp32":
                if count < 4096:
                    state_bytes += 8 * count
                else:
                    quantized_state = True
                    state_bytes += 2 * count + 8 * ((count + 255) // 256)
        if quantized_state:
            state_bytes += 2 * 256 * 4  # Shared signed/unsigned float32 quantization maps.
        optimizer = on_device(state_bytes) + MemoryPart(cpu_bytes=host_bytes)

    return AdapterBudget(
        profile=profile,
        stored_parameters=sum(tensor.numel for tensor in tensors),
        trainable_parameters=sum(tensor.numel for tensor in trainable),
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
        unsupported_reason=reason,
    )
