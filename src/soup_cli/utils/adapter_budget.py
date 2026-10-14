"""Persistent adapter tensor budgets from resolved shapes, without loading weights.

This is an arithmetic primitive, not a fit verdict or a peak-memory predictor.
The caller must supply distinct physical adapter tensors AFTER target resolution,
PEFT conversion and freezing, and their eventual materialized dtype (not the base
model dtype). Include frozen adapters too. Aliased parameters must appear once.

Profile names describe verified effective optimizer settings, not arbitrary YAML
optimizer names. Selection/validation against the actual training path belongs to
the caller. No streaming integration is enabled by importing this module.
"""

from dataclasses import dataclass
from math import prod
from typing import Literal, Optional, Sequence

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
_PROFILES = frozenset({
    "torch_adamw_fp32",  # non-fused, non-capturable, amsgrad=False
    "torch_sgd_fp32",  # momentum=0
    "torch_adagrad_fp32",  # non-fused
    "torch_rmsprop_fp32",  # momentum=0, centered=False, non-capturable
    "hf_adafactor_fp32",  # beta1=None
    "peft_lorafa_fp32",  # ordinary LoRA, A already frozen, B trainable
    "bnb_0.50.2_adamw8bit_fp32",  # non-paged, blockwise, min_8bit_size=4096
})


@dataclass(frozen=True)
class AdapterTensor:
    """Metadata only; dtype is the eventual adapter storage dtype."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("adapter tensor needs a non-empty name")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("adapter shape must be a non-empty tuple")
        if any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError("adapter dimensions must be positive integers")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported adapter storage dtype: {self.dtype!r}")
        if type(self.trainable) is not bool:
            raise ValueError("trainable must be boolean")

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def storage_bytes(self) -> int:
        return self.numel * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class MemoryPart:
    """Host RAM and CUDA memory are separate budgets, never interchangeable."""

    cpu_bytes: int = 0
    cuda_bytes: int = 0

    def __add__(self, other: "MemoryPart") -> "MemoryPart":
        return MemoryPart(self.cpu_bytes + other.cpu_bytes, self.cuda_bytes + other.cuda_bytes)


@dataclass(frozen=True)
class AdapterBudget:
    profile: str
    stored_parameters: int
    trainable_parameters: int
    weights: MemoryPart
    gradients: MemoryPart
    optimizer: Optional[MemoryPart]
    unsupported_reason: Optional[str] = None

    @property
    def total(self) -> Optional[MemoryPart]:
        """None means unknown, not zero or a successful fit."""
        if self.optimizer is None:
            return None
        return self.weights + self.gradients + self.optimizer


def estimate_adapter_budget(
    tensors: Sequence[AdapterTensor],
    *,
    profile: str,
    device: Literal["cpu", "cuda"],
) -> AdapterBudget:
    """Budget persistent tensors assuming all trainable adapters receive gradients.

    Excludes base weights, activations, optimizer workspaces, Python objects and
    allocator rounding. CPU scalar step tensors stay on CPU for the supported
    non-capturable Torch profiles. Other devices/options need separate profiles.
    The bnb profile includes shared maps once and per-tensor block scales.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("adapter budget device must be cpu or cuda")
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("empty adapter profile: check target resolution")
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise ValueError("duplicate adapter tensor names; deduplicate physical parameters")
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
    elif profile.startswith("bnb_") and device != "cuda":
        reason = "this bitsandbytes profile was verified on CUDA only"
    elif profile == "peft_lorafa_fp32":
        # Require full A/B pairs, not only requires_grad tensors: A still consumes memory.
        a_names = {tensor.name.replace(".lora_A.", ".lora_B.")
                   for tensor in tensors if ".lora_A." in tensor.name}
        b_names = {tensor.name for tensor in tensors if ".lora_B." in tensor.name}
        if (not a_names or a_names != b_names or any(
            len(tensor.shape) != 2
            or (".lora_A." not in tensor.name and ".lora_B." not in tensor.name)
            or tensor.trainable != (".lora_B." in tensor.name)
            for tensor in tensors
        )):
            reason = "LoRA-FA needs complete 2D A/B pairs with A frozen and B trainable"

    state_bytes = 0
    host_bytes = 0
    if reason is None:
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
                elements = (prod(shape[:-1]) + prod(shape[:-2]) * shape[-1]
                            if len(shape) >= 2 else count)
                state_bytes += 4 * elements + 4  # RMS scalar; step is a Python int
            elif profile == "peft_lorafa_fp32":
                state_bytes += 8 * count  # two B moments; step is a Python int
            elif profile == "bnb_0.50.2_adamw8bit_fp32":
                if count < 4096:
                    state_bytes += 8 * count
                else:
                    quantized_state = True
                    state_bytes += 2 * count + 8 * ((count + 255) // 256)
        if quantized_state:
            state_bytes += 2 * 256 * 4  # shared signed/unsigned fp32 quantization maps

    optimizer = None if reason else on_device(state_bytes) + MemoryPart(cpu_bytes=host_bytes)
    return AdapterBudget(
        profile=profile,
        stored_parameters=sum(tensor.numel for tensor in tensors),
        trainable_parameters=sum(tensor.numel for tensor in trainable),
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
        unsupported_reason=reason,
    )
