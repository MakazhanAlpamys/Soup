"""MultipackBatchSampler — First-Fit-Decreasing bin packing for SFT/pretrain.

Ports Axolotl's ``MultipackBatchSampler`` (utils/samplers/multipack.py:24-57)
without the numba runtime dependency. The pure-Python FFD here is fast enough
for typical dataset sizes (<1M samples); a future ``[multipack]`` extras can
add an optional numba JIT path keyed off the same ``ffd_bin_pack`` signature.

Compared with Axolotl, two intentional differences:

1. **Loud-fail on unknown architecture.** Axolotl's monkey-patch silently
   misses architectures absent from its allowlist
   (`monkeypatch/multipack.py:13-66`). Soup raises ``ValueError`` so the user
   knows multipack is not active. Mirrors the v0.33.0 #43 schema-gate policy.
2. **bool rejection on numeric inputs.** Mirrors v0.30.0 ``Candidate``
   policy — bool is a subclass of int and must not silently coerce.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Union

# Allow-list of HF model architectures that support the FA varlen path
# (``_get_unpad_data`` monkey-patch). Mirrors Axolotl's list plus the v0.31.0
# Soup recipe expansion. Keep frozenset to prevent runtime mutation.
MULTIPACK_ARCHITECTURES: frozenset[str] = frozenset({
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "QwenForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "Phi4ForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "FalconForCausalLM",
    "StableLmForCausalLM",
    "SmolLM2ForCausalLM",
})


# Worst-case FFD complexity is O(N^2). Cap N to prevent a crafted dataset
# from pinning a CPU. 1M samples is ~3 orders of magnitude beyond typical
# fine-tuning workloads.
_MAX_FFD_ITEMS: int = 1_000_000


def _check_int(name: str, value: object) -> int:
    """Reject bool, non-int, and return as int. Mirrors v0.30.0 policy."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool, got {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def ffd_bin_pack(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """First-Fit-Decreasing bin packing.

    Args:
        lengths: per-sample sequence lengths.
        max_len: maximum sum of lengths in any bin.

    Returns:
        A list of bins; each bin is a list of original indices into
        ``lengths``. Every index appears exactly once.

    Raises:
        ValueError: if any length is non-positive, exceeds ``max_len``,
            or if ``max_len`` is non-positive.
        TypeError: if ``max_len`` is bool.
    """
    _check_int("max_len", max_len)
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    # Materialise once — protects against a generator being exhausted by the
    # validation pass, which would silently produce empty bins on the sort.
    lengths = list(lengths)
    if not lengths:
        return []
    if len(lengths) > _MAX_FFD_ITEMS:
        raise ValueError(
            f"too many items for FFD bin-packing: {len(lengths)} > "
            f"{_MAX_FFD_ITEMS} (algorithm is O(N^2) worst-case)"
        )

    # Validate each length up-front so we fail loudly before packing.
    for idx, length in enumerate(lengths):
        _check_int(f"lengths[{idx}]", length)
        if length <= 0:
            raise ValueError(
                f"lengths[{idx}] must be positive, got {length}"
            )
        if length > max_len:
            raise ValueError(
                f"lengths[{idx}]={length} exceeds max_len={max_len}"
            )

    # Pair each length with its original index, then sort by length descending.
    indexed = sorted(
        enumerate(lengths), key=lambda pair: pair[1], reverse=True,
    )

    bins: list[list[int]] = []
    bin_remaining: list[int] = []  # parallel to bins: free space in each
    for orig_idx, length in indexed:
        placed = False
        for bin_idx, remaining in enumerate(bin_remaining):
            if length <= remaining:
                bins[bin_idx].append(orig_idx)
                bin_remaining[bin_idx] = remaining - length
                placed = True
                break
        if not placed:
            bins.append([orig_idx])
            bin_remaining.append(max_len - length)

    return bins


def validate_multipack_architecture(arch_name: str) -> None:
    """Raise ``ValueError`` if ``arch_name`` is not multipack-supported.

    Loud-fail policy (vs Axolotl's silent miss) — see module docstring.
    """
    if not isinstance(arch_name, str):
        raise TypeError(
            f"arch_name must be str, got {type(arch_name).__name__}"
        )
    if not arch_name:
        raise ValueError("arch_name must be non-empty")
    if "\x00" in arch_name:
        raise ValueError("arch_name must not contain null bytes")
    if arch_name not in MULTIPACK_ARCHITECTURES:
        supported = ", ".join(sorted(MULTIPACK_ARCHITECTURES))
        raise ValueError(
            f"Architecture {arch_name!r} not in multipack allowlist. "
            f"Either add it to MULTIPACK_ARCHITECTURES or set "
            f"multipack: false in your config. Supported: {supported}"
        )


# Type alias — a real_batches=False yield is a flat list of indices,
# real_batches=True is a list of bins (each bin a list of indices).
BatchType = Union[list[int], list[list[int]]]


class MultipackBatchSampler:
    """Yield batches of indices packed via FFD into bins of ``batch_max_len``.

    Two modes:

    * ``real_batches=False`` — each yielded value is a flat list of indices
      (one bin), to be flattened by the collator into a single packed
      sequence. Axolotl's ``multipack_real_batches=False`` trick.
    * ``real_batches=True`` — each yielded value is a list of bins
      (length ``<= batch_size``), where each bin is a list of indices.
      The collator stacks bins along the batch dimension.

    Determinism: a fixed ``seed`` produces a stable batch order across runs.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_max_len: int,
        batch_size: int,
        real_batches: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        batch_max_len = _check_int("batch_max_len", batch_max_len)
        batch_size = _check_int("batch_size", batch_size)
        seed = _check_int("seed", seed)
        if batch_max_len <= 0:
            raise ValueError(
                f"batch_max_len must be positive, got {batch_max_len}"
            )
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )
        if not lengths:
            raise ValueError("lengths must be non-empty")

        # Pack once at construction so __len__ is exact and iteration is
        # deterministic. Pre-pack validation in ffd_bin_pack catches
        # over-length items.
        self._lengths: list[int] = list(lengths)
        self._batch_max_len = batch_max_len
        self._batch_size = batch_size
        self._real_batches = bool(real_batches)
        self._seed = seed
        self._drop_last = bool(drop_last)

        bins = ffd_bin_pack(self._lengths, batch_max_len)
        # Shuffle the packed bins deterministically — keeps inter-bin order
        # randomised across epochs while preserving the FFD packing.
        rng = random.Random(seed)
        rng.shuffle(bins)
        self._bins: list[list[int]] = bins

    def __len__(self) -> int:
        if not self._real_batches:
            return len(self._bins)
        full = len(self._bins) // self._batch_size
        remainder = len(self._bins) % self._batch_size
        if self._drop_last or remainder == 0:
            return full
        return full + 1

    def __iter__(self) -> Iterator[BatchType]:
        if not self._real_batches:
            for bin_ in self._bins:
                yield list(bin_)
            return

        batch: list[list[int]] = []
        for bin_ in self._bins:
            batch.append(list(bin_))
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch and not self._drop_last:
            yield batch
