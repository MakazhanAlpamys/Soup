"""Cross-document attention masking for sample packing.

When multiple short documents are packed into a single sequence for faster
training (``training.packing: true``), the default causal mask allows tokens
in doc N to attend to doc N-1 — leaking unrelated context across unrelated
samples. This module builds a block-diagonal causal mask that prevents
attention from crossing document boundaries.

Axolotl, Unsloth, and recent TRL versions all support this. Our
implementation is a pure-numpy mask builder that plugs into the HF data
collator via the ``attention_mask`` tensor. The trainer wrapper is
responsible for attaching it during batch collation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def build_cross_doc_mask(
    boundaries: list[int], seq_length: int,
) -> "np.ndarray":
    """Build a block-diagonal causal attention mask.

    Args:
        boundaries: Sorted document boundary indices in ``[0, seq_length]``
            with ``boundaries[0] == 0`` and ``boundaries[-1] == seq_length``.
            Document N occupies positions ``boundaries[N] .. boundaries[N+1]-1``.
        seq_length: Total packed sequence length.

    Returns:
        A ``(seq_length, seq_length)`` uint8 numpy array where ``mask[i, j]``
        is 1 iff token i can attend to token j (same document *and* causal).

    Raises:
        ValueError: if boundaries are malformed.
    """
    import numpy as np

    if not boundaries:
        raise ValueError("boundaries must be non-empty")
    if boundaries[0] != 0:
        raise ValueError(
            f"boundaries must start at 0, got boundaries[0]={boundaries[0]}"
        )
    if boundaries[-1] != seq_length:
        raise ValueError(
            f"boundaries must end at seq_length={seq_length}, "
            f"got boundaries[-1]={boundaries[-1]}"
        )
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] >= boundaries[idx + 1]:
            raise ValueError(
                f"boundaries must be strictly increasing, "
                f"got {boundaries[idx]} >= {boundaries[idx+1]} at index {idx}"
            )

    mask = np.zeros((seq_length, seq_length), dtype=np.uint8)
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        block_size = end - start
        # Lower-triangular block (causal within this document) — vectorised to
        # avoid a pure-Python O(block_size**2) inner loop on long sequences.
        mask[start:end, start:end] = np.tril(
            np.ones((block_size, block_size), dtype=np.uint8)
        )
    return mask


def compute_doc_boundaries(document_lengths: list[int]) -> list[int]:
    """Convert a list of per-doc lengths into boundary positions.

    Example: ``[3, 2, 4]`` -> ``[0, 3, 5, 9]`` (seq_length=9).
    """
    if not document_lengths:
        raise ValueError("document_lengths must be non-empty")
    for length in document_lengths:
        if length <= 0:
            raise ValueError(
                f"document lengths must be positive, got {length}"
            )

    boundaries = [0]
    running = 0
    for length in document_lengths:
        running += length
        boundaries.append(running)
    return boundaries
