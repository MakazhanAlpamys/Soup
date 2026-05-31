"""neat_packing — 4D attention mask + FA/4D packing-strategy picker.

Mirrors LlamaFactory's "neat_packing" mode (`processor/supervised.py:214`,
`collator.py:93`):

1. Each token in a packed sequence carries a 1-indexed *segment ID* — all
   tokens from doc N share segment N. Padding tokens get segment 0.
2. The collator builds a 4D attention mask ``(B, 1, S, S)`` where
   ``mask[b, 0, i, j] = 0.0`` iff tokens i and j are in the SAME segment
   AND ``i >= j`` (causal); otherwise a large negative number is written
   to make the softmax effectively zero out that pair.

When FlashAttention is available, the trainer should prefer the varlen
path (``cu_seqlens``) — it is faster and uses less memory. The 4D mask
path here is the fallback for backends without FA support.

Compared to v0.28.0 :mod:`soup_cli.utils.cross_doc_attn`:

* That module produces a 2D ``(S, S)`` uint8 mask intended for the HF
  ``attention_mask`` tensor (boolean).
* This module produces a 4D float mask intended to be added directly into
  the attention logits — matching the LlamaFactory / Axolotl interface.
* They compose: cross_doc_attn picks WHICH boundaries; this module
  expresses those boundaries in the float-additive shape that newer HF
  kernels accept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np


def tag_sub_sequences(boundaries: Sequence[int]) -> list[int]:
    """Convert document boundaries into per-token 1-indexed segment IDs.

    Example: ``[0, 3, 5, 9]`` (3 docs of length 3, 2, 4) →
    ``[1, 1, 1, 2, 2, 3, 3, 3, 3]``.

    Args:
        boundaries: strictly-increasing sequence starting at 0. Length N+1
            for N documents.

    Returns:
        A list of segment IDs of length ``boundaries[-1]``.

    Raises:
        ValueError: if boundaries are empty, do not start at 0, or are
            not strictly increasing.
    """
    if not boundaries:
        raise ValueError("boundaries must be non-empty")
    if len(boundaries) > _MAX_BOUNDARY_SEGMENTS + 1:
        raise ValueError(
            f"too many segments ({len(boundaries) - 1}); cap is "
            f"{_MAX_BOUNDARY_SEGMENTS}"
        )
    if len(boundaries) < 2:
        raise ValueError(
            "boundaries must contain at least 2 entries (one document); "
            f"got {len(boundaries)} entries"
        )
    if boundaries[0] != 0:
        raise ValueError(
            f"boundaries must start at 0, got {boundaries[0]}"
        )
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] >= boundaries[idx + 1]:
            raise ValueError(
                "boundaries must be strictly increasing, "
                f"got {boundaries[idx]} >= {boundaries[idx + 1]} "
                f"at index {idx}"
            )

    seq_ids: list[int] = []
    for doc_idx in range(len(boundaries) - 1):
        length = boundaries[doc_idx + 1] - boundaries[doc_idx]
        seq_ids.extend([doc_idx + 1] * length)
    return seq_ids


# Caps to prevent resource-exhaustion on adversarial inputs. Both bounds are
# generous relative to realistic training workloads (max DataConfig.max_length
# is 1_048_576) but defend against accidental B*S^2 allocations >2GB.
_MAX_MASK_ELEMENTS: int = 2**31  # ~2.1B float32 cells = 8GB cap
_MAX_BOUNDARY_SEGMENTS: int = 1_000_000


def _neg_inf_for(dtype: np.dtype) -> float:
    """Largest negative finite value representable in ``dtype``.

    Avoids the ``RuntimeWarning: overflow`` that fires when a fp32 sentinel
    is cast down to fp16. Mirrors HF's convention in
    ``modeling_attn_mask_utils.py``.
    """
    import numpy as np

    return float(np.finfo(dtype).min)


def build_4d_attention_mask(
    seq_pos_ids: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Build a ``(B, 1, S, S)`` additive attention mask from segment IDs.

    Args:
        seq_pos_ids: integer array of shape ``(B, S)`` where entry
            ``[b, i]`` is the 1-indexed segment ID for token i in batch
            element b. Segment ID 0 is reserved for padding (token will
            be fully masked, both as query and key).
        dtype: float dtype of the output mask (``np.float32`` /
            ``np.float16`` / ``np.float64``).

    Returns:
        ``(B, 1, S, S)`` array — 0 where attention is allowed, large
        negative number where blocked. Suitable for direct addition to
        attention logits.

    Raises:
        ValueError: if input is not 2D or contains negative IDs.
    """
    import numpy as np

    if seq_pos_ids.ndim != 2:
        raise ValueError(
            f"seq_pos_ids must be 2D (B, S), got shape {seq_pos_ids.shape}"
        )
    if (seq_pos_ids < 0).any():
        raise ValueError("seq_pos_ids must be non-negative")

    elements = int(seq_pos_ids.shape[0]) * int(seq_pos_ids.shape[1]) ** 2
    if elements > _MAX_MASK_ELEMENTS:
        raise ValueError(
            f"requested 4D mask ({seq_pos_ids.shape[0]}, 1, "
            f"{seq_pos_ids.shape[1]}, {seq_pos_ids.shape[1]}) would allocate "
            f"{elements} cells, exceeding cap {_MAX_MASK_ELEMENTS}. "
            "Use the FlashAttention varlen path or smaller max_length."
        )

    if not np.issubdtype(dtype, np.floating):
        raise TypeError(
            f"dtype must be a numpy float dtype, got {dtype}"
        )
    batch, seq_len = seq_pos_ids.shape
    # Same-segment matrix: (B, S, S) bool — True iff query and key share id.
    same_segment = (
        seq_pos_ids[:, :, None] == seq_pos_ids[:, None, :]
    )
    # Padding (id=0) cannot attend to anyone, including itself.
    is_real = (seq_pos_ids != 0)
    # (B, S, S) — both query and key must be real tokens.
    real_pair = is_real[:, :, None] & is_real[:, None, :]
    # Causal: (S, S) lower-triangular True.
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    # Combine: allowed iff same segment AND both real AND causal.
    allowed = same_segment & real_pair & causal[None, :, :]
    # Build additive mask: 0.0 where allowed, dtype-specific min where blocked.
    neg_inf = _neg_inf_for(dtype)
    mask = np.where(allowed, 0.0, neg_inf).astype(dtype)
    # Add the leading "head" dim → (B, 1, S, S).
    return mask[:, None, :, :]


PackingStrategy = Literal["varlen", "4d_mask"]


def select_packing_strategy(*, flash_attn_available: bool) -> PackingStrategy:
    """Pick FA varlen path when available, otherwise 4D mask fallback.

    The varlen path is faster (no S² memory) but requires
    ``flash_attn>=2.0``. The 4D mask path matches every HF model with
    SDPA / eager attention.
    """
    if not isinstance(flash_attn_available, bool):
        raise TypeError(
            "flash_attn_available must be bool, got "
            f"{type(flash_attn_available).__name__}"
        )
    return "varlen" if flash_attn_available else "4d_mask"
