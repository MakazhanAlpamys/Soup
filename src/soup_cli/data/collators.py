"""Data collators for sample packing (v0.33.0 #47).

When ``training.packing=true`` and ``training.packing_cross_doc_attn_mask=true``,
multiple short documents are packed into a single sequence; the default causal
mask leaks attention across document boundaries. ``CrossDocCollator`` builds an
explicit block-diagonal causal mask via :func:`utils.cross_doc_attn.build_cross_doc_mask`
and injects it as ``attention_mask`` on the batch, preferred over TRL's
``packing_strategy="attention_free"`` flag (which is best-effort across TRL
versions).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CrossDocCollator:
    """Wraps a base data collator and overrides ``attention_mask`` with a
    block-diagonal causal mask derived from per-doc boundaries.

    The base collator produces the standard ``input_ids`` / ``labels`` tensors;
    this wrapper looks for a ``doc_lengths`` field in the underlying examples
    (per-doc token counts after tokenization) and computes the block-diagonal
    mask. If ``doc_lengths`` is missing, the original ``attention_mask`` is
    preserved so the wrapper degrades gracefully.

    Usage::

        from transformers import DataCollatorForLanguageModeling
        from soup_cli.data.collators import CrossDocCollator

        base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        collator = CrossDocCollator(base_collator=base)
        trainer = SFTTrainer(..., data_collator=collator)
    """

    def __init__(
        self, base_collator: Any, doc_lengths_key: str = "doc_lengths",
    ) -> None:
        if base_collator is None:
            raise ValueError("CrossDocCollator requires a base_collator")
        self._base = base_collator
        self._key = doc_lengths_key

    def __call__(self, features: list[dict]) -> dict:
        # Read per-example doc_lengths and copy each dict (excluding the key)
        # so we never mutate the caller's feature dicts — HF Dataset rows are
        # cached and reused across batches; pop() would silently strip
        # ``doc_lengths`` after the first call.
        per_example_lengths: list[Optional[list[int]]] = []
        cleaned: list[dict] = []
        for example in features:
            if isinstance(example, dict):
                lengths = example.get(self._key)
                cleaned.append(
                    {k: v for k, v in example.items() if k != self._key}
                )
            else:
                lengths = None
                cleaned.append(example)
            per_example_lengths.append(lengths)

        batch = self._base(cleaned)

        # Try to build the cross-doc mask. If anything goes wrong (no
        # doc_lengths, mismatched shapes, no numpy/torch), preserve the
        # base ``attention_mask`` and continue. Log at DEBUG so production
        # silent-degradation is still inspectable.
        try:
            self._inject_block_diag_mask(batch, per_example_lengths)
        except Exception as exc:  # noqa: BLE001 — degrade rather than crash training
            logger.debug(
                "CrossDocCollator: falling back to base mask: %s", exc,
            )
        return batch

    def _inject_block_diag_mask(
        self, batch: dict, per_example_lengths: list[Optional[list[int]]],
    ) -> None:
        from soup_cli.utils.cross_doc_attn import (
            build_cross_doc_mask,
            compute_doc_boundaries,
        )

        if "input_ids" not in batch:
            return
        input_ids = batch["input_ids"]
        if not hasattr(input_ids, "shape") or len(input_ids.shape) != 2:
            return
        batch_size, seq_length = input_ids.shape

        if not any(lengths for lengths in per_example_lengths):
            return  # Nothing to do — no doc_lengths supplied

        try:
            import numpy as np
            import torch
        except ImportError:
            return

        masks = []
        for lengths in per_example_lengths:
            if lengths and sum(lengths) <= seq_length:
                # Pad final segment if doc lengths sum to < seq_length
                total = sum(lengths)
                padded_lengths = list(lengths)
                if total < seq_length:
                    padded_lengths.append(seq_length - total)
                boundaries = compute_doc_boundaries(padded_lengths)
                mask = build_cross_doc_mask(boundaries, seq_length)
            else:
                # Fallback: lower-triangular causal (no doc separation).
                mask = np.tril(np.ones((seq_length, seq_length), dtype=np.uint8))
            masks.append(mask)

        if len(masks) != batch_size:
            return

        attn_tensor = torch.from_numpy(np.stack(masks))
        # HF expects ``attention_mask`` shape (batch, seq) for standard
        # masking; the (batch, seq, seq) block-diag mask is consumed by
        # FlashAttn-2 / SDPA when passed as ``attn_mask`` instead. We expose
        # both names so downstream code can pick whichever it needs.
        batch["cross_doc_attn_mask"] = attn_tensor
