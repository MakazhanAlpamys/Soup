"""Tests for v0.37.0 Part C — neat_packing 4D attention mask.

Covers:
- ``tag_sub_sequences`` — emit 1-indexed segment IDs per token
- ``build_4d_attention_mask`` — float ``(B, 1, S, S)`` mask shape, dtype,
  same-segment + causal semantics, padding handling
- ``select_packing_strategy`` — FA varlen when flash-attn available, 4D
  mask otherwise; fail-fast on unknown caller intent
"""

from __future__ import annotations

import numpy as np
import pytest

from soup_cli.utils.neat_packing import (
    build_4d_attention_mask,
    select_packing_strategy,
    tag_sub_sequences,
)

# ---- tag_sub_sequences ---------------------------------------------------


def test_tag_single_sequence():
    # one document of length 5 → all tokens get segment ID 1
    assert tag_sub_sequences([0, 5]) == [1, 1, 1, 1, 1]


def test_tag_three_sequences():
    # boundaries [0, 3, 5, 9] → docs of length 3, 2, 4
    assert tag_sub_sequences([0, 3, 5, 9]) == [1, 1, 1, 2, 2, 3, 3, 3, 3]


def test_tag_rejects_empty_boundaries():
    with pytest.raises(ValueError, match="non-empty"):
        tag_sub_sequences([])


def test_tag_single_token_document():
    # Minimum non-trivial case — boundary [0, 1] = 1 doc of 1 token.
    assert tag_sub_sequences([0, 1]) == [1]


def test_tag_rejects_zero_documents():
    # Single boundary [0] → 0 documents — likely a logic error, fail loudly.
    with pytest.raises(ValueError, match="at least 2 entries"):
        tag_sub_sequences([0])


def test_tag_rejects_non_zero_start():
    with pytest.raises(ValueError, match="must start at 0"):
        tag_sub_sequences([1, 5])


def test_tag_rejects_non_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        tag_sub_sequences([0, 5, 5, 9])


def test_tag_rejects_decreasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        tag_sub_sequences([0, 5, 3])


# ---- build_4d_attention_mask ---------------------------------------------


def test_4d_mask_shape_and_dtype():
    # seq_pos_ids: (B, S) = (1, 5) — single doc
    seq_ids = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
    mask = build_4d_attention_mask(seq_ids, dtype=np.float32)
    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == np.float32


def test_4d_mask_single_sequence_is_lower_triangular():
    seq_ids = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
    mask = build_4d_attention_mask(seq_ids, dtype=np.float32)
    # Token i can attend to token j iff i >= j → 0; else -inf-like
    plane = mask[0, 0]
    for i in range(5):
        for j in range(5):
            if i >= j:
                assert plane[i, j] == 0.0
            else:
                assert plane[i, j] < -1e9  # -inf marker


def test_4d_mask_blocks_cross_segment():
    # Two docs in one sequence: doc1=[0,1,2], doc2=[3,4]
    seq_ids = np.array([[1, 1, 1, 2, 2]], dtype=np.int32)
    mask = build_4d_attention_mask(seq_ids, dtype=np.float32)
    plane = mask[0, 0]
    # Cross-segment blocked
    assert plane[3, 0] < -1e9
    assert plane[3, 2] < -1e9
    assert plane[4, 0] < -1e9
    # Token 3 CAN attend to itself
    assert plane[3, 3] == 0.0
    # Token 4 attends to 3 (same doc, causal)
    assert plane[4, 3] == 0.0
    # Intra-segment causal — verify the full causal rule for both segments.
    # Doc1 (positions 0,1,2): positions only see preceding same-doc tokens.
    for i in range(3):
        for j in range(3):
            if i >= j:
                assert plane[i, j] == 0.0, f"intra-doc1 ({i},{j}) blocked"
            else:
                assert plane[i, j] < -1e9, f"intra-doc1 future ({i},{j}) leak"
    # Token 1 cannot attend forward to token 2 (intra-segment, future).
    assert plane[1, 2] < -1e9


def test_4d_mask_padding_segment_zero():
    # Convention: segment ID 0 = padding → token cannot attend to or be
    # attended to by anyone (full -inf row + col, including the diagonal —
    # padding queries are masked out entirely so softmax behaviour is
    # well-defined).
    seq_ids = np.array([[1, 1, 0, 0]], dtype=np.int32)
    mask = build_4d_attention_mask(seq_ids, dtype=np.float32)
    plane = mask[0, 0]
    # Padding tokens (rows 2,3) — every cell including diagonal is masked.
    for pad_row in (2, 3):
        for j in range(4):
            assert plane[pad_row, j] < -1e9, (
                f"padding row {pad_row} col {j} should be masked"
            )
    # No real token can attend to padding tokens (cols 2,3)
    for i in range(4):
        if i not in (2, 3):
            assert plane[i, 2] < -1e9
            assert plane[i, 3] < -1e9


def test_4d_mask_batch_dim():
    # Two batch elements with different segment layouts
    seq_ids = np.array([
        [1, 1, 2, 2],
        [1, 2, 2, 3],
    ], dtype=np.int32)
    mask = build_4d_attention_mask(seq_ids, dtype=np.float32)
    assert mask.shape == (2, 1, 4, 4)
    # Element 0: token 2 (doc2) can't see token 0 (doc1)
    assert mask[0, 0, 2, 0] < -1e9
    # Element 1: token 3 (doc3) can't see token 0 (doc1)
    assert mask[1, 0, 3, 0] < -1e9
    # Element 1: token 2 attends to token 1 (same doc2)
    assert mask[1, 0, 2, 1] == 0.0


def test_4d_mask_rejects_non_2d_seq_ids():
    seq_ids = np.array([1, 1, 2], dtype=np.int32)  # 1D
    with pytest.raises(ValueError, match="2D"):
        build_4d_attention_mask(seq_ids, dtype=np.float32)


def test_4d_mask_rejects_negative_segment_id():
    seq_ids = np.array([[1, -1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        build_4d_attention_mask(seq_ids, dtype=np.float32)


def test_4d_mask_rejects_oversize_allocation():
    # Defence against (B, S, S) OOM — cap rejects too-large allocations.
    import soup_cli.utils.neat_packing as np_mod
    original = np_mod._MAX_MASK_ELEMENTS
    try:
        np_mod._MAX_MASK_ELEMENTS = 10
        seq_ids = np.array([[1, 1, 1, 1]], dtype=np.int32)  # 1*4*4 = 16 cells
        with pytest.raises(ValueError, match="exceeding cap"):
            build_4d_attention_mask(seq_ids, dtype=np.float32)
    finally:
        np_mod._MAX_MASK_ELEMENTS = original


def test_tag_rejects_too_many_segments():
    import soup_cli.utils.neat_packing as np_mod
    original = np_mod._MAX_BOUNDARY_SEGMENTS
    try:
        np_mod._MAX_BOUNDARY_SEGMENTS = 2
        # 4 boundaries → 3 segments → exceeds cap of 2
        with pytest.raises(ValueError, match="too many segments"):
            tag_sub_sequences([0, 1, 2, 3])
    finally:
        np_mod._MAX_BOUNDARY_SEGMENTS = original


def test_4d_mask_dtype_choice():
    seq_ids = np.array([[1, 1, 2, 2]], dtype=np.int32)
    mask_f16 = build_4d_attention_mask(seq_ids, dtype=np.float16)
    assert mask_f16.dtype == np.float16


# ---- select_packing_strategy ---------------------------------------------


def test_strategy_prefers_fa_when_available():
    assert select_packing_strategy(flash_attn_available=True) == "varlen"


def test_strategy_falls_back_to_4d_mask():
    assert select_packing_strategy(flash_attn_available=False) == "4d_mask"


def test_strategy_rejects_non_bool():
    with pytest.raises(TypeError, match="must be bool"):
        select_packing_strategy(flash_attn_available=1)  # type: ignore[arg-type]


def test_strategy_rejects_none():
    with pytest.raises(TypeError, match="must be bool"):
        select_packing_strategy(flash_attn_available=None)  # type: ignore[arg-type]
