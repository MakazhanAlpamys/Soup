"""Tests for MultipackBatchSampler (v0.37.0 Part A).

Covers:
- ``ffd_bin_pack`` First-Fit-Decreasing algorithm correctness + invariants
- ``validate_multipack_architecture`` allowlist (loud-fail vs Axolotl)
- ``MultipackBatchSampler`` iter / len / determinism / real_batches modes
- bounds + bool rejection on numeric inputs
"""

from __future__ import annotations

import pytest

from soup_cli.utils.multipack_sampler import (
    MULTIPACK_ARCHITECTURES,
    MultipackBatchSampler,
    ffd_bin_pack,
    validate_multipack_architecture,
)

# ---- ffd_bin_pack ---------------------------------------------------------


def test_ffd_empty_returns_empty():
    assert ffd_bin_pack([], max_len=10) == []


def test_ffd_single_item_fits():
    assert ffd_bin_pack([5], max_len=10) == [[0]]


def test_ffd_packs_into_min_bins():
    # lengths [4, 3, 3, 2, 2] with max_len=6 => sorted desc = [4,3,3,2,2]
    # bin1: 4+2=6, bin2: 3+3=6, bin3: 2 — 3 bins
    bins = ffd_bin_pack([4, 3, 3, 2, 2], max_len=6)
    assert len(bins) == 3
    # Every original index appears exactly once across all bins
    flat = sorted(idx for b in bins for idx in b)
    assert flat == [0, 1, 2, 3, 4]
    # Each bin's total length <= max_len
    lengths = [4, 3, 3, 2, 2]
    for b in bins:
        assert sum(lengths[i] for i in b) <= 6


def test_ffd_full_coverage_invariant():
    # Property test: every index appears exactly once.
    import random
    rng = random.Random(42)
    lengths = [rng.randint(1, 20) for _ in range(100)]
    bins = ffd_bin_pack(lengths, max_len=32)
    flat = sorted(idx for b in bins for idx in b)
    assert flat == list(range(100))


def test_ffd_no_duplicates_across_packs():
    lengths = [5, 5, 5, 5, 5]
    bins = ffd_bin_pack(lengths, max_len=10)
    seen: set[int] = set()
    for b in bins:
        for idx in b:
            assert idx not in seen
            seen.add(idx)


def test_ffd_max_pack_len_invariant():
    lengths = [3, 7, 2, 8, 5, 1]
    max_len = 10
    bins = ffd_bin_pack(lengths, max_len=max_len)
    for b in bins:
        assert sum(lengths[i] for i in b) <= max_len


def test_ffd_rejects_item_larger_than_max():
    with pytest.raises(ValueError, match="exceeds max_len"):
        ffd_bin_pack([5, 15, 3], max_len=10)


def test_ffd_rejects_non_positive_length():
    with pytest.raises(ValueError, match="positive"):
        ffd_bin_pack([5, 0, 3], max_len=10)
    with pytest.raises(ValueError, match="positive"):
        ffd_bin_pack([5, -1, 3], max_len=10)


def test_ffd_rejects_non_positive_max_len():
    with pytest.raises(ValueError, match="max_len"):
        ffd_bin_pack([1, 2], max_len=0)
    with pytest.raises(ValueError, match="max_len"):
        ffd_bin_pack([1, 2], max_len=-5)


def test_ffd_rejects_bool_max_len():
    # bool is subclass of int; reject explicitly (matches v0.30.0 Candidate policy)
    with pytest.raises(TypeError, match="bool"):
        ffd_bin_pack([1, 2], max_len=True)


def test_ffd_all_lengths_equal_max_len():
    # Boundary: every item is exactly max_len → each gets its own bin.
    bins = ffd_bin_pack([10, 10, 10], max_len=10)
    assert len(bins) == 3
    flat = sorted(idx for b in bins for idx in b)
    assert flat == [0, 1, 2]


def test_ffd_rejects_too_many_items():
    # Defence against O(N^2) DoS — cap is 1M.
    from soup_cli.utils.multipack_sampler import _MAX_FFD_ITEMS
    too_many = _MAX_FFD_ITEMS + 1
    # Don't actually allocate 1M ints — just confirm the cap exists by
    # patching it lower for the test.
    import soup_cli.utils.multipack_sampler as ms
    original = ms._MAX_FFD_ITEMS
    try:
        ms._MAX_FFD_ITEMS = 5
        with pytest.raises(ValueError, match="too many items"):
            ffd_bin_pack([1, 2, 3, 4, 5, 6], max_len=10)
    finally:
        ms._MAX_FFD_ITEMS = original
    assert too_many > _MAX_FFD_ITEMS  # sanity


def test_ffd_handles_generator_input():
    # Generators are exhausted after one pass — implementation must
    # materialise to avoid a silent empty-bin result.
    bins = ffd_bin_pack((x for x in [4, 3, 2]), max_len=10)
    assert sum(len(b) for b in bins) == 3


# ---- validate_multipack_architecture --------------------------------------


def test_validate_arch_allows_known():
    # No raise for known arch
    validate_multipack_architecture("LlamaForCausalLM")
    validate_multipack_architecture("MistralForCausalLM")
    validate_multipack_architecture("Qwen2ForCausalLM")


def test_validate_arch_rejects_unknown_loudly():
    # Critical: vs Axolotl's silent-miss, we raise
    with pytest.raises(ValueError, match="not in multipack allowlist"):
        validate_multipack_architecture("BloomForCausalLM")


def test_validate_arch_error_lists_remediation():
    with pytest.raises(ValueError, match="multipack: false"):
        validate_multipack_architecture("UnknownForCausalLM")


def test_validate_arch_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_multipack_architecture("")


def test_validate_arch_rejects_non_string():
    with pytest.raises(TypeError, match="must be str"):
        validate_multipack_architecture(123)  # type: ignore[arg-type]


def test_validate_arch_rejects_null_byte():
    with pytest.raises(ValueError):
        validate_multipack_architecture("Llama\x00ForCausalLM")


def test_architectures_allowlist_is_frozen():
    # Module constant must be immutable — prevents runtime tampering.
    assert isinstance(MULTIPACK_ARCHITECTURES, frozenset)
    assert "LlamaForCausalLM" in MULTIPACK_ARCHITECTURES
    # frozenset has no .add
    with pytest.raises(AttributeError):
        MULTIPACK_ARCHITECTURES.add("X")  # type: ignore[attr-defined]


# ---- MultipackBatchSampler ------------------------------------------------


def test_sampler_iter_returns_index_lists():
    lengths = [3, 5, 2, 4, 1, 6]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=0,
    )
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        # batch is a flat list of indices when real_batches=False
        assert isinstance(batch, list)
        for idx in batch:
            assert 0 <= idx < len(lengths)


def test_sampler_full_coverage():
    lengths = [3, 5, 2, 4, 1, 6, 7, 2]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=0,
    )
    seen: list[int] = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))


def test_sampler_respects_batch_max_len():
    lengths = [3, 5, 2, 4, 1, 6]
    max_len = 10
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=max_len, batch_size=1, real_batches=False, seed=0,
    )
    for batch in sampler:
        total = sum(lengths[i] for i in batch)
        assert total <= max_len


def test_sampler_deterministic_with_seed():
    lengths = [3, 5, 2, 4, 1, 6, 7, 2, 8]
    sampler1 = MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=42,
    )
    sampler2 = MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=42,
    )
    assert list(sampler1) == list(sampler2)


def test_sampler_len_matches_iter():
    lengths = [3, 5, 2, 4, 1, 6]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=0,
    )
    assert len(sampler) == len(list(sampler))


def test_sampler_real_batches_groups_into_batch_size():
    # real_batches=True groups packed bins into chunks of batch_size
    lengths = [3] * 12
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=6, batch_size=2, real_batches=True, seed=0,
    )
    for batch in sampler:
        # Each batch is a list of bins; each bin is a list of indices.
        assert isinstance(batch, list)
        assert all(isinstance(bin_, list) for bin_ in batch)
        assert len(batch) <= 2  # batch_size


def test_sampler_drop_last():
    lengths = [3] * 13  # 13 / 2 doesn't divide evenly
    sampler_drop = MultipackBatchSampler(
        lengths, batch_max_len=6, batch_size=2, real_batches=True,
        seed=0, drop_last=True,
    )
    sampler_keep = MultipackBatchSampler(
        lengths, batch_max_len=6, batch_size=2, real_batches=True,
        seed=0, drop_last=False,
    )
    # drop_last=False keeps the trailing partial batch; True drops it.
    assert len(list(sampler_keep)) >= len(list(sampler_drop))


def test_sampler_rejects_empty_lengths():
    with pytest.raises(ValueError, match="lengths"):
        MultipackBatchSampler(
            [], batch_max_len=10, batch_size=1, real_batches=False, seed=0,
        )


def test_sampler_rejects_non_positive_batch_max_len():
    with pytest.raises(ValueError, match="batch_max_len must be positive"):
        MultipackBatchSampler(
            [3, 4], batch_max_len=0, batch_size=1, real_batches=False, seed=0,
        )


def test_sampler_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MultipackBatchSampler(
            [3, 4], batch_max_len=10, batch_size=0, real_batches=True, seed=0,
        )


def test_sampler_rejects_bool_batch_size():
    with pytest.raises(TypeError, match="bool"):
        MultipackBatchSampler(
            [3, 4], batch_max_len=10, batch_size=True, real_batches=True, seed=0,
        )


def test_sampler_rejects_bool_batch_max_len():
    with pytest.raises(TypeError, match="bool"):
        MultipackBatchSampler(
            [3, 4], batch_max_len=True, batch_size=1, real_batches=False, seed=0,
        )


def test_sampler_rejects_item_larger_than_max():
    with pytest.raises(ValueError, match="exceeds"):
        MultipackBatchSampler(
            [3, 100], batch_max_len=10, batch_size=1,
            real_batches=False, seed=0,
        )


def test_sampler_different_seeds_yield_different_orderings():
    lengths = [3, 5, 2, 4, 1, 6, 7, 2, 8, 4, 5]
    s1 = list(MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=1,
    ))
    s2 = list(MultipackBatchSampler(
        lengths, batch_max_len=10, batch_size=1, real_batches=False, seed=999,
    ))
    # Not strictly guaranteed but vanishingly improbable for 11 items.
    assert s1 != s2
