"""v0.37.0 Part E — packing correctness invariants across the multipack stack.

Mirrors Axolotl's ``test_packed_batch_sampler.py:111-117`` quality bar plus
extra cross-module checks that exercise sampler + 4D mask + Jinja analyzer
together. Each test asserts an invariant that, if broken, would silently
corrupt training (hardest class of bug to surface in production).

Invariants:

1. **No duplicates across packs** — every sample index appears at most once
   per epoch.
2. **Full coverage** — every sample index appears at least once per epoch.
3. **Pack-len bound** — no bin's total length exceeds
   ``batch_size × max_seq_length`` (flat mode) or ``max_seq_length``
   (real-batches mode).
4. **Mask-segment coherence** — the 4D mask built from a packed bin has
   no allowed cross-segment attention pair.
5. **Determinism on identical seed** — sampler order is reproducible.
6. **Stress** — invariants 1–3 hold on a 5,000-sample random workload.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from soup_cli.utils.jinja_analyzer import extract_message_fields
from soup_cli.utils.multipack_sampler import (
    MultipackBatchSampler,
    ffd_bin_pack,
)
from soup_cli.utils.neat_packing import (
    build_4d_attention_mask,
    tag_sub_sequences,
)


def _flatten_indices(sampler: MultipackBatchSampler) -> list[int]:
    flat: list[int] = []
    for batch in sampler:
        if batch and isinstance(batch[0], list):
            for bin_ in batch:
                flat.extend(bin_)
        else:
            flat.extend(batch)
    return flat


# ---- Invariants 1-3: sampler-level ---------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 999])
def test_no_duplicates_across_packs(seed):
    rng = random.Random(seed)
    lengths = [rng.randint(1, 50) for _ in range(200)]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=128, batch_size=4,
        real_batches=False, seed=seed,
    )
    seen: set[int] = set()
    for batch in sampler:
        for idx in batch:
            assert idx not in seen, f"duplicate index {idx} in sampler output"
            seen.add(idx)


@pytest.mark.parametrize("seed", [0, 1, 42, 999])
def test_full_coverage(seed):
    rng = random.Random(seed)
    n = 200
    lengths = [rng.randint(1, 50) for _ in range(n)]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=128, batch_size=4,
        real_batches=False, seed=seed,
    )
    flat = _flatten_indices(sampler)
    assert sorted(flat) == list(range(n))


@pytest.mark.parametrize(
    "real_batches,batch_size,batch_max_len",
    [(False, 1, 64), (True, 4, 64), (True, 8, 32)],
)
def test_pack_len_bound(real_batches, batch_size, batch_max_len):
    rng = random.Random(2026)
    lengths = [rng.randint(1, batch_max_len) for _ in range(150)]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=batch_max_len, batch_size=batch_size,
        real_batches=real_batches, seed=0,
    )
    if real_batches:
        for batch in sampler:
            for bin_ in batch:
                assert sum(lengths[i] for i in bin_) <= batch_max_len
    else:
        for bin_ in sampler:
            assert sum(lengths[i] for i in bin_) <= batch_max_len


# ---- Invariant 4: sampler + 4D mask coherence ----------------------------


def test_mask_built_from_pack_blocks_cross_doc():
    # Build a real packed bin via FFD, derive segment IDs, build 4D mask,
    # then verify no allowed cross-segment pair exists in the mask.
    lengths = [3, 5, 2, 4, 1, 6]
    bins = ffd_bin_pack(lengths, max_len=10)
    # Take the first multi-document bin — must have >=2 sub-seqs to test.
    target_bin = next((b for b in bins if len(b) >= 2), None)
    if target_bin is None:
        pytest.fail(
            "FFD must pack at least 2 docs into one bin given the test "
            f"input lengths={lengths}, max_len=10. Got bins={bins}. "
            "If this fires after a packer change, the cross-doc invariant "
            "is no longer being exercised."
        )

    boundaries = [0]
    cum = 0
    for idx in target_bin:
        cum += lengths[idx]
        boundaries.append(cum)
    seg_ids = tag_sub_sequences(boundaries)
    seq_arr = np.array([seg_ids], dtype=np.int32)
    mask = build_4d_attention_mask(seq_arr, dtype=np.float32)
    plane = mask[0, 0]

    # For every position i, every j with seg_ids[i] != seg_ids[j] must be
    # blocked (large negative).
    for i, seg_i in enumerate(seg_ids):
        for j, seg_j in enumerate(seg_ids):
            if seg_i != seg_j:
                assert plane[i, j] < -1e9, (
                    f"cross-segment leak at ({i},{j}) "
                    f"seg_i={seg_i} seg_j={seg_j}"
                )


# ---- Invariant 5: determinism --------------------------------------------


def test_determinism_across_processes_simulated():
    # Simulate two ranks building the sampler with the same seed → identical
    # batch order. Critical for DDP correctness.
    lengths = [random.Random(7).randint(1, 40) for _ in range(100)]
    s1 = list(MultipackBatchSampler(
        lengths, batch_max_len=64, batch_size=2,
        real_batches=True, seed=11,
    ))
    s2 = list(MultipackBatchSampler(
        lengths, batch_max_len=64, batch_size=2,
        real_batches=True, seed=11,
    ))
    assert s1 == s2


# ---- Invariant 6: stress test on 5k samples ------------------------------


def test_stress_5k_samples():
    rng = random.Random(31337)
    n = 5_000
    lengths = [rng.randint(1, 200) for _ in range(n)]
    sampler = MultipackBatchSampler(
        lengths, batch_max_len=512, batch_size=8,
        real_batches=False, seed=0,
    )
    flat = _flatten_indices(sampler)
    # No duplicates, full coverage — both invariants in one pass for speed.
    assert len(flat) == n
    assert sorted(flat) == list(range(n))


# ---- Cross-module: Jinja analyzer + sampler ------------------------------


def test_jinja_analyzer_finds_train_field_for_per_msg_masking():
    # The Axolotl chat-template flavour adds {% if m.train %} so per-message
    # training masking works. Confirms the analyzer picks it up — needed by
    # the v0.37.0 / v0.36.0 train_on_messages_with_train_field gate.
    template = (
        "{% for m in messages %}"
        "{% if m.train %}<train>{{ m.content }}</train>"
        "{% else %}{{ m.content }}"
        "{% endif %}"
        "{% endfor %}"
    )
    fields = extract_message_fields(template)
    assert "train" in fields
    assert "content" in fields
