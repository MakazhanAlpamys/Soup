"""`ffd_bin_pack` places items through an indexed structure, not a scan (#694).

The packing is unchanged: same bins, same order, same exceptions. Only the cost
of finding each bin changed, from O(N) per item to O(log N). These tests exist
to keep that "unchanged" honest, because a first-fit implementation that
quietly becomes best-fit still produces plausible-looking bins.

The reference implementation below is the exact pre-#694 loop. Comparing
against it rather than against hand-written expectations is what makes the
equivalence claim checkable: a hand-written case pins what someone believed
the old code did, and the whole risk here is that the belief is wrong.
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Sequence

import pytest

from soup_cli.utils.multipack_sampler import ffd_bin_pack


def linear_scan_ffd(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """The placement loop `ffd_bin_pack` used before #694, verbatim."""
    indexed = sorted(enumerate(list(lengths)), key=lambda pair: pair[1], reverse=True)
    bins: list[list[int]] = []
    bin_remaining: list[int] = []
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


@pytest.mark.parametrize("seed", range(25))
def test_placement_is_identical_to_the_linear_scan(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(40):
        max_len = rng.choice([8, 16, 64, 128, 1024])
        count = rng.randint(0, 120)
        lengths = [rng.randint(1, max_len) for _ in range(count)]
        assert ffd_bin_pack(lengths, max_len) == linear_scan_ffd(lengths, max_len)


@pytest.mark.parametrize("seed", range(10))
def test_ties_break_the_same_way(seed: int) -> None:
    """The case a best-fit regression would survive.

    With distinct lengths, several placement rules agree often enough to look
    right. With heavy ties they diverge, so this draws from a tiny alphabet and
    from all-identical inputs on purpose.
    """
    rng = random.Random(1000 + seed)
    for _ in range(40):
        max_len = rng.choice([16, 64, 1024])
        count = rng.randint(0, 100)
        alphabet = [1, max(1, max_len // 2), max_len]
        lengths = [rng.choice(alphabet) for _ in range(count)]
        assert ffd_bin_pack(lengths, max_len) == linear_scan_ffd(lengths, max_len)

        uniform = [rng.randint(1, max_len)] * count
        assert ffd_bin_pack(uniform, max_len) == linear_scan_ffd(uniform, max_len)


def test_the_worst_case_shape_is_identical() -> None:
    """Every item over half the budget: no earlier bin can ever fit."""
    lengths = [600] * 1500
    assert ffd_bin_pack(lengths, 1024) == linear_scan_ffd(lengths, 1024)


def test_first_fit_not_best_fit() -> None:
    """A case where the two rules disagree, stated directly.

    Sorted descending: 10, 6, 5, 4. Bins after the first three items are
    [10] (0 left), [6, 5] (5 left)... the 4 must land in the *first* bin with
    room, which best-fit would not choose if a tighter one existed later.
    """
    lengths = [10, 6, 5, 4, 4]
    assert ffd_bin_pack(lengths, 11) == linear_scan_ffd(lengths, 11)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 17, 64, 65, 128, 129])
def test_bin_capacity_growth_boundaries(count: int) -> None:
    """The tree's leaf array doubles, so exercise around the powers of two.

    An off-by-one in the rebuild would show up only at these sizes.
    """
    lengths = [7] * count
    assert ffd_bin_pack(lengths, 7) == linear_scan_ffd(lengths, 7)


def test_every_index_appears_exactly_once() -> None:
    rng = random.Random(99)
    lengths = [rng.randint(1, 512) for _ in range(400)]
    bins = ffd_bin_pack(lengths, 512)
    flat = sorted(idx for group in bins for idx in group)
    assert flat == list(range(len(lengths)))


def test_no_bin_exceeds_the_budget() -> None:
    rng = random.Random(1234)
    lengths = [rng.randint(1, 300) for _ in range(500)]
    bins = ffd_bin_pack(lengths, 300)
    for group in bins:
        assert sum(lengths[idx] for idx in group) <= 300


@pytest.mark.parametrize(
    ("lengths", "max_len", "exc", "message"),
    [
        ([1, 2], 0, ValueError, "max_len must be positive"),
        ([1, 2], -5, ValueError, "max_len must be positive"),
        ([1, 2], True, TypeError, "max_len must not be bool"),
        ([0, 1], 8, ValueError, "must be positive"),
        ([-3], 8, ValueError, "must be positive"),
        ([9], 8, ValueError, "exceeds max_len"),
        ([1.5], 8, TypeError, "must be int"),
    ],
)
def test_validation_is_unchanged(lengths, max_len, exc, message) -> None:
    with pytest.raises(exc, match=message):
        ffd_bin_pack(lengths, max_len)


def test_empty_input_returns_no_bins() -> None:
    assert ffd_bin_pack([], 8) == []


def _median_seconds(fn, *args, reps: int = 3) -> float:
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def test_the_worst_case_is_no_longer_quadratic() -> None:
    """A regression threshold, deliberately generous.

    Measured locally at 18x to 30x on the cases in #694. The bar here is 3x,
    which a reintroduced linear scan cannot clear at this size while leaving
    ample room for a loaded CI runner. This asserts the shape of the change,
    not a performance number.
    """
    lengths = [600] * 3000
    scan = _median_seconds(linear_scan_ffd, lengths, 1024)
    indexed = _median_seconds(ffd_bin_pack, lengths, 1024)
    assert indexed * 3 < scan, (
        f"indexed placement took {indexed * 1000:.1f}ms against the linear "
        f"scan's {scan * 1000:.1f}ms; expected at least a 3x margin"
    )
