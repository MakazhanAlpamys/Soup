"""Exact-output and performance guards for the kept-buffer optimization (#3250).

Despite the issue's original filename, this change uses matrix-vector products,
not blocked matrix-matrix products: raw audit scores must remain unchanged.
"""

import statistics
import time

import numpy as np
import pytest

from soup_cli.utils.semdedup import DedupReport, greedy_semdedup


def reference(vectors: np.ndarray, threshold: float) -> DedupReport:
    """The loop on main before this change (76f9b207), with #3254's comparison."""
    arr = np.asarray(vectors, dtype=np.float32)
    kept, dropped, pairs = [], [], []
    for idx in range(len(arr)):
        if kept:
            sims = arr[kept] @ arr[idx]
            best = int(np.argmax(sims))
            score = float(sims[best])
            matches = score >= threshold if threshold == 0 else (
                score > 0 and score >= threshold - 1e-6
            )
            if matches:
                dropped.append(idx)
                pairs.append((idx, kept[best], score))
                continue
        kept.append(idx)
    return DedupReport(tuple(kept), tuple(dropped), tuple(pairs), threshold)


def assert_identical(vectors: np.ndarray, threshold: float) -> None:
    before = vectors.copy()
    expected = reference(vectors, threshold)
    actual = greedy_semdedup(vectors, threshold=threshold)
    assert actual == expected
    # Also pin signed zero and the exact binary representation of audit scores.
    assert np.array([p[2] for p in actual.pairs], dtype=np.float64).tobytes() == (
        np.array([p[2] for p in expected.pairs], dtype=np.float64).tobytes()
    )
    np.testing.assert_array_equal(vectors, before)


@pytest.mark.parametrize("threshold", [0.0, 1e-7, 1e-6, 0.3, 0.9, 0.9999, 1.0])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed", "float64"])
def test_randomized_exact_reports(threshold: float, layout: str) -> None:
    rng = np.random.default_rng(3250)
    for dimension in (1, 3, 17, 64, 129, 384):
        vectors = rng.standard_normal((81, dimension)).astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        vectors[7:12] = vectors[0]
        vectors[40:50] = vectors[20:30]
        vectors[60] = 0
        if layout == "fortran":
            vectors = np.asfortranarray(vectors)
        elif layout == "strided":
            storage = np.zeros((len(vectors), dimension * 2), dtype=np.float32)
            storage[:, ::2] = vectors
            vectors = storage[:, ::2]
        elif layout == "reversed":
            vectors = vectors[:, ::-1]
        elif layout == "float64":
            vectors = vectors.astype(np.float64)
        vectors.flags.writeable = False
        assert_identical(vectors, threshold)


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (5, 0), (8, 1), (8, 3)])
@pytest.mark.parametrize("threshold", [0.0, 1e-7, 1e-6, 1.0])
def test_empty_single_and_zero_rows(shape: tuple[int, int], threshold: float) -> None:
    assert_identical(np.zeros(shape, dtype=np.float32), threshold)


def test_tie_uses_earliest_kept_original_index() -> None:
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0.6, 0.6]], dtype=np.float32)
    report = greedy_semdedup(vectors, threshold=0.6)
    assert report.kept == (0, 2)
    assert [(drop, keep) for drop, keep, _ in report.pairs] == [(1, 0), (3, 2), (4, 0)]
    assert_identical(vectors, 0.6)


def test_dropped_row_never_becomes_a_match_candidate() -> None:
    # B matches A, C matches B but not A: C must survive.
    angles = np.deg2rad([0, 30, 60])
    vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    report = greedy_semdedup(vectors, threshold=0.8)
    assert report.kept == (0, 2)
    assert report.dropped == (1,)
    assert_identical(vectors, 0.8)


@pytest.mark.parametrize("gap", [-2e-6, -0.5e-6, 0, 0.5e-6, 1e-6, 2e-6])
def test_exact_report_at_tolerance_boundary(gap: float) -> None:
    cosine = 0.8 - gap
    vectors = np.array([[1, 0], [cosine, np.sqrt(1 - cosine**2)]], dtype=np.float32)
    assert_identical(vectors, 0.8)


def test_buffer_avoids_repeated_copy_cost() -> None:
    """Same-process medians; restores-to-main mutation must fail this guard."""
    rng = np.random.default_rng(3250)
    vectors = rng.standard_normal((5000, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = reference(vectors, 0.9)
    assert len(expected.kept) == len(vectors)
    assert greedy_semdedup(vectors, threshold=0.9) == expected
    durations = {"reference": [], "candidate": []}
    calls = {"reference": reference, "candidate": greedy_semdedup}
    for repeat in range(5):
        order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
        for name in order:
            start = time.perf_counter()
            report = calls[name](vectors, threshold=0.9)
            durations[name].append(time.perf_counter() - start)
            assert report == expected
    ratio = statistics.median(durations["candidate"]) / statistics.median(durations["reference"])
    assert ratio < 0.5, f"candidate/reference={ratio:.3f}; seconds={durations}"
