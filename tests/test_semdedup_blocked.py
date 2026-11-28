"""Exact-output and performance guards for the kept-buffer optimization (#3250).

Despite the issue's original filename, this change uses matrix-vector products,
not blocked matrix-matrix products: raw audit scores must remain unchanged.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import soup_cli
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


def timing_samples() -> dict[str, list[float]]:
    """CPU seconds for both loops, alternating order, after one warmup call each."""
    rng = np.random.default_rng(3250)
    vectors = rng.standard_normal((5000, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = reference(vectors, 0.9)
    assert len(expected.kept) == len(vectors)
    assert greedy_semdedup(vectors, threshold=0.9) == expected
    seconds: dict[str, list[float]] = {"reference": [], "candidate": []}
    calls = {"reference": reference, "candidate": greedy_semdedup}
    for repeat in range(5):
        order = ("reference", "candidate") if repeat % 2 == 0 else ("candidate", "reference")
        for name in order:
            start = time.thread_time()
            report = calls[name](vectors, threshold=0.9)
            seconds[name].append(time.thread_time() - start)
            assert report == expected
    return seconds


def test_buffer_avoids_repeated_copy_cost() -> None:
    """Restoring the loop on main must fail this guard.

    Both loops run in one child process with a single BLAS thread, timed in CPU
    time of that thread. With the default thread count under parallel tests, BLAS
    workers wait on each other for CPU: wall time of the buffered loop varied by
    several times and the ratio crossed the bound. One thread does all the work
    on the timed thread, and time spent preempted is not counted.
    """
    child_env = dict(os.environ)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        child_env[name] = "1"  # read when NumPy loads, hence the child process
    # Import the same soup_cli, and this module, in the child.
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(soup_cli.__file__).parents[1]), str(Path(__file__).parent)]
    )
    code = f"import json, {__name__.rpartition('.')[2]} as t; print(json.dumps(t.timing_samples()))"
    run = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env, capture_output=True, text=True, timeout=480,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    seconds = json.loads(run.stdout.strip().splitlines()[-1])
    ratio = statistics.median(seconds["candidate"]) / statistics.median(seconds["reference"])
    assert ratio < 0.5, f"candidate/reference={ratio:.3f}; cpu_seconds={seconds}"
