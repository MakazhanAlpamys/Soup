"""THE growth check for the "stays linear" tests. One definition.

A wall-clock test of "linear, not quadratic" compares the time at ``scale * n`` with the time
at ``n``: a linear scan grows about ``scale`` times, a quadratic one ``scale ** 2`` times. The
tests that did this at a fixed ``n`` timed calls of 30 microseconds to a few milliseconds (most
of them under a millisecond), so one scheduler stall on a loaded runner was larger than the
whole signal: ``test_contains_grows_linearly[one-letter-run]`` read 69 microseconds against
1.77 ms (25.6x, limit 24) and the email growth test "8N took 0.024s vs N 0.001s" on macOS,
2026-10-20. A fixed bound flaked the same way before (#1932).

Three things keep a loaded runner from deciding the test:

* The reference run is calibrated to take at least ``MIN_BASE_SECONDS``: ``n`` doubles from a
  small start until it does (capped at ``MAX_GROWTH`` times the start), so a slow machine needs
  fewer doublings and a fast one more, and the signal is far above the timer and a stall.
  Starting small also keeps a quadratic scan cheap to catch: its reference reaches the floor
  at a small ``n``, so the larger run takes about ``scale ** 2`` times the floor, whatever the
  start was.
* Both sizes are the fastest of several runs, taken in alternation, so the same load phases
  hit both. The check ends at the first round that shows the larger input is fast enough; only
  a genuinely super-linear scan runs more than one, and it gives up after ``MIN_ROUNDS`` once
  ``BUDGET_SECONDS`` are spent.
* The collector is paused while timing, so one full collection cannot land on one size only.

A call that truncates its input (``detect_pii`` reads the first 50,000 characters) would stop
growing past that length and pass whatever it did before it: give ``cap`` the length it reads.

``TIMER_SLACK_SECONDS`` is added to the bound for the case where the call is so fast that the
cap is reached before the reference is long. A quadratic scan fails by a wide margin at either
size: 64x against a 32x limit.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

SCALE = 8
LIMIT = 4.0 * SCALE  # 4x slack over linear (8x), as in #1932; a quadratic scan measures 64x
MIN_BASE_SECONDS = 0.005
MAX_GROWTH = 4096  # how far calibration may scale the starting size
ROUNDS = 5
MIN_ROUNDS = 2  # a failing scan still gets this many rounds, however slow
BUDGET_SECONDS = 2.0  # ...and no more once the rounds have spent this long
TIMER_SLACK_SECONDS = 0.002


def _seconds(call: Callable[[Any], object], payload: Any) -> float:
    start = time.perf_counter()
    call(payload)
    return time.perf_counter() - start


def _calibrate(
    call: Callable[[Any], object],
    build: Callable[[int], Any],
    size: int,
    min_base: float,
    max_size: int,
) -> tuple[int, Any, float]:
    """Double ``size`` until the best of three runs reaches ``min_base`` (or ``max_size``)."""
    while True:
        payload = build(size)
        best = min(_seconds(call, payload) for _ in range(3))
        if best >= min_base or size * 2 > max_size:
            return size, payload, best
        size *= 2


def assert_grows_linearly(
    call: Callable[[Any], object],
    build: Callable[[int], Any],
    size: int,
    *,
    label: str = "",
    scale: int = SCALE,
    limit: float = LIMIT,
    min_base: float = MIN_BASE_SECONDS,
    cap: int | None = None,
) -> None:
    """Fail unless ``call(build(scale * n))`` takes under ``limit`` times ``call(build(n))``.

    ``n`` starts at ``size`` and is calibrated up so the reference run lasts ``min_base``.
    ``build(k)`` returns an input that grows in proportion to ``k``; ``call`` takes it whole.
    ``cap`` is the largest ``k`` the call still reads in full: past it the call truncates (or
    refuses) its input, so a larger run would measure nothing. The larger input never exceeds
    it, and the reference then stays shorter than ``min_base`` if it has to.
    """
    max_size = size * MAX_GROWTH
    if cap is not None:
        max_size = min(max_size, cap // scale)
        if size > max_size:
            raise ValueError(f"size {size} x {scale} is past the cap of {cap}")
    reenable = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        used, small, t_small = _calibrate(call, build, size, min_base, max_size)
        large = build(scale * used)
        t_large = float("inf")
        spent = 0.0
        for round_number in range(1, ROUNDS + 1):
            t_small = min(t_small, _seconds(call, small))
            latest = _seconds(call, large)
            t_large = min(t_large, latest)
            if t_large <= limit * t_small + TIMER_SLACK_SECONDS:
                return
            spent += latest
            if round_number >= MIN_ROUNDS and spent > BUDGET_SECONDS:
                break
    finally:
        if reenable:
            gc.enable()
    prefix = f"{label}: " if label else ""
    raise AssertionError(
        f"{prefix}{scale}x the input took {t_large / t_small:.0f}x as long "
        f"(limit {limit:.0f}x; n={used}: {t_small * 1e3:.2f} ms -> {t_large * 1e3:.2f} ms, "
        f"best of {round_number} rounds)"
    )
