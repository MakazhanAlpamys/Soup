"""Cost + budget guardrails for `soup loop` (v0.58.0 Part C).

Two orthogonal rate limits:

* ``monthly_budget_usd`` — composes with v0.34.0 per-run cost so the
  watch daemon pauses (graceful save, no kill) when projected spend
  would exceed the budget.
* ``max_runs_per_day`` — defends against runaway proxy loops by capping
  iteration starts per UTC day.

The math here is pure-Python so the daemon can call ``check()`` without
opening a SQLite handle. Persisted counters live in the ``LoopState``
shared store.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class BudgetDecision:
    """Decision returned by ``check_budget``.

    ``proceed`` is the only field a caller MUST inspect; the others are
    advisory for the user-facing dashboard.
    """

    proceed: bool
    reason: str
    projected_total_usd: float
    runs_today: int


def _utc_date_str(ts: Optional[datetime] = None) -> str:
    """Return today's UTC date as ``YYYY-MM-DD`` (testable via ``ts=...``)."""
    now = ts if ts is not None else datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def reset_daily_counter_if_new_day(
    runs_today: int,
    last_run_date: Optional[str],
    *,
    now: Optional[datetime] = None,
) -> "tuple[int, str]":
    """Reset ``runs_today`` to 0 when the UTC day rolls over.

    Returns ``(runs_today, last_run_date)`` — the caller updates the
    ``LoopState`` with the returned values before checking the cap.
    """
    if isinstance(runs_today, bool) or not isinstance(runs_today, int) or runs_today < 0:
        raise ValueError("runs_today must be a non-negative int")
    if last_run_date is not None and not isinstance(last_run_date, str):
        raise ValueError("last_run_date must be a str or None")
    today = _utc_date_str(now)
    if last_run_date != today:
        return 0, today
    return runs_today, today


def check_budget(
    *,
    estimated_run_usd: float,
    spent_so_far_usd: float,
    monthly_budget_usd: Optional[float],
    runs_today: int,
    max_runs_per_day: Optional[int],
) -> BudgetDecision:
    """Decide whether to proceed with another iteration.

    The decision composes three checks in this order:

    1. Run-cap (daily) — fast rejection when ``max_runs_per_day`` is set
       and ``runs_today >= max``.
    2. Estimate sanity — reject non-finite / negative cost estimates so
       a broken probe can't smuggle a negative refund.
    3. Budget — reject when projected spend would exceed the cap.
    """
    if (
        isinstance(estimated_run_usd, bool)
        or not isinstance(estimated_run_usd, (int, float))
        or not math.isfinite(estimated_run_usd)
        or estimated_run_usd < 0
    ):
        raise ValueError("estimated_run_usd must be a non-negative finite number")
    if (
        isinstance(spent_so_far_usd, bool)
        or not isinstance(spent_so_far_usd, (int, float))
        or not math.isfinite(spent_so_far_usd)
        or spent_so_far_usd < 0
    ):
        raise ValueError("spent_so_far_usd must be a non-negative finite number")
    if isinstance(runs_today, bool) or not isinstance(runs_today, int) or runs_today < 0:
        raise ValueError("runs_today must be a non-negative int")
    if max_runs_per_day is not None:
        if (
            isinstance(max_runs_per_day, bool)
            or not isinstance(max_runs_per_day, int)
            or max_runs_per_day < 1
        ):
            raise ValueError("max_runs_per_day must be a positive int or None")
        if runs_today >= max_runs_per_day:
            return BudgetDecision(
                proceed=False,
                reason=(
                    f"daily cap reached: {runs_today}/{max_runs_per_day} runs"
                ),
                projected_total_usd=float(spent_so_far_usd),
                runs_today=runs_today,
            )
    projected = float(spent_so_far_usd) + float(estimated_run_usd)
    if monthly_budget_usd is not None:
        if (
            isinstance(monthly_budget_usd, bool)
            or not isinstance(monthly_budget_usd, (int, float))
            or not math.isfinite(monthly_budget_usd)
            or monthly_budget_usd < 0
        ):
            raise ValueError("monthly_budget_usd must be >= 0 or None")
        if projected > float(monthly_budget_usd):
            return BudgetDecision(
                proceed=False,
                reason=(
                    f"would exceed monthly budget: ${projected:.2f} > "
                    f"${float(monthly_budget_usd):.2f}"
                ),
                projected_total_usd=projected,
                runs_today=runs_today,
            )
    return BudgetDecision(
        proceed=True,
        reason="within budget",
        projected_total_usd=projected,
        runs_today=runs_today,
    )


def parse_budget_string(raw: str) -> float:
    """Parse ``"50usd"`` / ``"100 USD"`` / ``"25"`` into a USD float.

    Trailing ``"usd"`` (case-insensitive) is optional. Bounds: ``[0,
    1_000_000]`` so a fat-finger ``"1000000000"`` cannot cause integer
    overflow in downstream arithmetic.
    """
    if not isinstance(raw, str):
        raise TypeError("budget must be a string")
    raw = raw.strip().lower()
    if not raw:
        raise ValueError("budget must not be empty")
    if "\x00" in raw:
        raise ValueError("budget must not contain NUL")
    if raw.endswith("usd"):
        raw = raw[:-3].strip()
    if not raw:
        # "usd" / "  usd  " — friendly explicit message (code-review M6).
        raise ValueError("budget must include a numeric value (e.g. '50usd')")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid budget value: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError("budget must be finite")
    if not (0.0 <= value <= 1_000_000.0):
        raise ValueError("budget must be in [0, 1_000_000] USD")
    return value
