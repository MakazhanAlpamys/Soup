"""Shared scaffold for the bundled rollout envs — v0.71.30.

Each env module supplies a per-row generator over a seeded ``random.Random``;
this helper owns the deterministic seeding + fixed row count so the three env
modules stay tiny and cannot drift on the boilerplate.
"""

from __future__ import annotations

import random
from collections.abc import Callable

# Fixed curriculum size per env (deterministic; overridable by callers).
DEFAULT_ROWS = 64


def seeded_rows(
    seed: int,
    make_row: Callable[[random.Random], dict[str, str]],
    count: int = DEFAULT_ROWS,
) -> list[dict[str, str]]:
    """Build ``count`` deterministic ``{"prompt","answer"}`` rows.

    A fresh ``random.Random(seed)`` is created on every call, so the output is
    identical across calls (determinism the tests assert). ``make_row`` receives
    that RNG and returns one ``{"prompt","answer"}`` row.
    """
    rng = random.Random(seed)
    return [make_row(rng) for _ in range(count)]
