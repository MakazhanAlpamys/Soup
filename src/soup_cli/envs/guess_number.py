"""Number-deduction rollout env — v0.71.30.

Generates deterministic single-shot number-deduction puzzles (constraints that
pin a unique integer) as GRPO prompt+answer rows. Score with
``reward_fn: verifiable`` + ``verifiable_domain: math`` (or ``reward_fn: accuracy``).

Honesty: this is single-shot *deduction*, not interactive guess-and-feedback —
the live openenv contract does not pass the model, so a true guessing loop is
out of scope (a follow-up once openenv gains a model-in-the-loop hook).

Usage: ``training.rollout_backend='openenv'`` +
``training.rollout_func='soup_cli.envs.guess_number:rollout'``.
"""

from __future__ import annotations

import random
from typing import Any

from soup_cli.envs._common import seeded_rows

_SEED = 20731


def _make_row(rng: random.Random) -> dict[str, str]:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    answer = a * b
    prompt = (
        f"I'm thinking of a number between 1 and 100. "
        f"It equals {a} times {b}. What is the number? "
        "Reply with just the number."
    )
    return {"prompt": prompt, "answer": str(answer)}


def rollout(prompts: Any = None) -> list[dict[str, str]]:
    """Return a deterministic list of ``{"prompt", "answer"}`` deduction rows.

    Each puzzle states two factors whose product is the answer, plus a range,
    so the answer is uniquely deducible. ``prompts`` is accepted for the
    openenv contract but does not change the curriculum.
    """
    return seeded_rows(_SEED, _make_row)
