"""Calculator tool-use rollout env — v0.71.30.

Generates a deterministic set of single-operation arithmetic problems as GRPO
prompt+answer rows. Score with ``reward_fn='math'`` (or ``'accuracy'``).

Usage: ``training.rollout_backend='openenv'`` +
``training.rollout_func='soup_cli.envs.calculator:rollout'``.
"""

from __future__ import annotations

import random
from typing import Any

from soup_cli.envs._common import seeded_rows

_SEED = 20730
_OPS = ("+", "-", "*")


def _make_row(rng: random.Random) -> dict[str, str]:
    op = _OPS[rng.randrange(len(_OPS))]
    if op == "*":
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
    else:
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
    result = {"+": a + b, "-": a - b, "*": a * b}[op]
    prompt = f"What is {a} {op} {b}? Reply with just the number."
    return {"prompt": prompt, "answer": str(result)}


def rollout(prompts: Any = None) -> list[dict[str, str]]:
    """Return a deterministic list of ``{"prompt", "answer"}`` arithmetic rows.

    ``prompts`` (the seed prompts from the GRPO dataset) is accepted for the
    openenv contract but does not change the generated curriculum — the env is
    a self-contained deterministic seeder.
    """
    return seeded_rows(_SEED, _make_row)
