"""Retrieval-QA rollout env — v0.71.30.

Generates deterministic short-document + question rows whose answer is a span
present in the document, as GRPO prompt+answer rows. Score with
``reward_fn='accuracy'`` (the answer span appears in the model's completion).

Usage: ``training.rollout_backend='openenv'`` +
``training.rollout_func='soup_cli.envs.retrieval_qa:rollout'``.
"""

from __future__ import annotations

import random
from typing import Any

_SEED = 20732
_DEFAULT_ROWS = 64

# (entity, attribute, value) fact templates — the answer is always ``value``,
# which is embedded verbatim in the document so it is a retrievable span.
_FACTS = (
    ("The Zephyr rover", "landed in the year", "2031"),
    ("The city of Aldermere", "has a population of", "48000"),
    ("The Blue Comet", "orbits every", "76 years"),
    ("The Marlow Bridge", "spans", "1200 metres"),
    ("The Quill library", "holds", "90000 books"),
    ("Mount Calder", "rises to", "3400 metres"),
    ("The Aster festival", "lasts", "9 days"),
    ("The Nadir mine", "reaches a depth of", "800 metres"),
)


def rollout(prompts: Any = None) -> list[dict]:
    """Return a deterministic list of ``{"prompt", "answer"}`` retrieval rows.

    Each row embeds a short document containing several facts and asks about
    one of them; the answer is a span present in the document. ``prompts`` is
    accepted for the openenv contract but does not change the curriculum.
    """
    rng = random.Random(_SEED)
    rows: list[dict] = []
    for _ in range(_DEFAULT_ROWS):
        # Pick 3 distinct facts as the document; ask about one of them.
        facts = rng.sample(_FACTS, 3)
        target = facts[rng.randrange(len(facts))]
        entity, attribute, value = target
        doc = " ".join(f"{e} {a} {v}." for (e, a, v) in facts)
        prompt = (
            f"Document: {doc}\n"
            f"Question: What does {entity} {attribute.rstrip()}? "
            "Reply with just the value."
        )
        rows.append({"prompt": prompt, "answer": value})
    return rows
