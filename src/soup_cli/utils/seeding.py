"""Resolve ``training.seed`` into the two forms a task trainer needs.

#341 added the ``training.seed`` knob and wired it into ``trainer/sft.py``.
Every other wrapper builds its own ``TrainingArguments`` subclass (``GRPOConfig``,
``DPOConfig``, ``RewardConfig``, ...) and none of them read the field, so a
``task: grpo`` run that set ``training.seed: 7`` trained at
``TrainingArguments``' default of 42 with no error and no warning (#353). Two
"different seeds" then produced identical initialisation and identical data
order, which is how STEP 25 of the H100 record ended up measuring GPU
nondeterminism against itself.

There are two separate jobs here, and closing #353 needs both:

``training_seed_kwargs``
    The ``seed`` / ``data_seed`` pair for a ``TrainingArguments`` subclass.
    Load-bearing rather than belt-and-braces: ``Trainer.__init__`` runs
    ``set_seed(args.seed)`` itself, so a seed applied before the trainer object
    exists is overwritten by ``args.seed`` (42, when nothing threads the config).

``apply_training_seed``
    The global seed for everything a wrapper draws BEFORE its trainer exists.
    That is not only LoRA's ``lora_A``: ``classifier`` / ``reward_model`` /
    ``prm`` load through ``AutoModelForSequenceClassification`` with a freshly
    initialised head, and ``unlearn`` never builds a ``Trainer`` at all. Call it
    at the top of ``setup()``, before the model is loaded.

Wrappers call both, rather than the CLI calling ``set_seed`` once on their
behalf. ``commands/train.py`` is not the only caller of ``setup()``:
``commands/sweep.py`` is another, and a sweep is precisely a reproducibility
study. Seeding at one entry point would leave the other silently unseeded, which
is the shape of the defect this module exists to close.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# What an UNSET ``training.seed`` resolves to. This is HF's own
# ``TrainingArguments.seed`` default, restated here so the "unset behaves exactly
# as before" guarantee is a written constant rather than an accident of whichever
# transformers version is installed. It carried this name in ``trainer/sft.py``
# from #341 until #353 moved it here, where every task trainer can reach it.
DEFAULT_TRAINING_SEED = 42


def resolve_training_seed(tcfg: Any) -> int:
    """The seed a run actually trains at.

    ``None`` means unset, and unset has to reproduce the pre-#341 numbers
    exactly, hence 42 rather than a new default of our own.
    """
    seed = getattr(tcfg, "seed", None)
    return DEFAULT_TRAINING_SEED if seed is None else int(seed)


def training_seed_kwargs(tcfg: Any) -> dict[str, Any]:
    """``seed`` / ``data_seed`` kwargs for any ``TrainingArguments`` subclass.

    ``data_seed`` stays ``None`` when unset: HF reads that as "follow ``seed``",
    and pinning it to the resolved seed instead would change the data order of
    every run that sets neither field.
    """
    return {
        "seed": resolve_training_seed(tcfg),
        "data_seed": getattr(tcfg, "data_seed", None),
    }


def apply_training_seed(tcfg: Any) -> int:
    """Seed python / numpy / torch before the model and adapter are built.

    Returns the seed applied so a caller can report it.
    """
    from transformers import set_seed

    seed = resolve_training_seed(tcfg)
    set_seed(seed)
    logger.debug("training seed %d applied before model build", seed)
    return seed
