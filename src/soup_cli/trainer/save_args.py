"""The checkpoint-saving arguments every HF-based wrapper hands to its trainer (#2575).

``training.save_strategy`` is ``steps`` (the default), ``epoch`` or ``no``. About twenty
wrappers used to spell ``save_steps=tcfg.save_steps, save_total_limit=3`` themselves and
never set a strategy, so transformers' own default (``"steps"``) applied, and with it
the save at the last step (``DefaultFlowCallback.on_step_end`` sets ``should_save`` once
``global_step >= max_steps``): every run left a ``checkpoint-N`` folder beside the final
adapter and nothing in a ``soup.yaml`` could turn that off.

Every wrapper now takes the arguments from :func:`save_kwargs`, so the setting cannot be
honoured by nineteen of them and forgotten by the twentieth.
``tests/test_save_strategy.py::TestTheRatchet`` fails on a wrapper that spells one of the
three names itself, and on one that builds HF arguments without calling this.

The final model or adapter is not a checkpoint: each wrapper writes it with
``trainer.save_model`` after training whatever the strategy is, and ``no`` leaves that
untouched. An explicit save request (the Ctrl-C save in ``utils/graceful_save.py`` sets
``should_save`` itself) is also honoured under ``no``.

The MLX loops (``mlx_sft`` / ``mlx_dpo`` / ``mlx_grpo``) are not ``TrainingArguments``:
they write a numbered adapter file every ``save_steps`` iterations themselves, so
``save_strategy`` other than ``steps`` is refused on ``backend: mlx`` at config load
(``SoupConfig._validate_save_strategy_is_honoured``) instead of being ignored.

Imports nothing heavy: it runs at config build time, before torch is wanted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: How many checkpoints a run keeps. It was ``save_total_limit=3`` in twenty places.
SAVE_TOTAL_LIMIT = 3


def save_kwargs(
    tcfg: Any,
    *,
    accepts: Callable[[str], bool] | None = None,
    config_name: str = "the trainer config",
) -> dict[str, Any]:
    """The ``save_*`` keyword arguments for a ``TrainingArguments`` or TRL config.

    ``steps``: strategy, ``save_steps`` and the limit (what every wrapper did before).
    ``epoch`` / ``no``: strategy and the limit only; ``save_steps`` means nothing to
    transformers there, and the schema refuses an explicit one.

    ``accepts`` is for a config class that varies across trl versions (PPO): when given,
    only the keywords it accepts are returned, as the wrapper filtered them before. A
    class that cannot take ``save_strategy`` cannot honour ``epoch`` / ``no``, which is
    refused (``ValueError``) rather than dropped.
    """
    strategy = tcfg.save_strategy
    kwargs: dict[str, Any] = {"save_strategy": strategy}
    if strategy == "steps":
        kwargs["save_steps"] = tcfg.save_steps
    kwargs["save_total_limit"] = SAVE_TOTAL_LIMIT
    if accepts is None:
        return kwargs
    if strategy != "steps" and not accepts("save_strategy"):
        raise ValueError(
            f"training.save_strategy: {strategy!r} cannot be honoured: {config_name} "
            "does not take save_strategy. Use save_strategy: steps, or upgrade trl."
        )
    return {name: value for name, value in kwargs.items() if accepts(name)}
