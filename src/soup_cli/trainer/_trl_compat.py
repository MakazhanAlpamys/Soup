"""Version-tolerant access to the ``trl`` APIs the preference trainers use.

``trl`` has broken Soup twice in the same place, and both times the diagnosis
was wrong because it was made by *reading source*. The rule this earned: a
version bound derived by reading source is a hypothesis; the experiment that
settles it is CONSTRUCTING THE OBJECT.

Two independent things move, and they move on different schedules:

**1. ``max_prompt_length`` was removed from the preference configs, in stages.**
Measured by constructing each config with the exact keyword the wrappers pass:

    trl              dpo   kto   orpo   cpo   bco
    0.14.0 - 0.26.2  yes   yes   yes    yes   yes
    0.27.0 - 0.27.2  yes   NO    yes    yes   yes
    0.28.0           yes   NO    NO     NO    NO
    0.29.0 - 1.9.2   NO    NO    NO     NO    NO

There is no successor field. ``max_length`` survives on every version, but
TRL 0.29.0 no longer applies it to prepared DPO prompts and ORPO's legacy
tokenizer can also emit an over-length pair once ``max_prompt_length`` is
gone. Soup therefore restores the removed prompt cap on the tokenized dataset
instead of merely dropping the rejected keyword.

**2. Three configs left the public ``trl`` namespace at 0.29.0.**
``ORPOConfig`` / ``CPOConfig`` / ``BCOConfig`` and their trainers were not
deleted; they live on under ``trl.experimental.<algo>``. That break is harder
than a rejected keyword — it is an ``ImportError`` at ``setup()``, so the task
dies before it can report anything useful.

Both helpers below are deliberately *capability* probes rather than version
comparisons. A version check re-encodes the very table that was wrong twice;
asking the installed object what it accepts cannot go stale.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def _installed_trl_version() -> str:
    """Best-effort version string, for error messages only."""
    try:
        import trl

        return str(getattr(trl, "__version__", "unknown"))
    except Exception:  # pragma: no cover - trl ships in the [train] extra
        return "unknown"


def resolve_trl_symbol(name: str, experimental_module: str | None = None) -> Any:
    """Return ``trl.<name>``, falling back to its ``trl.experimental`` home.

    The public namespace is tried FIRST, so any ``trl`` that still exports the
    symbol keeps the supported, non-experimental path and no warning is
    emitted. The fallback only engages on a ``trl`` that has already moved it.

    ``getattr`` rather than ``hasattr``: ``trl`` exposes these through a lazy
    module, so a submodule that fails to import raises something other than
    ``AttributeError`` and ``hasattr`` would report a clean ``False`` without
    saying why. The original exception is chained onto the raised
    ``ImportError`` rather than swallowed.
    """
    import trl

    public_error: Exception | None = None
    try:
        return getattr(trl, name)
    except Exception as exc:  # noqa: BLE001 - chained below, never swallowed
        public_error = exc

    experimental_error: Exception | None = None
    if experimental_module:
        try:
            return getattr(importlib.import_module(experimental_module), name)
        except Exception as exc:  # noqa: BLE001 - chained below
            experimental_error = exc

    where = f" nor in {experimental_module}" if experimental_module else ""
    raise ImportError(
        f"installed trl {_installed_trl_version()} does not provide {name!r} "
        f"in the `trl` namespace{where}. The [train] extra's version bounds and "
        f"this trainer disagree — see pyproject.toml. "
        f"(trl.{name}: {type(public_error).__name__}: {public_error}"
        + (
            f"; {experimental_module}.{name}: "
            f"{type(experimental_error).__name__}: {experimental_error})"
            if experimental_module
            else ")"
        )
    ) from public_error


def config_accepts(config_cls: type, field: str) -> bool:
    """Does this trl config class actually take ``field`` as a keyword?

    Asked of the class the caller is about to construct, so it stays correct
    across a namespace move as well as a version bump. Configs are dataclasses,
    so the generated ``__init__`` signature is the authoritative list of
    accepted keywords — including the ones inherited from ``TrainingArguments``.
    """
    try:
        return field in inspect.signature(config_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level __init__
        return False


def prompt_length_kwargs(config_cls: type, max_prompt_length: int) -> dict[str, int]:
    """``{'max_prompt_length': N}`` iff this trl's config still accepts it.

    Returns an empty dict on a trl that removed the field, which is the whole
    migration: there is no replacement keyword to pass instead. Behaviour on
    every trl that still has the field is byte-identical to passing it
    directly, so nothing changes for the versions Soup already supported.
    """
    if config_accepts(config_cls, "max_prompt_length"):
        return {"max_prompt_length": max_prompt_length}
    return {}


def kl_penalty_kwargs(config_cls: type, kl_penalty: float) -> dict[str, float]:
    """``{'kl_coef': x}`` or ``{'init_kl_coef': x}``, whichever this trl's config takes.

    trl 0.29 renamed ``init_kl_coef`` to ``kl_coef`` on ``PPOConfig``. The new
    name wins when a config accepts both. Returns an empty dict when it accepts
    neither, so a third rename leaves the coefficient visibly unforwarded
    rather than passing a keyword the config would reject.
    """
    for name in ("kl_coef", "init_kl_coef"):
        if config_accepts(config_cls, name):
            return {name: kl_penalty}
    return {}


def _truncate_tokens(tokens: list[int], limit: int, mode: str) -> list[int]:
    """Truncate one token sequence using TRL's preference-side convention."""
    if limit <= 0:
        return []
    if len(tokens) <= limit:
        return tokens
    if mode == "keep_end":
        return tokens[-limit:]
    return tokens[:limit]


def enforce_preference_sequence_limit(
    dataset: Any,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str,
) -> Any:
    """Restore TRL's removed prompt cap on an already-tokenized dataset.

    TRL 0.29 exposes two preference dataset layouts. DPO stores a shared
    ``prompt_ids`` plus separate completion ids; experimental ORPO stores two
    combined sequences whose prompt span is identified by ``-100`` labels.
    Applying the cap after TRL tokenizes keeps text and conversational inputs
    on the exact same chat-template path while guaranteeing that the tensors
    reaching the model obey ``data.max_length``.
    """
    columns = set(getattr(dataset, "column_names", ()))
    dpo_columns = {"prompt_ids", "chosen_ids", "rejected_ids"}
    orpo_columns = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    }

    if dpo_columns <= columns:

        def cap_dpo(row: dict[str, Any]) -> dict[str, Any]:
            prompt = _truncate_tokens(
                list(row["prompt_ids"]), max_prompt_length, truncation_mode
            )
            completion_limit = max(0, max_length - len(prompt))
            return {
                "prompt_ids": prompt,
                "chosen_ids": list(row["chosen_ids"])[:completion_limit],
                "rejected_ids": list(row["rejected_ids"])[:completion_limit],
            }

        return dataset.map(cap_dpo)

    if orpo_columns <= columns:

        def cap_combined(row: dict[str, Any], prefix: str) -> dict[str, list[int]]:
            input_ids = list(row[f"{prefix}_input_ids"])
            attention_mask = list(row[f"{prefix}_attention_mask"])
            labels = list(row[f"{prefix}_labels"])
            prompt_size = next(
                (index for index, label in enumerate(labels) if label != -100),
                len(labels),
            )
            prompt_ids = _truncate_tokens(
                input_ids[:prompt_size], max_prompt_length, truncation_mode
            )
            prompt_mask = _truncate_tokens(
                attention_mask[:prompt_size], max_prompt_length, truncation_mode
            )
            completion_limit = max(0, max_length - len(prompt_ids))
            completion_ids = input_ids[prompt_size:][:completion_limit]
            completion_mask = attention_mask[prompt_size:][:completion_limit]
            completion_labels = labels[prompt_size:][:completion_limit]
            return {
                f"{prefix}_input_ids": prompt_ids + completion_ids,
                f"{prefix}_attention_mask": prompt_mask + completion_mask,
                f"{prefix}_labels": [-100] * len(prompt_ids) + completion_labels,
            }

        def cap_orpo(row: dict[str, Any]) -> dict[str, Any]:
            prompt_ids = _truncate_tokens(
                list(row["prompt_input_ids"]), max_prompt_length, truncation_mode
            )
            prompt_mask = _truncate_tokens(
                list(row["prompt_attention_mask"]), max_prompt_length, truncation_mode
            )
            return {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_mask,
                **cap_combined(row, "chosen"),
                **cap_combined(row, "rejected"),
            }

        return dataset.map(cap_orpo)

    raise ValueError(
        "TRL prepared a preference dataset with an unknown token layout; "
        f"cannot enforce max_length={max_length}. Columns: {sorted(columns)}"
    )


def preference_rows_with_empty_completion(dataset: Any) -> list[int]:
    """Row indices that would train on zero completion tokens (#1208).

    trl 0.29's CPO truncation (the SimPO path) slices each answer to
    ``answer[: max_length - longer_response_length]`` — identical in 0.29.0 and
    0.29.1 — where ``longer_response_length`` is the *longer* of the two
    answers. A long ``chosen`` against a short ``rejected`` therefore leaves the
    short answer with zero tokens, and SimPO's length-normalised
    log-probability becomes 0/0: every LoRA tensor goes NaN while
    transformers' ``logging_nan_inf_filter`` reports the loss as 0.0.

    ``CPOTrainer`` tokenises and truncates in ``__init__``, so the dataset handed
    to a trainer has already been cut: the longer side is exactly
    ``max_length`` trainable tokens and ``max_length - longer`` is 0. Re-applying
    that arithmetic here flagged every truncated row, including pairs that train
    fine (#1208 review, measured: a 64/14 pair refused). The truncation is
    therefore not re-computed; the question is simply whether one side has no
    trainable label left, which is what a zero-token completion looks like after
    the fact.

    Returns the affected row indices; the caller decides between refusing and
    repairing. An unrecognised layout returns ``[]`` rather than a guess.
    """
    columns = set(getattr(dataset, "column_names", ()))
    if not {"chosen_labels", "rejected_labels"} <= columns:
        # SimPO only builds a CPOTrainer, whose prepared rows carry these
        # columns. The DPO ``chosen_ids`` / ``rejected_ids`` shape is that
        # trainer's collator input, never what a PPO-path dataset looks like.
        return []

    def completion_len(labels: list[int]) -> int:
        """Trainable tokens: everything that is not the ``-100`` prompt span."""
        return sum(1 for label in labels if label != -100)

    return [
        index
        for index, row in enumerate(dataset)
        if min(
            completion_len(list(row["chosen_labels"])),
            completion_len(list(row["rejected_labels"])),
        ) == 0
    ]


def _sharded_at_load(trainer: Any) -> bool:
    """``Trainer.__init__``'s own expression, asked of the trainer's model."""
    return getattr(
        getattr(trainer, "model", None), "is_distributed_loading_by_transformers", False
    )


def _false(trainer: Any) -> bool:
    return False


def _none(trainer: Any) -> None:
    return None


# Attributes ``transformers.Trainer.__init__`` assigns that another ``Trainer``
# method reads on a path trl's experimental ``PPOTrainer`` reaches. Each maps to a
# function of the trainer that returns what ``Trainer.__init__`` would have
# assigned on a single-process run:
#
# - ``is_distributed_loading_by_transformers`` — new in transformers 5.19.0, where
#   ``__init__`` takes it from the model (``False`` unless the model was loaded with
#   a ``distributed_config``) and ``save_model`` reads it on every ordinary save.
#   This is the one that broke ``task: ppo`` (#1675).
# - ``is_fsdp_xla_v1_enabled`` — read by ``save_model`` and
#   ``_save_optimizer_and_scheduler`` wherever ``torch_xla`` is importable; falsy
#   without FSDP.
# - ``hp_name`` — read by ``_get_output_dir`` for a hyperparameter-search trial;
#   ``None`` until ``hyperparameter_search`` names one.
_TRAINER_INIT_DEFAULTS: dict[str, Any] = {
    "is_distributed_loading_by_transformers": _sharded_at_load,
    "is_fsdp_xla_v1_enabled": _false,
    "hp_name": _none,
}


def _attrs_trainer_init_assigns() -> set[str]:
    """Attribute names the installed ``transformers.Trainer.__init__`` stores.

    Read from the bytecode of the function that would have run, through any
    decorator that wraps it, so the answer is about this install and needs no
    version table.
    """
    import dis

    from transformers import Trainer

    names: set[str] = set()
    func: Any = Trainer.__init__
    while func is not None:
        names.update(
            ins.argval for ins in dis.get_instructions(func) if ins.opname == "STORE_ATTR"
        )
        func = getattr(func, "__wrapped__", None)
    return names


def ensure_trainer_init_attrs(trainer: Any) -> list[str]:
    """Give a trainer that skipped ``Trainer.__init__`` the attributes it lacks.

    trl's experimental ``PPOTrainer`` subclasses ``transformers.Trainer`` but has
    an ``__init__`` of its own that never calls ``Trainer.__init__``, and it still
    borrows ``Trainer`` methods (``save_model``, ``_save_checkpoint``). Any
    attribute ``Trainer.__init__`` assigns and those methods read is then missing,
    and the first read raises ``AttributeError`` — which is how transformers
    5.19.0 stopped every PPO run at its first checkpoint (#1675).

    A capability probe on both sides, not a version comparison: a value from
    ``_TRAINER_INIT_DEFAULTS`` is set only when the installed ``Trainer.__init__``
    assigns that attribute AND this object does not have it. So a trainer that did
    run ``Trainer.__init__`` is left exactly as it was, a value trl or the caller
    already set is never replaced, and an older transformers that never had the
    attribute gains nothing. An object that is not a ``transformers.Trainer`` at
    all is returned untouched.

    Returns the names it added, in ``_TRAINER_INIT_DEFAULTS`` order.
    """
    from transformers import Trainer

    if not isinstance(trainer, Trainer):
        return []
    assigned = _attrs_trainer_init_assigns()
    added = []
    for name, default_for in _TRAINER_INIT_DEFAULTS.items():
        if name in assigned and not hasattr(trainer, name):
            setattr(trainer, name, default_for(trainer))
            added.append(name)
    return added
