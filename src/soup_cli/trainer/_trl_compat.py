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

# TRL's own KTO/BCO/CPO/ORPO tokenizers historically truncate an over-length
# prompt with ``keep_end`` — the tail carries a chat template's generation
# header (the assistant turn the completion must follow), so cutting from the
# start would strip that header. KTOConfig exposes no ``truncation_mode`` field
# for the wrappers to read, so the cap KTO applies through
# ``enforce_preference_sequence_limit`` uses this default; it must match TRL's
# historical direction rather than the ``keep_start`` most other configs default
# to.
DEFAULT_PROMPT_TRUNCATION_MODE = "keep_end"

#: Hugging Face's cross-entropy ignore index; masks prompt tokens out of the
#: completion loss.
_IGNORE_INDEX = -100


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


def _cap_combined_preference_sequence(
    row: dict[str, Any],
    prefix: str,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str,
) -> dict[str, list[int]]:
    """Cap a combined prompt/completion row split by the first trainable label."""
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
        f"{prefix}_labels": [_IGNORE_INDEX] * len(prompt_ids) + completion_labels,
    }


def _rebuild_unpaired_sequence(
    prompt_ids: list[int],
    prompt_mask: list[int],
    answer_ids: list[int],
    answer_mask: list[int],
    eos_id: int,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str,
) -> dict[str, list[int]]:
    """Build a capped KTO/BCO sequence while keeping EOS as the final token."""
    capped_prompt_ids = _truncate_tokens(list(prompt_ids), max_prompt_length, truncation_mode)
    capped_prompt_mask = _truncate_tokens(list(prompt_mask), max_prompt_length, truncation_mode)
    answer_limit = max(0, max_length - len(capped_prompt_ids) - 1)
    capped_answer_ids = list(answer_ids)[:answer_limit]
    capped_answer_mask = list(answer_mask)[:answer_limit]
    return {
        "input_ids": capped_prompt_ids + capped_answer_ids + [eos_id],
        "attention_mask": capped_prompt_mask + capped_answer_mask + [1],
        "labels": [_IGNORE_INDEX] * len(capped_prompt_ids) + capped_answer_ids + [eos_id],
    }


def _rotated_answer_index(index: int, row_count: int, chunk_size: int) -> int:
    """Return the answer row TRL rotates into KTO's KL completion for ``index``."""
    chunk = max(1, chunk_size)
    chunk_start = (index // chunk) * chunk
    chunk_end = min(chunk_start + chunk, row_count)
    chunk_width = chunk_end - chunk_start
    if chunk_width <= 1:
        return index
    return chunk_start + ((index - chunk_start + 1) % chunk_width)


def _rebuild_unpaired_completion_row(
    row: dict[str, Any],
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str,
) -> dict[str, list[int]]:
    """Reassemble the KTO/BCO completion from the untruncated answer tokens.

    TRL's KTO/BCO tokenizer caps the answer to ``max_length - len(prompt)``
    while ignoring ``max_prompt_length`` (``_process_tokens`` truncates
    ``answer_input_ids`` before concatenating). An over-length prompt therefore
    fills the whole budget and leaves ``completion_input_ids`` with the answer
    already sliced away, so capping that column can only recover an empty
    completion. The matched answer survives verbatim in ``answer_input_ids`` /
    ``answer_attention_mask`` (columns the loss never reads) and the prompt in
    ``prompt_input_ids``, so rebuild the combined sequence from those before the
    real cap runs. TRL guarantees ``completion_input_ids`` ends with EOS; reuse
    that token so a prompt-driven cap never drops the stop signal.
    """
    eos_id = list(row["completion_input_ids"])[-1]
    rebuilt = _rebuild_unpaired_sequence(
        list(row["prompt_input_ids"]),
        list(row["prompt_attention_mask"]),
        list(row["answer_input_ids"]),
        list(row["answer_attention_mask"]),
        eos_id,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        truncation_mode=truncation_mode,
    )
    return {
        "completion_input_ids": rebuilt["input_ids"],
        "completion_attention_mask": rebuilt["attention_mask"],
        "completion_labels": rebuilt["labels"],
    }


def enforce_preference_sequence_limit(
    dataset: Any,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str,
    only_when_overflow: bool = False,
    kl_chunk_size: int | None = None,
) -> Any:
    """Restore TRL's removed prompt cap on an already-tokenized dataset.

    TRL 0.29 exposes a few preference dataset layouts. DPO stores a shared
    ``prompt_ids`` plus separate completion ids; experimental ORPO/CPO store
    two combined sequences whose prompt span is identified by ``-100`` labels;
    BCO/KTO store one unpaired ``completion_*`` sequence, with KTO adding a
    ``KL_completion_*`` sequence as well. Applying the cap after TRL tokenizes
    keeps text and conversational inputs on the exact same chat-template path
    while guaranteeing that the tensors reaching the model obey
    ``data.max_length``.

    ``only_when_overflow`` preserves rows whose TRL-built model inputs already
    fit inside ``max_length``; DPO and ORPO keep the legacy unconditional cap,
    while BCO, IPO, KTO and SimPO use the conditional path.
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
    unpaired_columns = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "completion_input_ids",
        "completion_attention_mask",
        "completion_labels",
    }

    if dpo_columns <= columns:

        def cap_dpo(row: dict[str, Any]) -> dict[str, Any]:
            if only_when_overflow and len(row["prompt_ids"]) + max(
                len(row["chosen_ids"]), len(row["rejected_ids"])
            ) <= max_length:
                return {}
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

        def cap_orpo(row: dict[str, Any]) -> dict[str, Any]:
            if only_when_overflow and (
                len(row["chosen_input_ids"]) <= max_length
                and len(row["rejected_input_ids"]) <= max_length
            ):
                return {}
            prompt_ids = _truncate_tokens(
                list(row["prompt_input_ids"]), max_prompt_length, truncation_mode
            )
            prompt_mask = _truncate_tokens(
                list(row["prompt_attention_mask"]), max_prompt_length, truncation_mode
            )
            return {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_mask,
                **_cap_combined_preference_sequence(
                    row,
                    "chosen",
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                ),
                **_cap_combined_preference_sequence(
                    row,
                    "rejected",
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                ),
            }

        return dataset.map(cap_orpo)

    if unpaired_columns <= columns:
        answer_columns = {"answer_input_ids", "answer_attention_mask"}
        has_answer = answer_columns <= columns
        kl_columns = {
            "KL_prompt_input_ids",
            "KL_prompt_attention_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
            "KL_completion_labels",
        }
        has_kl = kl_columns <= columns
        answers = list(dataset["answer_input_ids"]) if has_answer and has_kl else []
        answer_masks = list(dataset["answer_attention_mask"]) if answers else []
        kl_chunk = max(1, int(kl_chunk_size or 1))

        def cap_unpaired(row: dict[str, Any], index: int) -> dict[str, Any]:
            fits = len(row["completion_input_ids"]) <= max_length and (
                not has_kl or len(row["KL_completion_input_ids"]) <= max_length
            )
            if only_when_overflow and fits:
                return {}
            prompt_ids = _truncate_tokens(
                list(row["prompt_input_ids"]), max_prompt_length, truncation_mode
            )
            prompt_mask = _truncate_tokens(
                list(row["prompt_attention_mask"]), max_prompt_length, truncation_mode
            )
            # TRL already sliced the answer out of ``completion_input_ids`` when
            # the prompt overflowed; rebuild it from the untruncated answer
            # columns so the cap keeps real completion tokens, not just the EOS.
            completion_source = (
                _rebuild_unpaired_completion_row(
                    row,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                )
                if has_answer
                else row
            )
            capped = {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_mask,
                **_cap_combined_preference_sequence(
                    completion_source,
                    "completion",
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                    truncation_mode=truncation_mode,
                ),
            }
            if {"KL_prompt_input_ids", "KL_prompt_attention_mask"} <= columns:
                capped["KL_prompt_input_ids"] = _truncate_tokens(
                    list(row["KL_prompt_input_ids"]),
                    max_prompt_length,
                    truncation_mode,
                )
                capped["KL_prompt_attention_mask"] = _truncate_tokens(
                    list(row["KL_prompt_attention_mask"]),
                    max_prompt_length,
                    truncation_mode,
                )
            if {
                "KL_completion_input_ids",
                "KL_completion_attention_mask",
                "KL_completion_labels",
            } <= columns:
                if answers:
                    eos_id = list(row["KL_completion_input_ids"])[-1]
                    lender = _rotated_answer_index(index, len(answers), kl_chunk)
                    kl = _rebuild_unpaired_sequence(
                        list(row["KL_prompt_input_ids"]),
                        list(row["KL_prompt_attention_mask"]),
                        list(answers[lender]),
                        list(answer_masks[lender]),
                        eos_id,
                        max_length=max_length,
                        max_prompt_length=max_prompt_length,
                        truncation_mode=truncation_mode,
                    )
                    capped["KL_completion_input_ids"] = kl["input_ids"]
                    capped["KL_completion_attention_mask"] = kl["attention_mask"]
                    capped["KL_completion_labels"] = kl["labels"]
                else:
                    capped.update(
                        _cap_combined_preference_sequence(
                            row,
                            "KL_completion",
                            max_length=max_length,
                            max_prompt_length=max_prompt_length,
                            truncation_mode=truncation_mode,
                        )
                    )
            return capped

        return dataset.map(cap_unpaired, with_indices=True)

    raise ValueError(
        "TRL prepared a preference dataset with an unknown token layout; "
        f"cannot enforce max_length={max_length}. Columns: {sorted(columns)}"
    )
