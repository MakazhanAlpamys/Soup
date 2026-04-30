"""Assistant-only loss masking (v0.36.0 Part A).

Builds ``{input_ids, labels, attention_mask}`` such that only assistant
content tokens contribute to the SFT loss; everything else is ``-100``
(``IGNORE_INDEX``).

Mirrors:
- LlamaFactory ``processor/supervised.py`` (IGNORE_INDEX on non-assistant).
- Axolotl ``prompt_strategies/chat_template.py`` (per-message train field).

Two strategies:

1. **Preferred**: ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True,
   return_dict=True)``. Available on HF templates that declare ``{% generation %}``
   markers. Honest, exact, no heuristic.

2. **Fallback**: Render ``messages[:i]`` vs ``messages[:i+1]`` for each turn and
   take the token delta. The delta is the new turn's tokens (prefix + content +
   suffix). Special tokens like BOS are added by the Jinja template itself
   (not by the tokenizer ``__call__``), so monotone-prefix templates produce
   stable deltas. We pass ``add_special_tokens=False`` to incremental tokenize
   calls so HF does not double-prepend BOS at the front of each render. This
   path is necessarily looser than the preferred path — the role-prefix tokens
   (e.g. ``<|assistant|>``) end up in the loss too. Users wanting strict
   assistant-content-only must pass a tokenizer with ``{% generation %}`` markers.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

IGNORE_INDEX = -100


def _validate_max_length(max_length: int) -> None:
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise ValueError("max_length must be an int")
    if max_length <= 0:
        raise ValueError("max_length must be positive")


def _check_messages(messages: Sequence[dict]) -> None:
    if not messages:
        raise ValueError("messages list is empty")


def _apply_template_with_mask(
    tokenizer: Any, messages: Sequence[dict]
) -> Optional[tuple[list[int], list[int]]]:
    """Try the preferred path. Returns (input_ids, mask) or None on failure."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template — cannot mask labels")
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
    except TypeError:
        # Old HF that doesn't recognise return_assistant_tokens_mask.
        return None
    if not isinstance(out, dict):
        return None
    masks = out.get("assistant_masks")
    ids = out.get("input_ids")
    if masks is None or ids is None:
        return None
    if len(masks) != len(ids):
        return None
    return list(ids), list(masks)


def _tokenize_only(tokenizer: Any, messages: Sequence[dict]) -> list[int]:
    """Render and tokenize ``messages``; never let HF auto-prepend BOS again."""
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
    except TypeError:
        # Older tokenizers that reject add_special_tokens kwarg.
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    if isinstance(out, dict):
        return list(out.get("input_ids", []))
    return list(out)


def _truncate(
    input_ids: list[int], labels: list[int], max_length: int
) -> dict[str, list[int]]:
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_assistant_only_labels(
    messages: Sequence[dict],
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """Build labels where only assistant tokens contribute to loss.

    Args:
        messages: Chat messages list (``{"role": ..., "content": ...}``).
        tokenizer: HF tokenizer with a ``chat_template`` set.
        max_length: Truncate to this many tokens.

    Returns:
        ``{"input_ids": [...], "labels": [...], "attention_mask": [...]}``
        where non-assistant positions in ``labels`` are ``IGNORE_INDEX``.

    Raises:
        ValueError: empty messages, non-positive max_length, or tokenizer
            lacking a chat_template.
    """
    _check_messages(messages)
    _validate_max_length(max_length)

    preferred = _apply_template_with_mask(tokenizer, messages)
    if preferred is not None:
        input_ids, mask = preferred
        labels = [
            tok if flag else IGNORE_INDEX
            for tok, flag in zip(input_ids, mask)
        ]
        return _truncate(input_ids, labels, max_length)

    # --- Fallback: incremental delta ---
    full_ids = _tokenize_only(tokenizer, messages)
    labels: list[int] = [IGNORE_INDEX] * len(full_ids)
    prev_len = 0
    cumulative: list[dict] = []
    for msg in messages:
        cumulative.append(msg)
        rendered = _tokenize_only(tokenizer, cumulative)
        new_len = len(rendered)
        if msg.get("role") == "assistant":
            end = min(new_len, len(full_ids))
            labels[prev_len:end] = full_ids[prev_len:end]
        prev_len = new_len
    return _truncate(full_ids, labels, max_length)


def build_per_message_train_labels(
    messages: Sequence[dict],
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """Build labels using per-message ``train: bool`` field.

    For each message, the ``train`` flag (defaulting to ``role == "assistant"``
    when missing) decides whether its tokens contribute to loss.

    Mirrors Axolotl ``message_field_training`` behaviour.
    """
    _check_messages(messages)
    _validate_max_length(max_length)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template — cannot mask labels")

    full_ids = _tokenize_only(tokenizer, messages)
    labels: list[int] = [IGNORE_INDEX] * len(full_ids)
    prev_len = 0
    cumulative: list[dict] = []
    for msg in messages:
        cumulative.append(msg)
        rendered = _tokenize_only(tokenizer, cumulative)
        new_len = len(rendered)
        train_flag = msg.get("train")
        if train_flag is None:
            train_flag = msg.get("role") == "assistant"
        if train_flag:
            end = min(new_len, len(full_ids))
            labels[prev_len:end] = full_ids[prev_len:end]
        prev_len = new_len
    return _truncate(full_ids, labels, max_length)
