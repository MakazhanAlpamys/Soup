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
   markers. Mapping outputs such as ``BatchEncoding`` are read through
   ``input_ids`` and tensor-like ids/masks are normalised to Python integers.
   If assistant messages are present but the returned mask is all zero, this
   path is rejected so the fallback can avoid a silent all-``-100`` training row.

2. **Fallback**: Render ``messages[:i]`` vs ``messages[:i+1]`` for each turn and
   take the token delta. The delta is the new turn's tokens (prefix + content +
   suffix). Leading non-trainable context is deferred until the first user turn,
   because some native templates reject an otherwise-transient system-only
   prefix. Special tokens like BOS are added by the Jinja template itself (not
   by the tokenizer ``__call__``), so monotone-prefix templates produce stable
   deltas. We pass ``add_special_tokens=False`` to incremental tokenize calls so
   HF does not double-prepend BOS at the front of each render. This path is
   necessarily looser than the preferred path — the role-prefix tokens (e.g.
   ``<|assistant|>``) end up in the loss too. Users wanting strict
   assistant-content-only must pass a tokenizer with ``{% generation %}`` markers.
"""

from __future__ import annotations

from collections.abc import Mapping
from difflib import SequenceMatcher
from operator import index
from typing import Any, Optional, Sequence

IGNORE_INDEX = -100
_MISSING = object()


class NoCausalLossTargetError(ValueError):
    """Raised when truncation leaves a row with no shifted causal-LM target."""


def _coerce_int_list(
    values: Any, *, field: str, allow_bool: bool = False
) -> list[int]:
    """Return a one-dimensional tokenizer sequence as Python ``int`` values."""
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            f"tokenizer returned invalid {field}; expected a sequence of integers"
        ) from exc

    result: list[int] = []
    for position, item in enumerate(items):
        if isinstance(item, bool) and not allow_bool:
            raise ValueError(
                f"tokenizer returned non-integer {field}[{position}]={item!r}"
            )
        try:
            item = index(item)
        except TypeError as exc:
            raise ValueError(
                f"tokenizer returned non-integer {field}[{position}]={item!r}"
            ) from exc
        result.append(int(item))
    return result


def _is_mapping_like(out: Any) -> bool:
    """True for ``Mapping`` *or* a duck-typed mapping with ``.get``.

    ``collections.abc.Mapping`` covers ``dict`` and HF ``BatchEncoding``
    (a ``UserDict``). A tokenizer that returns a dict-like object which is
    not registered as a Mapping used to miss this gate: ``coerce_token_ids``
    then iterated the object's *keys* and raised
    ``input_ids[0]='input_ids'``, and ``_apply_template_with_mask`` silently
    skipped the mask path. One predicate, two call sites (#441).
    """
    return isinstance(out, Mapping) or hasattr(out, "get")


def coerce_token_ids(out: Any) -> list[int]:
    """Extract and normalise token ids from a sequence or mapping output.

    Mapping-like outputs are read through ``input_ids``. This is the shared
    contract with ``utils.data_doctor`` — public so a private name cannot
    hide a second copy of the logic (#441 / #430).
    """
    values = out
    if _is_mapping_like(out):
        values = out.get("input_ids", _MISSING)
        if values is _MISSING:
            raise ValueError("tokenizer output mapping has no 'input_ids'")
    return _coerce_int_list(values, field="input_ids")


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
    if not _is_mapping_like(out):
        return None
    masks = out.get("assistant_masks")
    if masks is None:
        return None
    try:
        ids = coerce_token_ids(out)
        mask = _coerce_int_list(
            masks, field="assistant_masks", allow_bool=True
        )
    except ValueError:
        return None
    if len(mask) != len(ids):
        return None
    if any(msg.get("role") == "assistant" for msg in messages) and not any(mask):
        # Some templates accept the mask kwargs but have no {% generation %}
        # markers. Trusting their all-zero mask would silently train no tokens.
        return None
    return ids, mask


def _tokenize_only(tokenizer: Any, messages: Sequence[dict]) -> list[int]:
    """Render ``messages`` into Python int ids without auto-prepending BOS.

    Mapping outputs (including ``BatchEncoding``) are read through
    ``input_ids``. Tensor-like sequences are normalised element by element;
    missing or non-integer ids raise instead of flowing into a collator.
    """
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
    return coerce_token_ids(out)


def _has_user_turn(messages: Sequence[dict]) -> bool:
    """Whether a cumulative conversation contains a user query.

    Qwen3.8's native template refuses a system-only prefix. The incremental
    fallback does not need to render leading ignored context by itself: the
    first user render includes that context and establishes the same boundary
    before an assistant turn is unmasked.
    """
    return any(message.get("role") == "user" for message in messages)


def _unmask_render_insertions(
    labels: list[int],
    full_ids: Sequence[int],
    *,
    baseline_ids: Sequence[int],
    rendered_ids: Sequence[int],
) -> None:
    """Unmask tokens introduced by one message in a renderable prefix."""
    matcher = SequenceMatcher(a=baseline_ids, b=rendered_ids, autojunk=False)
    for tag, _a1, _a2, b1, b2 in matcher.get_opcodes():
        if tag not in {"insert", "replace"}:
            continue
        end = min(b2, len(full_ids))
        labels[b1:end] = full_ids[b1:end]


def _unmask_aligned_render(
    labels: list[int],
    full_ids: Sequence[int],
    *,
    selected_ids: Sequence[int],
    rendered_ids: Sequence[int],
) -> None:
    """Unmask the tokens from a valid standalone turn inside a larger render."""
    matcher = SequenceMatcher(a=selected_ids, b=rendered_ids, autojunk=False)
    for tag, _a1, _a2, b1, b2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        end = min(b2, len(full_ids))
        labels[b1:end] = full_ids[b1:end]


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


def ensure_causal_loss_target(labels: Sequence[int], *, max_length: int) -> None:
    """Reject a row that contributes no token to shifted causal-LM loss.

    Causal language-model losses compare ``logits[:-1]`` with ``labels[1:]``.
    A non-ignored label in position zero therefore does not make a trainable
    row. Check the shifted label surface, not merely ``labels`` itself.
    """
    _validate_max_length(max_length)
    if any(label != IGNORE_INDEX for label in labels[1:]):
        return
    raise NoCausalLossTargetError(
        "no causal-loss target remains after tokenization/truncation at "
        f"data.max_length={max_length}; the assistant response is absent or "
        "fully truncated. Increase data.max_length or shorten the prompt."
    )


def build_assistant_only_labels(
    messages: Sequence[dict],
    tokenizer: Any,
    max_length: int = 2048,
    *,
    include_eot: bool = False,
) -> dict[str, list[int]]:
    """Build labels where only assistant tokens contribute to loss.

    Args:
        messages: Chat messages list (``{"role": ..., "content": ...}``).
        tokenizer: HF tokenizer with a ``chat_template`` set.
        max_length: Truncate to this many tokens.
        include_eot: When True (axolotl ``train_on_eot``), extend each
            assistant span to include the immediately-following EOS / EOT
            token in the unmasked region — so the model learns to predict
            the turn terminator. Default False matches HF Trainer's standard
            chat-template loss-mask behaviour. (v0.53.2 #137)

    Returns:
        ``{"input_ids": [...], "labels": [...], "attention_mask": [...]}``
        where non-assistant positions in ``labels`` are ``IGNORE_INDEX``.

    Raises:
        ValueError: empty messages, non-positive max_length, or tokenizer
            lacking a chat_template or returning invalid token ids. When a
            template reports an all-zero assistant mask despite assistant
            messages, Soup falls back to incremental rendering instead.
        TypeError: ``include_eot`` not bool.
    """
    if not isinstance(include_eot, bool):
        raise TypeError(
            f"include_eot must be bool, got {type(include_eot).__name__}"
        )
    _check_messages(messages)
    _validate_max_length(max_length)

    eos_token_id = _resolve_eos_token_id(tokenizer) if include_eot else None

    preferred = _apply_template_with_mask(tokenizer, messages)
    if preferred is not None:
        input_ids, mask = preferred
        if include_eot and eos_token_id is not None:
            mask = _extend_mask_to_eot(input_ids, mask, eos_token_id)
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
        if msg.get("role") != "assistant" and not _has_user_turn(cumulative):
            # Do not feed a transient system/developer-only prefix to native
            # templates that require a user query (#540). Those tokens remain
            # ignored and are included in the first renderable user prefix.
            continue
        rendered = _tokenize_only(tokenizer, cumulative)
        new_len = len(rendered)
        if msg.get("role") == "assistant":
            end = min(new_len, len(full_ids))
            labels[prev_len:end] = full_ids[prev_len:end]
            if include_eot and eos_token_id is not None:
                # Extend through the immediately-following EOT/EOS run.
                extra = end
                while extra < len(full_ids) and full_ids[extra] == eos_token_id:
                    labels[extra] = full_ids[extra]
                    extra += 1
        prev_len = new_len
    return _truncate(full_ids, labels, max_length)


def build_full_sequence_labels(
    messages: Sequence[dict],
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """Build labels where EVERY token contributes to loss (no masking).

    The ``train_on_responses_only=False`` / ``train_on_messages_with_train_field
    =False`` path. Pre-tokenises with ``add_special_tokens=False`` (via
    ``_tokenize_only``) so the chat template is the single source of special
    tokens, matching inference and ``apply_chat_template(tokenize=True)`` (#785).

    Previously this path handed TRL a ``{"text"}`` column, and TRL's
    language-modeling ``tokenize_fn`` re-tokenised it with the tokenizer's
    default ``add_special_tokens=True``. A template that renders
    ``{{ bos_token }}`` then trained on a doubled BOS, one token off from
    post-#782 inference.
    """
    _check_messages(messages)
    _validate_max_length(max_length)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "tokenizer has no chat_template — cannot render messages for the "
            "text (train_on_responses_only=false) path"
        )
    full_ids = _tokenize_only(tokenizer, messages)
    # #785/#788: the BOS fix must not cost the training EOS. On ``main`` the
    # legacy text rows reached TRL as ``{"text": ...}``, and TRL 0.29.1's
    # language-modeling path appends ``eos_token`` as a STRING to every row that
    # does not already end with it (``sft_trainer.py`` ``add_eos``), before
    # tokenizing -- independent of the tokenizer's post-processor. Pre-tokenising
    # here skips that TRL step, so reproduce its rule in token space: append the
    # EOS id when the sequence does not already end on it. Probing the
    # post-processor instead (an earlier #788 attempt) under-appended for
    # BOS-only and Qwen-shaped tokenizers, dropping the stop token ``main``
    # trained on and teaching run-on generation (the failure `soup data doctor`'s
    # eos_in_labels check exists to catch).
    full_ids = append_training_eos(tokenizer, full_ids)
    labels = list(full_ids)
    return _truncate(full_ids, labels, max_length)


def append_training_eos(tokenizer: Any, input_ids: list[int]) -> list[int]:
    """Append the EOS the SFT language-modeling path trains on (TRL's rule).

    Reproduces TRL 0.29.1's ``add_eos`` (``sft_trainer.py``): the trained text row
    ends in ``eos_token`` unless it already does, regardless of whether the
    tokenizer's post-processor would append one. #785 pre-tokenises the rendered
    template so TRL never runs ``add_eos``; this puts the stop token back in token
    space so the pre-tokenised row trains on the same EOS ``main`` did.

    Used by the live-training path (:func:`build_full_sequence_labels`). The
    ``soup data preprocess`` cache path deliberately does NOT use this -- it
    tokenises exactly as ``main`` (``add_special_tokens=True``, so the
    post-processor supplies the EOS) and only drops #785's duplicated leading
    BOS via :func:`strip_doubled_leading_bos`. Its EOS is therefore whatever the
    post-processor added, not TRL's ``add_eos`` rule, and the resulting
    cache-vs-live mismatch is tracked separately in #791.
    """
    eos_id = _resolve_eos_token_id(tokenizer)
    if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
        return input_ids + [eos_id]
    return input_ids


def strip_doubled_leading_bos(
    tokenizer: Any, input_ids: list[int], attention_mask: list[int]
) -> tuple[list[int], list[int]]:
    """Drop only the one duplicated leading BOS #785 introduced on the cache path.

    ``soup data preprocess`` tokenises the rendered chat template exactly as
    ``main`` did (``add_special_tokens=True``), which keeps ``main``'s truncation
    reservation and post-processor EOS. The only #785 defect on this path is the
    doubled BOS: a template rendering ``{{ bos_token }}`` on a tokenizer whose
    post-processor also prepends BOS yields ``[bos, bos, ...]``. Drop exactly that
    duplicate (the post-processor's leading copy) and nothing else, so the row is
    byte-identical to ``main`` apart from the extra BOS and matches post-#782
    inference's one BOS.

    A single post-processor BOS (no template BOS, e.g. Zephyr/TinyLlama) or none
    at all (Qwen) is not a duplicate and is left as ``main`` had it -- the cache
    is deliberately pinned to ``main`` here, not to the live path's
    template-only rule, and that difference (the live path now yields zero
    leading BOS for a ``data.chat_template`` preset while the cache keeps
    ``main``'s one) is tracked in #876.
    """
    bos_id = _resolve_bos_token_id(tokenizer)
    if (
        bos_id is not None
        and len(input_ids) >= 2
        and input_ids[0] == bos_id
        and input_ids[1] == bos_id
    ):
        return input_ids[1:], attention_mask[1:]
    return input_ids, attention_mask


def _resolve_bos_token_id(tokenizer: Any) -> Optional[int]:
    """Return an int BOS token id, or None. Mirrors :func:`_resolve_eos_token_id`."""
    candidate = getattr(tokenizer, "bos_token_id", None)
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, list):
        for entry in candidate:
            if isinstance(entry, int) and not isinstance(entry, bool):
                return entry
    return None


def _resolve_eos_token_id(tokenizer: Any) -> Optional[int]:
    """Return an int EOS/EOT token id, or None if undetermined.

    Handles tokenizers exposing ``eos_token_id`` as int (most), list[int]
    (e.g. Llama 3 with the additional ``<|eot_id|>`` entry — we pick the
    first int entry), or anything else (str/None/bool → None).
    """
    candidate = getattr(tokenizer, "eos_token_id", None)
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, list):
        for entry in candidate:
            if isinstance(entry, int) and not isinstance(entry, bool):
                return entry
    return None


def _extend_mask_to_eot(
    input_ids: Sequence[int], mask: Sequence[int], eos_token_id: int
) -> list[int]:
    """Mark EOT/EOS tokens immediately following an assistant span as kept.

    Idempotent: a second pass over already-extended output produces the
    same result (no extra EOT absorbed downstream of the original span).
    """
    result = list(mask)
    n = len(input_ids)
    i = 0
    while i < n:
        if result[i]:
            # Walk to the end of this kept span, then absorb trailing EOS.
            j = i
            while j < n and result[j]:
                j += 1
            while j < n and input_ids[j] == eos_token_id:
                result[j] = 1
                j += 1
            # j > i guaranteed: the truthy-span walk advanced j at least once.
            i = j
        else:
            i += 1
    return result


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
    deferred: list[tuple[dict, bool]] = []
    for msg in messages:
        cumulative.append(msg)
        train_flag = msg.get("train")
        if train_flag is None:
            train_flag = msg.get("role") == "assistant"
        if not _has_user_turn(cumulative):
            # Template validity is independent of the per-message train flag:
            # defer every user-less prefix until it can be rendered (#540).
            deferred.append((msg, bool(train_flag)))
            continue
        rendered = _tokenize_only(tokenizer, cumulative)
        new_len = len(rendered)

        if deferred:
            # The first valid render contains both the deferred prefix and the
            # first user turn. Recover each requested prefix span by comparing
            # that render with the same conversation minus one deferred
            # message. This preserves an explicit ``train: true`` without ever
            # asking a native template to render an invalid system-only prefix.
            for deferred_msg, deferred_train_flag in deferred:
                if not deferred_train_flag:
                    continue
                without_message = [
                    item for item in cumulative if item is not deferred_msg
                ]
                _unmask_render_insertions(
                    labels,
                    full_ids,
                    baseline_ids=_tokenize_only(tokenizer, without_message),
                    rendered_ids=rendered,
                )

            # A trainable first user must not accidentally absorb ignored
            # system/developer tokens merely because the initial boundary was
            # deferred. Align its valid standalone render back into the full
            # prefix instead.
            if train_flag:
                _unmask_aligned_render(
                    labels,
                    full_ids,
                    selected_ids=_tokenize_only(tokenizer, [msg]),
                    rendered_ids=rendered,
                )
            deferred.clear()
            prev_len = new_len
            continue

        if train_flag:
            end = min(new_len, len(full_ids))
            labels[prev_len:end] = full_ids[prev_len:end]
        prev_len = new_len
    return _truncate(full_ids, labels, max_length)
