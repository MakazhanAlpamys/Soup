"""Response-only token masking for the MLX SFT path (#683).

``data.train_on_responses_only`` defaults to ``True`` and promises that only
assistant content contributes to the SFT loss. The MLX wrapper passed no mask
at all, so every MLX SFT run trained on system and user turns as well.

The obvious repair -- setting ``mask_prompt`` on the args object that
``mlx_lm.tuner.datasets.create_dataset`` reads -- is correct for
prompt/completion data and **wrong for multi-turn chat**. Upstream's
``ChatDataset`` masks a single prefix ending before ``messages[-1]``
(``datasets.py:65-75``), so on a conversation with two assistant turns it
supervises only the last one and silently drops the first from the loss.
Measured on Qwen2.5 with two assistant turns: 46 supervised tokens before,
4 after, and the 4 are not the ones the contract asks for.

The reason a prefix is all upstream can express is ``default_loss``
(``trainer.py:86-96``), which rebuilds the mask from an ``(offset, length)``
pair -- one contiguous span per sample. Multiple assistant spans are simply not
representable in that pair.

``train()`` takes ``loss`` and ``iterate_batches`` as callables
(``trainer.py:224-225``), so this module supplies a real per-token mask through
those hooks instead: a dataset that emits ``(tokens, mask)``, a batching
function that pads the mask alongside the tokens, and a loss that multiplies by
it. Same ``train()``, same callback, no fork and no vendored upstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

RESPONSE_ROLE = "assistant"

# Row-shape keys, mirroring `mlx_lm.tuner.datasets.create_dataset`'s defaults.
_CHAT_KEY = "messages"
_PROMPT_KEY = "prompt"
_COMPLETION_KEY = "completion"

# Padding granularity, mirroring `mlx_lm.tuner.trainer.iterate_batches`. Kept
# equal so a masked run batches to the same shapes an unmasked one would.
_PAD_TO = 32


@dataclass(frozen=True)
class MaskingPlan:
    """How one dataset shape must be masked, decided before any model loads.

    Split out of ``train()`` so the dispatch is testable on any platform: the
    three branches differ in which upstream code path they are correct for, and
    picking the wrong one is silent in two of the three cases.
    """

    token_mask: bool
    """Use Soup's per-token mask -- the only option correct for multi-turn chat."""

    mask_prompt: bool
    """Set upstream's single-prefix flag; correct only for prompt/completion."""

    warning: str = ""
    """Non-empty when the request cannot be honoured for this shape."""


def plan_response_masking(responses_only: bool, sample: Dict[str, Any]) -> MaskingPlan:
    """Choose a masking strategy from the config flag and one sample row.

    Dispatches on the same row shapes upstream's ``create_dataset`` does
    (``datasets.py:180-199``) rather than on a guess about the loader:

    * chat rows -> Soup's per-token mask. Upstream's flag would supervise only
      the final assistant turn.
    * prompt/completion rows -> upstream's flag. Its single masked prefix *is*
      the whole prompt here, so it is exactly right and needs no replacement.
    * anything else (plain text) -> neither. Upstream **raises**
      ``ValueError("Prompt masking not supported for text dataset.")`` when the
      flag is set on text rows, so setting it would convert a silently-wrong
      run into a crash. Warn instead, which is what the rest of
      ``_check_unsupported`` does for fields MLX cannot honour.
    """
    if not responses_only:
        return MaskingPlan(token_mask=False, mask_prompt=False)
    if _CHAT_KEY in sample:
        return MaskingPlan(token_mask=True, mask_prompt=False)
    if _PROMPT_KEY in sample and _COMPLETION_KEY in sample:
        return MaskingPlan(token_mask=False, mask_prompt=True)
    return MaskingPlan(
        token_mask=False,
        mask_prompt=False,
        warning=(
            "data.train_on_responses_only (plain-text rows carry no role "
            "boundaries to mask; use chatml or prompt/completion data)"
        ),
    )


class ResponseMaskError(ValueError):
    """Raised when assistant spans cannot be recovered for a conversation.

    The contract is explicit refusal rather than silent approximation: a run
    that quietly supervises the wrong tokens is the failure #683 is about, and
    producing a *different* wrong distribution would not be an improvement.
    """


def _apply(tokenizer, messages, *, tools=None, add_generation_prompt=False):
    return list(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    )


def build_response_mask(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Any,
    *,
    tools: Any = None,
) -> Tuple[List[int], List[int]]:
    """Tokenize ``messages`` and mark which tokens are assistant content.

    Returns ``(tokens, mask)`` with ``mask[i] == 1`` exactly when token ``i``
    belongs to an assistant turn -- every assistant turn, not just the last.

    The span of message ``k`` is recovered as the token delta between the
    rendering of ``messages[:k]`` and ``messages[:k+1]``, with
    ``add_generation_prompt`` set for assistant turns so the ``<|im_start|>
    assistant`` header falls on the masked side of the boundary rather than
    being supervised as if the model had produced it.

    That only holds if the chat template is *prefix-stable* -- each partial
    rendering must be a genuine prefix of the full one. Templates that emit a
    trailing marker, re-order turns, or inject a summary of earlier turns are
    not, and for those the spans this would compute are wrong. Every prefix is
    checked against the full sequence and ``ResponseMaskError`` is raised
    rather than returning a mask that looks plausible.
    """
    messages = list(messages)
    if not messages:
        raise ResponseMaskError("conversation has no messages")

    tokens = _apply(tokenizer, messages, tools=tools)
    mask = [0] * len(tokens)
    saw_response = False

    for k, message in enumerate(messages):
        is_response = message.get("role") == RESPONSE_ROLE
        if k == 0:
            # HuggingFace tokenizers raise on an empty conversation
            # ("Cannot apply chat template to an empty conversation"), so the
            # k == 0 prefix is taken as empty rather than rendered. A leading
            # assistant turn is refused instead: its generation-prompt header
            # cannot be rendered without a preceding turn, so it would be
            # supervised as if the model had produced it.
            if is_response:
                raise ResponseMaskError(
                    "conversation begins with an assistant turn; there is no "
                    "preceding context to render its generation prompt "
                    "against, so its header cannot be excluded from the loss"
                )
            prefix: List[int] = []
        else:
            prefix = _apply(
                tokenizer,
                messages[:k],
                tools=tools,
                add_generation_prompt=is_response,
            )
        upto = _apply(tokenizer, messages[: k + 1], tools=tools)

        # Prefix-stability, checked rather than assumed. `upto` is checked too:
        # a template that renders the final turn differently from an
        # intermediate one would pass the `prefix` check and still misplace the
        # span end.
        if tokens[: len(prefix)] != prefix or tokens[: len(upto)] != upto:
            raise ResponseMaskError(
                f"chat template is not prefix-stable at message {k} "
                f"(role={message.get('role')!r}): a partial rendering is not a "
                "prefix of the full one, so assistant spans cannot be located "
                "by token alignment"
            )
        if len(upto) < len(prefix):
            raise ResponseMaskError(
                f"chat template produced a shorter rendering for message {k} "
                "than for the turns before it"
            )

        if is_response:
            for i in range(len(prefix), len(upto)):
                mask[i] = 1
            saw_response = len(upto) > len(prefix) or saw_response

    if not saw_response:
        raise ResponseMaskError(
            "conversation contains no assistant content to supervise; with "
            "data.train_on_responses_only every row must have at least one "
            "non-empty assistant turn, or the row contributes no loss signal"
        )
    return tokens, mask


class MaskedChatDataset:
    """A chat dataset whose ``process`` returns ``(tokens, per-token mask)``.

    Deliberately the same shape as ``mlx_lm.tuner.datasets.ChatDataset`` --
    ``process`` / ``__getitem__`` / ``__len__`` -- so upstream's
    ``CacheDataset`` wraps it unchanged and the caching behaviour is identical.
    """

    def __init__(self, data: Sequence[Dict[str, Any]], tokenizer: Any, chat_key: str = "messages"):
        self._data = list(data)
        self.tokenizer = tokenizer
        self.chat_key = chat_key

    def process(self, d: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        return build_response_mask(
            d[self.chat_key], self.tokenizer, tools=d.get("tools")
        )

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def masked_iterate_batches(
    dataset,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: Any = None,
    comm_group: Any = None,
) -> Iterable[Tuple[Any, Any]]:
    """``iterate_batches`` yielding a padded per-token mask instead of spans.

    Signature matches ``mlx_lm.tuner.trainer.iterate_batches`` because
    ``train()`` and ``evaluate()`` both call it by keyword. Padding, the
    ``pad_to`` granularity and the truncation warning mirror upstream so a
    masked run batches into the same shapes an unmasked one would; the only
    difference is the second element of each yielded pair.

    Truncation is applied to the mask as well as the tokens. A conversation cut
    off before its assistant turn therefore contributes no supervised tokens
    rather than a mask that points past the end of the row.
    """
    import mlx.core as mx
    import numpy as np

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} examples "
            f"but only has {len(dataset)}."
        )

    idx = list(range(len(dataset)))
    if comm_group is not None:
        offset, step = comm_group.rank(), comm_group.size()
    else:
        offset, step = 0, 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    if seed:
        np.random.seed(seed)

    warned = False
    while True:
        for i in np.random.permutation(len(batch_idx)):
            rows = [dataset[j] for j in batch_idx[i]]
            toks, masks = zip(*rows)
            lengths = [len(t) for t in toks]
            if max(lengths) > max_seq_length and not warned:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} "
                    f"tokens. The longest sentence {max(lengths)} will be "
                    f"truncated to {max_seq_length}."
                )
                warned = True

            width = 1 + _PAD_TO * ((max(lengths) + _PAD_TO - 1) // _PAD_TO)
            width = min(width, max_seq_length)

            n = batch_size // step
            tok_arr = np.zeros((n, width), np.int32)
            mask_arr = np.zeros((n, width), np.int32)
            for j in range(n):
                keep = min(lengths[j], width)
                tok_arr[j, :keep] = toks[j][:keep]
                mask_arr[j, :keep] = masks[j][:keep]

            yield mx.array(tok_arr), mx.array(mask_arr)

        if not loop:
            break


def masked_loss(model, batch, masks):
    """Cross-entropy over the supervised tokens only.

    Mirrors ``mlx_lm.tuner.trainer.default_loss`` and returns the same
    ``(mean_loss, ntoks)`` pair, so ``train()``'s reporting, its
    ``TrainingCallback`` and its throughput accounting are unaffected. The one
    change is where the mask comes from: a per-token array rather than a span
    rebuilt from ``(offset, length)``.

    ``masks[:, 1:]`` is the alignment. ``targets`` is ``batch[:, 1:]``, so
    target position ``i`` is original token ``i + 1``, and the mask has to be
    shifted with it -- an unshifted mask supervises the token *before* each
    assistant token, which is off by one into the prompt.
    """
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    m = masks[:, 1:]
    ce = nn.losses.cross_entropy(logits, targets) * m
    ntoks = m.sum()
    # A batch whose supervised tokens were all truncated away yields 0/0.
    # Clamping the denominator makes that batch contribute exactly zero rather
    # than a nan that poisons every subsequent gradient. `ce.sum()` is zero in
    # that case, so the clamp cannot invent signal.
    ce = ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1)
    return ce, ntoks
