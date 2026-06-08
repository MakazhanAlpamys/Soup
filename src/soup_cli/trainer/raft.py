"""v0.71.10 #199 — live RAFT trainer pieces.

``make_raft_trainer_class(base_cls)`` builds an HF ``Trainer`` subclass whose
``compute_loss`` does a per-token WEIGHTED cross-entropy: the answer-only loss
mask is expressed via ``loss_weights`` (0.0 on the prompt span, 1.0 on the
answer, boosted on bracketed citation spans when ``citation_faithful`` is set
— #202). With all-1.0 answer weights this reduces exactly to answer-only CE.

``RaftDataCollator`` pads the pre-tokenised ``{input_ids, attention_mask,
labels, loss_weights}`` rows produced by ``utils.raft.tokenize_raft_example``
(``DataCollatorForSeq2Seq`` cannot pad the custom ``loss_weights`` column).

Mirrors the v0.53.11 ``make_prm_trainer_class`` / v0.53.2 ``_DistillTrainer``
factory pattern. Heavy imports (torch) are local.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from soup_cli.utils.raft import (
    _DEFAULT_CITATION_BOOST,
    build_raft_prompt,
    tokenize_raft_example,
)


def _resolve_pad_id(tokenizer: Any) -> int:
    """Resolve a pad token id (pad → eos → 0)."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def _pad_raft_features(features: List[dict], pad_id: int) -> dict:
    """Pad pre-tokenised RAFT rows into batched tensors.

    Shared by :class:`RaftDataCollator` (pre-tokenised rows) and
    :class:`RaftEpochShuffleCollator` (re-tokenised-per-epoch rows).
    """
    import torch

    if not features:
        raise ValueError("RaftDataCollator received an empty batch")
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    labels: List[List[int]] = []
    loss_weights: List[List[float]] = []
    for f in features:
        ids = list(f["input_ids"])
        n = len(ids)
        pad = max_len - n
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append(list(f.get("attention_mask", [1] * n)) + [0] * pad)
        labels.append(list(f["labels"]) + [-100] * pad)
        loss_weights.append(list(f.get("loss_weights", [1.0] * n)) + [0.0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
    }


class RaftDataCollator:
    """Pad pre-tokenised RAFT rows (incl. the custom ``loss_weights`` column)."""

    def __init__(self, tokenizer: Any) -> None:
        self.pad_id = _resolve_pad_id(tokenizer)

    def __call__(self, features: List[dict]) -> dict:
        return _pad_raft_features(features, self.pad_id)


# ---------------------------------------------------------------------------
# v0.71.17 #253 — epoch-aware RAFT document shuffle
# ---------------------------------------------------------------------------


class RaftEpochState:
    """Mutable epoch holder shared between the epoch-shuffle collator and the
    callback (#253).

    ``epoch`` advances at the start of each training epoch so the collator
    re-permutes the RAFT documents per epoch. Without this, the document order
    is baked once at tokenisation time and is fixed across all epochs.
    """

    __slots__ = ("epoch",)

    def __init__(self, epoch: int = 0) -> None:
        self.epoch = int(epoch)


class RaftEpochShuffleCollator:
    """Re-compose + re-tokenise RAW RAFT rows per batch with a per-epoch salt.

    The DataLoader holds RAW ``{query, golden_doc, distractor_docs, answer,
    _raft_row_index}`` rows (``remove_unused_columns=False`` keeps them). Each
    batch is composed with ``build_raft_prompt(..., epoch=state.epoch)`` so the
    document order changes each epoch, then tokenised with the answer-only mask
    (and citation-span boost when ``citation_faithful`` is set) and padded.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        epoch_state: RaftEpochState,
        shuffle_seed: Optional[int] = None,
        citation_faithful: bool = False,
        citation_style: str = "bracket",
        citation_boost: float = _DEFAULT_CITATION_BOOST,
    ) -> None:
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 8
        ):
            raise ValueError("max_length must be an int >= 8")
        if not isinstance(epoch_state, RaftEpochState):
            raise TypeError("epoch_state must be a RaftEpochState")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.epoch_state = epoch_state
        self.shuffle_seed = shuffle_seed
        self.citation_faithful = bool(citation_faithful)
        self.citation_style = citation_style
        self.citation_boost = citation_boost
        self.pad_id = _resolve_pad_id(tokenizer)

    def __call__(self, features: List[dict]) -> dict:
        epoch = int(self.epoch_state.epoch)
        tokenised: List[dict] = []
        for f in features:
            row_index = f.get("_raft_row_index", 0)
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                row_index = 0
            composed = build_raft_prompt(
                f,
                shuffle_seed=self.shuffle_seed,
                row_index=row_index,
                epoch=epoch,
            )
            tokenised.append(
                tokenize_raft_example(
                    self.tokenizer,
                    composed,
                    max_length=self.max_length,
                    citation_faithful=self.citation_faithful,
                    citation_style=self.citation_style,
                    citation_boost=self.citation_boost,
                )
            )
        return _pad_raft_features(tokenised, self.pad_id)


def _try_import_trainer_callback_base():
    """Return HF ``TrainerCallback`` or an ``object`` stand-in.

    Keeps the module importable (+ the callback constructable in tests) when
    ``transformers`` is unavailable — mirrors the curriculum_callback pattern.
    """
    try:
        from transformers import TrainerCallback  # noqa: PLC0415

        return TrainerCallback
    except Exception:  # noqa: BLE001 — transformers optional in some tests.
        return object


def make_raft_epoch_callback(epoch_state: RaftEpochState):
    """Build a ``TrainerCallback`` that advances ``epoch_state`` each epoch.

    ``on_epoch_begin`` reads ``state.epoch`` (a float HF Trainer maintains) and
    writes the int into the shared :class:`RaftEpochState` so the
    :class:`RaftEpochShuffleCollator` re-permutes documents for the new epoch.
    """
    if not isinstance(epoch_state, RaftEpochState):
        raise TypeError("epoch_state must be a RaftEpochState")
    base = _try_import_trainer_callback_base()

    class _RaftEpochCallback(base):  # type: ignore[misc, valid-type]
        def __init__(self, state: RaftEpochState) -> None:
            self._state = state

        def on_epoch_begin(self, args=None, state=None, control=None, **kwargs):
            if state is not None and getattr(state, "epoch", None) is not None:
                self._state.epoch = int(state.epoch)
            return control

    return _RaftEpochCallback(epoch_state)


@lru_cache(maxsize=None)
def make_raft_trainer_class(base_cls: type) -> type:
    """Build a ``Trainer`` subclass with per-token weighted-CE ``compute_loss``.

    Cached so repeated ``raft`` runs against the same base class share one
    subclass (mirrors ``make_prm_trainer_class`` / ``make_multipack_trainer_class``).
    """

    class _RaftTrainer(base_cls):  # type: ignore[valid-type, misc]
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            import torch
            from torch.nn.functional import cross_entropy

            labels = inputs.get("labels")
            loss_weights = inputs.get("loss_weights")
            model_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ("labels", "loss_weights")
            }
            outputs = model(**model_inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token = cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )  # [B*S'] — 0.0 at ignored positions.
            if loss_weights is not None:
                w = loss_weights[:, 1:].contiguous().reshape(-1).to(per_token.dtype)
                # Zero the weight wherever the label is ignored so padded /
                # prompt tokens never enter the weighted mean even if a
                # non-zero weight leaked in.
                valid = (shift_labels.reshape(-1) != -100).to(per_token.dtype)
                w = w * valid
            else:
                w = (shift_labels.reshape(-1) != -100).to(per_token.dtype)
            denom = w.sum().clamp(min=1.0)
            loss = (per_token * w).sum() / denom
            if not torch.isfinite(loss):
                # Degenerate batch (all-masked) OR a non-finite forward —
                # return a STRUCTURAL zero (never NaN even if per_token has a
                # NaN: `nan * 0.0 == nan`, so anchor on a fresh zeros tensor
                # with grad rather than `per_token.mean() * 0.0`).
                loss = torch.zeros(
                    (), device=per_token.device, dtype=per_token.dtype,
                    requires_grad=True,
                )
            return (loss, outputs) if return_outputs else loss

    _RaftTrainer.__name__ = f"_RaftTrainer_{base_cls.__name__}"
    return _RaftTrainer
