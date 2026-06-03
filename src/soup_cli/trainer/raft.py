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
from typing import Any, List


class RaftDataCollator:
    """Pad pre-tokenised RAFT rows (incl. the custom ``loss_weights`` column)."""

    def __init__(self, tokenizer: Any) -> None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        self.pad_id = int(pad_id)

    def __call__(self, features: List[dict]) -> dict:
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
            input_ids.append(ids + [self.pad_id] * pad)
            attention_mask.append(list(f.get("attention_mask", [1] * n)) + [0] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
            loss_weights.append(
                list(f.get("loss_weights", [1.0] * n)) + [0.0] * pad
            )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
        }


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
