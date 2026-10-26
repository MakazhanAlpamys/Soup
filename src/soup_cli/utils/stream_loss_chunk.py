"""The output head and the causal-LM loss in token chunks (#1693).

Layer streaming bounds the weights. What still grows with ``batch x seq`` is
the output head plus the loss: the head builds a ``[batch, seq, vocab]`` logits
tensor and ``ForCausalLMLoss`` upcasts it to fp32 and keeps two more buffers of
that shape at the backward peak. With a 128 256-token vocabulary that is
0.92 GB at 512 tokens and 3.68 GB at 2048, whatever the model's size.

Here the head and the loss run ``chunk`` positions at a time. Each chunk runs
the head, takes the cross-entropy sum, takes the gradient on its own slice of
the hidden states and drops its logits; the slices fill one ``[batch, seq,
hidden]`` tensor that the backward pass hands on. No graph over the logits
outlives its chunk, so the logits term is ``vocab x chunk x batch`` and no
longer depends on the sequence length.

The per-token terms are the ones ``ForCausalLMLoss`` computes. What differs is
the order of the final sum and the shape of each matmul, so the loss and the
gradients agree with the unchunked step to rounding, not bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChunkedStep:
    """What a labelled forward leaves for the trainer's metrics.

    TRL's ``SFTTrainer`` reads ``outputs.logits`` for its ``entropy`` and
    ``mean_token_accuracy`` columns, and the rewind log for its per-row
    losses; a chunked step has no logits to give them.
    """

    token_entropy: Any  # [batch, seq], the entropy of each position's distribution
    token_loss: Any  # [batch, seq] float32, each labelled position's cross-entropy
    correct: Any  # labelled positions whose argmax is the label
    labelled: Any  # positions with a label


def shift_for_next_token(labels: Any, ignore_index: int = IGNORE_INDEX) -> Any:
    """``labels`` moved one position left, as ``ForCausalLMLoss`` does: position
    ``t`` is scored against token ``t + 1`` and the last position against
    nothing."""
    from torch.nn import functional

    return functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]


def _chunk_terms(head: Callable[[Any], Any], piece: Any, target: Any, ignore_index: int):
    """One chunk: the cross-entropy sum, and what the metrics need, the latter
    without gradients."""
    import torch
    from torch.nn import functional

    logits = head(piece)
    with torch.no_grad():
        # In the head's own dtype, as TRL's `entropy_from_logits` reads them.
        log_probs = functional.log_softmax(logits, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(-1)
        labelled = target != ignore_index
        correct = ((logits.argmax(dim=-1) == target) & labelled).sum()
        # Also in the head's dtype and only then cast, as `row_losses` does.
        at_label = log_probs.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_loss = (-at_label).float() * labelled
        # Gone before the fp32 copy below is made, so the two do not add up.
        del log_probs
    # `ForCausalLMLoss` upcasts before the loss; so does each chunk.
    wide = logits.float()
    total = functional.cross_entropy(
        wide.view(-1, wide.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return total, entropy, token_loss, correct


def _chunked_function():
    import torch

    class _ChunkedHeadLoss(torch.autograd.Function):
        """The loss, with the gradient on the hidden states taken in ``forward``."""

        @staticmethod
        def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
            grad = torch.zeros_like(hidden) if with_grad else None
            total = torch.zeros((), dtype=torch.float32, device=hidden.device)
            entropies, losses = [], []
            correct = torch.zeros((), dtype=torch.long, device=hidden.device)
            for start in range(0, hidden.shape[1], chunk):
                stop = start + chunk
                piece = hidden[:, start:stop].detach()
                labels = target[:, start:stop]
                if with_grad:
                    piece.requires_grad_(True)
                    with torch.enable_grad():
                        part, entropy, token_loss, hits = _chunk_terms(
                            head, piece, labels, ignore_index
                        )
                        # Scaled BEFORE the gradient is taken, so each token's
                        # gradient is rounded to the head's dtype at the scale
                        # the unchunked step rounds it at. Scaling the summed
                        # gradient afterwards moved bf16 gradients by up to
                        # 9e-3 of the largest one.
                        (piece_grad,) = torch.autograd.grad(part / denominator, piece)
                    grad[:, start:stop] = piece_grad
                else:
                    part, entropy, token_loss, hits = _chunk_terms(
                        head, piece, labels, ignore_index
                    )
                total = total + part.detach()
                entropies.append(entropy)
                losses.append(token_loss)
                correct = correct + hits
            token_entropy = torch.cat(entropies, dim=1)
            token_loss = torch.cat(losses, dim=1)
            ctx.with_grad = with_grad
            if with_grad:
                ctx.save_for_backward(grad)
            ctx.mark_non_differentiable(token_entropy, token_loss, correct)
            return total / denominator, token_entropy, token_loss, correct

        @staticmethod
        def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
            if not ctx.with_grad:  # pragma: no cover — nothing asked for a gradient
                return (None,) * 7
            (grad,) = ctx.saved_tensors
            return (grad * grad_loss, None, None, None, None, None, None)

    return _ChunkedHeadLoss


def chunked_causal_lm_loss(
    hidden: Any,
    head: Callable[[Any], Any],
    labels: Any,
    chunk: int,
    *,
    num_items_in_batch: Any = None,
    shift_labels: Any = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Any, ChunkedStep]:
    """``ForCausalLMLoss(head(hidden), labels)`` without the full logits.

    ``head`` maps ``[batch, n, hidden]`` to ``[batch, n, vocab]`` and must be
    frozen: the gradient is taken on the hidden states only. ``labels``,
    ``shift_labels`` and ``num_items_in_batch`` mean what they mean to
    ``ForCausalLMLoss``: the mean over the labelled positions, or the sum over
    ``num_items_in_batch`` when the trainer passes one.
    """
    import torch

    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
    target = shift_labels if shift_labels is not None else shift_for_next_token(
        labels, ignore_index
    )
    target = target.to(hidden.device)
    if target.shape != hidden.shape[:2]:
        raise ValueError(
            f"labels of shape {tuple(target.shape)} do not match hidden states of "
            f"shape {tuple(hidden.shape)}"
        )
    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denominator = labelled
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_entropy, token_loss, correct = _chunked_function().apply(
        hidden, head, target, chunk, denominator, ignore_index, with_grad
    )
    step = ChunkedStep(
        token_entropy=token_entropy, token_loss=token_loss, correct=correct, labelled=labelled
    )
    return loss, step


class ChunkedHeadLoss:
    """A causal LM whose labelled forward runs its head and loss in chunks.

    The head wrapper of a streamed model never sees the labels, and a tied
    checkpoint has no streamed head at all, so the hook is on the causal-LM
    forward: with labels, the decoder runs as it always does, the head is held
    back, and the loss is built from the hidden states the head would have
    received. The output then carries the loss and no logits. A forward without
    labels (generation, ``predict``) is untouched.
    """

    def __init__(self, causal_lm: Any, chunk: int) -> None:
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
        head = causal_lm.get_output_embeddings()
        if head is None:
            raise ValueError(
                "training.stream_loss_chunk needs a model with an output head "
                f"(get_output_embeddings); {type(causal_lm).__name__} has none."
            )
        cap = getattr(getattr(causal_lm, "config", None), "final_logit_softcapping", None)
        if cap is not None:
            # Known before the first step; `_chunked_forward` checks any model.
            raise ValueError(
                "training.stream_loss_chunk cannot be used with "
                f"{type(causal_lm).__name__}: it soft-caps its logits after the "
                f"output head (final_logit_softcapping={cap}), so the loss cannot be "
                "rebuilt chunk by chunk from the head alone. Remove "
                "training.stream_loss_chunk."
            )
        trainable = [name for name, param in head.named_parameters() if param.requires_grad]
        if trainable:
            raise ValueError(
                "training.stream_loss_chunk takes the gradient on the hidden states "
                "only, so the output head must be frozen; trainable: "
                f"{', '.join(sorted(trainable))}."
            )
        self.chunk = chunk
        self.last_step: Optional[ChunkedStep] = None
        self._causal_lm = causal_lm
        self._head = head
        self._head_forward = head.forward
        self._forward = causal_lm.forward
        self._held: Optional[tuple[Any, Any]] = None
        self._holding = False
        # A forward already set on the instance (accelerate's hooks do that)
        # is put back as it was found; a class's own is uncovered again.
        self._before = [(module, module.__dict__.get("forward")) for module in (head, causal_lm)]
        head.forward = self._head_or_hold
        causal_lm.forward = self._chunked_forward

    def remove(self) -> None:
        """Put both forwards back."""
        for module, forward in self._before:
            if forward is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = forward

    def take_step(self) -> Optional[ChunkedStep]:
        """The metrics of the last labelled forward, once."""
        step, self.last_step = self.last_step, None
        return step

    def _head_or_hold(self, hidden: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._holding:
            return self._head_forward(hidden, *args, **kwargs)
        # Stands for the logits until the forward returns: no vocabulary axis,
        # so it costs nothing, and its identity says whether the model did
        # anything to "the logits" after the head.
        placeholder = hidden.new_empty((*hidden.shape[:-1], 0))
        self._held = (hidden, placeholder)
        return placeholder

    def _chunked_forward(self, *args: Any, **kwargs: Any) -> Any:
        labels = kwargs.pop("labels", None)
        if labels is None:
            return self._forward(*args, **kwargs)
        self._holding, self._held = True, None
        try:
            outputs = self._forward(*args, **kwargs)
        finally:
            self._holding = False
        held, self._held = self._held, None
        if held is None or getattr(outputs, "logits", None) is not held[1]:
            raise ValueError(
                "training.stream_loss_chunk cannot be used with "
                f"{type(self._causal_lm).__name__}: its forward does not hand the "
                "output head's result back unchanged (for example a final logit "
                "soft-cap), so the loss cannot be rebuilt chunk by chunk. Remove "
                "training.stream_loss_chunk."
            )
        loss, self.last_step = chunked_causal_lm_loss(
            held[0],
            self._head_forward,
            labels,
            self.chunk,
            num_items_in_batch=kwargs.get("num_items_in_batch"),
            shift_labels=kwargs.get("shift_labels"),
        )
        rest = {key: value for key, value in outputs.items() if key not in ("loss", "logits")}
        return type(outputs)(loss=loss, **rest)


def causal_lm_of(model: Any) -> Any:
    """The causal LM under a PEFT wrapper, or ``model`` itself."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


# The trainer holds the hook under this name, and leaves the last labelled
# forward's `ChunkedStep` under the other for the rewind recorder above it.
HOOK_ATTR = "_soup_chunked_head_loss"
STEP_ATTR = "_soup_chunked_step"


def make_chunked_loss_trainer_class(base_cls: type) -> type:
    """``base_cls`` (TRL's ``SFTTrainer``) for a model that returns no logits.

    ``SFTTrainer.compute_loss`` computes ``entropy`` and
    ``mean_token_accuracy`` from ``outputs.logits`` and raises without them.
    This goes to transformers' own ``compute_loss`` instead and logs the same
    columns, by TRL's formulas, from what the chunk loop collected. A step that
    is not chunked (no hook, or no labels) takes ``base_cls``'s path untouched.
    """
    from transformers import Trainer

    class ChunkedLossTrainer(base_cls):  # type: ignore[misc, valid-type]
        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            hook = getattr(self, HOOK_ATTR, None)
            if hook is None or "labels" not in inputs:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            mode = "train" if self.model.training else "eval"
            # As TRL's `compute_loss` does before it delegates.
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            loss, outputs = Trainer.compute_loss(
                self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            step = hook.take_step()
            setattr(self, STEP_ATTR, step)
            if step is not None:
                _log_trl_metrics(self, mode, inputs, step)
            return (loss, outputs) if return_outputs else loss

    ChunkedLossTrainer.__name__ = f"ChunkedLoss{base_cls.__name__}"
    ChunkedLossTrainer.__qualname__ = ChunkedLossTrainer.__name__
    return ChunkedLossTrainer


def _log_trl_metrics(trainer: Any, mode: str, inputs: Any, step: ChunkedStep) -> None:
    """``entropy``, ``num_tokens`` and ``mean_token_accuracy`` as TRL logs them.

    A trl without the ``_metrics`` store logs none of the three; the run does
    not depend on them.
    """
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
        tokens = gather(attention_mask.sum()).sum().item()
    else:
        entropy = step.token_entropy.mean()
        tokens = None
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train" and tokens is not None and hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)
