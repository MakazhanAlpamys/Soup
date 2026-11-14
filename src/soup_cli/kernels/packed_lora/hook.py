# SPDX-License-Identifier: Apache-2.0
#
# Written for Soup. It replaces the monkey-patch of ``Qwen2ForCausalLM.forward`` that the
# Soup Kernel Challenge submission of team Luck used by an instance-level hook that is installed
# after ``get_peft_model`` and removed when training ends, following
# ``utils/stream_loss_chunk.py::ChunkedHeadLoss``.
"""The packed forward on one causal LM, for the labelled training step only.

``PackedLoraHook`` puts a ``forward`` on the *instance* of the causal LM. A labelled forward in
training mode with gradients on (the trainer's step) runs the packed layers and the logit-free
head and loss of a :class:`~soup_cli.kernels.packed_lora.runtime.PackedRun`; its output carries
the loss and no logits. Every other call (evaluation, generation, a batch the kernels do not
take) runs the stock forward, with the host-parked tables put back on the GPU first. Nothing is
patched on a class and nothing is global, so ``remove`` puts the model back as it was and two
runs in one process do not see each other.

The hook leaves each labelled forward's metrics in a
:class:`~soup_cli.utils.stream_loss_chunk.ChunkedStep` for :meth:`take_step`, which is what
``make_chunked_loss_trainer_class`` reads to log TRL's ``entropy`` / ``mean_token_accuracy``
columns and what the rewind recorder reads for the per-row losses.

Without a run (``run=None``) the hook only does the part of the stock forward that the kernels'
set-up changed: ``keep_tables_in_compute_dtype`` leaves the embedding in the compute dtype, so
the stock forward is fed ``inputs_embeds`` upcast to fp32, which is what it would have computed
from the fp32 table of an ordinary ``prepare_model_for_kbit_training``.

A step is *stock* when the kernels cannot take it; the run counts those and says why (once).
"""

from __future__ import annotations

from typing import Any, Optional

#: The causal LM carries this when ``keep_tables_in_compute_dtype`` or ``model.prepare`` left its
#: embedding in the compute dtype (see ``model.py``).
FP32_EMBEDS_ATTR = "_packed_lora_fp32_embeds"


def _autocast_dtype():
    """The dtype of the CUDA autocast in effect, or ``None`` without one."""
    import torch

    if not torch.is_autocast_enabled("cuda"):
        return None
    return torch.get_autocast_dtype("cuda")


def _embedded_as_stock(causal_lm: Any, input_ids: Any) -> Any:
    """The embedding output of the stock path of a model whose table is in the compute dtype: the
    stock forward keeps an fp32 residual stream only from an fp32 embedding."""
    return causal_lm.get_input_embeddings()(input_ids).float()


def step_from_logits(logits: Any, labels: Any):
    """A :class:`~soup_cli.utils.stream_loss_chunk.ChunkedStep` from the logits of a stock
    forward, with TRL's formulas (``entropy_from_logits``, the accuracy of the shifted labels).
    ``token_loss`` is each labelled position's cross-entropy, read in the logits' dtype and then
    cast, as the rewind recorder's ``row_losses`` does."""
    import torch
    from torch.nn import functional

    from soup_cli.utils.stream_loss_chunk import ChunkedStep, shift_for_next_token

    with torch.no_grad():
        target = shift_for_next_token(labels).to(logits.device)
        labelled = target != -100
        log_probs = functional.log_softmax(logits, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(-1)
        correct = (logits.argmax(dim=-1) == target).sum()  # -100 is never an argmax
        at_label = log_probs.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_loss = (-at_label).float() * labelled
        return ChunkedStep(token_entropy=entropy, token_loss=token_loss, correct=correct,
                           labelled=labelled.sum())


class PackedLoraHook:
    """``causal_lm.forward`` on the packed kernels for the trainer's labelled step.

    ``run``: the :class:`~soup_cli.kernels.packed_lora.runtime.PackedRun` (prepared, with its
    tables not yet parked), or ``None`` for a hook that only takes care of the embedding of a
    model the kernels did not take."""

    def __init__(self, causal_lm: Any, run: Any = None) -> None:
        self._causal_lm = causal_lm
        self._run = run
        self._stock = causal_lm.forward
        # A forward already set on the instance (accelerate's hooks do that) is put back as it
        # was found; a class's own is uncovered again.
        self._before = causal_lm.__dict__.get("forward")
        self._parked = False  # the tables are where the plan keeps them (first step: not yet)
        self.last_step: Optional[Any] = None
        self.installed = True
        causal_lm.forward = self._forward

    # -- the trainer's side ---------------------------------------------------------------------
    def take_step(self) -> Optional[Any]:
        """The metrics of the last labelled forward, once."""
        step, self.last_step = self.last_step, None
        return step

    def remove(self) -> None:
        """Put the forward back. The run is closed by whoever made it."""
        if not self.installed:
            return
        self.installed = False
        if self._before is None:
            self._causal_lm.__dict__.pop("forward", None)
        else:
            self._causal_lm.forward = self._before

    # -- the forward ----------------------------------------------------------------------------
    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        run = self._run
        if (run is not None and not run.closed and kwargs.get("labels") is not None
                and self._causal_lm.training and _grad_enabled()):
            reason = self._unusable(args, kwargs)
            if reason is None:
                outputs = self._packed(kwargs)
                if outputs is not None:
                    return outputs
                reason = "a batch that is not right-padded"
            run.note_stock(reason)
        return self._stock_forward(args, kwargs)

    def _unusable(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Why the kernels cannot take this call, or ``None`` if they can."""
        run = self._run
        if args:
            return "a forward called with positional arguments"
        for name in ("position_ids", "past_key_values", "inputs_embeds", "shift_labels"):
            if kwargs.get(name) is not None:
                return f"a batch with {name}"
        input_ids = kwargs.get("input_ids")
        if input_ids is None or kwargs.get("attention_mask") is None:
            return "a batch without input_ids and attention_mask"
        if kwargs.get("use_cache") or kwargs.get("output_attentions") or kwargs.get(
                "output_hidden_states"):
            return "a call that asks for the cache, the attentions or the hidden states"
        if kwargs.get("return_dict") is False:
            return "a call that asks for a tuple"
        if kwargs.get("logits_to_keep"):
            return "a call that keeps only some logits"
        if input_ids.device != run.device:
            return f"a batch on {input_ids.device}, not on {run.device}"
        dtype = _autocast_dtype()
        if dtype is None or dtype != run.dtype:
            return (f"autocast is {dtype}, not {run.dtype} (is the trainer in "
                    f"{str(run.dtype)[6:]} mode?)")
        return None

    def _packed(self, kwargs: dict) -> Optional[Any]:
        """The packed step, or ``None`` if the batch is not right-padded."""
        import torch
        from transformers.modeling_outputs import CausalLMOutputWithPast

        run = self._run
        if not self._parked:
            run.park_tables()  # the first step, and the first one after an evaluation
            self._parked = True
        attention_mask = kwargs["attention_mask"]
        result = run.forward(kwargs["input_ids"], attention_mask, kwargs["labels"],
                             kwargs.get("num_items_in_batch"))
        if result is None:
            return None
        loss, stats = result
        with torch.no_grad():
            # the kernels give one entropy per real token; the trainer wants [batch, seq]
            token_entropy = torch.zeros(attention_mask.shape, dtype=stats["entropy"].dtype,
                                        device=stats["entropy"].device)
            token_entropy[attention_mask.bool()] = stats["entropy"]
            from soup_cli.utils.stream_loss_chunk import ChunkedStep

            self.last_step = ChunkedStep(
                token_entropy=token_entropy, token_loss=stats["token_loss"],
                correct=stats["hit"].sum(), labelled=stats["labelled"])
        return CausalLMOutputWithPast(loss=loss, logits=None)

    def _stock_forward(self, args: tuple, kwargs: dict) -> Any:
        """The model's own forward, with the tables back on the GPU and the embedding as the
        stock model computes it."""
        run = self._run
        if run is not None and not run.closed:
            run.restore_tables()  # the stock forward reads them on the GPU
            self._parked = False
        causal_lm = self._causal_lm
        if (getattr(causal_lm, FP32_EMBEDS_ATTR, False) and not args
                and kwargs.get("input_ids") is not None and kwargs.get("inputs_embeds") is None):
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = _embedded_as_stock(causal_lm, kwargs.pop("input_ids"))
            kwargs["input_ids"] = None
        outputs = self._stock(*args, **kwargs)
        labels = kwargs.get("labels")
        logits = getattr(outputs, "logits", None)
        if labels is not None and logits is not None and logits.shape[:-1] == labels.shape:
            self.last_step = step_from_logits(logits, labels)
        return outputs


def _grad_enabled() -> bool:
    import torch

    return torch.is_grad_enabled()
