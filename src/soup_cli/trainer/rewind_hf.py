"""HF/TRL side of the rewind flight recorder (see ``monitoring/rewind_log.py``).

Row identity comes from the train sampler (main process, consumption order); per-row
loss from ``compute_loss``, reduced from the logits TRL already computed. One
micro-batch = one record. Recorder failures never raise into training: every
unsupported configuration found so far (a resumed run, packed / padding-free
batches, missing logits, context-parallel ``shift_labels``) either disables the
recorder with one warning or falls through to the base trainer untouched.

Why the sampler and not a callback or a row-id column (all measured on
transformers 5.16.1 / trl 0.29.1, 2026-09-15):

* Stock HF ``Trainer`` does not pass ``inputs`` to ``on_step_end`` -- a
  :class:`~transformers.TrainerCallback` cannot see which rows were in a batch.
* An extra id column does not survive TRL's collator: ``compute_loss`` receives
  exactly ``input_ids`` / ``attention_mask`` / ``labels``.
* ``Trainer._get_train_sampler`` runs in the MAIN process and yields indices in
  consumption order even with ``dataloader_num_workers > 0``.

The DataLoader prefetches ahead of ``compute_loss``, so recorded indices are queued
FIFO and popped ``len(labels)`` at a time when the loss for that micro-batch runs.

``row_loss`` is plain token cross-entropy. Under ``loss_type="dft"``, label
smoothing, or a custom ``compute_loss_func`` it is NOT the loss that produced the
gradient -- it is still a consistent per-row difficulty signal, but it will not sum
to the logged training loss.

Usage from a trainer wrapper::

    from soup_cli.trainer.rewind_hf import (
        attach_rewind_state, make_rewind_trainer_class,
    )
    if tcfg.rewind_log:
        TrainerCls = make_rewind_trainer_class(SFTTrainer)
        trainer = TrainerCls(**trainer_kwargs)
        state = attach_rewind_state(trainer, log)
        trainer.train()
        if (note := state.summary()) is not None:
            console.print(note)

``attach_rewind_state`` must be called BEFORE ``train()`` -- the sampler is built
when the dataloader is built.
"""

from __future__ import annotations

import functools
from collections import deque
from typing import Any, Optional, Protocol, Sequence

from rich.console import Console
from rich.markup import escape

console = Console()

# Hidden attr on the trainer instance -- read by the overrides below.
_STATE_ATTR = "_soup_rewind_state"


class RewindSink(Protocol):
    """What the recorder writes to. ``monitoring.rewind_log.RewindLog`` satisfies it.

    Declared as a Protocol so this module never imports the log module: the two
    halves of the feature stay independently testable (tests inject a
    list-collecting fake).
    """

    def record_batch(
        self,
        *,
        step: int,
        micro: int,
        rows: Sequence[int],
        row_loss: Sequence[float],
        row_tokens: Sequence[int],
    ) -> None:
        ...


class RewindState:
    """FIFO of sampler indices plus step/micro bookkeeping.

    Pure Python -- no torch -- so the bookkeeping is unit-testable without a
    trainer, and so importing this module stays light.

    ``micro`` is derived from the step changing, not from the configured
    gradient-accumulation count (the log header records that, via ``RewindLog``).
    """

    def __init__(self, sink: RewindSink) -> None:
        self._sink = sink
        self._pending: deque[int] = deque()
        self._last_step: Optional[int] = None
        self._micro = -1
        self.failed = False
        self.dropped = 0

    def on_index(self, i: int) -> None:
        """Called by the sampler wrapper for every index it yields."""
        if self.failed:
            # A disabled recorder must not keep growing a FIFO nobody pops.
            return
        self._pending.append(int(i))

    def on_micro_batch(
        self,
        *,
        step: int,
        row_loss: Sequence[float],
        row_tokens: Sequence[int],
    ) -> None:
        """Pop ``len(row_loss)`` ids (oldest first) and write one batch record.

        ``micro`` restarts at 0 whenever ``step`` differs from the last step seen,
        otherwise it increments. If fewer ids are pending than there are rows the
        batch is dropped and counted -- that means the sampler was bypassed (a
        custom ``get_train_dataloader``, say) and guessing would put wrong row ids
        in the log, which is worse than a gap. A dropped micro-batch does not
        advance ``micro``, so the remaining micro-batches of that step are
        renumbered one lower than their true position.
        """
        n = len(row_loss)
        if len(self._pending) < n:
            self.dropped += 1
            return
        rows = [self._pending.popleft() for _ in range(n)]
        if step != self._last_step:
            self._micro = 0
            self._last_step = step
        else:
            self._micro += 1
        self._sink.record_batch(
            step=int(step),
            micro=self._micro,
            rows=rows,
            row_loss=[float(x) for x in row_loss],
            row_tokens=[int(t) for t in row_tokens],
        )

    def disable(self, reason: str) -> None:
        """Disable the recorder after one warning line. Later calls are silent."""
        if self.failed:
            return
        self.failed = True
        self._pending.clear()
        try:
            # escape(): the reason is arbitrary text (an exception message, a
            # path like "[/tmp/x]") and unescaped markup raises MarkupError.
            console.print(f"[yellow]Rewind recorder disabled:[/] {escape(reason)}")
        except Exception:  # a broken console must not stop training
            pass

    def fail(self, exc: BaseException) -> None:
        """Disable the recorder because ``exc`` was raised while recording."""
        self.disable(str(exc))

    def summary(self) -> Optional[str]:
        """One line describing dropped micro-batches, or ``None`` if there were none."""
        if self.dropped <= 0:
            return None
        noun = "micro-batch" if self.dropped == 1 else "micro-batches"
        return f"rewind: {self.dropped} {noun} dropped (FIFO underflow)"


def row_losses(logits: Any, labels: Any) -> tuple[list[float], list[int]]:
    """Per-row mean supervised-token cross-entropy, and the token counts.

    ``logits`` is ``(B, T, V)``, ``labels`` is ``(B, T)`` with ``-100`` at
    unsupervised positions. Causal-LM alignment: ``logits[:, :-1]`` predicts
    ``labels[:, 1:]``.

    Reuses the logits the forward pass already produced (no second forward),
    under ``no_grad`` so the recorder adds nothing to the autograd graph. The
    token cross-entropy is computed in the logits' dtype one row at a time, so
    no upcast copy of the ``(B, T, V)`` tensor is ever made; only the ``(T-1,)``
    per-token result is cast, and the per-row mean is accumulated in float32. A
    row with no supervised tokens yields ``(0.0, 0)``.

    This is plain token CE: see the module docstring for when it differs from
    the loss that produced the gradient.
    """
    import torch
    import torch.nn.functional as f

    row_loss: list[float] = []
    row_tokens: list[int] = []
    with torch.no_grad():
        shift_labels = labels[:, 1:]
        for i in range(shift_labels.size(0)):
            targets = shift_labels[i]
            # reduction="none" + ignore_index keeps ignored positions at exactly
            # 0.0; the explicit mask below makes that independent of that
            # guarantee.
            token_ce = f.cross_entropy(
                logits[i, :-1, :].detach(),  # a view: no copy of the logits
                targets,
                reduction="none",
                ignore_index=-100,
            ).float()
            mask = (targets != -100).to(token_ce.dtype)
            tokens = mask.sum()
            total = (token_ce * mask).sum()
            row_loss.append(float(total / tokens.clamp(min=1.0)))
            row_tokens.append(int(tokens.item()))
    return row_loss, row_tokens


@functools.lru_cache(maxsize=None)
def make_rewind_trainer_class(base_cls: type) -> type:
    """Return a subclass of ``base_cls`` that records rows and per-row losses.

    Cached via ``functools.lru_cache`` so two calls with the same ``base_cls``
    return the SAME subclass -- keeps ``isinstance`` checks consistent across
    sweep runs and avoids confusing pickle (mirrors
    ``utils/multipack_trainer.make_multipack_trainer_class``).

    Every override degrades to the base implementation when no state has been
    attached, so the subclass is safe to instantiate even when the recorder is
    later disabled.
    """
    from torch.utils.data import Sampler

    class IndexRecordingSampler(Sampler):
        """Wrap any ``Sampler[int]``; report every index before yielding it.

        Defined inside the factory because ``torch`` must not be imported at
        module scope (``tests/test_cli_startup_is_light.py``).
        """

        def __init__(self, inner: Any, state: RewindState) -> None:
            self._inner = inner
            self._state = state

        def __iter__(self):
            for index in self._inner:
                self._state.on_index(index)
                yield index

        def __len__(self) -> int:
            return len(self._inner)

        def __getattr__(self, name: str) -> Any:
            # The dataloader probes the sampler: ``set_epoch`` (reached via
            # ``hasattr(batch_sampler.sampler, "set_epoch")``) reseeds the
            # shuffle, and swallowing it would change the data order. Reached
            # only when normal lookup fails; reading ``_inner`` through
            # __getattribute__ keeps a half-built instance from recursing here.
            try:
                inner = object.__getattribute__(self, "_inner")
            except AttributeError:
                raise AttributeError(name) from None
            return getattr(inner, name)

    class RewindTrainer(base_cls):  # type: ignore[misc, valid-type]
        soup_rewind: bool = True
        # Class-level default: an instance that never went through
        # attach_rewind_state behaves exactly like the base trainer.
        _soup_rewind_state: Optional[RewindState] = None

        def train(  # type: ignore[override]
            self, resume_from_checkpoint: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            state = getattr(self, _STATE_ATTR, None)
            # On resume, HF wraps the dataloader in accelerate's
            # ``skip_first_batches``, whose ``SkipBatchSampler`` ENUMERATES the
            # wrapped sampler: every skipped row reaches ``on_index`` while
            # ``compute_loss`` never runs for it, so every logged id would be
            # wrong and nothing would be counted as dropped.
            if (
                state is not None
                and resume_from_checkpoint
                and not getattr(self.args, "ignore_data_skip", False)
            ):
                state.disable(
                    "resumed runs skip batches the recorder cannot see; "
                    "start a fresh run to record"
                )
            return super().train(resume_from_checkpoint, *args, **kwargs)

        def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            # ``*args/**kwargs`` accept the HF >=4.41 signature, where
            # ``train_dataset`` is passed positionally.
            sampler = super()._get_train_sampler(*args, **kwargs)
            state = getattr(self, _STATE_ATTR, None)
            if state is None or sampler is None:
                return sampler
            return IndexRecordingSampler(sampler, state)

        def compute_loss(  # type: ignore[override]
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            state = getattr(self, _STATE_ATTR, None)
            # ``compute_loss`` is also called from ``prediction_step`` during
            # evaluation; those rows never came from the train sampler, so
            # recording them would desynchronise the FIFO. Guard on the model's
            # own mode, which the eval loop flips. ``shift_labels`` means context
            # parallelism sharded the sequence, so the causal shift below would
            # misalign logits and labels.
            if (
                state is None
                or state.failed
                or not getattr(model, "training", False)
                or "labels" not in inputs
                or "shift_labels" in inputs
            ):
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            # Read the labels BEFORE delegating: HF's Trainer pops "labels" from
            # ``inputs`` when a label smoother or a custom compute_loss_func is
            # configured.
            labels = inputs["labels"]
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            try:
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise RuntimeError(
                        "the model returned no logits (use_liger_kernel skips them "
                        "during training), so per-row losses cannot be computed"
                    )
                row_loss, row_tokens = row_losses(logits, labels)
                # ``state.global_step`` is the number of COMPLETED optimizer
                # steps, so +1 is the 1-based step this micro-batch belongs to.
                state.on_micro_batch(
                    step=int(self.state.global_step) + 1,
                    row_loss=row_loss,
                    row_tokens=row_tokens,
                )
            except Exception as exc:  # never raise into training
                state.fail(exc)
            return (loss, outputs) if return_outputs else loss

    RewindTrainer.__name__ = f"Rewind{base_cls.__name__}"
    RewindTrainer.__qualname__ = RewindTrainer.__name__
    return RewindTrainer


def attach_rewind_state(trainer: Any, sink: RewindSink) -> RewindState:
    """Create a :class:`RewindState`, stash it on ``trainer``, and return it.

    Must be called before ``trainer.train()``: the sampler is wrapped when the
    train dataloader is built, which happens inside ``train()``.

    A trainer that flattens micro-batches (TRL's ``padding_free``, which
    ``packing=True`` turns on under the default ``packing_strategy="bfd"``) gets
    a state that is already disabled and is left unwrapped: one id would be
    popped per packed micro-batch and the rest would leak forever.
    """
    state = RewindState(sink)
    if getattr(trainer, "padding_free", False):
        state.disable(
            "packing / padding_free flattens each micro-batch into one sequence, "
            "so rows cannot be told apart; set packing=False and padding_free=False "
            "to record"
        )
        setattr(trainer, _STATE_ATTR, None)
        return state
    setattr(trainer, _STATE_ATTR, state)
    return state
