"""Trainer hook for data-parallel layer streaming: average the adapters' gradients (C3, C4).

A streamed model keeps its frozen base on ``meta`` and injects it per call, so
accelerate cannot wrap it in ``DistributedDataParallel`` (torch rejects a module whose
parameters sit on two device types), and on CPU accelerate never wraps at all. Soup
therefore owns the one collective data parallelism needs, an average of the LoRA
gradients across ranks (:class:`~soup_cli.utils.adapter_grad_sync.AdapterGradSync`),
and two things around it live here:

* :func:`make_adapter_sync_trainer_class` (C3) builds a trainer class whose
  ``training_step`` runs the sync on the last micro-batch of an accumulation window.
  The placement is the point. In transformers' inner loop the order is
  ``training_step`` -> ``_clip_grad_norm`` -> ``on_pre_optimizer_step`` callbacks ->
  ``optimizer.step()``, so a callback fires after clipping and would give
  ``mean(clip(g_rank))`` instead of ``clip(mean(g_rank))``. The override returns before
  HF clips. ``accelerator.sync_gradients`` is HF's own "last micro-batch" flag: HF sets
  it from the same condition that decides to clip and step.
* :func:`mark_accelerate_prepared` (C4) sets ``_is_accelerate_prepared``, which makes
  ``Accelerator.prepare_model`` return the model untouched, before ``.to(device)`` and
  before the DDP wrap. A CPU run cannot show what the flag prevents (``MULTI_CPU`` is
  never wrapped), so its capability is pinned in ``tests/test_stream_adapter_sync.py``.

The loss scale that makes a plain MEAN the right average is HF's. With
``average_tokens_across_devices`` (the default) and a model whose forward takes loss
kwargs, HF gathers ``num_items_in_batch`` over the ranks, divides each rank's summed
token loss by that global count and multiplies it by the number of processes, so the
mean over ranks of the per-rank gradients is the gradient of the global per-token mean
(the single-process gradient of the union batch). Averaging per-rank means instead, or
summing, would be wrong whenever the ranks hold different token counts.

Torch and transformers are imported inside the functions: ``soup`` stays importable
without them.
"""

from __future__ import annotations

from typing import Any

#: The attribute ``Accelerator.prepare_model`` reads to return a model unchanged.
ACCELERATE_PREPARED_FLAG = "_is_accelerate_prepared"

#: Class attribute that marks a trainer built by :func:`make_adapter_sync_trainer_class`.
ADAPTER_SYNC_MARKER = "_soup_adapter_grad_sync_trainer"


def mark_accelerate_prepared(model: Any) -> None:
    """Make ``Accelerator.prepare_model`` skip ``model``: no ``.to()``, no DDP wrap (C4)."""
    setattr(model, ACCELERATE_PREPARED_FLAG, True)


def is_accelerate_prepared(model: Any) -> bool:
    """True when :func:`mark_accelerate_prepared` was applied to ``model``."""
    return bool(getattr(model, ACCELERATE_PREPARED_FLAG, False))


def has_adapter_sync(trainer: Any) -> bool:
    """True when ``trainer`` comes from :func:`make_adapter_sync_trainer_class`."""
    return bool(getattr(trainer, ADAPTER_SYNC_MARKER, False))


def assert_adapter_sync_installed(trainer: Any) -> None:
    """Raise unless ``trainer`` averages the adapter gradients.

    A distributed streaming run whose trainer was built from another class (a branch
    that does not carry the sync) would train one unsynchronised copy per rank without
    an error, so the setup checks this once the trainer exists.
    """
    if not has_adapter_sync(trainer):
        raise RuntimeError(
            "distributed layer streaming needs the adapter gradient sync, but the trainer "
            f"built for this run ({type(trainer).__name__}) does not carry it; each rank "
            "would train its own copy without any error. This is a Soup bug: refusing to "
            "start."
        )


def _refuse_ddp_wrapped(model: Any) -> None:
    """The streamed model must reach ``training_step`` unwrapped (the flag of C4)."""
    import torch

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise RuntimeError(
            "accelerate wrapped the streamed model in DistributedDataParallel, which would "
            "average the gradients a second time (and cannot hold the streamed base). The "
            f"'{ACCELERATE_PREPARED_FLAG}' flag that makes accelerate skip the model is "
            "missing on it; refusing to train."
        )


def make_adapter_sync_trainer_class(base_cls: type) -> type:
    """``base_cls`` with the adapter-gradient average between backward and clipping.

    ``training_step`` runs ``base_cls``'s step (forward, loss scaling, backward), then,
    when ``accelerator.sync_gradients`` says this was the last micro-batch of the
    accumulation window, averages the trainable parameters' gradients across ranks. HF
    clips and steps after that. The :class:`AdapterGradSync` is built once, on the first
    synchronising step, from the model's trainable parameters.

    ``last_sync_stats`` is the :class:`SyncStats` of the latest sync (``None`` before the
    first), ``sync_calls`` counts the syncs. Both are meant for per-step metrics.
    Without a process group, or in a world of one, the sync is a no-op that reports zeros.
    """

    class AdapterSyncTrainer(base_cls):  # type: ignore[misc, valid-type]
        _soup_adapter_grad_sync_trainer = True
        last_sync_stats = None
        sync_calls = 0
        _adapter_grad_sync = None

        def _sync_adapter_grads(self, model: Any) -> None:
            if self._adapter_grad_sync is None:
                from soup_cli.utils.adapter_grad_sync import AdapterGradSync

                _refuse_ddp_wrapped(model)
                self._adapter_grad_sync = AdapterGradSync(model)
            self.last_sync_stats = self._adapter_grad_sync.sync()
            self.sync_calls += 1

        def training_step(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            loss = super().training_step(*args, **kwargs)
            if self.accelerator.sync_gradients:
                model = args[0] if args else kwargs["model"]
                self._sync_adapter_grads(model)
            return loss

    AdapterSyncTrainer.__name__ = f"AdapterSync{base_cls.__name__}"
    AdapterSyncTrainer.__qualname__ = AdapterSyncTrainer.__name__
    return AdapterSyncTrainer
