"""HF Trainer / TRL SFTTrainer subclass that swaps ``_get_train_sampler``.

The schema field ``training.multipack: bool`` shipped in v0.37.0 along
with the FFD bin-packing :class:`MultipackBatchSampler` and the
arch-allowlist validator. The live HF Trainer wiring was deliberately
deferred (mirrors the v0.27.0 MII / v0.38.0 quant-menu / v0.39.0 ReLoRA
stub-then-live pattern). v0.40.3 (#65) wires it up.

Usage from a trainer wrapper::

    from soup_cli.utils.multipack_trainer import (
        attach_multipack_state, lengths_from_dataset,
        make_multipack_trainer_class,
    )
    if tcfg.multipack:
        validate_multipack_architecture(arch)
        TrainerCls = make_multipack_trainer_class(SFTTrainer)
        trainer = TrainerCls(**trainer_kwargs)
        attach_multipack_state(
            trainer,
            lengths=lengths_from_dataset(train_ds),
            max_seq_len=cfg.data.max_length,
            batch_size=batch_size,
            seed=cfg.training.seed or 0,
        )
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Hidden attrs on the trainer instance — picked up by _get_train_sampler.
_LENGTHS_ATTR = "_soup_multipack_lengths"
_MAX_SEQ_ATTR = "_soup_multipack_max_seq_len"
_BATCH_SIZE_ATTR = "_soup_multipack_batch_size"
_SEED_ATTR = "_soup_multipack_seed"


def lengths_from_dataset(
    dataset: Any,
    *,
    key: str = "input_ids",
) -> list[int]:
    """Extract per-sample token lengths from a tokenized dataset.

    Falls back to a ``"length"`` column when ``input_ids`` is absent. A
    sample with neither contributes ``0`` (which the sampler will reject
    upstream — surfaces the misconfiguration loudly).
    """
    if dataset is None:
        return []
    lengths: list[int] = []
    try:
        iterator = iter(dataset)
    except TypeError:
        return []
    for row in iterator:
        if not isinstance(row, dict):
            lengths.append(0)
            continue
        value = row.get(key)
        if value is not None:
            try:
                lengths.append(len(value))
                continue
            except TypeError:
                pass
        fallback = row.get("length")
        if isinstance(fallback, int) and not isinstance(fallback, bool):
            lengths.append(fallback)
        else:
            lengths.append(0)

    # Surface the case where every length is 0 — the multipack sampler
    # would otherwise silently produce phantom batches and the trainer
    # would NaN-loss with no obvious cause. Loud-fail policy mirrors the
    # v0.37.0 multipack-arch allowlist.
    if lengths and not any(lengths):
        logger.warning(
            "lengths_from_dataset(key=%r) returned all zeros — dataset "
            "rows have neither '%s' nor 'length' keys. Multipack sampling "
            "will not work; check tokenisation pipeline.",
            key, key,
        )
    return lengths


def detect_arch_name(model: Any) -> Optional[str]:
    """Return the model's architecture class name, or ``None``.

    Probes ``model.config.architectures[0]`` first (HF convention), then
    falls back to ``type(model).__name__``.
    """
    if model is None:
        return None
    config = getattr(model, "config", None)
    if config is not None:
        archs = getattr(config, "architectures", None)
        if archs:
            try:
                first = archs[0]
            except (IndexError, TypeError):
                first = None
            if isinstance(first, str) and first:
                return first
    cls = type(model).__name__
    return cls or None


@functools.lru_cache(maxsize=None)
def make_multipack_trainer_class(base_cls: type) -> type:
    """Return a subclass of ``base_cls`` overriding ``_get_train_sampler``.

    The override returns a :class:`MultipackBatchSampler` built from
    instance attrs set by :func:`attach_multipack_state`. If state is
    missing the override delegates to the base implementation — so the
    subclass is safe to instantiate even when multipack is later disabled.

    Cached via ``functools.lru_cache`` so two calls with the same
    ``base_cls`` return the SAME subclass — this keeps ``isinstance``
    checks consistent across sweep runs and avoids confusing pickle.

    .. note::
       v0.40.4 (#65) wires this factory into the SFT / Pretrain trainer
       wrappers via a ``get_train_dataloader`` override. HF Trainer's
       ``_get_train_sampler`` returns a ``Sampler[int]`` which the
       DataLoader then consumes as scalar indices, while
       :class:`MultipackBatchSampler` yields ``list[list[int]]``. The
       solution is to bypass ``_get_train_sampler`` for the live path and
       install the multipack sampler as the DataLoader's
       ``batch_sampler=``. The ``_get_train_sampler`` override stays as a
       defensive no-op fallback (delegates to super) so the subclass
       remains safe to instantiate even when state was never attached.
    """
    from soup_cli.utils.multipack_sampler import MultipackBatchSampler

    class MultipackTrainer(base_cls):  # type: ignore[misc, valid-type]
        soup_multipack: bool = True

        def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            # v0.40.4 #65 — the live multipack path goes through
            # ``get_train_dataloader`` (below), which installs the
            # multipack ``MultipackBatchSampler`` directly as the
            # DataLoader's ``batch_sampler=``. This override stays as a
            # defensive no-op fallback that delegates to the base
            # implementation: HF Trainer's DataLoader iterates a
            # ``Sampler[int]`` (scalar indices) — returning a multipack
            # ``list[list[int]]`` here would be a shape mismatch if any
            # eval / prediction loop ever bypasses
            # ``get_train_dataloader`` and calls this directly.
            # ``*args/**kwargs`` accept the HF >=4.41 signature
            # (``train_dataset`` passed positionally).
            return super()._get_train_sampler(*args, **kwargs)

        def get_train_dataloader(self):  # type: ignore[override]
            """Return a DataLoader whose batch_sampler is multipack-aware.

            v0.40.4 #65 — when multipack state has been attached via
            :func:`attach_multipack_state`, build a flat-yield
            ``MultipackBatchSampler`` (``real_batches=False`` — yields
            ``list[int]`` per packed sequence, which is the contract HF
            ``DataLoader.batch_sampler`` expects). When state is missing,
            delegate to ``super().get_train_dataloader()`` so the subclass
            is safe to instantiate even when multipack is disabled at
            runtime (matches the ``_get_train_sampler`` fallback policy).
            """
            lengths = getattr(self, _LENGTHS_ATTR, None)
            max_seq = getattr(self, _MAX_SEQ_ATTR, None)
            batch_size = getattr(self, _BATCH_SIZE_ATTR, None)
            seed = getattr(self, _SEED_ATTR, 0)
            # `attach_multipack_state` rejects non-positive values up front;
            # the explicit None check below preserves the "state never
            # attached" path while letting an empty list (lengths=[]) fall
            # through to the same fallback (the sampler would reject it).
            if (
                lengths is None or not lengths
                or max_seq is None or batch_size is None
            ):
                return super().get_train_dataloader()

            from torch.utils.data import DataLoader

            train_dataset = getattr(self, "train_dataset", None)
            if train_dataset is None:
                # Defensive — HF Trainer always sets this before train()
                # invokes get_train_dataloader, but a user calling the
                # method directly might trip the unset case.
                return super().get_train_dataloader()

            args = getattr(self, "args", None)
            drop_last = False
            num_workers = 0
            pin_memory = False
            if args is not None:
                drop_last = bool(getattr(args, "dataloader_drop_last", False))
                num_workers = getattr(args, "dataloader_num_workers", 0) or 0
                pin_memory = bool(getattr(args, "dataloader_pin_memory", False))

            sampler = MultipackBatchSampler(
                lengths=list(lengths),
                batch_max_len=int(max_seq),
                batch_size=int(batch_size),
                real_batches=False,  # yield list[int] per pack — DataLoader-compatible
                seed=int(seed),
                drop_last=drop_last,
            )

            data_collator = getattr(self, "data_collator", None)
            return DataLoader(
                train_dataset,
                batch_sampler=sampler,
                collate_fn=data_collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    MultipackTrainer.__name__ = f"Multipack{base_cls.__name__}"
    MultipackTrainer.__qualname__ = MultipackTrainer.__name__
    return MultipackTrainer


def attach_multipack_state(
    trainer: Any,
    *,
    lengths: Sequence[int],
    max_seq_len: int,
    batch_size: int,
    seed: int = 0,
) -> None:
    """Stash the multipack sampler config on ``trainer``."""
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("max_seq_len must be int")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    lengths_list = list(lengths)
    if not lengths_list:
        raise ValueError(
            "lengths must not be empty — empty dataset cannot drive "
            "MultipackBatchSampler. Check the tokenisation pipeline."
        )
    setattr(trainer, _LENGTHS_ATTR, lengths_list)
    setattr(trainer, _MAX_SEQ_ATTR, max_seq_len)
    setattr(trainer, _BATCH_SIZE_ATTR, batch_size)
    setattr(trainer, _SEED_ATTR, seed)
