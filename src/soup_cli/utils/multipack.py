"""Multipack helper — task allowlist + sampler builder for trainer wrappers.

This is the high-level dispatcher that trainer wrappers call to get a
configured :class:`MultipackBatchSampler` from a tokenized dataset's per-
sample lengths. Keeping the construction logic out of the trainer modules
means the v0.37.0 wiring touches each trainer in ~3 lines.

Tasks wired in v0.37.0: ``sft``, ``pretrain``. Preference / RLHF / reward-
model trainers operate on paired data where multipack's index-bag
abstraction does not cleanly apply — those land in a future release.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from soup_cli.utils.multipack_sampler import MultipackBatchSampler

if TYPE_CHECKING:
    from soup_cli.config.schema import TrainingConfig


# v0.37.0 wiring — only flat-sequence tasks. Preference tasks deferred.
_MULTIPACK_SUPPORTED_TASKS: frozenset[str] = frozenset({"sft", "pretrain"})


def supports_multipack(task: str) -> bool:
    """Return True if ``task`` has multipack sampler wiring.

    Mirrors :func:`soup_cli.utils.v028_features.supports_v028_features`
    naming and shape so ``SoupConfig`` validators stay symmetric.
    """
    if not isinstance(task, str):
        return False
    return task in _MULTIPACK_SUPPORTED_TASKS


def build_multipack_sampler_for_lengths(
    *,
    lengths: Sequence[int],
    tcfg: TrainingConfig,
    max_seq_length: int,
    real_batches: bool = True,
    seed: int = 0,
) -> MultipackBatchSampler:
    """Build a :class:`MultipackBatchSampler` from per-sample token lengths.

    Args:
        lengths: token counts for each sample in the (already tokenized)
            training dataset.
        tcfg: the run's :class:`TrainingConfig`. Must have ``multipack=True``.
        max_seq_length: per-sample maximum length (matches
            ``DataConfig.max_length``).
        real_batches: when True, yields list-of-bins per call (collator
            stacks bins along the batch dim). When False, yields one flat
            bin at a time and ``batch_max_len = batch_size * max_seq_length``
            — Axolotl's "micro-batch as flat sequence" trick that maximises
            packing density.
        seed: deterministic shuffle seed (mirrored across DDP ranks).

    Returns:
        A configured :class:`MultipackBatchSampler`.

    Raises:
        ValueError: when ``multipack`` is not enabled on ``tcfg`` or when
            ``max_seq_length`` is non-positive.
    """
    if not getattr(tcfg, "multipack", False):
        raise ValueError(
            "build_multipack_sampler_for_lengths called but "
            "tcfg.multipack=False — guard upstream"
        )
    if isinstance(max_seq_length, bool):
        raise TypeError(
            f"max_seq_length must not be bool, got {max_seq_length!r}"
        )
    if not isinstance(max_seq_length, int):
        raise TypeError(
            f"max_seq_length must be int, got "
            f"{type(max_seq_length).__name__}"
        )
    if max_seq_length <= 0:
        raise ValueError(
            f"max_seq_length must be positive, got {max_seq_length}"
        )

    raw_batch_size = getattr(tcfg, "batch_size", 1) or 1
    # batch_size=='auto' must be resolved upstream before the sampler is
    # built — multipack needs an explicit integer to size each bin.
    if isinstance(raw_batch_size, str):
        raise ValueError(
            f"multipack requires an explicit integer batch_size, got "
            f"{raw_batch_size!r}. Resolve auto-batch-size before calling "
            "build_multipack_sampler_for_lengths (e.g. via "
            "soup_cli.utils.batch_probe.pick_batch_size)."
        )
    # bool is a subclass of int — reject explicitly (matches v0.30.0 policy).
    if isinstance(raw_batch_size, bool):
        raise TypeError(
            f"tcfg.batch_size must not be bool, got {raw_batch_size!r}"
        )
    if not isinstance(raw_batch_size, int):
        raise TypeError(
            f"tcfg.batch_size must be int, got {type(raw_batch_size).__name__}"
        )
    batch_size = raw_batch_size

    if real_batches:
        batch_max_len = max_seq_length
    else:
        # Axolotl trick: flat mode treats one bin as "batch_size copies of
        # max_seq_length stitched together" so FFD has more room to pack.
        batch_max_len = batch_size * max_seq_length

    return MultipackBatchSampler(
        lengths=lengths,
        batch_max_len=batch_max_len,
        batch_size=batch_size if real_batches else 1,
        real_batches=real_batches,
        seed=seed,
        drop_last=False,
    )
