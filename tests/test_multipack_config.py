"""Tests for v0.37.0 Part B — multipack config wiring + sampler builder.

Covers:
- ``TrainingConfig.multipack`` Pydantic field default + type
- Cross-validator: ``multipack`` and ``packing`` are mutually exclusive
- Cross-validator: SoupConfig restricts ``multipack`` to sft / pretrain
- Cross-validator: MLX backend rejects multipack (sampler injection is HF Trainer-specific)
- ``build_multipack_sampler_for_lengths`` helper — returns a configured
  :class:`MultipackBatchSampler` from a list of sample lengths
- ``supports_multipack`` task allowlist
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.utils.multipack import (
    build_multipack_sampler_for_lengths,
    supports_multipack,
)
from soup_cli.utils.multipack_sampler import MultipackBatchSampler

# ---- TrainingConfig.multipack field --------------------------------------


def test_multipack_default_false():
    tcfg = TrainingConfig()
    assert tcfg.multipack is False


def test_multipack_accepts_true():
    tcfg = TrainingConfig(multipack=True)
    assert tcfg.multipack is True


def test_multipack_rejects_non_bool():
    # Pydantic v2 coerces "true"/"false" strings to bool, but rejects
    # arbitrary objects. A list cannot be coerced to bool.
    with pytest.raises(ValidationError):
        TrainingConfig(multipack=[1, 2])  # type: ignore[arg-type]


# ---- mutually exclusive with packing -------------------------------------


def test_multipack_packing_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainingConfig(multipack=True, packing=True)


def test_multipack_alone_ok():
    tcfg = TrainingConfig(multipack=True, packing=False)
    assert tcfg.multipack is True
    assert tcfg.packing is False


def test_packing_alone_ok():
    tcfg = TrainingConfig(packing=True, multipack=False)
    assert tcfg.packing is True
    assert tcfg.multipack is False


# ---- SoupConfig task gate ------------------------------------------------


def _base_soup_kwargs(task: str = "sft", **overrides):
    cfg = {
        "base": "fake-org/fake-model",
        "task": task,
        "data": {"train": "data.jsonl", "format": "alpaca"},
        "training": {"epochs": 1, "lr": 1e-4, "multipack": True},
        "output": "./out",
    }
    cfg.update(overrides)
    return cfg


def test_multipack_allowed_for_sft():
    cfg = SoupConfig(**_base_soup_kwargs(task="sft"))
    assert cfg.training.multipack is True


def test_multipack_allowed_for_pretrain():
    kwargs = _base_soup_kwargs(task="pretrain")
    kwargs["data"]["format"] = "plaintext"
    cfg = SoupConfig(**kwargs)
    assert cfg.training.multipack is True


@pytest.mark.parametrize(
    "task",
    ["dpo", "grpo", "kto", "orpo", "simpo", "ipo", "ppo",
     "reward_model", "embedding"],
)
def test_multipack_rejected_for_non_sft_pretrain(task):
    kwargs = _base_soup_kwargs(task=task)
    # Adjust data format to satisfy each task's data validator before our
    # multipack guard fires.
    if task in {"dpo", "kto", "orpo", "simpo", "ipo"}:
        kwargs["data"]["format"] = "dpo"
    elif task == "embedding":
        kwargs["data"]["format"] = "embedding"
    elif task == "reward_model":
        kwargs["data"]["format"] = "dpo"
    with pytest.raises(ValidationError, match="multipack"):
        SoupConfig(**kwargs)


def test_multipack_off_does_not_trip_task_gate():
    # multipack=False on a non-sft task should NOT raise.
    kwargs = _base_soup_kwargs(task="dpo")
    kwargs["training"]["multipack"] = False
    kwargs["data"]["format"] = "dpo"
    cfg = SoupConfig(**kwargs)
    assert cfg.training.multipack is False


def test_multipack_rejected_on_mlx_backend():
    kwargs = _base_soup_kwargs(task="sft")
    kwargs["backend"] = "mlx"
    with pytest.raises(ValidationError, match="mlx"):
        SoupConfig(**kwargs)


# ---- supports_multipack helper -------------------------------------------


def test_supports_multipack_allowed_tasks():
    assert supports_multipack("sft") is True
    assert supports_multipack("pretrain") is True


@pytest.mark.parametrize(
    "task",
    ["dpo", "grpo", "kto", "orpo", "simpo", "ipo", "ppo",
     "reward_model", "embedding"],
)
def test_supports_multipack_rejected_tasks(task):
    assert supports_multipack(task) is False


def test_supports_multipack_unknown_task():
    assert supports_multipack("nonexistent_task") is False


# ---- build_multipack_sampler_for_lengths ---------------------------------


def test_build_sampler_returns_multipack_sampler():
    tcfg = TrainingConfig(
        multipack=True, batch_size=2, packing=False,
    )
    lengths = [3, 5, 2, 4, 1, 6]
    sampler = build_multipack_sampler_for_lengths(
        lengths=lengths, tcfg=tcfg, max_seq_length=10, seed=0,
    )
    assert isinstance(sampler, MultipackBatchSampler)


def test_build_sampler_real_batches_uses_batch_size():
    tcfg = TrainingConfig(
        multipack=True, batch_size=4, packing=False,
    )
    lengths = [3] * 16
    sampler = build_multipack_sampler_for_lengths(
        lengths=lengths, tcfg=tcfg, max_seq_length=12,
        real_batches=True, seed=0,
    )
    for batch in sampler:
        assert len(batch) <= 4


def test_build_sampler_flat_mode():
    # real_batches=False yields flat index lists, max_len = batch_size * max_seq_length
    tcfg = TrainingConfig(
        multipack=True, batch_size=4, packing=False,
    )
    lengths = [10, 8, 6, 4, 2]
    sampler = build_multipack_sampler_for_lengths(
        lengths=lengths, tcfg=tcfg, max_seq_length=8,
        real_batches=False, seed=0,
    )
    # max bin len in flat mode = 4 * 8 = 32, so total of all lengths (=30)
    # should fit in one bin given FFD.
    batches = list(sampler)
    assert len(batches) == 1, "expected single bin given budget"
    flat = sorted(idx for batch in batches for idx in batch)
    assert flat == list(range(5))


def test_build_sampler_requires_multipack_enabled():
    tcfg = TrainingConfig(multipack=False)
    with pytest.raises(ValueError, match="multipack"):
        build_multipack_sampler_for_lengths(
            lengths=[3, 4], tcfg=tcfg, max_seq_length=10, seed=0,
        )


def test_build_sampler_rejects_non_positive_max_seq_length():
    tcfg = TrainingConfig(multipack=True)
    with pytest.raises(ValueError, match="max_seq_length"):
        build_multipack_sampler_for_lengths(
            lengths=[3, 4], tcfg=tcfg, max_seq_length=0, seed=0,
        )


def test_build_sampler_rejects_bool_max_seq_length():
    # bool is subclass of int — reject explicitly per v0.30.0+ policy.
    tcfg = TrainingConfig(multipack=True)
    with pytest.raises(TypeError, match="bool"):
        build_multipack_sampler_for_lengths(
            lengths=[3, 4], tcfg=tcfg, max_seq_length=True, seed=0,
        )


def test_build_sampler_rejects_auto_batch_size():
    tcfg = TrainingConfig(multipack=True, batch_size="auto")
    with pytest.raises(ValueError, match="auto"):
        build_multipack_sampler_for_lengths(
            lengths=[3, 4], tcfg=tcfg, max_seq_length=10, seed=0,
        )


def test_build_sampler_seed_determinism():
    tcfg = TrainingConfig(multipack=True, batch_size=2)
    lengths = [3, 5, 2, 4, 1, 6, 7, 2]
    s1 = build_multipack_sampler_for_lengths(
        lengths=lengths, tcfg=tcfg, max_seq_length=10, seed=42,
    )
    s2 = build_multipack_sampler_for_lengths(
        lengths=lengths, tcfg=tcfg, max_seq_length=10, seed=42,
    )
    assert list(s1) == list(s2)
