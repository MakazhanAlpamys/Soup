"""Tests for v0.40.4 Part B — #65 multipack live HF Trainer wiring.

The v0.40.3 release shipped the ``make_multipack_trainer_class`` factory
plus ``attach_multipack_state``, ``lengths_from_dataset``, and
``detect_arch_name``, but did not wire them into the SFT / Pretrain
trainer wrappers — adversarial review surfaced that HF Trainer's
DataLoader expects ``Sampler[int]`` while ``MultipackBatchSampler``
yields ``list[list[int]]``. v0.40.4 fixes that by adding a
``get_train_dataloader`` override that installs the sampler as the
DataLoader's ``batch_sampler=`` kwarg.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from soup_cli.utils.multipack_trainer import (
    attach_multipack_state,
    make_multipack_trainer_class,
)


class TestGetTrainDataloaderOverrideExists:
    """The new override is part of the dynamically-generated subclass."""

    def test_subclass_has_method(self):
        class Base:
            def get_train_dataloader(self):
                return "default-dl"

        sub = make_multipack_trainer_class(Base)
        # The override is defined on the subclass itself.
        assert "get_train_dataloader" in sub.__dict__

    def test_falls_back_when_state_missing(self):
        class Base:
            def __init__(self):
                pass

            def get_train_dataloader(self):
                return "fallback-dl"

        sub = make_multipack_trainer_class(Base)
        instance = sub()
        # No state attached → defer to super().
        assert instance.get_train_dataloader() == "fallback-dl"

    def test_falls_back_when_train_dataset_missing(self):
        class Base:
            def __init__(self):
                pass

            def get_train_dataloader(self):
                return "fallback-no-ds"

        sub = make_multipack_trainer_class(Base)
        instance = sub()
        # State is set, but ``train_dataset`` attr is missing.
        attach_multipack_state(
            instance, lengths=[3, 4, 5], max_seq_len=64, batch_size=2,
        )
        assert instance.get_train_dataloader() == "fallback-no-ds"


class TestGetTrainDataloaderReturnsMultipackDataloader:
    """When state is attached and ``train_dataset`` is set, the override
    returns a ``DataLoader`` whose ``batch_sampler`` is the multipack
    sampler (NOT the standard scalar-Sampler from HF Trainer).
    """

    def test_returns_multipack_batch_sampler(self):
        try:
            from torch.utils.data import DataLoader, Dataset
        except ImportError:
            pytest.skip("torch not installed")

        class TinyDataset(Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {"x": idx}

        class Base:
            def __init__(self):
                self.train_dataset = TinyDataset(10)
                self.data_collator = None
                self.args = MagicMock(
                    dataloader_num_workers=0, dataloader_pin_memory=False,
                )

            def get_train_dataloader(self):
                return "should-not-be-called"

        sub = make_multipack_trainer_class(Base)
        instance = sub()
        attach_multipack_state(
            instance,
            lengths=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            max_seq_len=128,
            batch_size=2,
            seed=42,
        )

        dl = instance.get_train_dataloader()
        assert isinstance(dl, DataLoader)
        # The batch_sampler is a MultipackBatchSampler with real_batches=False.
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler

        assert isinstance(dl.batch_sampler, MultipackBatchSampler)
        # Sanity: real_batches=False → yields list[int] per pack
        # (NOT list[list[int]]).
        first_pack = next(iter(dl.batch_sampler))
        assert isinstance(first_pack, list)
        assert all(isinstance(x, int) for x in first_pack)


class TestGetTrainDataloaderForwardsDropLast:
    """v0.40.4 H3 — `args.dataloader_drop_last` must reach the
    MultipackBatchSampler; v0.40.4 first-cut hardcoded `drop_last=False`.
    """

    def test_drop_last_forwarded_to_sampler_constructor(self):
        # Source-level proof: the override reads ``dataloader_drop_last``
        # from ``self.args`` and passes it as ``drop_last=`` to
        # MultipackBatchSampler. (Live-spy patching is hard because the
        # factory function captures the symbol via free-variable closure
        # at definition time.)
        text = Path("src/soup_cli/utils/multipack_trainer.py").read_text(
            encoding="utf-8",
        )
        # The override block reads dataloader_drop_last from args.
        assert 'getattr(args, "dataloader_drop_last"' in text
        # And the sampler constructor receives it as `drop_last=drop_last`.
        assert "drop_last=drop_last" in text


class TestSftLiveWiring:
    """Source-level proof that sft.py instantiates the multipack subclass."""

    def test_sft_instantiates_subclass(self):
        text = Path("src/soup_cli/trainer/sft.py").read_text(encoding="utf-8")
        # The v0.40.3 yellow advisory string is GONE.
        assert "live HF Trainer wiring is deferred" not in text
        # The factory is invoked with SFTTrainer as the base.
        assert "make_multipack_trainer_class(SFTTrainer)" in text
        # State is attached.
        assert "attach_multipack_state(" in text
        # Architecture allowlist is consulted.
        assert "validate_multipack_architecture" in text

    def test_pretrain_instantiates_subclass(self):
        text = Path("src/soup_cli/trainer/pretrain.py").read_text(encoding="utf-8")
        assert "live HF Trainer wiring is deferred" not in text
        assert "make_multipack_trainer_class(SFTTrainer)" in text
        assert "attach_multipack_state(" in text


class TestRealTrainerSubclassHasOverride:
    """If transformers is installed, mix the override into the real
    Trainer MRO and confirm the new ``get_train_dataloader`` method
    shadows the parent.
    """

    def test_real_trainer_get_train_dataloader_in_dict(self):
        try:
            from transformers import Trainer
        except ImportError:
            pytest.skip("transformers not installed")
        sub = make_multipack_trainer_class(Trainer)
        # Our override is on the subclass itself, not inherited.
        assert "get_train_dataloader" in sub.__dict__
        assert sub.__dict__["get_train_dataloader"] is not Trainer.get_train_dataloader
