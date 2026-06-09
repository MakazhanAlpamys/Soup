"""Tests for v0.71.19 — Quant-menu + multipack hardening.

Closes:

* #81 — Quant Menu (gptq / awq / hqq:Nbit / aqlm / eetq / mxfp4 / fp8) was
  rejected by the SoupConfig modality gate for ``modality in {vision, audio}``.
  The vision / audio ``_setup_*`` paths in ``sft.py`` carried inline
  ``BitsAndBytesConfig`` blocks (4bit / 8bit only). v0.71.19 drops the gate and
  threads the unified ``build_quantization_config_for_loader`` through both
  paths so multi-modal training can use the full quant menu.

* #80 — the multipack ``get_train_dataloader`` override built a raw
  ``DataLoader`` and returned it directly, so under FSDP / DeepSpeed ZeRO / DDP
  every rank trained on the SAME packed bins (no data sharding). v0.71.19 routes
  the DataLoader through ``accelerator.prepare`` when ``num_processes > 1`` so
  accelerate's ``BatchSamplerShard`` shards whole bins across ranks. The
  single-process path is unchanged (raw DataLoader, the validated v0.40.4
  behaviour). Full multi-GPU validation stays a QA issue; this is a mocked-env
  test of the routing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from soup_cli.config.loader import load_config_from_string

# Anchor source reads on the repo root derived from this file's location so the
# source-grep tests survive another test's ``monkeypatch.chdir`` (cwd leak) in
# the full suite — matches the v0.71.5 precedent.
_REPO = Path(__file__).resolve().parent.parent
_SFT_SRC = (_REPO / "src/soup_cli/trainer/sft.py").read_text(encoding="utf-8")
_MP_SRC = (_REPO / "src/soup_cli/utils/multipack_trainer.py").read_text(encoding="utf-8")
_SCHEMA_SRC = (_REPO / "src/soup_cli/config/schema.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# #81 — Quant Menu for vision / audio modality (schema gate dropped)
# ---------------------------------------------------------------------------


class TestVisionAudioQuantMenuSchema:
    """The ``modality != 'text'`` Quant Menu rejection is gone; vision / audio
    configs with a quant-menu format now load."""

    # Every quant-menu format the unified loader supports, on BOTH multi-modal
    # modalities — covers the docstring-advertised menu incl. aqlm / mxfp4 / fp8.
    @pytest.mark.parametrize("modality", ["vision", "audio"])
    @pytest.mark.parametrize(
        "fmt", ["gptq", "awq", "hqq:4bit", "aqlm", "eetq", "mxfp4", "fp8"]
    )
    def test_quant_menu_format_accepted(self, modality, fmt):
        data_fmt = "llava" if modality == "vision" else "audio"
        cfg = load_config_from_string(
            f"base: m\ntask: sft\nmodality: {modality}\n"
            f"data: {{train: d.jsonl, format: {data_fmt}}}\n"
            f"training: {{quantization: {fmt}}}\n"
        )
        assert cfg.modality == modality
        assert cfg.training.quantization == fmt

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    @pytest.mark.parametrize("fmt", ["4bit", "8bit"])
    def test_bnb_formats_still_accepted(self, modality, fmt):
        # bnb 4bit/8bit were always universal — regression guard for both
        # modalities (the inline-BNB blocks they replace handled only these).
        data_fmt = "llava" if modality == "vision" else "audio"
        cfg = load_config_from_string(
            f"base: m\ntask: sft\nmodality: {modality}\n"
            f"data: {{train: d.jsonl, format: {data_fmt}}}\n"
            f"training: {{quantization: {fmt}}}\n"
        )
        assert cfg.training.quantization == fmt

    def test_text_gptq_still_accepted(self):
        # Text path unchanged — regression guard.
        cfg = load_config_from_string(
            """
base: TheBloke/Llama-2-7B-GPTQ
task: sft
data: {train: d.jsonl}
training: {quantization: gptq}
"""
        )
        assert cfg.modality == "text"
        assert cfg.training.quantization == "gptq"

    def test_mlx_quant_menu_still_rejected(self):
        # mlx backend gate is independent of modality and must still fire.
        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(
                """
base: m
task: sft
backend: mlx
data: {train: d.jsonl}
training: {quantization: hqq:4bit}
"""
            )


class TestVisionAudioQuantMenuWiring:
    """Source-level proof that the vision / audio setup paths use the unified
    quant-menu loader and dropped the inline ``BitsAndBytesConfig`` blocks."""

    def test_vision_setup_uses_unified_loader(self):
        # The vision setup method threads build_quantization_config_for_loader.
        # Split on the ``def`` (not the call site in setup()) to isolate the body.
        vision_block = _SFT_SRC.split("def _setup_vision_transformers")[1].split(
            "def _prepare_vision_dataset"
        )[0]
        assert "build_quantization_config_for_loader" in vision_block

    def test_audio_setup_uses_unified_loader(self):
        audio_block = _SFT_SRC.split("def _setup_audio_transformers")[1].split(
            "def _prepare_audio_dataset"
        )[0]
        assert "build_quantization_config_for_loader" in audio_block

    def test_vision_setup_no_inline_bnb_config(self):
        vision_block = _SFT_SRC.split("def _setup_vision_transformers")[1].split(
            "def _prepare_vision_dataset"
        )[0]
        # The inline BitsAndBytesConfig construction is gone (the unified
        # loader builds it lazily). Importing BitsAndBytesConfig in the method
        # is also gone.
        assert "BitsAndBytesConfig(" not in vision_block
        assert "import BitsAndBytesConfig" not in vision_block

    def test_audio_setup_no_inline_bnb_config(self):
        audio_block = _SFT_SRC.split("def _setup_audio_transformers")[1].split(
            "def _prepare_audio_dataset"
        )[0]
        assert "BitsAndBytesConfig(" not in audio_block
        assert "import BitsAndBytesConfig" not in audio_block

    def test_vision_setup_still_prepares_kbit(self):
        # prepare_model_for_kbit_training is still gated on bnb formats so
        # 4bit/8bit/mxfp4 still run through kbit-prep in vision/audio.
        vision_block = _SFT_SRC.split("def _setup_vision_transformers")[1].split(
            "def _prepare_vision_dataset"
        )[0]
        assert "prepare_model_for_kbit_training" in vision_block

    def test_schema_no_longer_rejects_non_text_modality(self):
        gate = _SCHEMA_SRC.split("_validate_quant_menu_supported_tasks")[1].split(
            "def _validate_preference_dispatcher"
        )[0]
        # The modality-specific rejection branch is removed.
        assert "modality='text' only" not in gate
        assert 'self.modality != "text"' not in gate
        # The mlx gate is retained.
        assert "mlx" in gate


# ---------------------------------------------------------------------------
# #80 — multipack get_train_dataloader hardening under FSDP / DeepSpeed / DDP
# ---------------------------------------------------------------------------


# Module-level sentinel: passed as ``accelerator`` to mean "the base class has
# no ``accelerator`` attribute at all" (vs ``accelerator=None``).
_NO_ACCEL = object()


def _torch_or_skip():
    try:
        from torch.utils.data import DataLoader, Dataset
    except ImportError:  # pragma: no cover - torch always present in [dev]
        pytest.skip("torch not installed")
    return DataLoader, Dataset


def _make_base_class(*, accelerator):
    """Return a fresh base class for ``make_multipack_trainer_class`` whose
    instances expose ``train_dataset`` / ``data_collator`` / ``args`` and the
    supplied ``accelerator`` (or no ``accelerator`` attr when ``_NO_ACCEL``)."""
    _, dataset_cls = _torch_or_skip()

    class TinyDataset(dataset_cls):
        def __len__(self):
            return 12

        def __getitem__(self, idx):
            return {"x": idx}

    class _Base:
        def __init__(self):
            self.train_dataset = TinyDataset()
            self.data_collator = None
            self.args = MagicMock(
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                dataloader_drop_last=False,
            )
            if accelerator is not _NO_ACCEL:
                self.accelerator = accelerator

        def get_train_dataloader(self):
            return "super-dl"

    return _Base


def _attach(instance):
    from soup_cli.utils.multipack_trainer import attach_multipack_state

    attach_multipack_state(
        instance,
        lengths=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 12, 24],
        max_seq_len=128,
        batch_size=2,
        seed=7,
    )


class TestMultipackDistributedDataLoader:
    def test_routes_through_accelerator_prepare_when_distributed(self):
        dataloader_cls, _ = _torch_or_skip()
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        sentinel = object()
        captured: dict = {}

        def _prepare(loader):
            captured["loader"] = loader
            return sentinel

        accel = MagicMock()
        accel.num_processes = 2  # simulate FSDP / ZeRO / DDP
        accel.prepare.side_effect = _prepare

        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        _attach(instance)

        result = instance.get_train_dataloader()

        accel.prepare.assert_called_once()
        assert result is sentinel
        prepared = captured["loader"]
        assert isinstance(prepared, dataloader_cls)
        assert isinstance(prepared.batch_sampler, MultipackBatchSampler)
        # The sampler is built with the ATTACHED seed (7), identical on every
        # rank — sharding happens after, in accelerate's BatchSamplerShard. A
        # regression that hardcoded seed=0 or added `+ rank` would break the
        # cross-rank global-order invariant; assert the seed propagated.
        assert prepared.batch_sampler._seed == 7
        # real_batches=False contract preserved: each pack is a flat list[int].
        first = next(iter(prepared.batch_sampler))
        assert isinstance(first, list)
        assert all(isinstance(x, int) for x in first)

    def test_single_process_returns_raw_dataloader(self):
        dataloader_cls, _ = _torch_or_skip()
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        accel = MagicMock()
        accel.num_processes = 1  # single GPU / CPU
        accel.prepare.side_effect = AssertionError("should not be called")

        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        _attach(instance)

        dl = instance.get_train_dataloader()
        accel.prepare.assert_not_called()
        assert isinstance(dl, dataloader_cls)
        assert isinstance(dl.batch_sampler, MultipackBatchSampler)
        # Single-process path must keep the v0.40.4 flat-yield contract
        # (real_batches=False → list[int] per pack), not silently flip to
        # real_batches=True.
        first = next(iter(dl.batch_sampler))
        assert isinstance(first, list)
        assert all(isinstance(x, int) for x in first)

    def test_no_accelerator_attr_returns_raw_dataloader(self):
        dataloader_cls, _ = _torch_or_skip()
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        # Build a base with NO accelerator attribute at all (older transformers
        # / direct construction).
        base_cls = _make_base_class(accelerator=_NO_ACCEL)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        assert not hasattr(instance, "accelerator")
        _attach(instance)

        dl = instance.get_train_dataloader()
        assert isinstance(dl, dataloader_cls)
        assert isinstance(dl.batch_sampler, MultipackBatchSampler)

    def test_accelerator_none_returns_raw_dataloader(self):
        dataloader_cls, _ = _torch_or_skip()
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        base_cls = _make_base_class(accelerator=None)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        _attach(instance)

        dl = instance.get_train_dataloader()
        assert isinstance(dl, dataloader_cls)
        assert isinstance(dl.batch_sampler, MultipackBatchSampler)

    def test_unconfigured_magicmock_accelerator_does_not_prepare(self):
        # Defence-in-depth: an accelerator whose num_processes is a MagicMock
        # (not a real int) must NOT route through prepare — `MagicMock() > 1`
        # is truthy, so the isinstance(int) guard is required.
        dataloader_cls, _ = _torch_or_skip()
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        accel = MagicMock()  # num_processes is an auto-MagicMock (not int)
        accel.prepare.side_effect = AssertionError("should not be called")

        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        _attach(instance)

        dl = instance.get_train_dataloader()
        accel.prepare.assert_not_called()
        assert isinstance(dl, dataloader_cls)

    def test_falls_back_to_super_when_state_missing(self):
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        accel = MagicMock()
        accel.num_processes = 4
        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        # No attach_multipack_state — must delegate to super (no prepare).
        assert instance.get_train_dataloader() == "super-dl"
        accel.prepare.assert_not_called()

    def test_empty_lengths_falls_back_to_super(self):
        # The `not lengths` arm of the state guard: attrs present but lengths is
        # an empty list (unreachable via attach_multipack_state, which rejects
        # it — so poke the attrs directly to exercise the defensive branch).
        from soup_cli.utils.multipack_trainer import (
            _BATCH_SIZE_ATTR,
            _LENGTHS_ATTR,
            _MAX_SEQ_ATTR,
            make_multipack_trainer_class,
        )

        accel = MagicMock()
        accel.num_processes = 1
        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        setattr(instance, _LENGTHS_ATTR, [])
        setattr(instance, _MAX_SEQ_ATTR, 128)
        setattr(instance, _BATCH_SIZE_ATTR, 2)
        assert instance.get_train_dataloader() == "super-dl"

    def test_drop_last_forwarded_through_override(self):
        # v0.40.4 H3 regression, behaviourally — `args.dataloader_drop_last`
        # must reach the MultipackBatchSampler through the #80-refactored path.
        _torch_or_skip()
        from soup_cli.utils.multipack_trainer import make_multipack_trainer_class

        accel = MagicMock()
        accel.num_processes = 1  # raw path so we can inspect the sampler
        base_cls = _make_base_class(accelerator=accel)
        sub = make_multipack_trainer_class(base_cls)
        instance = sub()
        instance.args.dataloader_drop_last = True
        _attach(instance)

        dl = instance.get_train_dataloader()
        assert dl.batch_sampler._drop_last is True


class TestMultipackSourceWiring:
    def test_override_routes_through_accelerator_prepare(self):
        # The override reads self.accelerator and routes the loader through
        # accelerate's prepare under distribution.
        assert 'getattr(self, "accelerator"' in _MP_SRC
        assert "prepare(loader)" in _MP_SRC
        # Gated on num_processes > 1 so the single-process path is unchanged.
        assert "num_processes" in _MP_SRC

    def test_drop_last_still_forwarded(self):
        # Regression guard from v0.40.4 H3 — must survive the #80 refactor.
        assert 'getattr(args, "dataloader_drop_last"' in _MP_SRC
        assert "drop_last=drop_last" in _MP_SRC


# ---------------------------------------------------------------------------
# Patch invariants
# ---------------------------------------------------------------------------


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(p) for p in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 19), soup_cli.__version__
