"""Tests for issue #797 item 4 — unsloth path must honour training.gradient_checkpointing.

Before the fix, ``load_model_and_tokenizer`` hardcoded
``use_gradient_checkpointing="unsloth"`` in ``get_peft_model``, so setting
``training.gradient_checkpointing: false`` was silently ignored on the unsloth
path. The fix resolves the flag via ``resolve_unsloth_gradient_checkpointing``
(built on the codebase's established ``should_enable_hf_gradient_checkpointing``
rule) and threads it through every trainer's ``_setup_unsloth``.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.unsloth import (
    load_model_and_tokenizer,
    resolve_unsloth_gradient_checkpointing,
)

# ─── Resolver unit tests ─────────────────────────────────────────────────────


class TestResolveUnslothGradientCheckpointing:
    def test_false_disables(self):
        """gradient_checkpointing: false → no checkpointing on the unsloth path."""
        assert resolve_unsloth_gradient_checkpointing(False) is False

    def test_none_disables(self):
        assert resolve_unsloth_gradient_checkpointing(None) is False

    def test_true_enables_unsloth_kernel(self):
        assert resolve_unsloth_gradient_checkpointing(True) == "unsloth"

    def test_full_tier_enables_unsloth_kernel(self):
        """Tier strings ('selective'/'medium'/'full'/'auto') mean on → 'unsloth'."""
        for tier in ("selective", "medium", "full", "auto"):
            assert resolve_unsloth_gradient_checkpointing(tier) == "unsloth"

    def test_stream_layers_forces_off(self):
        """Layer streaming already checkpoints every layer — streaming wins."""
        assert resolve_unsloth_gradient_checkpointing(True, stream_layers=True) is False

    def test_false_with_stream_layers_stays_off(self):
        assert resolve_unsloth_gradient_checkpointing(False, stream_layers=True) is False


# ─── Loader passthrough tests (unsloth mocked) ───────────────────────────────

_MODEL_KWARGS = dict(model_name="test-model", max_seq_length=2048, quantization="4bit")


def _mocked_unsloth_call(**load_kwargs):
    """Call load_model_and_tokenizer with unsloth mocked; return get_peft_model kwargs."""
    mock_flm = MagicMock()
    mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
    mock_flm.get_peft_model.return_value = MagicMock()
    with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
        load_model_and_tokenizer(**_MODEL_KWARGS, **load_kwargs)
    return mock_flm.get_peft_model.call_args[1]


class TestLoadModelAndTokenizerCheckpointing:
    def test_explicit_false_forwarded(self):
        """The #797 regression: False must reach get_peft_model, not the hardcoded default."""
        kwargs = _mocked_unsloth_call(use_gradient_checkpointing=False)
        assert kwargs["use_gradient_checkpointing"] is False

    def test_explicit_unsloth_forwarded(self):
        kwargs = _mocked_unsloth_call(use_gradient_checkpointing="unsloth")
        assert kwargs["use_gradient_checkpointing"] == "unsloth"

    def test_default_preserves_prior_behavior(self):
        """Default keeps today's behaviour ('unsloth') for existing call sites."""
        assert _mocked_unsloth_call()["use_gradient_checkpointing"] == "unsloth"


# ─── Trainer wiring tests (runtime, unsloth loader mocked) ───────────────────

TRAINER_MODULES = [
    "soup_cli.trainer.bco",
    "soup_cli.trainer.dpo",
    "soup_cli.trainer.embedding",
    "soup_cli.trainer.grpo",
    "soup_cli.trainer.ipo",
    "soup_cli.trainer.kto",
    "soup_cli.trainer.orpo",
    "soup_cli.trainer.ppo",
    "soup_cli.trainer.pretrain",
    "soup_cli.trainer.sft",
    "soup_cli.trainer.simpo",
]


def _trainer_cls(module_name):
    """First class defined in the module that owns ``_setup_unsloth``."""
    module = importlib.import_module(module_name)
    return next(
        obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and "_setup_unsloth" in obj.__dict__
    )


def _mock_configs(gradient_checkpointing):
    cfg = MagicMock()
    cfg.base = "test-model"
    cfg.data.max_length = 2048
    tcfg = MagicMock()
    tcfg.quantization = "4bit"
    tcfg.lora.r = 64
    tcfg.lora.alpha = 16
    tcfg.lora.dropout = 0.05
    tcfg.lora.target_modules = "auto"
    tcfg.lora.use_rslora = False
    tcfg.gradient_checkpointing = gradient_checkpointing
    tcfg.stream_layers = False
    return cfg, tcfg


def _run_setup_unsloth(module_name, gradient_checkpointing):
    """Invoke the real _setup_unsloth with the loader mocked; return loader kwargs."""
    trainer = _trainer_cls(module_name).__new__(_trainer_cls(module_name))
    cfg, tcfg = _mock_configs(gradient_checkpointing)
    with patch("soup_cli.utils.unsloth.load_model_and_tokenizer") as mock_loader:
        mock_loader.return_value = (MagicMock(), MagicMock())
        trainer._setup_unsloth(cfg, tcfg)
    return mock_loader.call_args[1]


class TestTrainerSetupUnslothWiring:
    """Every _setup_unsloth must resolve the flag and hand it to the loader."""

    @pytest.mark.parametrize("module_name", TRAINER_MODULES)
    def test_setup_unsloth_forwards_false(self, module_name):
        """The #797 case end-to-end: config false → loader receives False."""
        kwargs = _run_setup_unsloth(module_name, gradient_checkpointing=False)
        assert kwargs["use_gradient_checkpointing"] is False

    @pytest.mark.parametrize("module_name", TRAINER_MODULES)
    def test_setup_unsloth_forwards_unsloth(self, module_name):
        """Config true → loader receives 'unsloth'."""
        kwargs = _run_setup_unsloth(module_name, gradient_checkpointing=True)
        assert kwargs["use_gradient_checkpointing"] == "unsloth"


# ─── Real-config regression test (schema-derived default) ────────────────────


def _real_configs(**training_overrides):
    """Build a real SoupConfig; gradient_checkpointing comes from the schema default."""
    data = {"train": "dummy.jsonl"}
    if training_overrides:
        cfg = SoupConfig(base="test-model", data=data, training=training_overrides)
    else:
        cfg = SoupConfig(base="test-model", data=data)
    return cfg, cfg.training


def _run_real_setup_unsloth(module_name, cfg, tcfg):
    """Invoke the real _setup_unsloth with a real config and the loader mocked."""
    trainer = _trainer_cls(module_name).__new__(_trainer_cls(module_name))
    with patch("soup_cli.utils.unsloth.load_model_and_tokenizer") as mock_loader:
        mock_loader.return_value = (MagicMock(), MagicMock())
        trainer._setup_unsloth(cfg, tcfg)
    return mock_loader.call_args[1]


class TestRealConfigUnslothWiring:
    """Schema-derived config through the real trainer setup, Unsloth loader mocked."""

    @pytest.mark.parametrize("module_name", TRAINER_MODULES)
    def test_schema_default_false_reaches_loader(self, module_name):
        """Unset training.gradient_checkpointing (schema default False) → loader gets False."""
        cfg, tcfg = _real_configs()
        assert tcfg.gradient_checkpointing is False  # schema default, not a mock
        kwargs = _run_real_setup_unsloth(module_name, cfg, tcfg)
        assert kwargs["use_gradient_checkpointing"] is False

    @pytest.mark.parametrize("module_name", TRAINER_MODULES)
    def test_explicit_true_reaches_loader(self, module_name):
        """Explicit training.gradient_checkpointing: true → loader gets 'unsloth'."""
        cfg, _ = _real_configs(gradient_checkpointing=True)
        kwargs = _run_real_setup_unsloth(module_name, cfg, cfg.training)
        assert kwargs["use_gradient_checkpointing"] == "unsloth"

    def test_negative_control_old_hardcoded_behavior_fails(self):
        """Sensitivity check: with the old hardcoded 'unsloth', the assertion must fail."""
        cfg, tcfg = _real_configs()
        trainer = _trainer_cls("soup_cli.trainer.sft").__new__(_trainer_cls("soup_cli.trainer.sft"))
        with (
            patch("soup_cli.utils.unsloth.load_model_and_tokenizer") as mock_loader,
            patch(
                "soup_cli.utils.unsloth.resolve_unsloth_gradient_checkpointing",
                return_value="unsloth",
            ),
        ):
            mock_loader.return_value = (MagicMock(), MagicMock())
            trainer._setup_unsloth(cfg, tcfg)
        # Old behaviour: loader receives 'unsloth' even though the config says False
        assert mock_loader.call_args[1]["use_gradient_checkpointing"] == "unsloth"
        with pytest.raises(AssertionError):
            assert mock_loader.call_args[1]["use_gradient_checkpointing"] is False
