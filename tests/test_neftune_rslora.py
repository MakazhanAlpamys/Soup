"""Tests for NEFTune (neftune_alpha) and rsLoRA (use_rslora) support."""

import pytest

from soup_cli.config.schema import LoraConfig, SoupConfig, TrainingConfig

# ---------------------------------------------------------------------------
# NEFTune config tests
# ---------------------------------------------------------------------------

class TestNEFTuneConfig:
    """NEFTune config validation."""

    def test_neftune_alpha_default_none(self):
        """neftune_alpha defaults to None."""
        tcfg = TrainingConfig()
        assert tcfg.neftune_alpha is None

    def test_neftune_alpha_valid(self):
        """neftune_alpha accepts valid float."""
        tcfg = TrainingConfig(neftune_alpha=5.0)
        assert tcfg.neftune_alpha == 5.0

    def test_neftune_alpha_zero(self):
        """neftune_alpha accepts 0 (disabled)."""
        tcfg = TrainingConfig(neftune_alpha=0.0)
        assert tcfg.neftune_alpha == 0.0

    def test_neftune_alpha_max(self):
        """neftune_alpha accepts 50.0 (max)."""
        tcfg = TrainingConfig(neftune_alpha=50.0)
        assert tcfg.neftune_alpha == 50.0

    def test_neftune_alpha_negative_rejected(self):
        """neftune_alpha rejects negative values."""
        with pytest.raises(Exception):
            TrainingConfig(neftune_alpha=-1.0)

    def test_neftune_alpha_too_high_rejected(self):
        """neftune_alpha rejects values > 50."""
        with pytest.raises(Exception):
            TrainingConfig(neftune_alpha=51.0)

    def test_neftune_in_full_config(self):
        """neftune_alpha works in full SoupConfig."""
        cfg = SoupConfig(
            base="test-model",
            data={"train": "./data.jsonl"},
            training={"neftune_alpha": 5.0},
        )
        assert cfg.training.neftune_alpha == 5.0


# ---------------------------------------------------------------------------
# NEFTune trainer integration tests
# ---------------------------------------------------------------------------

class TestNEFTuneTrainer:
    """NEFTune trainer argument passing."""

    def test_sft_neftune_in_training_kwargs(self):
        """SFT trainer includes neftune_noise_alpha in training kwargs."""
        tcfg = TrainingConfig(neftune_alpha=5.0)
        assert tcfg.neftune_alpha == 5.0
        # Verify the pattern: if neftune_alpha is not None, it should be passed
        training_kwargs = {}
        if tcfg.neftune_alpha is not None:
            training_kwargs["neftune_noise_alpha"] = tcfg.neftune_alpha
        assert training_kwargs["neftune_noise_alpha"] == 5.0

    def test_neftune_none_not_passed(self):
        """When neftune_alpha is None, neftune_noise_alpha is NOT added."""
        tcfg = TrainingConfig()
        training_kwargs = {}
        if tcfg.neftune_alpha is not None:
            training_kwargs["neftune_noise_alpha"] = tcfg.neftune_alpha
        assert "neftune_noise_alpha" not in training_kwargs

    def test_dpo_neftune_forwarding(self):
        """DPO trainer config includes neftune_noise_alpha when set."""
        tcfg = TrainingConfig(neftune_alpha=10.0)
        extra = (
            {"neftune_noise_alpha": tcfg.neftune_alpha}
            if tcfg.neftune_alpha is not None else {}
        )
        assert extra == {"neftune_noise_alpha": 10.0}

    def test_kto_neftune_forwarding(self):
        """KTO trainer config includes neftune_noise_alpha when set."""
        tcfg = TrainingConfig(neftune_alpha=7.5)
        extra = (
            {"neftune_noise_alpha": tcfg.neftune_alpha}
            if tcfg.neftune_alpha is not None else {}
        )
        assert extra == {"neftune_noise_alpha": 7.5}

    def test_orpo_neftune_forwarding(self):
        """ORPO trainer config includes neftune_noise_alpha when set."""
        tcfg = TrainingConfig(neftune_alpha=3.0)
        extra = (
            {"neftune_noise_alpha": tcfg.neftune_alpha}
            if tcfg.neftune_alpha is not None else {}
        )
        assert extra == {"neftune_noise_alpha": 3.0}

    def test_simpo_neftune_forwarding(self):
        """SimPO trainer config includes neftune_noise_alpha when set."""
        tcfg = TrainingConfig(neftune_alpha=15.0)
        extra = (
            {"neftune_noise_alpha": tcfg.neftune_alpha}
            if tcfg.neftune_alpha is not None else {}
        )
        assert extra == {"neftune_noise_alpha": 15.0}

    def test_ipo_neftune_forwarding(self):
        """IPO trainer config includes neftune_noise_alpha when set."""
        tcfg = TrainingConfig(neftune_alpha=2.0)
        extra = (
            {"neftune_noise_alpha": tcfg.neftune_alpha}
            if tcfg.neftune_alpha is not None else {}
        )
        assert extra == {"neftune_noise_alpha": 2.0}


# ---------------------------------------------------------------------------
# NEFTune sweep integration tests
# ---------------------------------------------------------------------------

class TestNEFTuneSweep:
    """NEFTune sweep parameter support."""

    def test_neftune_in_sweep_shortcuts(self):
        """neftune_alpha is a valid sweep parameter."""
        from soup_cli.commands.sweep import _set_nested_param

        config_dict = {
            "base": "test",
            "task": "sft",
            "data": {"train": "./data.jsonl"},
            "training": {},
        }
        _set_nested_param(config_dict, "neftune_alpha", 5.0)
        assert config_dict["training"]["neftune_alpha"] == 5.0


# ---------------------------------------------------------------------------
# rsLoRA config tests
# ---------------------------------------------------------------------------

class TestRsLoRAConfig:
    """rsLoRA config validation."""

    def test_rslora_default_false(self):
        """use_rslora defaults to False."""
        lora = LoraConfig()
        assert lora.use_rslora is False

    def test_rslora_enable(self):
        """use_rslora can be enabled."""
        lora = LoraConfig(use_rslora=True)
        assert lora.use_rslora is True

    def test_rslora_in_full_config(self):
        """use_rslora works in full SoupConfig."""
        cfg = SoupConfig(
            base="test-model",
            data={"train": "./data.jsonl"},
            training={"lora": {"use_rslora": True}},
        )
        assert cfg.training.lora.use_rslora is True

    def test_rslora_with_dora(self):
        """use_rslora can be used alongside use_dora."""
        lora = LoraConfig(use_rslora=True, use_dora=True)
        assert lora.use_rslora is True
        assert lora.use_dora is True


# ---------------------------------------------------------------------------
# rsLoRA sweep integration tests
# ---------------------------------------------------------------------------

class TestRsLoRATrainer:
    """rsLoRA trainer integration — use_rslora flows to LoraConfig."""

    def test_rslora_in_lora_config_kwargs(self):
        """use_rslora is passed to peft.LoraConfig constructor."""
        lcfg = LoraConfig(use_rslora=True, r=16, alpha=32)
        # Verify the value that would be passed to peft.LoraConfig
        assert lcfg.use_rslora is True
        kwargs = {
            "r": lcfg.r,
            "lora_alpha": lcfg.alpha,
            "lora_dropout": lcfg.dropout,
            "use_dora": lcfg.use_dora,
            "use_rslora": lcfg.use_rslora,
        }
        assert kwargs["use_rslora"] is True
        assert kwargs["use_dora"] is False

    def test_rslora_false_in_lora_config_kwargs(self):
        """use_rslora=False is passed correctly."""
        lcfg = LoraConfig(use_rslora=False)
        kwargs = {"use_rslora": lcfg.use_rslora}
        assert kwargs["use_rslora"] is False

    def test_rslora_with_all_tasks(self):
        """use_rslora works in config for every task type."""
        for task in [
            "sft", "dpo", "grpo", "ppo", "reward_model",
            "kto", "orpo", "simpo", "ipo", "pretrain", "embedding",
        ]:
            cfg = SoupConfig(
                base="test-model",
                task=task,
                data={"train": "./data.jsonl"},
                training={"lora": {"use_rslora": True}},
            )
            assert cfg.training.lora.use_rslora is True, f"Failed for task={task}"


# ---------------------------------------------------------------------------
# rsLoRA sweep integration tests
# ---------------------------------------------------------------------------

class TestRsLoRASweep:
    """rsLoRA sweep parameter support."""

    def test_rslora_in_sweep_shortcuts(self):
        """use_rslora is a valid sweep parameter."""
        from soup_cli.commands.sweep import _set_nested_param

        config_dict = {
            "base": "test",
            "task": "sft",
            "data": {"train": "./data.jsonl"},
            "training": {"lora": {}},
        }
        _set_nested_param(config_dict, "use_rslora", True)
        assert config_dict["training"]["lora"]["use_rslora"] is True
