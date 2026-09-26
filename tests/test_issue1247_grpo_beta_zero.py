"""Tests for #1247: allow grpo_beta: 0 for KL-free GRPO recipes."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig

VARIANTS = ("standard", "gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft")


class TestSchemaGrpoBetaZero:
    """Schema-level validation of grpo_beta: 0."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_grpo_beta_zero_loads_for_all_variants(self, variant):
        delta_line = "  grpo_delta: 0.2\n" if variant in ("two_sided", "gspo") else ""
        cfg_str = f"""
base: test-model
task: grpo
data:
  train: dummy.jsonl
  max_length: 64
training:
  grpo_beta: 0
  grpo_variant: {variant}
{delta_line}
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.grpo_beta == 0.0

    def test_default_grpo_beta_unchanged(self):
        cfg_str = """
base: test-model
task: grpo
data:
  train: dummy.jsonl
  max_length: 64
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.grpo_beta == 0.1

    def test_grpo_beta_zero_float_loads(self):
        tcfg = TrainingConfig(grpo_beta=0.0)
        assert tcfg.grpo_beta == 0.0

    def test_grpo_beta_negative_rejected(self):
        with pytest.raises(ValidationError):
            TrainingConfig(grpo_beta=-0.01)

    def test_grpo_beta_bool_rejected(self):
        with pytest.raises(ValidationError, match="grpo_beta must not be a boolean"):
            TrainingConfig(grpo_beta=True)
        with pytest.raises(ValidationError, match="grpo_beta must not be a boolean"):
            TrainingConfig(grpo_beta=False)

    def test_grpo_beta_nan_inf_rejected(self):
        with pytest.raises(ValidationError, match="grpo_beta must be finite"):
            TrainingConfig(grpo_beta=float("nan"))
        with pytest.raises(ValidationError, match="grpo_beta must be finite"):
            TrainingConfig(grpo_beta=float("inf"))


class TestRewardHackMitigationConflict:
    """grpo_beta: 0 is mutually exclusive with beta-adjusting mitigation."""

    @pytest.mark.parametrize("mode", ["kl_control", "pid_lagrangian"])
    def test_grpo_beta_zero_refused_with_beta_adjusting_mitigation(self, mode):
        cfg_str = f"""
base: test-model
task: grpo
data:
  train: dummy.jsonl
  max_length: 64
training:
  grpo_beta: 0
  reward_hack_detector: info_rm
  reward_hack_mitigation: {mode}
"""
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(cfg_str)
        err = str(excinfo.value)
        assert "grpo_beta" in err
        assert "reward_hack_mitigation" in err
        assert mode in err

    def test_grpo_beta_zero_accepted_with_log_only_mitigation(self):
        cfg_str = """
base: test-model
task: grpo
data:
  train: dummy.jsonl
  max_length: 64
training:
  grpo_beta: 0
  reward_hack_detector: info_rm
  reward_hack_mitigation: log_only
"""
        cfg = load_config_from_string(cfg_str)
        assert cfg.training.grpo_beta == 0.0
        assert cfg.training.reward_hack_mitigation == "log_only"


class TestKernelLossBitForBit:
    """Variant loss with beta=0 matches KL-free loss bit-for-bit without reference."""

    @pytest.mark.parametrize("variant", ["gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft"])
    def test_variant_loss_matches_kl_free_bit_for_bit(self, variant):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.grpo_variants import apply_variant_loss

        gen = torch.Generator().manual_seed(1247)
        dtype = torch.float64
        logp_new = (-2.0 * torch.rand(3, 5, generator=gen, dtype=dtype)).requires_grad_(True)
        logp_old = logp_new.detach() + 0.2 * torch.randn(3, 5, generator=gen, dtype=dtype)
        advantages = torch.tensor([1.0, -0.5, 0.75], dtype=dtype)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=dtype)
        delta = 0.2 if variant in ("two_sided", "gspo") else None

        # Call with beta=0.0 and NO reference_logp (simulates TRL skipping ref forward)
        loss_no_ref = apply_variant_loss(
            variant,
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=advantages,
            beta=0.0,
            delta=delta,
            completion_mask=mask,
            reference_logp=None,
        )

        # Call with dummy reference
        dummy_ref = logp_new.detach() + 5.0
        loss_with_ref = apply_variant_loss(
            variant,
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=advantages,
            beta=0.0,
            delta=delta,
            completion_mask=mask,
            reference_logp=dummy_ref,
        )

        assert torch.equal(loss_no_ref, loss_with_ref), (
            f"{variant}: loss with beta=0 changed when reference was provided"
        )


class TestRealTrainerBetaZero:
    """Real GRPOTrainerWrapper on tiny CPU model skips reference forward when beta=0."""

    @pytest.mark.parametrize("variant", ["standard", "dapo", "dr_grpo"])
    def test_ref_forward_skipped_when_beta_zero(self, tmp_path, monkeypatch, variant):
        for mod in ("torch", "transformers", "peft", "trl", "datasets"):
            pytest.importorskip(mod)
        import yaml

        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        from tests.test_issue1232_grpo_variant_kl import (
            _tiny_llama_dir,
            _write_tiny_tokenizer,
        )

        weights = _tiny_llama_dir(tmp_path)
        _write_tiny_tokenizer(weights)
        monkeypatch.chdir(tmp_path)

        cfg_dict = {
            "base": weights,
            "task": "grpo",
            "backend": "transformers",
            "modality": "text",
            "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
            "training": {
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "num_generations": 2,
                "reward_fn": "format",
                "quantization": "none",
                "epochs": 1,
                "grpo_variant": variant,
                "grpo_beta": 0,
                "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            },
            "output": str(tmp_path / "out"),
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_dict))
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [{"prompt": "hi"} for _ in range(4)]})

        trainer = wrapper.trainer
        assert trainer.beta == 0.0

        batch = next(iter(trainer.get_train_dataloader()))
        inputs = trainer._prepare_inputs(batch)
        # When beta=0, trl skips computing ref_per_token_logps
        assert inputs.get("ref_per_token_logps") is None, (
            "TRL should skip reference forward when beta is 0.0"
        )

        # Ensure loss computation succeeds without error
        loss = trainer._compute_loss(trainer.model, inputs)
        assert loss is not None
        assert not loss.isnan()


class TestDocumentationMentionsKlFree:
    def test_training_docs_mention_grpo_beta_zero(self):
        docs_path = Path(__file__).parents[1] / "docs" / "training.md"
        content = docs_path.read_text(encoding="utf-8")
        assert "grpo_beta: 0" in content
        assert "KL-free" in content
        assert "1247" in content
