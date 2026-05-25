"""v0.70.0 Part D — Mid-epoch checkpoint for PPO/GRPO.

Optimizer-state serialization for long RL runs. TorchTune explicitly
punts this; Soup ships the schema + state-manifest builder here, with
the live save_state / load_state callback deferred to v0.70.1.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class TestRLCheckpointPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import rl_checkpoint

        assert hasattr(rl_checkpoint, "RLCheckpointConfig")
        assert hasattr(rl_checkpoint, "RLCheckpointState")
        assert hasattr(rl_checkpoint, "validate_save_every_steps")
        assert hasattr(rl_checkpoint, "build_rl_checkpoint_callback")


class TestValidateSaveEverySteps:
    def test_happy(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        assert validate_save_every_steps(100) == 100
        assert validate_save_every_steps(1) == 1

    def test_max_boundary(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        # 10M steps is plenty.
        assert validate_save_every_steps(10_000_000) == 10_000_000

    def test_zero_rejected(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        with pytest.raises(ValueError, match=">= 1"):
            validate_save_every_steps(0)

    def test_negative_rejected(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        with pytest.raises(ValueError, match=">= 1"):
            validate_save_every_steps(-10)

    def test_above_cap_rejected(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        with pytest.raises(ValueError, match="10000000"):
            validate_save_every_steps(10_000_001)

    def test_bool_rejected(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        with pytest.raises(ValueError, match="bool"):
            validate_save_every_steps(True)

    def test_non_int_rejected(self):
        from soup_cli.utils.rl_checkpoint import validate_save_every_steps

        with pytest.raises(ValueError, match="int"):
            validate_save_every_steps(100.5)


class TestRLCheckpointConfig:
    def test_defaults(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        cfg = RLCheckpointConfig(save_every_steps=100)
        assert cfg.save_every_steps == 100
        assert cfg.include_optimizer_state is True
        assert cfg.include_ref_model is False
        assert cfg.include_rollout_buffer is False
        assert cfg.keep_last == 3

    def test_frozen(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        cfg = RLCheckpointConfig(save_every_steps=100)
        with pytest.raises(FrozenInstanceError):
            cfg.save_every_steps = 50  # type: ignore[misc]

    def test_keep_last_bounds(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        cfg = RLCheckpointConfig(save_every_steps=100, keep_last=10)
        assert cfg.keep_last == 10

        with pytest.raises(ValueError, match="keep_last"):
            RLCheckpointConfig(save_every_steps=100, keep_last=0)

        with pytest.raises(ValueError, match="keep_last"):
            RLCheckpointConfig(save_every_steps=100, keep_last=101)

    def test_keep_last_bool_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        with pytest.raises(ValueError, match="bool"):
            RLCheckpointConfig(save_every_steps=100, keep_last=True)

    def test_invalid_save_every_propagates(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        with pytest.raises(ValueError):
            RLCheckpointConfig(save_every_steps=0)

    def test_bool_flags_must_be_bool(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointConfig

        with pytest.raises(TypeError, match="bool"):
            RLCheckpointConfig(
                save_every_steps=100,
                include_optimizer_state="yes",  # type: ignore[arg-type]
            )


class TestRLCheckpointState:
    """State manifest written to .soup-rl-ckpt/step-NNN/manifest.json.

    Frozen + JSON-serialisable + per-field validation.
    """

    def test_basic(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        st = RLCheckpointState(
            step=500,
            checkpoint_dir="./.soup-rl-ckpt/step-500",
            task="grpo",
            has_optimizer=True,
            has_ref_model=False,
            has_rollout_buffer=False,
            soup_version="0.70.0",
        )
        assert st.step == 500
        assert st.task == "grpo"

    def test_frozen(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        st = RLCheckpointState(
            step=500,
            checkpoint_dir="./.soup-rl-ckpt/step-500",
            task="grpo",
            has_optimizer=True,
            has_ref_model=False,
            has_rollout_buffer=False,
            soup_version="0.70.0",
        )
        with pytest.raises(FrozenInstanceError):
            st.step = 0  # type: ignore[misc]

    def test_invalid_step_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        with pytest.raises(ValueError, match="step"):
            RLCheckpointState(
                step=-1,
                checkpoint_dir="./x",
                task="grpo",
                has_optimizer=True,
                has_ref_model=False,
                has_rollout_buffer=False,
                soup_version="0.70.0",
            )

    def test_invalid_task_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        with pytest.raises(ValueError, match="task"):
            RLCheckpointState(
                step=10,
                checkpoint_dir="./x",
                task="sft",  # not an RL task
                has_optimizer=True,
                has_ref_model=False,
                has_rollout_buffer=False,
                soup_version="0.70.0",
            )

    def test_bool_step_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        with pytest.raises(ValueError, match="bool"):
            RLCheckpointState(
                step=True,
                checkpoint_dir="./x",
                task="grpo",
                has_optimizer=True,
                has_ref_model=False,
                has_rollout_buffer=False,
                soup_version="0.70.0",
            )

    def test_to_dict(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        st = RLCheckpointState(
            step=500,
            checkpoint_dir="./.soup-rl-ckpt/step-500",
            task="grpo",
            has_optimizer=True,
            has_ref_model=True,
            has_rollout_buffer=False,
            soup_version="0.70.0",
        )
        d = st.to_dict()
        assert isinstance(d, dict)
        assert d["step"] == 500
        assert d["task"] == "grpo"
        assert d["has_optimizer"] is True
        assert d["has_ref_model"] is True

    def test_null_byte_dir_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        with pytest.raises(ValueError, match="null byte"):
            RLCheckpointState(
                step=10,
                checkpoint_dir="./bad\x00",
                task="grpo",
                has_optimizer=True,
                has_ref_model=False,
                has_rollout_buffer=False,
                soup_version="0.70.0",
            )

    def test_bool_has_optimizer_rejected(self):
        from soup_cli.utils.rl_checkpoint import RLCheckpointState

        with pytest.raises(TypeError, match="has_optimizer"):
            RLCheckpointState(
                step=10,
                checkpoint_dir="./x",
                task="grpo",
                has_optimizer="yes",  # type: ignore[arg-type]
                has_ref_model=False,
                has_rollout_buffer=False,
                soup_version="0.70.0",
            )


class TestBuildRLCheckpointCallback:
    """Live callback deferred to v0.70.1."""

    def test_non_config_rejected(self):
        from soup_cli.utils.rl_checkpoint import build_rl_checkpoint_callback

        with pytest.raises(TypeError, match="RLCheckpointConfig"):
            build_rl_checkpoint_callback({"save_every_steps": 100})  # type: ignore[arg-type]

    def test_deferred(self):
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        cfg = RLCheckpointConfig(save_every_steps=100)
        with pytest.raises(NotImplementedError, match="v0.70.1"):
            build_rl_checkpoint_callback(cfg)


# ---------------------------------------------------------------------------
# Schema integration — TrainingConfig + SoupConfig
# ---------------------------------------------------------------------------


class TestSchemaTrainingConfig:
    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.rl_checkpoint_save_every_steps is None
        assert tcfg.rl_checkpoint_keep_last == 3

    def test_set_save_every(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(rl_checkpoint_save_every_steps=500)
        assert tcfg.rl_checkpoint_save_every_steps == 500

    def test_zero_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(rl_checkpoint_save_every_steps=0)

    def test_keep_last_bounds(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(rl_checkpoint_keep_last=0)

        with pytest.raises(ValidationError):
            TrainingConfig(rl_checkpoint_keep_last=101)


class TestSchemaSoupConfigTaskGate:
    """rl_checkpoint_save_every_steps only meaningful on RL tasks
    (grpo/ppo)."""

    def _yaml(self, task: str, save_every: int = 500) -> str:
        return f"""
base: meta-llama/Llama-3.1-8B
task: {task}
data:
  train: ./data/train.jsonl
  format: chatml
training:
  rl_checkpoint_save_every_steps: {save_every}
"""

    def test_grpo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("grpo"))
        assert cfg.training.rl_checkpoint_save_every_steps == 500

    def test_ppo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("ppo"))
        assert cfg.training.rl_checkpoint_save_every_steps == 500

    def test_sft_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="rl_checkpoint"):
            load_config_from_string(self._yaml("sft"))


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli"
            / "utils"
            / "rl_checkpoint.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body
