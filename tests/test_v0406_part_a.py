"""v0.40.6 Part A — ReLoRA + surgical PEFT non-SFT (#67).

Source-level proofs that all 11 non-SFT trainers wire:
  * `apply_gemma4_clippable_patch` (pre-LoRA, gated by `is_gemma4_model`)
  * `strip_lora_dropout_for_3d_experts` (post-LoRA)
  * `ReLoRACallback` (when `relora_steps` is set)

Plus schema-gate widening: every transformer-backend non-SFT task now
accepts `relora_steps`; MLX backend still rejected with distinct message.

Source-level grep mirrors the v0.40.4 #63 / #65 pattern — a parametrized
matrix proves wiring landed in every trainer file, without booting heavy
deps (torch, transformers, peft, trl) inside CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NON_SFT_TRAINERS = [
    "dpo",
    "grpo",
    "kto",
    "orpo",
    "simpo",
    "ipo",
    "ppo",
    "reward_model",
    "pretrain",
    "embedding",
    "bco",
]

# v0.40.6 review fix — SFT is migrated to the shared helpers as well, so the
# centralisation invariant covers ALL transformer-backend trainers (no drift).
ALL_TRANSFORMER_TRAINERS = ["sft", *NON_SFT_TRAINERS]


def _trainer_source(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer" / f"{name}.py"
    assert path.is_file(), f"missing trainer source: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("trainer", ALL_TRANSFORMER_TRAINERS)
def test_trainer_calls_pre_and_post_lora_helpers(trainer: str) -> None:
    """v0.39.0 Part D surgical patches wired via shared `peft_wiring` helpers
    in every non-SFT trainer (v0.40.6 #67)."""
    src = _trainer_source(trainer)
    assert "apply_pre_lora_patches(" in src, (
        f"{trainer}.py must call peft_wiring.apply_pre_lora_patches"
    )
    assert "apply_post_lora_patches(" in src, (
        f"{trainer}.py must call peft_wiring.apply_post_lora_patches"
    )


@pytest.mark.parametrize("trainer", ALL_TRANSFORMER_TRAINERS)
def test_trainer_attaches_relora_callback(trainer: str) -> None:
    """v0.39.0 Part B `ReLoRACallback` attached via shared helper."""
    src = _trainer_source(trainer)
    assert "attach_relora_callback(" in src, (
        f"{trainer}.py must call peft_wiring.attach_relora_callback"
    )


@pytest.mark.parametrize("trainer", ALL_TRANSFORMER_TRAINERS)
def test_surgical_patches_pre_and_post_lora(trainer: str) -> None:
    """The pre-LoRA helper must run BEFORE `get_peft_model` (so PEFT's
    matcher sees the swapped `nn.Linear`); the post-LoRA helper must run
    AFTER.
    """
    src = _trainer_source(trainer)
    pre_idx = src.find("apply_pre_lora_patches(")
    peft_idx = src.find("get_peft_model(")
    post_idx = src.find("apply_post_lora_patches(")
    assert pre_idx != -1 and peft_idx != -1 and post_idx != -1, (
        f"{trainer}.py must call apply_pre_lora_patches, get_peft_model, "
        "and apply_post_lora_patches"
    )
    assert pre_idx < peft_idx < post_idx, (
        f"{trainer}.py: surgical patches in wrong order — pre-LoRA must run "
        "before get_peft_model, post-LoRA must run after."
    )


def test_peft_wiring_helpers_are_importable() -> None:
    """The shared helpers must be importable without instantiating heavy deps."""
    from soup_cli.utils.peft_wiring import (
        apply_post_lora_patches,
        apply_pre_lora_patches,
        attach_relora_callback,
    )

    assert callable(apply_pre_lora_patches)
    assert callable(apply_post_lora_patches)
    assert callable(attach_relora_callback)


def test_apply_pre_lora_patches_skips_non_gemma4() -> None:
    """Non-Gemma4 model name short-circuits without touching the model."""
    from unittest.mock import MagicMock

    from soup_cli.utils.peft_wiring import apply_pre_lora_patches

    sentinel = MagicMock()
    apply_pre_lora_patches(sentinel, "meta-llama/Llama-3.1-8B")
    # Sentinel must NOT have been mutated — the gate kept us out of the model tree.
    assert not sentinel.method_calls


def test_attach_relora_callback_returns_false_when_disabled() -> None:
    """No `relora_steps` set -> helper returns False, no add_callback fired."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = MagicMock()
    tcfg = SimpleNamespace(relora_steps=None)
    assert attach_relora_callback(trainer, tcfg) is False
    trainer.add_callback.assert_not_called()


def test_attach_relora_callback_attaches_when_enabled() -> None:
    """relora_steps > 0 -> helper attaches a ReLoRACallback exactly once."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from soup_cli.utils.peft_wiring import attach_relora_callback
    from soup_cli.utils.relora import ReLoRACallback

    trainer = MagicMock()
    tcfg = SimpleNamespace(
        relora_steps=100,
        relora_warmup_ratio=0.1,
        relora_reset_optimizer=True,
        relora_prune_ratio=0.5,
    )
    assert attach_relora_callback(trainer, tcfg) is True
    trainer.add_callback.assert_called_once()
    cb = trainer.add_callback.call_args[0][0]
    assert isinstance(cb, ReLoRACallback)


def test_attach_relora_callback_forwards_policy_fields() -> None:
    """Review-fix: assert each TrainingConfig field reaches the policy.

    Without this, a future refactor that hardcoded a constant default would
    pass the existing ``isinstance`` check.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = MagicMock()
    tcfg = SimpleNamespace(
        relora_steps=250,
        relora_warmup_ratio=0.37,
        relora_reset_optimizer=False,
        relora_prune_ratio=0.42,
    )
    attach_relora_callback(trainer, tcfg)
    cb = trainer.add_callback.call_args[0][0]
    assert cb.policy.steps == 250
    assert cb.policy.warmup_ratio == pytest.approx(0.37)
    assert cb.policy.reset_optimizer is False
    assert cb.policy.prune_ratio == pytest.approx(0.42)


def test_apply_pre_lora_patches_runs_on_gemma4(monkeypatch) -> None:
    """Gemma4 model name -> patch is invoked exactly once."""
    from unittest.mock import MagicMock

    from soup_cli.utils import peft_wiring

    fake_patcher = MagicMock(return_value=1)
    # Patch the symbol resolved inside the helper's lazy import. We do this
    # via the source module, which `peft_wiring` imports lazily by name.
    monkeypatch.setattr(
        "soup_cli.utils.peft_patches.apply_gemma4_clippable_patch", fake_patcher
    )
    model = MagicMock()
    peft_wiring.apply_pre_lora_patches(model, "google/gemma-4-2b-it")
    fake_patcher.assert_called_once_with(model)


def test_apply_pre_lora_patches_swallows_exceptions(monkeypatch) -> None:
    """A failed Gemma4 swap must NOT propagate to training."""
    from unittest.mock import MagicMock

    from soup_cli.utils import peft_wiring

    def _boom(_model):
        raise RuntimeError("synthetic patch failure")

    monkeypatch.setattr(
        "soup_cli.utils.peft_patches.apply_gemma4_clippable_patch", _boom
    )
    # Must not raise.
    peft_wiring.apply_pre_lora_patches(MagicMock(), "google/gemma-4-9b")


def test_apply_post_lora_patches_invokes_strip(monkeypatch) -> None:
    """Happy path: helper calls `strip_lora_dropout_for_3d_experts` once."""
    from unittest.mock import MagicMock

    from soup_cli.utils import peft_wiring

    fake_strip = MagicMock(return_value=0)
    monkeypatch.setattr(
        "soup_cli.utils.peft_patches.strip_lora_dropout_for_3d_experts",
        fake_strip,
    )
    model = MagicMock()
    peft_wiring.apply_post_lora_patches(model)
    fake_strip.assert_called_once_with(model)


def test_apply_post_lora_patches_swallows_exceptions(monkeypatch) -> None:
    """A failed dropout strip must NOT propagate to training."""
    from unittest.mock import MagicMock

    from soup_cli.utils import peft_wiring

    def _boom(_model):
        raise RuntimeError("synthetic strip failure")

    monkeypatch.setattr(
        "soup_cli.utils.peft_patches.strip_lora_dropout_for_3d_experts", _boom
    )
    peft_wiring.apply_post_lora_patches(MagicMock())


def test_attach_relora_callback_rejects_zero_via_policy() -> None:
    """Review-fix: schema-bypassing caller passing relora_steps=0 surfaces
    as a loud `ReLoRAPolicy` ValueError, NOT a silent skip.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = MagicMock()
    tcfg = SimpleNamespace(
        relora_steps=0,
        relora_warmup_ratio=0.0,
        relora_reset_optimizer=True,
        relora_prune_ratio=0.5,
    )
    with pytest.raises((ValueError, TypeError)):
        attach_relora_callback(trainer, tcfg)


class TestPreferenceTaskGate:
    """Review-fix: cover the dispatcher task ('preference') with `relora_steps`."""

    def test_preference_task_accepts_relora(self) -> None:
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(yaml.safe_dump({
            "base": "meta-llama/Llama-3.1-8B",
            "task": "preference",
            "backend": "transformers",
            "data": {"train": "./data.jsonl"},
            "training": {
                "relora_steps": 100,
                "preference_loss": "dpo",
            },
        }))
        assert cfg.training.relora_steps == 100
        assert cfg.task == "preference"


class TestReLoRASchemaGateWidened:
    """v0.40.6 lifts the `task != 'sft'` rejection. Every transformer-backend
    task now accepts `relora_steps`; MLX backend still rejected.
    """

    def _base_cfg(
        self, task: str = "sft", backend: str = "transformers"
    ) -> dict:
        return {
            "base": "meta-llama/Llama-3.1-8B",
            "task": task,
            "backend": backend,
            "data": {"train": "./data.jsonl"},
            "training": {"relora_steps": 100},
        }

    @pytest.mark.parametrize(
        "task",
        [
            "sft",
            "dpo",
            "grpo",
            "kto",
            "orpo",
            "simpo",
            "ipo",
            "ppo",
            "reward_model",
            "pretrain",
            "embedding",
            "bco",
        ],
    )
    def test_all_transformer_tasks_accepted(self, task: str) -> None:
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(yaml.safe_dump(self._base_cfg(task)))
        assert cfg.training.relora_steps == 100
        assert cfg.task == task

    def test_mlx_backend_still_rejected(self) -> None:
        import yaml

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(yaml.safe_dump(self._base_cfg("sft", "mlx")))
