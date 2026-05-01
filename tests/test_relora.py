"""Tests for v0.39.0 Part B — ReLoRA callback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TrainingConfig


class TestReLoRASchema:
    def test_default_disabled(self):
        cfg = TrainingConfig()
        assert cfg.relora_steps is None
        assert cfg.relora_warmup_ratio == 0.1
        assert cfg.relora_reset_optimizer is True
        assert 0.0 < cfg.relora_prune_ratio <= 1.0

    def test_relora_steps_positive(self):
        cfg = TrainingConfig(relora_steps=500)
        assert cfg.relora_steps == 500

    def test_relora_steps_rejects_zero(self):
        with pytest.raises(ValidationError):
            TrainingConfig(relora_steps=0)

    def test_relora_steps_rejects_negative(self):
        with pytest.raises(ValidationError):
            TrainingConfig(relora_steps=-1)

    def test_relora_steps_upper_bound(self):
        # Cap at 10**7 to prevent overflow / nonsensical values
        with pytest.raises(ValidationError):
            TrainingConfig(relora_steps=10**8)

    def test_relora_warmup_ratio_bounds(self):
        TrainingConfig(relora_warmup_ratio=0.0)
        TrainingConfig(relora_warmup_ratio=1.0)
        with pytest.raises(ValidationError):
            TrainingConfig(relora_warmup_ratio=-0.01)
        with pytest.raises(ValidationError):
            TrainingConfig(relora_warmup_ratio=1.01)

    def test_relora_prune_ratio_bounds(self):
        TrainingConfig(relora_prune_ratio=0.5)
        TrainingConfig(relora_prune_ratio=0.99)
        with pytest.raises(ValidationError):
            TrainingConfig(relora_prune_ratio=0.0)
        with pytest.raises(ValidationError):
            TrainingConfig(relora_prune_ratio=1.0)
        with pytest.raises(ValidationError):
            TrainingConfig(relora_prune_ratio=1.01)


class TestReLoRAPolicy:
    def test_policy_frozen(self):
        from dataclasses import FrozenInstanceError

        from soup_cli.utils.relora import ReLoRAPolicy
        p = ReLoRAPolicy(steps=500, warmup_ratio=0.1, reset_optimizer=True, prune_ratio=0.9)
        with pytest.raises(FrozenInstanceError):
            p.steps = 999  # type: ignore

    def test_policy_should_fire_step_zero_no(self):
        from soup_cli.utils.relora import ReLoRAPolicy
        p = ReLoRAPolicy(steps=500)
        assert p.should_fire(global_step=0) is False

    def test_policy_should_fire_at_multiple(self):
        from soup_cli.utils.relora import ReLoRAPolicy
        p = ReLoRAPolicy(steps=500)
        assert p.should_fire(global_step=500) is True
        assert p.should_fire(global_step=1000) is True

    def test_policy_should_fire_skips_warmup(self):
        from soup_cli.utils.relora import ReLoRAPolicy
        # warmup_ratio=0.5 over total 1000 = first 500 skipped
        p = ReLoRAPolicy(steps=200, warmup_ratio=0.5)
        assert p.should_fire(global_step=200, total_steps=1000) is False
        assert p.should_fire(global_step=600, total_steps=1000) is True

    def test_policy_rejects_invalid_steps(self):
        from soup_cli.utils.relora import ReLoRAPolicy
        with pytest.raises(ValueError):
            ReLoRAPolicy(steps=0)
        with pytest.raises(ValueError):
            ReLoRAPolicy(steps=-1)

    def test_policy_rejects_invalid_prune_ratio(self):
        from soup_cli.utils.relora import ReLoRAPolicy
        with pytest.raises(ValueError):
            ReLoRAPolicy(steps=500, prune_ratio=0.0)
        with pytest.raises(ValueError):
            ReLoRAPolicy(steps=500, prune_ratio=1.0)
        with pytest.raises(ValueError):
            ReLoRAPolicy(steps=500, prune_ratio=1.5)


class TestMagnitudePrune:
    def test_magnitude_prune_zeroes_low_magnitude(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        x = torch.tensor([0.01, 0.02, 0.5, 1.0, 2.0])
        # prune_ratio=0.6 → keep top 40% (2 of 5) → smallest 3 zeroed
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.6)
        nonzero = (out != 0).sum().item()
        assert nonzero == 2
        # the two largest must survive
        assert (out.abs() == 2.0).any()
        assert (out.abs() == 1.0).any()

    def test_magnitude_prune_single_element_no_crash(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        # 1-element tensor — kthvalue(_, 0) would raise; helper must short-circuit.
        x = torch.tensor([3.14])
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.5)
        # untouched (use approx for float32 storage)
        assert out.item() == pytest.approx(3.14, abs=1e-5)

    def test_magnitude_prune_keep_all(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        x = torch.tensor([1.0, 2.0, 3.0])
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.001)
        # near-zero prune ratio → at least one element kept
        assert (out != 0).any()

    def test_magnitude_prune_rejects_invalid_ratio(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor
        x = torch.zeros(3)
        with pytest.raises(ValueError):
            magnitude_prune_tensor(x, prune_ratio=0.0)
        with pytest.raises(ValueError):
            magnitude_prune_tensor(x, prune_ratio=1.0)

    def test_magnitude_prune_rejects_non_tensor_input(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor
        with pytest.raises(TypeError):
            magnitude_prune_tensor([1.0, 2.0, 3.0], prune_ratio=0.5)
        with pytest.raises(TypeError):
            magnitude_prune_tensor("not a tensor", prune_ratio=0.5)


class TestReLoRACallback:
    def test_callback_disabled_no_op(self):
        from soup_cli.utils.relora import ReLoRACallback
        cb = ReLoRACallback(policy=None)
        # disabled callback never fires
        state = MagicMock(global_step=500, max_steps=1000)
        ctrl = MagicMock()
        args = MagicMock()
        cb.on_step_end(args, state, ctrl)
        assert cb.fire_count == 0

    def test_callback_fires_on_relora_step(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy
        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100))
        # mock model with PEFT-style lora_A / lora_B parameters
        cb._prune_and_reset = MagicMock()  # type: ignore[method-assign]
        state = MagicMock(global_step=100, max_steps=1000)
        ctrl = MagicMock()
        args = MagicMock()
        cb.on_step_end(args, state, ctrl, model=MagicMock(), optimizer=MagicMock())
        assert cb.fire_count == 1
        cb._prune_and_reset.assert_called_once()

    def test_callback_does_not_fire_off_step(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy
        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100))
        cb._prune_and_reset = MagicMock()  # type: ignore[method-assign]
        state = MagicMock(global_step=99, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock())
        assert cb.fire_count == 0
        cb._prune_and_reset.assert_not_called()

    def test_callback_skips_warmup(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy
        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100, warmup_ratio=0.5))
        cb._prune_and_reset = MagicMock()  # type: ignore[method-assign]
        # at step 100 with total 1000 → warmup is 500 → skip
        state = MagicMock(global_step=100, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock(), model=MagicMock(), optimizer=MagicMock())
        assert cb.fire_count == 0


class TestReLoRATaskGate:
    def _base_cfg(self, task: str = "sft", backend: str = "transformers") -> dict:
        return {
            "base": "meta-llama/Llama-3.1-8B",
            "task": task,
            "backend": backend,
            "data": {"train": "./data.jsonl"},
            "training": {"relora_steps": 100},
        }

    def test_sft_accepted(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string
        cfg = load_config_from_string(yaml.safe_dump(self._base_cfg("sft")))
        assert cfg.training.relora_steps == 100

    @pytest.mark.parametrize(
        "task",
        ["dpo", "grpo", "kto", "orpo", "simpo", "ipo",
         "ppo", "reward_model", "pretrain", "embedding"],
    )
    def test_other_tasks_rejected(self, task):
        import yaml

        from soup_cli.config.loader import load_config_from_string
        # load_config_from_string surfaces a ValueError (Pydantic validation
        # error) that wraps the cross-validator's message.
        with pytest.raises(ValueError, match="relora_steps"):
            load_config_from_string(yaml.safe_dump(self._base_cfg(task)))

    def test_mlx_backend_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string
        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(yaml.safe_dump(self._base_cfg("sft", "mlx")))

    def test_no_relora_no_gate(self):
        # multi-task without relora_steps stays valid
        import yaml

        from soup_cli.config.loader import load_config_from_string
        d = self._base_cfg("dpo")
        d["training"].pop("relora_steps")
        cfg = load_config_from_string(yaml.safe_dump(d))
        assert cfg.training.relora_steps is None


class TestPruneAndReset:
    def test_prune_and_reset_targets_lora_modules(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        class FakeLoraModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Linear(4, 2, bias=False)
                self.lora_B = nn.Linear(2, 4, bias=False)
                self.base = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.base(x) + self.lora_B(self.lora_A(x))

        model = FakeLoraModule()
        # set known weight values
        with torch.no_grad():
            model.lora_A.weight.fill_(1.0)
            model.lora_A.weight[0, 0] = 100.0
            model.lora_B.weight.fill_(0.5)
            model.base.weight.fill_(7.0)
        before_base = model.base.weight.clone()

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=10, prune_ratio=0.9))
        opt = MagicMock()
        opt.state = {}
        cb._prune_and_reset(model, opt)

        # base must be untouched
        assert torch.equal(model.base.weight, before_base)
        # lora_A retains highest-magnitude entry
        assert (model.lora_A.weight.abs() >= 100.0).any()
        # lora_A overall has many zeros now (prune_ratio=0.9)
        zero_frac = (model.lora_A.weight == 0).float().mean().item()
        assert zero_frac > 0.5

    def test_prune_and_reset_clears_real_optimizer_state(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        class FakeLoraModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.lora_A(x)

        model = FakeLoraModule()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Run a step so opt.state is populated for the param.
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        param = model.lora_A.weight
        assert param in opt.state
        assert len(opt.state[param]) > 0  # exp_avg, exp_avg_sq, step

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=10, prune_ratio=0.9))
        cb._prune_and_reset(model, opt)
        # Optimizer state for the pruned param must have been reset to empty
        assert len(opt.state[param]) == 0

    def test_prune_and_reset_respects_reset_optimizer_false(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        class FakeLoraModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.lora_A(x)

        model = FakeLoraModule()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        param = model.lora_A.weight
        before = len(opt.state[param])

        cb = ReLoRACallback(
            policy=ReLoRAPolicy(steps=10, prune_ratio=0.9, reset_optimizer=False)
        )
        cb._prune_and_reset(model, opt)
        # State preserved when reset_optimizer=False
        assert len(opt.state[param]) == before
