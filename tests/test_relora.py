"""Tests for v0.39.0 Part B — ReLoRA callback (#693 paper-faithful restarts)."""

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
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.6)
        nonzero = (out != 0).sum().item()
        assert nonzero == 2
        assert (out.abs() == 2.0).any()
        assert (out.abs() == 1.0).any()

    def test_magnitude_prune_tie_keeps_exact_count(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        x = torch.ones(10)
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.9)
        assert (out != 0).sum().item() == 1

    def test_magnitude_prune_single_element_no_crash(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        x = torch.tensor([3.14])
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.5)
        assert out.item() == pytest.approx(3.14, abs=1e-5)

    def test_magnitude_prune_keep_all(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import magnitude_prune_tensor

        x = torch.tensor([1.0, 2.0, 3.0])
        out = magnitude_prune_tensor(x.clone(), prune_ratio=0.001)
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
        state = MagicMock(global_step=500, max_steps=1000)
        ctrl = MagicMock()
        cb.on_step_end(MagicMock(), state, ctrl)
        assert cb.fire_count == 0

    def test_callback_fires_on_relora_step(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100))
        cb._merge_reinit_and_reset = MagicMock()  # type: ignore[method-assign]
        state = MagicMock(global_step=100, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock(), model=MagicMock(), optimizer=MagicMock())
        assert cb.fire_count == 1
        cb._merge_reinit_and_reset.assert_called_once()

    def test_callback_does_not_fire_off_step(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100))
        cb._merge_reinit_and_reset = MagicMock()  # type: ignore[method-assign]
        state = MagicMock(global_step=99, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock())
        assert cb.fire_count == 0
        cb._merge_reinit_and_reset.assert_not_called()

    def test_callback_skips_warmup(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=100, warmup_ratio=0.5))
        cb._merge_reinit_and_reset = MagicMock()  # type: ignore[method-assign]
        state = MagicMock(global_step=100, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock(), model=MagicMock(), optimizer=MagicMock())
        assert cb.fire_count == 0


class FakeLoraModule:
    """Minimal LoRA module matching soup_cli.utils.relora iteration."""

    pass


def _make_fake_lora_module():
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")

    class _Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)
            self.base = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(x))

    return _Fake()


class TestReLoRARestart:
    def test_effective_map_preserved_after_restart(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(0.1)
            model.lora_B.weight.fill_(0.2)
            model.base.weight.fill_(1.0)
        before_base = model.base.weight.clone()
        x = torch.randn(3, 4)
        before_out = model(x).detach()

        _merge_reinit_and_reset(model, None, reset_optimizer=False)

        after_out = model(x).detach()
        mse = ((before_out - after_out) ** 2).mean()
        assert mse.item() < 1e-5
        assert not torch.equal(model.base.weight, before_base)

    def test_merge_false_mutation_large_mse(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(1.0)
            model.lora_B.weight.fill_(1.0)
            model.base.weight.fill_(1.0)
        x = torch.ones(3, 4)
        before_out = model(x).detach()

        _merge_reinit_and_reset(model, None, reset_optimizer=False, merge=False)

        after_out = model(x).detach()
        mse = ((before_out - after_out) ** 2).mean()
        assert mse.item() > 1e-2  # merge skipped => lost LoRA delta

    def test_successive_restarts_accumulate_in_base(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(0.1)
            model.lora_B.weight.fill_(0.2)
            model.base.weight.zero_()
        base0 = model.base.weight.clone()

        _merge_reinit_and_reset(model, None, reset_optimizer=False)
        base1 = model.base.weight.clone()
        delta1 = base1 - base0
        assert delta1.abs().sum() > 0

        with torch.no_grad():
            model.lora_A.weight.fill_(0.3)
            model.lora_B.weight.fill_(0.4)
        _merge_reinit_and_reset(model, None, reset_optimizer=False)
        base2 = model.base.weight.clone()
        delta2 = base2 - base1
        assert delta2.abs().sum() > 0
        assert (base2 - base0).abs().sum() > delta1.abs().sum()

    def test_restart_cannot_dead_branch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(0.5)
            model.lora_B.weight.fill_(0.5)
            model.base.weight.fill_(1.0)

        _merge_reinit_and_reset(model, None, reset_optimizer=False)

        assert not torch.all(model.lora_A.weight == 0)
        x = torch.randn(2, 4)
        out = model(x).sum()
        out.backward()
        assert model.lora_B.weight.grad is not None
        assert model.lora_B.weight.grad.abs().sum().item() > 0


class TestMergeReinitAndReset:
    def test_merge_reinit_targets_lora_modules(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(0.1)
            model.lora_B.weight.fill_(0.2)
            model.base.weight.fill_(7.0)
        before_base = model.base.weight.clone()

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=10, prune_ratio=0.9))
        cb._merge_reinit_and_reset(model, None)

        assert not torch.equal(model.base.weight, before_base)
        assert not torch.all(model.lora_A.weight == 0)
        assert torch.all(model.lora_B.weight == 0)

    def test_merge_reinit_clears_real_optimizer_state(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        model = _make_fake_lora_module()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        params = [model.lora_A.weight, model.lora_B.weight]
        for param in params:
            assert param in opt.state
            assert len(opt.state[param]) > 0

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=10, prune_ratio=0.9))
        cb._merge_reinit_and_reset(model, opt)
        for param in params:
            assert len(opt.state[param]) == 0


    def test_optimizer_reset_rejected_before_merge(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_fake_lora_module()
        with torch.no_grad():
            model.lora_A.weight.fill_(0.1)
            model.lora_B.weight.fill_(0.2)
            model.base.weight.fill_(1.0)
        before_base = model.base.weight.clone()

        optimizer = MagicMock()
        del optimizer.state

        with pytest.raises(RuntimeError, match="optimizer has no"):
            _merge_reinit_and_reset(model, optimizer, reset_optimizer=True)

        assert torch.equal(model.base.weight, before_base)

    def test_merge_reinit_respects_reset_optimizer_false(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        model = _make_fake_lora_module()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        param = model.lora_A.weight
        before = len(opt.state[param])

        cb = ReLoRACallback(
            policy=ReLoRAPolicy(steps=10, prune_ratio=0.9, reset_optimizer=False)
        )
        cb._merge_reinit_and_reset(model, opt)
        assert len(opt.state[param]) == before


class TestReLoRATaskGate:
    def _base_cfg(self, task: str = "sft", backend: str = "transformers") -> dict:
        return {
            "base": "meta-llama/Llama-3.1-8B",
            "task": task,
            "backend": backend,
            "data": {"train": "./data.jsonl"},
            "training": {"relora_steps": 100, "quantization": "none"},
        }

    def test_sft_accepted(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(yaml.safe_dump(self._base_cfg("sft")))
        assert cfg.training.relora_steps == 100

    @pytest.mark.parametrize(
        "task",
        [
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
    def test_other_tasks_accepted(self, task):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(yaml.safe_dump(self._base_cfg(task)))
        assert cfg.training.relora_steps == 100
        assert cfg.task == task

    def test_mlx_backend_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(yaml.safe_dump(self._base_cfg("sft", "mlx")))

    def test_quantization_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg()
        d["training"]["quantization"] = "4bit"
        with pytest.raises(ValueError, match="quantization"):
            load_config_from_string(yaml.safe_dump(d))

    def test_stream_layers_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg()
        d["training"]["stream_layers"] = True
        with pytest.raises(ValueError, match="stream_layers"):
            load_config_from_string(yaml.safe_dump(d))

    def test_vera_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg()
        d["training"] = {"relora_steps": 100, "quantization": "none", "lora": {"use_vera": True}}
        with pytest.raises(ValueError, match="use_vera"):
            load_config_from_string(yaml.safe_dump(d))

    def test_dora_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg()
        d["training"] = {"relora_steps": 100, "quantization": "none", "lora": {"use_dora": True}}
        with pytest.raises(ValueError, match="use_dora"):
            load_config_from_string(yaml.safe_dump(d))

    def test_fsdp2_compile_rejected(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg()
        d["training"]["use_fsdp2_compile"] = True
        with pytest.raises(ValueError, match="use_fsdp2_compile"):
            load_config_from_string(yaml.safe_dump(d))

    def test_no_relora_no_gate(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        d = self._base_cfg("dpo")
        d["training"].pop("relora_steps")
        cfg = load_config_from_string(yaml.safe_dump(d))
        assert cfg.training.relora_steps is None
