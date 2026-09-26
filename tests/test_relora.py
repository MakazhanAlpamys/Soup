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

    def test_lr_warmup_via_param_groups(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        class _Opt:
            def __init__(self, lr):
                self.param_groups = [{"lr": lr}]
                self.state = {}

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=20))
        opt = _Opt(0.01)
        cb._start_lr_warmup(opt)
        assert opt.param_groups[0]["lr"] == 0.0
        assert cb._warmup_target_lrs == [0.01]
        cb._maybe_advance_lr_warmup(opt)
        assert opt.param_groups[0]["lr"] == 0.005
        cb._maybe_advance_lr_warmup(opt)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_lr_warmup_advances_on_next_step_after_fire(self):
        from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

        class _Opt:
            def __init__(self, lr):
                self.param_groups = [{"lr": lr}]
                self.state = {}

        cb = ReLoRACallback(policy=ReLoRAPolicy(steps=10, warmup_ratio=0.0))
        model = _make_fake_lora_module()
        opt = _Opt(1e-3)
        state = MagicMock(global_step=10, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock(), model=model, optimizer=opt)
        assert cb.fire_count == 1
        assert opt.param_groups[0]["lr"] == 0.0
        state = MagicMock(global_step=11, max_steps=1000)
        cb.on_step_end(MagicMock(), state, MagicMock(), model=model, optimizer=opt)
        assert cb.fire_count == 1
        assert opt.param_groups[0]["lr"] == 1e-3


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


def _make_quantized_lora_module():
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")

    class _QuantWeight:
        quant_state = object()

    class _Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)

            class _Base:
                weight = _QuantWeight()

            self.base = _Base()

    return _Fake()


def _make_mixed_writable_quantized_parent():
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")

    class _Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _make_fake_lora_module()
            self.second = _make_quantized_lora_module()

    return _Parent()


def _single_process_trainer():
    trainer = MagicMock()
    trainer.args.world_size = 1
    trainer.is_deepspeed_enabled = False
    trainer.is_fsdp_enabled = False
    return trainer


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

    def test_skips_embedding_lora_modules(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        class _EmbedLora(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({})
                self.lora_B = nn.ModuleDict({})
                self.lora_embedding_A = nn.Parameter(torch.ones(2, 4))
                self.lora_embedding_B = nn.Parameter(torch.ones(4, 2))

        class _Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = _EmbedLora()
                self.linear = _make_fake_lora_module()

        model = _Parent()
        with torch.no_grad():
            model.linear.lora_A.weight.fill_(0.1)
            model.linear.lora_B.weight.fill_(0.2)
            model.linear.base.weight.fill_(1.0)
        embed_a_before = model.embed.lora_embedding_A.detach().clone()
        embed_b_before = model.embed.lora_embedding_B.detach().clone()
        linear_base_before = model.linear.base.weight.detach().clone()

        _merge_reinit_and_reset(model, None, reset_optimizer=False)

        assert torch.equal(model.embed.lora_embedding_A, embed_a_before)
        assert torch.equal(model.embed.lora_embedding_B, embed_b_before)
        assert not torch.equal(model.linear.base.weight, linear_base_before)

    def test_iter_yields_linear_with_empty_embedding_dicts(self):
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _iter_lora_modules

        class _LinearWithEmptyEmbedDicts(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
                self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
                self.lora_embedding_A = nn.ParameterDict()
                self.lora_embedding_B = nn.ParameterDict()
                self.base_layer = nn.Linear(4, 4, bias=False)

        module = _LinearWithEmptyEmbedDicts()
        yielded = list(_iter_lora_modules(module))
        assert len(yielded) == 1
        assert yielded[0][0] is module

    def test_restart_refuses_partial_when_later_module_quantized(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from soup_cli.utils.relora import _merge_reinit_and_reset

        model = _make_mixed_writable_quantized_parent()
        with torch.no_grad():
            model.first.lora_A.weight.fill_(0.1)
            model.first.lora_B.weight.fill_(0.2)
            model.first.base.weight.fill_(1.0)
        before_base = model.first.base.weight.detach().clone()
        before_b = model.first.lora_B.weight.detach().clone()

        with pytest.raises(RuntimeError, match="quantized or sharded"):
            _merge_reinit_and_reset(model, None, reset_optimizer=False)

        assert torch.equal(model.first.base.weight, before_base)
        assert torch.equal(model.first.lora_B.weight, before_b)

    def test_wrapped_optimizer_rejected_even_when_reset_false(self):
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
            _merge_reinit_and_reset(model, optimizer, reset_optimizer=False)

        assert torch.equal(model.base.weight, before_base)


def test_attach_relora_callback_warns_prune_ratio_ignored(caplog):
    import logging

    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    tcfg = TrainingConfig(relora_steps=100, relora_prune_ratio=0.9)
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.peft_wiring"):
        attach_relora_callback(trainer, tcfg)
    assert "relora_prune_ratio" in caplog.text
    assert "ignored" in caplog.text.lower()


def test_attach_relora_callback_default_prune_ratio_does_not_warn(caplog):
    import logging

    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    tcfg = TrainingConfig(relora_steps=100)
    assert "relora_prune_ratio" not in tcfg.model_fields_set
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.peft_wiring"):
        attach_relora_callback(trainer, tcfg)
    assert "relora_prune_ratio" not in caplog.text


def test_attach_relora_refuses_world_size_gt_1():
    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    trainer.args.world_size = 2
    with pytest.raises(ValueError, match="world_size"):
        attach_relora_callback(trainer, TrainingConfig(relora_steps=100))
    trainer.add_callback.assert_not_called()


def test_attach_relora_refuses_deepspeed():
    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    trainer.is_deepspeed_enabled = True
    with pytest.raises(ValueError, match="DeepSpeed"):
        attach_relora_callback(trainer, TrainingConfig(relora_steps=100))
    trainer.add_callback.assert_not_called()


def test_attach_relora_refuses_fsdp():
    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    trainer.is_fsdp_enabled = True
    with pytest.raises(ValueError, match="FSDP"):
        attach_relora_callback(trainer, TrainingConfig(relora_steps=100))
    trainer.add_callback.assert_not_called()


def test_attach_relora_single_process_still_attaches():
    from soup_cli.utils.peft_wiring import attach_relora_callback
    from soup_cli.utils.relora import ReLoRACallback

    trainer = _single_process_trainer()
    assert attach_relora_callback(trainer, TrainingConfig(relora_steps=100)) is True
    trainer.add_callback.assert_called_once()
    assert isinstance(trainer.add_callback.call_args[0][0], ReLoRACallback)


def test_attach_relora_preflight_rejects_quantized_base():
    from soup_cli.utils.peft_wiring import attach_relora_callback

    trainer = _single_process_trainer()
    trainer.model = _make_mixed_writable_quantized_parent()
    with pytest.raises(RuntimeError, match="quantized or sharded"):
        attach_relora_callback(trainer, TrainingConfig(relora_steps=100))
    trainer.add_callback.assert_not_called()


class TestReLoRARealPeft:
    def test_restart_merges_real_peft_linear_skips_embedding(self):
        try:
            import torch
            import torch.nn as nn
            from peft import LoraConfig, get_peft_model
            from transformers import LlamaConfig, LlamaForCausalLM
        except (ImportError, OSError) as exc:
            pytest.skip(f"torch / transformers / peft not available: {exc}")
        from soup_cli.utils.relora import (
            _iter_lora_modules,
            _merge_reinit_and_reset,
            _resolve_base_weight,
            _resolve_lora_weight,
        )

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        try:
            model = LlamaForCausalLM(config).to(torch.float32).eval()
            peft_cfg = LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "embed_tokens"],
            )
            model = get_peft_model(model, peft_cfg)
            model.eval()
        except (ImportError, OSError) as exc:
            pytest.skip(f"torch / transformers / peft not available: {exc}")

        yielded = list(_iter_lora_modules(model))
        assert len(yielded) == 2

        for module, adapter in yielded:
            weight_a = _resolve_lora_weight(module, "lora_A", adapter)
            weight_b = _resolve_lora_weight(module, "lora_B", adapter)
            with torch.no_grad():
                weight_a.fill_(0.25)
                weight_b.fill_(0.5)

        embed_before = {}
        for name, module in model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            if not isinstance(lora_a, nn.ModuleDict) or len(lora_a) != 0:
                continue
            if not hasattr(module, "lora_embedding_A"):
                continue
            with torch.no_grad():
                for key, param in module.lora_embedding_A.items():
                    param.fill_(0.3)
                    embed_before[(name, "A", key)] = param.detach().clone()
                for key, param in module.lora_embedding_B.items():
                    param.fill_(0.4)
                    embed_before[(name, "B", key)] = param.detach().clone()
        assert embed_before

        snapshots = []
        lora_params = []
        for module, adapter in yielded:
            weight_a = _resolve_lora_weight(module, "lora_A", adapter)
            weight_b = _resolve_lora_weight(module, "lora_B", adapter)
            base = _resolve_base_weight(module)
            delta = module.get_delta_weight(adapter)
            snapshots.append(
                (
                    module,
                    adapter,
                    base.detach().clone(),
                    delta.detach().clone(),
                    weight_a.detach().clone(),
                )
            )
            lora_params.extend([weight_a, weight_b])

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits

        opt = torch.optim.AdamW(lora_params, lr=1e-3)
        for param in lora_params:
            opt.state[param] = {
                "exp_avg": torch.ones_like(param),
                "exp_avg_sq": torch.ones_like(param),
            }

        _merge_reinit_and_reset(model, opt, reset_optimizer=True)

        for module, adapter, old_base, delta, old_a in snapshots:
            base = _resolve_base_weight(module)
            weight_a = _resolve_lora_weight(module, "lora_A", adapter)
            weight_b = _resolve_lora_weight(module, "lora_B", adapter)
            assert torch.allclose(base, old_base + delta, atol=1e-5, rtol=1e-5)
            assert torch.all(weight_b == 0)
            assert not torch.equal(weight_a, old_a)
        for param in lora_params:
            assert len(opt.state[param]) == 0

        with torch.no_grad():
            after_logits = model(input_ids=input_ids).logits
        assert torch.allclose(before_logits, after_logits, atol=1e-5, rtol=1e-5)

        for name, module in model.named_modules():
            for key, param in getattr(module, "lora_embedding_A", {}).items():
                assert torch.equal(param, embed_before[(name, "A", key)])
            for key, param in getattr(module, "lora_embedding_B", {}).items():
                assert torch.equal(param, embed_before[(name, "B", key)])


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
        d["training"]["batch_size"] = 1
        with pytest.raises(
            ValueError,
            match="relora_steps is incompatible with training.stream_layers",
        ):
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
