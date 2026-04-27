"""Part E — v0.32.1 stability live (#56, #57, #58, #59) for v0.33.0.

Covers:
  - #56 run_lr_sweep — in-process LR-sweep loop with mocked model + DataLoader.
  - #57 SoupTrainerCallback._write_spike_recovery_hint — writes JSON hint
    when watchdog fires and loss_spike_recovery is enabled.
  - #58 SFTTrainerWrapper._resolve_mixed_precision — wires
    pick_mixed_precision into bf16/fp16 flags; preserves legacy default
    when auto flag is False.
  - #59 SoupTrainerCallback grad-accum advisory — fires once on threshold
    crossing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# #56 — run_lr_sweep
# ---------------------------------------------------------------------------


class TestRunLRSweep:
    def test_empty_schedule_rejected(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        with pytest.raises(ValueError, match="schedule must be non-empty"):
            run_lr_sweep(
                model=MagicMock(), dataloader=iter([]),
                schedule=[], optimizer_factory=lambda p: MagicMock(),
            )

    def test_loop_records_loss_per_step(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        # Fake model returning a tensor-like loss
        def _fake_loss_value(value):
            obj = MagicMock()
            obj.detach = MagicMock(return_value=obj)
            obj.item = MagicMock(return_value=value)
            obj.backward = MagicMock(return_value=None)
            return obj

        loss_values = [3.0, 2.0, 1.5, 1.0]

        class FakeModel:
            def __init__(self):
                self._idx = 0

            def parameters(self):
                return []

            def __call__(self, **batch):
                value = loss_values[self._idx]
                self._idx += 1
                return {"loss": _fake_loss_value(value)}

        model = FakeModel()

        # Fake optimizer with mutable param_groups
        class FakeOptim:
            def __init__(self, _params):
                self.param_groups = [{"lr": 0.0}]

            def zero_grad(self, set_to_none: bool = False):  # noqa: ARG002
                pass

            def step(self):
                pass

        dl = iter([{"input_ids": MagicMock()}] * 4)
        schedule = [1e-6, 1e-5, 1e-4, 1e-3]

        losses = run_lr_sweep(
            model=model, dataloader=dl, schedule=schedule,
            optimizer_factory=FakeOptim,
        )
        assert losses == loss_values

    def test_diverged_loss_breaks_loop(self):
        from soup_cli.utils.lr_finder import run_lr_sweep

        loss_values = [3.0, float("inf"), 1.0]

        def _wrap(value):
            obj = MagicMock()
            obj.detach = MagicMock(return_value=obj)
            obj.item = MagicMock(return_value=value)
            obj.backward = MagicMock(return_value=None)
            return obj

        class FakeModel:
            def __init__(self):
                self._idx = 0

            def parameters(self):
                return []

            def __call__(self, **batch):
                value = loss_values[self._idx]
                self._idx += 1
                return {"loss": _wrap(value)}

        class FakeOptim:
            def __init__(self, _params):
                self.param_groups = [{"lr": 0.0}]

            def zero_grad(self, set_to_none: bool = False):  # noqa: ARG002
                pass

            def step(self):
                pass

        dl = iter([{"x": MagicMock()}] * 3)
        losses = run_lr_sweep(
            model=FakeModel(), dataloader=dl,
            schedule=[1e-6, 1e-5, 1e-4],
            optimizer_factory=FakeOptim,
        )
        # Loop terminates after the inf — only the first finite loss kept.
        assert losses == [3.0]


# ---------------------------------------------------------------------------
# #58 — auto mixed-precision push
# ---------------------------------------------------------------------------


class TestResolveMixedPrecision:
    def test_auto_flag_off_preserves_legacy(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        tcfg = SimpleNamespace(auto_mixed_precision=False)
        bf16, fp16 = wrapper._resolve_mixed_precision(tcfg, "any")
        assert bf16 is True
        assert fp16 is False

    def test_auto_flag_off_cpu(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cpu"
        tcfg = SimpleNamespace(auto_mixed_precision=False)
        bf16, fp16 = wrapper._resolve_mixed_precision(tcfg, "any")
        assert (bf16, fp16) == (False, False)

    def test_auto_flag_cpu_returns_no(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cpu"
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        assert wrapper._resolve_mixed_precision(tcfg, "any") == (False, False)

    def test_auto_flag_picks_bf16_on_ampere(self, monkeypatch):
        """Ampere (cc 8.6) + non-quirk model → bf16."""
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda *_a, **_k: (8, 6),
            raising=False,
        )
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        bf16, fp16 = wrapper._resolve_mixed_precision(tcfg, "neutral-model")
        assert (bf16, fp16) == (True, False)

    def test_auto_flag_picks_fp16_for_qwen2_on_ampere(self, monkeypatch):
        import torch

        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda *_a, **_k: (8, 6),
            raising=False,
        )
        tcfg = SimpleNamespace(auto_mixed_precision=True)
        bf16, fp16 = wrapper._resolve_mixed_precision(
            tcfg, "Qwen/Qwen2-7B-Instruct",
        )
        assert (bf16, fp16) == (False, True)


# ---------------------------------------------------------------------------
# #57 — spike recovery hint
# ---------------------------------------------------------------------------


def _make_callback(tmp_path, **kwargs):
    from soup_cli.monitoring.callback import SoupTrainerCallback

    display = MagicMock()
    return SoupTrainerCallback(
        display=display,
        tracker=None,
        run_id="t",
        output_dir=str(tmp_path),
        **kwargs,
    )


class TestSpikeRecoveryHint:
    def test_writes_hint_file(self, tmp_path):
        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=2,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        hint = tmp_path / "spike_recovery.json"
        assert hint.exists()
        data = json.loads(hint.read_text(encoding="utf-8"))
        assert data["previous_lr"] == pytest.approx(1e-3)
        assert data["recommended_lr"] == pytest.approx(5e-4)
        assert data["should_recover"] is True
        assert data["attempts"] == 1

    def test_attempts_counter_increments(self, tmp_path):
        cb = _make_callback(
            tmp_path,
            spike_recovery=True,
            spike_recovery_max_attempts=3,
            spike_recovery_lr_decay=0.5,
        )
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        cb._write_spike_recovery_hint(args, loss=10.0)
        data = json.loads((tmp_path / "spike_recovery.json").read_text())
        assert data["attempts"] == 2

    def test_disabled_when_strategy_not_set(self, tmp_path):
        cb = _make_callback(tmp_path, spike_recovery=False)
        args = SimpleNamespace(
            learning_rate=1e-3, output_dir=str(tmp_path),
        )
        cb._write_spike_recovery_hint(args, loss=10.0)
        # No hint file written.
        assert not (tmp_path / "spike_recovery.json").exists()


# ---------------------------------------------------------------------------
# #59 — grad-accum advisory
# ---------------------------------------------------------------------------


class TestGradAccumAdvisory:
    def test_advise_fires_once_under_pressure(self, tmp_path, monkeypatch, capsys):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.5,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )

        # Mock torch presence + memory probe — high pressure (8 GB / 10 GB = 80%).
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 8 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is True

        # Second call is a no-op (one-shot).
        cb._grad_accum_monitor.observe = MagicMock()
        cb._maybe_advise_grad_accum()
        # Already advised, so monitor.observe shouldn't be called.
        cb._grad_accum_monitor.observe.assert_not_called()

    def test_no_advice_when_under_threshold(self, tmp_path, monkeypatch):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.9,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 5 * (1024**3)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False

    def test_no_advice_when_disabled(self, tmp_path):
        cb = _make_callback(tmp_path, grad_accum_auto_tune=False)
        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False

    def test_no_advice_when_cuda_unavailable(self, tmp_path, monkeypatch):
        cb = _make_callback(
            tmp_path,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.5,
            grad_accum_total_vram_gb=10.0,
            grad_accum_current_steps=1,
            grad_accum_current_batch=4,
        )
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        cb._maybe_advise_grad_accum()
        assert cb._grad_accum_advised is False
