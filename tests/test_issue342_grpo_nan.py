"""Tests for #342 — GRPO numerical stability (gradient watchdog).

CPU-only, no model, no GPU.  Four test groups:

1. Watchdog behavior — detect NaN, zero grads, track count
2. Run record — skip count always persisted (including zero)
3. Microbatch placement — on_pre_optimizer_step vs training_step
4. Defensive callback wiring — ensure_grpo_stability_callback contract
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_state(global_step: int = 100) -> SimpleNamespace:
    """Mock HF TrainerState with log_history."""
    return SimpleNamespace(global_step=global_step, log_history=[])


def _make_mock_control() -> MagicMock:
    """Mock HF TrainerControl."""
    return MagicMock()


def _make_mock_model_with_grads(*grad_tensors: torch.Tensor) -> MagicMock:
    """Create a mock model whose parameters() yields params with the given grads."""
    params = []
    for g in grad_tensors:
        p = MagicMock()
        p.grad = g
        params.append(p)
    model = MagicMock()
    model.parameters.return_value = params
    return model


def _get_callback_instance():
    """Get a fresh GRPOStabilityCallback instance with defaults."""
    from soup_cli.monitoring.grpo_stability_callback import (
        GRPOStabilityCallback,
    )

    return GRPOStabilityCallback()


# ===========================================================================
# Group 1 — Watchdog behavior [C7.5-C7.9]
# ===========================================================================


class TestGradWatchdog:
    """Pre-step gradient watchdog: detect NaN, zero grads, track count."""

    def test_nan_grad_detected_and_zeroed(self):
        """Watchdog zeros all gradients when any param has NaN grad."""
        cb = _get_callback_instance()
        nan_grad = torch.tensor([float("nan"), 1.0])
        model = _make_mock_model_with_grads(nan_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        cb.on_pre_optimizer_step(None, state, control, model=model)
        model.zero_grad.assert_called_once()
        assert cb._nan_skip_count == 1

    def test_finite_grad_untouched(self):
        """Watchdog does nothing when all gradients are finite."""
        cb = _get_callback_instance()
        ok_grad = torch.tensor([1.0, 2.0])
        model = _make_mock_model_with_grads(ok_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        cb.on_pre_optimizer_step(None, state, control, model=model)
        model.zero_grad.assert_not_called()
        assert cb._nan_skip_count == 0

    def test_inf_grad_detected(self):
        """Watchdog catches inf (not just NaN)."""
        cb = _get_callback_instance()
        inf_grad = torch.tensor([float("inf"), 1.0])
        model = _make_mock_model_with_grads(inf_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        cb.on_pre_optimizer_step(None, state, control, model=model)
        model.zero_grad.assert_called_once()
        assert cb._nan_skip_count == 1

    def test_multiple_nans_counted(self):
        """Cumulative counter increments on each NaN step."""
        cb = _get_callback_instance()
        nan_grad = torch.tensor([float("nan")])
        model = _make_mock_model_with_grads(nan_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        for _ in range(3):
            cb.on_pre_optimizer_step(None, state, control, model=model)
        assert cb._nan_skip_count == 3

    def test_warning_logged(self, caplog):
        """Watchdog emits a WARNING-level log on NaN detection."""
        cb = _get_callback_instance()
        nan_grad = torch.tensor([float("nan")])
        model = _make_mock_model_with_grads(nan_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        with caplog.at_level(logging.WARNING):
            cb.on_pre_optimizer_step(None, state, control, model=model)
        assert any("non-finite" in r.message for r in caplog.records)


# ===========================================================================
# Group 2 — Run record (always persisted, including zero)
# ===========================================================================


class TestRunRecord:
    """Skip fraction persisted in log_history for audit."""

    def test_nonzero_count_surfaced(self):
        """Non-zero skip count appears in state.log_history."""
        cb = _get_callback_instance()
        cb._nan_skip_count = 3
        state = _make_mock_state(global_step=100)
        control = _make_mock_control()

        cb.on_train_end(None, state, control)
        record = state.log_history[-1]
        assert record["nan_skip_count"] == 3
        assert record["nan_skip_fraction"] == 0.03

    def test_zero_count_surfaced(self):
        """Zero skip count must still appear in log_history.

        A missing record means 'trained before the watchdog existed';
        a zero record means 'the watchdog ran and saw no NaN'.
        """
        cb = _get_callback_instance()
        state = _make_mock_state(global_step=50)
        control = _make_mock_control()

        cb.on_train_end(None, state, control)
        record = state.log_history[-1]
        assert record["nan_skip_count"] == 0
        assert record["nan_skip_fraction"] == 0.0

    def test_loud_threshold_at_5_percent(self, caplog):
        """>=5% triggers WARNING; <5% triggers INFO."""
        cb = _get_callback_instance()
        cb._nan_skip_count = 5
        state = _make_mock_state(global_step=100)
        control = _make_mock_control()

        with caplog.at_level(logging.WARNING):
            cb.on_train_end(None, state, control)
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), "5% fraction must trigger WARNING"


# ===========================================================================
# Group 3 — Microbatch placement
# ===========================================================================


class TestMicrobatchPlacement:
    """Unit-level check that on_pre_optimizer_step handles accumulated gradients.

    These tests verify the watchdog's behavior on pre-accumulated gradient
    buffers.  The e2e smoke test (TestGRPOWatchdogE2E) proves the actual
    placement in the trainer lifecycle.
    """

    def test_first_microbatch_survives_second_nan(self):
        """Accumulated gradient from microbatch 1 must survive.

        Simulate gradient accumulation: microbatch 1 is finite,
        microbatch 2 is NaN.  After on_pre_optimizer_step, the grads
        should be zeroed (or None), NOT the finite microbatch 1 grads.
        The watchdog sees the accumulated result, which includes NaN.
        """
        cb = _get_callback_instance()

        # After accumulation: mixed finite + NaN in the gradient buffer.
        mixed_grad = torch.tensor([1.0, float("nan")])
        model = _make_mock_model_with_grads(mixed_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        cb.on_pre_optimizer_step(None, state, control, model=model)

        # The watchdog zeroes because NaN is present.
        model.zero_grad.assert_called_once()
        assert cb._nan_skip_count == 1

    def test_clean_accumulation_untouched(self):
        """When both microbatches are finite, no zeroing occurs."""
        cb = _get_callback_instance()

        # After accumulation: all finite.
        finite_grad = torch.tensor([1.0, 2.0])
        model = _make_mock_model_with_grads(finite_grad)
        state = _make_mock_state()
        control = _make_mock_control()

        cb.on_pre_optimizer_step(None, state, control, model=model)

        model.zero_grad.assert_not_called()
        assert cb._nan_skip_count == 0


# ---------------------------------------------------------------------------
# Group 4 — ensure_grpo_stability_callback defensive contract
# ---------------------------------------------------------------------------
# Concrete test doubles (NOT MagicMock) so missing attributes surface as
# AttributeError rather than silently auto-generating child mocks.
# ---------------------------------------------------------------------------

class _CallbackHandler:
    """Minimal callback_handler stand-in with a mutable callbacks list."""

    def __init__(self):
        self.callbacks: list = []


class _TrainerWithCallbacks:
    """Has both add_callback and callback_handler (normal HF Trainer)."""

    def __init__(self):
        self.callback_handler = _CallbackHandler()

    def add_callback(self, cb):
        self.callback_handler.callbacks.append(cb)


class _TrainerWithoutCallbackHandler:
    """Has add_callback but NO callback_handler (future TRL / slim mock)."""

    def __init__(self):
        self._attached: list = []

    def add_callback(self, cb):
        self._attached.append(cb)


class _BareTrainer:
    """Has neither add_callback nor callback_handler (FakeGRPOTrainer)."""

    def __init__(self, **kwargs):
        pass


class TestEnsureGRPOStabilityCallback:
    """Group 4 — defensive callback wiring for issue #342."""

    def test_attaches_when_callback_handler_present(self):
        """Normal path: trainer with both add_callback and handler."""
        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        trainer = _TrainerWithCallbacks()
        result = ensure_grpo_stability_callback(trainer)

        assert result is True, "Must attach on first call"
        assert any(
            isinstance(cb, GRPOStabilityCallback)
            for cb in trainer.callback_handler.callbacks
        ), "Callback must be in the handler's list"

    def test_skips_duplicate_when_already_attached(self):
        """Duplicate detection via callback_handler.callbacks."""
        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        trainer = _TrainerWithCallbacks()
        ensure_grpo_stability_callback(trainer)
        result = ensure_grpo_stability_callback(trainer)

        assert result is False, "Second call must be no-op"
        count = sum(
            1
            for cb in trainer.callback_handler.callbacks
            if isinstance(cb, GRPOStabilityCallback)
        )
        assert count == 1, f"Expected exactly 1 callback, got {count}"

    def test_attaches_without_callback_handler(self):
        """Trainer with add_callback but no callback_handler."""
        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        trainer = _TrainerWithoutCallbackHandler()
        result = ensure_grpo_stability_callback(trainer)

        assert result is True, (
            "Must attach even without callback_handler"
        )
        assert any(
            isinstance(cb, GRPOStabilityCallback)
            for cb in trainer._attached
        ), "Callback must be in the trainer's internal list"

    def test_returns_false_for_bare_trainer(self):
        """Bare trainer without add_callback must not crash."""
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        trainer = _BareTrainer()
        result = ensure_grpo_stability_callback(trainer)

        assert result is False, (
            "Must return False for trainers without add_callback"
        )

    def test_returns_false_for_none_and_non_trainer(self):
        """Degenerate inputs (None, string, int) must degrade safely."""
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        for degenerate in (None, "trainer", 42, [], object()):
            result = ensure_grpo_stability_callback(degenerate)
            assert result is False, f"Expected False for {degenerate!r}, got {result!r}"

    def test_attaches_when_callbacks_attribute_is_non_iterable(self):
        """Non-iterable callback_handler.callbacks must not crash."""
        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        class _BrokenHandlerTrainer:
            def __init__(self):
                self.callback_handler = type("Handler", (), {"callbacks": 123})()
                self._attached = []

            def add_callback(self, cb):
                self._attached.append(cb)

        trainer = _BrokenHandlerTrainer()
        result = ensure_grpo_stability_callback(trainer)
        assert result is True
        assert any(isinstance(cb, GRPOStabilityCallback) for cb in trainer._attached)

    def test_returns_false_when_add_callback_raises(self):
        """Failing add_callback degrades to False without crashing."""
        from soup_cli.utils.peft_wiring import (
            ensure_grpo_stability_callback,
        )

        class _FailingTrainer:
            def add_callback(self, cb):
                raise RuntimeError("Trainer rejects callback")

        trainer = _FailingTrainer()
        result = ensure_grpo_stability_callback(trainer)
        assert result is False


# ---------------------------------------------------------------------------
# Group 5 — `soup adapters audit` reads the skip record (#342 item 3)
# ---------------------------------------------------------------------------

_PEFT_RECORD = {
    "peft_type": "LORA",
    "base_model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
}


def _audit_config(task: str = "grpo") -> dict:
    return {
        "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task": task,
        "data": {"train": "./d.jsonl"},
        "training": {"lora": {"r": 8, "alpha": 16}},
    }


class TestTheAuditRow:
    @pytest.mark.parametrize(
        ("extra", "status", "ran"),
        [
            ({}, "unknown", None),
            ({"nan_skip_count": 0, "nan_skip_fraction": 0.0}, "ok", 0.0),
            ({"nan_skip_count": 1, "nan_skip_fraction": 0.5}, "diverged", 0.5),
        ],
    )
    def test_the_row_reads_the_record(self, extra, status, ran):
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_audit_config(), {**_PEFT_RECORD, **extra})
        row = next(r for r in result.rows if r.setting == "nan_skip_fraction")
        assert (row.status, row.ran) == (status, ran)

    def test_a_non_grpo_adapter_gets_no_row(self):
        """Only GRPO runs carry the watchdog; elsewhere the row could never be known."""
        from soup_cli.utils.adapter_audit import audit_adapter

        result = audit_adapter(_audit_config(task="sft"), dict(_PEFT_RECORD))
        assert all(r.setting != "nan_skip_fraction" for r in result.rows)


class TestTheAuditCommandReadsTrainerState:
    def _row(self, tmp_path, monkeypatch, trainer_state):
        import json

        import yaml
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "adapter_config.json").write_text(json.dumps(_PEFT_RECORD))
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(_audit_config()))
        if trainer_state is not None:
            (tmp_path / "trainer_state.json").write_text(trainer_state)
        res = CliRunner().invoke(app, ["audit", ".", "--config", "soup.yaml", "--json"])
        assert res.exit_code in (0, 2), (res.output, repr(res.exception))
        rows = json.loads(res.stdout)["rows"]
        return next(r for r in rows if r["setting"] == "nan_skip_fraction")

    def test_the_last_record_in_trainer_state_is_reported(self, tmp_path, monkeypatch):
        import json

        state = {
            "log_history": [
                {"loss": 1.0, "step": 1},
                {"nan_skip_count": 1, "nan_skip_fraction": 0.5},
            ]
        }
        row = self._row(tmp_path, monkeypatch, json.dumps(state))
        assert (row["status"], row["ran"]) == ("diverged", 0.5)

    def test_no_trainer_state_reads_unknown(self, tmp_path, monkeypatch):
        assert self._row(tmp_path, monkeypatch, None)["status"] == "unknown"

    @pytest.mark.parametrize(
        "bad", ["[]", '{"log_history": 5}', '{"log_history": [3]}', "not json"]
    )
    def test_a_malformed_trainer_state_reads_unknown(self, tmp_path, monkeypatch, bad):
        assert self._row(tmp_path, monkeypatch, bad)["status"] == "unknown"


# ---------------------------------------------------------------------------
# Group 6 — Message text pinning (#342 review round 2)
# ---------------------------------------------------------------------------


class TestTheMessagesPeopleSee:
    def test_a_skip_below_five_percent_is_still_reported(self, caplog):
        cb = _get_callback_instance()
        cb._nan_skip_count = 1
        with caplog.at_level(logging.INFO):
            cb.on_train_end(None, _make_mock_state(global_step=100), _make_mock_control())
        assert "1 of 100 optimizer steps were skipped" in caplog.text

    def test_the_warning_names_the_step_the_trainer_logs(self, caplog):
        """global_step is incremented after on_pre_optimizer_step, so step 5 is 4 + 1."""
        cb = _get_callback_instance()
        model = _make_mock_model_with_grads(torch.tensor([float("nan")]))
        with caplog.at_level(logging.WARNING):
            cb.on_pre_optimizer_step(
                None, _make_mock_state(global_step=4), _make_mock_control(), model=model
            )
        assert "step 5:" in caplog.text
