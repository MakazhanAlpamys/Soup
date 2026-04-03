"""Tests for loss watchdog: auto-stop on loss spikes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from soup_cli.config.schema import TrainingConfig

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestWatchdogConfig:
    """Tests for loss_watchdog config fields."""

    def test_loss_watchdog_default_false(self):
        """loss_watchdog defaults to False."""
        cfg = TrainingConfig()
        assert cfg.loss_watchdog is False

    def test_loss_watchdog_threshold_default(self):
        """loss_watchdog_threshold defaults to 3.0."""
        cfg = TrainingConfig()
        assert cfg.loss_watchdog_threshold == 3.0

    def test_loss_watchdog_patience_default(self):
        """loss_watchdog_patience defaults to 5."""
        cfg = TrainingConfig()
        assert cfg.loss_watchdog_patience == 5

    def test_loss_watchdog_threshold_positive(self):
        """loss_watchdog_threshold must be positive."""
        with pytest.raises(Exception):
            TrainingConfig(loss_watchdog_threshold=0.0)

    def test_loss_watchdog_threshold_negative_rejected(self):
        """loss_watchdog_threshold rejects negative values."""
        with pytest.raises(Exception):
            TrainingConfig(loss_watchdog_threshold=-1.0)

    def test_loss_watchdog_patience_positive(self):
        """loss_watchdog_patience must be >= 1."""
        with pytest.raises(Exception):
            TrainingConfig(loss_watchdog_patience=0)

    def test_loss_watchdog_enable(self):
        """loss_watchdog can be enabled."""
        cfg = TrainingConfig(loss_watchdog=True)
        assert cfg.loss_watchdog is True

    def test_loss_watchdog_custom_values(self):
        """Custom threshold and patience values are accepted."""
        cfg = TrainingConfig(
            loss_watchdog=True,
            loss_watchdog_threshold=5.0,
            loss_watchdog_patience=10,
        )
        assert cfg.loss_watchdog_threshold == 5.0
        assert cfg.loss_watchdog_patience == 10


# ---------------------------------------------------------------------------
# Watchdog logic in callback
# ---------------------------------------------------------------------------


class TestWatchdogCallback:
    """Tests for loss watchdog logic in SoupTrainerCallback."""

    def _make_callback(self, threshold: float = 3.0, patience: int = 5):
        """Create a callback with watchdog enabled."""
        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.monitoring.display import TrainingDisplay

        display = MagicMock(spec=TrainingDisplay)
        cb = SoupTrainerCallback(
            display=display,
            loss_watchdog=True,
            loss_watchdog_threshold=threshold,
            loss_watchdog_patience=patience,
        )
        return cb

    def test_watchdog_no_stop_normal_loss(self):
        """Normal loss values do not trigger watchdog."""
        cb = self._make_callback(threshold=3.0, patience=3)

        state = MagicMock()
        state.global_step = 1
        state.epoch = 1.0
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # Simulate 5 normal loss values
        for step in range(5):
            state.global_step = step + 1
            cb.on_log(args, state, control, logs={"loss": 1.5})

        assert control.should_training_stop is False

    def test_watchdog_triggers_after_patience(self):
        """Watchdog stops training after patience consecutive high losses."""
        cb = self._make_callback(threshold=3.0, patience=3)

        state = MagicMock()
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # Simulate patience+1 high loss values
        for step in range(4):
            state.global_step = step + 1
            state.epoch = 1.0
            cb.on_log(args, state, control, logs={"loss": 5.0})

        assert control.should_training_stop is True

    def test_watchdog_resets_on_good_loss(self):
        """Counter resets when a normal loss appears."""
        cb = self._make_callback(threshold=3.0, patience=5)

        state = MagicMock()
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # 3 high, then 1 normal, then 3 high — should NOT trigger (patience=5)
        for loss_val in [5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0]:
            state.global_step = 1
            state.epoch = 1.0
            cb.on_log(args, state, control, logs={"loss": loss_val})

        assert control.should_training_stop is False

    def test_watchdog_disabled_by_default(self):
        """Watchdog does not interfere when disabled."""
        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.monitoring.display import TrainingDisplay

        display = MagicMock(spec=TrainingDisplay)
        cb = SoupTrainerCallback(display=display)

        state = MagicMock()
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # Very high loss — should NOT trigger since watchdog is off
        for step in range(20):
            state.global_step = step
            state.epoch = 1.0
            cb.on_log(args, state, control, logs={"loss": 100.0})

        assert control.should_training_stop is False

    def test_watchdog_exact_patience_boundary(self):
        """Exactly patience high-loss steps triggers on the next one."""
        cb = self._make_callback(threshold=3.0, patience=3)

        state = MagicMock()
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # Exactly 3 high losses (patience=3) — should trigger on 4th
        for step in range(3):
            state.global_step = step + 1
            state.epoch = 1.0
            cb.on_log(args, state, control, logs={"loss": 5.0})

        # After patience steps of high loss, next one triggers
        assert control.should_training_stop is True

    def test_watchdog_no_loss_in_logs(self):
        """Watchdog handles logs without loss key gracefully."""
        cb = self._make_callback(threshold=3.0, patience=3)

        state = MagicMock()
        args = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # Logs without loss key
        cb.on_log(args, state, control, logs={"learning_rate": 1e-5})
        assert control.should_training_stop is False


# ---------------------------------------------------------------------------
# Sweep integration
# ---------------------------------------------------------------------------


class TestWatchdogSweep:
    """Tests for watchdog fields in sweep param support."""

    def test_watchdog_threshold_in_sweep(self):
        """loss_watchdog_threshold is a valid sweep param."""
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(
            ["training.loss_watchdog_threshold=2.0,3.0,5.0"]
        )
        assert "training.loss_watchdog_threshold" in params
        assert len(params["training.loss_watchdog_threshold"]) == 3

    def test_watchdog_patience_in_sweep(self):
        """loss_watchdog_patience is a valid sweep param."""
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(
            ["training.loss_watchdog_patience=3,5,10"]
        )
        assert "training.loss_watchdog_patience" in params
        assert len(params["training.loss_watchdog_patience"]) == 3
