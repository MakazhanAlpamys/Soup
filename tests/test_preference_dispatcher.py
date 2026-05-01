"""Tests for v0.40.0 Part B — unified preference loss dispatcher.

Adds ``task: preference`` + ``training.preference_loss`` Literal
(dpo/simpo/orpo/ipo/bco). Existing per-task forms (``task: dpo`` etc) keep
working unchanged — the unified surface is *additive*, not a breaking
collapse. Backward-compat ``resolve_preference_loss`` maps legacy task
strings to their preference_loss equivalents for callers that want a
single dispatch entry-point.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

# ─── Schema Tests ───────────────────────────────────────────────────────────


class TestPreferenceTaskField:
    def test_preference_task_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        assert cfg.task == "preference"
        assert cfg.training.preference_loss == "dpo"

    @pytest.mark.parametrize("loss", ["dpo", "simpo", "orpo", "ipo", "bco"])
    def test_preference_loss_each_value(self, loss):
        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": loss},
        )
        assert cfg.training.preference_loss == loss

    def test_preference_loss_unknown_rejected(self):
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="preference",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss": "garbage"},
            )

    def test_preference_task_requires_preference_loss(self):
        """task=preference without preference_loss must error."""
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="preference",
                data={"train": "./data.jsonl", "format": "dpo"},
            )

    def test_preference_loss_default_none_for_other_tasks(self):
        """Non-preference tasks default preference_loss=None."""
        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.training.preference_loss is None

    def test_preference_loss_set_outside_preference_task_rejected(self):
        """preference_loss is meaningful only when task=preference."""
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="dpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss": "dpo"},
            )


# ─── Resolver ──────────────────────────────────────────────────────────────


class TestResolvePreferenceLoss:
    def test_resolve_preference_task_returns_loss(self):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "simpo"},
        )
        assert resolve_preference_loss(cfg) == "simpo"

    @pytest.mark.parametrize(
        "task,expected",
        [
            ("dpo", "dpo"),
            ("simpo", "simpo"),
            ("orpo", "orpo"),
            ("ipo", "ipo"),
            ("bco", "bco"),
        ],
    )
    def test_legacy_task_maps_to_loss(self, task, expected):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task=task,
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert resolve_preference_loss(cfg) == expected

    def test_resolve_non_preference_task_returns_none(self):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl"},
        )
        assert resolve_preference_loss(cfg) is None


# ─── Wrapper Tests ─────────────────────────────────────────────────────────


class TestPreferenceTrainerWrapper:
    def test_import_exists(self):
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        assert PreferenceTrainerWrapper is not None

    @pytest.mark.parametrize(
        "loss,wrapper_path",
        [
            ("dpo", "soup_cli.trainer.dpo.DPOTrainerWrapper"),
            ("simpo", "soup_cli.trainer.simpo.SimPOTrainerWrapper"),
            ("orpo", "soup_cli.trainer.orpo.ORPOTrainerWrapper"),
            ("ipo", "soup_cli.trainer.ipo.IPOTrainerWrapper"),
            ("bco", "soup_cli.trainer.bco.BCOTrainerWrapper"),
        ],
    )
    def test_dispatcher_routes_to_correct_wrapper(self, loss, wrapper_path):
        """PreferenceTrainerWrapper.setup() must delegate to the right wrapper.

        Asserts the inner cfg sent to the wrapper has task=loss and that
        preference_loss has been cleared (defends _make_inner_cfg's contract).
        """
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": loss},
        )
        with mock_patch(wrapper_path) as mock_inner:
            mock_instance = MagicMock()
            mock_instance.train.return_value = {
                "initial_loss": 1.0, "final_loss": 0.5,
                "duration": "1m", "duration_secs": 60.0,
                "output_dir": "./out", "total_steps": 10,
            }
            mock_inner.return_value = mock_instance

            wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})
            wrapper.train()

            mock_inner.assert_called_once()
            inner_cfg = mock_inner.call_args[0][0]
            assert inner_cfg.task == loss
            assert inner_cfg.training.preference_loss is None
            mock_instance.setup.assert_called_once()
            mock_instance.train.assert_called_once()

    def test_setup_does_not_mutate_caller_cfg(self):
        """_make_inner_cfg must return a copy; cfg.task unchanged after setup."""
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        with mock_patch("soup_cli.trainer.dpo.DPOTrainerWrapper") as mock_inner:
            mock_inner.return_value = MagicMock()
            wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})
        # Caller's cfg untouched.
        assert cfg.task == "preference"
        assert cfg.training.preference_loss == "dpo"

    def test_train_before_setup_raises(self):
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="setup"):
            wrapper.train()

    def test_unknown_loss_raises_at_setup(self):
        """Defence-in-depth: schema gate prevents this, but if it slipped, raise."""
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        # Mutate after construction to bypass schema validation.
        cfg.training.preference_loss = "garbage"
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(ValueError, match="preference_loss"):
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})


# ─── Train Routing ─────────────────────────────────────────────────────────


class TestPreferenceTrainRouting:
    def test_sweep_routes_to_preference_wrapper(self):
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        fake_dataset = {"train": [{"prompt": "Q", "chosen": "A", "rejected": "B"}]}
        fake_result = {
            "initial_loss": 1.0, "final_loss": 0.5,
            "duration": "1m", "duration_secs": 60.0,
            "output_dir": "./out", "total_steps": 10,
        }
        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}

        with mock_patch(
            "soup_cli.data.loader.load_dataset", return_value=fake_dataset,
        ), mock_patch(
            "soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU"),
        ), mock_patch(
            "soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info,
        ), mock_patch(
            "soup_cli.experiment.tracker.ExperimentTracker",
        ) as mock_tracker_cls, mock_patch(
            "soup_cli.monitoring.display.TrainingDisplay",
        ), mock_patch(
            "soup_cli.trainer.preference.PreferenceTrainerWrapper.setup",
        ), mock_patch(
            "soup_cli.trainer.preference.PreferenceTrainerWrapper.train",
            return_value=fake_result,
        ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-pref-1"
            mock_tracker_cls.return_value = mock_tracker
            result = _run_single(cfg, {}, "pref_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-pref-1"
