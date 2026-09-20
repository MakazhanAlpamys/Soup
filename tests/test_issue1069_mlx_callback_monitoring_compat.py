"""Tests for Issue #1069: loss_watchdog, loss_spike_recovery and grad_accum_auto_tune
are rejected and reported as unread on backend: mlx.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from soup_cli.config.backend_support import REJECTED, check_config, unsupported_for
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi


class TestSchemaRejectionOnMlxBackend:
    """Validate that backend: mlx rejects watchdog, spike recovery,
    and grad_accum_auto_tune at config validation time with exact reasons.
    """

    @pytest.mark.parametrize(
        ("field", "training_payload", "reason"),
        [
            (
                "loss_watchdog",
                {"loss_watchdog": True},
                "Soup does not implement the watchdog on the MLX callback, "
                "which has no stop control",
            ),
            (
                "loss_spike_recovery",
                {"loss_watchdog": True, "loss_spike_recovery": True},
                "there is no checkpoint rollback or LR decay on MLX",
            ),
            (
                "grad_accum_auto_tune",
                {"grad_accum_auto_tune": True},
                "there is no VRAM-pressure signal on unified memory",
            ),
        ],
    )
    def test_mlx_backend_rejects_monitoring_flags(
        self, field: str, training_payload: dict, reason: str
    ) -> None:
        cfg_kwargs = {
            "base": "sshleifer/tiny-gpt2",
            "task": "sft",
            "backend": "mlx",
            "data": {"train": "train.jsonl"},
            "training": dict(training_payload),
        }
        pattern = (
            rf"training\.{field} is not supported for backend='mlx' "
            rf"because {re.escape(reason)}"
        )
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(**cfg_kwargs)

    def test_transformers_backend_accepts_monitoring_flags(self) -> None:
        cfg = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="sft",
            backend="transformers",
            data={"train": "train.jsonl"},
            training={
                "loss_watchdog": True,
                "loss_spike_recovery": True,
                "grad_accum_auto_tune": True,
            },
        )
        assert cfg.training.loss_watchdog is True
        assert cfg.training.loss_spike_recovery is True
        assert cfg.training.grad_accum_auto_tune is True


class TestBackendSupportRegistryAndDoctor:
    """Verify backend_support registry and check_config declare and report
    all three unread monitoring fields for (sft, mlx) as REJECTED.
    """

    def test_registry_contains_all_three_monitoring_fields_as_rejected(self) -> None:
        entries = {e.field: e for e in unsupported_for("sft", "mlx")}
        assert entries["training.loss_watchdog"].status == REJECTED
        assert "watchdog on the MLX callback" in entries["training.loss_watchdog"].reason
        assert entries["training.loss_watchdog"].trainer_reads is True

        assert entries["training.loss_spike_recovery"].status == REJECTED
        assert "checkpoint rollback" in entries["training.loss_spike_recovery"].reason
        assert entries["training.loss_spike_recovery"].trainer_reads is True

        assert entries["training.grad_accum_auto_tune"].status == REJECTED
        assert "VRAM-pressure" in entries["training.grad_accum_auto_tune"].reason
        assert entries["training.grad_accum_auto_tune"].trainer_reads is True

    def test_check_config_reports_declared_gaps(self) -> None:
        cfg = SoupConfig.model_construct(
            base="sshleifer/tiny-gpt2",
            task="sft",
            backend="mlx",
            data=SimpleNamespace(model_fields_set=set()),
            training=TrainingConfig(
                loss_watchdog=True,
                loss_spike_recovery=True,
                grad_accum_auto_tune=True,
            ),
        )
        reported = {e.field for e in check_config(cfg)}
        assert "training.loss_watchdog" in reported
        assert "training.loss_spike_recovery" in reported
        assert "training.grad_accum_auto_tune" in reported


class TestMlxTrainerCheckUnsupported:
    """Verify MLXSFTTrainerWrapper._check_unsupported prints warnings for all three fields."""

    def test_trainer_warns_on_all_three_fields(self, capsys) -> None:
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        wrapper = object.__new__(MLXSFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            training=TrainingConfig(
                loss_watchdog=True,
                loss_spike_recovery=True,
                grad_accum_auto_tune=True,
            ),
            data=SimpleNamespace(),
        )

        wrapper._check_unsupported()
        out = " ".join(strip_ansi(capsys.readouterr().out).split())

        expected_watchdog = (
            "training.loss_watchdog (Soup does not implement the watchdog "
            "on the MLX callback, which has no stop control)"
        )
        expected_spike = (
            "training.loss_spike_recovery "
            "(there is no checkpoint rollback or LR decay on MLX)"
        )
        expected_tune = (
            "training.grad_accum_auto_tune "
            "(there is no VRAM-pressure signal on unified memory)"
        )

        assert expected_watchdog in out
        assert expected_spike in out
        assert expected_tune in out
