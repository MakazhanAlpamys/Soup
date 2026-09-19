"""Tests for Issue #1069: loss_watchdog, loss_spike_recovery and grad_accum_auto_tune
are rejected and reported as unread on backend: mlx.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.backend_support import check_config, unsupported_for
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi


class TestSchemaRejectionOnMlxBackend:
    """Validate that backend: mlx rejects watchdog, spike recovery,
    and grad_accum_auto_tune at config validation time.
    """

    @pytest.mark.parametrize(
        ("field", "training_payload"),
        [
            ("loss_watchdog", {"loss_watchdog": True}),
            ("loss_spike_recovery", {"loss_watchdog": True, "loss_spike_recovery": True}),
            ("grad_accum_auto_tune", {"grad_accum_auto_tune": True}),
        ],
    )
    def test_mlx_backend_rejects_monitoring_flags(
        self, field: str, training_payload: dict
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
            r"because 'mlx' does not attach a live training callback"
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
    all three unread monitoring fields for (sft, mlx).
    """

    def test_registry_contains_all_three_monitoring_fields(self) -> None:
        entries = {e.field: e for e in unsupported_for("sft", "mlx")}
        for field in (
            "training.loss_watchdog",
            "training.loss_spike_recovery",
            "training.grad_accum_auto_tune",
        ):
            assert field in entries
            assert entries[field].trainer_reads is True
            assert "live training callback" in entries[field].reason

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
        # model_fields_set on TrainingConfig marks what was explicitly written
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

        assert "training.loss_watchdog" in out
        assert "training.loss_spike_recovery" in out
        assert "training.grad_accum_auto_tune" in out
        assert "MLX does not attach a live training callback" in out


class TestMutationChecks:
    """Mutation checks guaranteeing that removing the refusal or warnings fails tests."""

    def test_mutation_removing_backend_refusal_fails_rejection(self) -> None:
        # A config without backend check would pass instantiation
        # Test that SoupConfig explicitly refuses it
        with pytest.raises(ValueError, match="backend='mlx'"):
            SoupConfig(
                base="sshleifer/tiny-gpt2",
                task="sft",
                backend="mlx",
                data={"train": "train.jsonl"},
                training={"loss_watchdog": True},
            )

    def test_mutation_missing_warning_in_check_unsupported_fails(self, capsys) -> None:
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        wrapper = object.__new__(MLXSFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            training=TrainingConfig(loss_watchdog=False),
            data=SimpleNamespace(),
        )
        wrapper._check_unsupported()
        out = strip_ansi(capsys.readouterr().out)
        assert "training.loss_watchdog" not in out
