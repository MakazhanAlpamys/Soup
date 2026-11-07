"""#1069 monitoring refusals, updated for native SFT watchdog support (#1989).

Recovery and auto-tuning remain refused on MLX.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from soup_cli.config.backend_support import check_config, unsupported_for
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi


class TestSchemaRejectionOnMlxBackend:
    """MLX still rejects recovery and auto-tuning with exact reasons.
    """

    @pytest.mark.parametrize(
        ("field", "training_payload", "reason"),
        [
            (
                "loss_spike_recovery",
                {"loss_watchdog": True, "loss_spike_recovery": True},
                "MLX does not implement checkpoint rollback or recovery hints",
            ),
            (
                "grad_accum_auto_tune",
                {"grad_accum_auto_tune": True},
                "there is no VRAM total to measure pressure against on unified memory",
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
    """Schema-refused monitoring settings have no unreachable doctor rows."""

    def test_registry_omits_schema_refused_monitoring_fields(self) -> None:
        fields = {entry.field for entry in unsupported_for("sft", "mlx")}
        assert fields.isdisjoint({
            "training.loss_watchdog",
            "training.loss_spike_recovery",
            "training.grad_accum_auto_tune",
        })

    def test_loadable_monitoring_defaults_report_no_gaps(self) -> None:
        cfg = SoupConfig(
            base="test/model", task="sft", backend="mlx",
            data={"train": "data.jsonl"},
            training={"loss_watchdog": False, "loss_spike_recovery": False,
                      "grad_accum_auto_tune": False},
        )
        assert check_config(cfg) == []


class TestMlxTrainerCheckUnsupported:
    """The internal trainer guard warns only about the remaining refused fields."""

    def test_trainer_warns_only_on_refused_fields(self, capsys) -> None:
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

        expected_spike = (
            "training.loss_spike_recovery "
            "(MLX does not implement checkpoint rollback or recovery hints)"
        )
        expected_tune = (
            "training.grad_accum_auto_tune "
            "(there is no VRAM total to measure pressure against on unified memory)"
        )

        assert "training.loss_watchdog" not in out
        assert expected_spike in out
        assert expected_tune in out


class TestReasonStringsByteIdenticalAcrossSurfaces:
    """Verify the reason strings across schema.py, backend_support.py, and mlx_sft.py
    are byte-identical for the refused monitoring fields.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "loss_spike_recovery",
            "grad_accum_auto_tune",
        ],
    )
    def test_reasons_are_byte_identical(self, field: str) -> None:
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        # 1. Schema reason extracted from ValueError
        payload = (
            {"loss_watchdog": True, "loss_spike_recovery": True}
            if field == "loss_spike_recovery"
            else {field: True}
        )
        cfg_kwargs = {
            "base": "sshleifer/tiny-gpt2",
            "task": "sft",
            "backend": "mlx",
            "data": {"train": "train.jsonl"},
            "training": payload,
        }
        with pytest.raises(ValueError) as exc_info:
            SoupConfig(**cfg_kwargs)
        msg = exc_info.value.errors()[0]["msg"]
        prefix = f"Value error, training.{field} is not supported for backend='mlx' because "
        assert prefix in msg
        schema_reason = msg.split(prefix)[1].strip()

        # Refused values cannot reach doctor, so there is no registry reason.
        assert f"training.{field}" not in {e.field for e in unsupported_for("sft", "mlx")}

        # 3. Trainer warning reason extracted from MLXSFTTrainerWrapper._check_unsupported
        wrapper = object.__new__(MLXSFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            training=TrainingConfig(**payload),
            data=SimpleNamespace(),
        )
        lines = []
        import unittest.mock
        with unittest.mock.patch("soup_cli.trainer.mlx_sft.console.print") as mock_print:
            wrapper._check_unsupported()
            if mock_print.called:
                raw_call = mock_print.call_args[0][0]
                lines.append(strip_ansi(raw_call))

        assert len(lines) == 1
        line = lines[0]
        # Format: "[yellow]MLX backend ignores: training.<field> (<reason>)[/]"
        match = re.search(rf"training\.{field} \((.*?)\)(?:, |\[/\]|$)", line)
        assert match is not None
        trainer_reason = match.group(1)

        # Keep the load-time refusal and internal trainer guard consistent.
        assert schema_reason == trainer_reason
