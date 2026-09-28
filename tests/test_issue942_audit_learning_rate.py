"""Issue #942 — audit the learning rate a finished run actually used."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


def _config(lr: float = 5e-4, **training):
    return {
        "backend": "mlx",
        "training": {"lr": lr, **training},
        "data": {},
    }


def _mlx_record(peak_lr=5e-4, **record):
    return {"fine_tune_type": "lora", "peak_lr": peak_lr, **record}


def _lr_row(config, record):
    from soup_cli.utils.adapter_audit import audit_adapter

    result = audit_adapter(config, record)
    return result, next(row for row in result.rows if row.setting == "learning_rate")


def test_matching_mlx_peak_is_checked_not_merely_echoed():
    result, row = _lr_row(
        _config(5e-4),
        _mlx_record(5e-4, learning_rate=9e-4),
    )

    assert row.status == "ok"
    assert row.asked == 5e-4
    assert row.ran == 5e-4
    assert result.checked_count >= 1


def test_different_mlx_peak_is_a_divergence_and_fails_the_gate():
    """Mutation control: replacing the comparison with unconditional equality fails."""
    result, row = _lr_row(_config(5e-4), _mlx_record(2e-4))

    assert row.status == "diverged"
    assert row.asked == 5e-4
    assert row.ran == 2e-4
    assert "training.lr=0.0005" in row.detail
    assert "peak_lr=0.0002" in row.detail
    assert result.exit_code == 2


def test_config_echo_is_not_substituted_for_missing_effective_peak():
    record = _mlx_record(learning_rate=5e-4)
    del record["peak_lr"]

    _, row = _lr_row(_config(5e-4), record)

    assert row.status == "unknown"
    assert row.ran is None
    assert "not checked" in row.detail
    assert "peak_lr" in row.detail


def test_transformers_record_plainly_says_learning_rate_was_not_checked():
    from soup_cli.utils.adapter_audit import unknown_reason

    config = _config(5e-4)
    config["backend"] = "transformers"
    _, row = _lr_row(config, {"peft_type": "LORA", "r": 8, "lora_alpha": 16})

    assert row.status == "unknown"
    assert "effective learning rate" in row.detail
    assert "not checked" in row.detail
    assert "transformers" in row.detail
    reason = unknown_reason("peft")
    assert reason is not None
    assert "effective learning rate" in reason
    assert "not checked" in reason


def test_warmup_and_decay_do_not_turn_the_configured_peak_into_a_false_positive():
    result, row = _lr_row(
        _config(5e-4, warmup_ratio=0.25, scheduler="cosine"),
        _mlx_record(
            5e-4,
            warmup_updates=3,
            total_updates=12,
            scheduler="cosine",
        ),
    )

    assert row.status == "ok"
    assert not any(r.setting == "learning_rate" for r in result.rows if r.status == "diverged")


@pytest.mark.parametrize("peak_lr", [True, "0.0005", math.nan, math.inf, -math.inf])
def test_malformed_or_non_finite_recorded_peak_is_a_verdict_not_agreement(peak_lr):
    result, row = _lr_row(_config(5e-4), _mlx_record(peak_lr))

    assert row.status == "diverged"
    assert "finite numbers" in row.detail
    assert result.exit_code == 2
