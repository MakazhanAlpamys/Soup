"""Issue #2370 — an int too large for a float is not a number, not a crash."""

from __future__ import annotations

import pytest

from tests.test_issue762_adapters_audit import _config, _mlx_record

pytestmark = pytest.mark.unit

HUGE = 10**400


def _row(config, record, setting):
    from soup_cli.utils.adapter_audit import audit_adapter

    result = audit_adapter(config, record)
    return result, next(r for r in result.rows if r.setting == setting)


@pytest.mark.parametrize("value", [HUGE, -HUGE])
def test_an_int_past_the_double_range_is_not_a_number(value):
    from soup_cli.utils.adapter_audit import _is_number

    assert _is_number(value) is False


@pytest.mark.parametrize("value", [10**15, 2e-5, 0])
def test_ordinary_numbers_still_are(value):
    from soup_cli.utils.adapter_audit import _is_number

    assert _is_number(value) is True


@pytest.mark.parametrize(
    "lr,peak_lr", [(1e-4, HUGE), (HUGE, 1e-4)], ids=["recorded", "configured"]
)
def test_a_huge_learning_rate_is_a_diverged_row(lr, peak_lr):
    result, row = _row(
        _config(training={"lr": lr}), _mlx_record(peak_lr=peak_lr), "learning_rate"
    )

    assert row.status == "diverged"
    assert "finite numbers" in row.detail
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "overrides",
    [{"warmup_updates": HUGE}, {"total_updates": HUGE}],
    ids=["warmup_updates", "total_updates"],
)
def test_a_huge_warmup_total_is_a_diverged_row(overrides):
    _, row = _row(_config(), _mlx_record(**overrides), "warmup_ratio")

    assert row.status == "diverged"
    assert "must be numbers" in row.detail


@pytest.mark.parametrize(
    "params",
    [{"rank": 8, "scale": HUGE}, {"rank": HUGE, "scale": 2.0}],
    ids=["scale", "rank"],
)
def test_a_huge_lora_scale_or_rank_leaves_alpha_unread(params):
    _, row = _row(_config(), _mlx_record(lora_parameters=params), "lora.alpha")

    assert row.status == "unknown"
