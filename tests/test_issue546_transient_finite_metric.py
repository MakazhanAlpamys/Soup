"""Regression coverage for issue #546's over-broad log_history finite-state gate."""

from __future__ import annotations


def test_self_corrected_transient_metric_is_accepted() -> None:
    from soup_cli.trainer.sft import _assert_finite_training_state

    _assert_finite_training_state(
        [
            {"loss": 0.0, "grad_norm": float("nan"), "step": 2},
            {"loss": 0.4, "grad_norm": 0.089, "step": 9},
        ]
    )


def test_non_finite_final_metric_is_still_rejected() -> None:
    import pytest

    from soup_cli.trainer.sft import _assert_finite_training_state

    with pytest.raises(
        RuntimeError,
        match=r"non-finite training metric grad_norm=nan.*step 9.*refusing to save",
    ):
        _assert_finite_training_state(
            [
                {"loss": 0.4, "grad_norm": 1.1, "step": 2},
                {"loss": 0.0, "grad_norm": float("nan"), "step": 9},
            ]
        )


def test_recovered_metric_checked_independently_per_metric() -> None:
    """A self-corrected ``loss`` must not mask a still-broken ``grad_norm``."""
    import pytest

    from soup_cli.trainer.sft import _assert_finite_training_state

    with pytest.raises(RuntimeError, match=r"grad_norm=nan.*step 9"):
        _assert_finite_training_state(
            [
                {"loss": float("nan"), "grad_norm": 0.5, "step": 2},
                {"loss": 0.4, "grad_norm": float("nan"), "step": 9},
            ]
        )
