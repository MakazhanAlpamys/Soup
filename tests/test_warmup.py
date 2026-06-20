import pytest
from soup_cli.utils.warmup import compute_warmup_steps, MIN_WARMUP, MAX_WARMUP


# --- Happy path ---

def test_typical_inputs_returns_clamped_value():
    result = compute_warmup_steps(
        num_examples=1000, batch_size=8, grad_accum=1, epochs=3
    )
    assert MIN_WARMUP <= result <= MAX_WARMUP


def test_default_ratio_used_when_not_specified():
    result = compute_warmup_steps(1000, 8, 1, 3)
    assert isinstance(result, int)


# --- ratio=0 special case ---

def test_zero_ratio_returns_zero():
    result = compute_warmup_steps(1000, 8, 1, 3, ratio=0.0)
    assert result == 0


# --- Clamping ---

def test_clamps_to_min_warmup_for_tiny_dataset():
    result = compute_warmup_steps(
        num_examples=10, batch_size=8, grad_accum=1, epochs=1, ratio=0.03
    )
    assert result == MIN_WARMUP


def test_clamps_to_max_warmup_for_huge_dataset():
    result = compute_warmup_steps(
        num_examples=1_000_000, batch_size=1, grad_accum=1, epochs=10, ratio=0.5
    )
    assert result == MAX_WARMUP


# --- Bounds ---

def test_max_ratio_accepted():
    result = compute_warmup_steps(1000, 8, 1, 3, ratio=0.5)
    assert MIN_WARMUP <= result <= MAX_WARMUP


def test_grad_accum_reduces_steps():
    result_no_accum = compute_warmup_steps(1000, 8, 1, 3)
    result_with_accum = compute_warmup_steps(1000, 8, 4, 3)
    assert result_with_accum <= result_no_accum


# --- Edge cases ---

def test_single_example():
    result = compute_warmup_steps(1, 1, 1, 1)
    assert result == MIN_WARMUP


def test_single_epoch():
    result = compute_warmup_steps(500, 8, 1, 1)
    assert MIN_WARMUP <= result <= MAX_WARMUP


# --- Validation errors ---

def test_raises_for_zero_examples():
    with pytest.raises(ValueError, match="num_examples"):
        compute_warmup_steps(0, 8, 1, 3)


def test_raises_for_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        compute_warmup_steps(1000, 0, 1, 3)


def test_raises_for_zero_grad_accum():
    with pytest.raises(ValueError, match="grad_accum"):
        compute_warmup_steps(1000, 8, 0, 3)


def test_raises_for_zero_epochs():
    with pytest.raises(ValueError, match="epochs"):
        compute_warmup_steps(1000, 8, 1, 0)


def test_raises_for_ratio_above_max():
    with pytest.raises(ValueError, match="ratio"):
        compute_warmup_steps(1000, 8, 1, 3, ratio=0.6)


def test_raises_for_negative_ratio():
    with pytest.raises(ValueError, match="ratio"):
        compute_warmup_steps(1000, 8, 1, 3, ratio=-0.1)