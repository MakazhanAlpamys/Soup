"""Issue #1609: unclamped importance ratio in grpo_variants causes finite loss + NaN gradients.

In `src/soup_cli/utils/grpo_variants.py`, under extreme policy divergence,
`log_ratio = logp_new - logp_old` causes `torch.exp(log_ratio)` to overflow to
`+inf` in half-precision (bfloat16 / float16). Downstream surrogate clipping or
reduction masks could yield a finite scalar forward loss, but PyTorch autograd's
backward chain rule evaluated `0.0 * (+inf) = NaN`, polluting all parameter gradients.

Additionally, `_masked_mean` and `dr_grpo` previously used arithmetic masking
(`token_loss * completion_mask`), which produced `+inf * 0.0 = NaN` in forward
when padding tokens experienced overflow.

Hardened with:
1. Dynamic upper bound clamping on derived `log_ratio` (and `seq_log_ratio` in gspo,
   and `ref_minus_logp` in `_per_token_kl`) adapted to advantages
   `max_log_ratio - log(max(|A|, 1))` prior to `torch.exp()`.
2. Boolean masking `torch.where(completion_mask.bool(), token_loss, torch.zeros_like(token_loss))`
   across `_masked_mean`, `dr_grpo`, `gspo`, and `rft`.
"""

from __future__ import annotations

import math

import pytest

from soup_cli.utils.grpo_variants import _masked_mean, apply_variant_loss, variant_kl_metric

torch = pytest.importorskip("torch")

VARIANTS_WITH_RATIO = ("dapo", "dr_grpo", "bnpo", "two_sided", "gspo")
ALL_ACTIVE_VARIANTS = ("dapo", "dr_grpo", "bnpo", "two_sided", "gspo", "rft")
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
@pytest.mark.parametrize("dtype", DTYPES)
def test_issue1609_extreme_divergence_finite_loss_and_grad(variant, dtype):
    """Under extreme delta_logp (overflowing exp), loss and gradients must stay finite."""
    finfo = torch.finfo(dtype)
    delta_logp = math.log(finfo.max) + 5.0

    old = torch.tensor([[-20.0, -20.0]], dtype=dtype)
    new = torch.tensor([[-20.0 + delta_logp, -20.0 + delta_logp]], dtype=dtype, requires_grad=True)
    adv = torch.tensor([[1.0, 1.0]], dtype=dtype)
    # Token 0 is completion; Token 1 is padding (completion_mask = 0.0)
    mask = torch.tensor([[1.0, 0.0]], dtype=dtype)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )

    assert torch.isfinite(loss).item(), f"{variant} ({dtype}) loss {loss} is not finite"
    loss.backward()

    assert new.grad is not None, f"{variant} ({dtype}) grad is None"
    assert torch.isfinite(new.grad).all().item(), f"{variant} ({dtype}) non-finite: {new.grad}"
    # Masked token (index 1) must receive exactly 0.0 gradient, never NaN
    assert new.grad[0, 1].item() == 0.0, f"{variant} ({dtype}) mask grad non-zero: {new.grad}"


@pytest.mark.parametrize("variant", ALL_ACTIVE_VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_issue1609_normal_range_matches_canonical_math(variant, dtype):
    """Within normal ratio bounds (|delta_logp| <= 1.0), clamping is an exact identity."""
    old = torch.tensor([[-1.5, -2.0]], dtype=dtype)
    new = torch.tensor([[-1.3, -2.2]], dtype=dtype, requires_grad=True)
    adv = torch.tensor([[0.8, -0.4]], dtype=dtype)
    mask = torch.tensor([[1.0, 1.0]], dtype=dtype)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(new.grad).all().item()

    log_ratio = new.detach() - old
    ratio = torch.exp(log_ratio)
    tol = 1e-3 if dtype == torch.float16 else 1e-5
    if variant == "dapo":
        clipped = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.28)
        expected = -torch.min(ratio * adv, clipped * adv).mean()
        assert torch.allclose(loss.detach(), expected, atol=tol)
    elif variant == "dr_grpo":
        expected = -(ratio * adv).sum(dim=-1).mean()
        assert torch.allclose(loss.detach(), expected, atol=tol)


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_kl_penalty_extreme_divergence_stays_finite(variant):
    """When beta > 0 and reference policy severely diverges, KL penalty does not leak NaN."""
    # ref - new = 90.0 in bfloat16
    ref = torch.tensor([[+70.0, +70.0]], dtype=torch.bfloat16)
    new = torch.tensor([[-20.0, -20.0]], dtype=torch.bfloat16, requires_grad=True)
    old = torch.tensor([[-20.0, -20.0]], dtype=torch.bfloat16)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        beta=0.04,
        delta=0.2,
        reference_logp=ref,
    )
    loss.backward()

    assert torch.isfinite(loss).item(), f"{variant} KL loss is not finite: {loss}"
    assert torch.isfinite(new.grad).all().item(), f"{variant} KL grad contains NaN: {new.grad}"
    assert new.grad[0, 1].item() == 0.0


def test_issue1609_boolean_masking_blocks_nan_backward():
    """_masked_mean with non-finite values at masked positions does not leak NaN."""
    token_loss = torch.tensor([1.5, float("inf")], requires_grad=True)
    mask = torch.tensor([1.0, 0.0])

    mean_loss = _masked_mean(token_loss, mask)
    mean_loss.backward()

    assert torch.isfinite(mean_loss).item()
    assert mean_loss.item() == 1.5
    assert token_loss.grad is not None
    assert token_loss.grad[0].item() == 1.0
    assert token_loss.grad[1].item() == 0.0


@pytest.mark.parametrize("variant", ALL_ACTIVE_VARIANTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mask_dtype", (None, "same", torch.int64))
def test_issue1609_half_precision_loss_keeps_input_dtype(variant, dtype, mask_dtype):
    """Half-precision inputs preserve their dtype through GSPO/RFT and other reductions."""
    old = torch.tensor([[-1.5, -2.0]], dtype=dtype)
    new = torch.tensor([[-1.3, -2.2]], dtype=dtype, requires_grad=True)
    adv = torch.tensor([[0.8, 0.4]], dtype=dtype)
    if mask_dtype is None:
        mask = None
    elif mask_dtype == "same":
        mask = torch.ones(1, 2, dtype=dtype)
    else:
        mask = torch.ones(1, 2, dtype=mask_dtype)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert loss.dtype == dtype


@pytest.mark.parametrize("variant", ("dapo", "dr_grpo", "bnpo", "two_sided", "gspo"))
def test_issue1609_fp16_moderate_ratio_is_not_clamped(variant):
    """In float16, delta logp = 8.0 (exp = 2981) with A = -1.0 is not clamped prematurely."""
    old = torch.tensor([[-9.0, -2.0]], dtype=torch.float16)
    new = torch.tensor([[-1.0, -2.0]], dtype=torch.float16, requires_grad=True)
    adv = torch.tensor([[-1.0, -1.0]], dtype=torch.float16)
    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=torch.ones(1, 2, dtype=torch.float16),
        delta=0.2,
    )
    loss.backward()
    assert new.grad[0, 0].item() != 0.0


def test_issue1609_fp16_kl_gradient_survives_moderate_gap():
    """In float16, reference - policy = 7 nats preserves pull-back gradient (<-50.0)."""
    logp = torch.tensor([[-8.0, -1.0]], dtype=torch.float16, requires_grad=True)
    loss = apply_variant_loss(
        "dapo",
        logp_old=logp.detach(),
        logp_new=logp,
        advantages=torch.zeros(1, 2, dtype=torch.float16),
        completion_mask=torch.ones(1, 2, dtype=torch.float16),
        beta=0.1,
        reference_logp=torch.tensor([[-1.0, -1.0]], dtype=torch.float16),
    )
    loss.backward()
    assert logp.grad[0, 0].item() < -50.0


@pytest.mark.parametrize("variant", ALL_ACTIVE_VARIANTS)
def test_issue1609_all_masked_batch_degenerate_safe(variant):
    """Completely masked batch (lengths = 0) returns structural zero safely."""
    old = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
    new = torch.tensor([[-1.0, -2.0]], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert new.grad is None or torch.isfinite(new.grad).all().item()


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_negative_advantages_extreme_divergence(variant):
    """Extreme divergence under negative advantages remains finite and gradient safe."""
    old = torch.tensor([[-20.0, -20.0]], dtype=torch.bfloat16)
    new = torch.tensor([[+70.0, +70.0]], dtype=torch.bfloat16, requires_grad=True)
    adv = torch.tensor([[-5.0, -5.0]], dtype=torch.bfloat16)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert new.grad is None or torch.isfinite(new.grad).all().item()
    if new.grad is not None:
        assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_extreme_underflow_ratio_zero(variant):
    """Extreme underflow (delta_logp = -1000.0) yields finite loss and valid gradients."""
    old = torch.tensor([[+500.0, +500.0]], dtype=torch.float32)
    new = torch.tensor([[-500.0, -500.0]], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad).all().item()
    assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_kl_with_nan_in_reference_at_padding(variant):
    """NaN in reference_logp at masked padding positions does not corrupt loss or gradients."""
    ref = torch.tensor([[-1.0, float("nan")]], dtype=torch.float32)
    old = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    new = torch.tensor([[-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        beta=0.04,
        delta=0.2,
        reference_logp=ref,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad).all().item()
    assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_negative_advantages_headroom_no_overflow(variant):
    """Large positive divergence with negative advantages does not overflow to -inf in bfloat16."""
    old = torch.tensor([[0.0, 0.0]], dtype=torch.bfloat16)
    new = torch.tensor([[85.0, 85.0]], dtype=torch.bfloat16, requires_grad=True)
    adv = torch.tensor([[-5.0, -5.0]], dtype=torch.bfloat16)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)

    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad).all().item()
    assert new.grad[0, 1].item() == 0.0


def test_issue1609_rft_padding_tokens_with_neg_inf():
    """RFT with -inf logp_new at padding produces finite loss and gradients."""
    old = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    # Token 1 is valid, token 2 is padding with -inf log prob
    new = torch.tensor([[-1.0, -float("inf")]], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = apply_variant_loss(
        "rft",
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad[0, 0]).item()
    assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("variant", ["dapo", "dr_grpo", "bnpo", "two_sided", "gspo", "rft"])
def test_issue1609_variant_kl_metric_with_padding_extreme_values(variant):
    """variant_kl_metric does not leak NaN when padding positions have extreme divergence."""
    ref = torch.tensor([[-1.0, 100.0]], dtype=torch.float32)
    new = torch.tensor([[-1.0, -100.0]], dtype=torch.float32)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    metric = variant_kl_metric(
        variant,
        logp_new=new,
        advantages=adv,
        beta=0.04,
        completion_mask=mask,
        reference_logp=ref,
    )
    assert metric is not None
    assert torch.isfinite(metric).item()


def test_issue1609_gspo_empty_sequence_in_batch():
    """GSPO handles batches where a sequence has length 0 without producing NaN."""
    old = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    new = torch.tensor([[0.1, 0.2], [90.0, 90.0]], dtype=torch.float32, requires_grad=True)
    adv = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    # Sequence 0 has 2 tokens; sequence 1 is completely masked out
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    loss = apply_variant_loss(
        "gspo",
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad[0]).all().item()
    assert (new.grad[1] == 0.0).all().item()


@pytest.mark.parametrize("variant", VARIANTS_WITH_RATIO)
def test_issue1609_fp16_large_advantage_ratio_remains_finite(variant):
    """Large advantages (|A| = 5.0) with large ratio (delta = 9.5) remain finite in float16."""
    old = torch.tensor([[-10.5, -2.0]], dtype=torch.float16)
    new = torch.tensor([[-1.0, -2.0]], dtype=torch.float16, requires_grad=True)
    adv = torch.tensor([[-5.0, -5.0]], dtype=torch.float16)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=new,
        advantages=adv,
        completion_mask=mask,
        delta=0.2,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(new.grad).all().item()
    assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("variant", ALL_ACTIVE_VARIANTS)
def test_issue1609_variant_kl_metric_with_nan_reference_at_padding(variant):
    """variant_kl_metric does not produce NaN when padding positions have NaN reference_logp."""
    ref = torch.tensor([[-1.0, float("nan")]], dtype=torch.float32)
    new = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    adv = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    metric = variant_kl_metric(
        variant,
        logp_new=new,
        advantages=adv,
        beta=0.04,
        completion_mask=mask,
        reference_logp=ref,
    )
    assert metric is not None
    assert torch.isfinite(metric).item()


@pytest.mark.parametrize("variant", ("dapo", "dr_grpo", "bnpo", "two_sided", "gspo", "rft"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_issue1609_variant_kl_metric_counts_past_half_precision(variant, dtype):
    # 20 x 4000 = 80,000 completion tokens: float16 cannot count that many.
    shape = (20, 4000)
    mask = torch.ones(shape, dtype=torch.int32)
    mask[:, -1] = 0
    metric = variant_kl_metric(
        variant,
        logp_new=torch.full(shape, -1.0, dtype=dtype),
        advantages=torch.ones(20),
        beta=0.04,
        completion_mask=mask,
        reference_logp=torch.full(shape, -2.0, dtype=dtype),
    )
    # reference - policy = -1 on every token: kl_t = exp(-1)
    assert metric.item() == pytest.approx(math.exp(-1.0), rel=1e-2)


def test_issue1609_gspo_padded_token_values_cannot_reach_the_loss():
    new = torch.tensor([[-1.2, -1.0]], requires_grad=True)
    loss = apply_variant_loss(
        "gspo",
        logp_old=torch.tensor([[-1.0, float("-inf")]]),
        logp_new=new,
        advantages=torch.tensor([[0.5, float("nan")]]),
        completion_mask=torch.tensor([[1, 0]]),
    )
    loss.backward()
    assert torch.isfinite(loss).item() and torch.isfinite(new.grad).all().item()


def test_issue1609_gspo_empty_sequence_with_nan_advantage_is_dropped():
    new = torch.tensor([[-1.2, -0.8], [-1.0, -1.0]], requires_grad=True)
    loss = apply_variant_loss(
        "gspo",
        logp_old=torch.full((2, 2), -1.0),
        logp_new=new,
        advantages=torch.tensor([0.5, float("nan")]),
        completion_mask=torch.tensor([[1, 1], [0, 0]]),
    )
    loss.backward()
    assert torch.isfinite(loss).item() and torch.isfinite(new.grad).all().item()


@pytest.mark.parametrize("variant", ("dapo", "gspo"))
def test_issue1609_clamp_bound_is_not_differentiated_through_advantages(variant):
    finfo = torch.finfo(torch.float32)
    old = torch.tensor([[-20.0, -20.0]])
    adv = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    loss = apply_variant_loss(
        variant,
        logp_old=old,
        logp_new=old + math.log(finfo.max) + 5.0,
        advantages=adv,
        completion_mask=torch.ones(1, 2),
        delta=0.2,
    )
    loss.backward()
    clamped_ratio = math.exp(math.log(finfo.max) - 1.0 - math.log(2.0))
    assert adv.grad.tolist()[0] == pytest.approx([-clamped_ratio / 2] * 2, rel=1e-3)


@pytest.mark.parametrize("variant", ("dapo", "dr_grpo", "bnpo", "two_sided", "gspo", "rft"))
def test_issue1609_variant_kl_metric_sums_past_half_precision(variant):
    # 20 x 5000 tokens, kl_t = exp(1) - 2 = 0.718: the SUM (71,800) is past float16's 65,504.
    shape = (20, 5000)
    metric = variant_kl_metric(
        variant,
        logp_new=torch.full(shape, -1.0, dtype=torch.float16),
        advantages=torch.ones(20),
        beta=0.04,
        completion_mask=torch.ones(shape, dtype=torch.int32),
        reference_logp=torch.full(shape, 0.0, dtype=torch.float16),
    )
    assert metric.item() == pytest.approx(math.e - 2.0, rel=1e-2)


def test_issue1609_variant_kl_metric_keeps_float64():
    shape = (3, 5)
    wiggle = 1e-9 * torch.arange(15, dtype=torch.float64).reshape(shape)
    metric = variant_kl_metric(
        "dapo",
        logp_new=torch.full(shape, -1.0, dtype=torch.float64) + wiggle,
        advantages=torch.ones(3, dtype=torch.float64),
        beta=0.1,
        completion_mask=torch.ones(shape, dtype=torch.int32),
        reference_logp=torch.zeros(shape, dtype=torch.float64),
    )
    assert metric.dtype == torch.float64
