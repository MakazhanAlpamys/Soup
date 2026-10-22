"""Real native KD values/gradients against independent float64 arithmetic."""

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from soup_cli.trainer.mlx_distill_loss import token_loss_sums  # noqa: E402


def fixture(dtype=mx.float32):
    rng = np.random.default_rng(17)
    student = mx.array(rng.normal(size=(2, 4, 5)), dtype=dtype)
    teacher = mx.array(rng.normal(size=(2, 4, 5)), dtype=dtype)
    labels = mx.array([[-100, 2, -100, 1], [3, 1, 0, -100]])
    # Indices refer to the flattened shifted labels, not input positions.
    active = mx.array([0, 2, 3, 4])
    return student, teacher, labels, active


def oracle(student, teacher, labels, divergence, temperature):
    """Independent probability-space float64 reference; no production helpers."""
    student, teacher = (np.asarray(x, dtype=np.float64)[:, :-1] for x in (student, teacher))
    targets = np.asarray(labels)[:, 1:]
    active = targets != -100
    student, teacher, targets = student[active], teacher[active], targets[active]
    if not len(targets):
        return np.zeros(3)

    def logprob(x):
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    ce = -logprob(student)[np.arange(len(targets)), targets].sum()
    ps, pt = (np.exp(logprob(x / temperature)) for x in (student, teacher))
    # The ordinary fixtures avoid underflow, so these independent formulas
    # can use explicit probabilities rather than repeating the native log-space kernel.
    if divergence == "forward_kl":
        kd = (pt * np.log(pt / ps)).sum()
    elif divergence == "reverse_kl":
        kd = (ps * np.log(ps / pt)).sum()
    else:
        mixture = (ps + pt) / 2
        kd = ((ps * np.log(ps / mixture)).sum() + (pt * np.log(pt / mixture)).sum()) / 2
    kd *= temperature**2
    return np.array([(ce + kd) / 2, ce, kd])


def numerical_gradient(student, teacher, labels, divergence, temperature):
    student = np.asarray(student, dtype=np.float64).copy()
    result = np.zeros_like(student)
    delta = 1e-4
    for index in np.ndindex(student.shape):
        original = student[index]
        student[index] = original + delta
        plus = oracle(student, teacher, labels, divergence, temperature)[0]
        student[index] = original - delta
        minus = oracle(student, teacher, labels, divergence, temperature)[0]
        student[index] = original
        result[index] = (plus - minus) / (2 * delta)
    return result


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("temperature", [0.05, 2.0, 100.0])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_values_and_gradients_match_float64_oracle(divergence, temperature, dtype):
    student, teacher, labels, active = fixture(dtype)

    def objective(logits):
        return token_loss_sums(logits, teacher, labels, active,
                               divergence=divergence, temperature=temperature)

    (total, metrics), grad = mx.value_and_grad(objective)(student)
    rounded_s, rounded_t = (np.array(v.astype(mx.float32)) for v in (student, teacher))
    expected = oracle(rounded_s, rounded_t, np.array(labels), divergence, temperature)
    # FP32 cancellation in KL/JS is magnified by T^2 near the upper bound.
    tolerance = 5e-7 * temperature**2 * len(active) + 3e-6
    np.testing.assert_allclose(
        [total.item(), metrics["ce_sum"].item(), metrics["kd_sum"].item()],
        expected, atol=tolerance, rtol=2e-5,
    )
    expected_grad = numerical_gradient(
        rounded_s, rounded_t, np.array(labels), divergence, temperature,
    )
    # The derivative is returned in the input dtype; BF16 introduces its own rounding.
    np.testing.assert_allclose(np.array(grad.astype(mx.float32)), expected_grad,
                               atol=3e-3 if dtype == mx.bfloat16 else 3e-4, rtol=7e-3)
    assert total.dtype == mx.float32
    assert metrics["teacher_finite"].item() and metrics["student_finite"].item()


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7])
@pytest.mark.parametrize("recompute", [False, True])
def test_chunk_and_recompute_preserve_values_and_gradients(divergence, chunk_size, recompute):
    student, teacher, labels, active = fixture()

    def run(chunk, checkpoint):
        return mx.value_and_grad(lambda s: token_loss_sums(
            s, teacher, labels, active, divergence=divergence, temperature=2,
            chunk_size=chunk, recompute=checkpoint,
        ))(student)

    (expected, expected_metrics), expected_grad = run(None, False)
    (actual, metrics), grad = run(chunk_size, recompute)
    assert mx.allclose(actual, expected, atol=2e-6).item()
    assert mx.allclose(grad, expected_grad, atol=2e-6).item()
    assert mx.allclose(metrics["ce_sum"], expected_metrics["ce_sum"], atol=2e-6).item()
    assert mx.allclose(metrics["kd_sum"], expected_metrics["kd_sum"], atol=2e-6).item()


def test_identical_distributions_keep_ce_and_freeze_teacher_gradient():
    student, _, labels, active = fixture()
    (loss, metrics), grads = mx.value_and_grad(
        lambda s, t: token_loss_sums(s, t, labels, active), argnums=(0, 1),
    )(student, student)
    assert abs(metrics["kd_sum"].item()) < 1e-6
    assert metrics["ce_sum"].item() > 0
    assert mx.allclose(loss, metrics["ce_sum"] / 2).item()
    assert mx.any(grads[0] != 0).item()
    assert mx.all(grads[1] == 0).item()


@pytest.mark.parametrize("recompute", [False, True])
def test_empty_selection_is_finite_differentiable_zero(recompute):
    student, teacher, labels, _ = fixture()
    empty = mx.array([], dtype=mx.int32)
    (loss, metrics), grad = mx.value_and_grad(lambda s: token_loss_sums(
        s, teacher, labels, empty, recompute=recompute, chunk_size=1,
    ))(student)
    assert loss.item() == metrics["ce_sum"].item() == metrics["kd_sum"].item() == 0
    assert mx.all(grad == 0).item()


def test_masked_and_last_logits_have_no_effect_but_active_boundary_does():
    student, teacher, labels, active = fixture()
    baseline = token_loss_sums(student, teacher, labels, active)[0].item()
    masked = mx.array(student)
    masked[:, -1] = float("nan")
    masked[0, 1] = float("inf")
    assert token_loss_sums(masked, teacher, labels, active)[0].item() == baseline
    changed = mx.array(student)
    changed[0, 0, 2] += 3
    assert abs(token_loss_sums(changed, teacher, labels, active)[0].item() - baseline) > 0.1


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("temperature", [0.05, 100.0])
def test_extreme_finite_logits_and_underflow_keep_finite_loss_and_gradients(
    divergence, temperature,
):
    student = mx.array([[[10000.0, -10000.0, 0.0], [0.0, 0.0, 0.0]]])
    teacher = -student
    labels, active = mx.array([[-100, 2]]), mx.array([0])
    (loss, metrics), grad = mx.value_and_grad(lambda s: token_loss_sums(
        s, teacher, labels, active, divergence=divergence, temperature=temperature,
        chunk_size=1, recompute=True,
    ))(student)
    assert mx.isfinite(loss).item() and mx.all(mx.isfinite(grad)).item()
    assert metrics["student_finite"].item() and metrics["teacher_finite"].item()


def test_temperature_does_not_change_supervised_ce():
    student, teacher, labels, active = fixture()
    values = [token_loss_sums(student, teacher, labels, active, temperature=t)[1]
              for t in (0.05, 2.0, 100.0)]
    assert values[0]["ce_sum"].item() == values[1]["ce_sum"].item() == values[2]["ce_sum"].item()
    assert abs(values[0]["kd_sum"].item() - values[1]["kd_sum"].item()) > 0.1
