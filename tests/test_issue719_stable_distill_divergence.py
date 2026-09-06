"""Regression coverage for stable reverse-KL and JS distillation (#719)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("divergence", ["reverse_kl", "js"])
@pytest.mark.parametrize("mask_kind", ["labels", "attention_mask"])
def test_low_temperature_loss_and_gradients_stay_finite(
    dtype_name: str, divergence: str, mask_kind: str
) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    dtype = getattr(torch, dtype_name)
    student = torch.tensor(
        [[[0.0, 0.0], [0.0, -6.0], [0.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    teacher = torch.tensor([[[0.0, 0.0], [0.0, -0.5], [0.0, 0.0]]], dtype=dtype)
    mask = torch.tensor([[0, 0, 1]])
    kwargs = (
        {"labels": mask.masked_fill(mask == 0, -100)}
        if mask_kind == "labels"
        else {"attention_mask": mask}
    )

    loss = _compute_distill_term(student, teacher, divergence, temperature=0.05, **kwargs)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad[:, 0, :]) == 0
    assert torch.count_nonzero(student.grad[:, 2, :]) == 0


def test_forward_kl_matches_probability_space_reference() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    student = torch.tensor([[[0.2, -0.4, 0.8]]], requires_grad=True)
    teacher = torch.tensor([[[0.5, 0.1, -0.2]]])
    temperature = 2.0
    log_student = torch.log_softmax(student / temperature, dim=-1)
    teacher_prob = torch.softmax(teacher / temperature, dim=-1)
    expected = (
        torch.nn.functional.kl_div(log_student, teacher_prob, reduction="batchmean")
        * temperature**2
    )

    actual = _compute_distill_term(student, teacher, "forward_kl", temperature=temperature)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("which", ["student", "teacher"])
def test_non_finite_input_is_reported_separately(which: str) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    student = torch.zeros(1, 1, 2)
    teacher = torch.zeros(1, 1, 2)
    target = student if which == "student" else teacher
    target[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match=rf"{which}_logits.*finite"):
        _compute_distill_term(student, teacher, "reverse_kl", temperature=1.0)
