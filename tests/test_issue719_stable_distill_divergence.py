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
@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_non_finite_logits_propagate_to_amp(which, divergence, nonfinite) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    student = torch.zeros(1, 1, 2, requires_grad=True)
    teacher = torch.zeros(1, 1, 2)
    with torch.no_grad():
        (student if which == "student" else teacher)[0, 0, 0] = nonfinite

    loss = _compute_distill_term(student, teacher, divergence, temperature=1.0)
    loss.backward()

    assert not torch.isfinite(loss)
    assert not torch.isfinite(student.grad).all()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
def test_divergence_matches_double_precision_reference(dtype_name, divergence) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.trainer.distill import _compute_distill_term

    dtype = getattr(torch, dtype_name)
    student = torch.tensor([[[0.2, -0.4, 0.8]]], dtype=dtype, requires_grad=True)
    teacher = torch.tensor([[[0.5, 0.1, -0.2]]], dtype=dtype)
    temperature = 2.0
    # Use independent probability-space arithmetic on the actual rounded inputs.
    ps = torch.softmax(student.detach().double() / temperature, dim=-1)
    pt = torch.softmax(teacher.double() / temperature, dim=-1)
    if divergence == "forward_kl":
        expected = (pt * torch.log(pt / ps)).sum()
    elif divergence == "reverse_kl":
        expected = (ps * torch.log(ps / pt)).sum()
    else:
        mixture = (ps + pt) / 2
        expected = ((ps * torch.log(ps / mixture)).sum()
                    + (pt * torch.log(pt / mixture)).sum()) / 2
    expected *= temperature**2

    actual = _compute_distill_term(student, teacher, divergence, temperature=temperature)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-7)
