"""Separate cross-backend parity; standalone MLX operation is tested without Torch."""

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("torch")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402

from soup_cli.trainer.distill import _compute_distill_term  # noqa: E402
from soup_cli.trainer.mlx_distill_loss import token_loss_sums  # noqa: E402


def compare(divergence, temperature, dtype, chunk, recompute, shape=(2, 4, 5)):
    rng = np.random.default_rng(34)
    student, teacher = (mx.array(rng.normal(size=shape), dtype=dtype) for _ in range(2))
    labels = rng.integers(0, shape[-1], size=shape[:2]).astype(np.int32)
    labels[:, 0] = -100
    labels[0, -1] = -100
    active = mx.array(np.flatnonzero(labels[:, 1:].reshape(-1) != -100).astype(np.int32))
    # Deliberately keep at least one active target for the short S=2 case.
    if active.size == 0:
        labels[0, -1] = 0
        active = mx.array([0])
    (native, metrics), grad = mx.value_and_grad(lambda s: token_loss_sums(
        s, teacher, mx.array(labels), active, divergence=divergence,
        temperature=temperature, chunk_size=chunk, recompute=recompute,
    ))(student)
    torch_dtype = {mx.float32: torch.float32, mx.float16: torch.float16,
                   mx.bfloat16: torch.bfloat16}[dtype]
    s = torch.tensor(np.array(student.astype(mx.float32)), dtype=torch_dtype, requires_grad=True)
    t = torch.tensor(np.array(teacher.astype(mx.float32)), dtype=torch_dtype)
    target = torch.tensor(labels, dtype=torch.long)
    kd = _compute_distill_term(
        s, t, divergence, temperature=temperature, labels=target,
        chunk_size=chunk, use_checkpoint=recompute,
    ) * active.size
    ce = functional.cross_entropy(s[:, :-1].float().reshape(-1, shape[-1]),
                                 target[:, 1:].reshape(-1), reduction="sum")
    expected = 0.5 * ce + 0.5 * kd
    expected.backward()
    # Float32 KL cancellation is amplified at the high supported T boundary.
    tolerance = 1e-6 * temperature**2 * active.size + 4e-6
    np.testing.assert_allclose(
        [native.item(), metrics["ce_sum"].item(), metrics["kd_sum"].item()],
        [expected.item(), ce.item(), kd.item()], atol=tolerance, rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.array(grad.astype(mx.float32)), s.grad.float().numpy(),
        atol=3e-3 if dtype == mx.bfloat16 else 4e-4, rtol=7e-3,
    )


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("temperature", [0.05, 2.0, 100.0])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_existing_transformers_objective_and_gradient(divergence, temperature, dtype):
    compare(divergence, temperature, dtype, None, False)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("chunk", [1, 3, 4, 9])
@pytest.mark.parametrize("recompute", [False, True])
def test_existing_chunked_recomputed_objective(divergence, chunk, recompute):
    compare(divergence, 2.0, mx.float32, chunk, recompute)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize("shape", [(1, 2, 2), (3, 7, 17)])
def test_other_batch_sequence_and_vocabulary_dimensions(divergence, shape):
    compare(divergence, 0.7, mx.float32, 3, True, shape)
