"""Batch invariance of streamed expert linear and routing-weight gradients."""

import os
import subprocess
import sys

import pytest


def rowwise_matmul(left, right):
    import torch

    return torch.cat([row @ right for row in left.split(1)])


def rowwise_dot(left, right, dtype):
    import torch

    rows = [(a * b).sum(dim=-1, keepdim=True) for a, b in zip(left.split(1), right.split(1))]
    return torch.cat(rows).to(dtype)


def _check_linear(device, dtype):
    import torch

    from soup_cli.utils.invariant_gemm import linear

    generator = torch.Generator().manual_seed(23)
    rows = torch.randn(71, 137, generator=generator).to(device, dtype)
    weight = (torch.randn(79, 137, generator=generator) / 137**0.5).to(device, dtype)
    grad = torch.randn(71, 79, generator=generator).to(device, dtype)

    def run(start, stop):
        local = rows[start:stop].detach().requires_grad_(True)
        result = linear(local, weight)
        result.backward(grad[start:stop])
        return result.detach(), local.grad

    whole, whole_grad = run(0, len(rows))
    for start, stop in ((0, 1), (4, 9), (3, 39), (15, 71)):
        part, part_grad = run(start, stop)
        assert torch.equal(part, whole[start:stop])
        assert torch.equal(part_grad, whole_grad[start:stop])
    torch.testing.assert_close(
        whole.double(), rows.double() @ weight.double().t(), rtol=1e-2, atol=1e-2,
    )


def test_expert_matmul_is_independent_of_batch_shape_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from soup_cli.utils import invariant_gemm

    monkeypatch.setattr(invariant_gemm, "matmul", rowwise_matmul)
    _check_linear("cpu", torch.float32)


def test_triton_matmul_is_independent_of_batch_shape_under_cpu_interpreter():
    pytest.importorskip("torch")
    pytest.importorskip("triton")

    result = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; from tests.test_invariant_gemm import _check_linear; "
            "_check_linear('cpu', torch.float32)",
        ],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cuda_expert_matmul_is_independent_of_batch_shape(dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if dtype == "bfloat16" and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 needs compute capability 8.0+")
    _check_linear("cuda", getattr(torch, dtype))


@pytest.mark.gpu
def test_cuda_routing_weight_gradient_is_independent_of_batch_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from soup_cli.utils.invariant_gemm import scale_rows

    generator = torch.Generator().manual_seed(23)
    values = torch.randn(67, 2049, generator=generator).to("cuda", torch.float16)
    weights = torch.rand(67, 1, generator=generator).cuda()
    grad = torch.randn(67, 2049, generator=generator).cuda()

    def run(start, stop):
        local = weights[start:stop].detach().requires_grad_(True)
        out = scale_rows(values[start:stop], local)
        return torch.autograd.grad(out, local, grad[start:stop])[0]

    full = run(0, len(weights))
    for start, stop in ((0, 1), (4, 9), (3, 39), (15, 67)):
        assert torch.equal(run(start, stop), full[start:stop])
    reference = (values.float() * grad).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(full, reference, rtol=1e-5, atol=1e-4)
