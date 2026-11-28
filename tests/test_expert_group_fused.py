"""Packed NF4 bounds and CUDA launches for grouped forward and input gradients."""

from __future__ import annotations

from dataclasses import replace

import pytest

from soup_cli.utils.expert_group_fused import PackedNF4, plan_fused_group


@pytest.mark.parametrize("rows", [[True], [-1], [513], [1] * 5])
def test_fused_plan_refuses_invalid_rows(rows):
    with pytest.raises(ValueError):
        plan_fused_group(rows, (32, 64))


def test_call_budget_includes_outputs_descriptors_and_one_dense_fallback():
    shape = (2048, 1024)
    plan = plan_fused_group([0, 128, 256, 512], shape, native_small_rows=True, transpose=True)
    assert plan["output_bytes"] == (128 + 256 + 512) * 1024 * 2
    assert plan["descriptor_bytes"] == 512
    assert plan["dense_weight_bytes"] == 2048 * 1024 * 2
    assert plan["call_bytes"] == sum(
        plan[field] for field in ("output_bytes", "descriptor_bytes", "dense_weight_bytes")
    )
    assert (plan["fallback_experts"], plan["fused_experts"]) == (1, 2)


def _packed(torch, shape, *, nested, device):
    rows, columns = shape
    blocks = rows * columns // 64
    # High nibble 1 and low nibble 2. These values make forward and transposed
    # sums exactly representable, so decoding or orientation bugs are visible.
    packed = torch.full((rows * columns // 2,), 0x12, dtype=torch.uint8, device=device)
    code = torch.zeros(16, dtype=torch.float32, device=device)
    code[1], code[2] = 0.5, -0.25
    if not nested:
        return PackedNF4(packed, torch.ones(blocks, device=device), code, shape)
    nested_code = torch.zeros(256, device=device)
    nested_code[3] = 0.5
    return PackedNF4(
        packed, torch.full((blocks,), 3, dtype=torch.uint8, device=device), code, shape,
        nested_scales=torch.ones((blocks + 255) // 256, device=device),
        nested_code=nested_code, offset=torch.tensor([0.5], device=device),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_packed_metadata_validation_and_budget_boundary(nested):
    import torch

    from soup_cli.utils.expert_group_fused import fused_group_linear, validate_fused_group

    weight = _packed(torch, (32, 64), nested=nested, device="cpu")
    inputs = [torch.ones((17, 64), dtype=torch.bfloat16)]
    budget = plan_fused_group([17], weight.shape)["call_bytes"]
    assert validate_fused_group(inputs, [weight], call_budget_bytes=budget)["call_bytes"] == budget
    with pytest.raises(ValueError, match="fused group exceeds call budget before allocation"):
        validate_fused_group(inputs, [weight], call_budget_bytes=budget - 1)
    with pytest.raises(ValueError, match="invalid frozen NF4 tensor extent"):
        validate_fused_group(
            inputs, [replace(weight, packed=weight.packed[:-1])], call_budget_bytes=budget,
        )
    with pytest.raises(ValueError, match="requires CUDA compute capability"):
        fused_group_linear(inputs, [weight], call_budget_bytes=budget)


@pytest.mark.gpu(reason="packed expert kernels require CUDA")
@pytest.mark.parametrize("backend", ["triton", "wmma"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_cuda_grouped_packed_forward_and_transposed_projection(backend, nested, transpose):
    import sys

    import torch

    pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires native CUDA BF16")
    if backend == "wmma" and sys.platform != "linux":
        pytest.skip("WMMA uses Linux NVRTC and the CUDA driver")
    from soup_cli.utils.expert_group_fused import fused_group_linear

    device = torch.device("cuda", torch.cuda.current_device())
    weights = [_packed(torch, (32, 64), nested=nested, device=device) for _ in range(3)]
    others = weights[1].code
    others[1], others[2] = 0.25, -0.5
    inputs = [
        torch.ones((rows, 32 if transpose else 64), dtype=torch.bfloat16, device=device)
        for rows in (129, 17, 0)
    ]
    outputs = fused_group_linear(
        inputs, weights, transpose=transpose, call_budget_bytes=2**20, backend=backend,
    )
    for states, weight, output in zip(inputs, weights, outputs):
        dense = weight.code[1:3].repeat(1024).reshape(32, 64).to(torch.bfloat16)
        expected = states @ (dense if transpose else dense.T)
        assert torch.equal(output, expected)


@pytest.mark.parametrize("tuple_api", [False, True])
def test_bf16_policy_uses_the_torch_api_variant(tuple_api):
    from soup_cli.utils.expert_group_gemm import bf16_policy, set_bf16_policy

    class Matmul:
        allow_bf16_reduced_precision_reduction_split_k = True
        reduction = (True, True) if tuple_api else True

        @property
        def allow_bf16_reduced_precision_reduction(self):
            return self.reduction

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value):
            if isinstance(value, tuple) != tuple_api:
                raise RuntimeError("wrong setting type")
            self.reduction = value

    matmul = Matmul()
    assert bf16_policy(matmul) == (True, True)
    set_bf16_policy(matmul, False, False)
    assert bf16_policy(matmul) == (False, False)
    set_bf16_policy(matmul, True, True)
    assert bf16_policy(matmul) == (True, True)


@pytest.mark.gpu(reason="packed expert training requires CUDA")
@pytest.mark.parametrize("two_reads", [False, True])
def test_cuda_packed_expert_adapter_step(tmp_path, two_reads):
    import torch

    pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires native CUDA BF16")
    pytest.importorskip("bitsandbytes")
    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.expert_training import ExpertGroupCompute
    from soup_cli.utils.layer_shard import NF4_CODE_KEY

    device = f"cuda:{torch.cuda.current_device()}"
    torch.manual_seed(17)
    path = str(tmp_path / "experts.safetensors")
    pairs = (
        (torch.randn(64, 32, dtype=torch.bfloat16) / 16,
         torch.randn(32, 32, dtype=torch.bfloat16) / 16)
        for _ in range(2)
    )
    info = write_expert_shard(path, pairs, quantise=True, double_quant=False, device=device)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype="bfloat16",
    )
    codes = {NF4_CODE_KEY: info.codes[0]}
    options = ExpertStreamConfig(backend="fused_nf4", native_rows_from=None, group_width=2)
    dense_pool = ExpertBufferPool(
        reader, device=device, dtype=reader.dtype, codes=codes,
        weight_budget_bytes=options.weight_budget_bytes,
    )
    packed_pool = ExpertBufferPool(
        reader, device=device, dtype=reader.dtype, codes=codes,
        weight_budget_bytes=options.weight_budget_bytes, group_width=2, packed_only=True,
    )
    try:
        packed_pool.group_compute = ExpertGroupCompute(
            reader, options, device=device, dtype=reader.dtype,
        )
        cls = streamed_experts_class()
        modules = [
            cls(reader, pool, torch.nn.SiLU(), 0, two_reads=two_reads)
            for pool in (dense_pool, packed_pool)
        ]
        index = torch.arange(258, device=device).reshape(-1, 1) % 2
        hidden = torch.randn(258, 32, dtype=torch.bfloat16, device=device)
        parameters = [
            torch.nn.Parameter(torch.full((32,), 0.125, device=device)) for _ in modules
        ]
        optimizers = [torch.optim.SGD([parameter], lr=0.5) for parameter in parameters]
        for _ in range(2):
            observed = []
            for module, parameter, optimizer in zip(modules, parameters, optimizers):
                optimizer.zero_grad()
                route = (hidden.float() @ parameter).sigmoid().to(hidden.dtype)[:, None]
                output = module(hidden + parameter.to(hidden.dtype), index, route)
                output.float().square().mean().backward()
                observed.append((output.detach(), parameter.grad.detach().clone()))
                optimizer.step()
            for actual, reference in zip(observed[1], observed[0]):
                torch.testing.assert_close(actual, reference, rtol=0.03, atol=2e-4)
            torch.testing.assert_close(parameters[1], parameters[0], rtol=0.001, atol=2e-5)
        assert not torch.equal(parameters[1], torch.full_like(parameters[1], 0.125))
        assert packed_pool.plan.dense_bytes == 0 and packed_pool.groups_loaded == 2
    finally:
        packed_pool.close()
        dense_pool.close()
        reader.close()
