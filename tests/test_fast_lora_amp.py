"""Fast-LoRA under autocast: cast-preserving gradients, cast caching, optimizer trajectory."""

import copy
from types import SimpleNamespace

import pytest


@pytest.fixture
def amp_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(792)
    yield torch
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def make_model(torch, path):
    from peft import LoraConfig, inject_adapter_in_model

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shapes = {
                "single": {"o_proj": (32, 24)},
                "qkv": {"q_proj": (32, 32), "k_proj": (32, 16), "v_proj": (32, 16)},
                "mlp": {"gate_proj": (32, 64), "up_proj": (32, 64), "down_proj": (64, 32)},
            }[path]
            for name, (inputs, outputs) in shapes.items():
                setattr(self, name, torch.nn.Linear(inputs, outputs, dtype=torch.bfloat16))
            if path == "mlp":
                self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            if path == "single":
                return self.o_proj(x)
            if path == "qkv":
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    model = Block()
    model.config = SimpleNamespace(hidden_act="silu")
    targets = [name for name, _ in model.named_children() if name.endswith("proj")]
    inject_adapter_in_model(
        LoraConfig(r=3, lora_alpha=7, target_modules=targets, init_lora_weights=False), model
    )
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    return model


def patch(model, path):
    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    count = {
        "single": patch_fast_lora_single_projection,
        "qkv": patch_fast_lora_qkv,
        "mlp": patch_fast_lora_mlp,
    }[path](model)
    assert count == 1


@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
@pytest.mark.parametrize("input_dtype", ["float32", "half"])
def test_nf4_autocast_master_gradients_and_fp32_input_fallback(
    amp_torch, path, precision, input_dtype,
):
    torch = amp_torch
    bnb = pytest.importorskip("bitsandbytes")
    dtype = getattr(torch, precision)
    compute_dtype = torch.float32 if input_dtype == "float32" else dtype
    reference = make_model(torch, path)
    for projection in [module for module in reference.modules() if hasattr(module, "lora_A")]:
        dense = projection.get_base_layer()
        base = bnb.nn.Linear4bit(
            dense.in_features, dense.out_features, bias=dense.bias is not None,
            compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4",
        )
        try:
            base.weight = bnb.nn.Params4bit(
                dense.weight.detach().to(dtype), requires_grad=False,
                compress_statistics=True, quant_type="nf4",
            ).to("cpu")
        except (RuntimeError, NotImplementedError, ImportError) as error:
            pytest.skip(f"CPU 4-bit quantisation unavailable: {error}")
        if dense.bias is not None:
            base.bias = torch.nn.Parameter(dense.bias.detach().to(dtype), requires_grad=False)
        projection.base_layer = base
    fast = copy.deepcopy(reference)
    patch(fast, path)
    value = torch.randn(2, 7, 32, dtype=torch.float32 if input_dtype == "float32" else dtype)
    results, upstream = [], None
    for model in (reference, fast):
        x = value.clone().requires_grad_()
        with torch.autocast("cpu", dtype=dtype):
            output = model(x)
        output = output if isinstance(output, tuple) else (output,)
        if upstream is None:
            # A scaled backward exercises the same cast boundaries as GradScaler.
            upstream = [torch.randn_like(item) * 128 for item in output]
        torch.autograd.backward(output, upstream)
        results.append((output, {"x": x.grad, **{
            name: parameter.grad for name, parameter in model.named_parameters() if "lora_" in name
        }}))
    nodes = [item.grad_fn for item in results[1][0]]
    seen, custom = set(), []
    while nodes:
        node = nodes.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__.startswith("_FastLora"):
            custom.append(node)
        nodes.extend(child for child, _index in node.next_functions)
    if input_dtype == "float32":
        assert not custom
    else:
        assert custom and all(node.amp_dtype == dtype for node in custom)
    # The fp32-input rows delegate to PEFT, so they are bitwise equal. For the
    # half-input rows the base forward here is dequantise + F.linear, while PEFT
    # runs bitsandbytes' MatMul4Bit on a transposed operand; the two are not
    # guaranteed to round identically, and one forward ulp cascades into every
    # gradient. Those rows compare within 16 ulp of the tensor's largest value.
    exact = input_dtype == "float32"
    bound = 0.0 if exact else 16 * torch.finfo(dtype).eps

    def assert_within(left, right):
        assert torch.isfinite(right).all()
        scale = left.abs().max().clamp_min(torch.finfo(left.dtype).tiny)
        assert (left.float() - right.float()).abs().max() <= bound * scale.float()

    for left, right in zip(results[0][0], results[1][0], strict=True):
        assert_within(left, right)
    for name, gradient in results[0][1].items():
        assert_within(gradient, results[1][1][name])


@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("input_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("input_kind", ["leaf", "nonleaf", "view"])
def test_dense_bf16_autocast_forward_and_all_gradients(
    amp_torch, path, input_dtype, cache, input_kind
):
    torch = amp_torch
    reference = make_model(torch, path)
    fast = copy.deepcopy(reference)
    patch(fast, path)
    value = torch.randn(2, 7, 32, dtype=getattr(torch, input_dtype))
    outputs, inputs, gradients = [], [], []
    upstream = None
    for model in (reference, fast):
        x = value.clone().requires_grad_()
        if input_kind == "nonleaf":
            x = x * 1.0
        elif input_kind == "view":
            x = x.transpose(0, 1)
        x.retain_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
            output = model(x)
        output = output if isinstance(output, tuple) else (output,)
        if upstream is None:
            upstream = [torch.randn_like(item) for item in output]
        torch.autograd.backward(output, upstream)
        outputs.append(output)
        inputs.append(x.grad)
        gradients.append({n: p.grad for n, p in model.named_parameters() if "lora_" in n})
    for item in outputs[1]:
        assert type(item.grad_fn).__name__.startswith("_FastLora")
        assert item.grad_fn.amp_dtype == torch.bfloat16
    for left, right in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)


def test_bf16_llama_accumulated_optimizer_trajectory(amp_torch):
    torch = amp_torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    reference = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                use_cache=False,
            )
        ).bfloat16(),
        LoraConfig(
            r=3, lora_alpha=7, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            ],
            init_lora_weights=False,
        ),
    )
    for name, parameter in reference.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    fast = copy.deepcopy(reference)
    assert patch_fast_lora_mlp(fast) == patch_fast_lora_qkv(fast) == 2
    assert patch_fast_lora_single_projection(fast) > 0
    batches = []
    for _ in range(2):
        ids = torch.randint(0, 128, (2, 12))
        labels = ids.clone()
        labels[:, :7] = -100
        batches.append({"input_ids": ids, "labels": labels})
    denominator = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in batches)
    finals, trajectories = [], []
    for model in (reference, fast):
        parameters = [p for n, p in model.named_parameters() if "lora_" in n]
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0, foreach=False)
        trajectory = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0
            for batch in batches:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model(**batch, num_items_in_batch=denominator).loss
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
            optimizer.step()
            trajectory.append(loss_sum)
        trajectories.append(trajectory)
        finals.append({n: p.detach() for n, p in model.named_parameters() if "lora_" in n})
    assert [f"{loss:.3f}" for loss in trajectories[0]] == [
        f"{loss:.3f}" for loss in trajectories[1]
    ]
    for name in finals[0]:
        torch.testing.assert_close(finals[0][name], finals[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["single", "qkv", "mlp"])
def test_float64_gemms_are_not_narrowed_by_enabled_autocast(amp_torch, path):
    torch = amp_torch
    reference = make_model(torch, path).double()
    fast = copy.deepcopy(reference)
    patch(fast, path)
    value = torch.randn(2, 7, 32, dtype=torch.float64)
    gradients, upstream = [], None
    for model in (reference, fast):
        x = value.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x)
        output = output if isinstance(output, tuple) else (output,)
        if upstream is None:
            upstream = [torch.randn_like(item) for item in output]
        torch.autograd.backward(output, upstream)
        gradients.append(
            {"x": x.grad, **{n: p.grad for n, p in model.named_parameters() if "lora_" in n}}
        )
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-12, atol=1e-12)
