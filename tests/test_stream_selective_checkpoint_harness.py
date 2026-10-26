"""Benchmark-only selective checkpointing must preserve streamed gradients."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _harness():
    path = Path(__file__).parents[1] / "benchmarks/harness/stream_selective_checkpoint.py"
    spec = importlib.util.spec_from_file_location("stream_selective_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy,width", [("qkv", 6144), ("qkvo", 10240)])
def test_mistral_activation_bytes_are_outputs_only(policy, width):
    helper = _harness()
    for seq in (512, 2048):
        assert helper.saved_projection_bytes(policy, 32, 4096, 1024, seq) == (32 * width * seq * 2)


def test_projection_byte_predictor_refuses_unknown_policy():
    helper = _harness()
    with pytest.raises(ValueError, match="unknown policy"):
        helper.saved_projection_bytes("all", 32, 4096, 1024, 512)


def test_projection_byte_predictor_refuses_zero_sequence_length():
    helper = _harness()
    with pytest.raises(ValueError, match="positive"):
        helper.saved_projection_bytes("qkv", 32, 4096, 1024, 0)


def _fixture():
    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    nn = torch.nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(8, 8, bias=False)
            self.base_layer.weight.requires_grad_(False)
            self.lora_A = nn.Linear(8, 2, bias=False)
            self.lora_B = nn.Linear(2, 8, bias=False)

        def forward(self, value):
            return self.base_layer(value) + self.lora_B(self.lora_A(value)) * 2

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, Adapter())

        def forward(self, value):
            query, key, val = self.q_proj(value), self.k_proj(value), self.v_proj(value)
            scores = (query @ key.transpose(-1, -2) / 8**0.5).softmax(-1)
            return self.o_proj(scores @ val)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = nn.Linear(8, 8, bias=False)
            self.mlp.weight.requires_grad_(False)

        def forward(self, value):
            value = value + self.self_attn(value)
            return value + self.mlp(value).tanh()

    class Pool:
        def __init__(self, states):
            self.states = states
            self.buffers = {name: value.clone() for name, value in states[0].items()}

        def wait(self, idx):
            with torch.no_grad():
                for name, value in self.states[idx].items():
                    self.buffers[name].copy_(value)
            return self.buffers

    class Prefetcher:
        def advance(self, idx):
            pass

    torch.manual_seed(713)
    inners = [Decoder().double() for _ in range(3)]
    states = [
        {
            name: value.detach().clone()
            for name, value in layer.named_parameters()
            if not value.requires_grad
        }
        for layer in inners
    ]
    pool = Pool(states)
    return nn.Sequential(
        *[
            StreamedDecoderLayer(
                layer, idx, pool, Prefetcher(), name_map={name: name for name in states[idx]}
            )
            for idx, layer in enumerate(inners)
        ]
    )


@pytest.mark.parametrize("policy,count", [("qkv", 9), ("qkvo", 12)])
def test_three_layers_recycle_one_slot_with_exact_gradients_and_updates(policy, count):
    helper = _harness()
    expected_model = _fixture()
    actual_model = _fixture()
    controller = helper.install(actual_model, policy)
    assert controller.layer_count == 3
    values = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(51)).double()
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.001) for m in (expected_model, actual_model)]
    for step in range(2):
        rows = []
        for model, optimizer in zip((expected_model, actual_model), optimizers):
            optimizer.zero_grad(set_to_none=True)
            inputs = values.clone().requires_grad_(True)
            output = model(inputs)
            if model is actual_model:
                assert controller.cache_bytes() == count * 5 * 8 * 8
                assert len(controller.cached_tensors()) == count
            loss = output.square().mean()
            loss.backward()
            rows.append(
                (
                    output.detach().clone(),
                    inputs.grad.clone(),
                    {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad},
                )
            )
            optimizer.step()
        assert torch.equal(rows[0][0], rows[1][0]), step
        assert torch.equal(rows[0][1], rows[1][1]), step
        for name in rows[0][2]:
            assert torch.equal(rows[0][2][name], rows[1][2][name]), (step, name)
        for (name, expected), (_, actual) in zip(
            expected_model.named_parameters(), actual_model.named_parameters()
        ):
            assert torch.equal(expected, actual), (step, name)
    assert controller.counts["saved_forward"] == count * 2
    assert controller.counts["recompute_projection_calls"] > 0
    controller.remove()


def test_mutated_cache_is_refused():
    helper = _harness()
    model = _fixture()
    controller = helper.install(model, "qkv")
    output = model(torch.ones(1, 5, 8, dtype=torch.float64, requires_grad=True))
    with torch.no_grad():
        controller.cached_tensors()[0].add_(1)
    with pytest.raises(RuntimeError, match="mutat"):
        output.square().sum().backward()
    controller.remove()


def test_remove_restores_instance_forwards_and_invalid_scope_is_refused():
    helper = _harness()
    model = _fixture()
    forwards = {name: module.forward for name, module in model.named_modules()}
    controller = helper.install(model, "qkvo")
    controller.remove()
    assert all(module.forward == forwards[name] for name, module in model.named_modules())
    with pytest.raises(ValueError, match="policy"):
        helper.install(model, "all")
    with pytest.raises(ValueError, match="streamed"):
        helper.install(torch.nn.Linear(8, 8), "qkv")


@pytest.mark.parametrize("policy", ["qkv", "qkvo"])
def test_retained_layer_is_refused_before_any_forward_is_patched(policy):
    helper = _harness()
    model = _fixture()
    for layer in model:
        layer.recompute = True
    model[-1].recompute = False
    modules = []
    model.apply(modules.append)
    originals = [
        (module, module.forward, "forward" in module.__dict__, module.__dict__.get("forward"))
        for module in modules
    ]

    with pytest.raises(ValueError, match="recompute"):
        helper.install(model, policy)

    for module, forward, had_instance_forward, instance_forward in originals:
        assert module.forward == forward
        assert ("forward" in module.__dict__) == had_instance_forward
        assert module.__dict__.get("forward") is instance_forward

    model[-1].recompute = True
    controller = helper.install(model, policy)
    assert controller.layer_count == 3
    controller.remove()


def _mistral_weights(tmp_path):
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(73)
    config = MistralConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=64,
        attention_dropout=0.0,
    )
    resident = MistralForCausalLM(config)
    path = str(tmp_path / "weights")
    resident.save_pretrained(path)
    return path


def _real_mistral_comparison(tmp_path, policy, device, quant):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    helper = _harness()
    weights = _mistral_weights(tmp_path)
    shards = str(tmp_path / "shards")
    dtype = "bfloat16" if device == "cuda" else "float32"
    suffixes = ()
    if quant == "nf4":
        probe = build_meta_skeleton(weights, dtype=dtype, quant=quant)
        suffixes = quantised_layer_suffixes(probe)
        del probe
    index = shard_checkpoint(
        weights,
        shards,
        dtype=dtype,
        arch="mistral",
        quant=quant,
        quant_suffixes=suffixes,
        quant_device=device,
        double_quant=True,
    )
    models, runtimes = [], []
    controller = None
    try:
        # Repeat full/full first: a nondeterministic reference cannot certify SAC.
        for mode in ("full", "full", policy):
            model, runtime = build_streamed_model(
                model_id=weights,
                shard_dir=shards,
                index=index,
                lora_config=LoraConfig(
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                    task_type=TaskType.CAUSAL_LM,
                ),
                device=device,
                dtype=dtype,
                buffers=2,
                pin=device == "cuda",
                require_pin=device == "cuda",
                seed=3,
                quant=quant,
                weights_dir=weights,
            )
            model.config._attn_implementation = "eager"
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        # Zero B can hide errors in input gradients on step one.
                        param.fill_(0.01)
            model.train()
            models.append(model)
            runtimes.append(runtime)
            if mode != "full":
                controller = helper.install(model, mode)
        ids = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(17))
        ids = ids.to(device)
        optimizers = [
            torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=2e-4,
            )
            for model in models
        ]
        for step in range(2):
            rows = []
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                out = model(input_ids=ids, labels=ids)
                if model is models[-1]:
                    expected = helper.saved_projection_bytes(
                        policy,
                        4,
                        64,
                        32,
                        16,
                        element_bytes=2 if device == "cuda" else 4,
                    )
                    assert controller.cache_bytes() == expected
                    assert len(controller.cached_tensors()) == 4 * len(helper.POLICIES[policy])
                out.loss.backward()
                row = {
                    "logits": out.logits.detach().clone(),
                    "loss": out.loss.detach().clone(),
                    **{
                        f"grad:{n}": p.grad.detach().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    },
                }
                optimizer.step()
                row.update(
                    {
                        f"param:{n}": p.detach().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    }
                )
                rows.append(row)
            assert rows[0].keys() == rows[1].keys() == rows[2].keys()
            for name in rows[0]:
                assert torch.equal(rows[0][name], rows[1][name]), ("full/full", step, name)
                assert torch.equal(rows[0][name], rows[2][name]), (policy, step, name)
    finally:
        if controller is not None:
            controller.remove()
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("policy", ["qkv", "qkvo"])
def test_real_mistral_gqa_cpu_exact_two_updates(tmp_path, policy):
    _real_mistral_comparison(tmp_path, policy, "cpu", "none")


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["qkv", "qkvo"])
def test_real_mistral_gqa_nf4_cuda_exact_two_updates(tmp_path, policy):
    # Eager attention removes SDPA flash backward nondeterminism from this gate.
    _real_mistral_comparison(tmp_path, policy, "cuda", "nf4")
