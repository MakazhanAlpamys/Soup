"""Exercise real Qwen3/PEFT saved operands, masks and asynchronous pool reuse."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness.streaming_residency_controls import (
    activation_retention,
    retention_bench,
    retention_replay,
)


def qwen_fixture(device, dtype="float32"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("requires Qwen3 architecture support")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from soup_cli.utils.layer_stream_runtime import (
        LayerBufferPool,
        StreamedDecoderLayer,
        StreamPrefetcher,
        _layer_name_map,
    )

    torch.manual_seed(792)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=6, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8,
                         attention_dropout=0, pad_token_id=0, use_cache=False)
    config._attn_implementation = "eager"
    model = get_peft_model(Qwen3ForCausalLM(config).to(device=device, dtype=getattr(torch, dtype)),
                          LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, value in sorted(model.named_parameters()):
            if value.requires_grad:
                value.copy_(torch.randn(value.shape, generator=generator)
                            * (.003 if "lora_B" in name else .02))
    layers = model.get_base_model().model.layers
    maps = [{name: name.replace(".base_layer.", ".")
             for name, value in layer.named_parameters() if not value.requires_grad}
            for layer in layers]
    weights = []
    for layer, mapping in zip(layers, maps, strict=True):
        held = {key: layer.get_parameter(name).detach().cpu().clone()
                for name, key in mapping.items()}
        if device == "cuda":
            held = {key: value.pin_memory() for key, value in held.items()}
        weights.append(held)
        for name in mapping:
            prefix, _, leaf = name.rpartition(".")
            owner = layer.get_submodule(prefix) if prefix else layer
            value = getattr(owner, leaf)
            setattr(owner, leaf, torch.nn.Parameter(torch.empty_like(value, device="meta"),
                                                    requires_grad=False))
        assert mapping and _layer_name_map(layer) == mapping

    class Source:
        nbytes = sum(value.numel() * value.element_size()
                     for held in weights for value in held.values())

        def get(self, index, name):
            return weights[index][name]

    source = Source()
    specs = {name: (tuple(value.shape), str(value.dtype).removeprefix("torch."))
             for name, value in weights[0].items()}
    pool = LayerBufferPool(specs, device=device)
    prefetcher = StreamPrefetcher(pool, source, len(layers),
                                 stream=torch.cuda.Stream() if device == "cuda" else None)
    for index in range(len(layers)):
        layers[index] = StreamedDecoderLayer(layers[index], index, pool, prefetcher, maps[index])
    runtime = SimpleNamespace(pool=pool, source=source, prefetcher=prefetcher,
                              n_layers=len(layers), tier="ram")
    return model, runtime


def run_qwen(model, runtime, device, dtype="float32"):
    import torch

    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                 lr=.00002)
    tokens = (torch.arange(24, device=device).reshape(2, 12) % 50) + 3
    mask = torch.ones_like(tokens)
    mask[1, :4] = 0
    tokens[1, :4] = 0
    labels = tokens.masked_fill(mask == 0, -100)
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    values = []
    for _update in range(3):
        for _micro in range(2):
            runtime.prefetcher.prime()
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=dtype == "bfloat16"):
                output = model(input_ids=tokens, attention_mask=mask,
                               position_ids=positions, labels=labels)
            values.extend([output.logits.detach().clone(), output.loss.detach().clone()])
            (output.loss / 2).backward()
        values.extend(value.grad.detach().clone() for value in model.parameters()
                      if value.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    values.extend(value.detach().clone() for value in model.parameters() if value.requires_grad)
    if device == "cuda":
        torch.cuda.synchronize()
    return values


@pytest.mark.parametrize("arm", ["baseline", "resident", "retained", "both"])
def test_real_qwen_peft_cpu_four_arms(arm):
    check_qwen(arm, "cpu")


def check_qwen(arm, device, dtype="float32"):
    torch = pytest.importorskip("torch")

    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        reference, native = qwen_fixture(device, dtype)
        model, runtime = qwen_fixture(device, dtype)
        if arm in {"resident", "both"}:
            install_persistent_layers(runtime, [2, 3])
        state = None
        if arm in {"retained", "both"}:
            state = activation_retention.install_retention(model, [2, 3], resident=arm == "both")
        first = run_qwen(reference, native, device, dtype)
        second = run_qwen(model, runtime, device, dtype)
        assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
        if state:
            assert state["retained_forward_calls"] == 12
            assert bool(state["weight_snapshot_tensors"]) == (arm == "retained")
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ["baseline", "resident", "retained", "both"])
def test_real_qwen_peft_cuda_four_arms(arm):
    check_qwen(arm, "cuda")


@pytest.mark.parametrize("arm", ["baseline", "resident", "retained", "both"])
def test_real_qwen_peft_cpu_bf16_four_arms(arm):
    check_qwen(arm, "cpu", "bfloat16")


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ["baseline", "resident", "retained", "both"])
def test_real_qwen_peft_cuda_bf16_four_arms(arm):
    check_qwen(arm, "cuda", "bfloat16")


def test_real_qwen_mutable_alias_mutation_detected(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(activation_retention, "_snapshot", lambda value: value.detach())
    reference, native = qwen_fixture("cpu")
    model, runtime = qwen_fixture("cpu")
    activation_retention.install_retention(model, [2, 3], resident=False)
    first = run_qwen(reference, native, "cpu")
    second = run_qwen(model, runtime, "cpu")
    assert any(not torch.equal(left, right) for left, right in zip(first, second, strict=True))


def test_frozen_upstream_missing_report_metadata_is_supported():
    wrapper = SimpleNamespace(config=SimpleNamespace(
        training=SimpleNamespace(use_liger=False, fast_lora=False)))
    provenance = retention_bench.baseline_patch_metadata(wrapper)
    assert wrapper._streaming_liger_patches == {}
    assert provenance == {"metadata_only": True, "production_arithmetic_changed": False}
    wrapper._liger_applied = True
    with pytest.raises(ValueError, match="unpatched"):
        retention_bench.baseline_patch_metadata(wrapper)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_bounded_replay_matches_original_full_comparator(tmp_path, dtype):
    import ast
    import zipfile

    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    repo = Path(__file__).parents[1]
    archive = repo / "tests/fixtures/streaming_residency/planned-harness.zip"
    with zipfile.ZipFile(archive) as source:
        tree = ast.parse(source.read("overlay/benchmarks/harness/liger_streaming_parity.py"))
    compare = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "compare")
    namespace = {"Path": Path}
    code = compile(ast.Module(body=[compare], type_ignores=[]), "original-compare", "exec")
    exec(code, namespace)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    tensors = {"logits/0": torch.linspace(-1, 1, 128 * 17000).reshape(1, 128, 17000)
               .to(getattr(torch, dtype)), "batch/0/input_ids": torch.arange(12),
               "initial/adapter": torch.arange(64).float(),
               "gradient/0/adapter": torch.linspace(.1, 1, 64),
               "loss": torch.tensor([2., 1.9999], dtype=torch.float64)}
    changed = {name: value.clone() for name, value in tensors.items()}
    changed["gradient/0/adapter"][7] += .000001
    for corrupt in (False, True):
        if corrupt:
            changed["logits/0"][0, 99, 200] = float("inf")
        save_file(tensors, str(first / "trace.safetensors"))
        save_file(changed, str(second / "trace.safetensors"))
        original = namespace["compare"](first, second)
        replay = retention_replay.compare_traces(
            first / "trace.safetensors", second / "trace.safetensors"
        )
        assert replay["passed"] == original["passed"]
        assert replay["loss_three_decimals"] == original["loss_three_decimals"]
        for group, stats in original["groups"].items():
            assert all(stats[key] == replay["groups"][group][key]
                       for key in ("tensors", "failed_tensors", "max_abs", "failures"))
