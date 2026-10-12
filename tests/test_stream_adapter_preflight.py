"""Early adapter budgets use real PEFT shapes and stop before checkpoint I/O."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from soup_cli.utils.stream_adapter_preflight import (
    build_stream_adapter_plan,
    check_stream_adapter_budget,
)


def _settings(**overrides):
    values = dict(
        optimizer="adamw_torch", moe_lora=False, use_lorafa=False,
        lora=SimpleNamespace(
            r=8, alpha=8, dropout=0, target_modules="auto", use_dora=False,
            use_rslora=False, rank_pattern=None, alpha_pattern=None, init_strategy="random",
        ),
        quantization="none", double_quant_on=True, stream_vram_override=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def tiny_config(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path), config


def _plan(tiny_config, settings=None):
    base, config = tiny_config
    return build_stream_adapter_plan(
        base, config, settings or _settings(), dtype="bfloat16", quant="none",
        double_quant=True, trust_remote_code=False, on_cuda=True,
    )


def test_real_shapes_and_materialization(tiny_config):
    from copy import deepcopy

    from peft import get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, materialize_meta_adapters
    from soup_cli.utils.peft_wiring import apply_pre_lora_patches

    plan = _plan(tiny_config)
    assert plan.budget.stored_parameters == 3584
    assert plan.budget.total.cuda_bytes == 3584 * 16
    assert plan.budget.total.cpu_bytes == 8 * 4
    base, _ = tiny_config
    actual = build_meta_skeleton(base, dtype="bfloat16")
    for param in actual.parameters():
        param.requires_grad = False
    apply_pre_lora_patches(actual, base)
    actual = get_peft_model(actual, deepcopy(plan.lora_config))
    materialize_meta_adapters(actual, device="cpu")
    real = {name: (tuple(param.shape), str(param.dtype))
            for name, param in actual.named_parameters() if param.requires_grad}
    assert real == {tensor.name: (tensor.shape, "torch.float32") for tensor in plan.tensors}


def test_known_budget_boundary_and_unknown_warning(tiny_config):
    output = StringIO()
    console = Console(file=output)
    plan = _plan(tiny_config)
    check_stream_adapter_budget(plan, available_cuda_bytes=57344, console=console)
    with pytest.raises(ValueError, match="before checkpoint loading"):
        check_stream_adapter_budget(plan, available_cuda_bytes=57343, console=console)
    unknown = _plan(tiny_config, _settings(optimizer="muon"))
    assert unknown.budget.total is None
    check_stream_adapter_budget(unknown, available_cuda_bytes=10**9, console=console)
    assert "No adapter-fit verdict" in output.getvalue()
    with pytest.raises(ValueError, match="weights alone"):
        check_stream_adapter_budget(unknown, available_cuda_bytes=1, console=console)


def test_lorafa_and_override_profiles(tiny_config):
    plan = _plan(tiny_config, _settings(use_lorafa=True))
    assert plan.budget.total is not None
    assert all(not t.trainable for t in plan.tensors if ".lora_A." in t.name)
    assert plan.budget.weights.cuda_bytes == 3584 * 4
    unknown = _plan(tiny_config, _settings(use_galore=True))
    assert unknown.budget.total is None


@pytest.mark.parametrize("free_bytes,rejected", [(1, True), (10**9, False)])
def test_setup_stops_before_weight_loading(tiny_config, monkeypatch, free_bytes, rejected):
    import torch
    from transformers import AutoTokenizer

    from soup_cli.trainer import stream_setup
    from soup_cli.utils import layer_shard, layer_stream, spectrum_scan, stripe_roots

    base, _ = tiny_config
    monkeypatch.setattr(stream_setup, "refuse_if_data_parallel", lambda device: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=SimpleNamespace(
        pad_token="pad", eos_token="eos"
    )))
    monkeypatch.setattr(layer_stream, "resolve_stream_dtype", lambda device: "float32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, 10**9))
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda base: "unused-cache")
    monkeypatch.setattr(stripe_roots, "resolve_stripe_roots", lambda *args, **kwargs: ())
    weights = Mock(side_effect=RuntimeError("REACHED_WEIGHT_LOADING"))
    sharder = Mock(side_effect=AssertionError("must not shard"))
    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", weights)
    monkeypatch.setattr(layer_shard, "shard_checkpoint", sharder)
    wrapper = stream_setup.StreamingSetupMixin()
    wrapper.device = "cuda:0"
    wrapper._trust_remote_code = False
    expected = "before checkpoint loading" if rejected else "REACHED_WEIGHT_LOADING"
    with pytest.raises(ValueError if rejected else RuntimeError, match=expected):
        wrapper._setup_streaming_transformers(SimpleNamespace(base=base, task="sft"), _settings())
    assert weights.call_count == (0 if rejected else 1)
    sharder.assert_not_called()


def test_peak_replaces_legacy_adapter_term():
    from soup_cli.utils.layer_stream import estimate_stream_peak_vram

    settings = dict(layer_bytes=100, buffers=2, extras_bytes=50, adapter_params=1000,
                    vocab_size=128, hidden_size=64, intermediate_size=128, n_layers=2,
                    seq_len=16)
    old = estimate_stream_peak_vram(**settings)
    new = estimate_stream_peak_vram(**settings, adapter_training_bytes=1234)
    assert new == old - 16000 + 1234
    with pytest.raises(ValueError, match="non-negative"):
        estimate_stream_peak_vram(**settings, adapter_training_bytes=-1)


def test_fused_moe_plan_keeps_real_build_config_unmodified(tmp_path):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, moe_intermediate_size=128,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    plan = build_stream_adapter_plan(
        str(tmp_path), config, _settings(moe_lora=True), dtype="bfloat16", quant="none",
        double_quant=True, trust_remote_code=False, on_cuda=True,
    )
    assert not plan.lora_config.target_parameters
    assert not plan.lora_config.rank_pattern
    actual = get_peft_model(AutoModelForCausalLM.from_config(config), plan.lora_config)
    real = {name: tuple(param.shape) for name, param in actual.named_parameters()
            if param.requires_grad}
    assert real == {tensor.name: tensor.shape for tensor in plan.tensors}
    assert any(".experts." in name for name in real)
    assert plan.budget.stored_parameters == 60416
    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    settings = _settings(moe_lora=True)
    message = recommend_stream_adapter_targets(
        plan, str(tmp_path), config, settings, available_cuda_bytes=200000,
        dtype="bfloat16", quant="none", double_quant=True,
        trust_remote_code=False, on_cuda=True,
    )
    assert "attention only:" in message
    assert "attention + shared expert:" not in message
    assert "training.moe_lora: false" in message
    assert settings.moe_lora is True


def test_bnb_profile_version_and_override_selection(monkeypatch):
    from soup_cli.utils import stream_adapter_preflight as preflight

    monkeypatch.setattr(preflight, "version", lambda package: "0.50.2")
    assert preflight._optimizer_profile(_settings(optimizer="adamw_8bit")) == (
        "bnb_0.50.2_adamw8bit_fp32"
    )
    monkeypatch.setattr(preflight, "version", lambda package: "0.50.3")
    assert preflight._optimizer_profile(_settings(optimizer="adamw_8bit")).startswith(
        "unsupported:"
    )
    assert preflight._optimizer_profile(_settings(loraplus_lr_ratio=16)).startswith("unsupported:")


def test_nf4_meta_plan_keeps_adapter_shapes(tiny_config):
    pytest.importorskip("bitsandbytes")
    base, config = tiny_config
    plain = _plan(tiny_config)
    nf4 = build_stream_adapter_plan(
        base, config, _settings(), dtype="float32", quant="nf4", double_quant=False,
        trust_remote_code=False, on_cuda=True,
    )
    assert nf4.tensors == plain.tensors
    assert nf4.budget == plain.budget
    assert nf4.quant_suffixes


def test_attention_recommendation_rebuilds_and_preserves_settings(tiny_config):
    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    settings = _settings()
    settings.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]
    settings.lora.rank_pattern = {"q_proj": 4}
    original = _plan(tiny_config, settings)
    base, config = tiny_config
    kwargs = dict(dtype="float32", quant="none", double_quant=True,
                  trust_remote_code=False, on_cuda=True)
    text = recommend_stream_adapter_targets(
        original, base, config, settings, available_cuda_bytes=120000, **kwargs,
    )
    assert "attention only:" in text
    assert "attention + shared expert:" not in text
    assert "training.moe_lora: false" in text
    assert "not a full-training fit" in text
    assert settings.lora.rank_pattern == {"q_proj": 4}
    assert "gate_proj" in settings.lora.target_modules
    # The emitted value is valid YAML/JSON and works through actual PEFT.
    import json
    from copy import deepcopy

    proposed = deepcopy(settings)
    value = text.split("training.lora.target_modules: ", 1)[1].split("; keep", 1)[0]
    proposed.lora.target_modules = json.loads(value)
    rerun = _plan(tiny_config, proposed)
    assert rerun.budget.total.cuda_bytes <= 120000
    assert all(".self_attn." in t.name for t in rerun.tensors)
    assert any(t.shape[0] == 4 for t in rerun.tensors if ".q_proj.lora_A." in t.name)
    assert "No smaller verified" in recommend_stream_adapter_targets(
        original, base, config, settings, available_cuda_bytes=1, **kwargs,
    )


def test_shared_expert_recommendation_excludes_other_mlp(tiny_config, monkeypatch):
    import torch
    from transformers import LlamaForCausalLM

    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    base, config = tiny_config

    def skeleton(*args, **kwargs):
        model = LlamaForCausalLM(config)
        for layer in model.model.layers:
            shared = torch.nn.Module()
            shared.gate_proj = torch.nn.Linear(64, 128, bias=False)
            shared.up_proj = torch.nn.Linear(64, 128, bias=False)
            shared.down_proj = torch.nn.Linear(128, 64, bias=False)
            layer.mlp.shared_expert = shared
        return model.to("meta")

    monkeypatch.setattr("soup_cli.utils.layer_stream_runtime.build_meta_skeleton", skeleton)
    settings = _settings()
    settings.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]
    original = _plan(tiny_config, settings)
    text = recommend_stream_adapter_targets(
        original, base, config, settings, available_cuda_bytes=300000,
        dtype="float32", quant="none", double_quant=True,
        trust_remote_code=False, on_cuda=True,
    )
    assert "attention only:" in text
    assert "attention + shared expert:" in text
    import json

    value = text.rsplit("training.lora.target_modules: ", 1)[1].split("; keep", 1)[0]
    settings.lora.target_modules = json.loads(value)
    rerun = _plan(tiny_config, settings)
    assert all(".self_attn." in t.name or ".shared_expert." in t.name for t in rerun.tensors)
    assert rerun.budget.total.cuda_bytes < original.budget.total.cuda_bytes


def test_recommendations_lazy_and_unknown_has_no_fit(tiny_config):
    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    plan = _plan(tiny_config)
    callback = Mock(return_value=" VERIFIED_ALTERNATIVE")
    check_stream_adapter_budget(plan, available_cuda_bytes=10**9,
                                console=Console(file=StringIO()), recommendations=callback)
    callback.assert_not_called()
    with pytest.raises(ValueError, match="VERIFIED_ALTERNATIVE"):
        check_stream_adapter_budget(plan, available_cuda_bytes=1,
                                    console=Console(file=StringIO()), recommendations=callback)
    callback.assert_called_once()
    unknown = _plan(tiny_config, _settings(optimizer="muon"))
    assert "unknown" in recommend_stream_adapter_targets(
        unknown, *tiny_config, _settings(), available_cuda_bytes=10**9,
    )


@pytest.mark.parametrize("optimizer_name", ["adamw_torch", "sgd", "adagrad", "rmsprop",
                                           "adafactor", "lorafa"])
def test_trainer_created_optimizer_matches_profile(tiny_config, tmp_path, optimizer_name):
    """Exercise Trainer's factory/grouping rather than construct optimizers by hand."""
    import torch
    from peft import get_peft_model
    from transformers import LlamaForCausalLM, Trainer, TrainingArguments

    from soup_cli.utils.adapter_budget import estimate_adapter_budget

    use_lorafa = optimizer_name == "lorafa"
    settings = _settings(optimizer="adamw_torch" if use_lorafa else optimizer_name,
                         use_lorafa=use_lorafa)
    plan = _plan(tiny_config, settings)
    model = get_peft_model(LlamaForCausalLM(tiny_config[1]), plan.lora_config)
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path / "trainer"), optim=settings.optimizer,
        use_cpu=True, report_to="none",
    ))
    if use_lorafa:
        from soup_cli.utils.peft_wiring import attach_lorafa_optimizer

        assert attach_lorafa_optimizer(trainer, settings)
    optimizer = trainer.create_optimizer()
    predicted = estimate_adapter_budget(plan.tensors, profile=plan.budget.profile, device="cpu")
    params = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in params} == {
        id(p) for group in optimizer.param_groups for p in group["params"] if p.requires_grad}
    for _ in range(2):
        for p in params:
            p.grad = torch.ones_like(p)
        optimizer.step()
        seen = set()
        actual = 0
        for state in optimizer.state.values():
            for tensor in state.values():
                if not isinstance(tensor, torch.Tensor):
                    continue
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    actual += storage.nbytes()
                    seen.add(key)
        assert actual == predicted.optimizer.cpu_bytes


@pytest.mark.parametrize("optimizer_name", ["adamw_8bit", "adamw_bnb_8bit"])
def test_trainer_bnb_factory_uses_profiled_options(tmp_path, optimizer_name):
    pytest.importorskip("bitsandbytes")
    import torch
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
                             report_to="none", optim=optimizer_name)
    optimizer_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    assert optimizer_cls.__module__.startswith("bitsandbytes.")
    assert kwargs["optim_bits"] == 8
    assert kwargs["is_paged"] is False
    optimizer = optimizer_cls([torch.nn.Parameter(torch.zeros(4096))], **kwargs)
    assert optimizer.args.min_8bit_size == 4096
    assert optimizer.is_paged is False


def test_refusal_apply_recommendation_then_real_cpu_setup(tmp_path, monkeypatch):
    """Real CUDA gate (capacity mocked), then emitted YAML through real CPU runtime."""
    import json

    import torch
    import yaml
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer import stream_setup
    from soup_cli.utils import spectrum_scan

    base = tmp_path / "model"
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    LlamaForCausalLM(config).save_pretrained(base)
    raw = dict(base=str(base), task="sft", backend="transformers", modality="text",
               data=dict(train="unused.jsonl", max_length=64), training=dict(
                   quantization="none", stream_layers=True, batch_size=1,
                   gradient_accumulation_steps=1, stream_pin=False,
                   optimizer="adamw_torch", lora=dict(r=8, alpha=8, dropout=0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])))
    cfg = load_config_from_string(yaml.safe_dump(raw))
    wrapper = stream_setup.StreamingSetupMixin()
    wrapper.device, wrapper._trust_remote_code = "cuda:0", False
    monkeypatch.setattr(stream_setup, "refuse_if_data_parallel", lambda device: None)
    monkeypatch.setattr("soup_cli.utils.layer_stream.resolve_stream_dtype",
                        lambda device: "float32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (120000, 120000))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", Mock(return_value=
                        SimpleNamespace(pad_token="pad", eos_token="eos")))
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    from unittest.mock import patch

    with patch.object(spectrum_scan, "resolve_model_weights") as loader:
        with pytest.raises(ValueError, match="attention only:") as error:
            wrapper._setup_streaming_transformers(cfg, cfg.training)
        loader.assert_not_called()
    message = str(error.value)
    targets = json.loads(message.split("training.lora.target_modules: ", 1)[1].split("; keep")[0])
    raw["training"]["moe_lora"] = False
    raw["training"]["lora"]["target_modules"] = targets
    proposed = load_config_from_string(yaml.safe_dump(raw))
    plan = _plan((str(base), config), proposed.training)
    assert f"total={plan.budget.total.cuda_bytes} CUDA bytes" in message
    check_stream_adapter_budget(plan, available_cuda_bytes=120000,
                                console=Console(file=StringIO()))
    wrapper.device = "cpu"
    try:
        wrapper._setup_streaming_transformers(proposed, proposed.training)
        actual = {n: tuple(p.shape) for n, p in wrapper.model.named_parameters() if p.requires_grad}
        assert actual == {t.name: t.shape for t in plan.tensors if t.trainable}
        output = wrapper.model(input_ids=torch.randint(0, 128, (1, 8)))
        assert torch.isfinite(output.logits).all()
    finally:
        wrapper._close_stream_runtime()
