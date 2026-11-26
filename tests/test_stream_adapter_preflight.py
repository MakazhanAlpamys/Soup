"""Check persistent adapter memory before opening a streamed checkpoint."""

import json
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from rich.console import Console


def _settings(**overrides: Any) -> SimpleNamespace:
    from soup_cli.config.schema import LoraConfig

    values = dict(
        optimizer="adamw_torch",
        moe_lora=False,
        use_lorafa=False,
        lora=LoraConfig(r=8, alpha=8, dropout=0),
        quantization="none",
        double_quant_on=True,
        stream_vram_override=None,
        stream_disk_kind=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def tiny_config(tmp_path: Any) -> tuple[str, Any]:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path), config


def _build_kwargs(**overrides: Any) -> dict[str, Any]:
    return dict(
        dtype="bfloat16",
        quant="none",
        double_quant=True,
        trust_remote_code=False,
        on_cuda=True,
    ) | overrides


def _plan(tiny_config: tuple[str, Any], settings: Any = None, **kwargs: Any) -> Any:
    from soup_cli.utils.stream_adapter_preflight import build_stream_adapter_plan

    return build_stream_adapter_plan(
        *tiny_config, settings if settings is not None else _settings(), **_build_kwargs(**kwargs)
    )


def test_known_budget_fits_silently_at_exact_capacity(tiny_config: tuple[str, Any]) -> None:
    from soup_cli.utils.stream_adapter_preflight import check_stream_adapter_budget

    output = StringIO()
    recommendations = Mock(return_value="")
    check_stream_adapter_budget(
        _plan(tiny_config),
        available_cuda_bytes=57344,
        console=Console(file=output),
        recommendations=recommendations,
    )

    assert output.getvalue() == ""
    recommendations.assert_not_called()


def test_known_budget_refuses_one_byte_over_capacity(tiny_config: tuple[str, Any]) -> None:
    from soup_cli.utils.stream_adapter_preflight import check_stream_adapter_budget

    recommendations = Mock(return_value="\nverified alternative")
    with pytest.raises(ValueError) as error:
        check_stream_adapter_budget(
            _plan(tiny_config),
            available_cuda_bytes=57343,
            console=Console(file=StringIO()),
            recommendations=recommendations,
        )

    assert str(error.value) == (
        "Adapter memory budget exceeded before checkpoint loading: "
        "3,584 adapter parameters; weights=14336, gradients=14336, optimizer=28672 CUDA bytes; "
        "CPU tensors=32 bytes; required=57344 bytes (0.000 GB), available=57343 bytes (0.000 GB). "
        "Reduce LoRA targets or rank. The full training step needs additional memory."
        "\nverified alternative"
    )
    recommendations.assert_called_once_with()


@pytest.mark.parametrize("available", [-1, True, 57344.0])
def test_rejects_invalid_cuda_capacity(tiny_config: tuple[str, Any], available: Any) -> None:
    from soup_cli.utils.stream_adapter_preflight import check_stream_adapter_budget

    output = StringIO()
    with pytest.raises(ValueError) as error:
        check_stream_adapter_budget(
            _plan(tiny_config), available_cuda_bytes=available, console=Console(file=output)
        )
    assert str(error.value) == (
        f"available CUDA memory must be a non-negative integer byte count; got {available!r}"
    )
    assert output.getvalue() == ""


def test_unknown_optimizer_warns_and_checks_weights_only(tiny_config: tuple[str, Any]) -> None:
    from soup_cli.utils.stream_adapter_preflight import (
        check_stream_adapter_budget,
        recommend_stream_adapter_targets,
    )

    output = StringIO()
    console = Console(file=output, width=240)
    settings = _settings(optimizer="muon")
    plan = _plan(tiny_config, settings)
    assert plan.budget.total is None
    check_stream_adapter_budget(plan, available_cuda_bytes=14336, console=console)
    assert output.getvalue() == (
        "Adapter optimizer budget unavailable: unverified optimizer profile: "
        "'unsupported: muon'. No adapter-fit verdict is available.\n"
    )
    with pytest.raises(ValueError) as error:
        check_stream_adapter_budget(plan, available_cuda_bytes=14335, console=console)
    assert str(error.value) == (
        "Adapter memory budget exceeded before checkpoint loading: 3,584 adapter parameters; "
        "adapter weights alone=14336 bytes; unverified optimizer profile: 'unsupported: muon'; "
        "required=14336 bytes (0.000 GB), available=14335 bytes (0.000 GB). "
        "Reduce LoRA targets or rank. The full training step needs additional memory."
    )
    assert recommend_stream_adapter_targets(
        plan, *tiny_config, settings, available_cuda_bytes=14335, **_build_kwargs()
    ) == " No verified target alternative: optimizer budget is unknown."


def test_real_shapes_and_materialization(tiny_config: tuple[str, Any]) -> None:
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, materialize_meta_adapters
    from soup_cli.utils.peft_wiring import apply_pre_lora_patches, attach_adapter

    plan = _plan(tiny_config)
    assert plan.budget.stored_parameters == 3584
    assert plan.budget.total.cuda_bytes == 57344
    assert plan.budget.total.cpu_bytes == 32
    base, _ = tiny_config
    actual = build_meta_skeleton(base, dtype="bfloat16")
    for param in actual.parameters():
        param.requires_grad = False
    apply_pre_lora_patches(actual, base)
    actual = attach_adapter(actual, deepcopy(plan.lora_config))
    materialize_meta_adapters(actual, device="cpu")
    real = {
        name: (tuple(param.shape), str(param.dtype))
        for name, param in actual.named_parameters()
        if param.requires_grad
    }
    assert real == {tensor.name: (tensor.shape, "torch.float32") for tensor in plan.tensors}


def test_probe_preserves_training_rng(tiny_config: tuple[str, Any]) -> None:
    import torch

    rng_state = torch.get_rng_state()
    _plan(tiny_config)
    assert torch.equal(torch.get_rng_state(), rng_state)


@pytest.mark.parametrize(
    ("device", "message"),
    [
        ("cpu", "stream adapter preflight unexpectedly materialized parameter 'unexpected'"),
        ("meta", "stream adapter preflight cannot account for 'unexpected'"),
    ],
)
def test_probe_refuses_unaccounted_parameters(
    tiny_config: tuple[str, Any], monkeypatch: pytest.MonkeyPatch, device: str, message: str
) -> None:
    import torch

    from soup_cli.utils import peft_wiring

    real_attach = peft_wiring.attach_adapter

    def attach_extra(model: Any, config: Any) -> Any:
        adapted = real_attach(model, config)
        adapted.unexpected = torch.nn.Parameter(torch.empty(1, device=device))
        return adapted

    monkeypatch.setattr(peft_wiring, "attach_adapter", attach_extra)
    with pytest.raises(ValueError) as error:
        _plan(tiny_config)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("free_bytes", "reserved_bytes", "override_bytes", "optimizer_backend"),
    [
        (1, 0, None, None),
        (10**9, 0, None, None),
        (1, 10**9, None, None),
        (1, 10**9, 1, None),
        (14336, 0, None, "deepspeed_config"),
        (14336, 0, None, "fsdp_config"),
    ],
)
def test_setup_checks_budget_before_weight_resolution(
    tiny_config: tuple[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    free_bytes: int,
    reserved_bytes: int,
    override_bytes: int | None,
    optimizer_backend: str | None,
) -> None:
    import torch
    from transformers import AutoTokenizer

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer import stream_setup
    from soup_cli.utils import layer_shard, layer_stream, spectrum_scan, stripe_roots

    cfg = SoupConfig.model_validate({
        "base": tiny_config[0],
        "training": {
            "stream_layers": True, "quantization": "none", "batch_size": 1,
            "lora": {"r": 8}, "stream_vram_override": override_bytes,
        },
        "data": {"train": "unused.jsonl"},
    })
    monkeypatch.setattr(stream_setup, "refuse_if_data_parallel", lambda device: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=SimpleNamespace(
        pad_token="pad", eos_token="eos",
    )))
    monkeypatch.setattr(layer_stream, "resolve_stream_dtype", lambda device: "float32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, 10**9))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved_bytes)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda base: "unused-cache")
    monkeypatch.setattr(stripe_roots, "resolve_stripe_roots", lambda *args, **kwargs: ())
    weights = Mock(side_effect=RuntimeError("reached weight resolution"))
    sharder = Mock(side_effect=AssertionError("must not shard"))
    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", weights)
    monkeypatch.setattr(layer_shard, "shard_checkpoint", sharder)
    wrapper = stream_setup.StreamingSetupMixin()
    wrapper.device = "cuda:0"
    wrapper._trust_remote_code = False
    if optimizer_backend is not None:
        setattr(wrapper, optimizer_backend, {"enabled": True})

    if (free_bytes == 1 and not reserved_bytes) or override_bytes == 1:
        with pytest.raises(ValueError, match="before checkpoint loading") as error:
            wrapper._setup_streaming_transformers(cfg, cfg.training)
        assert "required=57344 bytes (0.000 GB), available=1 bytes (0.000 GB)." in str(error.value)
        weights.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="^reached weight resolution$"):
            wrapper._setup_streaming_transformers(cfg, cfg.training)
        weights.assert_called_once()
    sharder.assert_not_called()


def test_lorafa_counts_frozen_a_as_weights_only(tiny_config: tuple[str, Any]) -> None:
    plan = _plan(tiny_config, _settings(use_lorafa=True))
    assert plan.budget.profile == "peft_lorafa_fp32"
    assert plan.budget.weights.cuda_bytes == 14336
    assert plan.budget.gradients.cuda_bytes == 6144
    assert plan.budget.optimizer.cuda_bytes == 12288
    assert all(not tensor.trainable for tensor in plan.tensors if ".lora_A." in tensor.name)


def test_optimizer_override_profiles_are_unverified() -> None:
    from soup_cli.utils.stream_adapter_preflight import _optimizer_profile

    assert _optimizer_profile(_settings(use_galore=True)) == "unsupported: GaLore"
    assert _optimizer_profile(_settings(loraplus_lr_ratio=16)) == "unsupported: LoRA+"
    assert _optimizer_profile(_settings(use_lorafa=True, optimizer="sgd")) == (
        "unsupported: LoRA-FA optimizer conflict"
    )
    assert _optimizer_profile(_settings(use_lorafa=True, optimizer="adamw")) == "peft_lorafa_fp32"
    assert _optimizer_profile(_settings(use_lorafa=True, optimizer=None)) == "peft_lorafa_fp32"
    assert _optimizer_profile(_settings(use_lorafa=True, optimizer="adamw_hf")) == (
        "unsupported: LoRA-FA optimizer conflict"
    )
    assert _optimizer_profile(_settings(), optimizer_backend="DeepSpeed") == (
        "unsupported: DeepSpeed"
    )
    assert _optimizer_profile(_settings(), optimizer_backend="FSDP") == "unsupported: FSDP"


@pytest.mark.parametrize(
    ("package_version", "expected"),
    [("0.50.2", "bnb_0.50.2_adamw8bit_fp32"), ("0.50.3", "unsupported: adamw_8bit")],
)
def test_bnb_optimizer_requires_profiled_version(
    monkeypatch: pytest.MonkeyPatch, package_version: str, expected: str
) -> None:
    from soup_cli.utils import stream_adapter_preflight as preflight

    monkeypatch.setattr(preflight, "version", lambda package: package_version)
    assert preflight._optimizer_profile(_settings(optimizer="adamw_8bit")) == expected


def test_nf4_probe_uses_real_quantised_targets(tiny_config: tuple[str, Any]) -> None:
    pytest.importorskip("bitsandbytes")
    plain = _plan(tiny_config)
    nf4 = _plan(tiny_config, dtype="float32", quant="nf4", double_quant=False)
    assert nf4.tensors == plain.tensors
    assert nf4.budget == plain.budget
    assert nf4.quant_suffixes == (
        "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    )


def test_fused_moe_probe_keeps_build_config_unmodified(tmp_path: Any) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from soup_cli.utils.peft_wiring import attach_adapter

    config = transformers.Qwen3MoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, moe_intermediate_size=128,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    plan = _plan((str(tmp_path), config), _settings(moe_lora=True))
    assert plan.lora_config.target_parameters
    assert not plan.lora_config.rank_pattern
    actual = attach_adapter(
        transformers.AutoModelForCausalLM.from_config(config), plan.lora_config
    )
    real = {
        name: tuple(param.shape) for name, param in actual.named_parameters() if param.requires_grad
    }
    assert real == {tensor.name: tensor.shape for tensor in plan.tensors}
    assert any(".experts." in name for name in real)


@pytest.fixture
def deepseek_config(tmp_path: Any) -> tuple[str, Any]:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=128,
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path), config


@pytest.mark.parametrize("use_regex", [False, True])
def test_probe_counts_shared_expert_targets_without_routed_experts(
    deepseek_config: tuple[str, Any], use_regex: bool,
) -> None:
    from soup_cli.config.schema import LoraConfig

    target_modules = (
        r".*\.(?:self_attn\.(?:q_a_proj|q_b_proj)|mlp\.shared_experts\.(?:gate|up)_proj)"
        if use_regex else ["shared_experts.gate_proj", "shared_experts.up_proj"]
    )
    settings = _settings(
        lora=LoraConfig(r=4, alpha=8, dropout=0, target_modules=target_modules)
    )
    plan = _plan(deepseek_config, settings, dtype="float32", on_cuda=False)
    owners = {
        tensor.name.rsplit(".lora_", 1)[0].removeprefix("base_model.model.")
        for tensor in plan.tensors
    }
    expected = {f"model.layers.{idx}.mlp.shared_experts.{proj}_proj"
                for idx in (1, 2) for proj in ("gate", "up")}
    if use_regex:
        expected.update(f"model.layers.{idx}.self_attn.{proj}"
                        for idx in range(3) for proj in ("q_a_proj", "q_b_proj"))
    assert owners == expected
    assert plan.budget.stored_parameters == (3840 if use_regex else 1536)


def test_probe_refuses_a_missing_shared_expert_projection(tiny_config: tuple[str, Any]) -> None:
    settings = _settings()
    settings.lora.target_modules = ["q_proj", "shared_experts.up_proj"]
    with pytest.raises(ValueError) as error:
        _plan(tiny_config, settings)
    assert str(error.value) == (
        "training.lora.target_modules entry 'shared_experts.up_proj' matches no shared-expert "
        "gate_proj, up_proj or down_proj module."
    )


def test_probe_refuses_ambiguous_shared_expert_targets(deepseek_config: tuple[str, Any]) -> None:
    settings = _settings()
    settings.lora.target_modules = ["shared_experts.up_proj", "gate_proj"]
    with pytest.raises(ValueError) as error:
        _plan(deepseek_config, settings)
    assert str(error.value) == (
        "training.lora.target_modules ['gate_proj'] sit beside shared-expert targets. "
        "Name these projections by full path, or set training.moe_lora: true "
        "to target routed experts."
    )


def test_moe_lora_with_explicit_shared_target_matches_native_peft(
    deepseek_config: tuple[str, Any],
) -> None:
    from peft import get_peft_model

    from soup_cli.config.schema import LoraConfig
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    settings = _settings(
        moe_lora=True,
        lora=LoraConfig(r=4, alpha=8, dropout=0, target_modules=["shared_experts.up_proj"]),
    )
    plan = _plan(deepseek_config, settings, dtype="float32")
    model = build_meta_skeleton(deepseek_config[0], dtype="float32")
    native = get_peft_model(model, deepcopy(plan.lora_config))
    expected = {
        name: tuple(param.shape) for name, param in native.named_parameters() if ".lora_" in name
    }
    assert any(".experts." in name for name in expected)
    assert {tensor.name: tensor.shape for tensor in plan.tensors} == expected


def test_shared_expert_attach_restores_peft_conversion_on_exception(
    deepseek_config: tuple[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft
    from peft.utils import transformers_weight_conversion as conversion

    original = conversion.convert_peft_config_for_transformers

    def fail_attach(model: Any, config: Any) -> Any:
        assert conversion.convert_peft_config_for_transformers is not original
        raise RuntimeError("injected PEFT attach failure")

    settings = _settings()
    settings.lora.target_modules = ["shared_experts.up_proj"]
    monkeypatch.setattr(peft, "get_peft_model", fail_attach)
    with pytest.raises(RuntimeError, match="^injected PEFT attach failure$"):
        _plan(deepseek_config, settings)
    assert conversion.convert_peft_config_for_transformers is original


def test_attention_recommendation_rebuilds_and_preserves_rank_pattern(
    tiny_config: tuple[str, Any],
) -> None:
    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    settings = _settings()
    settings.lora.target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    settings.lora.rank_pattern = {"q_proj": 4}
    original = _plan(tiny_config, settings)
    message = recommend_stream_adapter_targets(
        original, *tiny_config, settings, available_cuda_bytes=120000, **_build_kwargs()
    )
    assert "attention only:" in message
    assert "attention + shared expert:" not in message
    assert "training.moe_lora: false" in message
    assert settings.lora.rank_pattern == {"q_proj": 4}
    assert "gate_proj" in settings.lora.target_modules

    candidate = deepcopy(settings)
    value = message.split("training.lora.target_modules: ", 1)[1].split("; keep", 1)[0]
    candidate.lora.target_modules = json.loads(value)
    plan = _plan(tiny_config, candidate)
    assert plan.budget.total.cuda_bytes == 98304
    assert all(".self_attn." in tensor.name for tensor in plan.tensors)
    assert all(
        tensor.shape[0] == 4 for tensor in plan.tensors if ".q_proj.lora_A." in tensor.name
    )
    assert recommend_stream_adapter_targets(
        original, *tiny_config, settings, available_cuda_bytes=1, **_build_kwargs()
    ) == " No smaller verified attention/shared-expert alternative fits the adapter budget."


def test_recommendation_contains_attention_and_shared_experts_only(
    deepseek_config: tuple[str, Any],
) -> None:
    from soup_cli.utils.stream_adapter_preflight import recommend_stream_adapter_targets

    settings = _settings(moe_lora=True)
    original = _plan(deepseek_config, settings, dtype="float32")
    message = recommend_stream_adapter_targets(
        original, *deepseek_config, settings, available_cuda_bytes=300000,
        **_build_kwargs(dtype="float32"),
    )
    assert "attention only:" in message
    assert "attention + shared expert:" in message
    assert settings.moe_lora is True
    assert settings.lora.target_modules == "auto"

    proposed = deepcopy(settings)
    proposed.moe_lora = False
    value = message.rsplit("training.lora.target_modules: ", 1)[1].split("; keep", 1)[0]
    proposed.lora.target_modules = json.loads(value)
    plan = _plan(deepseek_config, proposed, dtype="float32")
    assert any(".self_attn." in tensor.name for tensor in plan.tensors)
    assert any(".shared_experts." in tensor.name for tensor in plan.tensors)
    assert all(
        ".self_attn." in tensor.name or ".shared_experts." in tensor.name
        for tensor in plan.tensors
    )
    assert plan.budget.total.cuda_bytes < original.budget.total.cuda_bytes


def test_refusal_recommends_targets_that_train_in_streamed_setup(
    tiny_config: tuple[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import yaml
    from transformers import AutoTokenizer, LlamaForCausalLM

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer import stream_setup
    from soup_cli.utils import layer_stream, spectrum_scan

    base, config = tiny_config
    LlamaForCausalLM(config).save_pretrained(base)
    raw = dict(
        base=base,
        data=dict(train="unused.jsonl", max_length=64),
        training=dict(
            quantization="none",
            stream_layers=True,
            batch_size=1,
            gradient_accumulation_steps=1,
            stream_pin=False,
            optimizer="adamw_torch",
            lora=dict(r=8, alpha=8, dropout=0, target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            ]),
        ),
    )
    cfg = load_config_from_string(yaml.safe_dump(raw))
    monkeypatch.setattr(stream_setup, "refuse_if_data_parallel", lambda device: None)
    monkeypatch.setattr(layer_stream, "resolve_stream_dtype", lambda device: "float32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (120000, 120000))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=SimpleNamespace(
        pad_token="pad", eos_token="eos",
    )))
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    wrapper = stream_setup.StreamingSetupMixin()
    wrapper.device = "cuda:0"
    wrapper._trust_remote_code = False
    with monkeypatch.context() as patch:
        weights = Mock(side_effect=AssertionError("must refuse before weight resolution"))
        patch.setattr(spectrum_scan, "resolve_model_weights", weights)
        with pytest.raises(ValueError, match="attention only:") as error:
            wrapper._setup_streaming_transformers(cfg, cfg.training)
        weights.assert_not_called()

    message = str(error.value)
    assert "total=114688 CUDA bytes" in message
    value = message.split("training.lora.target_modules: ", 1)[1].split("; keep", 1)[0]
    raw["training"]["moe_lora"] = False
    raw["training"]["lora"]["target_modules"] = json.loads(value)
    proposed = load_config_from_string(yaml.safe_dump(raw))
    plan = _plan(tiny_config, proposed.training, dtype="float32")
    assert plan.budget.total.cuda_bytes == 114688

    wrapper.device = "cpu"
    try:
        wrapper._setup_streaming_transformers(proposed, proposed.training)
        trainable = {
            name.replace(".inner.", "."): tuple(param.shape)
            for name, param in wrapper.model.named_parameters()
            if param.requires_grad
        }
        resident = {
            tensor.name.replace(".inner.", "."): tensor.shape
            for tensor in plan.tensors if tensor.trainable
        }
        assert trainable == resident
        tokens = torch.arange(8).unsqueeze(0)
        loss = wrapper.model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        assert torch.isfinite(loss)
        assert any(
            param.grad is not None and torch.count_nonzero(param.grad) > 0
            for param in wrapper.model.parameters() if param.requires_grad
        )
    finally:
        wrapper._close_stream_runtime()


@pytest.mark.parametrize("optimizer_name", ["adamw_torch", "sgd", "adagrad", "rmsprop",
                                           "adafactor", "lorafa"])
def test_trainer_optimizer_state_matches_adapter_profile(
    tiny_config: tuple[str, Any], tmp_path: Any, optimizer_name: str,
) -> None:
    import torch
    from transformers import LlamaForCausalLM, Trainer, TrainingArguments

    from soup_cli.utils.peft_wiring import attach_adapter, attach_lorafa_optimizer

    use_lorafa = optimizer_name == "lorafa"
    settings = _settings(optimizer="adamw_torch" if use_lorafa else optimizer_name,
                         use_lorafa=use_lorafa)
    plan = _plan(tiny_config, settings, on_cuda=False)
    model = attach_adapter(LlamaForCausalLM(tiny_config[1]), plan.lora_config)
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path / "trainer"), optim=settings.optimizer,
        use_cpu=True, report_to="none",
    ))
    if use_lorafa:
        assert attach_lorafa_optimizer(trainer, settings)
    optimizer = trainer.create_optimizer()
    params = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in params} == {
        id(p) for group in optimizer.param_groups for p in group["params"] if p.requires_grad
    }
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    storages = {
        (str(tensor.device), tensor.untyped_storage().data_ptr()): tensor.untyped_storage().nbytes()
        for state in optimizer.state.values()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    }
    assert sum(storages.values()) == plan.budget.optimizer.cpu_bytes


@pytest.mark.parametrize("optimizer_name", ["adamw_8bit", "adamw_bnb_8bit"])
def test_trainer_bnb_factory_uses_profiled_options(tmp_path: Any, optimizer_name: str) -> None:
    pytest.importorskip("bitsandbytes")
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
                             report_to="none", optim=optimizer_name)
    optimizer_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    assert optimizer_cls.__module__.startswith("bitsandbytes.")
    assert kwargs["optim_bits"] == 8
    assert kwargs["is_paged"] is False
