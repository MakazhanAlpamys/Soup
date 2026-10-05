"""CUDA decode backend safety, static-input handling, and generation compatibility."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.utils import cuda_graphs


@pytest.fixture(autouse=True)
def _validated_torch(monkeypatch):
    """Every test here runs as if on the validated PyTorch; the gate has its own tests.
    Without this the suite would depend on whichever torch CI resolved."""
    monkeypatch.setattr(cuda_graphs, "_installed_torch_version", lambda: "2.14.0+cu130")


def _model(device: str = "cuda:0") -> object:
    model_type = type(
        "Qwen2ForCausalLM", (), {"__module__": "transformers.models.qwen2.modeling_qwen2"}
    )
    model = model_type()
    model.training = False
    model.device = device
    model.config = SimpleNamespace(model_type="qwen2", is_encoder_decoder=False)
    model.generation_config = SimpleNamespace(cache_implementation=None, temperature=0.7)
    model.hf_device_map = {"": 0}
    model.parameters = lambda: iter([SimpleNamespace(device=device)])
    model.buffers = lambda: iter([])
    model.modules = lambda: iter([model])
    model._supports_default_dynamic_cache = lambda: True
    return model


def test_generation_kwargs_preserve_defaults_and_register_once(monkeypatch):
    model = _model()
    before = vars(model.generation_config).copy()
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    result = cuda_graphs.cuda_graph_generation_kwargs(model)
    assert vars(model.generation_config) == before
    assert result["cache_implementation"] == "static"
    assert result["disable_compile"] is False
    config = result["compile_config"]
    assert config.backend == "soup_cudagraphs"
    assert config.mode is None
    assert config.fullgraph is True
    assert config.dynamic is False
    register.assert_called_once()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda model: setattr(model, "training", True), "eval"),
        (lambda model: setattr(model, "is_quantized", True), "quantiz"),
        (lambda model: setattr(model, "hf_quantizer", object()), "quantiz"),
        (lambda model: setattr(model, "hf_device_map", {"model": 0, "head": "cpu"}), "offload"),
        (lambda model: setattr(model, "hf_device_map", {"model": 0, "head": "disk"}), "offload"),
        (lambda model: setattr(model, "_supports_default_dynamic_cache", lambda: False), "cache"),
        (lambda model: setattr(model.config, "model_type", "gpt2"), "Qwen2.*Llama"),
        (lambda model: setattr(model, "_hf_hook", SimpleNamespace(offload=True)), "offload"),
    ],
)
def test_rejects_unsupported_models_before_registering(monkeypatch, change, message):
    model = _model()
    change(model)
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    register.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "meta", "mps"])
def test_rejects_non_cuda_parameters(monkeypatch, device):
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(_model(device))


def test_rejects_parameters_spread_across_gpus(monkeypatch):
    model = _model()
    model.parameters = lambda: iter(
        [SimpleNamespace(device="cuda:0"), SimpleNamespace(device="cuda:1")]
    )
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="single CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


def test_rejects_cpu_buffers_and_nested_offload_hooks(monkeypatch):
    model = _model()
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    model.buffers = lambda: iter([SimpleNamespace(device="cpu")])
    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    model.buffers = lambda: iter([])
    model._hf_hook = SimpleNamespace(hooks=[SimpleNamespace(offload=True)])
    with pytest.raises(RuntimeError, match="offload"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


def test_rejects_streaming_wrappers(monkeypatch):
    model = _model()
    streamed = type(
        "StreamedDecoderLayer", (), {"__module__": "soup_cli.utils.layer_stream_runtime"}
    )()
    model.modules = lambda: iter([model, streamed])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="stream"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_cache", False, "use_cache"),
        ("num_beams", 2, "beam"),
        ("prompt_lookup_num_tokens", 3, "assisted"),
        ("assistant_early_exit", 1, "assisted"),
        ("prefill_chunk_size", 32, "prefill"),
        ("cache_implementation", "offloaded", "cache"),
        ("cache_implementation", "quantized", "cache"),
    ],
)
def test_rejects_generation_defaults_that_bypass_or_conflict_with_graphs(
    monkeypatch, field, value, message
):
    model = _model()
    setattr(model.generation_config, field, value)
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    register.assert_not_called()


def test_cache_disabled_in_model_config_and_explicit_generation_override(monkeypatch):
    model = _model()
    model.config.use_cache = False
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="use_cache"):
        cuda_graphs.cuda_graph_generation_kwargs(model)
    model.generation_config.use_cache = True
    assert cuda_graphs.cuda_graph_generation_kwargs(model)["cache_implementation"] == "static"


def test_rejects_training_submodule_in_eval_model(monkeypatch):
    model = _model()
    model.modules = lambda: iter([model, SimpleNamespace(training=True)])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="eval"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize("implementation", ["flash_attention_2", "flash_attention_3"])
def test_rejects_flash_attention_partial_graph_override(monkeypatch, implementation):
    model = _model()
    model.config._attn_implementation = implementation
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="attention"):
        cuda_graphs.cuda_graph_generation_kwargs(model)


@pytest.mark.parametrize(
    "metadata", [None, SimpleNamespace(), SimpleNamespace(static_input_indices=[])]
)
def test_missing_static_metadata_fails_closed(metadata):
    context = SimpleNamespace(fw_metadata=metadata)
    with pytest.raises(RuntimeError, match="static input"):
        cuda_graphs._static_input_indices(context, 4)


@pytest.mark.parametrize("indices", [[-1], [4], [True], [1, 1], ["1"]])
def test_invalid_static_metadata_fails_closed(indices):
    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=indices))
    with pytest.raises(RuntimeError, match="static input"):
        cuda_graphs._static_input_indices(context, 4)


def test_static_indices_use_aot_coordinates_including_inlined_weights():
    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=[0, 2, 3]))
    assert cuda_graphs._static_input_indices(context, 4) == [0, 2, 3]


def test_missing_private_api_has_actionable_error(monkeypatch):
    def missing(name):
        raise ImportError("removed private module")

    monkeypatch.setattr(cuda_graphs, "import_module", missing)
    with pytest.raises(RuntimeError, match="PyTorch.*CUDA graph"):
        cuda_graphs._load_backend_api()


def test_backend_registration_is_idempotent_and_rejects_name_collision(monkeypatch):
    registry = {}

    def register(backend, *, name):
        assert name not in registry
        registry[name] = backend

    register_mock = Mock(side_effect=register)
    monkeypatch.setattr(
        cuda_graphs,
        "_load_backend_api",
        lambda: SimpleNamespace(registry=registry, register_backend=register_mock),
    )
    cuda_graphs._register_backend()
    cuda_graphs._register_backend()
    register_mock.assert_called_once_with(cuda_graphs._soup_cudagraphs, name="soup_cudagraphs")
    registry["soup_cudagraphs"] = object()
    with pytest.raises(RuntimeError, match="already registered"):
        cuda_graphs._register_backend()


def test_backend_refuses_training_before_graph_capture(monkeypatch):
    import torch

    monkeypatch.setattr(cuda_graphs, "_load_backend_api", lambda: SimpleNamespace(torch=torch))
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        cuda_graphs._soup_cudagraphs(object(), [])


def test_backend_uses_static_metadata_and_preserves_functionalization(monkeypatch):
    import torch

    context = SimpleNamespace(fw_metadata=SimpleNamespace(static_input_indices=[0, 2]))
    fake_graph = SimpleNamespace(graph=object())
    captured = {}

    def replay(inputs):
        return inputs

    def aot_autograd(**kwargs):
        captured.update(kwargs)
        return lambda graph, inputs: kwargs["inference_compiler"](graph, inputs)

    api = SimpleNamespace(
        torch=torch,
        tracing_context=SimpleNamespace(try_get=lambda: context),
        aot_autograd=aot_autograd,
        boxed_nop=lambda graph, inputs: replay,
        find_input_mutations=lambda graph: set(),
        get_device_node_mapping=lambda graph: {torch.device("cuda:0"): object()},
        check_devices=lambda devices: None,
        incompatible_node=lambda graph: None,
        get_stack_traces=lambda graph: [],
        get_placeholder_info=lambda graph: [],
        cudagraphify=Mock(return_value=replay),
    )
    monkeypatch.setattr(cuda_graphs, "_load_backend_api", lambda: api)
    with torch.no_grad():
        result = cuda_graphs._soup_cudagraphs(fake_graph, [1, 2, 3])
    assert captured["keep_inference_input_mutations"] is False
    assert api.cudagraphify.call_args.args[2] == [0, 2]
    assert api.cudagraphify.call_args.kwargs["is_inference"] is True
    assert result is replay
    with pytest.raises(RuntimeError, match="inference"):
        captured["bw_compiler"]()
    with pytest.raises(RuntimeError, match="inference"):
        captured["fw_compiler"]()

    # If metadata disappears or mutation targets an ordinary input, no capture
    # (and therefore no graph-pool weight allocation) may be attempted.
    api.cudagraphify.reset_mock()
    context.fw_metadata = None
    with pytest.raises(RuntimeError, match="static input"):
        captured["inference_compiler"](fake_graph, [1, 2, 3])
    api.cudagraphify.assert_not_called()
    context.fw_metadata = SimpleNamespace(static_input_indices=[0, 2])
    api.find_input_mutations = lambda graph: {1}
    with pytest.raises(RuntimeError, match="mutation"):
        captured["inference_compiler"](fake_graph, [1, 2, 3])
    api.cudagraphify.assert_not_called()


@pytest.mark.gpu
def test_tiny_llama_repeated_generation_keeps_tokens_and_default_config():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        .to(device="cuda", dtype=torch.float16)
        .eval()
    )
    before = model.generation_config.to_dict()
    prompts = ([1, 5, 9], [1, 7, 8, 4], [1, 5, 9])
    kwargs = cuda_graphs.cuda_graph_generation_kwargs(model)
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([prompt], device="cuda")
            mask = torch.ones_like(ids)
            normal = model.generate(ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
            graphed = model.generate(
                ids, attention_mask=mask, max_new_tokens=8, do_sample=False, **kwargs
            )
            torch.testing.assert_close(graphed, normal, rtol=0, atol=0)
    assert model.generation_config.to_dict() == before


def _peft(base=None, **overrides):
    """A PeftModelForCausalLM stand-in forwarding the attributes the validator reads."""
    base = base if base is not None else _model()
    wrapper_type = type("PeftModelForCausalLM", (), {"__module__": "peft.peft_model"})
    model = wrapper_type()
    model.training = False
    model.device = base.device
    model.config = base.config
    model.generation_config = base.generation_config
    model.hf_device_map = base.hf_device_map
    model.parameters = base.parameters
    model.buffers = base.buffers
    model.modules = lambda: iter([model, base])
    model._supports_default_dynamic_cache = base._supports_default_dynamic_cache
    model.get_base_model = lambda: base
    model.peft_config = {"default": SimpleNamespace(peft_type="LORA", use_dora=False)}
    model.active_adapters = ["default"]
    for name, value in overrides.items():
        setattr(model, name, value)
    return model


def test_accepts_one_unmerged_lora_over_a_stock_model(monkeypatch):
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    assert cuda_graphs.cuda_graph_generation_kwargs(_peft())["cache_implementation"] == "static"


def test_accepts_the_peft_enum_spelling_and_a_string_active_adapter(monkeypatch):
    # peft.PeftType is a str Enum whose .value is "LORA"; active_adapter can be a str.
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    config = SimpleNamespace(peft_type=SimpleNamespace(value="LORA"), use_dora=False)
    model = _peft(peft_config={"default": config}, active_adapters="default")
    assert cuda_graphs.cuda_graph_generation_kwargs(model)["cache_implementation"] == "static"


_LORA = SimpleNamespace(peft_type="LORA", use_dora=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"peft_config": {"default": SimpleNamespace(peft_type="IA3", use_dora=False)}},
            "LoRA adapters only",
        ),
        ({"peft_config": {"default": SimpleNamespace(peft_type="LORA", use_dora=True)}}, "DoRA"),
        ({"active_adapters": ["a", "b"], "peft_config": {"a": _LORA, "b": _LORA}}, "one active"),
        ({"active_adapters": []}, "one active"),
        ({"active_adapters": ["missing"]}, "one active"),
        ({"peft_config": None}, "single LoRA"),
        ({"get_base_model": None}, "single LoRA"),
    ],
)
def test_rejects_unsupported_peft_wrappers_before_registering(monkeypatch, overrides, message):
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    with pytest.raises(RuntimeError, match=message):
        cuda_graphs.cuda_graph_generation_kwargs(_peft(**overrides))
    register.assert_not_called()


def test_rejects_a_merged_adapter(monkeypatch):
    base = _model()
    base.modules = lambda: iter([base, SimpleNamespace(training=False, merged=True)])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="unmerged"):
        cuda_graphs.cuda_graph_generation_kwargs(_peft(base))


def test_a_peft_wrapper_over_an_unsupported_base_is_refused(monkeypatch):
    base = _model()
    base.config.model_type = "gpt2"
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="Qwen2.*Llama"):
        cuda_graphs.cuda_graph_generation_kwargs(_peft(base))


@pytest.mark.gpu
def test_tiny_llama_unmerged_lora_matches_eager_exactly(monkeypatch):
    """THE PROBE (spec 1.1). Mirrors Soup's real load path: fp16 base on CUDA, then
    PEFT, which autocasts the adapter to fp32; the adapter stays unmerged.

    Counts compiles through the backend (cudagraphify_impl calls): generate() silently
    runs eager when its auto-compile criteria fail, and eager matches eager trivially,
    so identical tokens alone would not show the backend ran at all."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(23)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).to(device="cuda", dtype=torch.float16)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            init_lora_weights=False,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    ).eval()
    ids = torch.tensor([[1, 5, 9]], device="cuda")
    with torch.no_grad():
        adapted = model(input_ids=ids).logits
        with model.disable_adapter():
            plain = model(input_ids=ids).logits
    assert not torch.equal(adapted, plain), "control: the adapter must change the logits"

    captures = []
    real_api = cuda_graphs._load_backend_api

    def counting_api():
        api = real_api()
        capture = api.cudagraphify

        def counted(*args, **kwargs):
            captures.append(1)
            return capture(*args, **kwargs)

        api.cudagraphify = counted
        return api

    monkeypatch.setattr(cuda_graphs, "_load_backend_api", counting_api)
    kwargs = cuda_graphs.cuda_graph_generation_kwargs(model)
    with torch.no_grad():
        for prompt in ([1, 5, 9], [1, 7, 8, 4, 3, 6], [1, 5, 9]):
            ids = torch.tensor([prompt], device="cuda")
            mask = torch.ones_like(ids)
            normal = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False
            )
            graphed = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=8,
                do_sample=False,
                **kwargs,
            )
            torch.testing.assert_close(graphed, normal, rtol=0, atol=0)
    assert captures, "the decode never compiled through the backend: generate() ran eager"


@pytest.mark.parametrize(
    "version", ["2.14.0", "2.14.0+cu130", "2.14.0a0+git1234", "2.15.1", "3.0.0"]
)
def test_validated_or_newer_torch_passes_the_gate(version):
    cuda_graphs._require_tested_torch(version)


@pytest.mark.parametrize("version", ["2.6.0", "2.13.1+cu126", "1.13.1"])
def test_older_torch_is_refused_with_the_tested_version_named(version):
    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.14 \(validated on 2\.14\.0\)"):
        cuda_graphs._require_tested_torch(version)


@pytest.mark.parametrize("version", ["", "nightly", "two.fourteen"])
def test_an_unreadable_torch_version_is_refused(version):
    with pytest.raises(RuntimeError, match="Cannot read the PyTorch version"):
        cuda_graphs._require_tested_torch(version)


def test_generation_kwargs_check_torch_before_the_model_and_registration(monkeypatch):
    monkeypatch.setattr(cuda_graphs, "_installed_torch_version", lambda: "2.6.0")
    register = Mock()
    monkeypatch.setattr(cuda_graphs, "_register_backend", register)
    broken = _model()
    broken.training = True  # would raise "eval" if the model were checked first
    with pytest.raises(RuntimeError, match=r"2\.14"):
        cuda_graphs.cuda_graph_generation_kwargs(broken)
    register.assert_not_called()


def test_the_real_cudagraphify_accepts_the_call_the_backend_makes():
    """cudagraphify_impl only STORES its keyword arguments; the real cudagraphify is
    called later, inside deferred_cudagraphify, at graph run time in the middle of a
    generate(). Binding that call at pre-flight is where a drift can be refused cleanly."""
    from torch._inductor import cudagraph_trees

    cuda_graphs._require_cudagraphify_signature(cudagraph_trees.cudagraphify)


def test_a_drifted_cudagraphify_signature_is_refused_at_pre_flight():
    def drifted(
        model,
        inputs,
        static_input_idxs=(),
        *,
        device_index,
        is_backward,
        is_inference,
        stack_traces=None,
        mutated_input_idxs=(),
    ):
        raise AssertionError("binding must not call it")  # 'placeholders' was dropped

    with pytest.raises(RuntimeError, match="API changed.*Omit --cuda-graphs"):
        cuda_graphs._require_cudagraphify_signature(drifted)


def test_loading_the_backend_api_checks_the_deferred_signature(monkeypatch):
    from torch._inductor import cudagraph_trees

    seen = []
    monkeypatch.setattr(cuda_graphs, "_require_cudagraphify_signature", seen.append)
    cuda_graphs._load_backend_api()
    assert seen == [cudagraph_trees.cudagraphify]


def test_rejects_a_lora_variant_such_as_alora_or_arrow(monkeypatch):
    base = _model()
    variant = SimpleNamespace(training=False, merged=False, lora_variant={"default": object()})
    base.modules = lambda: iter([base, variant])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    with pytest.raises(RuntimeError, match="plain LoRA"):
        cuda_graphs.cuda_graph_generation_kwargs(_peft(base))


def test_a_plain_lora_layer_has_an_empty_variant_map_and_passes(monkeypatch):
    base = _model()
    plain = SimpleNamespace(training=False, merged=False, lora_variant={})
    base.modules = lambda: iter([base, plain])
    monkeypatch.setattr(cuda_graphs, "_register_backend", Mock())
    assert cuda_graphs.cuda_graph_generation_kwargs(_peft(base))["cache_implementation"] == "static"
