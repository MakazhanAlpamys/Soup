"""Opt-in CUDA graph decoding with resident weights and static KV buffers.

Transformers retains ownership of prefill, sampling, stopping and cache setup.
The backend captures its decode forward using PyTorch's graph trees. Static
input indices come from AOT metadata, including parameters already inlined by
Dynamo: counting only newly lifted parameters would duplicate model weights in
the graph pool and copy them on every token.
"""

from __future__ import annotations

import functools
import inspect
import re
import threading
from importlib import import_module
from itertools import chain
from types import SimpleNamespace
from typing import Any

_BACKEND_NAME = "soup_cudagraphs"
_REGISTRATION_LOCK = threading.Lock()
_STOCK_MODELS = {
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    "llama": ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
}
# The private compiler APIs below were validated on this PyTorch only (spec 1.2).
CUDA_GRAPHS_MIN_TORCH = (2, 14)
_TESTED_TORCH = "2.14.0"
_TORCH_RELEASE_RE = re.compile(r"(\d+)\.(\d+)")


def _installed_torch_version() -> str:
    return str(import_module("torch").__version__)


def _require_tested_torch(version: str) -> None:
    """Refuse a PyTorch older than the one these private APIs were validated on."""
    match = _TORCH_RELEASE_RE.match(version)
    if match is None:
        raise RuntimeError(f"Cannot read the PyTorch version {version!r}; omit --cuda-graphs")
    if (int(match.group(1)), int(match.group(2))) < CUDA_GRAPHS_MIN_TORCH:
        floor = ".".join(str(part) for part in CUDA_GRAPHS_MIN_TORCH)
        raise RuntimeError(
            f"--cuda-graphs needs PyTorch >= {floor} (validated on {_TESTED_TORCH}); "
            f"this is {version}. Omit --cuda-graphs or upgrade PyTorch"
        )


def _cuda_device_key(device: Any) -> str:
    value = f"cuda:{device}" if isinstance(device, int) else str(device)
    if value == "cuda":
        value = "cuda:0"
    if not value.startswith("cuda:"):
        raise RuntimeError("CUDA graphs require every model tensor on a single CUDA device")
    return value


def _has_offload_hook(hook: Any) -> bool:
    return bool(getattr(hook, "offload", False)) or any(
        _has_offload_hook(child) for child in getattr(hook, "hooks", ())
    )


_PEFT_REFUSAL = "CUDA graph decoding supports a single LoRA adapter over a stock model"


def _unwrap_peft(model: Any) -> Any:
    """Return the transformers model under a PEFT wrapper; refuse what graphs cannot hold.

    The adapter stays UNMERGED: without --cuda-graphs Soup runs it unmerged, and
    merging would change outputs by rounding, breaking this option's contract of
    identical tokens.
    """
    if not type(model).__module__.startswith("peft."):
        return model
    get_base_model = getattr(model, "get_base_model", None)
    configs = getattr(model, "peft_config", None)
    if not callable(get_base_model) or not isinstance(configs, dict):
        raise RuntimeError(_PEFT_REFUSAL)
    active = getattr(model, "active_adapters", None)
    if isinstance(active, str):
        active = [active]
    if not isinstance(active, (list, tuple)) or len(active) != 1 or active[0] not in configs:
        raise RuntimeError("CUDA graph decoding supports exactly one active LoRA adapter")
    adapter = configs[active[0]]
    peft_type = getattr(adapter, "peft_type", None)
    peft_type = getattr(peft_type, "value", peft_type)
    if peft_type != "LORA":
        raise RuntimeError(f"CUDA graph decoding supports LoRA adapters only, not {peft_type}")
    if getattr(adapter, "use_dora", False):
        raise RuntimeError("CUDA graph decoding does not support DoRA adapters")
    base = get_base_model()
    for module in base.modules():
        if getattr(module, "merged", False) is True:
            raise RuntimeError(
                "CUDA graph decoding requires an unmerged adapter; merged weights would not "
                "match normal generation"
            )
        # peft keeps {} here for plain LoRA; DoRA, aLoRA, Arrow and QALoRA register a
        # variant whose forward was never validated under capture.
        if getattr(module, "lora_variant", None):
            raise RuntimeError(
                "CUDA graph decoding supports plain LoRA only; this adapter uses a LoRA "
                "variant (DoRA, aLoRA, Arrow, ...)"
            )
    return base


def _validate_model(model: Any) -> None:
    if getattr(model, "training", True):
        raise RuntimeError("CUDA graph decoding requires model.eval()")
    base = _unwrap_peft(model)
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    expected = _STOCK_MODELS.get(model_type)
    if expected != (type(base).__module__, type(base).__name__):
        raise RuntimeError("CUDA graph decoding supports stock Qwen2 and Llama causal models")
    if getattr(config, "is_encoder_decoder", False):
        raise RuntimeError("CUDA graph decoding requires a decoder-only model")
    attention = getattr(config, "_attn_implementation", None)
    if isinstance(attention, str) and attention.startswith("flash_attention"):
        raise RuntimeError(
            "CUDA graph decoding does not support flash attention's partial-graph override"
        )
    if (
        any(
            bool(getattr(model, flag, False))
            for flag in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit")
        )
        or getattr(model, "hf_quantizer", None) is not None
        or getattr(config, "quantization_config", None) is not None
    ):
        raise RuntimeError("CUDA graph decoding currently requires an unquantized model")
    supports_cache = getattr(model, "_supports_default_dynamic_cache", None)
    if not callable(supports_cache) or not supports_cache():
        raise RuntimeError("CUDA graph decoding requires Transformers' standard cache API")
    generation = getattr(model, "generation_config", None)
    use_cache = getattr(generation, "use_cache", None)
    if use_cache is None:
        use_cache = getattr(config, "use_cache", True)
    if use_cache is False:
        raise RuntimeError("CUDA graph decoding requires use_cache=True")
    if getattr(generation, "num_beams", None) not in (None, 1):
        raise RuntimeError("CUDA graph decoding supports greedy or sampling, without beam search")
    if any(
        getattr(generation, name, None) is not None
        for name in ("prompt_lookup_num_tokens", "assistant_early_exit")
    ) or getattr(generation, "is_assistant", False):
        raise RuntimeError("CUDA graph decoding does not support assisted generation")
    if getattr(generation, "cache_implementation", None) not in (None, "dynamic", "static"):
        raise RuntimeError("CUDA graphs cannot replace an offloaded or quantized cache")
    if getattr(generation, "prefill_chunk_size", None) is not None:
        raise RuntimeError("CUDA graph decoding does not support compiled chunked prefill")
    for module in model.modules():
        if getattr(module, "training", False):
            raise RuntimeError("CUDA graph decoding requires every model module in eval mode")
        if type(module).__module__.startswith("soup_cli.utils.layer_stream"):
            raise RuntimeError("CUDA graph decoding does not support layer streaming")
        if _has_offload_hook(getattr(module, "_hf_hook", None)):
            raise RuntimeError("CUDA graph decoding does not support CPU or disk offload")
    expected_device = _cuda_device_key(model.device)
    devices = {
        _cuda_device_key(tensor.device) for tensor in chain(model.parameters(), model.buffers())
    }
    if devices != {expected_device}:
        raise RuntimeError("CUDA graph decoding requires all weights on a single CUDA device")
    for device in getattr(model, "hf_device_map", {}).values():
        if str(device) in {"cpu", "disk", "meta"}:
            raise RuntimeError("CUDA graph decoding does not support CPU or disk offload")
        if _cuda_device_key(device) != expected_device:
            raise RuntimeError("CUDA graph decoding requires a single CUDA device")


def _load_backend_api() -> SimpleNamespace:
    """Fail clearly when this optional backend's private PyTorch APIs differ."""
    names = {
        "aot_autograd": ("torch._dynamo.backends.common", "aot_autograd"),
        "boxed_nop": ("torch._dynamo.backends.debugging", "boxed_nop"),
        "find_input_mutations": ("torch._dynamo.backends.cudagraphs", "find_input_mutations"),
        "get_device_node_mapping": ("torch._dynamo.backends.cudagraphs", "get_device_node_mapping"),
        "get_stack_traces": ("torch._dynamo.backends.cudagraphs", "get_stack_traces"),
        "cudagraphify": ("torch._inductor.cudagraph_trees", "cudagraphify_impl"),
        "cudagraphify_target": ("torch._inductor.cudagraph_trees", "cudagraphify"),
        "check_devices": (
            "torch._inductor.cudagraph_utils",
            "check_multiple_devices_or_any_cpu_nodes",
        ),
        "get_placeholder_info": ("torch._inductor.cudagraph_utils", "get_placeholder_info"),
        "incompatible_node": ("torch._inductor.utils", "get_first_incompatible_cudagraph_node"),
        "register_backend": ("torch._dynamo.backends.registry", "register_backend"),
    }
    try:
        values = {
            key: getattr(import_module(module), name) for key, (module, name) in names.items()
        }
        if not all(callable(value) for value in values.values()):
            raise AttributeError("required backend function is not callable")
        values["torch"] = import_module("torch")
        values["tracing_context"] = import_module("torch._guards").TracingContext
        values["registry"] = import_module("torch._dynamo.backends.registry")._COMPILER_FNS
        if not callable(values["torch"].compile) or not callable(values["tracing_context"].try_get):
            raise AttributeError("required compiler capability is missing")
        if not isinstance(values["registry"], dict):
            raise AttributeError("backend registry is unavailable")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This PyTorch installation lacks Soup's required CUDA graph compiler APIs; "
            "omit --cuda-graphs or install a compatible PyTorch version"
        ) from exc
    _require_cudagraphify_signature(values["cudagraphify_target"])
    return SimpleNamespace(**values)


def _require_cudagraphify_signature(target: Any) -> None:
    """Bind, at pre-flight, the call cudagraph trees make only at graph run time.

    cudagraphify_impl stores its keyword arguments and forwards them to cudagraphify
    inside deferred_cudagraphify, i.e. mid-generate. A drifted keyword would surface
    there as a bare TypeError; binding the same call here refuses it before any output.
    """
    try:
        inspect.signature(target).bind(
            None, [], [], device_index=0, is_backward=False, is_inference=True,
            stack_traces=[], placeholders=[], mutated_input_idxs=set(),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"PyTorch's CUDA graph API changed ({exc}); Soup's backend was validated on "
            f"{_TESTED_TORCH}. Omit --cuda-graphs"
        ) from exc


def _register_backend() -> None:
    api = _load_backend_api()
    with _REGISTRATION_LOCK:
        existing = api.registry.get(_BACKEND_NAME)
        if existing is None:
            api.register_backend(_soup_cudagraphs, name=_BACKEND_NAME)
        elif existing is not _soup_cudagraphs:
            raise RuntimeError(f"PyTorch backend name {_BACKEND_NAME!r} is already registered")


def cuda_graph_generation_kwargs(model: Any) -> dict[str, Any]:
    """Return opt-in generate kwargs without changing model generation defaults.

    Only resident, unquantized, eval-mode Qwen2/Llama models are supported.
    Compilation happens on the first decode; subsequent calls reuse compiled
    code. Graph trees validate static addresses when a request creates a new
    cache. Callers must serialize requests using the same model instance.
    """
    _require_tested_torch(_installed_torch_version())
    _validate_model(model)
    try:
        from transformers import CompileConfig

        compile_config = CompileConfig(
            backend=_BACKEND_NAME, mode=None, dynamic=False, fullgraph=True
        )
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "This Transformers version lacks CUDA graph generation configuration; "
            "omit --cuda-graphs or install a compatible Transformers version"
        ) from exc
    _register_backend()
    return {
        "cache_implementation": "static",
        "disable_compile": False,
        "compile_config": compile_config,
    }


def _static_input_indices(context: Any, input_count: int) -> list[int]:
    metadata = getattr(context, "fw_metadata", None)
    indices = getattr(metadata, "static_input_indices", None)
    if (
        not isinstance(indices, (list, tuple))
        or not indices
        or any(type(index) is not int or not 0 <= index < input_count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise RuntimeError(
            "CUDA graph compiler did not provide valid static input metadata; "
            "refusing capture that could duplicate model weights. Omit --cuda-graphs"
        )
    return list(indices)


def _soup_cudagraphs(dynamo_model: Any, dynamo_inputs: list[Any], **kwargs: Any) -> Any:
    api = _load_backend_api()
    if api.torch.is_grad_enabled():
        raise RuntimeError("Soup's CUDA graph backend supports inference with gradients disabled")

    def forward_compiler(
        aot_model: Any, aot_inputs: list[Any], *, is_inference: bool = False
    ) -> Any:
        if not is_inference:
            raise RuntimeError("Soup's CUDA graph backend supports inference only")
        indices = _static_input_indices(api.tracing_context.try_get(), len(aot_inputs))
        mutated = api.find_input_mutations(aot_model.graph)
        unsupported = mutated - set(indices)
        devices = api.get_device_node_mapping(aot_model)
        reason = api.check_devices(devices)
        if not reason and (len(devices) != 1 or next(iter(devices)).type != "cuda"):
            reason = "graph operations must use a single CUDA device"
        incompatible = api.incompatible_node(aot_model)
        if not reason and incompatible is not None:
            reason = f"incompatible operation {incompatible.name}"
        if not reason and unsupported:
            reason = "mutation of inputs without stable addresses"
        if reason:
            raise RuntimeError(f"Cannot capture CUDA decode: {reason}. Omit --cuda-graphs")
        device = next(device for device in devices if device.type == "cuda")
        # These keywords are only bound when the graph first runs (deferred_cudagraphify);
        # _require_cudagraphify_signature checks them at pre-flight instead.
        compiled = api.cudagraphify(
            api.boxed_nop(aot_model, aot_inputs),
            aot_inputs,
            indices,
            device_index=device.index,
            is_backward=False,
            is_inference=True,
            stack_traces=api.get_stack_traces(aot_model),
            placeholders=api.get_placeholder_info(aot_model.graph),
            mutated_input_idxs=mutated,
        )
        compiled._boxed_call = True
        return compiled

    def reject_backward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Soup's CUDA graph backend supports inference only")

    compiler = api.aot_autograd(
        fw_compiler=reject_backward,
        bw_compiler=reject_backward,
        inference_compiler=functools.partial(forward_compiler, is_inference=True),
        keep_inference_input_mutations=False,
    )
    return compiler(dynamo_model, dynamo_inputs)
