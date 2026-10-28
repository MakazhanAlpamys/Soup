"""A model's shape for the VRAM estimate: dims and exact size, without torch or weights.

#2049: the two VRAM predictors guessed the hidden size from the parameter count and
had no vocabulary at all, so the logits term, which dominates on 150k-250k
vocabularies, could not be priced. The shape comes from ``config.json``: a local
model directory first, then the Hugging Face cache, then the Hub unless
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` is set. The exact parameter count
comes from the safetensors headers (a local directory, a cached snapshot, or the
Hub's header-only metadata call); without them it is counted from ``config.json``.

With no ``config.json`` anywhere, the shape is guessed from the parameter count
(``model_size_from_name``) and ``vocab_size`` is ``None``: the estimate then says
what it could not price instead of inventing a vocabulary.

No torch import, and no weight is downloaded.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

#: A config.json larger than this is not a config.json.
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
#: The largest hidden, intermediate, vocabulary, head or head-width figure a config.json
#: may state, and the most layers. Real models are two orders of magnitude below both;
#: a config past them is treated as unreadable.
_MAX_DIM = 2**24
_MAX_LAYERS = 2**16
_MAX_EXPERTS = 2**16

_OFFLINE_VALUES = frozenset({"1", "TRUE", "YES", "ON"})

# ``owner/name`` or ``name``: what ``hf_hub_download`` takes as a repo id.
_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*(/[\w.-]+)?")

#: Seconds allowed for each optional Hub metadata request.
HUB_TIMEOUT_S = 5.0
#: Seconds allowed for the whole safetensors-metadata lookup, which makes one request
#: per shard; past this the count comes from config.json instead.
HUB_METADATA_DEADLINE_S = 20.0

# The sizes ``get_model_arch`` used to guess from a parameter count, kept as the
# fallback when no config.json can be read: params (B) -> (hidden, layers, intermediate).
_KNOWN_ARCHS: dict[float, tuple[int, int, int]] = {
    0.5: (896, 24, 4864),
    1.0: (2048, 22, 5632),
    1.5: (2048, 28, 5632),
    3.0: (3072, 28, 8192),
    7.0: (4096, 32, 11008),
    8.0: (4096, 32, 14336),
    13.0: (5120, 40, 13824),
    14.0: (5120, 40, 14336),
    32.0: (5120, 64, 27648),
    34.0: (8192, 48, 22016),
    70.0: (8192, 80, 28672),
}


@dataclass(frozen=True)
class ModelShape:
    """What the estimate needs to know about a model."""

    params: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: Optional[int]
    tie_word_embeddings: bool
    model_type: str
    #: "safetensors", "hub safetensors metadata", "config.json" or "name".
    params_source: str
    #: "config.json" when the dims were read, "guessed" when derived from params.
    shape_source: str
    #: A mixture-of-experts config (it names a count of experts). The estimate prices
    #: such a model as a dense one, and says it is outside what it was fitted on.
    moe: bool = False

    @property
    def guessed(self) -> bool:
        return self.shape_source != "config.json"


def hub_offline() -> bool:
    """True when the environment asks Hugging Face libraries not to use the network."""
    return any(
        os.environ.get(name, "").upper() in _OFFLINE_VALUES
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _read_json_file(path: str) -> Optional[dict]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return None
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, RecursionError):
        # Unreadable or malformed: no config.json.
        return None
    return data if isinstance(data, dict) else None


def _hub_lookup(repo: str, *, allow_network: bool) -> tuple[Optional[dict], Optional[str]]:
    """``config.json`` of a Hub repo and the cached snapshot directory, if any.

    The single place this module reads the Hugging Face cache or the Hub for a
    config; the test suite replaces it so no test depends on a cache or a network.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None, None
    snapshot: Optional[str] = None
    try:
        cached = try_to_load_from_cache(repo, "config.json")
    except Exception:  # noqa: BLE001 - a malformed cache is the same as no cache
        cached = None
    if isinstance(cached, str):
        snapshot = os.path.dirname(cached)
        return _read_json_file(cached), snapshot
    if not allow_network or hub_offline():
        return None, None
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

        # The size first, from a HEAD request: a file over the cap is never fetched.
        head = get_hf_file_metadata(hf_hub_url(repo, "config.json"), timeout=HUB_TIMEOUT_S)
        size = getattr(head, "size", None)
        if not isinstance(size, int) or size > _MAX_CONFIG_BYTES:
            return None, None
        path = hf_hub_download(repo, "config.json", etag_timeout=HUB_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - gated, missing or offline: no config, and no echo
        return None, None
    return _read_json_file(path), os.path.dirname(path)


def _config_cached(repo: str) -> bool:
    """Whether the Hugging Face cache already holds ``repo``'s ``config.json``."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:  # noqa: BLE001 - a malformed cache is the same as no cache
        return False


def _within_deadline(call: Any, seconds: float) -> Any:
    """``call()``'s result, or ``None`` if it raises or has not returned in ``seconds``.

    It runs on a daemon thread, so a request that stalls past the deadline is left
    behind instead of holding up the caller or the interpreter's exit.
    """
    result: list[Any] = []

    def run() -> None:
        try:
            result.append(call())
        except Exception:  # noqa: BLE001 - optional metadata; never echo provider errors
            pass

    worker = threading.Thread(target=run, name="soup-hub-metadata", daemon=True)
    worker.start()
    worker.join(seconds)
    return result[0] if result else None


def _hub_param_count(repo: str, *, allow_network: bool) -> Optional[int]:
    """Exact parameter count from the Hub's safetensors headers, without the weights.

    ``HfApi.get_safetensors_metadata`` arrived in huggingface-hub 0.20; Soup's floor
    is 0.16, so an older hub returns ``None`` and the count comes from config.json.
    The suite replaces this function too.
    """
    if not allow_network or hub_offline():
        return None
    try:
        from soup_cli.utils.hf import get_hf_api

        api = get_hf_api()
        getter = getattr(api, "get_safetensors_metadata", None)
        if getter is None:
            return None
        # Newer hubs take a per-request timeout; the deadline bounds the whole lookup.
        kwargs = (
            {"timeout": HUB_TIMEOUT_S}
            if "timeout" in inspect.signature(getter).parameters
            else {}
        )
        metadata = _within_deadline(lambda: getter(repo, **kwargs), HUB_METADATA_DEADLINE_S)
        counts = getattr(metadata, "parameter_count", None)
        if not isinstance(counts, dict) or not counts:
            return None
        total = sum(int(v) for v in counts.values())
    except Exception:  # noqa: BLE001 - optional metadata; never echo provider errors
        return None
    return total if total > 0 else None


def _int_field(config: dict, *names: str) -> Optional[int]:
    for name in names:
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _text_config(config: dict) -> dict:
    """The language model's own section of a multimodal config, else the config."""
    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and _int_field(nested, "hidden_size", "n_embd", "d_model"):
            merged = dict(nested)
            merged.setdefault("model_type", config.get("model_type", ""))
            return merged
    return config


_EXPERT_COUNT_KEYS = (
    "num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts",
    "num_routed_experts",
)


def _is_moe(config: dict) -> bool:
    """Whether a config.json describes a mixture-of-experts model."""
    model_type = str(config.get("model_type") or "").lower()
    if "moe" in model_type or "mixtral" in model_type:
        return True
    return any((_int_field(config, key) or 0) > 1 for key in _EXPERT_COUNT_KEYS)


def params_from_config(config: dict) -> Optional[int]:
    """Parameter count of a decoder-only transformer, counted from its config.

    Embeddings (twice when untied), attention projections, a gated MLP and two
    norms per layer. Biases and rotary buffers are left out; the safetensors
    count is preferred whenever it can be read. A mixture-of-experts config prices
    every expert (and a shared one) in every layer: approximate, since some models
    keep a few dense layers.
    """
    shape = _dims(config)
    if shape is None:
        return None
    hidden, inter, layers, heads, kv, head_dim, vocab, tied = shape
    attn = hidden * heads * head_dim * 2 + hidden * kv * head_dim * 2
    mlp = 3 * hidden * inter
    experts = max((_int_field(config, key) or 0) for key in _EXPERT_COUNT_KEYS)
    if experts > _MAX_EXPERTS:
        return None
    if experts > 1:
        expert_inter = _int_field(config, "moe_intermediate_size") or inter
        shared = _int_field(config, "n_shared_experts", "num_shared_experts") or 0
        if expert_inter > _MAX_DIM or shared > _MAX_EXPERTS:
            return None
        mlp = 3 * hidden * expert_inter * (experts + shared)
    per_layer = attn + mlp + 2 * hidden
    embed = vocab * hidden * (1 if tied else 2)
    return layers * per_layer + embed + hidden


def _dims(config: dict) -> Optional[tuple[int, int, int, int, int, int, int, bool]]:
    hidden = _int_field(config, "hidden_size", "n_embd", "d_model")
    layers = _int_field(config, "num_hidden_layers", "n_layer", "num_layers")
    vocab = _int_field(config, "vocab_size")
    if hidden is None or layers is None or vocab is None:
        return None
    heads = _int_field(config, "num_attention_heads", "n_head") or max(1, hidden // 128)
    kv = _int_field(config, "num_key_value_heads", "num_kv_heads") or heads
    head_dim = _int_field(config, "head_dim") or max(1, hidden // heads)
    inter = _int_field(config, "intermediate_size", "n_inner", "ffn_dim") or 4 * hidden
    tied = config.get("tie_word_embeddings", False) is True
    if layers > _MAX_LAYERS or any(
        value > _MAX_DIM for value in (hidden, inter, heads, kv, head_dim, vocab)
    ):
        return None  # a figure no model has: treated as no config.json
    return hidden, inter, layers, heads, kv, head_dim, vocab, tied


def guessed_arch(params_b: float) -> tuple[int, int, int]:
    """(hidden, layers, intermediate) from a parameter count alone.

    The nearest known size within 20%, else ``params ~ 12 * layers * hidden^2``
    with 32 layers. This was ``profiler.get_model_arch``; it now only serves the
    case with no config.json.
    """
    closest = min(_KNOWN_ARCHS, key=lambda size: abs(size - params_b))
    if abs(closest - params_b) / max(params_b, 0.1) <= 0.2:
        return _KNOWN_ARCHS[closest]
    hidden = int(math.sqrt(params_b * 1e9 / (12 * 32)))
    hidden = max(128, (hidden // 128) * 128)
    layers = max(1, int(params_b * 1e9 / (12 * hidden * hidden)))
    return hidden, layers, hidden * 3


def shape_from_config(
    config: dict, *, params: Optional[int], params_source: str
) -> Optional[ModelShape]:
    text = _text_config(config)
    dims = _dims(text)
    if dims is None:
        return None
    hidden, inter, layers, heads, kv, head_dim, vocab, tied = dims
    if params is None:
        params, params_source = params_from_config(text), "config.json"
    if not params:
        return None
    return ModelShape(
        params=int(params),
        hidden_size=hidden,
        intermediate_size=inter,
        num_layers=layers,
        num_attention_heads=heads,
        num_kv_heads=kv,
        head_dim=head_dim,
        vocab_size=vocab,
        tie_word_embeddings=tied,
        model_type=str(text.get("model_type") or config.get("model_type") or ""),
        params_source=params_source,
        shape_source="config.json",
        moe=_is_moe(text) or _is_moe(config),
    )


def guessed_shape(params_b: float, *, params_source: str = "name") -> ModelShape:
    """A shape from a parameter count alone; ``vocab_size`` stays unknown."""
    hidden, layers, inter = guessed_arch(params_b)
    heads = max(1, hidden // 128)
    return ModelShape(
        params=int(params_b * 1e9),
        hidden_size=hidden,
        intermediate_size=inter,
        num_layers=layers,
        num_attention_heads=heads,
        num_kv_heads=heads,
        head_dim=hidden // heads,
        vocab_size=None,
        tie_word_embeddings=False,
        model_type="",
        params_source=params_source,
        shape_source="guessed",
    )


def read_model_shape(
    base: str,
    *,
    allow_network: bool = True,
    on_hub_request: Optional[Callable[[], None]] = None,
) -> ModelShape:
    """The shape of ``base`` (a local directory or a Hub id); never raises.

    ``allow_network=False`` keeps to the local directory and the cache.
    ``on_hub_request`` is called once, just before the first request to the Hub, so a
    command can say it is about to wait; it is not called when nothing will be sent
    (a local directory, a cached snapshot with its weights, offline mode).
    """
    from soup_cli.utils.gpu import model_size_from_name, param_count_from_local_safetensors

    announced = False

    def reach_hub() -> None:
        nonlocal announced
        if on_hub_request is not None and not announced and allow_network and not hub_offline():
            announced = True
            on_hub_request()

    base = base or ""
    config: Optional[dict] = None
    snapshot: Optional[str] = None
    if os.path.isdir(base):
        config = _read_json_file(os.path.join(base, "config.json"))
        snapshot = base
    elif _REPO_ID.fullmatch(base) and ".." not in base:
        if on_hub_request is not None and not _config_cached(base):
            reach_hub()
        config, snapshot = _hub_lookup(base, allow_network=allow_network)

    params: Optional[int] = None
    params_source = "config.json"
    if snapshot is not None:
        params = param_count_from_local_safetensors(snapshot)
        params_source = "safetensors"
    if params is None and not os.path.isdir(base) and config is not None:
        reach_hub()
        params = _hub_param_count(base, allow_network=allow_network)
        params_source = "hub safetensors metadata"
    if params is None:
        params_source = "config.json"

    if config is not None:
        shape = shape_from_config(config, params=params, params_source=params_source)
        if shape is not None:
            return shape
    if params:
        return guessed_shape(params / 1e9, params_source="safetensors")
    return guessed_shape(model_size_from_name(base), params_source="name")


# peft's default LoRA targets for ``target_modules: auto`` (TRANSFORMERS_MODELS_TO_
# LORA_TARGET_MODULES_MAPPING), for the architectures whose projections this
# estimate can size. Anything else falls back to q_proj and v_proj.
_PEFT_DEFAULT_TARGETS: dict[str, tuple[str, ...]] = {
    "llama": ("q_proj", "v_proj"),
    "mistral": ("q_proj", "v_proj"),
    "mixtral": ("q_proj", "v_proj"),
    "qwen2": ("q_proj", "v_proj"),
    "qwen3": ("q_proj", "v_proj"),
    "gemma": ("q_proj", "v_proj"),
    "gemma2": ("q_proj", "v_proj"),
    "gemma3_text": ("q_proj", "v_proj"),
    "phi3": ("qkv_proj",),
}

_ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_targets(target_modules: Any, model_type: str) -> tuple[str, ...]:
    """The module names LoRA adapts, as the estimate counts them."""
    if target_modules is None or target_modules == "auto":
        return _PEFT_DEFAULT_TARGETS.get(model_type, ("q_proj", "v_proj"))
    if target_modules == "all-linear":
        return _ALL_LINEAR
    if isinstance(target_modules, str):
        return (target_modules,)
    if isinstance(target_modules, (list, tuple)):
        return tuple(str(name) for name in target_modules)
    return ("q_proj", "v_proj")


def module_dims(shape: ModelShape, name: str) -> tuple[int, int]:
    """(in_features, out_features) of one projection in one layer.

    Unknown names are sized hidden -> hidden, which the output does not hide:
    it is what the old flat 1% stood in for, only per module.
    """
    hidden, inter = shape.hidden_size, shape.intermediate_size
    q_out = shape.num_attention_heads * shape.head_dim
    kv_out = shape.num_kv_heads * shape.head_dim
    dims = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "qkv_proj": (hidden, q_out + 2 * kv_out),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "gate_up_proj": (hidden, 2 * inter),
        "down_proj": (inter, hidden),
    }
    return dims.get(name, (hidden, hidden))


def lora_param_count(shape: ModelShape, *, r: int, target_modules: Any) -> int:
    """Trainable LoRA parameters: r * (in + out) per adapted module per layer."""
    if r <= 0:
        return 0
    per_layer = 0
    for name in lora_targets(target_modules, shape.model_type):
        fan_in, fan_out = module_dims(shape, name)
        per_layer += r * (fan_in + fan_out)
    return per_layer * shape.num_layers


__all__ = [
    "HUB_TIMEOUT_S",
    "ModelShape",
    "guessed_arch",
    "guessed_shape",
    "hub_offline",
    "lora_param_count",
    "lora_targets",
    "module_dims",
    "params_from_config",
    "read_model_shape",
    "shape_from_config",
]
