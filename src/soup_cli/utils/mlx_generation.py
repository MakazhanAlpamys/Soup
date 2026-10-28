"""Shared native text consumers. MLX imports stay behind explicit backend selection."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


def validate_native_options(
    *, temperature: float, max_tokens: int, device: str | None, hub: str,
    task: str = "text", batch_size: int = 1, cuda_graphs: bool = False,
) -> None:
    """Refuse options whose Transformers semantics the native path cannot provide."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("MLX temperature must be finite and nonnegative")
    if max_tokens < 1 or max_tokens > 16384:
        raise ValueError("MLX max_tokens must be between 1 and 16384")
    if device is not None:
        raise ValueError("MLX selects its native device; omit --device")
    if hub != "hf":
        raise ValueError("MLX generation supports --hub hf only")
    if task != "text":
        raise ValueError("MLX generation requires --task text")
    if batch_size != 1:
        raise ValueError("MLX generation requires --batch-size 1")
    if cuda_graphs:
        raise ValueError("MLX generation does not support --cuda-graphs")


def _same_base(left: str, right: str) -> bool:
    if Path(left).exists() or Path(right).exists():
        return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))
    return left == right


#: Bounds on native adapter metadata; a file past them is refused.
_MAX_ADAPTER_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_ADAPTER_RANK = 4096
_MAX_ADAPTER_LAYERS = 2**16
_MAX_ADAPTER_KEYS = 1024
_MAX_ADAPTER_KEY_LEN = 256


def _finite(value: Any) -> bool:
    """``value`` is an int or float that is finite (a huge int is not: it has no float)."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_native_adapter(path: Path, base: str | None = None) -> dict | None:
    """Validate metadata before loading optional dependencies or allocating a model."""
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        if (path / "adapters.safetensors").exists():
            raise ValueError("Native MLX adapter weights require adapter_config.json")
        if base is not None:
            raise ValueError("--base is only valid for a native MLX adapter")
        return None
    try:
        if config_path.stat().st_size > _MAX_ADAPTER_CONFIG_BYTES:
            raise ValueError("adapter_config.json is larger than 4 MiB")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid native MLX metadata in adapter_config.json") from exc
    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise ValueError("Expected native MLX metadata with a model identity, not a PEFT adapter")
    if not config["model"].strip() or "base_model_name_or_path" in config:
        raise ValueError("Ambiguous or empty native MLX metadata model identity")
    if base is not None and not _same_base(base, config["model"]):
        raise ValueError("--base does not match the native MLX metadata model identity")
    if config.get("fine_tune_type") != "lora" or config.get("use_dora", False):
        raise ValueError("Native MLX generation currently supports ordinary linear LoRA only")
    params = config.get("lora_parameters")
    if not isinstance(params, dict):
        raise ValueError("Native MLX metadata requires lora_parameters")
    rank = params.get("rank")
    layers = config.get("num_layers")
    if type(rank) is not int or rank < 1 or type(layers) is not int or layers < 1:
        raise ValueError("Native MLX metadata rank and num_layers must be positive integers")
    if rank > _MAX_ADAPTER_RANK:
        raise ValueError(f"Native MLX metadata rank must be at most {_MAX_ADAPTER_RANK}")
    if layers > _MAX_ADAPTER_LAYERS:
        raise ValueError(f"Native MLX metadata num_layers must be at most {_MAX_ADAPTER_LAYERS}")
    for key in ("scale", "dropout"):
        if not _finite(params.get(key)):
            raise ValueError(f"Native MLX metadata {key} must be finite")
    if params["scale"] <= 0 or not 0 <= params["dropout"] < 1:
        raise ValueError("Native MLX metadata requires positive scale and dropout in [0, 1)")
    keys = params.get("keys")
    if isinstance(keys, list) and (
        len(keys) > _MAX_ADAPTER_KEYS
        or any(isinstance(key, str) and len(key) > _MAX_ADAPTER_KEY_LEN for key in keys)
    ):
        raise ValueError(
            f"Native MLX metadata keys must be at most {_MAX_ADAPTER_KEYS} names of "
            f"{_MAX_ADAPTER_KEY_LEN} characters"
        )
    if not isinstance(keys, list) or not keys or any(
        not isinstance(key, str) or not key or key == "auto" for key in keys
    ) or len(set(keys)) != len(keys):
        raise ValueError("Native MLX metadata requires unique resolved target keys")
    if not (path / "adapters.safetensors").is_file():
        raise ValueError("Native MLX adapter is missing adapters.safetensors")
    return config


def reject_native_adapter(model: str) -> None:
    """Avoid sending an unmistakable native artifact through PEFT by accident."""
    path = Path(model) / "adapter_config.json"
    if path.is_file():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return
        if isinstance(config, dict) and ("model" in config or "lora_parameters" in config):
            raise ValueError("Native MLX adapter detected; select --backend mlx")


def load_native_model(
    model_path: str, base_model: str | None = None, *, trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a base and strictly apply every saved ordinary linear LoRA tensor."""
    path = Path(model_path)
    config = read_native_adapter(path, base_model)
    from rich.console import Console

    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    base = config["model"] if config else model_path
    trust_remote_code = resolve_trust_remote_code(
        base, requested=trust_remote_code, console=Console(),
        requires_remote_code=bool(model_requires_trust_remote_code(base)),
    )
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise ImportError(
            "Native MLX generation requires soup-cli[mlx]; install it on Apple Silicon."
        ) from exc

    # 0.31.3 accepts tokenizer trust but predates custom architecture loading;
    # newer releases additionally require an explicit architecture trust flag.
    trust = ({"trust_remote_code": trust_remote_code}
             if "trust_remote_code" in inspect.signature(load).parameters else {})
    model, tokenizer, base_config = load(
        base,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        return_config=True, **trust,
    )
    # MLX-LM 0.31.3 replaces tokenizer EOS with model EOS; retain both as newer
    # versions do. Use the declared ID, never a guessed control-token string.
    if tokenizer.eos_token_id is not None:
        tokenizer.add_eos_token(str(tokenizer.eos_token_id))
    if base_config.get("vision_config") or base_config.get("audio_config") or any(
        "Classification" in architecture or "Embedding" in architecture
        for architecture in base_config.get("architectures", [])
    ):
        raise ValueError("Native MLX generation supports text causal language models only")
    if config:
        layers = getattr(model, "layers", [])
        if config["num_layers"] > len(layers):
            raise ValueError("Native MLX metadata num_layers exceeds the base model's layers")
        keys = set(config["lora_parameters"]["keys"])
        matches = [
            (name, module)
            for parent in [*layers[-config["num_layers"]:], model]
            for name, module in parent.named_modules() if name in keys
        ]
        if keys != {name for name, _ in matches} or any(
            not isinstance(module, (nn.Linear, nn.QuantizedLinear))
            or (isinstance(module, nn.QuantizedLinear) and module.bits != 4)
            for _, module in matches
        ):
            raise ValueError("Native MLX targets must all match dense or 4-bit linear modules")
        model.freeze()
        linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
        expected = dict(tree_flatten(model.trainable_parameters()))
        try:
            weights = mx.load(str(path / "adapters.safetensors"))
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "Cannot read native MLX adapter weights in adapters.safetensors"
            ) from exc
        if not expected or weights.keys() != expected.keys() or any(
            weights[name].shape != expected[name].shape for name in weights.keys() & expected
        ):
            raise ValueError("Native MLX adapter tensor keys/shapes must exactly match the base")
        if any(
            not mx.issubdtype(weight.dtype, mx.floating)
            or not bool(mx.all(mx.isfinite(weight)).item())
            for weight in weights.values()
        ):
            raise ValueError("Native MLX adapter tensors must be finite floating point values")
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
    model.eval()
    return model, tokenizer


def generate_native(
    model: Any, tokenizer: Any, messages: list[dict], *, max_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, int]:
    """Return completion text and sampled token count (including a terminating EOS)."""
    validate_native_options(temperature=temperature, max_tokens=max_tokens, device=None, hub="hf")
    from contextlib import closing

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.utils.vllm import _render_chat_prompt

    # Pass IDs, never a rendered string which the engine could tokenize a second time.
    rendered, templated = _render_chat_prompt(messages, tokenizer, fallback_on_error=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=not templated)
    if not prompt:
        raise ValueError("MLX generation requires a nonempty tokenized prompt")
    text = []
    count = 0
    with closing(stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=0.9 if temperature > 0 else 1.0),
    )) as responses:
        for response in responses:
            text.append(response.text)
            count = response.generation_tokens
    return "".join(text).strip(), count
