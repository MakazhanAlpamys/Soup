"""Shared native text consumers. MLX imports stay behind explicit backend selection."""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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


def _resolve_adapter_base(stored: str, base: str | None) -> tuple[str, str, bool]:
    """Keep the selected spelling alongside the resolved legacy local identity."""
    path = Path(stored)
    legacy = not path.is_absolute() and (
        stored.startswith(("./", "../", ".\\", "..\\")) or path.is_dir()
    )
    if legacy and base is not None and Path(base).is_absolute():
        if not Path(base).is_dir():
            raise ValueError(
                f"Native MLX legacy relative base {stored!r}: "
                "--base must name an existing absolute local directory"
            )
        return base, os.path.realpath(base), True
    if base is not None and not _same_base(base, stored):
        raise ValueError("--base does not match the native MLX metadata model identity")
    if legacy:
        if not path.is_dir():
            raise ValueError(
                f"Native MLX legacy relative base {stored!r} is not a directory in cwd; "
                "pass --base with the absolute path to the original local base directory"
            )
        return stored, os.path.realpath(stored), True
    return stored, stored, False


#: Bounds on native adapter metadata; a file past them is refused.
_MAX_ADAPTER_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_ADAPTER_RANK = 4096
_MAX_ADAPTER_LAYERS = 2**16
_MAX_ADAPTER_KEYS = 1024
_MAX_ADAPTER_KEY_LEN = 256


class NativePromptError(ValueError):
    """A tokenized prompt is empty or exceeds the caller's input limit."""


def _finite(value: Any) -> bool:
    """``value`` is an int or float that is finite (a huge int is not: it has no float)."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_adapter_rope_scaling(config: dict[str, Any]) -> None:
    """Validate RoPE scaling metadata on a native adapter."""
    if "rope_scaling" not in config:
        return
    rope_cfg = config["rope_scaling"]
    if not isinstance(rope_cfg, dict):
        raise ValueError("Native MLX metadata rope_scaling must be a dictionary")
    rope_type = rope_cfg.get("type") or rope_cfg.get("rope_type")
    if rope_type != "linear":
        raise ValueError(
            f"Native MLX generation currently supports linear RoPE scaling only, "
            f"got {rope_type!r}"
        )
    factor = rope_cfg.get("factor")
    if not _finite(factor) or factor <= 1.0:
        raise ValueError(
            "Native MLX metadata rope_scaling factor must be a finite number > 1.0"
        )
    orig_max = rope_cfg.get("original_max_position_embeddings")
    if (
        orig_max is None
        or isinstance(orig_max, bool)
        or not isinstance(orig_max, int)
        or orig_max <= 0
    ):
        raise ValueError(
            "Native MLX metadata rope_scaling original_max_position_embeddings "
            "must be a positive integer"
        )


def read_native_adapter(
    path: Path, base: str | None = None, *, announce: bool = True,
    validate_base: Callable[[str], None] | None = None,
) -> dict | None:
    """Validate metadata and the selected base spelling before announcing or loading it."""
    if (path / "added_tokens.safetensors").exists():
        raise ValueError(
            "Native MLX generation does not currently support artifacts "
            "with expanded vocabularies (#3089)"
        )
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
    selected_base, resolved_base, legacy = _resolve_adapter_base(config["model"], base)
    if config.get("vocab_expansion"):
        raise ValueError(
            "Native MLX generation does not currently support artifacts "
            "with expanded vocabularies (#3089)"
        )
    if config.get("fine_tune_type") != "lora" or config.get("use_dora", False):
        raise ValueError("Native MLX generation currently supports ordinary linear LoRA only")
    _validate_adapter_rope_scaling(config)
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
    if validate_base is not None:
        validate_base(selected_base)
    if legacy:
        if announce:
            from rich.console import Console

            from soup_cli.utils.terminal import for_terminal

            Console().print(for_terminal(
                f"Native MLX legacy relative base {config['model']!r} "
                f"resolved to {resolved_base!r}; the original training cwd is unknown."
            ))
        config["model"] = resolved_base
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
    require_dense: bool = False, return_config: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, dict]:
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

    model_config_arg: dict[str, Any] = {}
    if config and "rope_scaling" in config:
        from soup_cli.utils.mlx_rope import _read_checkpoint_config

        raw_base_config, _ = _read_checkpoint_config(base)
        orig_len = raw_base_config.get("max_position_embeddings")
        if (
            orig_len is None
            or isinstance(orig_len, bool)
            or not isinstance(orig_len, int)
            or orig_len <= 0
        ):
            raise ValueError(
                f"Base checkpoint {base!r} original context length ('max_position_embeddings') "
                f"is missing or invalid (got {orig_len!r}); "
                "positive integer required for MLX RoPE scaling."
            )
        expected_orig = config["rope_scaling"]["original_max_position_embeddings"]
        if orig_len != expected_orig:
            raise ValueError(
                f"Base model max_position_embeddings ({orig_len!r}) does not match "
                f"adapter original_max_position_embeddings ({expected_orig!r})"
            )
        existing = raw_base_config.get("rope_scaling") or raw_base_config.get("rope_parameters")
        if isinstance(existing, dict):
            if any(isinstance(v, dict) for v in existing.values()):
                raise ValueError(
                    "Base checkpoint has per-layer or nested RoPE configuration; "
                    "not supported on MLX"
                )
            rope_type = existing.get("rope_type") or existing.get("type")
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"Base checkpoint already carries RoPE scaling ({existing!r}); "
                    "cannot apply adapter RoPE scaling."
                )
        model_config_arg["model_config"] = {
            "rope_scaling": config["rope_scaling"],
            "max_position_embeddings": config.get(
                "max_seq_length", int(round(orig_len * config["rope_scaling"]["factor"]))
            ),
        }

    model, tokenizer, base_config = load(
        base,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        return_config=True, **trust, **model_config_arg,
    )
    if require_dense and (
        base_config.get("quantization") or base_config.get("quantization_config")
        or any(isinstance(module, (nn.QuantizedLinear,
                                  getattr(nn, "QuantizedEmbedding", nn.QuantizedLinear)))
               for _, module in model.named_modules())
    ):
        raise ValueError("Native MLX serving currently supports dense checkpoints only")
    # MLX-LM 0.31.3 replaces tokenizer EOS with model EOS; retain both as newer
    # versions do. Use the declared ID, never a guessed control-token string.
    if tokenizer.eos_token_id is not None:
        tokenizer.add_eos_token(str(tokenizer.eos_token_id))
    if base_config.get("vision_config") or base_config.get("audio_config") or any(
        "Classification" in architecture or "Embedding" in architecture
        for architecture in base_config.get("architectures", [])
    ):
        raise ValueError("Native MLX generation supports text causal language models only")
    if config and "rope_scaling" in config:
        quantized_types = [
            getattr(nn, "QuantizedLinear", ()),
            getattr(nn, "QuantizedEmbedding", ()),
        ]
        try:
            from mlx_lm.models.switch_layers import QuantizedSwitchLinear

            quantized_types.append(QuantizedSwitchLinear)
        except ImportError:
            pass
        quantized_tuple = tuple(t for t in quantized_types if isinstance(t, type))
        for name, mod in model.named_modules():
            if isinstance(mod, quantized_tuple):
                raise ValueError(
                    "Native MLX generation with RoPE scaling does not support "
                    "quantized base models; use an unquantized base."
                )
        expected_scale = 1.0 / config["rope_scaling"]["factor"]
        rope_scales = [
            float(module.scale)
            for _, module in model.named_modules()
            if isinstance(module, nn.RoPE)
        ]
        if not rope_scales:
            raise ValueError(
                f"Base checkpoint {base!r} does not contain rotary embedding (nn.RoPE) modules; "
                "cannot apply native MLX linear RoPE context extension."
            )
        if any(abs(scale - expected_scale) > 1e-5 for scale in rope_scales):
            raise ValueError(
                f"Base checkpoint {base!r} does not apply native MLX RoPE scaling "
                f"(expected scale {expected_scale:.6g}, found {rope_scales}); "
                "model architecture may not support native linear RoPE scaling."
            )
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
    return (model, tokenizer, base_config) if return_config else (model, tokenizer)


@dataclass(frozen=True)
class NativeGenerationChunk:
    """One detokenized increment, with actual engine token counts."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None


def stream_native(
    model: Any, tokenizer: Any, messages: list[dict], *, max_tokens: int = 256,
    temperature: float = 0.7, top_p: float = 0.9, stop: tuple[str, ...] = (),
    max_prompt_tokens: int | None = None, seed: int | None = None,
) -> Iterator[NativeGenerationChunk]:
    """Stream native text once, retaining only a possible stop-string prefix.

    Closing the iterator closes the engine and its request-local KV cache. Counts
    include a sampled EOS or a token containing a stop string, even when hidden.
    """
    validate_native_options(temperature=temperature, max_tokens=max_tokens, device=None, hub="hf")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError(
            "MLX top_p must be finite and in (0, 1]; zero disables native nucleus filtering"
        )
    if len(stop) > 4 or any(
        not isinstance(item, str) or not 1 <= len(item) <= 256 for item in stop
    ):
        raise ValueError("MLX stop accepts at most four nonempty strings of at most 256 characters")
    from contextlib import closing

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.utils.vllm import _render_chat_prompt

    # Pass IDs, never a rendered string which the engine could tokenize a second time.
    rendered, templated = _render_chat_prompt(messages, tokenizer, fallback_on_error=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=not templated)
    if not prompt:
        raise NativePromptError("MLX generation requires a nonempty tokenized prompt")
    if max_prompt_tokens is not None and len(prompt) > max_prompt_tokens:
        raise NativePromptError("MLX prompt exceeds the input token limit")
    if seed is not None:
        import mlx.core as mx

        mx.random.seed(seed)
    pending = ""
    count = 0
    with closing(stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 1.0),
    )) as responses:
        for response in responses:
            pending += response.text
            count = response.generation_tokens
            positions = [pending.index(item) for item in stop if item in pending]
            if positions:
                yield NativeGenerationChunk(pending[:min(positions)], len(prompt), count, "stop")
                return
            reason = getattr(response, "finish_reason", None)
            if reason in {"stop", "length"}:
                yield NativeGenerationChunk(pending, len(prompt), count, reason)
                return
            # Never expose a prefix which a later token could complete as a stop.
            retained = max((
                length for item in stop for length in range(1, len(item))
                if pending.endswith(item[:length])
            ), default=0)
            emitted = pending[:-retained] if retained else pending
            pending = pending[-retained:] if retained else ""
            yield NativeGenerationChunk(emitted, len(prompt), count)
    yield NativeGenerationChunk(
        pending, len(prompt), count, "length" if count >= max_tokens else "stop",
    )


def generate_native(
    model: Any, tokenizer: Any, messages: list[dict], *, max_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, int]:
    """Return completion text and sampled token count (including a terminating EOS)."""
    from contextlib import closing

    text = []
    count = 0
    with closing(stream_native(
        model, tokenizer, messages, max_tokens=max_tokens, temperature=temperature,
    )) as chunks:
        for chunk in chunks:
            text.append(chunk.text)
            count = chunk.completion_tokens
    return "".join(text).strip(), count
