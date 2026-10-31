"""Strict, staged native LoRA fusion without importing the PyTorch stack."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _local_base_path(model: str) -> Path:
    """Resolve the files already cached by MLX-LM's model loader, without network I/O."""
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    # Match MLX-LM's load filters on both the minimum and current supported stacks.
    # An unfiltered lookup also requires uncached repository files such as README.md.
    return Path(snapshot_download(model, local_files_only=True, allow_patterns=[
        '*.json', 'model*.safetensors', '*.py', 'tokenizer.model',
        '*.tiktoken', 'tiktoken.model', '*.txt', '*.jsonl', '*.jinja',
    ]))


def _validate_destination(output: Path, adapter: Path, base: str) -> None:
    from soup_cli.utils.paths import is_under, is_under_cwd

    if not is_under_cwd(output):
        raise ValueError("MLX merge --output must stay under cwd")
    for source in (adapter, Path(base)):
        if source.exists() and (is_under(output, source) or is_under(source, output)):
            raise ValueError("MLX merge output must not overlap the adapter or base")
    if output.exists() or output.is_symlink():
        raise ValueError("MLX merge output already exists")


def merge_native_adapter(
    adapter: str, output: str, *, base: str | None = None, dtype: str = "float16",
    dequantize: bool = False, trust_remote_code: bool = False,
) -> Path:
    """Write a complete dense model, publishing the staged directory only on success."""
    from soup_cli.utils.mlx_generation import load_native_model, read_native_adapter

    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("MLX merge dtype must be float16, bfloat16 or float32")
    metadata = read_native_adapter(Path(adapter), base)
    if metadata is None:
        raise ValueError("MLX merge requires a native MLX adapter, not a base-only model")
    destination = Path(output)
    _validate_destination(destination, Path(adapter), metadata["model"])

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.utils import dequantize_model, save
    except ImportError as exc:
        raise ImportError(
            "Native MLX merge requires soup-cli[mlx]; install it on Apple Silicon."
        ) from exc

    model, tokenizer, config = load_native_model(
        adapter, base, trust_remote_code=trust_remote_code, return_config=True,
    )
    # Loading has already populated the cache for a Hub identity. Pass the local
    # snapshot to save(), which otherwise downloads a README via ModelCard.load.
    base_path = _local_base_path(metadata["model"])
    _validate_destination(destination, Path(adapter), str(base_path))
    quantized_types = (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)
    quantized = [
        module for _, module in model.named_modules()
        if isinstance(module, quantized_types)
    ]
    if quantized and not dequantize:
        raise ValueError("Quantized MLX fusion requires explicit --dequantize for dense output")
    if any(module.bits != 4 for module in quantized):
        raise ValueError("MLX merge currently qualifies dense and 4-bit bases only")
    fused = [
        (name, module.fuse(dequantize=True))
        for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    if not fused:
        raise ValueError("MLX merge found no supported linear LoRA layers to fuse")
    model.update_modules(tree_unflatten(fused))
    model = dequantize_model(model)
    if any(isinstance(module, (LoRALinear, *quantized_types))
           for _, module in model.named_modules()):
        raise ValueError("MLX merge did not produce a fully dense model")
    model.set_dtype(getattr(mx, dtype))
    parameters = dict(tree_flatten(model.parameters()))
    mx.eval(parameters)
    if any(not bool(mx.all(mx.isfinite(value)).item()) for value in parameters.values()):
        raise ValueError("MLX merge produced nonfinite weights at the requested dtype")
    config = dict(config)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["torch_dtype"] = dtype
    config["dtype"] = dtype

    _validate_destination(destination, Path(adapter), str(base_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".soup_mlx_merge_", dir=destination.parent) as root:
        staged = Path(root) / "model"
        save(staged, base_path, model, tokenizer, config, donate_model=False)
        _validate_destination(destination, Path(adapter), str(base_path))
        os.rename(staged, destination)
    return destination
