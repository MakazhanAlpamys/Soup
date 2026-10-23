"""Resolved ordinary linear LoRA patterns and strict native reconstruction."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


def _recipe(lora: Any) -> dict:
    return {
        "r": lora.r, "alpha": lora.alpha, "dropout": lora.dropout,
        "use_rslora": lora.use_rslora,
        "rank_pattern": lora.rank_pattern or {},
        "alpha_pattern": lora.alpha_pattern or {},
    }


def resolve_module_parameters(names: list[str], recipe: dict) -> dict:
    """PEFT 0.20+ first-match suffix regex, independently for rank and alpha."""
    from soup_cli.config.schema import LoraConfig

    # Reuse bounded/safe pattern validation, including artifacts read from disk.
    cfg = LoraConfig.model_validate(recipe)
    for key in ("r", "alpha"):
        if type(recipe.get(key)) is not int or recipe[key] <= 0:
            raise ValueError("MLX patterned LoRA requires positive integer rank/alpha defaults")
    if (type(recipe.get("dropout")) not in (int, float) or
            not math.isfinite(recipe["dropout"]) or not 0 <= recipe["dropout"] < 1 or
            type(recipe.get("use_rslora")) is not bool):
        raise ValueError("MLX patterned LoRA requires finite dropout in [0, 1) and bool rsLoRA")
    compiled = {
        field: [(key, re.compile(rf"(.*\.)?({key})$"), value)
                for key, value in (getattr(cfg, field) or {}).items()]
        for field in ("rank_pattern", "alpha_pattern")
    }
    for field, patterns in compiled.items():
        unmatched = [key for key, pattern, _ in patterns
                     if not any(pattern.match(name) for name in names)]
        if unmatched:
            raise ValueError(f"MLX {field} matches no selected module: {unmatched}")
    resolved = {}
    for name in names:
        rank = next((value for _, pattern, value in compiled["rank_pattern"]
                     if pattern.match(name)), cfg.r)
        alpha = next((value for _, pattern, value in compiled["alpha_pattern"]
                      if pattern.match(name)), cfg.alpha)
        scale = alpha / (math.sqrt(rank) if cfg.use_rslora else rank)
        resolved[name] = {"rank": rank, "alpha": alpha, "scale": scale,
                          "dropout": cfg.dropout}
    return resolved


def selected_linear_modules(model: Any, keys: list[str]) -> dict:
    """Keep upstream's layer-relative/exact full-path target selection."""
    import mlx.nn as nn

    selected = {}
    found = set()
    for parent in [*model.layers, model]:
        for name, module in parent.named_modules():
            if name in keys:
                found.add(name)
                selected[id(module)] = module
    modules = {name: module for name, module in model.named_modules() if id(module) in selected}
    if found != set(keys) or not modules or any(
        not isinstance(module, (nn.Linear, nn.QuantizedLinear)) or
        (isinstance(module, nn.QuantizedLinear) and module.bits != 4)
        for module in modules.values()
    ):
        raise ValueError(
            "MLX patterns require matched dense or 4-bit linear targets; "
            f"unmatched targets: {sorted(set(keys) - found)}"
        )
    return modules


def build_pattern_manifest(model: Any, lora: Any, keys: list[str]) -> dict:
    """Resolve before mutating the base; store both recipe and effective modules."""
    modules = selected_linear_modules(model, keys)
    recipe = _recipe(lora)
    return {"version": 1, "recipe": recipe, "target_keys": keys,
            "modules": resolve_module_parameters(list(modules), recipe)}


def validate_pattern_manifest(manifest: Any) -> dict:
    """Validate all dimensions/scales and recipe drift without importing MLX."""
    if (not isinstance(manifest, dict) or
            set(manifest) != {"version", "recipe", "target_keys", "modules"} or
            type(manifest["version"]) is not int or manifest["version"] != 1):
        raise ValueError("Unsupported Soup MLX per-module manifest version/fields")
    keys, modules, recipe = (manifest[key] for key in ("target_keys", "modules", "recipe"))
    if (not isinstance(keys, list) or not keys or
            any(not isinstance(key, str) or not key or key == "auto" for key in keys) or
            len(set(keys)) != len(keys) or not isinstance(modules, dict) or not modules or
            any(not isinstance(name, str) or not name for name in modules) or
            not isinstance(recipe, dict) or set(recipe) != {
                "r", "alpha", "dropout", "use_rslora", "rank_pattern", "alpha_pattern",
            }):
        raise ValueError("Invalid Soup MLX per-module targets/recipe")
    expected = resolve_module_parameters(list(modules), recipe)
    for name, params in modules.items():
        if (not isinstance(params, dict) or set(params) != {"rank", "alpha", "scale", "dropout"}
                or type(params["rank"]) is not int or type(params["alpha"]) is not int
                or any(type(params[key]) not in (int, float) or
                       not math.isfinite(params[key]) for key in ("scale", "dropout"))
                or params != expected[name]):
            raise ValueError(f"Soup MLX per-module metadata drift for {name}")
    return manifest


def apply_pattern_manifest(model: Any, manifest: dict) -> None:
    """Construct the exact module set before loading any adapter tensors."""
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    validate_pattern_manifest(manifest)
    modules = selected_linear_modules(model, manifest["target_keys"])
    if modules.keys() != manifest["modules"].keys():
        raise ValueError("Soup MLX per-module targets do not match the base model")
    model.freeze()
    converted = [
        (name, LoRALinear.from_base(
            module, r=manifest["modules"][name]["rank"],
            scale=manifest["modules"][name]["scale"],
            dropout=manifest["modules"][name]["dropout"],
        )) for name, module in modules.items()
    ]
    model.update_modules(tree_unflatten(converted))


def load_pattern_checkpoint(model: Any, checkpoint: str, manifest: dict) -> None:
    """Warm-start only an identical resolved recipe and complete finite tensors."""
    import json

    import mlx.core as mx
    from mlx.utils import tree_flatten

    try:
        metadata = json.loads((Path(checkpoint).parent / "adapter_config.json").read_text())
        saved = validate_pattern_manifest(metadata.get("soup_lora"))
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError("MLX patterns warm-start requires valid per-module metadata") from exc
    if saved != manifest:
        raise ValueError("MLX patterns warm-start metadata is incompatible with this recipe")
    weights = mx.load(checkpoint)
    expected = dict(tree_flatten(model.trainable_parameters()))
    if (weights.keys() != expected.keys() or any(
            weights[name].shape != expected[name].shape or
            not mx.issubdtype(weights[name].dtype, mx.floating) or
            not bool(mx.all(mx.isfinite(weights[name])).item()) for name in weights)):
        raise ValueError("MLX patterns warm-start tensor keys/shapes/values are incompatible")
    model.load_weights(list(weights.items()), strict=False)


def check_pattern_output(output: Path, metadata: dict) -> None:
    """Never relabel existing checkpoint weights with a different recipe."""
    import json

    if not any(output.glob("*adapters.safetensors")):
        return
    try:
        previous = json.loads((output / "adapter_config.json").read_text())
        if (previous.get("soup_lora") == metadata["soup_lora"] and
                previous.get("model") == metadata["model"]):
            return
    except (OSError, ValueError, AttributeError):
        pass
    raise ValueError("MLX patterns require a fresh output directory for a different adapter recipe")


def write_pattern_metadata(output: Path, metadata: dict) -> None:
    """Publish complete metadata atomically alongside periodic/final weights."""
    import json
    import os
    import tempfile

    # Training output is allowed outside cwd; the general report writer is not.
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=output)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, indent=2))
        os.replace(temporary, output / "adapter_config.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
