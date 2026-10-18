"""Merge a LoRA adapter into a base checkpoint one tensor at a time."""

import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_ADAPTER_PREFIX = "base_model.model."
_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"
_SAFETENSORS_DTYPE = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}
_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_UNSUPPORTED_CONFIG = (
    "use_dora",
    "modules_to_save",
    "trainable_token_indices",
    "rank_pattern",
    "alpha_pattern",
    "loftq_config",
)
_COPIED_SUFFIXES = (".json", ".model", ".txt", ".tiktoken", ".jinja", ".py")

Progress = Optional[Callable[[str], None]]


class LowMemoryMergeError(ValueError):
    """The adapter or checkpoint needs the regular merge path."""


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_adapter(adapter_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, Any]]]:
    from safetensors.torch import load_file

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / _ADAPTER_WEIGHTS
    if not weights_path.is_file():
        raise LowMemoryMergeError(f"{_ADAPTER_WEIGHTS} not found in {adapter_dir}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    for field in _UNSUPPORTED_CONFIG:
        if config.get(field):
            raise LowMemoryMergeError(
                f"adapter_config.json sets {field!r}, which the low-memory merge does not support"
            )

    half_a: Dict[str, Any] = {}
    half_b: Dict[str, Any] = {}
    for key, tensor in load_file(str(weights_path)).items():
        name = key[len(_ADAPTER_PREFIX):] if key.startswith(_ADAPTER_PREFIX) else key
        if name.endswith(_LORA_A):
            half_a[name[: -len(_LORA_A)] + ".weight"] = tensor
        elif name.endswith(_LORA_B):
            half_b[name[: -len(_LORA_B)] + ".weight"] = tensor
        else:
            raise LowMemoryMergeError(
                f"adapter tensor {key!r} is not a plain LoRA A/B weight"
            )
    if half_a.keys() != half_b.keys():
        raise LowMemoryMergeError("adapter has unpaired lora_A / lora_B tensors")
    return config, {name: (half_a[name], half_b[name]) for name in half_a}


def lora_scaling(config: Dict[str, Any]) -> float:
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def merged_tensor(
    weight: Any,
    lora_a: Any,
    lora_b: Any,
    *,
    scaling: float,
    fan_in_fan_out: bool,
    dtype: Any,
) -> Any:
    delta = lora_b.float() @ lora_a.float()
    if fan_in_fan_out:
        delta = delta.T
    delta = delta * scaling
    base = weight.to(dtype)
    if tuple(delta.shape) != tuple(base.shape):
        raise LowMemoryMergeError(
            f"LoRA delta {tuple(delta.shape)} does not match weight {tuple(base.shape)}"
        )
    return (base.float() + delta).to(dtype)


def _weight_files(base_dir: Path) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    index_path = base_dir / _INDEX_NAME
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        return sorted(set(index["weight_map"].values())), index
    if (base_dir / _SINGLE_NAME).is_file():
        return [_SINGLE_NAME], None
    raise LowMemoryMergeError(f"no .safetensors weights found in {base_dir}")


def _write_shard(
    source_path: Path,
    target_path: Path,
    out_dtype: str,
    adapter: Dict[str, Tuple[Any, Any]],
    config: Dict[str, Any],
    used: set,
) -> int:
    import torch
    from safetensors import safe_open

    scaling = lora_scaling(config)
    fan_in_fan_out = bool(config.get("fan_in_fan_out"))
    torch_dtype = _torch_dtype(out_dtype)
    header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    order: List[str] = []
    floats: set = set()
    with safe_open(str(source_path), framework="pt") as source:
        for key in sorted(source.keys()):
            piece = source.get_slice(key)
            stored = piece.get_dtype()
            target = _SAFETENSORS_DTYPE[out_dtype] if stored in _FLOAT_DTYPES else stored
            if target not in _ITEMSIZE:
                raise LowMemoryMergeError(f"unsupported tensor dtype {target!r}")
            shape = list(piece.get_shape())
            size = math.prod(shape) * _ITEMSIZE[target]
            if stored in _FLOAT_DTYPES:
                floats.add(key)
            header[key] = {
                "dtype": target,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
            order.append(key)

        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with open(target_path, "wb") as sink:
            sink.write(struct.pack("<Q", len(encoded)))
            sink.write(encoded)
            for key in order:
                tensor = source.get_tensor(key)
                if key in adapter:
                    lora_a, lora_b = adapter[key]
                    tensor = merged_tensor(
                        tensor,
                        lora_a,
                        lora_b,
                        scaling=scaling,
                        fan_in_fan_out=fan_in_fan_out,
                        dtype=torch_dtype,
                    )
                    used.add(key)
                elif key in floats:
                    tensor = tensor.to(torch_dtype)
                flat = tensor.contiguous().reshape(-1).view(torch.uint8)
                sink.write(memoryview(flat.numpy()))
    return offset


def merge_low_memory(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    dtype: str,
    progress: Progress = None,
) -> int:
    """Write the merged checkpoint to ``output_dir``; return its weight bytes."""
    if dtype not in _SAFETENSORS_DTYPE:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if os.path.realpath(base_dir) == os.path.realpath(output_dir):
        raise ValueError("output directory must differ from the base model directory")

    config, adapter = load_adapter(adapter_dir)
    files, index = _weight_files(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used: set = set()
    total = 0
    for position, name in enumerate(files, start=1):
        if progress is not None:
            progress(f"Merging {name} ({position}/{len(files)})...")
        total += _write_shard(
            base_dir / name, output_dir / name, dtype, adapter, config, used
        )

    missing = sorted(set(adapter) - used)
    if missing:
        raise LowMemoryMergeError(
            f"{len(missing)} adapter weights match no base tensor, e.g. {missing[0]!r}"
        )

    if index is not None:
        index.setdefault("metadata", {})["total_size"] = total
        with open(output_dir / _INDEX_NAME, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

    _copy_support_files(base_dir, output_dir, dtype)
    return total


def _copy_support_files(base_dir: Path, output_dir: Path, dtype: str) -> None:
    for entry in base_dir.iterdir():
        if not entry.is_file() or entry.name == _INDEX_NAME:
            continue
        if not entry.name.endswith(_COPIED_SUFFIXES):
            continue
        shutil.copyfile(entry, output_dir / entry.name)

    config_path = output_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            model_config = json.load(handle)
        for field in ("torch_dtype", "dtype"):
            if field in model_config:
                model_config[field] = dtype
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(model_config, handle, indent=2)
