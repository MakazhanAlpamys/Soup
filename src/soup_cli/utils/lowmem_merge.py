"""Merge a LoRA adapter into a base checkpoint one tensor at a time."""

import json
import math
import os
import re
import shutil
import struct
import tempfile
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
    "lora_bias",
    "modules_to_save",
    "trainable_token_indices",
    "rank_pattern",
    "alpha_pattern",
    "loftq_config",
    "layer_replication",
    "target_parameters",
    "alora_invocation_tokens",
    "use_qalora",
)
_COPIED_SUFFIXES = (".json", ".model", ".txt", ".tiktoken", ".jinja")
_SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)

Progress = Optional[Callable[[str], None]]
Finalize = Optional[Callable[[Path], None]]


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
    if config.get("peft_type", "LORA") != "LORA":
        raise LowMemoryMergeError(
            f"adapter_config.json has peft_type {config['peft_type']!r}; only LORA is supported"
        )
    if config.get("bias", "none") != "none":
        raise LowMemoryMergeError(
            f"adapter_config.json sets bias={config['bias']!r}, "
            "which the low-memory merge does not support"
        )
    for field in _UNSUPPORTED_CONFIG:
        if config.get(field):
            raise LowMemoryMergeError(
                f"adapter_config.json sets {field!r}, which the low-memory merge does not support"
            )

    lora_scaling(config)
    _flag(config, "fan_in_fan_out")

    half_a: Dict[str, Any] = {}
    half_b: Dict[str, Any] = {}
    for key, tensor in load_file(str(weights_path)).items():
        name = key[len(_ADAPTER_PREFIX):] if key.startswith(_ADAPTER_PREFIX) else key
        if name.endswith(_LORA_A):
            half_a[name[: -len(_LORA_A)] + ".weight"] = tensor
        elif name.endswith(_LORA_B):
            half_b[name[: -len(_LORA_B)] + ".weight"] = tensor
        elif "lm_head" in key or "embed" in key:
            raise LowMemoryMergeError(
                f"adapter tensor {key!r}: adapters on embeddings or lm_head are not supported"
            )
        else:
            raise LowMemoryMergeError(
                f"adapter tensor {key!r} is not a plain LoRA A/B weight"
            )
    if half_a.keys() != half_b.keys():
        raise LowMemoryMergeError("adapter has unpaired lora_A / lora_B tensors")
    return config, {name: (half_a[name], half_b[name]) for name in half_a}


def _flag(config: Dict[str, Any], field: str) -> bool:
    value = config.get(field, False)
    if not isinstance(value, bool):
        raise LowMemoryMergeError(
            f"adapter_config.json field {field!r} must be true or false, got {value!r}"
        )
    return value


def lora_scaling(config: Dict[str, Any]) -> float:
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise LowMemoryMergeError(
            f"adapter_config.json field 'r' must be a positive integer, got {rank!r}"
        )
    alpha = config.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise LowMemoryMergeError(
            f"adapter_config.json field 'lora_alpha' must be a finite number, got {alpha!r}"
        )
    if _flag(config, "use_rslora"):
        return float(alpha) / math.sqrt(rank)
    return float(alpha) / rank


def _check_delta_shape(
    lora_a_shape: Any, lora_b_shape: Any, weight_shape: Any, fan_in_fan_out: bool, key: str = ""
) -> None:
    """Compare shapes before any matmul, so a hostile adapter cannot force a huge delta."""
    label = f" for {key!r}" if key else ""
    if len(lora_a_shape) != 2 or len(lora_b_shape) != 2:
        raise LowMemoryMergeError(f"LoRA A/B tensors must be 2-D{label}")
    if lora_b_shape[1] != lora_a_shape[0]:
        raise LowMemoryMergeError(
            f"LoRA rank mismatch{label}: lora_B {tuple(lora_b_shape)}, lora_A {tuple(lora_a_shape)}"
        )
    delta = (lora_b_shape[0], lora_a_shape[1])
    if fan_in_fan_out:
        delta = delta[::-1]
    if delta != tuple(weight_shape):
        raise LowMemoryMergeError(
            f"LoRA delta {delta} does not match weight {tuple(weight_shape)}{label}"
        )


def merged_tensor(
    weight: Any,
    lora_a: Any,
    lora_b: Any,
    *,
    scaling: float,
    fan_in_fan_out: bool,
    dtype: Any,
) -> Any:
    _check_delta_shape(tuple(lora_a.shape), tuple(lora_b.shape), tuple(weight.shape),
                       fan_in_fan_out)
    delta = lora_b.float() @ lora_a.float()
    if fan_in_fan_out:
        delta = delta.T
    delta = delta * scaling
    base = weight.to(dtype)
    return (base.float() + delta).to(dtype)


def _inside(path: Any, root: str) -> bool:
    real_root = os.path.realpath(root)
    try:
        return os.path.commonpath([os.path.realpath(path), real_root]) == real_root
    except ValueError:
        return False


def _source_roots(base_dir: Path) -> List[str]:
    """Directories a base file may resolve into: the model dir and its HF cache blobs."""
    roots = [os.path.realpath(base_dir)]
    absolute = Path(os.path.abspath(base_dir))
    if absolute.parent.name == "snapshots":
        roots.append(os.path.realpath(absolute.parent.parent / "blobs"))
    return roots


def _checked_source(base_dir: Path, name: str, roots: List[str]) -> Path:
    path = base_dir / name
    if not any(_inside(path, root) for root in roots):
        raise LowMemoryMergeError(f"{name!r} resolves outside the base model directory")
    if not path.is_file():
        raise LowMemoryMergeError(f"{name!r} not found in {base_dir}")
    return path


def _checked_shard_name(name: Any) -> str:
    if (
        not isinstance(name, str)
        or _SHARD_NAME.fullmatch(name) is None
        or name.split(".")[0].upper() in _WINDOWS_DEVICES
    ):
        raise LowMemoryMergeError(
            f"{_INDEX_NAME} lists {name!r}, which is not a plain .safetensors file name"
        )
    return name


def _weight_files(
    base_dir: Path, roots: List[str]
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    index_path = base_dir / _INDEX_NAME
    if index_path.is_file():
        _checked_source(base_dir, _INDEX_NAME, roots)
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise LowMemoryMergeError(f"{_INDEX_NAME} has no weight_map")
        names = sorted({_checked_shard_name(name) for name in weight_map.values()})
        if len({name.lower() for name in names}) != len(names):
            raise LowMemoryMergeError(f"{_INDEX_NAME} lists shard names that differ only by case")
        return names, index
    if (base_dir / _SINGLE_NAME).is_file():
        return [_SINGLE_NAME], None
    raise LowMemoryMergeError(f"no .safetensors weights found in {base_dir}")


def _preflight(
    sources: Dict[str, Path],
    adapter: Dict[str, Tuple[Any, Any]],
    config: Dict[str, Any],
) -> None:
    """Check the adapter against the base tensors before anything is written."""
    from safetensors import safe_open

    fan_in_fan_out = _flag(config, "fan_in_fan_out")
    found: Dict[str, Tuple[str, List[int]]] = {}
    for path in sources.values():
        with safe_open(str(path), framework="pt") as source:
            for key in source.keys():
                piece = source.get_slice(key)
                found[key] = (piece.get_dtype(), list(piece.get_shape()))

    missing = sorted(set(adapter) - set(found))
    if missing:
        raise LowMemoryMergeError(
            f"{len(missing)} adapter weights match no base tensor, e.g. {missing[0]!r}"
        )
    for key, (lora_a, lora_b) in adapter.items():
        stored, shape = found[key]
        if stored not in _FLOAT_DTYPES:
            raise LowMemoryMergeError(
                f"base tensor {key!r} is stored as {stored}, not a float type "
                "(quantized weights cannot take a LoRA delta here)"
            )
        _check_delta_shape(
            tuple(lora_a.shape), tuple(lora_b.shape), tuple(shape), fan_in_fan_out, key
        )


def _write_shard(
    source_path: Path,
    target_path: Path,
    out_dtype: str,
    adapter: Dict[str, Tuple[Any, Any]],
    config: Dict[str, Any],
) -> int:
    import torch
    from safetensors import safe_open

    scaling = lora_scaling(config)
    fan_in_fan_out = _flag(config, "fan_in_fan_out")
    torch_dtype = _torch_dtype(out_dtype)
    header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    order: List[str] = []
    sizes: Dict[str, int] = {}
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
            elif key in adapter:
                raise LowMemoryMergeError(
                    f"base tensor {key!r} is stored as {stored}, not a float type"
                )
            header[key] = {
                "dtype": target,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
            sizes[key] = size
            order.append(key)

        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with open(target_path, "xb") as sink:
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
                elif key in floats:
                    tensor = tensor.to(torch_dtype)
                flat = tensor.contiguous().reshape(-1).view(torch.uint8)
                if flat.numel() != sizes[key]:
                    raise LowMemoryMergeError(
                        f"tensor {key!r} is {flat.numel()} bytes, its header declares {sizes[key]}"
                    )
                sink.write(memoryview(flat.numpy()))
    return offset


def _reject_overlap(directories: Dict[str, Path]) -> None:
    real = {label: os.path.realpath(path) for label, path in directories.items()}
    labels = list(real)
    for position, first in enumerate(labels):
        for second in labels[position + 1:]:
            try:
                shared = os.path.commonpath([real[first], real[second]])
            except ValueError:
                continue
            if shared in (real[first], real[second]):
                raise ValueError(f"{first} and {second} directories must not overlap")


def merge_low_memory(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    dtype: str,
    progress: Progress = None,
    finalize: Finalize = None,
    trust_remote_code: bool = False,
) -> int:
    """Write the merged checkpoint to ``output_dir``; return its weight bytes.

    Everything is written to a fresh staging directory next to ``output_dir`` and
    moved into place only when complete. ``finalize`` runs on the staging
    directory first, so a failure there leaves no output either. An existing,
    non-empty ``output_dir`` is never overwritten. ``.py`` files are copied from the
    base only when ``trust_remote_code`` is set.
    """
    if dtype not in _SAFETENSORS_DTYPE:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if os.path.realpath(base_dir) == os.path.realpath(output_dir):
        raise ValueError("output directory must differ from the base model directory")
    _reject_overlap({"base": base_dir, "adapter": adapter_dir, "output": output_dir})
    if os.path.islink(output_dir):
        raise ValueError("output directory must not be a symbolic link")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory {output_dir} already exists and is not empty")

    config, adapter = load_adapter(adapter_dir)
    roots = _source_roots(base_dir)
    files, index = _weight_files(base_dir, roots)
    sources = {name: _checked_source(base_dir, name, roots) for name in files}
    support = _support_files(base_dir, roots, trust_remote_code)
    _preflight(sources, adapter, config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        total = 0
        for position, name in enumerate(files, start=1):
            if progress is not None:
                progress(f"Merging {name} ({position}/{len(files)})...")
            total += _write_shard(sources[name], staging / name, dtype, adapter, config)

        if index is not None:
            index.setdefault("metadata", {})["total_size"] = total
            with open(staging / _INDEX_NAME, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)

        _copy_support_files(support, staging, dtype)
        if finalize is not None:
            finalize(staging)

        mask = os.umask(0)
        os.umask(mask)
        os.chmod(staging, 0o777 & ~mask)
        if output_dir.exists():
            output_dir.rmdir()
        os.rename(staging, output_dir)
        published = True
        return total
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _support_files(base_dir: Path, roots: List[str], trust_remote_code: bool) -> List[Path]:
    suffixes = _COPIED_SUFFIXES + ((".py",) if trust_remote_code else ())
    found = []
    for entry in sorted(base_dir.iterdir()):
        if entry.name == _INDEX_NAME or not entry.name.endswith(suffixes):
            continue
        if not entry.is_file():
            continue
        if not any(_inside(entry, root) for root in roots):
            raise LowMemoryMergeError(f"{entry.name!r} resolves outside the base model directory")
        found.append(entry)
    return found


def _copy_support_files(files: List[Path], staging: Path, dtype: str) -> None:
    for entry in files:
        shutil.copyfile(entry, staging / entry.name)

    config_path = staging / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as handle:
            model_config = json.load(handle)
        for field in ("torch_dtype", "dtype"):
            if field in model_config:
                model_config[field] = dtype
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(model_config, handle, indent=2)
