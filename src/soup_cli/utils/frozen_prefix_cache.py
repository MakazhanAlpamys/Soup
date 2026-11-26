"""Frozen-prefix activation cache: keys, fingerprints and on-disk storage.

With LoRA on the top layers only, the bottom layers and the embeddings never
change, so their output for a given batch is the same every epoch. The cache
stores that output keyed by everything it depends on, so a later epoch reads it
back instead of running the frozen layers again.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: Directory name, inside the training output directory, that holds the entries.
CACHE_DIRNAME = "_frozen_prefix_cache"


def cache_key(metadata: dict[str, Any]) -> str:
    """Create a deterministic key from cache metadata."""
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_cache_directory(output_dir: str | Path) -> Path:
    """Return the cache directory inside ``output_dir``, refusing any escape.

    The directory is checked after symlinks are resolved, so a planted
    ``_frozen_prefix_cache`` link that points elsewhere is refused instead of
    being written through.
    """
    root = os.path.realpath(os.fspath(output_dir))
    target = os.path.realpath(os.path.join(root, CACHE_DIRNAME))
    try:
        inside = os.path.commonpath([root, target]) == root and target != root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise ValueError(
            "The frozen-prefix cache directory must stay inside the training output "
            f"directory; {CACHE_DIRNAME!r} resolves to {target!r}."
        )
    return Path(target)


def fingerprint_prefix_configuration(decoder: Any, cutoff: int) -> str:
    """Hash decoder configuration and effective RoPE state for persistent keys.

    Configuration hashing is deliberately conservative: any configuration change
    is a cache miss, even if that change is unrelated to the frozen prefix.
    Non-persistent rotary buffers (not included in state_dict) are included.
    """
    import torch

    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("cutoff must be an integer")
    if not 0 < cutoff <= len(decoder.layers):
        raise ValueError("Invalid frozen-prefix cutoff")

    rotary = decoder.rotary_emb
    digest = hashlib.sha256()
    digest.update(b"frozen-prefix-configuration-v1\0")
    digest.update(str(cutoff).encode("ascii"))
    digest.update(decoder.config.to_json_string(use_diff=False).encode("utf-8"))
    digest.update(f"{type(rotary).__module__}.{type(rotary).__qualname__}".encode())

    # Most RoPE variants use non-persistent inv_freq buffers. Iterating over
    # state_dict() alone silently misses them.
    tensor_items = {
        f"buffer.{name}": value
        for name, value in rotary.named_buffers(recurse=True)
    }
    tensor_items.update({
        f"parameter.{name}": value
        for name, value in rotary.named_parameters(recurse=True)
    })
    for name in ("inv_freq", "original_inv_freq", "cos_cached", "sin_cached"):
        value = getattr(rotary, name, None)
        if isinstance(value, torch.Tensor):
            tensor_items.setdefault(f"runtime.{name}", value)

    with torch.no_grad():
        for name, value in sorted(tensor_items.items()):
            if value.is_meta:
                raise ValueError("Cannot fingerprint meta rotary tensors")
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())

    for name in (
        "attention_scaling",
        "rope_type",
        "rope_kwargs",
        "max_seq_len_cached",
        "original_max_seq_len",
    ):
        if not hasattr(rotary, name):
            continue
        value = getattr(rotary, name)
        if isinstance(value, torch.Tensor):
            raise TypeError(f"RoPE {name} must be handled as a tensor")
        try:
            serialized = json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot fingerprint RoPE attribute {name}") from exc
        digest.update(name.encode("utf-8"))
        digest.update(serialized.encode("utf-8"))

    rotary_config = getattr(rotary, "config", None)
    if rotary_config is not None and rotary_config is not decoder.config:
        digest.update(rotary_config.to_json_string(use_diff=False).encode("utf-8"))

    return digest.hexdigest()


def fingerprint_frozen_prefix(decoder: Any, cutoff: int) -> str:
    """SHA-256 fingerprint of the embeddings and the frozen decoder layers."""
    import torch

    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("cutoff must be an integer")

    if not 0 < cutoff <= len(decoder.layers):
        raise ValueError("Invalid frozen-prefix cutoff")

    digest = hashlib.sha256()
    digest.update(b"frozen-prefix-v1")
    digest.update(fingerprint_prefix_configuration(decoder, cutoff).encode("ascii"))

    modules = [
        ("embed_tokens", decoder.embed_tokens),
        *[(f"layers.{index}", decoder.layers[index]) for index in range(cutoff)],
    ]

    with torch.no_grad():
        for prefix, module in modules:
            for name, tensor in sorted(module.state_dict().items()):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Unexpected state entry: {prefix}.{name}")

                if tensor.is_meta:
                    raise ValueError(
                        "Cannot fingerprint meta tensors; "
                        "use a checkpoint/shard fingerprint instead"
                    )

                value = tensor.detach().cpu().contiguous()

                digest.update(f"{prefix}.{name}".encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(memoryview(value.view(torch.uint8).numpy()))

    return digest.hexdigest()


def build_cache_metadata(
    *,
    model_revision: str,
    frozen_prefix_fingerprint: str,
    cutoff: int,
    input_ids: Any,
    attention_mask: Any = None,
    position_ids: Any = None,
    config_fingerprint: str,
) -> dict[str, Any]:
    """Build cache metadata from the actual model inputs."""
    import torch

    if not model_revision or not frozen_prefix_fingerprint:
        raise ValueError("Model identity and fingerprint are required")

    if not config_fingerprint:
        raise ValueError("Configuration fingerprint is required")

    if cutoff < 1:
        raise ValueError("cutoff must be positive")

    def tensor_hash(value: Any) -> str | None:
        if value is None:
            return None

        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")

        tensor = value.detach().cpu().contiguous()

        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())

        return digest.hexdigest()

    return {
        "schema_version": 1,
        "model_revision": model_revision,
        "frozen_prefix_fingerprint": frozen_prefix_fingerprint,
        "config_fingerprint": config_fingerprint,
        "cutoff": cutoff,
        "input_ids_hash": tensor_hash(input_ids),
        "attention_mask_hash": tensor_hash(attention_mask),
        "position_ids_hash": tensor_hash(position_ids),
    }


class FrozenPrefixCache:
    """Store and retrieve frozen-prefix activations on disk."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, metadata: dict[str, Any]) -> Path:
        return self.directory / f"{cache_key(metadata)}.safetensors"

    def save(self, activation: Any, metadata: dict[str, Any]) -> Path:
        """Atomically save a frozen-prefix activation."""
        from safetensors.torch import save_file

        path = self._path(metadata)
        tensor = activation.detach().cpu().contiguous()

        fd, temporary_path = tempfile.mkstemp(
            prefix=".frozen-prefix-",
            suffix=".tmp",
            dir=self.directory,
        )

        try:
            os.close(fd)
            save_file({"hidden_states": tensor}, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

        return path

    def load(self, metadata: dict[str, Any]) -> Any | None:
        """Return the stored activation, or ``None`` when there is no entry."""
        from safetensors.torch import load_file

        path = self._path(metadata)

        if not path.is_file():
            return None

        tensors = load_file(str(path), device="cpu")
        return tensors["hidden_states"]
