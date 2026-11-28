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
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

#: Directory name, inside the training output directory, that holds the entries.
CACHE_DIRNAME = "_frozen_prefix_cache"

#: ``ignore_patterns`` for any upload of a training output directory: the cache
#: is scratch data, large, and not part of the model.
UPLOAD_IGNORE_PATTERNS = [CACHE_DIRNAME, f"{CACHE_DIRNAME}/*"]

#: Writing stops when it would leave less free space than the larger of these.
MIN_FREE_BYTES = 2 * 1024**3
MIN_FREE_FRACTION = 0.05


def free_space_reserve(total_bytes: int) -> int:
    """Free space that must stay on the volume: max(2 GiB, 5 % of the volume)."""
    return max(MIN_FREE_BYTES, int(total_bytes * MIN_FREE_FRACTION))


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
    """Store and retrieve frozen-prefix activations on disk.

    Writing is best effort: when the volume would drop below the free-space
    reserve, or a write fails, the cache stops writing, tells ``notify`` once,
    and keeps serving the entries it already has. Training is never stopped by
    the cache running out of room.
    """

    def __init__(
        self,
        directory: str | Path,
        notify: Callable[[str], None] | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.notify = notify
        self.writing = True
        self.stop_reason: str | None = None

    def _path(self, metadata: dict[str, Any]) -> Path:
        return self.directory / f"{cache_key(metadata)}.safetensors"

    def _stop_writing(self, reason: str) -> None:
        self.writing = False
        self.stop_reason = reason
        if self.notify is not None:
            self.notify(
                "Frozen-prefix cache stopped writing: "
                f"{reason}. Existing entries are still read; new batches are recomputed."
            )

    def save(self, activation: Any, metadata: dict[str, Any]) -> Path | None:
        """Atomically save a frozen-prefix activation, or return ``None`` if not writing."""
        from safetensors.torch import save_file

        if not self.writing:
            return None

        path = self._path(metadata)
        tensor = activation.detach().cpu().contiguous()

        try:
            usage = shutil.disk_usage(self.directory)
            needed = tensor.numel() * tensor.element_size()
            reserve = free_space_reserve(usage.total)
            if usage.free - needed < reserve:
                self._stop_writing(
                    f"{usage.free / 1e9:.1f} GB free would drop below the "
                    f"{reserve / 1e9:.1f} GB reserve (the larger of 2 GiB and 5% of the volume)"
                )
                return None
        except OSError as exc:
            self._stop_writing(f"free space could not be read ({type(exc).__name__}: {exc})")
            return None

        temporary_path = None
        try:
            fd, temporary_path = tempfile.mkstemp(
                prefix=".frozen-prefix-",
                suffix=".tmp",
                dir=self.directory,
            )
            os.close(fd)
            save_file({"hidden_states": tensor}, temporary_path)
            os.replace(temporary_path, path)
        except OSError as exc:
            self._stop_writing(f"a write failed ({type(exc).__name__}: {exc})")
            return None
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
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
