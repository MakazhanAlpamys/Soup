"""Committed native pretraining snapshots and strict weights-only warm starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from soup_cli.data.mlx_pretrain import file_sha256


def tokenizer_identity(tokenizer) -> str:
    """Hash the tokenizer implementation and declared special-token policy, never corpus text."""
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    backend = getattr(hf, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(
            "MLX pretrain requires a serializable fast tokenizer for warm-start identity",
        )
    identity = {"tokenizer": backend.to_str(), "bos": hf.bos_token_id,
                "eos": hf.eos_token_id, "pad": hf.pad_token_id,
                "special": hf.special_tokens_map}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def base_identity(base: str) -> dict:
    """Resolve the already-loaded snapshot and hash its model/config/tokenizer files."""
    path = Path(base)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(base, local_files_only=True))
    names = [p for p in path.iterdir() if p.is_file() and (
        p.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "model.safetensors.index.json"}
        or (p.name.startswith("model") and p.suffix == ".safetensors")
    )]
    if not any(p.suffix == ".safetensors" for p in names):
        raise ValueError("MLX pretrain base identity requires local model safetensors")
    return {p.name: file_sha256(p) for p in sorted(names)}


def read_snapshot(path: Path, expected_contract: dict | None = None) -> dict:
    """A sidecar commits a complete weight file to its exact training contract."""
    try:
        sidecar = path.with_suffix(".json")
        if sidecar.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("snapshot metadata exceeds 4 MiB")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if (meta.get("snapshot_version") != 1 or meta.get("weights_sha256") != file_sha256(path)
                or not isinstance(meta.get("contract"), dict)):
            raise ValueError("incomplete or corrupt snapshot")
        if expected_contract is not None and meta["contract"] != expected_contract:
            differences = sorted(k for k in meta["contract"].keys() | expected_contract.keys()
                                 if meta["contract"].get(k) != expected_contract.get(k))
            raise ValueError(f"incompatible warm-start contract: {', '.join(differences)}")
    except (OSError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError(f"MLX pretrain cannot warm-start {path}: {exc}") from exc
    return meta


def save_snapshot(model, path: Path, metadata: dict) -> None:
    """Publish weights atomically, then their digest/contract; failed writes are not valid."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from soup_cli.trainer.mlx_grpo import write_atomically

    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.eval(weights)
    if not weights or any(not mx.issubdtype(v.dtype, mx.floating)
                          or not mx.all(mx.isfinite(v)).item() for v in weights.values()):
        raise ValueError("MLX pretrain cannot save empty or non-finite adapter tensors")
    write_atomically(path, lambda temp: mx.save_safetensors(temp, weights))
    committed = {**metadata, "snapshot_version": 1, "weights_sha256": file_sha256(path)}
    write_atomically(path.with_suffix(".json"), lambda temp: Path(temp).write_text(
        json.dumps(committed, indent=2, allow_nan=False), encoding="utf-8",
    ))


def load_snapshot(model, path: Path, contract: dict) -> None:
    """Validate identity and every trainable tensor before modifying the live model."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    read_snapshot(path, contract)
    try:
        weights = mx.load(str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"MLX pretrain cannot read warm-start weights: {path}") from exc
    expected = dict(tree_flatten(model.trainable_parameters()))
    if not expected or weights.keys() != expected.keys() or any(
        weights[k].shape != v.shape for k, v in expected.items()
    ):
        raise ValueError("MLX pretrain warm-start tensor keys/shapes must match exactly")
    if any(not mx.issubdtype(v.dtype, mx.floating) or not mx.all(mx.isfinite(v)).item()
           for v in weights.values()):
        raise ValueError("MLX pretrain warm-start tensors must be finite floating point values")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
