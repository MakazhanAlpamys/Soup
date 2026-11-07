"""Decide whether the saved checkpoint needs llama.cpp's no-MTP conversion."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from soup_cli.utils.safetensors_reader import read_header


def _read_metadata(path: Path, limit: int) -> object:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{path.name} exceeds the {limit}-byte metadata limit")
    return json.loads(raw)


def _checkpoint_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        payload = _read_metadata(index, 32 * 1024 * 1024)
        mapping = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("safetensors index must contain a nonempty weight_map")
        names = set(mapping.values())
        root = os.path.realpath(model_dir)
        shards = []
        for name in sorted(names):
            if not isinstance(name, str) or not name.endswith(".safetensors"):
                raise ValueError("safetensors index contains an invalid shard filename")
            candidate = os.path.realpath(model_dir / name)
            if os.path.commonpath([root, candidate]) != root:
                raise ValueError("safetensors index shard escapes the model directory")
            shards.append(Path(candidate))
        return shards
    single = model_dir / "model.safetensors"
    shards = [single] if single.exists() else sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors checkpoint found")
    return shards


def needs_no_mtp(model_dir: Path, converter: Path) -> bool:
    """Inspect headers only; never edit config or load tensor payloads.

    A declared zero is included: changing the declaration to zero is a known
    unsuccessful workaround for converters that assert a positive MTP count.
    """
    try:
        config = _read_metadata(model_dir / "config.json", 1024 * 1024)
        if not isinstance(config, dict):
            raise ValueError("config.json must contain an object")
        text = config.get("text_config")
        if text is not None and not isinstance(text, dict):
            raise ValueError("text_config must contain an object")
        effective = text if isinstance(text, dict) and "mtp_num_hidden_layers" in text else config
        if "mtp_num_hidden_layers" not in effective:
            return False
        count = effective["mtp_num_hidden_layers"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("mtp_num_hidden_layers must be a nonnegative integer")
        # Validate every shard, even after finding MTP: unreadable metadata must
        # not silently select a conversion mode for an incomplete checkpoint.
        names = set()
        for shard in _checkpoint_shards(model_dir):
            names.update(read_header(str(shard)))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot inspect MTP safetensors metadata: {exc}") from exc

    if any(name.startswith(("mtp.", "model.mtp.")) for name in names):
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(converter), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        supported = result.returncode == 0 and re.search(
            r"(?<![\w-])--no-mtp(?![\w-])",
            result.stdout + "\n" + result.stderr,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        supported = False
    if not supported:
        raise ValueError(
            "MTP is declared but no MTP tensors were saved. The selected converter "
            "does not advertise --no-mtp successfully; use --llama-cpp with a "
            "compatible llama.cpp checkout supporting --no-mtp."
        )
    return True
