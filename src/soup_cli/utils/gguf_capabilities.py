"""Read model metadata and probe the selected text GGUF converter's capabilities."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def model_text_architecture(model: Path) -> str:
    """Match the converter's root architecture and optional text-config override."""
    path = model / "config.json"
    try:
        with path.open("rb") as handle:
            raw = handle.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("config exceeds the 1 MiB metadata limit")
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")

        def architecture(value: object, field: str) -> str:
            if (not isinstance(value, list) or not value or not isinstance(value[0], str)
                    or not value[0].strip()):
                raise ValueError(f"{field} must be a nonempty list with a string architecture")
            return value[0]

        result = architecture(config.get("architectures"), "architectures")
        text = config.get("text_config", {})
        if not isinstance(text, dict):
            raise ValueError("text_config must be an object")
        if text.get("architectures") is not None:
            result = architecture(text["architectures"], "text_config.architectures")
        return result
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid GGUF model metadata in {path}: {exc}") from exc


def pinned_text_architectures(tag: str) -> frozenset[str]:
    """Load generated capabilities; a pin change requires regenerating the data."""
    path = Path(__file__).with_name("llama_cpp_capabilities.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["tag"] != tag:
        raise ValueError(
            f"llama.cpp pin {tag} does not match generated capability data {data['tag']}; "
            "regenerate it with scripts/gen_llama_cpp_capabilities.py"
        )
    return frozenset(data["architectures"]["TEXT"])


def probe_text_architectures(converter: Path) -> frozenset[str] | None:
    """Return advertised TEXT models, or None when the probe is inconclusive.

    b5270 writes grouped capabilities to stderr. A nonzero exit, timeout, missing
    interpreter/dependency or an unrecognized/empty listing proves no exclusion.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(converter), "--print-supported-models"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names: set[str] = set()
    for output in (result.stderr, result.stdout):
        kind = None
        for line in output.splitlines():
            heading = re.search(r"(?:^|:)\s*(TEXT|VISION) models:\s*$", line)
            if heading:
                kind = heading[1]
                continue
            entry = re.search(r"(?:^|:)\s*-\s+([A-Za-z0-9_]+)\s*$", line)
            if kind == "TEXT" and entry:
                names.add(entry[1])
    return frozenset(names) if names else None
