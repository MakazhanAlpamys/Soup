"""v0.44.0 Part D — `soup fetch` example/config catalog.

Maps short names (e.g. `llama-3.1-8b-lora`) to ready-to-edit YAML payloads.
The catalog is a frozen registry; payload bodies live in
`templates/fetch_examples/*.yaml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

# Closed-allowlist of fetch namespaces.
_VALID_NAMESPACES = frozenset({"examples", "configs", "deepspeed_configs"})

_MAX_NAME_LEN = 96


@dataclass(frozen=True)
class FetchEntry:
    """One catalog entry."""

    name: str
    namespace: str
    filename: str
    description: str


def _build_catalog() -> Mapping[str, FetchEntry]:
    """Construct the static catalog. Keep small and curated."""
    raw = [
        FetchEntry(
            name="llama-3.1-8b-lora",
            namespace="examples",
            filename="llama-3.1-8b-lora.yaml",
            description="Llama 3.1 8B SFT LoRA recipe (4-bit + r=16).",
        ),
        FetchEntry(
            name="qwen2.5-7b-dpo",
            namespace="examples",
            filename="qwen2.5-7b-dpo.yaml",
            description="Qwen 2.5 7B DPO preference recipe.",
        ),
        FetchEntry(
            name="zero3-cpu-offload",
            namespace="deepspeed_configs",
            filename="zero3-cpu-offload.json",
            description="DeepSpeed ZeRO-3 with CPU offload (24-32GB GPUs).",
        ),
    ]
    return MappingProxyType({entry.name: entry for entry in raw})


CATALOG: Mapping[str, FetchEntry] = _build_catalog()


def list_entries(namespace: Optional[str] = None) -> Mapping[str, FetchEntry]:
    """Return entries, optionally filtered by namespace."""
    if namespace is None:
        return CATALOG
    if namespace not in _VALID_NAMESPACES:
        raise ValueError(
            f"namespace must be one of {sorted(_VALID_NAMESPACES)}; "
            f"got {namespace!r}"
        )
    return MappingProxyType(
        {
            name: entry
            for name, entry in CATALOG.items()
            if entry.namespace == namespace
        }
    )


def get_entry(name: str) -> Optional[FetchEntry]:
    """Look up a single entry by short name."""
    if not isinstance(name, str):
        return None
    if not name or "\x00" in name or len(name) > _MAX_NAME_LEN:
        return None
    return CATALOG.get(name)


def fetch_examples_dir() -> str:
    """Filesystem path to the bundled fetch-example directory.

    Uses `os.path.realpath` (project policy) so symlinked installs resolve
    to the real package root, not the symlink target's parent.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "templates",
        "fetch_examples",
    )
