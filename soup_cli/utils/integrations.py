"""v0.45.0 Part C — External integrations catalog (schema-only).

Closed allowlist of integration descriptors so a future ``soup deploy
<target>`` can know about LM Studio, ComfyUI, stable-diffusion.cpp,
Open WebUI, etc. without each command growing its own ad-hoc list. Live
launch wiring lands in v0.45.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

_INTEGRATION_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,31}$")
_MAX_DESCRIPTION = 256


@dataclass(frozen=True)
class IntegrationSpec:
    """One external integration entry."""

    name: str
    description: str
    target_artifacts: Tuple[str, ...]


def _make(
    name: str,
    description: str,
    target_artifacts: Tuple[str, ...],
) -> IntegrationSpec:
    if not _INTEGRATION_NAME_RE.match(name):
        raise ValueError(
            "integration name must be kebab-case ([a-z0-9][a-z0-9-]{0,31})"
        )
    if not isinstance(description, str) or "\x00" in description:
        raise ValueError("description must be a NUL-free string")
    if len(description) > _MAX_DESCRIPTION:
        raise ValueError(f"description exceeds {_MAX_DESCRIPTION} chars")
    if not isinstance(target_artifacts, tuple) or not target_artifacts:
        raise ValueError("target_artifacts must be a non-empty tuple")
    for artifact in target_artifacts:
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("target_artifacts entries must be non-empty str")
    return IntegrationSpec(
        name=name, description=description, target_artifacts=target_artifacts
    )


_BUILTIN: Mapping[str, IntegrationSpec] = MappingProxyType(
    {
        "lm-studio": _make(
            "lm-studio",
            "Push merged GGUF to LM Studio via the lms CLI",
            ("gguf",),
        ),
        "comfyui": _make(
            "comfyui",
            "Export node-graph compatible models for ComfyUI",
            ("safetensors", "gguf"),
        ),
        "stable-diffusion-cpp": _make(
            "stable-diffusion-cpp",
            "Image-output model export for stable-diffusion.cpp",
            ("safetensors",),
        ),
        "open-webui": _make(
            "open-webui",
            "Auto-register the served model with Open WebUI",
            ("served-endpoint",),
        ),
        "ollama": _make(
            "ollama",
            "Deploy GGUF model to local Ollama instance",
            ("gguf",),
        ),
        "tei": _make(
            "tei",
            "HuggingFace Text-Embeddings-Inference deployment",
            ("safetensors",),
        ),
        "pgvector": _make(
            "pgvector",
            "Postgres pgvector schema + connection pattern",
            ("safetensors",),
        ),
        "faiss": _make(
            "faiss",
            "FAISS index pattern for embedding-model output",
            ("safetensors",),
        ),
        "weaviate": _make(
            "weaviate",
            "Weaviate vector-DB connection pattern",
            ("safetensors",),
        ),
        "sentence-transformers": _make(
            "sentence-transformers",
            "Sentence-Transformers compatible embedding export",
            ("safetensors",),
        ),
        "claude-code": _make(
            "claude-code",
            "Claude Code client SDK integration example",
            ("served-endpoint",),
        ),
        "cursor": _make(
            "cursor",
            "Cursor IDE custom-model connection guide",
            ("served-endpoint",),
        ),
        "continue": _make(
            "continue",
            "Continue VS Code extension custom-model guide",
            ("served-endpoint",),
        ),
        "cline": _make(
            "cline",
            "Cline (autonomous IDE agent) connection guide",
            ("served-endpoint",),
        ),
        "sillytavern": _make(
            "sillytavern",
            "SillyTavern chat front-end connection guide",
            ("served-endpoint",),
        ),
    }
)


def list_integrations() -> Mapping[str, IntegrationSpec]:
    """Return an immutable view of the integration registry."""
    return _BUILTIN


def get_integration(name: str) -> IntegrationSpec:
    """Return the integration entry for ``name`` or raise ``KeyError``."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    canonical = name.strip().lower()
    if not canonical or "\x00" in canonical:
        raise ValueError("name must be a non-empty NUL-free string")
    if canonical not in _BUILTIN:
        raise KeyError(canonical)
    return _BUILTIN[canonical]


def has_integration(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return name.strip().lower() in _BUILTIN


__all__ = [
    "IntegrationSpec",
    "list_integrations",
    "get_integration",
    "has_integration",
]
