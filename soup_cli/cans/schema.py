"""Pydantic schemas for the ``.can`` artifact format (v0.26.0 Part E)."""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CAN_FORMAT_VERSION = 2  # v0.33.0: deploy_targets + env capture (additive over v1)
SUPPORTED_CAN_FORMAT_VERSIONS = (1, 2)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,127}$")
_HF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]{0,127}$")
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,95}/[A-Za-z0-9][A-Za-z0-9_\-.]{0,95}$")


class DeployTarget(BaseModel):
    """One declarative deploy target embedded in a can manifest (v2+).

    ``kind`` selects the deploy backend:
      - ``ollama``: model name to deploy via ``soup deploy ollama``
      - ``gguf``: relative path inside the can to a GGUF artifact
      - ``vllm``: model id to serve via ``soup serve --backend vllm``
    """

    kind: Literal["ollama", "gguf", "vllm"] = Field(description="Deploy backend")
    name: Optional[str] = Field(
        default=None, max_length=128,
        description="Model / artifact name (kind-specific)",
    )
    path: Optional[str] = Field(
        default=None, max_length=512,
        description="Relative path within the can for kind=gguf",
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("deploy name must not contain null bytes or newlines")
        return value

    @field_validator("path")
    @classmethod
    def _safe_relpath(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("deploy path contains null byte")
        if value.startswith("/") or value.startswith("\\"):
            raise ValueError(f"deploy path '{value}' must be relative")
        if ".." in value.split("/") or ".." in value.split("\\"):
            raise ValueError(f"deploy path '{value}' may not contain '..'")
        return value


class DataRef(BaseModel):
    """How to fetch the training data after unpacking a can.

    ``kind`` is one of:
      - ``url``: HTTPS URL pointing at a JSONL file
      - ``hf``: HuggingFace dataset id (``org/dataset``)
      - ``local``: relative path — user must supply it locally
    """

    kind: Literal["url", "hf", "local"] = Field(
        description="Data source type",
    )
    url: Optional[str] = Field(default=None, description="HTTPS URL")
    hf_dataset: Optional[str] = Field(default=None, description="HF dataset id")
    local_path: Optional[str] = Field(default=None, description="Relative local path")

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value.startswith("https://"):
            raise ValueError(
                f"data_ref.url must be https:// (got: {value}) - "
                "plain http is forbidden for remote fetches"
            )
        return value

    @field_validator("hf_dataset")
    @classmethod
    def _valid_hf_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not _HF_NAME_RE.match(value):
            raise ValueError(
                f"hf_dataset '{value}' is invalid - "
                "use 'org/name' with alphanumeric + _-./"
            )
        return value


class Manifest(BaseModel):
    """Top-level metadata for a ``.can`` file."""

    can_format_version: int = Field(description="Format version integer")
    name: str = Field(description="Recipe name")
    author: str = Field(description="Author handle", max_length=128)
    created_at: str = Field(description="ISO-8601 timestamp or YYYY-MM-DD")
    base_hash: str = Field(description="SHA-256 of the config (from registry)")
    description: Optional[str] = Field(default=None, max_length=4096)
    tags: list[str] = Field(default_factory=list)
    deploy_targets: list[DeployTarget] = Field(
        default_factory=list,
        description="Optional declarative deploy targets (v2+)",
    )

    @field_validator("can_format_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value not in SUPPORTED_CAN_FORMAT_VERSIONS:
            raise ValueError(
                f"unknown can_format_version {value}; this build of Soup "
                f"supports versions {SUPPORTED_CAN_FORMAT_VERSIONS}. "
                "Upgrade Soup or re-pack the can with a supported format."
            )
        return value

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(
                f"can name '{value}' is invalid - "
                "alphanumeric + _-. only, must start with alphanumeric"
            )
        return value

    @field_validator("author")
    @classmethod
    def _clean_author(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("author must not contain null bytes or newlines")
        return value

    @field_validator("created_at")
    @classmethod
    def _parseable_created_at(cls, value: str) -> str:
        # Accept YYYY-MM-DD or any ISO-8601 datetime parseable by fromisoformat
        from datetime import datetime as _dt

        try:
            _dt.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"created_at '{value}' is not valid ISO-8601"
            ) from exc
        return value
