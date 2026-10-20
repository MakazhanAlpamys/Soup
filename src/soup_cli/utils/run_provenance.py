"""Run-scoped base revisions and portable, explicitly nullable artifact metadata.

No library monkeypatches or new config fields. HF loaders opt in through
``base_revision_kwargs``; unsupported backends retain their existing behavior.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from soup_cli.utils.hf import get_hf_api

PROVENANCE_FILENAME = "soup_run.json"
_MAX_METADATA_BYTES = 64 * 1024
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCES = {
    "hub_commit", "cache_path", "local", "unknown", "unsupported_backend", "offline",
    "resolution_failed",
}


def validate_identity(revision: str | None, source: str | None) -> None:
    """Reject malformed metadata without inventing a commit for unknown runs."""
    if source is not None and (not isinstance(source, str) or source not in _SOURCES):
        raise ValueError("invalid base revision source")
    if revision is not None:
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError("base revision must be a lowercase 40-hex commit or None")
        if source not in {"hub_commit", "cache_path"}:
            raise ValueError("a recorded revision needs its identity source")


def validate_training_hash(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
        raise ValueError("training data hash must be a lowercase SHA-256 or None")


@dataclass
class BaseRevision:
    """Resolve at most once, immediately before the first supported base load.

    A cache directory's name is metadata, not an integrity check on local bytes.
    Offline/probe failure preserves the prior loader flow and records unknown.
    """

    base: str
    binding_supported: bool = True
    revision: str | None = field(default=None, init=False)
    source: str = field(default="unknown", init=False)
    _pending: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if os.path.isdir(self.base):
            path = Path(os.path.abspath(self.base))
            self.source = "local"
            if path.parent.name == "snapshots" and _REVISION.fullmatch(path.name):
                self.revision, self.source = path.name, "cache_path"
            self._pending = False
        elif not self.binding_supported:
            self.source, self._pending = "unsupported_backend", False

    def load_kwargs(self, model_id: str) -> dict[str, str]:
        if not isinstance(model_id, str) or model_id != self.base:
            return {}
        if self._pending:
            self._pending = False
            if any(os.environ.get(name, "").upper() in {"1", "TRUE", "YES", "ON"}
                   for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
                self.source = "offline"
            else:
                try:
                    # Bound the optional metadata request, not model downloads.
                    revision = get_hf_api().model_info(self.base, timeout=5).sha
                    if isinstance(revision, str) and _REVISION.fullmatch(revision):
                        self.revision, self.source = revision, "hub_commit"
                    else:
                        self.source = "resolution_failed"
                except Exception:
                    # Do not echo provider exceptions: they can contain tokens/paths.
                    self.source = "resolution_failed"
        if self.source == "hub_commit" and self.revision is not None:
            return {"revision": self.revision}
        return {}

    def metadata(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1, "run_id": run_id, "base_model": self.base,
            "base_revision": self.revision, "base_revision_source": self.source,
            # The train command fills this from the recorded startup source hash.
            "training_data_sha256": None,
        }


_ACTIVE_BASE: ContextVar[BaseRevision | None] = ContextVar("soup_base_revision", default=None)


@contextmanager
def base_revision_scope(identity: BaseRevision) -> Iterator[None]:
    """Keep one execution identity across setup/training; restore nested callers."""
    token = _ACTIVE_BASE.set(identity)
    try:
        yield
    finally:
        _ACTIVE_BASE.reset(token)


def base_revision_kwargs(model_id: str) -> dict[str, str]:
    """Pin only this run's base, never unrelated teachers/rewards/scorers."""
    identity = _ACTIVE_BASE.get()
    return identity.load_kwargs(model_id) if identity is not None else {}


def _validate_metadata(data: Any) -> dict[str, Any]:
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise ValueError("unsupported run metadata")
    if not isinstance(data.get("run_id"), str) or not data["run_id"]:
        raise ValueError("run metadata needs a run ID")
    if not isinstance(data.get("base_model"), str) or not data["base_model"]:
        raise ValueError("run metadata needs the original base ID")
    validate_identity(data.get("base_revision"), data.get("base_revision_source"))
    validate_training_hash(data.get("training_data_sha256"))
    return data


def write_run_provenance(output: str | Path, metadata: dict[str, Any]) -> None:
    """Atomically save the final artifact's run identity without a DB dependency."""
    _validate_metadata(metadata)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("run metadata is too large")
    fd, temporary = tempfile.mkstemp(prefix=".soup_run_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, directory / PROVENANCE_FILENAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_run_provenance(artifact: str | Path) -> dict[str, Any]:
    """Unknown/malformed/legacy metadata must not introduce new load refusals."""
    try:
        with (Path(artifact) / PROVENANCE_FILENAME).open("rb") as handle:
            payload = handle.read(_MAX_METADATA_BYTES + 1)
        if len(payload) > _MAX_METADATA_BYTES:
            return {}
        return _validate_metadata(json.loads(payload))
    except (OSError, ValueError, TypeError, RecursionError):
        return {}


def recorded_base_kwargs(artifact: str | None, base: str) -> dict[str, str]:
    """Reuse a copied artifact's Hub commit only for its matching base ID.

    An explicit base override, a local source, or a cache-path hint is not a
    bound Hub identity. Leave those existing flows alone instead of guessing.
    """
    if artifact is None or os.path.isdir(base):
        return {}
    data = read_run_provenance(artifact)
    if (data.get("base_model") == base and data.get("base_revision_source") == "hub_commit"
            and data.get("base_revision") is not None):
        return {"revision": data["base_revision"]}
    return {}


def recorded_adapter_base(artifact: str) -> str | None:
    """Recognize a local adapter whose saved base agrees with its run metadata.

    Transformers otherwise auto-loads the base from adapter_config.json, which
    can resolve a moving Hub HEAD. Preserve legacy/explicit-revision behavior
    when the two local records disagree instead of rewriting an override.
    """
    data = read_run_provenance(artifact)
    base = data.get("base_model")
    if not isinstance(base, str) or not recorded_base_kwargs(artifact, base):
        return None
    try:
        with (Path(artifact) / "adapter_config.json").open("rb") as handle:
            payload = handle.read(_MAX_METADATA_BYTES + 1)
        if len(payload) > _MAX_METADATA_BYTES:
            return None
        config = json.loads(payload)
        if (isinstance(config, dict) and config.get("base_model_name_or_path") == base
                and config.get("revision") in (None, data["base_revision"])):
            return base
    except (OSError, ValueError, TypeError, RecursionError):
        pass
    return None
