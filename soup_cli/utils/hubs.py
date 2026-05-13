"""v0.51.0 Part E — Alternative model hubs (ModelScope / Modelers).

Schema-only support for selecting a non-HF model hub for downloads + pushes.
Each hub gets a closed allowlist + SSRF-hardened endpoint validator that
mirrors the v0.29.0 ``HF_ENDPOINT`` policy in ``utils/hf.py``:

* scheme allowlist (http/https only)
* null-byte rejection
* ``0.0.0.0`` explicitly rejected
* plain HTTP only permitted for loopback hosts
  (``localhost`` / ``127.0.0.1`` / ``::1``)
* private / link-local / cloud-metadata IPs (RFC1918, 169.254.x) rejected
  for plain HTTP.

Live download / upload wiring is deferred to v0.51.1 — this module ships the
schema lock-in (``hub`` Literal on ``TrainingConfig``) plus the validators
that the live wiring will call.
"""
from __future__ import annotations

import ipaddress
import os
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse

# Closed allowlist of supported hubs. Wrapped in MappingProxyType so callers
# cannot mutate the registry at runtime (matches v0.36.0 _REGISTRY policy).
SUPPORTED_HUBS: frozenset[str] = frozenset({"hf", "modelscope", "modelers"})

_HUB_DEFAULT_ENDPOINTS: Mapping[str, str] = MappingProxyType({
    "hf": "https://huggingface.co",
    "modelscope": "https://modelscope.cn",
    "modelers": "https://modelers.cn",
})

# Per-hub env var that overrides the default endpoint (mirrors HF_ENDPOINT).
_HUB_ENDPOINT_ENV: Mapping[str, str] = MappingProxyType({
    "hf": "HF_ENDPOINT",
    "modelscope": "MODELSCOPE_ENDPOINT",
    "modelers": "MODELERS_ENDPOINT",
})

# Per-hub pip-install hint, surfaced when the live downloader complains.
_HUB_PACKAGE: Mapping[str, str] = MappingProxyType({
    "hf": "huggingface-hub",
    "modelscope": "modelscope",
    "modelers": "openmind-hub",
})

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

_MAX_HUB_NAME_LEN: int = 32


def validate_hub_name(name: str) -> str:
    """Validate ``name`` against ``SUPPORTED_HUBS`` and return canonical form.

    Mirrors v0.41.0 ``validate_optimizer_name`` policy:
    rejects non-string / bool / empty / null-byte / oversize / unknown, and
    lower-cases the input for deterministic lookup.
    """
    if isinstance(name, bool):
        raise TypeError(f"hub name must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(f"hub name must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("hub name must be non-empty")
    if "\x00" in name:
        raise ValueError("hub name must not contain null bytes")
    if len(name) > _MAX_HUB_NAME_LEN:
        raise ValueError(
            f"hub name too long (max {_MAX_HUB_NAME_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_HUBS:
        supported = ", ".join(sorted(SUPPORTED_HUBS))
        raise ValueError(
            f"hub {name!r} not supported. Supported: {supported}"
        )
    return canonical


def required_hub_package(hub: str) -> str | None:
    """Return the pip-installable package name for ``hub``, or ``None``.

    Non-string / unknown returns ``None`` (no advisory available).
    """
    if not isinstance(hub, str):
        return None
    return _HUB_PACKAGE.get(hub.lower())


def default_endpoint(hub: str) -> str:
    """Return the canonical default endpoint URL for ``hub``.

    Raises ``ValueError`` if ``hub`` is not in :data:`SUPPORTED_HUBS`.
    """
    canonical = validate_hub_name(hub)
    return _HUB_DEFAULT_ENDPOINTS[canonical]


def endpoint_env_var(hub: str) -> str:
    """Return the env-var name that overrides the default endpoint."""
    canonical = validate_hub_name(hub)
    return _HUB_ENDPOINT_ENV[canonical]


def _is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` is a private / link-local / loopback IP.

    DNS resolution intentionally not performed (matches v0.29.0 hf.py policy).
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_link_local or addr.is_loopback


def validate_hub_endpoint(endpoint: str, *, hub: str | None = None) -> str:
    """SSRF-hardened endpoint validator. Returns the stripped endpoint.

    Mirrors ``utils/hf.resolve_endpoint`` exactly so all three hubs share the
    same security posture. Raises ``ValueError`` on bad input.

    Args:
        endpoint: candidate URL.
        hub: optional hub name, threaded into the error messages.
    """
    label = (hub or "hub_endpoint").strip() or "hub_endpoint"

    if isinstance(endpoint, bool):
        raise TypeError(f"{label} must not be bool, got {endpoint!r}")
    if not isinstance(endpoint, str):
        raise TypeError(
            f"{label} must be str, got {type(endpoint).__name__}"
        )
    if not endpoint:
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in endpoint:
        raise ValueError(f"{label} must not contain null bytes")
    # Defence-in-depth: control characters (CR/LF/etc.) inside a URL would
    # be a CRLF-injection hazard if the URL ever flowed into a raw HTTP
    # client. Reject them here even though urlparse silently strips them.
    if any(ord(c) < 0x20 for c in endpoint):
        raise ValueError(f"{label} must not contain control characters")

    stripped = endpoint.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{label} must use http/https scheme, got: {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"{label} is missing a host")

    host = parsed.hostname or ""
    if host == "0.0.0.0":
        raise ValueError(
            f"{label} 0.0.0.0 is ambiguous; use 127.0.0.1 or localhost"
        )
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        if _is_private_or_link_local(host):
            raise ValueError(
                f"{label} plain HTTP is only allowed for loopback "
                f"(localhost / 127.0.0.1 / ::1); private/link-local hosts "
                f"require HTTPS"
            )
        raise ValueError(
            f"{label} for remote hosts must use HTTPS "
            f"(localhost HTTP allowed)"
        )
    return stripped


def resolve_endpoint(hub: str, *, env: Mapping[str, str] | None = None) -> str:
    """Return the active endpoint for ``hub`` after env-override + validation.

    Looks up the per-hub env var (e.g. ``MODELSCOPE_ENDPOINT``) — if set, runs
    it through :func:`validate_hub_endpoint` and returns the stripped URL.
    Otherwise returns :func:`default_endpoint`.

    The default endpoints are baked-in HTTPS URLs so they don't need
    re-validation on every call.
    """
    canonical = validate_hub_name(hub)
    source = env if env is not None else os.environ
    raw = source.get(_HUB_ENDPOINT_ENV[canonical])
    if not raw:
        return _HUB_DEFAULT_ENDPOINTS[canonical]
    return validate_hub_endpoint(raw, hub=canonical)


def is_hf(hub: str) -> bool:
    """Convenience: True iff ``hub`` canonicalises to ``'hf'``.

    Rejects ``bool`` explicitly (matches v0.30.0 ``Candidate`` /
    v0.34.0 ``estimate_run_cost_usd`` policy) so a stray ``True`` cannot
    silently pretend to be a hub name.
    """
    if isinstance(hub, bool):
        return False
    if not isinstance(hub, str):
        return False
    return hub.lower() == "hf"


# v0.53.8 #130 — Live download / upload dispatcher.
# Each backend lazy-imports its SDK so a missing optional dep only surfaces
# when the user actually selects that hub. Mirrors v0.51.0 stub-then-live
# pattern: schema (TrainingConfig.hub Literal) shipped v0.51.0; live wiring
# ships now.

_REPO_ID_MAX = 200


def _validate_repo_id_shape(repo_id: str) -> str:
    """Cheap shape-only repo-id validator shared by all hub adapters.

    Does NOT mirror the full v0.29.0 HF ``validate_repo_id`` regex (which is
    HF-specific). Each hub's SDK applies its own canonicalisation; we just
    reject obviously dangerous shapes (null bytes, leading slash, ``..``,
    oversize) before forwarding.
    """
    if isinstance(repo_id, bool):
        raise TypeError(f"repo_id must not be bool, got {repo_id!r}")
    if not isinstance(repo_id, str):
        raise TypeError(
            f"repo_id must be str, got {type(repo_id).__name__}"
        )
    if not repo_id:
        raise ValueError("repo_id must be non-empty")
    if "\x00" in repo_id:
        raise ValueError("repo_id must not contain null bytes")
    if len(repo_id) > _REPO_ID_MAX:
        raise ValueError(
            f"repo_id too long (max {_REPO_ID_MAX} chars)"
        )
    if repo_id.startswith("/") or repo_id.startswith("\\"):
        raise ValueError("repo_id must not start with a path separator")
    if ".." in repo_id.split("/"):
        raise ValueError("repo_id must not contain '..' segments")
    # Defence-in-depth: control chars (incl. CR / LF) would be a header
    # injection hazard if the id ever flowed into an HTTP request line.
    if any(ord(c) < 0x20 for c in repo_id):
        raise ValueError("repo_id must not contain control characters")
    return repo_id


def _missing_dep_message(hub: str) -> str:
    """Friendly ImportError message naming the pip install command."""
    pkg = required_hub_package(hub) or hub
    return (
        f"hub={hub!r} requires the '{pkg}' package. "
        f"Install with: pip install {pkg}"
    )


def _validate_local_path(value: str, *, field: str) -> str:
    """Cwd-containment + shape check for ``local_dir`` / ``folder_path``.

    Mirrors the project-standard ``utils.paths.is_under_cwd`` policy used by
    every other path-accepting helper since v0.26.0. Rejects bool BEFORE
    `isinstance(str)` (matches v0.30.0 ``Candidate`` policy).
    """
    from soup_cli.utils.paths import is_under_cwd

    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if not is_under_cwd(value):
        raise ValueError(
            f"{field} must stay under the current working directory"
        )
    return value


def download_repo(
    hub: str,
    repo_id: str,
    *,
    local_dir: str,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    repo_type: str = "model",
) -> str:
    """Snapshot-download ``repo_id`` from ``hub`` into ``local_dir``.

    Returns the absolute local path to the downloaded snapshot. Lazy-imports
    the appropriate SDK per ``hub``:

    * ``hf``      → :func:`huggingface_hub.snapshot_download`
    * ``modelscope`` → :func:`modelscope.snapshot_download`
    * ``modelers``   → :func:`openmind_hub.snapshot_download`

    Raises ``ImportError`` (with pip-install hint) when the SDK is missing,
    ``ValueError`` for invalid args, ``TypeError`` for wrong types.
    """
    canonical = validate_hub_name(hub)
    _validate_repo_id_shape(repo_id)
    _validate_local_path(local_dir, field="local_dir")
    if revision is not None:
        if not isinstance(revision, str):
            raise TypeError("revision must be str or None")
        if "\x00" in revision or any(ord(c) < 0x20 for c in revision):
            raise ValueError("revision must not contain control characters")
    if repo_type not in ("model", "dataset", "space"):
        raise ValueError(
            "repo_type must be one of 'model' / 'dataset' / 'space'"
        )

    if canonical == "hf":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(_missing_dep_message("hf")) from exc
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )

    if canonical == "modelscope":
        try:
            from modelscope import (
                snapshot_download as ms_download,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise ImportError(_missing_dep_message("modelscope")) from exc
        # modelscope's snapshot_download uses a different kwarg set; we map
        # the canonical args here so callers see one consistent API.
        ms_kwargs: dict[str, object] = {
            "model_id": repo_id,
            "cache_dir": local_dir,
        }
        if revision is not None:
            ms_kwargs["revision"] = revision
        if allow_patterns is not None:
            ms_kwargs["allow_file_pattern"] = allow_patterns
        return ms_download(**ms_kwargs)

    if canonical == "modelers":
        try:
            from openmind_hub import (
                snapshot_download as om_download,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise ImportError(_missing_dep_message("modelers")) from exc
        om_kwargs: dict[str, object] = {
            "repo_id": repo_id,
            "local_dir": local_dir,
        }
        if revision is not None:
            om_kwargs["revision"] = revision
        if allow_patterns is not None:
            om_kwargs["allow_patterns"] = allow_patterns
        return om_download(**om_kwargs)

    # Unreachable — validate_hub_name has already rejected unknown hubs.
    raise ValueError(f"hub {canonical!r} has no download adapter")


def upload_repo(
    hub: str,
    repo_id: str,
    *,
    folder_path: str,
    commit_message: str = "Upload via Soup",
    token: str | None = None,
    repo_type: str = "model",
) -> None:
    """Upload ``folder_path`` to ``repo_id`` on ``hub``.

    Same lazy-import policy as :func:`download_repo`. Token resolution is
    left to the caller (each backend has its own conventions); pass
    ``token`` explicitly or rely on the SDK's env-var defaults.
    """
    canonical = validate_hub_name(hub)
    _validate_repo_id_shape(repo_id)
    _validate_local_path(folder_path, field="folder_path")
    if not isinstance(commit_message, str) or not commit_message:
        raise ValueError("commit_message must be a non-empty string")
    # Mirror v0.29.0 push policy: first line only, ≤200 chars (prevents
    # multi-line injection into public commit history).
    commit_message = commit_message.splitlines()[0][:200]
    if repo_type not in ("model", "dataset", "space"):
        raise ValueError(
            "repo_type must be one of 'model' / 'dataset' / 'space'"
        )

    if canonical == "hf":
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(_missing_dep_message("hf")) from exc
        api = HfApi(token=token)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        return

    if canonical == "modelscope":
        try:
            from modelscope.hub.api import HubApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_missing_dep_message("modelscope")) from exc
        api = HubApi()
        if token:
            api.login(token)
        # ModelScope's `push_model` does not accept `commit_message` — pass
        # only the model id + dir. The sanitised commit_message is recorded
        # in the operator's local git log via the HF/Modelers backends.
        api.push_model(
            model_id=repo_id,
            model_dir=folder_path,
        )
        return

    if canonical == "modelers":
        try:
            from openmind_hub import HubApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_missing_dep_message("modelers")) from exc
        api = HubApi(token=token)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        return

    raise ValueError(f"hub {canonical!r} has no upload adapter")
