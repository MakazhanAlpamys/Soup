"""HuggingFace Hub integration utilities (v0.29.0).

Single source of truth for HF token/endpoint resolution, repo ID validation,
and an ``HfApi`` factory used by push, data push, deploy hf-space, and
auto-push callbacks.

Design goals:
- **No custom token flags**: respect ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``
  env vars and the cached login from ``huggingface-cli login``.
- **Self-hosted Hub**: ``HF_ENDPOINT`` env var overrides the public endpoint.
- **SSRF hardening**: only HTTPS remotes allowed; localhost HTTP permitted
  for self-hosted dev rigs.
- **Lazy import**: ``huggingface_hub`` is optional and imported inside
  ``get_hf_api``.
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_ENDPOINT = "https://huggingface.co"

# Loopback hosts that may legitimately use plain HTTP (dev / self-hosted).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# repo IDs are either "name" or "owner/name"; both parts must be safe.
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

# Collection slugs use "owner/title-<hash>" with longer hashes allowed.
_COLLECTION_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def resolve_token(explicit: Optional[str] = None) -> Optional[str]:
    """Return the HF token following the documented precedence.

    Order:
    1. ``explicit`` argument (for tests / programmatic callers).
    2. ``HF_TOKEN`` env var.
    3. ``HUGGINGFACE_HUB_TOKEN`` env var.
    4. Cached login at ``~/.cache/huggingface/token`` (new location).
    5. Legacy cached login at ``~/.huggingface/token``.
    6. ``None`` if no token is available.
    """
    if explicit:
        stripped = explicit.strip()
        if not stripped:
            # Whitespace-only — fall through to env/cache lookup rather
            # than return an empty string as a token.
            pass
        elif not stripped.isprintable():
            raise ValueError("explicit token contains non-printable characters")
        else:
            return stripped

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        stripped = env_token.strip()
        if stripped:
            return stripped

    home = Path.home()
    for candidate in (
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ):
        if candidate.is_file():
            try:
                value = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                return value
    return None


def resolve_endpoint() -> str:
    """Return the HF endpoint, validating ``HF_ENDPOINT`` when set.

    Strips one trailing slash, rejects non-HTTPS remotes (localhost HTTP is
    allowed for self-hosted dev setups), and rejects null bytes / bogus
    schemes.
    """
    raw = os.environ.get("HF_ENDPOINT")
    if not raw:
        return DEFAULT_ENDPOINT

    if "\x00" in raw:
        raise ValueError("HF_ENDPOINT must not contain null bytes")

    stripped = raw.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"HF_ENDPOINT must use http/https scheme, got: {parsed.scheme}")
    if not parsed.netloc:
        raise ValueError("HF_ENDPOINT is missing a host")

    host = parsed.hostname or ""
    if host == "0.0.0.0":
        raise ValueError(
            "HF_ENDPOINT 0.0.0.0 is ambiguous; use 127.0.0.1 or localhost"
        )
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        # Reject plain HTTP for RFC1918 / link-local / cloud metadata too —
        # HF_ENDPOINT=http://169.254.169.254 or http://192.168.1.1 would
        # otherwise route SDK traffic to internal targets.
        if _is_private_or_link_local(host):
            raise ValueError(
                "HF_ENDPOINT plain HTTP is only allowed for loopback "
                "(localhost / 127.0.0.1 / ::1); private/link-local hosts "
                "require HTTPS"
            )
        raise ValueError(
            "HF_ENDPOINT for remote hosts must use HTTPS (localhost HTTP allowed)"
        )
    return stripped


def _is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` resolves to a private / link-local / loopback IP."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Hostname — we don't resolve DNS here (the SDK does), so fall back
        # to "treat as public". A malicious DNS record pointing to a private
        # IP is out of scope for this local-tool threat model.
        return False
    return addr.is_private or addr.is_link_local or addr.is_loopback


def validate_repo_id(repo_id: str) -> None:
    """Validate a HuggingFace repo ID (raises ``ValueError`` on bad input).

    Accepts ``owner/name`` or ``name``. Each component must start with an
    alphanumeric character and contain only ``[A-Za-z0-9._-]``, ≤96 chars.
    """
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("repo_id must be a non-empty string")
    if "\x00" in repo_id:
        raise ValueError("repo_id must not contain null bytes")
    if len(repo_id) > 200:
        raise ValueError("repo_id too long (max 200 chars)")
    if any(ch.isspace() for ch in repo_id):
        raise ValueError("repo_id must not contain whitespace")
    if ".." in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError(f"repo_id contains invalid path segments: {repo_id!r}")

    parts = repo_id.split("/")
    if len(parts) > 2:
        raise ValueError(f"repo_id must be 'owner/name' or 'name', got: {repo_id!r}")
    for part in parts:
        if not _REPO_PART_RE.match(part):
            raise ValueError(f"repo_id component invalid: {part!r}")


def validate_collection_slug(slug: str) -> None:
    """Validate a HuggingFace collection slug (``owner/slug-hash``)."""
    if not isinstance(slug, str) or not slug:
        raise ValueError("collection slug must be a non-empty string")
    if "\x00" in slug or any(ch.isspace() for ch in slug):
        raise ValueError("collection slug must not contain whitespace or null bytes")
    if ".." in slug or slug.startswith("/") or slug.endswith("/"):
        raise ValueError(f"collection slug has invalid segments: {slug!r}")
    if len(slug) > 256:
        raise ValueError("collection slug too long (max 256 chars)")
    if not _COLLECTION_SLUG_RE.match(slug):
        raise ValueError(f"collection slug must be 'owner/slug-hash', got: {slug!r}")


def get_hf_api(
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Any:
    """Return a configured ``HfApi`` instance.

    Lazy-imports ``huggingface_hub`` so the rest of the CLI works without
    the optional dependency. Raises ``ImportError`` with install hint when
    missing.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF Hub operations. "
            "Install with: pip install huggingface-hub"
        ) from exc

    return HfApi(token=token, endpoint=endpoint)


def add_to_collection(
    collection_slug: str,
    repo_id: str,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    item_type: str = "model",
    ignore_duplicate: bool = True,
) -> None:
    """Add a repo to an existing HF Collection.

    Validates inputs and swallows "already exists" errors when
    ``ignore_duplicate`` is set (default) so repeated pushes are idempotent.
    """
    validate_collection_slug(collection_slug)
    validate_repo_id(repo_id)
    if item_type not in ("model", "dataset", "space"):
        raise ValueError(f"item_type must be model|dataset|space, got: {item_type!r}")

    api = get_hf_api(token=token, endpoint=endpoint)
    try:
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type=item_type,
        )
    except Exception as exc:
        if not ignore_duplicate:
            raise
        # Prefer HTTP status (409 = Conflict) when huggingface_hub raises a
        # typed HfHubHTTPError; fall back to a string-match on the exception
        # message for older hub versions that lack a typed response.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 409:
            return
        message = str(exc).lower()
        if "already" in message or "exists" in message:
            return
        raise
