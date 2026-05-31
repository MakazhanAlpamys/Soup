"""v0.43.0 Part A — Tracker integrations + PostHog telemetry opt-out.

Closed allowlist of HF Trainer `report_to` backends Soup recognises.
Adds mlflow / swanlab / trackio to the legacy `wandb` / `tensorboard` / `none`
set. Live integrations rely on HF Trainer's built-in callbacks (mlflow,
swanlab) plus the third-party `trackio` callback when installed; Soup only
validates the name and surfaces a friendly error when the backing package
is missing.

Telemetry: opt-out via `SOUP_TELEMETRY=0` env var. Default is OFF until a
public privacy policy ships — `is_telemetry_enabled` returns False unless
the user explicitly enables it. Hardware-info-only payload schema lives in
`build_telemetry_payload` for documentation/testing; no network calls in
v0.43.0 (PostHog wire-up deferred to v0.43.1).
"""
from __future__ import annotations

import math
import os
import platform
from types import MappingProxyType
from typing import Mapping

# Closed allowlist of report_to backends.
_REPORT_TO_BACKENDS: Mapping[str, str | None] = MappingProxyType({
    "none": None,
    "wandb": "wandb",
    "tensorboard": "tensorboard",
    "mlflow": "mlflow",
    "swanlab": "swanlab",
    "trackio": "trackio",
})

SUPPORTED_TRACKERS = frozenset(_REPORT_TO_BACKENDS.keys())

# v0.43.0 additions (HF-native wandb/tensorboard already supported).
NEW_TRACKERS_V0_43 = frozenset({"mlflow", "swanlab", "trackio"})

_MAX_NAME_LEN = 32


def validate_tracker_name(name: object) -> str:
    """Validate and lowercase a `report_to` tracker name.

    Returns the canonical lower-cased name. Raises ValueError on invalid
    input. Mirrors v0.41.0 `validate_optimizer_name` policy.
    """
    if not isinstance(name, str):
        raise ValueError(f"tracker name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("tracker name must not be empty")
    if "\x00" in name:
        raise ValueError("tracker name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"tracker name length {len(name)} exceeds max {_MAX_NAME_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_TRACKERS:
        supported = ", ".join(sorted(SUPPORTED_TRACKERS))
        raise ValueError(
            f"unknown tracker '{name}'. Supported: {supported}"
        )
    return canonical


def required_tracker_package(name: str) -> str | None:
    """Return the pip-installable package name for a tracker, or None.

    Non-string input returns None (mirrors `is_new_v0_43_tracker`).
    """
    if not isinstance(name, str):
        return None
    return _REPORT_TO_BACKENDS.get(name.lower())


def is_new_v0_43_tracker(name: object) -> bool:
    """True if the name is an additive v0.43.0 tracker, False otherwise."""
    if not isinstance(name, str):
        return False
    return name.lower() in NEW_TRACKERS_V0_43


# --- Telemetry (opt-out, default OFF) ----------------------------------

_TELEMETRY_ENV_VAR = "SOUP_TELEMETRY"


def is_telemetry_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Telemetry is opt-IN until v0.43.1 ships the network code.

    The roadmap entry calls this opt-out, but until the privacy policy
    + PostHog wire-up land we keep it default-OFF so no payload is built
    or sent. Users may enable explicitly with `SOUP_TELEMETRY=1`.
    """
    source = env if env is not None else os.environ
    raw = source.get(_TELEMETRY_ENV_VAR)
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    return False


def build_telemetry_payload(
    *,
    soup_version: str,
    command: str,
    duration_seconds: float | int | None = None,
) -> dict:
    """Build the hardware-info-only telemetry payload.

    The payload contains NO user data, dataset paths, model names, or
    config contents. Documented schema:

      - `soup_version`: caller-supplied
      - `command`: top-level CLI command (e.g. `train`, `data ingest`)
      - `python`: major.minor only
      - `os`: platform.system()
      - `arch`: platform.machine()
      - `duration_seconds`: optional, finite float / int / None

    Raises ValueError for non-string `command` / `soup_version` and for
    non-finite `duration_seconds`.
    """
    if not isinstance(soup_version, str) or not soup_version:
        raise ValueError("soup_version must be a non-empty string")
    if "\x00" in soup_version:
        raise ValueError("soup_version must not contain null bytes")
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")
    if "\x00" in command:
        raise ValueError("command must not contain null bytes")
    if duration_seconds is not None:
        # bool is a subclass of int — reject explicitly (project policy)
        if isinstance(duration_seconds, bool) or not isinstance(
            duration_seconds, (int, float)
        ):
            raise ValueError("duration_seconds must be int / float / None")
        if not math.isfinite(float(duration_seconds)):
            raise ValueError("duration_seconds must be finite")
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")

    py = platform.python_version_tuple()
    py_major_minor = f"{py[0]}.{py[1]}"
    return {
        "soup_version": soup_version,
        "command": command,
        "python": py_major_minor,
        "os": platform.system(),
        "arch": platform.machine(),
        "duration_seconds": (
            float(duration_seconds) if duration_seconds is not None else None
        ),
    }


# v0.53.8 #90 — PostHog telemetry live wiring.
# Opt-IN via SOUP_TELEMETRY=1; silent-fail on any network/transport error
# so telemetry can NEVER crash training. 1s hard timeout, HTTPS-only.

_POSTHOG_HOST = "https://us.i.posthog.com"
_POSTHOG_ENDPOINT = f"{_POSTHOG_HOST}/i/v0/e/"
# v0.53.10 #154 — bundled public write-only project key for Soup CLI
# telemetry. The key is INTENTIONALLY hard-coded: PostHog "phc_*" keys are
# write-only (cannot read events back); rotating it requires a release.
# Operators wanting to point telemetry at their own PostHog project should
# set ``SOUP_POSTHOG_KEY`` AND ``SOUP_POSTHOG_ENDPOINT`` together; both env
# vars are validated by :func:`_resolve_posthog_target`.
_POSTHOG_DEFAULT_KEY = "phc_soup_public_write_only"
_TELEMETRY_TIMEOUT_S = 1.0


# Sentinel for "caller did not pass an endpoint, fall back to default + env".
_POSTHOG_ENDPOINT_DEFAULT = object()


def _resolve_posthog_target(
    api_key: str | None,
    endpoint: object = _POSTHOG_ENDPOINT_DEFAULT,
    env: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Resolve ``(key, endpoint)`` from explicit args + ``SOUP_POSTHOG_*`` env.

    v0.53.10 #154 — adds env-var overrides for the bundled defaults so users
    on private PostHog instances can point Soup telemetry at their own
    project without a code change. Precedence:

    1. Explicit ``api_key`` / ``endpoint`` kwargs (caller wins).
    2. ``SOUP_POSTHOG_KEY`` env var (overrides ``_POSTHOG_DEFAULT_KEY``).
    3. ``SOUP_POSTHOG_ENDPOINT`` env var (overrides
       ``_POSTHOG_ENDPOINT``; must be HTTPS + pass the v0.51.0 SSRF policy).
    4. Bundled defaults.

    Returns ``None`` when any input fails validation (silent no-op so
    telemetry can never crash training).
    """
    import os  # noqa: PLC0415 — local lazy import

    src = env if env is not None else os.environ
    # Endpoint resolution: explicit caller > env override > default.
    # Use a sentinel default so a caller who passes
    # ``endpoint=_POSTHOG_ENDPOINT`` (locking in the default) is NOT silently
    # overridden by ``SOUP_POSTHOG_ENDPOINT`` (code-review HIGH fix).
    if endpoint is _POSTHOG_ENDPOINT_DEFAULT:
        env_endpoint = src.get("SOUP_POSTHOG_ENDPOINT")
        resolved_endpoint = env_endpoint or _POSTHOG_ENDPOINT
    else:
        resolved_endpoint = endpoint
    if not isinstance(resolved_endpoint, str):
        return None
    if not _telemetry_endpoint_is_safe(resolved_endpoint):
        return None
    # Key resolution: explicit caller > env override > default.
    if api_key is not None:
        key = api_key
    else:
        key = src.get("SOUP_POSTHOG_KEY") or _POSTHOG_DEFAULT_KEY
    if not isinstance(key, str) or not key:
        return None
    # Reject control chars / whitespace in the key — defends against an
    # operator dropping ``\nAuthorization:...`` into SOUP_POSTHOG_KEY.
    if "\x00" in key or any(ord(c) < 0x20 for c in key) or len(key) > 256:
        return None
    return key, resolved_endpoint


def _telemetry_endpoint_is_safe(endpoint: str) -> bool:
    """Re-validate the telemetry endpoint via the v0.51.0 SSRF policy.

    Even though :func:`send_telemetry_payload` only POSTs to a static
    PostHog URL by default, callers can override ``endpoint``. Re-run the
    same private-IP / link-local rejection used for hub endpoints so a
    crafted ``endpoint='https://10.0.0.1/'`` cannot reach an internal
    network from a misconfigured caller.
    """
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        return False
    try:
        from soup_cli.utils.hubs import validate_hub_endpoint

        validate_hub_endpoint(endpoint, hub="telemetry")
    except (TypeError, ValueError):
        return False
    return True


def send_telemetry_payload(
    payload: dict[str, object],
    *,
    api_key: str | None = None,
    timeout: float = _TELEMETRY_TIMEOUT_S,
    endpoint: object = _POSTHOG_ENDPOINT_DEFAULT,
) -> bool:
    """POST ``payload`` to PostHog if telemetry is enabled, else no-op.

    Returns ``True`` on a 2xx response, ``False`` on any failure or skip.
    NEVER raises — telemetry is best-effort and must never crash training.

    Args:
        payload: dict built by :func:`build_telemetry_payload`. Required keys
            are validated upstream by the builder.
        api_key: PostHog project key. Defaults to the bundled write-only key.
        timeout: hard wall-clock cap (default 1 s).
        endpoint: full PostHog capture URL (must be HTTPS).
    """
    if not is_telemetry_enabled():
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return False
    if not math.isfinite(float(timeout)) or timeout <= 0:
        return False
    # v0.53.10 #154 — resolve key + endpoint via env-override-aware helper.
    # Returns ``None`` when either input fails validation; treat as silent
    # no-op so telemetry remains best-effort.
    resolved = _resolve_posthog_target(api_key, endpoint)
    if resolved is None:
        return False
    key, endpoint = resolved
    try:
        import httpx  # lazy — optional dep, surfaces no advisory
    except ImportError:
        return False
    body = {
        "api_key": key,
        "event": payload.get("command", "soup_event"),
        "properties": {k: v for k, v in payload.items() if k != "command"},
    }
    try:
        resp = httpx.post(endpoint, json=body, timeout=timeout)
        return 200 <= resp.status_code < 300
    except Exception:  # noqa: BLE001 — telemetry must never crash training
        return False


# v0.53.8 #89 — Friendly missing-dep panel for HF Trainer `--tracker`.
# When user passes `--tracker mlflow` without mlflow installed, HF raises
# a generic ImportError mid-training; this helper lets the CLI surface a
# pip-install advisory BEFORE construction.


def tracker_missing_dep_message(name: str) -> str | None:
    """Return a friendly install advisory for ``name`` if the package is
    missing, else None.

    Always returns ``None`` for `wandb` / `tensorboard` / `none` (the
    legacy backends), since those are part of the standard HF Trainer
    extra and not part of v0.43.0's additive set.
    """
    if not isinstance(name, str):
        return None
    canonical = name.lower()
    if canonical not in NEW_TRACKERS_V0_43:
        return None
    pkg = required_tracker_package(canonical)
    if not pkg:
        return None
    # Use ``importlib.util.find_spec`` (non-executing probe) so we don't
    # incur side effects from the tracker's top-level module (e.g. swanlab
    # initialises network threads on import). ``sys.modules[pkg] = None``
    # raises ``ValueError`` on find_spec — treat that as missing too so
    # tests can simulate the absent-package path without subprocess.
    import importlib.util
    import sys

    sentinel = object()
    cached = sys.modules.get(pkg, sentinel)
    if cached is None:
        missing = True
    elif cached is not sentinel:
        # Module is already imported (or test injected a real-shaped mock).
        missing = False
    else:
        try:
            missing = importlib.util.find_spec(pkg) is None
        except (ImportError, ValueError):
            missing = True
    if missing:
        return (
            f"--tracker {canonical} requires the '{pkg}' package. "
            f"Install with: pip install soup-cli[trackers] "
            f"(or pip install {pkg})"
        )
    return None


def resolve_report_to(
    *,
    wandb: bool = False,
    tensorboard: bool = False,
    tracker: str | None = None,
) -> str:
    """Resolve the HF Trainer `report_to` value from CLI flags + --tracker.

    Mutual-exclusion: only one of (wandb, tensorboard, tracker) may be set.
    Empty string / None on `tracker` is treated as unset.
    """
    set_count = sum(
        1
        for x in (
            bool(wandb),
            bool(tensorboard),
            bool(tracker) if isinstance(tracker, str) and tracker else False,
        )
        if x
    )
    if set_count > 1:
        raise ValueError(
            "--wandb, --tensorboard, and --tracker are mutually exclusive"
        )
    if wandb:
        return "wandb"
    if tensorboard:
        return "tensorboard"
    if tracker:
        return validate_tracker_name(tracker)
    return "none"
