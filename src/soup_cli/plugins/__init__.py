"""v0.45.0 Part A — Plugin / hook system.

Public API for third-party plugins. Plugins register themselves at module
import time via ``register_plugin(...)`` and provide hooks the trainer fires
at well-known points (``pre_train`` / ``post_train`` / ``pre_step`` /
``post_step``). Plugins may also expose chat-template and model-group names as
descriptive metadata in ``soup plugins``.

Bundled modules and third-party ``soup_cli.plugins`` entry points are discovered
lazily by plugin-aware CLI and training paths. Third-party code and hooks are
loaded only after explicit opt-in, and their enabled state persists across Soup
processes.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import pkgutil
import re
import stat
import tempfile
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Plugin name: kebab-case, alphanumeric + hyphens; 1..40 chars, leading alnum.
_PLUGIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,39}$")
# Semver-ish: MAJOR.MINOR.PATCH with optional ``-tag`` / ``+build`` suffix.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[\-+][A-Za-z0-9.\-]{1,32})?$")
_MAX_PLUGINS = 64
_MAX_DESCRIPTION = 256
_MAX_TEMPLATES_PER_PLUGIN = 32
_MAX_MODEL_GROUPS_PER_PLUGIN = 32
_MAX_NAME_ENTRY_LEN = 128
_MAX_STATE_BYTES = 64 * 1024
_ENTRY_POINT_GROUP = "soup_cli.plugins"
_STATE_PATH_ENV = "SOUP_PLUGIN_STATE_PATH"

_HOOK_NAMES: Tuple[str, ...] = (
    "pre_train",
    "post_train",
    "pre_step",
    "post_step",
)


@runtime_checkable
class BasePlugin(Protocol):
    """Duck-typed plugin protocol.

    Plugins are objects (instance or class) carrying any subset of the
    four hook methods. Each hook accepts a single ``context`` argument
    (an opaque dict the trainer fills in) and returns ``None``.
    """

    def pre_train(self, context: Dict[str, Any]) -> None: ...

    def post_train(self, context: Dict[str, Any]) -> None: ...

    def pre_step(self, context: Dict[str, Any]) -> None: ...

    def post_step(self, context: Dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PluginSpec:
    """One registered plugin."""

    name: str
    version: str
    plugin: Any
    description: str = ""
    enabled: bool = True
    templates: Tuple[str, ...] = ()
    model_groups: Tuple[str, ...] = ()


_PLUGINS: Dict[str, PluginSpec] = {}
_ENTRY_POINTS: Dict[str, Any] = {}
_LOCK = RLock()
_LOAD_LOCK = RLock()
_DISCOVERY_COMPLETE = False


def _plugin_state_path() -> str:
    """Return the opt-in state file path (the env override primarily aids tests)."""
    override = os.environ.get(_STATE_PATH_ENV)
    if override:
        if "\x00" in override or len(override) > 4096:
            raise ValueError(f"{_STATE_PATH_ENV} must be a valid filesystem path")
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), ".soup", "plugins.json")


def _read_enabled_state() -> Dict[str, bool]:
    """Read explicit plugin choices; malformed state is ignored safely."""
    path = "<invalid>"
    try:
        path = _plugin_state_path()
        if not os.path.exists(path):
            return {}
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError("plugin state file must not be a symlink")
        with open(path, encoding="utf-8") as handle:
            raw = handle.read(_MAX_STATE_BYTES + 1)
        if len(raw.encode("utf-8")) > _MAX_STATE_BYTES:
            raise ValueError("plugin state file is too large")
        payload = json.loads(raw)
        enabled = payload.get("enabled") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("plugin state file has an unsupported schema")
        if not isinstance(enabled, dict):
            raise ValueError("plugin state file has an unsupported schema")
        clean: Dict[str, bool] = {}
        for name, value in enabled.items():
            if (
                isinstance(name, str)
                and _PLUGIN_NAME_RE.match(name)
                and isinstance(value, bool)
            ):
                clean[name] = value
        return clean
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        logger.warning("Ignoring unreadable plugin state %s: %s", path, exc)
        return {}


def _write_enabled_state(enabled: Mapping[str, bool]) -> None:
    """Persist explicit enable/disable choices with an atomic replacement."""
    path = _plugin_state_path()
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.path.lexists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
        raise ValueError("plugin state file must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=".plugins.", suffix=".tmp", dir=parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "enabled": dict(sorted(enabled.items()))}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _replace_enabled(name: str, enabled: bool) -> None:
    existing = _PLUGINS[name]
    _PLUGINS[name] = PluginSpec(
        name=existing.name,
        version=existing.version,
        plugin=existing.plugin,
        description=existing.description,
        enabled=enabled,
        templates=existing.templates,
        model_groups=existing.model_groups,
    )


def _apply_enabled_state(
    names: set[str], saved: Mapping[str, bool], *, default: bool
) -> None:
    with _LOCK:
        for name in names:
            _replace_enabled(name, saved.get(name, default))


def _iter_plugin_entry_points() -> tuple[Any, ...]:
    """Return entry points across the Python 3.10+ metadata API variants."""
    discovered = importlib.metadata.entry_points()
    selected = (
        discovered.select(group=_ENTRY_POINT_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(_ENTRY_POINT_GROUP, ())
    )
    return tuple(selected)


def _disabled_entry_point_spec(entry_point: Any) -> PluginSpec:
    """Represent installed metadata without importing third-party code."""
    try:
        version = str(entry_point.dist.version)
    except (AttributeError, TypeError):
        version = "unknown"
    return PluginSpec(
        name=entry_point.name,
        version=version,
        plugin=None,
        description="Installed entry point; enable it to load third-party code.",
        enabled=False,
    )


def _remember_disabled_entry_point(entry_point: Any) -> None:
    """Publish one metadata-only placeholder while preserving the registry cap."""
    with _LOCK:
        if entry_point.name in _PLUGINS:
            return
        if len(_PLUGINS) >= _MAX_PLUGINS:
            raise RuntimeError(f"too many plugins (max {_MAX_PLUGINS})")
        _PLUGINS[entry_point.name] = _disabled_entry_point_spec(entry_point)


def _load_enabled_entry_point(
    entry_point: Any, saved: Mapping[str, bool]
) -> set[str]:
    """Import one explicitly enabled entry point and apply its saved state."""
    before = set(list_plugins())
    try:
        loaded = entry_point.load()
        # The supported contract is a zero-argument registration callable.
        # Import-side-effect plugins remain compatible: if load() already
        # registered something, do not call the returned object again.
        if set(list_plugins()) == before and callable(loaded):
            loaded()
        registered = set(list_plugins()) - before
        if entry_point.name not in registered:
            raise ValueError(
                f"entry point {entry_point.name!r} must register a plugin "
                "with the same name"
            )
        _apply_enabled_state(registered, saved, default=True)
        return registered
    except Exception:
        # A registrar may have registered one plugin before failing. Never
        # leave partially loaded third-party code enabled.
        _apply_enabled_state(set(list_plugins()) - before, {}, default=False)
        raise


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("plugin name must be a string")
    if not _PLUGIN_NAME_RE.match(name):
        raise ValueError(
            "plugin name must be kebab-case ([a-z0-9][a-z0-9-]{0,39})"
        )


def _validate_version(version: str) -> None:
    if not isinstance(version, str):
        raise TypeError("plugin version must be a string")
    if not _VERSION_RE.match(version):
        raise ValueError(
            "plugin version must match MAJOR.MINOR.PATCH (semver)"
        )


def _validate_description(description: str) -> None:
    if not isinstance(description, str):
        raise TypeError("description must be a string")
    if "\x00" in description:
        raise ValueError("description must not contain null bytes")
    if len(description) > _MAX_DESCRIPTION:
        raise ValueError(f"description exceeds {_MAX_DESCRIPTION} chars")


def list_hook_names() -> Tuple[str, ...]:
    """Return the canonical hook names recognised by the trainer."""
    return _HOOK_NAMES


def discover_hooks(plugin: Any) -> Dict[str, Callable[[Dict[str, Any]], None]]:
    """Return the subset of canonical hooks the plugin actually implements.

    A hook is considered implemented when ``getattr(plugin, name)`` is a
    callable. Missing or non-callable attributes are silently skipped —
    plugins are not required to implement every hook.
    """
    found: Dict[str, Callable[[Dict[str, Any]], None]] = {}
    for hook in _HOOK_NAMES:
        candidate = getattr(plugin, hook, None)
        if callable(candidate):
            found[hook] = candidate
    return found


def register_plugin(
    *,
    name: str,
    version: str,
    plugin: Any,
    description: str = "",
    templates: Optional[List[str]] = None,
    model_groups: Optional[List[str]] = None,
) -> PluginSpec:
    """Register a plugin. Idempotent for an identical spec; rejects
    re-registration with a different version or plugin object."""
    _validate_name(name)
    _validate_version(version)
    _validate_description(description)
    if plugin is None:
        raise ValueError("plugin object must not be None")
    # Hook discovery is best-effort: we don't require any hook, but at
    # least one of {hooks, templates, model_groups} must be non-empty so a
    # totally-empty plugin is rejected loudly.
    hooks = discover_hooks(plugin)
    tpls = tuple(templates or ())
    grps = tuple(model_groups or ())
    if len(tpls) > _MAX_TEMPLATES_PER_PLUGIN:
        raise ValueError(
            f"templates exceeds {_MAX_TEMPLATES_PER_PLUGIN} entries"
        )
    if len(grps) > _MAX_MODEL_GROUPS_PER_PLUGIN:
        raise ValueError(
            f"model_groups exceeds {_MAX_MODEL_GROUPS_PER_PLUGIN} entries"
        )
    for tpl in tpls:
        if not isinstance(tpl, str) or not tpl or "\x00" in tpl:
            raise ValueError("template name must be non-empty NUL-free str")
        if len(tpl) > _MAX_NAME_ENTRY_LEN:
            raise ValueError(
                f"template name exceeds {_MAX_NAME_ENTRY_LEN} chars"
            )
    for grp in grps:
        if not isinstance(grp, str) or not grp or "\x00" in grp:
            raise ValueError("model_group name must be non-empty NUL-free str")
        if len(grp) > _MAX_NAME_ENTRY_LEN:
            raise ValueError(
                f"model_group name exceeds {_MAX_NAME_ENTRY_LEN} chars"
            )
    if not hooks and not tpls and not grps:
        raise ValueError(
            "plugin must implement at least one hook OR register a template "
            "OR register a model group"
        )
    spec = PluginSpec(
        name=name,
        version=version,
        plugin=plugin,
        description=description,
        templates=tpls,
        model_groups=grps,
    )
    with _LOCK:
        if len(_PLUGINS) >= _MAX_PLUGINS and name not in _PLUGINS:
            raise RuntimeError(f"too many plugins (max {_MAX_PLUGINS})")
        existing = _PLUGINS.get(name)
        if existing is not None:
            if (
                existing.version != version
                or existing.plugin is not plugin
                or existing.templates != tpls
                or existing.model_groups != grps
                or existing.description != description
            ):
                raise ValueError(
                    f"plugin {name!r} already registered with a different spec"
                )
            # Identical re-register: keep enabled state.
            return existing
        _PLUGINS[name] = spec
    return spec


def list_plugins() -> Mapping[str, PluginSpec]:
    """Return an immutable view of registered plugins."""
    with _LOCK:
        return MappingProxyType(dict(_PLUGINS))


def get_plugin(name: str) -> Optional[PluginSpec]:
    """Return the registered plugin spec for ``name``, or ``None``."""
    if not isinstance(name, str):
        return None
    with _LOCK:
        return _PLUGINS.get(name)


def enable_plugin(name: str) -> bool:
    """Mark a registered plugin enabled and persist the choice."""
    _validate_name(name)
    with _LOAD_LOCK:
        existing = get_plugin(name)
        if existing is None:
            raise KeyError(name)
        was_enabled = existing.enabled
        saved = _read_enabled_state()
        if existing.plugin is None:
            entry_point = _ENTRY_POINTS.get(name)
            if entry_point is None:
                raise KeyError(name)
            before = set(list_plugins())
            with _LOCK:
                _PLUGINS.pop(name, None)
            try:
                _load_enabled_entry_point(
                    entry_point,
                    {**saved, name: True},
                )
            except Exception as exc:
                with _LOCK:
                    for registered_name in set(_PLUGINS) - (before - {name}):
                        _PLUGINS.pop(registered_name, None)
                    _PLUGINS[name] = existing
                raise ValueError(
                    f"failed to enable plugin {name!r}: {type(exc).__name__}"
                ) from exc
            existing = get_plugin(name)
            if existing is None:
                raise ValueError(f"entry point {name!r} registered no plugin")
        saved[name] = True
        _write_enabled_state(saved)
        changed = not was_enabled
        if changed:
            _replace_enabled(name, True)
        return changed


def disable_plugin(name: str) -> bool:
    """Mark a registered plugin disabled and persist the choice."""
    _validate_name(name)
    with _LOCK:
        existing = _PLUGINS.get(name)
        if existing is None:
            raise KeyError(name)
        saved = _read_enabled_state()
        saved[name] = False
        _write_enabled_state(saved)
        changed = existing.enabled
        if changed:
            _replace_enabled(name, False)
        return changed


def is_enabled(name: str) -> bool:
    """Return True iff ``name`` is registered and enabled."""
    spec = get_plugin(name)
    return bool(spec and spec.enabled)


def clear_plugins() -> None:
    """Remove all registered plugins. Used by tests."""
    global _DISCOVERY_COMPLETE
    with _LOAD_LOCK:
        with _LOCK:
            _PLUGINS.clear()
            _ENTRY_POINTS.clear()
            _DISCOVERY_COMPLETE = False


def load_plugins() -> int:
    """Discover built-in modules and ``soup_cli.plugins`` entry points once.

    Plugin failures are caught and logged at WARNING — one bad plugin
    must not crash a listing or training run. Bundled modules are enabled by
    default. Third-party entry points are listed from package metadata but not
    imported until the user explicitly enables them.
    """
    with _LOAD_LOCK:
        return _load_plugins_once()


def _load_plugins_once() -> int:
    """Run discovery while :func:`load_plugins` holds the load lock."""
    global _DISCOVERY_COMPLETE
    with _LOCK:
        if _DISCOVERY_COMPLETE:
            return 0
        # Set before importing plugins so a plugin that asks for the registry
        # during its own import cannot recursively start discovery again.
        _DISCOVERY_COMPLETE = True

    saved = _read_enabled_state()
    count = 0
    pkg = importlib.import_module(__name__)
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        before = set(list_plugins())
        try:
            importlib.import_module(f"{__name__}.{module_info.name}")
            _apply_enabled_state(set(list_plugins()) - before, saved, default=True)
            count += 1
        except Exception:  # noqa: BLE001 — plugin failure must not crash CLI
            logger.exception(
                "Failed to load Soup plugin: %s", module_info.name
            )

    try:
        entry_points = _iter_plugin_entry_points()
    except Exception:  # noqa: BLE001 — broken metadata must not crash training
        logger.exception("Failed to enumerate Soup plugin entry points")
        entry_points = ()

    for entry_point in entry_points:
        try:
            _validate_name(entry_point.name)
            _ENTRY_POINTS[entry_point.name] = entry_point
            if saved.get(entry_point.name, False):
                _load_enabled_entry_point(entry_point, saved)
            else:
                _remember_disabled_entry_point(entry_point)
            count += 1
        except Exception:  # noqa: BLE001 — one plugin must not crash training
            entry_point_name = getattr(entry_point, "name", "")
            if isinstance(entry_point_name, str) and entry_point_name in _ENTRY_POINTS:
                try:
                    _remember_disabled_entry_point(entry_point)
                except RuntimeError:
                    pass
            logger.exception(
                "Failed to discover Soup plugin entry point: %s",
                entry_point_name or "<unknown>",
            )
    return count


__all__ = [
    "BasePlugin",
    "PluginSpec",
    "discover_hooks",
    "list_hook_names",
    "register_plugin",
    "list_plugins",
    "get_plugin",
    "enable_plugin",
    "disable_plugin",
    "is_enabled",
    "clear_plugins",
    "load_plugins",
]
