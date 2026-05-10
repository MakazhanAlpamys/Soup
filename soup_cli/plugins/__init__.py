"""v0.45.0 Part A — Plugin / hook system.

Public API for third-party plugins. Plugins register themselves at module
import time via ``register_plugin(...)`` and provide hooks the trainer fires
at well-known points (``pre_train`` / ``post_train`` / ``pre_step`` /
``post_step``). Plugins may also register chat templates and model groups
via ``register_template`` / ``register_model_group``.

This release ships the registry and CLI surface; live trainer-callback
wiring lands in v0.45.1 (mirrors the v0.27.0 MII / v0.37.0 multipack
stub-then-live pattern).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
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
_LOCK = RLock()


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
    """Mark a registered plugin enabled. Returns True iff it changed state."""
    _validate_name(name)
    with _LOCK:
        existing = _PLUGINS.get(name)
        if existing is None:
            raise KeyError(name)
        if existing.enabled:
            return False
        _PLUGINS[name] = PluginSpec(
            name=existing.name,
            version=existing.version,
            plugin=existing.plugin,
            description=existing.description,
            enabled=True,
            templates=existing.templates,
            model_groups=existing.model_groups,
        )
        return True


def disable_plugin(name: str) -> bool:
    """Mark a registered plugin disabled. Returns True iff it changed state."""
    _validate_name(name)
    with _LOCK:
        existing = _PLUGINS.get(name)
        if existing is None:
            raise KeyError(name)
        if not existing.enabled:
            return False
        _PLUGINS[name] = PluginSpec(
            name=existing.name,
            version=existing.version,
            plugin=existing.plugin,
            description=existing.description,
            enabled=False,
            templates=existing.templates,
            model_groups=existing.model_groups,
        )
        return True


def is_enabled(name: str) -> bool:
    """Return True iff ``name`` is registered and enabled."""
    spec = get_plugin(name)
    return bool(spec and spec.enabled)


def clear_plugins() -> None:
    """Remove all registered plugins. Used by tests."""
    with _LOCK:
        _PLUGINS.clear()


def load_plugins() -> int:
    """Import every ``soup_cli.plugins.*`` submodule. Returns count loaded.

    Plugin failures are caught and logged at WARNING — one bad plugin
    must not crash ``soup`` startup (mirrors the v0.44.0 Web UI plugin
    loader policy).
    """
    count = 0
    pkg = importlib.import_module(__name__)
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module_info.name}")
            count += 1
        except Exception:  # noqa: BLE001 — plugin failure must not crash CLI
            logger.exception(
                "Failed to load Soup plugin: %s", module_info.name
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
