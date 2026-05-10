"""v0.44.0 Part C — Web UI plugin registry.

Drop-in `soup_cli/ui/plugins/*.py` files register tabs by calling
`register_tab(...)` at import time. The FastAPI app discovers them via
`load_plugins()` at startup.

Plugin contract:

```python
from soup_cli.ui.plugins import register_tab

def render_my_tab(request) -> str:
    return "<div>my tab body</div>"

register_tab(name="my-tab", title="My Tab", render=render_my_tab)
```

Pure-Python: no FastAPI dep at module level.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# Tab name regex — kebab-case, alphanumeric + hyphen.
_TAB_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,30}$")
_MAX_TITLE_LEN = 64
_MAX_TABS = 32

_TABS: Dict[str, "TabSpec"] = {}
_LOCK = Lock()


@dataclass(frozen=True)
class TabSpec:
    """One registered Web UI tab."""

    name: str
    title: str
    render: Callable[..., str]


def register_tab(
    *,
    name: str,
    title: str,
    render: Callable[..., str],
) -> TabSpec:
    """Register a tab. Idempotent for identical (title, render); rejects
    re-registration with a different title or render fn."""
    if not isinstance(name, str) or not _TAB_NAME_RE.match(name):
        raise ValueError(
            "name must be kebab-case ([a-z0-9][a-z0-9-]{0,30})"
        )
    if not isinstance(title, str) or not title or "\x00" in title:
        raise ValueError("title must be a non-empty NUL-free str")
    if len(title) > _MAX_TITLE_LEN:
        raise ValueError(f"title exceeds {_MAX_TITLE_LEN} chars")
    if not callable(render):
        raise TypeError("render must be callable")
    spec = TabSpec(name=name, title=title, render=render)
    with _LOCK:
        if len(_TABS) >= _MAX_TABS and name not in _TABS:
            raise RuntimeError(f"too many tabs (max {_MAX_TABS})")
        existing = _TABS.get(name)
        if existing is not None and (
            existing.title != title or existing.render is not render
        ):
            raise ValueError(
                f"tab name {name!r} already registered with a different spec"
            )
        _TABS[name] = spec
    return spec


def list_tabs() -> Mapping[str, TabSpec]:
    """Return an immutable view of currently-registered tabs."""
    with _LOCK:
        return MappingProxyType(dict(_TABS))


def get_tab(name: str) -> Optional[TabSpec]:
    if not isinstance(name, str):
        return None
    with _LOCK:
        return _TABS.get(name)


def clear_tabs() -> None:
    """Remove all registered tabs. Used by tests."""
    with _LOCK:
        _TABS.clear()


def load_plugins() -> int:
    """Import every `soup_cli.ui.plugins.*` submodule. Returns count loaded."""
    count = 0
    pkg = importlib.import_module(__name__)
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module_info.name}")
            count += 1
        except Exception:  # noqa: BLE001 — plugin failure must not crash UI
            logger.exception(
                "Failed to load Web UI plugin: %s", module_info.name
            )
    return count
