"""#1465: the Web UI plugin registry is gone, and nothing still names it.

``soup_cli.ui.plugins`` (``register_tab`` and friends, importable since v0.44.0) was a
registry nobody read: ``create_app`` never imported it, no route and nothing in
``ui/static/`` listed the registered tabs, and ``docs/serving-and-export.md`` promised a
drop-in tab that was never rendered. The maintainer's ruling on #1465 was to remove it
rather than wire it. ``soup_cli.plugins`` (``soup plugins``, the working plugin system)
and ``utils/ui_env.py`` are untouched.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_DOCS = _ROOT / "docs"

#: The spellings the removed registry went by. ``soup_cli.plugins`` (no ``ui``) is a
#: different, working feature and is deliberately not in this list.
_REGISTRY_MARKERS = (
    "soup_cli.ui.plugins",
    "soup_cli/ui/plugins",
    "register_tab",
    "Web UI Plugin Registry",
)

#: The routes the Web UI serves today, read from ``ui/app.py`` before the removal.
_ROUTES_TODAY = {
    "/",
    "/api/auth/ticket",
    "/api/runs",
    "/api/runs/compare",
    "/api/runs/{run_id}",
    "/api/runs/{run_id}/metrics",
    "/api/runs/{run_id}/eval",
    "/api/system",
    "/api/templates",
    "/api/config/validate",
    "/api/config/schema",
    "/api/config/from-form",
    "/api/train/start",
    "/api/train/status",
    "/api/train/stop",
    "/api/train/logs",
    "/api/train/metrics/live",
    "/api/train/progress",
    "/api/data/inspect",
    "/api/recipes",
}


def _files(root: Path, suffixes: tuple[str, ...]):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
            yield path


class TestTheRegistryIsGone:
    def test_the_module_cannot_be_imported(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("soup_cli.ui.plugins")

    def test_the_package_directory_is_gone(self):
        assert not (_SRC / "soup_cli" / "ui" / "plugins").exists()

    def test_nothing_under_src_names_it(self):
        hits = [
            f"{path.relative_to(_ROOT)}: {marker}"
            for path in _files(_SRC, (".py", ".html", ".js", ".css", ".md"))
            for marker in _REGISTRY_MARKERS
            if marker in path.read_text(encoding="utf-8", errors="replace")
        ]
        assert hits == [], hits

    def test_nothing_under_docs_names_it(self):
        hits = [
            f"{path.relative_to(_ROOT)}: {marker}"
            for path in _files(_DOCS, (".md",))
            for marker in _REGISTRY_MARKERS
            if marker in path.read_text(encoding="utf-8")
        ]
        assert hits == [], hits

    def test_the_docs_contents_list_has_no_dangling_anchor(self):
        """The section and its contents-list line go together: a surviving
        ``(#web-ui-plugin-registry--env-knobs)`` entry would point at nothing."""
        text = (_DOCS / "serving-and-export.md").read_text(encoding="utf-8")
        assert "web-ui-plugin-registry" not in text
        assert not re.search(r"^## .*Plugin Registry", text, re.M)


class TestNothingElseMoved:
    def test_the_working_plugin_system_is_still_there(self):
        """``soup plugins`` is a different feature and the README's one mention of
        plugins points at it; it must not be caught in this removal."""
        plugins = importlib.import_module("soup_cli.plugins")
        assert callable(plugins.load_plugins)
        assert "Plugin System" in (_DOCS / "backends-and-ops.md").read_text(encoding="utf-8")

    def test_ui_env_is_untouched(self):
        from soup_cli.utils.ui_env import UiEnv, resolve_ui_env

        assert resolve_ui_env({}) == UiEnv(None, None, None, None, None)

    def test_the_web_ui_still_serves_todays_routes(self):
        pytest.importorskip("fastapi")
        from soup_cli.ui.app import create_app

        served = {getattr(route, "path", None) for route in create_app().routes}
        missing = _ROUTES_TODAY - served
        assert not missing, sorted(missing)
