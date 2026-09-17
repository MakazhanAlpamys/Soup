"""Regression tests for issue #819: make the plugin system reachable."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli import plugins as plugins_pkg

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI styling and collapse Rich wrapping before matching output."""
    return " ".join(_ANSI_RE.sub("", text).split())


class _HookPlugin:
    def pre_train(self, _context):
        return None


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path):
    state_path = tmp_path / "plugins.json"
    monkeypatch.setenv("SOUP_PLUGIN_STATE_PATH", str(state_path))
    monkeypatch.setattr(plugins_pkg, "_iter_plugin_entry_points", lambda: ())
    plugins_pkg.clear_plugins()
    yield state_path
    plugins_pkg.clear_plugins()


def _install_fake_entry_point(monkeypatch, name: str = "hello") -> list[str]:
    module_name = f"issue819_{name.replace('-', '_')}"
    module = ModuleType(module_name)
    registrations: list[str] = []

    def register() -> None:
        registrations.append(name)
        plugins_pkg.register_plugin(
            name=name,
            version="1.0.0",
            plugin=_HookPlugin(),
            description="entry-point plugin",
        )

    module.register = register
    monkeypatch.setitem(sys.modules, module_name, module)
    entry_point = importlib.metadata.EntryPoint(
        name=name,
        value=f"{module_name}:register",
        group="soup_cli.plugins",
    )
    monkeypatch.setattr(
        plugins_pkg, "_iter_plugin_entry_points", lambda: (entry_point,)
    )
    return registrations


def test_plugins_cli_discovers_entry_point_without_manual_load(monkeypatch):
    from soup_cli.commands import plugins as plugins_cli

    monkeypatch.setattr(plugins_cli, "console", Console(width=120))
    registrations = _install_fake_entry_point(monkeypatch)
    result = CliRunner().invoke(plugins_cli.app, ["list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "hello" in plain
    assert "disabled" in plain
    assert registrations == []
    placeholder = plugins_pkg.get_plugin("hello")
    assert placeholder is not None
    assert placeholder.plugin is None


def test_explicit_enable_loads_the_selected_entry_point(monkeypatch):
    registrations = _install_fake_entry_point(monkeypatch)
    plugins_pkg.load_plugins()

    assert registrations == []
    assert plugins_pkg.enable_plugin("hello") is True
    assert registrations == ["hello"]
    loaded = plugins_pkg.get_plugin("hello")
    assert loaded is not None
    assert loaded.plugin is not None
    assert plugins_pkg.is_enabled("hello") is True


def test_attach_callback_discovers_enabled_entry_point(monkeypatch, _isolated_registry):
    pytest.importorskip("transformers")
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    _install_fake_entry_point(monkeypatch)
    _isolated_registry.write_text(
        json.dumps({"version": 1, "enabled": {"hello": True}}),
        encoding="utf-8",
    )
    trainer = MagicMock()

    assert attach_plugin_callback(trainer) is True
    trainer.add_callback.assert_called_once()


def test_disable_survives_a_fresh_process(tmp_path):
    package = tmp_path / "fake_soup_plugin.py"
    imported_marker = tmp_path / "third-party-code-ran"
    package.write_text(
        "from pathlib import Path\n"
        f"Path({str(imported_marker)!r}).write_text('imported', encoding='utf-8')\n"
        "from soup_cli.plugins import register_plugin\n"
        "class Plugin:\n"
        "    def pre_train(self, context):\n"
        "        return None\n"
        "def register():\n"
        "    register_plugin(name='hello', version='1.0.0', plugin=Plugin())\n",
        encoding="utf-8",
    )
    dist_info = tmp_path / "fake_soup_plugin-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-soup-plugin\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[soup_cli.plugins]\nhello = fake_soup_plugin:register\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "state" / "plugins.json"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env.update(
        {
            "COLUMNS": "120",
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(repo_src))),
            "SOUP_PLUGIN_STATE_PATH": str(state_path),
            "SOUP_NO_AUDIT_LOG": "1",
            "SOUP_TELEMETRY": "0",
        }
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "soup_cli", "plugins", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    discovered = run("list")
    assert discovered.returncode == 0, (discovered.stdout, discovered.stderr)
    assert "hello" in _plain(discovered.stdout)
    assert "disabled" in _plain(discovered.stdout)
    assert not imported_marker.exists()

    enabled = run("enable", "hello")
    assert enabled.returncode == 0, (enabled.stdout, enabled.stderr)
    assert imported_marker.exists()
    imported_marker.unlink()

    disabled = run("disable", "hello")
    assert disabled.returncode == 0, (disabled.stdout, disabled.stderr)
    assert imported_marker.exists()
    imported_marker.unlink()

    fresh = run("list")
    assert fresh.returncode == 0, (fresh.stdout, fresh.stderr)
    plain = _plain(fresh.stdout)
    assert "hello" in plain
    assert "disabled" in plain
    assert not imported_marker.exists()
    assert json.loads(state_path.read_text(encoding="utf-8"))["enabled"]["hello"] is False


def test_install_stub_fails_instead_of_claiming_success():
    from soup_cli.commands import plugins as plugins_cli

    result = CliRunner().invoke(plugins_cli.app, ["install", "anything"])

    assert result.exit_code != 0
    assert "does not install" in _plain(result.output)


def test_plugin_resources_are_visible_in_the_cli(monkeypatch):
    from soup_cli.commands import plugins as plugins_cli

    monkeypatch.setattr(plugins_cli, "console", Console(width=120))
    plugins_pkg.register_plugin(
        name="resources",
        version="1.0.0",
        plugin=_HookPlugin(),
        templates=["my-template"],
        model_groups=["my-models"],
    )

    result = CliRunner().invoke(plugins_cli.app, ["list"])

    assert result.exit_code == 0
    plain = _plain(result.output)
    assert "my-template" in plain
    assert "my-models" in plain


def test_plugin_resources_are_rendered_as_literal_text(monkeypatch):
    from soup_cli.commands import plugins as plugins_cli

    monkeypatch.setattr(plugins_cli, "console", Console(width=120))
    plugins_pkg.register_plugin(
        name="resources",
        version="1.0.0",
        plugin=_HookPlugin(),
        templates=["[blink]TPL-MARKUP[/]"],
        model_groups=["[bold red]GROUP-MARKUP[/]"],
    )

    result = CliRunner().invoke(plugins_cli.app, ["list"])

    assert result.exit_code == 0
    plain = _plain(result.output)
    assert "[blink]TPL-MARKUP[/]" in plain
    assert "[bold red]GROUP-MARKUP[/]" in plain


def test_partially_failing_entry_point_cannot_leave_plugin_enabled(
    monkeypatch, _isolated_registry
):
    class BrokenEntryPoint:
        name = "partial"

        @staticmethod
        def load():
            def register_then_fail():
                plugins_pkg.register_plugin(
                    name="partial",
                    version="1.0.0",
                    plugin=_HookPlugin(),
                )
                raise RuntimeError("broken registrar")

            return register_then_fail

    monkeypatch.setattr(
        plugins_pkg, "_iter_plugin_entry_points", lambda: (BrokenEntryPoint(),)
    )
    _isolated_registry.write_text(
        json.dumps({"version": 1, "enabled": {"partial": True}}),
        encoding="utf-8",
    )

    plugins_pkg.load_plugins()

    assert plugins_pkg.get_plugin("partial") is not None
    assert plugins_pkg.is_enabled("partial") is False


def test_recursive_state_json_is_ignored_without_crashing(_isolated_registry):
    nested = "[" * 5_000 + "0" + "]" * 5_000
    _isolated_registry.write_text(
        '{"version":1,"enabled":' + nested + "}",
        encoding="utf-8",
    )

    assert plugins_pkg._read_enabled_state() == {}


def test_enable_state_is_published_with_atomic_replace(monkeypatch, _isolated_registry):
    plugins_pkg.register_plugin(
        name="atomic", version="1.0.0", plugin=_HookPlugin()
    )
    real_replace = os.replace
    replacements: list[tuple[str, str]] = []

    def recording_replace(source: str, destination: str) -> None:
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(plugins_pkg.os, "replace", recording_replace)

    plugins_pkg.enable_plugin("atomic")

    assert len(replacements) == 1
    source, destination = replacements[0]
    assert source != destination
    assert destination == str(_isolated_registry)
