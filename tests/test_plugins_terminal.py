"""soup plugins prints typed and plugin-supplied text as typed, not as Rich markup."""

import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli import plugins as plugins_pkg
from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


class _HookPlugin:
    def pre_train(self, _context):
        return None


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("SOUP_PLUGIN_STATE_PATH", str(tmp_path / "plugins.json"))
    monkeypatch.setattr(plugins_pkg, "_iter_plugin_entry_points", lambda: ())
    monkeypatch.setattr("soup_cli.commands.plugins.console", Console(width=500))
    plugins_pkg.clear_plugins()
    yield
    plugins_pkg.clear_plugins()


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _register(**fields) -> None:
    plugins_pkg.register_plugin(
        name="demo", version="1.0.0", plugin=_HookPlugin(), **fields
    )


@pytest.mark.parametrize("value", _SPELLINGS)
def test_install_names_the_plugin_as_typed(value):
    result = runner.invoke(app, ["plugins", "install", value])

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Soup does not install plugin {_shown(value)}. Install a trusted" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_list_shows_the_description_as_typed(value):
    _register(description=value)

    result = runner.invoke(app, ["plugins", "list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # \W is the cell border: a box character, or "|" on a terminal without them.
    assert re.search(r"demo \W 1\.0\.0 \W .* \W " + re.escape(_shown(value)) + r" \W", text)
    assert _not_shown(text)


@pytest.mark.parametrize("keyword, label", [
    pytest.param("templates", "templates", id="templates"),
    pytest.param("model_groups", "model groups", id="model-groups"),
])
@pytest.mark.parametrize("value", _SPELLINGS)
def test_list_shows_the_resource_names_as_typed(keyword, label, value):
    _register(**{keyword: [value]})

    result = runner.invoke(app, ["plugins", "list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(
        r"demo \W 1\.0\.0 \W .* \W " + label + ": " + re.escape(_shown(value)) + r" \W", text
    )
    assert _not_shown(text)


@pytest.mark.parametrize("command", ["enable", "disable"])
@pytest.mark.parametrize("value", _SPELLINGS)
def test_a_hostile_plugin_name_is_rejected_before_it_can_be_printed(command, value):
    result = runner.invoke(app, ["plugins", command, value])

    assert result.exit_code == 2
    text = _plain(result)
    assert "kebab-case" in text
    assert _shown(value) not in text
    assert _not_shown(text)


class _Dist:
    def __init__(self, version):
        self.version = version


class _InstalledEntryPoint:
    """An installed third-party plugin, listed from its package metadata, never imported."""

    name = "third-party"

    def __init__(self, version):
        self.dist = _Dist(version)

    def load(self):
        raise AssertionError("a plugin that is not enabled must not be imported")


@pytest.mark.parametrize("value", _SPELLINGS)
def test_list_shows_an_installed_version_as_typed(monkeypatch, value):
    version = f"1.0.0{value}"
    monkeypatch.setattr(
        plugins_pkg, "_iter_plugin_entry_points", lambda: (_InstalledEntryPoint(version),)
    )

    result = runner.invoke(app, ["plugins", "list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"third-party \W " + re.escape(_shown(version)) + r" \W disabled", text)
    assert _not_shown(text)
