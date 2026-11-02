"""soup bom emit prints formats, paths and error text as typed, not as Rich markup."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import RegistryStore
from tests.conftest import strip_ansi

runner = CliRunner()

_SHA = "a" * 64
_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.bom.console", Console(width=500))


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    return tmp_path


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _emit(*args: str):
    return runner.invoke(
        app,
        [
            "bom", "emit", "--name", "adapter", "--base-model", "org/model",
            "--base-sha", _SHA, "--config-sha", _SHA, *args,
        ],
    )


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    "value", ["x[red]f", "x[/]f", pytest.param("f:warning:", id="emoji-code")]
)
def test_unsupported_format_is_named_as_typed(value):
    result = _emit("--format", value)

    assert result.exit_code == 2
    assert f"Unsupported --format: {value} (use cyclonedx" in _plain(result)


def test_unsupported_format_strips_controls_from_the_display():
    result = _emit("--format", f"x[bold]f{_CONTROLS}")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Unsupported --format: x[bold]f (use cyclonedx" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "name",
    [
        "../x[red]e.json",
        "../x[/]e.json",
        pytest.param("../e:warning:.json", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    ],
)
def test_rejected_energy_path_is_named_as_typed(name):
    result = _emit("--energy", name)

    assert result.exit_code == 2
    assert f"Energy path rejected: --energy '{name}' must stay under cwd" in _plain(result)


def test_rejected_energy_path_strips_bidi_controls_from_the_display():
    result = _emit("--energy", f"../x[bold]e{_CONTROLS}.json")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Energy path rejected: --energy '../x[bold]e" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "name",
    ["x[red]e.json", pytest.param("e:warning:.json", marks=_NO_COLON_ON_WINDOWS, id="emoji-code")],
)
def test_missing_energy_file_is_named_as_typed(name):
    result = _emit("--energy", name)

    assert result.exit_code == 2
    assert f"Energy file not found: {name}" in _plain(result)


def test_missing_energy_file_strips_controls_from_the_display():
    result = _emit("--energy", f"x[bold]e{_CONTROLS}.json")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Energy file not found: x[bold]e.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "message", ["denied [/] here", "[red]denied", "stop:warning:", f"x[bold]m{_CONTROLS}"]
)
def test_unreadable_energy_file_is_shown_as_typed(monkeypatch, message):
    Path("energy.json").write_text("{}", encoding="utf-8")
    real_read_text = Path.read_text

    def _deny(self, *args, **kwargs):
        if self.name == "energy.json":
            raise OSError(message)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _deny)

    result = _emit("--energy", "energy.json")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"Cannot read energy file: {message.replace(_CONTROLS, '')}" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "key", ["x[red]k", "x[/]k", "k:warning:", "x[bold]k\u202e", "x[bold]k\u009b"]
)
def test_invalid_energy_measurement_names_the_key_as_typed(key):
    Path("energy.json").write_text(json.dumps({key: 1}), encoding="utf-8")

    result = _emit("--energy", "energy.json")

    assert result.exit_code == 2
    text = _plain(result)
    shown = key.replace("\u202e", "").replace("\u009b", "")
    assert f"unexpected keyword argument '{shown}'" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "output",
    [
        "../x[red]o",
        "../x[/]o",
        pytest.param("../o:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    ],
)
def test_write_failure_is_named_as_typed(output):
    result = _emit("--output", output)

    assert result.exit_code == 2
    assert f"Write failed: output '{output}' must stay under cwd" in _plain(result)


def test_write_failure_strips_bidi_controls_from_the_display():
    result = _emit("--output", f"../x[bold]o{_CONTROLS}")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Write failed: output '../x[bold]o" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "output",
    ["../x[red]o", pytest.param("../o:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code")],
)
def test_write_failure_of_both_formats_is_named_as_typed(output):
    result = _emit("--format", "both", "--output", output)

    assert result.exit_code == 2
    assert f"Write failed: output '{output}.cdx.json' must stay under cwd" in _plain(result)


def test_write_failure_of_both_formats_strips_bidi_controls_from_the_display():
    result = _emit("--format", "both", "--output", f"../x[bold]o{_CONTROLS}")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Write failed: output '../x[bold]o" in text
    assert _not_shown(text)


def _written(text: str, label: str, name: str) -> bool:
    return re.search(rf"{re.escape(label)} -> \S*{re.escape(name)}", text) is not None


@pytest.mark.parametrize(
    "name",
    [
        "x[red]o.json",
        "x[bold]o.json",
        pytest.param("o:warning:.json", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    ],
)
def test_written_bom_path_is_named_as_typed(name):
    result = _emit("--output", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _written(_plain(result), "Wrote BOM (cyclonedx)", name)


def test_written_bom_path_strips_controls_from_the_display():
    result = _emit("--output", f"x[bold]o{_CONTROLS}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _written(text, "Wrote BOM (cyclonedx)", "x[bold]o.json")
    assert _not_shown(text)


@pytest.mark.parametrize(
    "prefix",
    ["x[red]o", pytest.param("o:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code")],
)
def test_written_both_bom_paths_are_named_as_typed(prefix):
    result = _emit("--format", "both", "--output", prefix)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _written(text, "Wrote CycloneDX BOM", f"{prefix}.cdx.json")
    assert _written(text, "Wrote SPDX BOM", f"{prefix}.spdx.json")


def test_written_both_bom_paths_strip_controls_from_the_display():
    result = _emit("--format", "both", "--output", f"x[bold]o{_CONTROLS}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _written(text, "Wrote CycloneDX BOM", "x[bold]o.cdx.json")
    assert _written(text, "Wrote SPDX BOM", "x[bold]o.spdx.json")
    assert _not_shown(text)


@pytest.mark.parametrize("ref", ["x[red]r", "x[/]r", "r:warning:", f"x[bold]r{_CONTROLS}"])
def test_attach_failure_names_the_reference_as_typed(ref):
    result = _emit("--output", "bom.json", "--attach-to-registry", ref)

    assert result.exit_code == 1
    text = _plain(result)
    shown = ref.replace(_CONTROLS, "")
    assert f"could not attach to registry: registry entry not found: {shown}" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "name",
    [
        "x[red]o.json",
        pytest.param("o:warning:.json", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
        f"x[bold]o{_CONTROLS}.json",
    ],
)
def test_attached_bom_path_is_named_as_typed(name):
    with RegistryStore() as store:
        entry_id = store.push(
            name="phi", tag="v1", base_model="org/model", task="sft", run_id=None,
            config={"base": "org/model", "task": "sft"}, notes=None,
        )

    result = _emit("--output", name, "--attach-to-registry", entry_id)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(
        rf"Attached bom to registry entry \S+ \(\S*{re.escape(name.replace(_CONTROLS, ''))}\)", text
    )
    assert _not_shown(text)
