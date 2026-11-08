"""soup spectrum scan prints the model, layer names and errors as typed, not as Rich markup."""

import re
import sys

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import spectrum_scan as scan_utils
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n.yaml",
    pytest.param("n:warning:.yaml", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.yaml",
]


class _OddError(Exception):
    """An exception type the command does not name, like a hub library's own."""


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.spectrum.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _layers(group="mlp.up_proj", stem="layers"):
    return tuple(
        scan_utils.LayerSNR(
            name=f"{stem}.{index}.proj.weight", module_type="mlp", group=group,
            snr=1.0 + index, shape=(2, 2),
        )
        for index in range(2)
    )


def _stub_scan(monkeypatch, *, model="org/model", group="mlp.up_proj", stem="layers"):
    result = scan_utils.ScanResult(model=model, modules="all", layers=_layers(group, stem))
    monkeypatch.setattr(scan_utils, "scan_model", lambda *args, **kwargs: result)


def _scan(*args: str):
    return runner.invoke(app, ["spectrum", "scan", "--model", "org/model", *args])


@pytest.mark.parametrize("value", _VALUES)
def test_the_table_title_names_the_model_as_typed(monkeypatch, value):
    _stub_scan(monkeypatch, model=value)

    result = _scan()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Spectrum SNR \u2014 {_shown(value)} (top 50% per group)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_group_column_shows_a_layer_group_as_typed(monkeypatch, value):
    _stub_scan(monkeypatch, group=value)

    result = _scan()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"mlp \W+ " + re.escape(_shown(value)) + r" \W+ 2 \W+ 1 ", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_yaml_block_shows_a_layer_name_as_typed(monkeypatch, value):
    _stub_scan(monkeypatch, stem=f"layers.{value}")

    result = _scan("--top-percent", "100")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    prefix = f"layers.{value}.0.proj"
    # The block is YAML: it writes a control character as an escape, so compare with its text.
    expected = " ".join(yaml.safe_dump([prefix], default_flow_style=False).split())
    assert expected in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_cwd_is_shown_as_typed(name):
    result = _scan("-o", f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    quoted = quote_path(f"../{name}").replace("\u202e", "")
    assert f"output {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, FileNotFoundError, RuntimeError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_scan_is_shown_as_typed(monkeypatch, error, value):
    def fail(*args, **kwargs):
        raise error(f"scan said {value}")

    monkeypatch.setattr(scan_utils, "scan_model", fail)

    result = _scan()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spectrum scan failed: scan said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_unexpected_scan_error_is_shown_as_typed(monkeypatch, value):
    def fail(*args, **kwargs):
        raise _OddError(f"hub said {value}")

    monkeypatch.setattr(scan_utils, "scan_model", fail)

    result = _scan()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Spectrum scan failed: _OddError: hub said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refused_selection_is_shown_as_typed(monkeypatch, value):
    _stub_scan(monkeypatch)

    def refuse(*args, **kwargs):
        raise ValueError(f"selection said {value}")

    monkeypatch.setattr(scan_utils, "select_unfrozen_parameters", refuse)

    result = _scan()

    assert result.exit_code == 2
    text = _plain(result)
    assert f"selection said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_written_patch_is_named_as_typed(monkeypatch, name):
    _stub_scan(monkeypatch)

    result = _scan("-o", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Wrote patch: {_shown(name)}" in text
    assert _not_shown(text)


def test_the_table_title_keeps_its_italic_style(monkeypatch):
    _stub_scan(monkeypatch)
    monkeypatch.setattr(
        "soup_cli.commands.spectrum.console",
        Console(width=500, force_terminal=True, color_system="standard", legacy_windows=False),
    )

    result = _scan()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Italic is SGR 3.
    assert re.search(r"\x1b\[3m\s*Spectrum SNR", result.output)
