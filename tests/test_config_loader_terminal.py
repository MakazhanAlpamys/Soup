"""Loader diagnostics are literal; files, API errors and configuration stay intact."""

import copy
import io
import sys
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, RootModel, ValidationError, field_validator, model_validator
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config import loader
from soup_cli.config.deprecation import SoupConfigDeprecationWarning
from soup_cli.config.unknown_keys import find_unknown_config_keys, format_unknown_keys
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary", "ordinary", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("[red]name", "[red]name", id="markup"),
    pytest.param(r"name\[unfinished", r"name\[unfinished", id="backslash"),
    pytest.param("name\x1b\x07\x9b\u202e[end]", "name[end]", id="controls"),
]


def _plain(text):
    text = strip_ansi(text)
    assert not any(char in text for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


@pytest.fixture
def printed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    monkeypatch.setattr(
        loader, "console",
        Console(
            file=stream, width=500, force_terminal=True, color_system="standard", no_color=False
        ),
    )
    return stream


def _document():
    return {"base": "org/model", "task": "sft", "data": {"train": "train.jsonl"},
            "training": {"epochs": 1}}


def _path(value):
    # C0 bytes cannot name files on Windows. C1/bidi remain in the real path;
    # raw C0 bytes are exercised in diagnostic values and exception messages.
    return Path("cfg-" + value.replace("\x1b", "").replace("\x07", "") + ".yaml")


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["directory", "utf8", "yaml", "os-error"])
def test_file_refusals_keep_paths_literal_and_preserve_input(
    printed, monkeypatch, kind, value, shown
):
    if ":" in value:
        # Only real filenames need the platform restriction.
        if sys.platform == "win32":
            pytest.skip("':' is not valid in a Windows filename")
    path = _path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"Config file {_path(shown)}"
    body = {"utf8": b"base: \xff\n", "yaml": b"base: [\n", "os-error": b"base: org/model\n"}
    if kind == "directory":
        path.mkdir()
        reason = "is a directory, not a file"
        original = None
    else:
        original = body[kind]
        path.write_bytes(original)
        reason = {
            "utf8": "is not valid UTF-8: invalid start byte at byte 6",
            "yaml": "is not valid YAML: expected the node content, but found '<stream end>' "
                    "(line 2, column 1)",
            "os-error": f"cannot be read: Permission denied {shown}",
        }[kind]
    if kind == "os-error":
        def denied(*args, **kwargs):
            raise PermissionError(13, f"Permission denied {value}")

        monkeypatch.setattr(loader, "load_yaml_file_limited", denied)

    with pytest.raises(SystemExit) as error:
        loader.load_config(path)

    assert error.value.code == 1
    assert _plain(printed.getvalue()) == f"{prefix} {reason}"
    assert "\x1b[31m" in printed.getvalue()
    if original is None:
        assert path.is_dir() and not list(path.iterdir())
    else:
        assert path.read_bytes() == original


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("severity", ["warn", "error"])
def test_unknown_key_reports_keep_original_mapping_and_api_reason(
    printed, monkeypatch, value, shown, severity
):
    raw = _document()
    raw["training"][f"bad-{value}"] = 1
    before = copy.deepcopy(raw)
    monkeypatch.setattr(loader, "UNKNOWN_KEY_SEVERITY", severity)
    reason = format_unknown_keys(find_unknown_config_keys(raw), include_deadline=severity == "warn")
    display = reason.replace(value, shown)

    if severity == "warn":
        result = loader.validate_config_mapping(raw)
        assert result.base == raw["base"] and result.training.epochs == 1
        text = _plain(printed.getvalue())
        assert text.startswith("Warning: " + " ".join(display.split()))
        assert "An unapplied key is ignored, not defaulted" in text
        assert "\x1b[33mWarning:" in printed.getvalue()
    else:
        with pytest.raises(SystemExit) as error:
            loader.validate_config_mapping(raw)
        assert error.value.code == 1
        assert _plain(printed.getvalue()) == (
            "Config validation error: " + " ".join(display.split())
        )
        with pytest.raises(ValueError) as api:
            loader.config_from_mapping(raw)
        assert str(api.value) == reason
    assert raw == before


@pytest.mark.parametrize("value,shown", VALUES)
def test_staged_warning_is_literal_and_preserves_formatter_inputs(
    printed, monkeypatch, value, shown
):
    raw = _document()
    config = loader.SoupConfig(**raw)
    before = copy.deepcopy(raw)
    findings = [object()]
    monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")

    def find(received, *, config):
        assert received is raw
        assert config.base == "org/model"
        return findings

    def format_message(received, *, include_deadline):
        assert received is findings and include_deadline is True
        return f"Staged {value}"

    monkeypatch.setattr(loader, "find_staged_config_fields", find)
    monkeypatch.setattr(loader, "format_staged_fields", format_message)
    assert loader._report_staged_fields(raw, config=config) is None
    assert _plain(printed.getvalue()) == f"Warning: Staged {shown}"
    assert "\x1b[33mWarning:" in printed.getvalue()
    assert raw == before


@pytest.mark.parametrize("value,shown", VALUES)
def test_deprecations_keep_deduplication_and_foreign_warning_values(
    printed, monkeypatch, value, shown
):
    raw = {"base": value}
    received = []
    result = object()

    def build(**kwargs):
        received.append(kwargs)
        warnings.warn(f"Deprecated {value}", SoupConfigDeprecationWarning)
        warnings.warn(f"Deprecated {value}", SoupConfigDeprecationWarning)
        warnings.warn(f"Foreign {value}", UserWarning)
        return result

    monkeypatch.setattr(loader, "SoupConfig", build)
    with pytest.warns(UserWarning) as foreign:
        assert loader._build_config(raw) is result
        assert loader._build_config(raw) is result

    assert [str(item.message) for item in foreign] == [f"Foreign {value}"] * 2
    assert received == [raw, raw] and raw == {"base": value}
    assert _plain(printed.getvalue()) == f"Warning: Deprecated {shown} Warning: Deprecated {shown}"
    assert strip_ansi(printed.getvalue()).splitlines() == [f"Warning: Deprecated {shown}"] * 2
    assert "\x1b[33mWarning:" in printed.getvalue()


class _FieldRefusal(BaseModel):
    base: str

    @field_validator("base")
    @classmethod
    def refuse(cls, value):
        raise ValueError(f"Cannot use {value}")


class _ModelRefusal(BaseModel):
    base: str

    @model_validator(mode="after")
    def refuse(self):
        raise ValueError(f"Cannot use {self.base}")


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["location", "field-message", "model-message"])
def test_pydantic_errors_are_literal_while_api_diagnostics_stay_original(
    printed, monkeypatch, value, shown, kind
):
    raw = {"base": value}

    def build(received):
        assert received == raw
        if kind == "location":
            return RootModel[dict[str, int]].model_validate({received["base"]: "not-an-int"})
        model = _FieldRefusal if kind == "field-message" else _ModelRefusal
        return model.model_validate(received)

    with pytest.raises(ValidationError) as reference:
        build(raw)
    entry = reference.value.errors()[0]
    loc = " -> ".join(str(part) for part in entry["loc"])
    original_api = f"{loc}: {entry['msg']}"
    if kind == "location":
        displayed = f"  {shown}: {entry['msg']}"
    else:
        displayed = ("  base: " if kind == "field-message" else "  ") + f"Cannot use {shown}"
    monkeypatch.setattr(loader, "_build_config", build)

    with pytest.raises(SystemExit) as cli:
        loader.validate_config_mapping(raw)
    assert cli.value.code == 1
    assert _plain(printed.getvalue()) == "Config validation error: " + displayed.strip()
    lines = strip_ansi(printed.getvalue()).splitlines()
    assert lines[:2] == ["Config validation error:", ""]
    assert lines[2].startswith("  ")
    if kind == "field-message":
        assert "\x1b[31mbase:\x1b[0m" in printed.getvalue()
    with pytest.raises(ValueError) as api:
        loader.load_config_from_string(yaml.safe_dump(raw))
    assert str(api.value) == original_api
    assert raw == {"base": value}


@pytest.mark.parametrize(
    "command,exit_code", [(["train", "--dry-run"], 1), (["doctor"], 2)],
)
def test_real_cli_prints_literal_unknown_key_without_changing_config(
    tmp_path, monkeypatch, command, exit_code
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loader, "console", Console(width=300))
    raw = _document()
    raw["training"][r"name\[unfinished:rocket:"] = True
    path = Path("soup.yaml")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    before = path.read_bytes()

    result = CliRunner().invoke(app, [*command, "--config", str(path)])

    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    assert r"training.name\[unfinished:rocket:" in _plain(result.output)
    assert "Config validation error:" in _plain(result.output)
    assert path.read_bytes() == before


def test_raise_unreadable_keeps_the_raw_exception_and_silent_output(printed):
    path = Path("cfg-[red]name.yaml")
    path.write_bytes(b"base: \xff\n")
    with pytest.raises(UnicodeDecodeError) as error:
        loader.load_config(path, raise_unreadable=True)
    assert error.value.object == b"base: \xff\n" and error.value.start == 6
    assert printed.getvalue() == ""
