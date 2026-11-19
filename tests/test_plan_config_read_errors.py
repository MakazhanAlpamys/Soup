"""Unreadable preflight inputs get a useful refusal and never change saved state."""

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import plan
from tests.conftest import strip_ansi

COMMANDS = [["plan"], ["apply"], ["apply", "--dry-run"]]


def _plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plan, "console", Console(width=180))
    return tmp_path


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("existing_state", [False, True])
@pytest.mark.parametrize(
    ("content", "reason"),
    [(b"base: [\n", "not valid YAML"),
     (b"base: !!python/object/apply:os.system []\n", "not valid YAML"),
     (b"base: \xff\n", "not valid UTF-8")],
)
def test_bad_config_file_is_named_and_state_is_unchanged(
    workdir, command, existing_state, content, reason
):
    config = Path("[bold]bad[end].yaml")
    config.write_bytes(content)
    state = Path("soup.tfstate")
    before = b"operator's existing state\n"
    if existing_state:
        state.write_bytes(before)

    result = CliRunner().invoke(app, [*command, "--config", str(config)])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert str(config) in text and reason in text
    assert "Traceback" not in text
    assert config.read_bytes() == content
    if existing_state:
        assert state.read_bytes() == before
    else:
        assert not state.exists()
    if reason == "not valid YAML":
        assert "line 1" in text or "line 2" in text
    else:
        assert "byte 6" in text


@pytest.mark.parametrize("command", COMMANDS)
def test_a_directory_config_is_reported_before_reading_state(workdir, command):
    Path("folder.yaml").mkdir()
    result = CliRunner().invoke(app, [*command, "--config", "folder.yaml"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "folder.yaml" in _plain(result) and "cannot be read" in _plain(result)
    assert not Path("soup.tfstate").exists()


@pytest.mark.parametrize("command", COMMANDS)
def test_read_error_reports_the_os_reason_without_terminal_controls(workdir, monkeypatch, command):
    Path("soup.yaml").write_text("base: unused\n", encoding="utf-8")
    original = b"state must stay intact"
    Path("soup.tfstate").write_bytes(original)

    def denied(*args, **kwargs):
        raise PermissionError(13, "[bold]denied\x1b\x9b\u202e[/]", "soup.yaml")

    monkeypatch.setattr(plan, "load_yaml_file_limited", denied)
    result = CliRunner().invoke(app, [*command])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = strip_ansi(result.output)
    assert "soup.yaml" in text and "[bold]denied[/]" in text
    assert not any(char in text for char in ("\x1b", "\x9b", "\u202e"))
    assert Path("soup.tfstate").read_bytes() == original


@pytest.mark.parametrize("command", COMMANDS)
def test_missing_config_keeps_its_exit_one_and_hint(workdir, command):
    result = CliRunner().invoke(app, [*command, "--config", "missing.yaml"])
    assert result.exit_code == 1
    assert "Config not found: missing.yaml" in _plain(result)
    assert not Path("soup.tfstate").exists()


@pytest.mark.parametrize("command", COMMANDS)
def test_unsupported_yaml_tag_is_bounded_without_echoing_the_document(workdir, command):
    Path("soup.yaml").write_text("base: !" + "a" * 10000 + " value\n", encoding="utf-8")
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert "not valid YAML" in text and "soup.yaml" in text
    assert "line 1" in text
    assert len(text) < 500
    assert not Path("soup.tfstate").exists()
