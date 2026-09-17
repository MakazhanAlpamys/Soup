"""soup can prints manifest fields without control bytes."""

from __future__ import annotations

import io
import tarfile

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# ``name`` is held to ``[A-Za-z0-9_.-]`` by the schema, so a control byte there
# refuses the whole manifest; the other printed fields accept one.
MANIFEST = (
    'can_format_version: 1\nname: "n"\nauthor: "a\\e[3A\\e[2K"\n'
    'created_at: "2026-01-01"\nbase_hash: "h\\e]8;;http://x\\e\\\\"\n'
    'tags: ["t\\e[31m"]\ndescription: "\\e]0;TITLE\\e\\\\"\n'
)

BAD_NAME_MANIFEST = (
    'can_format_version: 1\nname: "n\\e[2Kx"\nauthor: "a"\n'
    'created_at: "2026-01-01"\nbase_hash: "h"\n'
)

BAD_DATE_MANIFEST = (
    'can_format_version: 1\nname: "n"\nauthor: "a"\n'
    'created_at: "\\e]0;WHEN\\e\\\\"\nbase_hash: "h"\n'
)


def _write_can(path, text, config=None):
    with tarfile.open(path, "w:gz") as tf:
        members = [("manifest.yaml", text)]
        if config is not None:
            members.append(("config.yaml", config))
        for name, body in members:
            payload = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture
def can_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return _write_can("c.can", MANIFEST, config="base: m\n")


def test_manifest_fixture_carries_escape_bytes(can_file):
    from soup_cli.cans.unpack import inspect_can

    manifest = inspect_can(can_file)
    assert "\x1b" in manifest.author
    assert "\x1b" in manifest.base_hash
    assert "\x1b" in manifest.tags[0]
    assert "\x1b" in manifest.description


def test_inspect_output_has_no_escape_bytes(can_file):
    from soup_cli.cli import app

    result = runner.invoke(app, ["can", "inspect", can_file])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b" not in result.output, repr(result.output)
    assert "TITLE" in result.output


def test_run_confirmation_has_no_escape_bytes(can_file):
    from soup_cli.cli import app

    result = runner.invoke(app, ["can", "run", can_file])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "--yes" in result.output
    assert "\x1b" not in result.output, repr(result.output)


def test_inspect_error_has_no_escape_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cli import app

    _write_can("bad.can", BAD_NAME_MANIFEST)
    result = runner.invoke(app, ["can", "inspect", "bad.can"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Cannot inspect can" in result.output
    assert "\x1b" not in result.output, repr(result.output)


def test_verify_message_has_no_escape_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.cli import app

    _write_can("bad.can", BAD_DATE_MANIFEST, config="base: m\n")
    result = runner.invoke(app, ["can", "verify", "bad.can"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Verify failed" in result.output
    assert "WHEN" in result.output
    assert "\x1b" not in result.output, repr(result.output)
