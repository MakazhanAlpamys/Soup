"""Literal text in the public can commands, with the original styles and arguments."""

import io
import tarfile
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from soup_cli.cans import pack, publish
from soup_cli.cans import run as can_run
from soup_cli.cans.unpack import inspect_can
from soup_cli.cli import app
from soup_cli.commands import can
from tests.conftest import make_cli_runner, strip_ansi

RAW = r"label:warning:[red]\[end"
STORED = RAW + "\x9b\u202e"
MISSING = r"missing-[red]\[end.can"


@pytest.fixture(autouse=True)
def cwd_and_console(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        can,
        "console",
        Console(
            width=500,
            height=100,
            force_terminal=True,
            no_color=False,
            color_system="standard",
            legacy_windows=False,
        ),
    )


def run(*args):
    return make_cli_runner().invoke(app, ["can", *args])


@pytest.fixture
def archive():
    with tarfile.open("item.can", "w:gz") as tf:
        members = {
            "manifest.yaml": yaml.safe_dump(
                {
                    "can_format_version": 1,
                    "name": "demo",
                    "author": RAW + "\u202e",
                    "created_at": "2026-01-01",
                    "base_hash": "hash",
                    "tags": [STORED],
                    "description": STORED,
                }
            ),
            "config.yaml": "base: model\n",
        }
        for name, body in members.items():
            data = body.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return "item.can"


def test_inspect_reads_real_archive_metadata(archive):
    result = run("inspect", archive)
    assert result.exit_code == 0, (result.output, result.exception)
    plain = strip_ansi(result.output)
    assert "Can: demo" in plain and plain.count(RAW) == 3
    assert "\x9b" not in plain and "\u202e" not in plain
    assert "\x1b[1;36mauthor" in result.output
    manifest = inspect_can(archive)
    assert manifest.author == RAW + "\u202e"
    assert manifest.description == STORED and manifest.tags == [STORED]


def test_run_confirmation_keeps_metadata_and_indentation(archive, monkeypatch):
    monkeypatch.setattr(
        can_run, "run_can", lambda *a, **k: pytest.fail("confirmation must not train")
    )
    result = run("run", archive)
    assert result.exit_code == 1, (result.output, result.exception)
    rows = [
        line.removeprefix("│ ").removesuffix("│").rstrip()
        for line in strip_ansi(result.output).splitlines()
        if line.startswith("│")
    ]
    assert rows == [
        "\x60soup can run\x60 will:",
        "  - Extract the can",
        "  - Auto-fetch any data referenced inside",
        "  - Run soup train against the embedded config",
        "",
        "Recipe: demo",
        f"Author: {RAW}",
        "",
        "Pass --yes to confirm.",
    ]
    for styled in (
        "\x1b[33m\x60soup can run\x60 will:\x1b[0m",
        "\x1b[1msoup train\x1b[0m",
        "\x1b[1mdemo\x1b[0m",
        "\x1b[1m--yes\x1b[0m",
    ):
        assert styled in result.output


@pytest.mark.parametrize("success", [True, False])
def test_verify_receipts(archive, success):
    result = run("verify", archive if success else MISSING + "\u202e")
    if success:
        assert result.exit_code == 0
        assert "\x1b[32mOK:\x1b[0m OK (format v1, name=demo)" in result.output
    else:
        assert result.exit_code == 1
        assert f"\x1b[31mVerify failed:\x1b[0m can not found: {MISSING}" in result.output


def test_inspect_missing_file_is_literal():
    result = run("inspect", MISSING + "\u202e")
    assert result.exit_code == 1, (result.output, result.exception)
    assert f"\x1b[31mcan not found: {MISSING}\x1b[0m" in result.output


@pytest.mark.parametrize("code", [0, 7])
def test_run_result_receipt_is_literal(monkeypatch, code):
    path = Path(STORED)
    monkeypatch.setattr(can_run, "run_can", lambda *a, **k: can_run.CanRunResult(path, code))
    result = run("run", "item.can", "--yes")
    assert result.exit_code == code, (result.output, result.exception)
    shown = str(path).replace("\x9b\u202e", "")
    if code:
        assert f"\x1b[31mtrain failed (rc=7). Extract dir: {shown}\x1b[0m" in result.output
    else:
        assert (
            f"\x1b[32mCan run complete\x1b[0m - extract dir: \x1b[1m{shown}\x1b[0m" in result.output
        )


@pytest.mark.parametrize("name", ["pack", "fork", "publish"])
def test_artifact_receipts_are_literal(name, monkeypatch):
    if name == "pack":

        def pack_entry(*, entry_id, out_path, **kwargs):
            assert (entry_id, out_path) == (STORED, "out.can")
            return STORED

        monkeypatch.setattr(pack, "pack_entry", pack_entry)
        args = ["--entry-id", STORED, "-o", "out.can"]
        expected = f"Packed \x1b[36m{RAW}\x1b[0m -> \x1b[1m{RAW}\x1b[0m"
    elif name == "fork":
        monkeypatch.setattr(pack, "fork_can", lambda **kwargs: STORED)
        args = ["item.can", "-o", "out.can"]
        expected = f"\x1b[32mForked\x1b[0m -> \x1b[1m{RAW}\x1b[0m"
    else:
        monkeypatch.setattr(publish, "publish_can", lambda *args, **kwargs: STORED)
        args = ["item.can", "--hf-hub", "owner/item"]
        expected = f"\x1b[32mPublished\x1b[0m -> \x1b[1m{RAW}\x1b[0m"
    result = run(name, *args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert expected in result.output
