"""``soup can pack`` / ``soup can fork``: credentials are not written into a can.

``pack_entry`` and ``fork_can`` run the config (and the description) through the same
redaction helper as the crash bundle (``utils/redaction.py``) before anything is
packed, and list the KEYS that were changed on the console: key paths only, never a
value. The produced archive is opened and scanned in each test.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cans import unpack
from soup_cli.cans.pack import fork_can, pack_entry
from soup_cli.registry.store import RegistryStore
from soup_cli.utils.redaction import REDACTED
from tests.conftest import strip_ansi
from tests.test_crash_bundle_redaction import HF, ROW_IDS, ROWS, URL_SECRET

BASE = {"base": "llama-3.1-8b", "task": "sft", "training": {"lr": 2e-5, "max_new_tokens": 128}}


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
    return tmp_path


def _push(config: dict, notes: str = "demo") -> str:
    store = RegistryStore()
    try:
        return store.push(
            name="recipe", tag="v1", base_model="llama-3.1-8b", task="sft", run_id=None,
            config=config, notes=notes,
        )
    finally:
        store.close()


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=240, force_terminal=False, color_system=None), buffer


def _members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read() for member in tar.getmembers()
            if member.isfile()
        }


def _archive_text(path: Path) -> str:
    return "\n".join(data.decode("utf-8") for data in _members(path).values())


def _write_can(path: Path, config: dict, description: str = "old can") -> None:
    manifest = {
        "can_format_version": 3, "name": "old", "author": "someone",
        "created_at": "2026-01-01T00:00:00", "base_hash": "abc", "description": description,
        "tags": [], "deploy_targets": [], "attestations": [],
    }
    with tarfile.open(path, mode="w:gz") as tar:
        for name, content in (
            ("manifest.yaml", yaml.safe_dump(manifest)),
            ("config.yaml", yaml.safe_dump(config)),
            ("data_ref.yaml", yaml.safe_dump({"kind": "local"})),
        ):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# pack_entry
# ---------------------------------------------------------------------------


def test_pack_entry_replaces_credentials_and_lists_the_key_paths(workdir):
    config = {
        **BASE,
        "data": {
            "train": f"https://user:{URL_SECRET}@data.example/t.jsonl?api_key={URL_SECRET}",
            "hf_token": "plainvalue-1",
            "sources": [{"path": "a.jsonl", "api_key": "plainvalue-2"}],
        },
    }
    console, buffer = _console()
    out = pack_entry(entry_id=_push(config), out_path=str(workdir / "r.can"), console=console)

    packed = unpack.read_config(str(out))
    assert packed["data"]["hf_token"] == REDACTED
    assert packed["data"]["sources"] == [{"path": "a.jsonl", "api_key": REDACTED}]
    assert packed["data"]["train"] == (
        f"https://{REDACTED}@data.example/t.jsonl?api_key={REDACTED}"
    )
    assert packed["training"] == BASE["training"] and packed["base"] == BASE["base"]

    text = _archive_text(out)
    for secret in ("plainvalue-1", "plainvalue-2", URL_SECRET):
        assert secret not in text, secret

    shown = " ".join(buffer.getvalue().split())
    assert "Redacted 4 credential value(s) before packing, keys changed:" in shown
    # the registry stores the config with sorted keys
    assert "data.hf_token, data.sources[0].api_key, data.train" in shown
    for secret in ("plainvalue-1", "plainvalue-2", URL_SECRET):
        assert secret not in shown, secret


def test_a_config_without_credentials_is_packed_as_it_was_and_prints_nothing(workdir):
    console, buffer = _console()
    out = pack_entry(entry_id=_push(BASE), out_path=str(workdir / "r.can"), console=console)
    assert unpack.read_config(str(out)) == BASE
    assert unpack.inspect_can(str(out)).description == "demo"
    assert buffer.getvalue() == ""


def test_description_and_notes_are_redacted_too(workdir):
    console, buffer = _console()
    out = pack_entry(
        entry_id=_push(BASE, notes=f"trained with {HF}"), out_path=str(workdir / "r.can"),
        console=console,
    )
    assert HF not in _archive_text(out)
    assert unpack.inspect_can(str(out)).description == f"trained with {REDACTED}"
    assert "keys changed: description" in " ".join(buffer.getvalue().split())

    explicit = pack_entry(
        entry_id=_push(BASE), out_path=str(workdir / "e.can"), description=f"key {HF}",
        console=_console()[0],
    )
    assert HF not in _archive_text(explicit)


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_packed_can_carries_no_credential_from_the_spelling_table(workdir, row):
    config = {
        **BASE,
        "notes": row.positive,
        "nested": {"items": [row.positive, {"text": row.positive}]},
        "hf_token": row.secret,
    }
    out = pack_entry(
        entry_id=_push(config, notes=row.positive), out_path=str(workdir / "r.can"),
        description=row.positive, console=_console()[0],
    )
    members = _members(out)
    assert set(members) == {"manifest.yaml", "config.yaml", "data_ref.yaml", "recipe.md"}
    for name, data in members.items():
        assert row.secret.encode() not in data, name
        assert row.secret.replace("\n", "\\n").encode() not in data, name
    assert unpack.read_config(str(out))["training"] == BASE["training"]


def test_listing_is_capped_but_the_count_covers_every_value(workdir):
    config = {**BASE, "keys": {f"api_key_{index:02d}": f"plain-{index}" for index in range(30)}}
    console, buffer = _console()
    pack_entry(entry_id=_push(config), out_path=str(workdir / "r.can"), console=console)
    shown = " ".join(buffer.getvalue().split())
    assert "Redacted 30 credential value(s)" in shown
    assert "keys.api_key_00" in shown and "keys.api_key_19" in shown
    assert "keys.api_key_20" not in shown
    assert "(+10 more)" in shown
    assert "plain-" not in shown


def test_markup_in_a_key_path_is_printed_literally(workdir):
    config = {**BASE, "[red]api_key[/red]": "plainvalue-9"}
    console, buffer = _console()
    pack_entry(entry_id=_push(config), out_path=str(workdir / "r.can"), console=console)
    assert "[red]api_key[/red]" in buffer.getvalue()
    assert "plainvalue-9" not in buffer.getvalue()


def test_a_control_character_in_a_key_path_never_reaches_the_terminal(workdir):
    config = {**BASE, "api_key]0;titlex": "plainvalue-9"}
    console, buffer = _console()
    pack_entry(entry_id=_push(config), out_path=str(workdir / "r.can"), console=console)
    shown = buffer.getvalue()
    assert "" not in shown and "" not in shown
    assert "plainvalue-9" not in shown
    assert "keys changed: api_key" in " ".join(shown.split())


def test_a_key_that_is_itself_a_credential_is_not_printed(workdir):
    config = {**BASE, "mirrors": {HF: f"https://u:{URL_SECRET}@h.example/x"}}
    console, buffer = _console()
    pack_entry(entry_id=_push(config), out_path=str(workdir / "r.can"), console=console)
    shown = " ".join(buffer.getvalue().split())
    assert HF not in shown and URL_SECRET not in shown
    assert f"keys changed: mirrors.{REDACTED}" in shown


def test_pack_cli_lists_the_keys_that_were_changed(workdir):
    from soup_cli.cli import app

    eid = _push({**BASE, "hf_token": "plainvalue-cli"})
    result = CliRunner().invoke(app, ["can", "pack", "--entry-id", eid, "--out", "r.can"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    shown = " ".join(strip_ansi(result.output).split())
    assert "Redacted 1 credential value(s) before packing, keys changed: hf_token" in shown
    assert "plainvalue-cli" not in _archive_text(workdir / "r.can")


def test_pack_cli_prints_nothing_extra_when_there_is_nothing_to_redact(workdir):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["can", "pack", "--entry-id", _push(BASE), "--out", "r.can"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Redacted" not in strip_ansi(result.output)


# ---------------------------------------------------------------------------
# fork_can
# ---------------------------------------------------------------------------


def test_fork_cli_lists_the_keys_that_were_changed(workdir):
    from soup_cli.cli import app

    _write_can(workdir / "old.can", BASE)
    result = CliRunner().invoke(app, [
        "can", "fork", "old.can", "--out", "f.can", "--modify", "training.hf_token=plainvalue-cli",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    shown = " ".join(strip_ansi(result.output).split())
    assert "Redacted 1 credential value(s) before packing, keys changed: training.hf_token" in shown
    assert "plainvalue-cli" not in shown
    assert "plainvalue-cli" not in _archive_text(workdir / "f.can")


def test_fork_redacts_a_credential_set_by_a_modification(workdir):
    source = pack_entry(entry_id=_push(BASE), out_path=str(workdir / "r.can"))
    console, buffer = _console()
    forked = fork_can(
        source=str(source), out_path=str(workdir / "f.can"), console=console,
        modifications=[
            "training.hf_token=plainvalue-fork", "training.lr=5e-5",
            f"data.train=https://u:{URL_SECRET}@h.example/t.jsonl",
        ],
    )
    config = unpack.read_config(str(forked))
    assert config["training"]["hf_token"] == REDACTED and config["training"]["lr"] == 5e-5
    assert config["data"]["train"] == f"https://{REDACTED}@h.example/t.jsonl"
    text = _archive_text(forked)
    assert "plainvalue-fork" not in text and URL_SECRET not in text
    shown = " ".join(buffer.getvalue().split())
    assert "Redacted 2 credential value(s) before packing, keys changed:" in shown
    assert "training.hf_token" in shown and "data.train" in shown
    assert "plainvalue-fork" not in shown and URL_SECRET not in shown


def test_fork_cleans_a_can_that_already_carries_a_credential(workdir):
    _write_can(workdir / "old.can", {**BASE, "api_key": "plainvalue-old", "note": HF},
               description=f"made with {HF}")
    console, buffer = _console()
    forked = fork_can(
        source=str(workdir / "old.can"), out_path=str(workdir / "f.can"), modifications=[],
        console=console,
    )
    text = _archive_text(forked)
    assert "plainvalue-old" not in text and HF not in text
    assert unpack.inspect_can(str(forked)).description == f"made with {REDACTED}"
    shown = " ".join(buffer.getvalue().split())
    assert "Redacted 3 credential value(s)" in shown
    assert "api_key, note, description" in shown


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_forked_can_carries_no_credential_from_the_spelling_table(workdir, row):
    _write_can(workdir / "old.can", {**BASE, "notes": row.positive}, description=row.positive)
    forked = fork_can(
        source=str(workdir / "old.can"), out_path=str(workdir / "f.can"),
        modifications=[f"extra.value={json.dumps(row.positive)}"], console=_console()[0],
    )
    for name, data in _members(forked).items():
        assert row.secret.encode() not in data, name
        assert row.secret.replace("\n", "\\n").encode() not in data, name


def test_fork_without_credentials_prints_nothing_and_keeps_the_config(workdir):
    source = pack_entry(entry_id=_push(BASE), out_path=str(workdir / "r.can"))
    console, buffer = _console()
    forked = fork_can(
        source=str(source), out_path=str(workdir / "f.can"), console=console,
        modifications=["training.lr=5e-5"],
    )
    assert unpack.read_config(str(forked))["training"] == {"lr": 5e-5, "max_new_tokens": 128}
    assert buffer.getvalue() == ""
