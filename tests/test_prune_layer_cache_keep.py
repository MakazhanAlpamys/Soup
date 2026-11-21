"""Explicit revision pins protect only matching cache slots, before deletion."""

import json
import os

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import cache
from soup_cli.utils.layer_cache_slots import list_base_slots
from tests.conftest import strip_ansi

OLD = "a" * 40
MID = "b" * 40
NEW = "c" * 40
runner = CliRunner()


def _slot(root, name, *, age=100):
    path = root / name
    path.mkdir()
    index = path / "index.json"
    index.write_text(json.dumps({
        "n_layers": 1, "layer_keys": [], "extra_keys": [], "dtype": "bf16",
        "total_params": 1,
    }))
    os.utime(index, (age, age))
    return path


@pytest.fixture
def slots(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_current", lambda *a: "org__m@current")
    paths = {}
    for sha, age in [(OLD, 100), (MID, 200), (NEW, 300)]:
        paths[sha] = _slot(tmp_path, "org__m@" + sha, age=age)
    return tmp_path, paths


def invoke(root, *args):
    result = runner.invoke(app, [
        "prune-layer-cache", "--model", "org/m", "--cache-dir", str(root), *args,
    ])
    return result, " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("yes", [False, True])
def test_old_pin_keeps_only_exact_revision(slots, yes):
    root, paths = slots
    other = _slot(root, "other__m@" + OLD, age=100)
    result, output = invoke(root, "--keep", OLD, *(["--yes"] if yes else []))
    assert result.exit_code == 0, output
    assert paths[OLD].exists() and paths[NEW].exists() and other.exists()
    assert paths[MID].exists() is not yes
    assert str(paths[MID]) in output
    assert str(paths[OLD]) not in output
    if not yes:
        assert "Use --keep <commit>" in output


def test_nothing_to_prune_notice_names_keep(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_current", lambda *a: "org__m@current")
    only = _slot(tmp_path, "org__m@" + NEW)
    result, output = invoke(tmp_path)
    assert result.exit_code == 0, output
    assert "Nothing to prune." in output and "Use --keep <commit>" in output
    assert only.exists()


def test_newest_pin_does_not_promote_another_slot(slots):
    root, paths = slots
    result, output = invoke(root, "--keep", NEW, "--yes")
    assert result.exit_code == 0, output
    assert paths[NEW].exists()
    assert not paths[OLD].exists() and not paths[MID].exists()


def test_repeated_pins_and_duplicates(slots):
    root, paths = slots
    result, output = invoke(root, "--keep", OLD, "--keep", MID, "--keep", OLD, "--yes")
    assert result.exit_code == 0, output
    assert all(p.exists() for p in paths.values())


@pytest.mark.parametrize("yes", [False, True])
def test_missing_pin_aborts_all_deletion(slots, yes):
    root, paths = slots
    missing = "d" * 40
    _slot(root, "other__m@" + missing)
    result, output = invoke(root, "--keep", OLD, "--keep", missing,
                            *(["--yes"] if yes else []))
    assert result.exit_code == 1, output
    assert missing in output and "No cache slot" in output
    assert all(p.exists() for p in paths.values())


@pytest.mark.parametrize("bad", ["A" * 40, "abc", "../slot", "a" * 41, "", "a" * 39])
@pytest.mark.parametrize("yes", [False, True])
def test_invalid_pin_rejected_before_current_or_prune(slots, monkeypatch, bad, yes):
    root, paths = slots
    def forbidden(*args, **kwargs):
        pytest.fail("validation must precede cache resolution and pruning")
    monkeypatch.setattr(cache, "_current", forbidden)
    monkeypatch.setattr("soup_cli.utils.layer_cache_slots.prune_slots", forbidden)
    result, output = invoke(root, "--keep", bad, *(["--yes"] if yes else []))
    assert result.exit_code == 1, output
    assert "Invalid --keep:" in output and "lowercase 40-hex" in output
    assert all(p.exists() for p in paths.values())


def test_pin_count_bounded_before_resolution(slots, monkeypatch):
    root, _ = slots
    monkeypatch.setattr(cache, "_current", lambda *a: pytest.fail("too many pins"))
    result, output = invoke(root, *[v for _ in range(65) for v in ("--keep", OLD)])
    assert result.exit_code == 1, output
    assert "64" in output and "Invalid --keep:" in output


@pytest.mark.parametrize("kind,reason", [
    ("current", "current slot"), ("writing", "write in progress, kept"),
    ("incomplete", "incomplete, kept"), ("resharding", "resharding"),
])
def test_pins_preserve_automatic_reason(slots, kind, reason):
    root, paths = slots
    current = None
    if kind == "current":
        current = str(paths[OLD])
    elif kind == "writing":
        (paths[OLD] / ".soup.test.tmp").touch()
    elif kind == "incomplete":
        (paths[OLD] / "index.json").unlink()
    else:
        (paths[OLD] / "index.json.resharding").touch()
    scanned = list_base_slots("org/m", str(root), current, keep_revisions={OLD})
    assert next(s for s in scanned if s.name.endswith(OLD)).reason == reason


def test_no_keep_retains_existing_behavior(slots):
    root, paths = slots
    result, output = invoke(root, "--yes")
    assert result.exit_code == 0, output
    assert not paths[OLD].exists() and not paths[MID].exists() and paths[NEW].exists()


def test_legacy_slot_cannot_match_pin(slots):
    root, paths = slots
    legacy = _slot(root, "org__m", age=50)
    result, output = invoke(root, "--keep", OLD, "--yes")
    assert result.exit_code == 0, output
    assert not legacy.exists() and paths[OLD].exists()
