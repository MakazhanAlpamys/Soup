"""Stale layer-stream cache slots: listing, preflight panel, opt-in prune (#2196)."""

import os
import shutil
import time

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import cache as cache_cmd
from soup_cli.trainer import stream_setup
from soup_cli.utils import layer_cache_slots
from soup_cli.utils.layer_cache_slots import CacheSlot, StripeFolder, list_base_slots, prune_slots
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

OLD = time.time() - 7200
INDEX = (
    '{"n_layers": 1, "dtype": "bf16", "source_fingerprint": "x", "extras": [], '
    '"large_keys": []}'
)
runner = CliRunner()


def _slot(root, name, size=100, *, index=True, age=OLD, extra=()):
    path = root / name
    path.mkdir(parents=True)
    (path / "layer_0.safetensors").write_bytes(b"x" * size)
    if index:
        (path / "index.json").write_text(INDEX)
    for fname in extra:
        (path / fname).write_text("")
    for f in path.iterdir():
        os.utime(f, (age, age))
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "soup_cli.utils.layer_shard.read_shard_index",
        lambda p, **k: type("I", (), {"stripe_roots": ()})()
        if os.path.exists(os.path.join(p, "index.json"))
        else (_ for _ in ()).throw(FileNotFoundError(p)),
    )
    return tmp_path


def _by_name(root, model="org/m", current=None):
    return {s.name: s for s in list_base_slots(model, str(root), current)}


def test_lists_same_base_only(root):
    _slot(root, "org__m@aaa", 100)
    _slot(root, "org__m@bbb", 200)
    _slot(root, "org__m", 50)
    _slot(root, "org__m-v2@x")
    (root / "stray.txt").write_text("")
    slots = _by_name(root)
    assert set(slots) == {"org__m@aaa", "org__m@bbb", "org__m"}
    assert slots["org__m@bbb"].bytes >= 200


def test_reference_rules(root):
    _slot(root, "org__m@cur")
    _slot(root, "org__m@rs", extra=("index.json.resharding",))
    _slot(root, "org__m@inc", index=False)
    _slot(root, "org__m@tmp", extra=(".soup.abc.tmp",))
    _slot(root, "org__m@old", age=OLD - 100)
    _slot(root, "org__m@new", age=OLD + 100)
    slots = _by_name(root, current="org__m@cur")
    assert {n: s.referenced for n, s in slots.items()} == {
        "org__m@cur": True, "org__m@rs": True, "org__m@inc": True,
        "org__m@tmp": True, "org__m@old": False, "org__m@new": True,
    }
    assert slots["org__m@inc"].reason == "incomplete, kept"
    assert slots["org__m@tmp"].reason == "write in progress, kept"


def test_keeps_newest_complete_slot_without_current(root):
    old = _slot(root, "org__m@aaa", age=OLD - 500)
    new = _slot(root, "org__m@bbb", age=OLD)
    res = prune_slots("org/m", str(root), None, confirm=True)
    assert [s.name for s, _ in res] == ["org__m@aaa"]
    assert not old.exists() and new.exists()


def test_single_complete_slot_never_removed(root):
    only = _slot(root, "org__m@aaa")
    assert prune_slots("org/m", str(root), None, confirm=True) == [] and only.exists()


def test_prune_dry_run_then_yes(root):
    _slot(root, "org__m@cur")
    old = _slot(root, "org__m@old", age=OLD - 100)
    _slot(root, "org__m@new", age=OLD + 100)
    dry = prune_slots("org/m", str(root), "org__m@cur", confirm=False)
    assert [s.name for s, _ in dry] == ["org__m@old"] and old.exists()
    done = prune_slots("org/m", str(root), "org__m@cur", confirm=True)
    assert done[0][1] is None and not old.exists() and (root / "org__m@cur").exists()


def test_prune_failure_does_not_stop_others(root, monkeypatch):
    a, b = _slot(root, "org__m@a", age=OLD - 200), _slot(root, "org__m@b", age=OLD - 100)
    _slot(root, "org__m@z")

    real = shutil.rmtree

    def flaky(path, *args, **kw):
        if path.endswith("@a"):
            raise OSError("boom")
        real(path, *args, **kw)

    monkeypatch.setattr(shutil, "rmtree", flaky)
    res = prune_slots("org/m", str(root), None, confirm=True)
    assert [e for _, e in res] == ["boom", None]
    assert a.exists() and not b.exists()


@pytest.mark.requires_symlink
def test_symlinked_slot_ignored(root, tmp_path_factory):
    target = tmp_path_factory.mktemp("elsewhere")
    (target / "f").write_text("x")
    os.symlink(target, root / "org__m@link", target_is_directory=True)
    assert _by_name(root) == {}
    prune_slots("org/m", str(root), None, confirm=True)
    assert (target / "f").exists()


@pytest.mark.parametrize(
    "suffix",
    ["[red]x", pytest.param("cache:warning:", marks=skip_colon_on_windows, id="emoji-code")],
)
@pytest.mark.parametrize("confirm", [False, True], ids=["dry-run", "remove"])
def test_hostile_name_renders(root, monkeypatch, suffix, confirm):
    # The path must stay on one line even in a narrow terminal.
    monkeypatch.setattr(cache_cmd.console, "_width", 20)
    slot = _slot(root, f"org__m@{suffix}", age=OLD - 100)
    displayed_path = os.path.realpath(slot)
    _slot(root, "org__m@z")
    args = ["prune-layer-cache", "--model", "org/m", "--cache-dir", str(root)]
    if confirm:
        args.append("--yes")

    result = runner.invoke(app, args, color=True)

    verb = "Removed" if confirm else "Would remove"
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"{verb} {displayed_path} (" in strip_ansi(result.stdout)
    assert slot.exists() is not confirm


@skip_colon_on_windows
def test_cli_prune_reports_failed_slot_path_literally(root, monkeypatch):
    slot = _slot(root, "org__m@cache:warning:", age=OLD - 100)
    _slot(root, "org__m@new")
    displayed_path = os.path.realpath(slot)
    rmtree = shutil.rmtree

    def fail_on_slot(path, *args, **kwargs):
        if path == displayed_path:
            raise OSError("boom")
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_on_slot)

    result = runner.invoke(
        app, ["prune-layer-cache", "--model", "org/m", "--cache-dir", str(root), "--yes"],
        color=True,
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Failed {displayed_path}: boom" in strip_ansi(result.stdout)
    assert slot.is_dir()


def test_cli_prune_reports_kept_stripe_path_literally(tmp_path, monkeypatch):
    # This path is only displayed, so the case can also run on Windows.
    stripe = StripeFolder(path=str(tmp_path / "stripe:warning:"), bytes=100)
    slot = CacheSlot(
        name="org__m@old",
        path=str(tmp_path / "org__m@old"),
        bytes=100,
        referenced=False,
        stripes=(stripe,),
    )
    monkeypatch.setattr(layer_cache_slots, "prune_slots", lambda *args, **kwargs: [(slot, None)])

    result = runner.invoke(
        app, ["prune-layer-cache", "--model", "org/m", "--cache-dir", str(tmp_path)],
        color=True,
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"stripe folder kept, delete manually: {stripe.path} (" in strip_ansi(result.stdout)


def test_cli_prune_requires_yes(root):
    old = _slot(root, "org__m@old", age=OLD - 100)
    _slot(root, "org__m@new")
    args = ["prune-layer-cache", "--model", "org/m", "--cache-dir", str(root)]
    assert runner.invoke(app, args).exit_code == 0 and old.exists()
    assert runner.invoke(app, [*args, "--yes"]).exit_code == 0 and not old.exists()


def _preflight(slots, free=10**15):
    return stream_setup._render_stream_disk_preflight(
        source_bytes=1, materialized_copy_bytes=0, materialize_bytes=0,
        materialized_path=".", shard_bytes=1, shard_write_bytes=1, shard_path=".",
        other_slots=slots, base="org/m",
    )


def test_preflight_lists_slots(root, capsys, monkeypatch):
    _slot(root, "org__m@a", 100)
    _slot(root, "org__m@b", 200)
    slots = list(_by_name(root).values())
    printed = []
    monkeypatch.setattr(stream_setup.console, "print", lambda p: printed.append(str(p.renderable)))
    _preflight(slots)
    text = printed[0]
    assert "org__m@a" in text and "org__m@b" in text and str(root) in text
    assert "kept: newest complete slot" in text
    assert "soup prune-layer-cache --model org/m --yes" in text


def test_preflight_default_unchanged(monkeypatch):
    printed = []
    monkeypatch.setattr(stream_setup.console, "print", lambda p: printed.append(str(p.renderable)))
    _preflight(())
    assert "cache slots" not in printed[0]
