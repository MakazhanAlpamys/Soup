"""#2196 — stale layer-stream cache slots: listing, preflight panel, opt-in prune."""

import os
import time

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.trainer import stream_setup
from soup_cli.utils.layer_cache_slots import list_base_slots, prune_slots

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
    _slot(root, "org__m@new", age=time.time())
    _slot(root, "org__m@old")
    slots = _by_name(root, current="org__m@cur")
    assert {n: s.referenced for n, s in slots.items()} == {
        "org__m@cur": True, "org__m@rs": True, "org__m@inc": True,
        "org__m@new": True, "org__m@old": False,
    }
    assert slots["org__m@inc"].reason == "incomplete, kept"


def test_prune_dry_run_then_yes(root):
    _slot(root, "org__m@cur")
    old = _slot(root, "org__m@old")
    dry = prune_slots("org/m", str(root), "org__m@cur", confirm=False)
    assert [s.name for s, _ in dry] == ["org__m@old"] and old.exists()
    done = prune_slots("org/m", str(root), "org__m@cur", confirm=True)
    assert done[0][1] is None and not old.exists() and (root / "org__m@cur").exists()


def test_prune_failure_does_not_stop_others(root, monkeypatch):
    a, b = _slot(root, "org__m@a"), _slot(root, "org__m@b")
    import shutil

    real = shutil.rmtree

    def flaky(path, *args, **kw):
        if path.endswith("@a"):
            raise OSError("boom")
        real(path, *args, **kw)

    monkeypatch.setattr(shutil, "rmtree", flaky)
    res = prune_slots("org/m", str(root), None, confirm=True)
    assert [e for _, e in res] == ["boom", None]
    assert a.exists() and not b.exists()


def test_symlinked_slot_ignored(root, tmp_path_factory):
    target = tmp_path_factory.mktemp("elsewhere")
    (target / "f").write_text("x")
    try:
        os.symlink(target, root / "org__m@link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlink support")
    assert _by_name(root) == {}
    prune_slots("org/m", str(root), None, confirm=True)
    assert (target / "f").exists()


def test_hostile_name_renders(root):
    _slot(root, "org__m@[red]x[/]")
    result = runner.invoke(app, ["cache", "list", "--model", "org/m", "--cache-dir", str(root)])
    assert result.exit_code == 0 and "[red]x" in result.output


def test_cli_prune_requires_yes(root):
    old = _slot(root, "org__m@old")
    args = ["cache", "prune", "--model", "org/m", "--cache-dir", str(root)]
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
    _slot(root, "org__m@b", 200, age=time.time())
    slots = list(_by_name(root).values())
    printed = []
    monkeypatch.setattr(stream_setup.console, "print", lambda p: printed.append(str(p.renderable)))
    _preflight(slots)
    text = printed[0]
    assert "org__m@a" in text and "org__m@b" in text and str(root) in text
    assert "kept: modified within the last hour" in text
    assert "soup cache prune --model org/m --yes" in text


def test_preflight_default_unchanged(monkeypatch):
    printed = []
    monkeypatch.setattr(stream_setup.console, "print", lambda p: printed.append(str(p.renderable)))
    _preflight(())
    assert "cache slots" not in printed[0]
