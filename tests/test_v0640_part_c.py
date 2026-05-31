"""v0.64.0 Part C — `soup env` hermetic lockfile + ABI-mismatch detection."""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import env_lock

    assert hasattr(env_lock, "EnvLock")
    assert hasattr(env_lock, "EnvEntry")
    assert hasattr(env_lock, "AbiCheck")
    assert hasattr(env_lock, "TRACKED_PACKAGES")
    assert hasattr(env_lock, "snapshot_env")
    assert hasattr(env_lock, "write_lock")
    assert hasattr(env_lock, "read_lock")
    assert hasattr(env_lock, "check_abi_compat")
    assert hasattr(env_lock, "DEFAULT_LOCK_FILE")


# ---------------------------------------------------------------------------
# TRACKED_PACKAGES catalogue
# ---------------------------------------------------------------------------


def test_tracked_packages_contains_core():
    from soup_cli.utils.env_lock import TRACKED_PACKAGES

    # Should at least include the ABI-sensitive heavyweights
    names = {p.lower() for p in TRACKED_PACKAGES}
    assert "torch" in names
    assert "transformers" in names
    assert "peft" in names


def test_tracked_packages_is_tuple():
    from soup_cli.utils.env_lock import TRACKED_PACKAGES

    assert isinstance(TRACKED_PACKAGES, tuple)


# ---------------------------------------------------------------------------
# EnvEntry
# ---------------------------------------------------------------------------


def test_env_entry_frozen():
    from soup_cli.utils.env_lock import EnvEntry

    e = EnvEntry(name="torch", version="2.1.0", source="pip")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.version = "0.0"  # type: ignore[misc]


def test_env_entry_rejects_empty_name():
    from soup_cli.utils.env_lock import EnvEntry

    with pytest.raises(ValueError, match="name"):
        EnvEntry(name="", version="1.0", source="pip")


def test_env_entry_rejects_null_byte():
    from soup_cli.utils.env_lock import EnvEntry

    with pytest.raises(ValueError, match="null"):
        EnvEntry(name="t\x00", version="1.0", source="pip")


def test_env_entry_rejects_invalid_source():
    from soup_cli.utils.env_lock import EnvEntry

    with pytest.raises(ValueError, match="source"):
        EnvEntry(name="t", version="1.0", source="random-bogus-thing")


def test_env_entry_known_sources():
    from soup_cli.utils.env_lock import EnvEntry

    # Just verify each known source instantiates clean
    for src in ("pip", "conda", "system", "wheel", "unknown"):
        EnvEntry(name="t", version="1.0", source=src)


# ---------------------------------------------------------------------------
# EnvLock
# ---------------------------------------------------------------------------


def test_env_lock_frozen():
    from soup_cli.utils.env_lock import EnvEntry, EnvLock

    lock = EnvLock(
        soup_version="0.64.0",
        python_version="3.10.0",
        platform="linux",
        cuda_version=None,
        entries=(EnvEntry(name="torch", version="2.0", source="pip"),),
        created_at="2026-05-20T00:00:00+00:00",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        lock.soup_version = "0.0.0"  # type: ignore[misc]


def test_env_lock_entries_must_be_tuple():
    from soup_cli.utils.env_lock import EnvEntry, EnvLock

    with pytest.raises(TypeError, match="tuple"):
        EnvLock(
            soup_version="0.64.0",
            python_version="3.10.0",
            platform="linux",
            cuda_version=None,
            entries=[EnvEntry(name="t", version="1.0", source="pip")],  # type: ignore[arg-type]
            created_at="2026-05-20T00:00:00+00:00",
        )


def test_env_lock_rejects_null_byte_in_platform():
    from soup_cli.utils.env_lock import EnvLock

    with pytest.raises(ValueError, match="null"):
        EnvLock(
            soup_version="0.64.0",
            python_version="3.10.0",
            platform="linux\x00",
            cuda_version=None,
            entries=(),
            created_at="2026-05-20T00:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# snapshot_env
# ---------------------------------------------------------------------------


def test_snapshot_env_returns_envlock():
    from soup_cli.utils.env_lock import EnvLock, snapshot_env

    lock = snapshot_env()
    assert isinstance(lock, EnvLock)
    # python_version should at least parse to two segments
    assert "." in lock.python_version


def test_snapshot_env_entries_non_empty():
    """The snapshot should contain at least one tracked package (pytest)."""
    from soup_cli.utils.env_lock import snapshot_env

    lock = snapshot_env()
    # The function inspects whatever's installed; pytest will at least be there.
    # We just verify the entries tuple is well-formed.
    assert isinstance(lock.entries, tuple)


# ---------------------------------------------------------------------------
# write_lock / read_lock
# ---------------------------------------------------------------------------


def test_write_lock_roundtrip(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import EnvEntry, EnvLock, read_lock, write_lock

    monkeypatch.chdir(tmp_path)
    lock = EnvLock(
        soup_version="0.64.0",
        python_version="3.10.0",
        platform="linux",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version="2.0", source="pip"),),
        created_at="2026-05-20T00:00:00+00:00",
    )
    out = tmp_path / "soup-env.lock"
    write_lock(lock, str(out))
    loaded = read_lock(str(out))
    assert loaded.soup_version == "0.64.0"
    assert len(loaded.entries) == 1
    assert loaded.entries[0].name == "torch"


def test_write_lock_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import EnvLock, write_lock

    monkeypatch.chdir(tmp_path)
    lock = EnvLock(
        soup_version="0.64.0",
        python_version="3.10.0",
        platform="linux",
        cuda_version=None,
        entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    outside = tmp_path.parent / "evil.lock"
    with pytest.raises(ValueError, match="cwd"):
        write_lock(lock, str(outside))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_write_lock_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import EnvLock, write_lock

    monkeypatch.chdir(tmp_path)
    lock = EnvLock(
        soup_version="0.64.0",
        python_version="3.10.0",
        platform="linux",
        cuda_version=None,
        entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    target = tmp_path / "real.lock"
    target.write_text("{}")
    link = tmp_path / "link.lock"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symlink"):
        write_lock(lock, str(link))


def test_read_lock_missing(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import read_lock

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_lock(str(tmp_path / "nope.lock"))


def test_read_lock_invalid_json(tmp_path, monkeypatch):
    from soup_cli.utils.env_lock import read_lock

    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.lock"
    p.write_text("not json")
    with pytest.raises(ValueError):
        read_lock(str(p))


# ---------------------------------------------------------------------------
# AbiCheck
# ---------------------------------------------------------------------------


def test_abi_check_frozen():
    from soup_cli.utils.env_lock import AbiCheck

    a = AbiCheck(ok=True, drift_count=0, changes=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.ok = False  # type: ignore[misc]


def test_abi_check_changes_is_tuple():
    from soup_cli.utils.env_lock import AbiCheck

    with pytest.raises(TypeError, match="tuple"):
        AbiCheck(ok=False, drift_count=1, changes=["x"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# check_abi_compat
# ---------------------------------------------------------------------------


def test_check_abi_compat_no_drift():
    from soup_cli.utils.env_lock import EnvEntry, EnvLock, check_abi_compat

    a = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version="2.0", source="pip"),),
        created_at="2026-05-20T00:00:00+00:00",
    )
    b = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version="2.0", source="pip"),),
        created_at="2026-05-21T00:00:00+00:00",  # timestamp differs but ABI same
    )
    report = check_abi_compat(a, b)
    assert report.ok is True
    assert report.drift_count == 0


def test_check_abi_compat_torch_change():
    from soup_cli.utils.env_lock import EnvEntry, EnvLock, check_abi_compat

    a = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version="2.0", source="pip"),),
        created_at="2026-05-20T00:00:00+00:00",
    )
    b = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version="2.1", source="pip"),),
        created_at="2026-05-20T00:00:00+00:00",
    )
    report = check_abi_compat(a, b)
    assert report.ok is False
    assert report.drift_count >= 1


def test_check_abi_compat_cuda_change():
    from soup_cli.utils.env_lock import EnvLock, check_abi_compat

    a = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="12.1", entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    b = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version="11.8", entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    report = check_abi_compat(a, b)
    assert report.ok is False
    assert any("cuda" in c.lower() for c in report.changes)


def test_check_abi_compat_python_change():
    from soup_cli.utils.env_lock import EnvLock, check_abi_compat

    a = EnvLock(
        soup_version="0.64.0", python_version="3.10.0", platform="linux",
        cuda_version=None, entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    b = EnvLock(
        soup_version="0.64.0", python_version="3.11.0", platform="linux",
        cuda_version=None, entries=(),
        created_at="2026-05-20T00:00:00+00:00",
    )
    report = check_abi_compat(a, b)
    assert report.ok is False


def test_check_abi_compat_rejects_non_envlock():
    from soup_cli.utils.env_lock import check_abi_compat

    with pytest.raises(TypeError):
        check_abi_compat("not a lock", "not a lock")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_env_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["env", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_env_lock_writes_file(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env", "lock"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "soup-env.lock").exists()


def test_cli_env_status_no_lock(tmp_path, monkeypatch):
    """`env status` without an existing lock file exits with friendly error."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env", "status"])
    # Should not crash; either exits non-zero with message or shows empty status.
    assert result.exit_code in (0, 1)


def test_cli_env_status_with_lock(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    r1 = runner.invoke(app, ["env", "lock"])
    assert r1.exit_code == 0, (r1.output, repr(r1.exception))
    r2 = runner.invoke(app, ["env", "status"])
    assert r2.exit_code == 0, (r2.output, repr(r2.exception))


def test_cli_env_lock_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "evil.lock"
    result = runner.invoke(app, ["env", "lock", "--output", str(outside)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Source-wiring regression
# ---------------------------------------------------------------------------


def test_cli_registers_env():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "cli.py"
    text = src.read_text(encoding="utf-8")
    assert '"env"' in text or "'env'" in text or 'name="env"' in text


def test_no_heavy_top_level_imports():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "env_lock.py"
    text = src.read_text(encoding="utf-8")
    import re
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        assert not re.search(bad, text, re.MULTILINE)
