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


# ---------------------------------------------------------------------------
# v0.71.1 #224 — compute_env_hash (auto-glue for `soup lock write`)
# ---------------------------------------------------------------------------


def _make_env_lock(*, version="2.1.0", created_at="2026-01-01T00:00:00+00:00"):
    from soup_cli.utils.env_lock import EnvEntry, EnvLock

    return EnvLock(
        soup_version="0.71.1",
        python_version="3.10.5",
        platform="linux-x86_64",
        cuda_version="12.1",
        entries=(EnvEntry(name="torch", version=version, source="pip"),),
        created_at=created_at,
    )


def test_compute_env_hash_is_64_hex():
    import re

    from soup_cli.utils.env_lock import compute_env_hash

    h = compute_env_hash(_make_env_lock())
    assert re.match(r"^[0-9a-f]{64}$", h)


def test_compute_env_hash_deterministic():
    from soup_cli.utils.env_lock import compute_env_hash

    assert compute_env_hash(_make_env_lock()) == compute_env_hash(_make_env_lock())


def test_compute_env_hash_excludes_created_at():
    # Two locks differing only in created_at must hash identically — the
    # env-hash is content-only so re-snapshotting the same env is stable.
    from soup_cli.utils.env_lock import compute_env_hash

    a = _make_env_lock(created_at="2026-01-01T00:00:00+00:00")
    b = _make_env_lock(created_at="2026-09-09T12:34:56+00:00")
    assert compute_env_hash(a) == compute_env_hash(b)


def test_compute_env_hash_content_sensitive():
    from soup_cli.utils.env_lock import compute_env_hash

    a = _make_env_lock(version="2.1.0")
    b = _make_env_lock(version="2.2.0")
    assert compute_env_hash(a) != compute_env_hash(b)


def test_compute_env_hash_rejects_non_lock():
    from soup_cli.utils.env_lock import compute_env_hash

    with pytest.raises(TypeError):
        compute_env_hash({"soup_version": "x"})  # type: ignore[arg-type]


def test_compute_env_hash_matches_lock_closure_regex():
    # The hash must be accepted by soup_lock.compute_lock_closure (which
    # requires each input to be 64-hex).
    from soup_cli.utils.env_lock import compute_env_hash
    from soup_cli.utils.soup_lock import compute_lock_closure

    env_hash = compute_env_hash(_make_env_lock())
    closure = compute_lock_closure(
        base_model_sha="a" * 64,
        dataset_sha="b" * 64,
        env_hash=env_hash,
    )
    assert len(closure) == 64


# ---------------------------------------------------------------------------
# v0.71.1 #209 — `soup env fix` install-plan renderer
# ---------------------------------------------------------------------------


def _lock_with_conda():
    from soup_cli.utils.env_lock import EnvEntry, EnvLock

    return EnvLock(
        soup_version="0.71.1",
        python_version="3.10.5",
        platform="linux-x86_64",
        cuda_version="12.1",
        entries=(
            EnvEntry(name="torch", version="2.1.0", source="pip"),
            EnvEntry(name="mkl", version="2023.1", source="conda"),
        ),
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_render_install_plan_uv_format():
    from soup_cli.utils.env_lock import render_install_plan

    plan = render_install_plan(_make_env_lock(), fmt="uv-pip")
    assert "uv pip install" in plan
    assert "torch==2.1.0" in plan
    # The python pin is surfaced so the operator recreates the same minor.
    assert "3.10" in plan


def test_render_install_plan_requirements_format():
    from soup_cli.utils.env_lock import render_install_plan

    plan = render_install_plan(_make_env_lock(), fmt="requirements")
    assert "torch==2.1.0" in plan
    assert "uv pip install" not in plan


def test_render_install_plan_skips_non_pip_as_comment():
    from soup_cli.utils.env_lock import render_install_plan

    plan = render_install_plan(_lock_with_conda(), fmt="uv-pip")
    # conda entry is surfaced as a comment, not an install line.
    assert "torch==2.1.0" in plan
    lines = [ln for ln in plan.splitlines() if "mkl" in ln]
    assert lines and all(ln.lstrip().startswith("#") for ln in lines)


def test_render_install_plan_rejects_unknown_format():
    from soup_cli.utils.env_lock import render_install_plan

    with pytest.raises(ValueError, match="format"):
        render_install_plan(_make_env_lock(), fmt="bogus")


def test_render_install_plan_rejects_non_lock():
    from soup_cli.utils.env_lock import render_install_plan

    with pytest.raises(TypeError):
        render_install_plan({"soup_version": "x"}, fmt="uv-pip")  # type: ignore[arg-type]


def test_write_requirements_txt_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.env_lock import write_requirements_txt

    write_requirements_txt(_make_env_lock(), "requirements.txt")
    text = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==2.1.0" in text


def test_write_requirements_txt_outside_cwd_rejected(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    from soup_cli.utils.env_lock import write_requirements_txt

    with pytest.raises(ValueError, match="cwd"):
        write_requirements_txt(_make_env_lock(), str(outside / "requirements.txt"))


# --- CLI: soup env fix ---


def test_cli_env_fix_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["env", "fix", "--help"])
    assert result.exit_code == 0, result.output


def test_cli_env_fix_renders_plan(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(app, ["env", "fix"])
    assert result.exit_code == 0, result.output
    assert "uv pip install" in result.output


def test_cli_env_fix_requirements_format(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(app, ["env", "fix", "--format", "requirements"])
    assert result.exit_code == 0, result.output
    assert "uv pip install" not in result.output


def test_cli_env_fix_missing_lock(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env", "fix"])
    assert result.exit_code == 1
    assert "soup env lock" in result.output


def test_cli_env_fix_writes_output(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(app, ["env", "fix", "--output", "requirements.txt"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "requirements.txt").exists()


def test_cli_env_fix_output_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(
        app, ["env", "fix", "--output", str(tmp_path / "req.txt")]
    )
    assert result.exit_code == 2


def test_cli_env_fix_output_null_byte_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(app, ["env", "fix", "--output", "a\x00b"])
    assert result.exit_code == 2, result.output


def test_render_install_plan_requirements_conda_comment():
    from soup_cli.utils.env_lock import render_install_plan

    # In requirements format a non-pip (conda) entry is surfaced as a comment
    # line rather than a bare `name==version` pip pin (v0.71.1 #209).
    plan = render_install_plan(_lock_with_conda(), fmt="requirements")
    assert "# conda: mkl==2023.1" in plan


def test_cli_env_fix_corrupt_lock(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup-env.lock").write_text("{ this is not valid json", encoding="utf-8")
    result = runner.invoke(app, ["env", "fix"])
    assert result.exit_code == 2, result.output


def test_cli_env_fix_bad_format(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    result = runner.invoke(app, ["env", "fix", "--format", "bogus-format"])
    assert result.exit_code == 2, result.output


def test_cli_env_check_no_drift(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["env", "lock"]).exit_code == 0
    # Nothing changed between snapshots, so the env is ABI-clean.
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 0, result.output
    assert "ABI-clean" in result.output


def test_cli_env_check_missing_lock(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 1
    assert "soup env lock" in result.output


def test_cli_env_check_drift_exits_3(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.utils.env_lock import snapshot_env, write_lock

    monkeypatch.chdir(tmp_path)
    # Write a lock that claims a different Python version → ABI drift on check.
    env = snapshot_env()
    drifted = dataclasses.replace(env, python_version="2.0.0")
    write_lock(drifted, "soup-env.lock")
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 3, result.output


def test_cli_env_lock_null_byte_output_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env", "lock", "--output", "a\x00b"])
    assert result.exit_code == 2, result.output


# ---------------------------------------------------------------------------
# check_declared_bounds — an installed package that violates soup-cli's OWN
# declared version bound (#368). Reads the bound from package metadata, never
# from a hardcoded copy, so it also catches every future instance of the shape.
# ---------------------------------------------------------------------------

# A SYNTHETIC un-gated requirement, kept because it is the smallest input that
# exercises the un-gated path. It is deliberately NOT what real metadata looks
# like: since the v0.71.0 deps-split, `importlib.metadata.requires("soup-cli")`
# lists `transformers` only under `extra == "train"/"all"/"dev"`, so no un-gated
# transformers row exists — see `_TRAIN_TRANSFORMERS` for the real shape and the
# tests that cover #368 end to end.
_UNGATED_TRANSFORMERS = ("transformers>=4.36.0,<5.0.0",)

# The real shape: the `transformers` cap #368 was reported against, exactly as
# metadata states it — gated behind the extras that pull it in.
_TRAIN_TRANSFORMERS = (
    'transformers>=4.36.0,<5.0.0; extra == "train"',
    'transformers>=4.36.0,<5.0.0; extra == "all"',
    'transformers>=4.36.0,<5.0.0; extra == "dev"',
)


def test_check_declared_bounds_flags_core_bound_over_cap():
    from soup_cli.utils.env_lock import check_declared_bounds

    # `pip install vllm` moved transformers to 5.14.1, past the core <5.0.0 cap.
    report = check_declared_bounds(_UNGATED_TRANSFORMERS, {"transformers": "5.14.1"})
    assert not report.ok
    assert report.violation_count == 1
    (v,) = report.violations
    assert v.name == "transformers"
    assert v.installed == "5.14.1"
    # The bound is quoted from the requirement string, not a hardcoded copy of
    # `<5.0.0` — that second copy is exactly the drift this catches.
    assert "<5.0.0" in v.specifier


def test_check_declared_bounds_control_compliant_passes():
    from soup_cli.utils.env_lock import check_declared_bounds

    # Control: an in-bounds environment passes, so the check cannot be
    # satisfied by always failing.
    report = check_declared_bounds(_UNGATED_TRANSFORMERS, {"transformers": "4.57.6"})
    assert report.ok
    assert report.violations == ()


def test_check_declared_bounds_skips_uninstalled_package():
    from soup_cli.utils.env_lock import check_declared_bounds

    # A package you never installed is not drift — only installed packages that
    # violate a bound are reported.
    report = check_declared_bounds(_UNGATED_TRANSFORMERS, {"typer": "0.19.0"})
    assert report.ok


def test_check_declared_bounds_reads_bound_from_given_requirements():
    from soup_cli.utils.env_lock import check_declared_bounds

    # Proof the bound is data-driven: change ONLY the requirement string and
    # the same installed version flips from compliant to violating.
    installed = {"widget": "2.0.0"}
    assert check_declared_bounds(("widget<3.0.0",), installed).ok
    bad = check_declared_bounds(("widget<2.0.0",), installed)
    assert not bad.ok
    assert bad.violations[0].name == "widget"


def test_check_declared_bounds_ignores_unparseable_requirement():
    from soup_cli.utils.env_lock import check_declared_bounds

    # A malformed requirement line must not crash the diagnostic.
    report = check_declared_bounds(("!!! not a requirement",), {"x": "1.0.0"})
    assert report.ok


def test_extra_gated_bound_not_opted_into_is_not_flagged():
    # #368 review finding 2: a package required only under `extra == "wandb"`
    # that the running environment never opted into is NOT a violation, even when
    # it is installed and out of range — the marker is EVALUATED, not just named.
    from soup_cli.utils.env_lock import check_declared_bounds

    reqs = ('wandb>=0.15.0,<0.18.0; extra == "wandb"',)
    report = check_declared_bounds(reqs, {"wandb": "0.19.1"})
    assert report.ok, report.violations


def test_core_bound_counted_once_despite_extra_duplicates():
    # #368 review finding 4: metadata restates the same bound under the base
    # deps AND under several extras. The violation is counted ONCE.
    from soup_cli.utils.env_lock import check_declared_bounds

    reqs = (
        "transformers>=4.36.0,<5.0.0",
        'transformers>=4.36.0,<5.0.0; extra == "all"',
        'transformers>=4.36.0,<5.0.0; extra == "dev"',
        'transformers>=4.36.0,<5.0.0; extra == "train"',
    )
    report = check_declared_bounds(reqs, {"transformers": "5.14.1"})
    assert report.violation_count == 1
    assert report.violations[0].name == "transformers"


def test_extra_gated_tracked_bound_is_enforced():
    # #368 review blocker: since the v0.71.0 deps-split, the `transformers`
    # <5.0.0 cap exists in metadata ONLY under `extra == "train"/"all"/"dev"`.
    # Deselecting every gated requirement made the case #368 was FILED ABOUT
    # unreachable — measured as `ok=True, violation_count=0` against real
    # installed metadata. A TRACKED_PACKAGES name keeps its bound enforced.
    from soup_cli.utils.env_lock import check_declared_bounds

    report = check_declared_bounds(_TRAIN_TRANSFORMERS, {"transformers": "5.14.1"})
    assert not report.ok, report
    assert report.violation_count == 1
    (v,) = report.violations
    assert v.name == "transformers"
    assert v.installed == "5.14.1"
    assert "<5.0.0" in v.specifier


def test_extra_gated_untracked_bound_stays_deselected():
    # The other half of the blocker fix: enforcing extra-gated bounds must NOT
    # reintroduce the false positive it replaced. `wandb` is installed out of
    # range but is not in TRACKED_PACKAGES, so it stays deselected.
    from soup_cli.utils.env_lock import TRACKED_PACKAGES, check_declared_bounds

    assert "wandb" not in {p.lower() for p in TRACKED_PACKAGES}
    reqs = ('wandb>=0.15.0,<0.18.0; extra == "wandb"',)
    report = check_declared_bounds(reqs, {"wandb": "0.19.1"})
    assert report.ok, report.violations


def test_mlx_only_bound_on_tracked_package_is_not_enforced():
    # A tracked NAME is not sufficient to enforce a gated bound. `[mlx]` declares
    # `transformers>=5.0.0` against `[train]`'s `<5.0.0` — deliberately
    # incompatible extras. Enforcing both by name would make EVERY installed
    # transformers violate exactly one of them, a false positive on every
    # machine. Only the training-install extras lift the marker.
    from soup_cli.utils.env_lock import check_declared_bounds

    mlx_only = ('transformers>=5.0.0; extra == "mlx"',)
    # A version the [train] cap is happy with, that the [mlx] floor is not.
    assert check_declared_bounds(mlx_only, {"transformers": "4.57.6"}).ok

    # And the two together resolve to the training bound alone, not to "both
    # bounds violated".
    both = _TRAIN_TRANSFORMERS + mlx_only
    assert check_declared_bounds(both, {"transformers": "4.57.6"}).ok
    breached = check_declared_bounds(both, {"transformers": "5.14.1"})
    assert breached.violation_count == 1
    assert "<5.0.0" in breached.violations[0].specifier


def test_extra_gated_tracked_bound_counted_once_across_extras():
    # #368 review finding: the dedup guard was dead while every gated row was
    # evaluated away. Enforcing tracked bounds brings the duplicates back —
    # metadata states the SAME transformers bound under train, all AND dev —
    # so the guard is now load-bearing. Deleting it makes this fail with 3.
    from soup_cli.utils.env_lock import check_declared_bounds

    assert len(_TRAIN_TRANSFORMERS) == 3
    report = check_declared_bounds(_TRAIN_TRANSFORMERS, {"transformers": "5.14.1"})
    assert report.violation_count == 1


def test_declared_bounds_warns_when_packaging_missing(caplog, monkeypatch):
    # #368 review: `packaging` is a declared core dependency now, so it should
    # never be absent — but if it is, the audit must not report a silent clean.
    # A checker that says "clean" without having checked fails the wrong way.
    import logging

    from soup_cli.utils.env_lock import check_declared_bounds

    monkeypatch.setitem(sys.modules, "packaging.markers", None)
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.env_lock"):
        report = check_declared_bounds(_UNGATED_TRANSFORMERS, {"transformers": "5.14.1"})
    # Still degrades to clean so `env check` keeps working...
    assert report.ok
    # ...but says so, naming the audit it could not run.
    assert any("declared-bounds audit was skipped" in r.message for r in caplog.records)


def test_current_declared_bounds_check_returns_bounds_check_for_absent_dist():
    from soup_cli.utils.env_lock import BoundsCheck, current_declared_bounds_check

    # An uninstalled or packaging-less environment degrades to a clean report.
    report = current_declared_bounds_check("this-distribution-does-not-exist")
    assert isinstance(report, BoundsCheck)
    assert report.ok


def test_current_declared_bounds_check_real_path_detects_violation(monkeypatch):
    # #368 review finding 1: exercise the REAL detection path end to end —
    # requires() → installed versions → check. Making the function a no-op, or
    # dropping the version lookup, fails this.
    import importlib.metadata as md

    from soup_cli.utils import env_lock

    monkeypatch.setattr(md, "requires", lambda dist: ["transformers>=4.36.0,<5.0.0"])
    monkeypatch.setattr(
        env_lock,
        "_detect_package_version",
        lambda name: "5.14.1" if name == "transformers" else None,
    )
    report = env_lock.current_declared_bounds_check("soup-cli")
    assert not report.ok
    assert report.violation_count == 1
    assert report.violations[0].name == "transformers"


def test_cli_env_check_declared_bound_violation_exits_3(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.commands import env as env_cmd
    from soup_cli.utils.env_lock import BoundsCheck, BoundViolation

    monkeypatch.chdir(tmp_path)
    violation = BoundViolation(
        name="transformers",
        installed="5.14.1",
        specifier=">=4.36.0,<5.0.0",
        extra=None,
    )
    monkeypatch.setattr(
        env_cmd,
        "current_declared_bounds_check",
        lambda: BoundsCheck(ok=False, violation_count=1, violations=(violation,)),
    )
    # No lock file present: the bound violation is caught AND the "no lock file"
    # diagnostic still prints — neither short-circuits the other (finding 5).
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 3, result.output
    assert "declared bounds" in result.output
    assert "lock" in result.output.lower()
    assert "transformers" in result.output


def test_cli_env_check_bound_violation_escapes_rich_markup(tmp_path, monkeypatch):
    # #368 review: the escape() calls were unasserted — stripping every one of
    # them still passed. A package name carrying Rich markup must render
    # LITERALLY, not be swallowed as a style tag.
    from soup_cli.cli import app
    from soup_cli.commands import env as env_cmd
    from soup_cli.utils.env_lock import BoundsCheck, BoundViolation

    monkeypatch.chdir(tmp_path)
    violation = BoundViolation(
        name="[bold]evil[/bold]",
        installed="5.14.1",
        specifier=">=4.36.0,<5.0.0",
        extra=None,
    )
    monkeypatch.setattr(
        env_cmd,
        "current_declared_bounds_check",
        lambda: BoundsCheck(ok=False, violation_count=1, violations=(violation,)),
    )
    result = runner.invoke(app, ["env", "check"])
    assert result.exit_code == 3, result.output
    assert "[bold]evil[/bold]" in result.output
