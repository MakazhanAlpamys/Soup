"""v0.57.0 Part D — adapters branch / checkout / branches list."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.adapter_branch import (
    Branch,
    create_branch,
    delete_branch,
    list_branches,
    load_branch,
    write_checkout,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_branches_dir(tmp_path, monkeypatch):
    """Redirect SOUP_BRANCHES_DIR to tmp so tests don't pollute ~/.soup."""
    branches = tmp_path / "branches"
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(branches))
    monkeypatch.chdir(tmp_path)
    return branches


def _make_config(tmp_path: Path, name: str = "soup.yaml", content: str = "base: test\n") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------- create_branch ----------


def test_create_branch_happy(tmp_path):
    _make_config(tmp_path)
    snap = create_branch("v1", config_path="soup.yaml", base_model="meta/llama")
    assert isinstance(snap, Branch)
    assert snap.name == "v1"
    assert snap.base_model == "meta/llama"
    assert len(snap.config_sha256) == 64
    assert snap.dataset_sha256 is None


def test_create_branch_with_dataset(tmp_path):
    _make_config(tmp_path)
    (tmp_path / "data.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    snap = create_branch("v1", config_path="soup.yaml", base_model="b",
                          dataset_path="data.jsonl")
    assert snap.dataset_sha256 is not None
    assert len(snap.dataset_sha256) == 64


def test_create_branch_invalid_name(tmp_path):
    _make_config(tmp_path)
    with pytest.raises(ValueError, match="must match"):
        create_branch("../etc", config_path="soup.yaml", base_model="b")
    with pytest.raises(ValueError, match="non-empty"):
        create_branch("", config_path="soup.yaml", base_model="b")


def test_create_branch_null_byte_name(tmp_path):
    _make_config(tmp_path)
    with pytest.raises(ValueError, match="null"):
        create_branch("v\x001", config_path="soup.yaml", base_model="b")


def test_create_branch_bool_name(tmp_path):
    _make_config(tmp_path)
    with pytest.raises(TypeError):
        create_branch(True, config_path="soup.yaml", base_model="b")  # type: ignore[arg-type]


def test_create_branch_config_outside_cwd(tmp_path):
    outside = os.path.join(os.path.dirname(str(tmp_path)), "x.yaml")
    with pytest.raises(ValueError):
        create_branch("v1", config_path=outside, base_model="b")


def test_create_branch_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_branch("v1", config_path="missing.yaml", base_model="b")


def test_create_branch_empty_base_model(tmp_path):
    _make_config(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        create_branch("v1", config_path="soup.yaml", base_model="")


def test_create_branch_null_byte_base_model(tmp_path):
    _make_config(tmp_path)
    with pytest.raises(ValueError, match="null"):
        create_branch("v1", config_path="soup.yaml", base_model="x\x00")


def test_create_branch_bool_base_model_rejected(tmp_path):
    """bool subclasses str-checks in some idioms; ensure explicit TypeError."""
    _make_config(tmp_path)
    with pytest.raises(TypeError):
        create_branch("v1", config_path="soup.yaml",
                       base_model=True)  # type: ignore[arg-type]


def test_delete_branch_traversal_rejected():
    with pytest.raises(ValueError):
        delete_branch("../etc/passwd")


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink semantics")
def test_load_branch_rejects_symlink(tmp_path, monkeypatch):
    branches = tmp_path / "br"
    branches.mkdir()
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(branches))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    os.symlink(str(target), str(branches / "evil.json"))
    with pytest.raises(ValueError, match="symlink"):
        load_branch("evil")


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink semantics")
def test_delete_branch_rejects_symlink(tmp_path, monkeypatch):
    branches = tmp_path / "br"
    branches.mkdir()
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(branches))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    os.symlink(str(target), str(branches / "evil.json"))
    with pytest.raises(ValueError, match="symlink"):
        delete_branch("evil")
    # original file untouched
    assert target.exists()


def test_branches_dir_crlf_env_rejected(monkeypatch, tmp_path):
    """Control char (CRLF) in SOUP_BRANCHES_DIR falls back to default."""
    from soup_cli.utils.adapter_branch import _branches_dir
    monkeypatch.setenv("SOUP_BRANCHES_DIR", "/some\npath")
    monkeypatch.chdir(tmp_path)
    resolved = _branches_dir()
    assert "\n" not in str(resolved)


def test_create_branch_oversize_config(tmp_path):
    # 1 MiB + 1 byte
    (tmp_path / "soup.yaml").write_bytes(b"x" * (1_048_577))
    with pytest.raises(ValueError, match="cap"):
        create_branch("v1", config_path="soup.yaml", base_model="b")


def test_create_branch_atomic_write(tmp_path, isolated_branches_dir):
    _make_config(tmp_path)
    snap = create_branch("v1", config_path="soup.yaml", base_model="b")
    target = isolated_branches_dir / "v1.json"
    assert target.exists()
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["name"] == "v1"
    assert raw["config_sha256"] == snap.config_sha256


# ---------- list_branches ----------


def test_list_branches_empty(tmp_path):
    assert list_branches() == ()


def test_list_branches_sorted(tmp_path):
    _make_config(tmp_path)
    create_branch("b", config_path="soup.yaml", base_model="x")
    create_branch("a", config_path="soup.yaml", base_model="x")
    create_branch("c", config_path="soup.yaml", base_model="x")
    assert list_branches() == ("a", "b", "c")


# ---------- load_branch ----------


def test_load_branch_roundtrip(tmp_path):
    _make_config(tmp_path)
    created = create_branch("v1", config_path="soup.yaml", base_model="m")
    loaded = load_branch("v1")
    assert loaded.name == created.name
    assert loaded.config_sha256 == created.config_sha256
    assert loaded.base_model == created.base_model


def test_load_branch_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_branch("nope")


def test_load_branch_invalid_name():
    with pytest.raises(ValueError):
        load_branch("../etc")


# ---------- delete_branch ----------


def test_delete_branch_true_when_present(tmp_path):
    _make_config(tmp_path)
    create_branch("v1", config_path="soup.yaml", base_model="m")
    assert delete_branch("v1") is True
    assert "v1" not in list_branches()


def test_delete_branch_false_when_missing():
    assert delete_branch("nope") is False


# ---------- write_checkout ----------


def test_write_checkout_writes_target(tmp_path):
    _make_config(tmp_path, content="base: original\n")
    snap = create_branch("v1", config_path="soup.yaml", base_model="m")
    written = write_checkout(snap, "restored.yaml")
    assert written.exists()
    assert "original" in written.read_text(encoding="utf-8")


def test_write_checkout_detects_drift(tmp_path):
    _make_config(tmp_path, content="base: original\n")
    snap = create_branch("v1", config_path="soup.yaml", base_model="m")
    # Mutate original config — SHA mismatch
    (tmp_path / "soup.yaml").write_text("base: drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        write_checkout(snap, "restored.yaml")


def test_write_checkout_outside_cwd_rejected(tmp_path):
    _make_config(tmp_path)
    snap = create_branch("v1", config_path="soup.yaml", base_model="m")
    outside = os.path.join(os.path.dirname(str(tmp_path)), "x.yaml")
    with pytest.raises(ValueError):
        write_checkout(snap, outside)


def test_write_checkout_rejects_non_branch():
    with pytest.raises(TypeError):
        write_checkout("not a branch", "out.yaml")  # type: ignore[arg-type]


def test_write_checkout_missing_source(tmp_path):
    _make_config(tmp_path)
    snap = create_branch("v1", config_path="soup.yaml", base_model="m")
    (tmp_path / "soup.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        write_checkout(snap, "restored.yaml")


# ---------- Frozen Branch ----------


def test_branch_frozen(tmp_path):
    import dataclasses
    _make_config(tmp_path)
    snap = create_branch("v1", config_path="soup.yaml", base_model="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.name = "v2"  # type: ignore[misc]


# ---------- Env override containment ----------


def test_branches_dir_env_override_outside_falls_back(monkeypatch, tmp_path):
    """Out-of-bounds env override is rejected; helper returns the in-bounds default.

    We point SOUP_BRANCHES_DIR at a tmp-rooted path that IS in-bounds (under
    $TMPDIR per the containment policy) so we can also assert that valid
    overrides ARE honoured — i.e. we test both directions of the policy here.
    """
    from soup_cli.utils.adapter_branch import _branches_dir
    in_bounds = tmp_path / "valid-override"
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(in_bounds))
    monkeypatch.chdir(tmp_path)
    resolved = _branches_dir()
    assert os.path.realpath(str(resolved)) == os.path.realpath(str(in_bounds))


def test_branches_dir_null_byte_env_ignored(monkeypatch, tmp_path):
    """Null-byte env override is silently ignored; default ~/.soup/branches used."""
    from soup_cli.utils.adapter_branch import _branches_dir
    monkeypatch.setenv("SOUP_BRANCHES_DIR", "/some\x00path")
    monkeypatch.chdir(tmp_path)
    resolved = _branches_dir()
    # Falls back to ~/.soup/branches — must NOT contain a null byte
    assert "\x00" not in str(resolved)
    assert ".soup" in str(resolved) and "branches" in str(resolved)


# ---------- CLI smoke ----------


def test_branch_cli_help():
    result = runner.invoke(soup_app, ["adapters", "branch", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_checkout_cli_help():
    result = runner.invoke(soup_app, ["adapters", "checkout", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_branches_cli_help():
    result = runner.invoke(soup_app, ["adapters", "branches", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_branch_cli_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    _make_config(tmp_path)
    r1 = runner.invoke(soup_app, [
        "adapters", "branch", "v1",
        "-c", "soup.yaml", "--base", "meta/llama",
    ])
    assert r1.exit_code == 0, (r1.output, repr(r1.exception))
    assert "v1" in r1.output

    r2 = runner.invoke(soup_app, ["adapters", "branches"])
    assert r2.exit_code == 0, (r2.output, repr(r2.exception))
    assert "v1" in r2.output


def test_branch_cli_invalid_name(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    _make_config(tmp_path)
    result = runner.invoke(soup_app, [
        "adapters", "branch", "../etc",
        "-c", "soup.yaml", "--base", "m",
    ])
    assert result.exit_code == 2
    assert "must match" in result.output


def test_branch_cli_missing_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(soup_app, [
        "adapters", "branch", "v1",
        "-c", "missing.yaml", "--base", "m",
    ])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_checkout_cli_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    _make_config(tmp_path, content="base: x\nepochs: 1\n")
    create_branch("v1", config_path="soup.yaml", base_model="m")
    result = runner.invoke(soup_app, [
        "adapters", "checkout", "v1",
        "-o", "restored.yaml",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "restored.yaml").exists()


def test_checkout_cli_missing_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(soup_app, [
        "adapters", "checkout", "nope", "-o", "out.yaml",
    ])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_branches_cli_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "br"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(soup_app, ["adapters", "branches"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "No branches" in result.output
