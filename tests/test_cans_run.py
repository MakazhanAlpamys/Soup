"""Tests for can run temp directory cleanup on failure.

Refs #1464.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pytest
import yaml

from soup_cli.cans.run import CanRunResult, run_can


def _make_can(
    path: Path,
    *,
    deploy_targets: Optional[list[dict]] = None,
    with_symlink: bool = False,
) -> Path:
    manifest: dict[str, object] = {
        "can_format_version": 2 if deploy_targets else 1,
        "name": "test-recipe",
        "author": "tester",
        "created_at": "2026-01-01",
        "base_hash": "0123456789abcdef",
    }
    if deploy_targets:
        manifest["deploy_targets"] = deploy_targets

    manifest_bytes = yaml.safe_dump(manifest).encode("utf-8")
    config_bytes = b"base: test\ntask: sft\n"
    data_ref_bytes = b"kind: local\nlocal_path: data.jsonl\n"

    with tarfile.open(path, "w:gz") as tf:
        for name, payload in [
            ("manifest.yaml", manifest_bytes),
            ("config.yaml", config_bytes),
            ("data_ref.yaml", data_ref_bytes),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        if with_symlink:
            info = tarfile.TarInfo("bad_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)

    return path


@pytest.fixture()
def run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    systmp = tmp_path / "systmp"
    systmp.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(systmp))
    monkeypatch.setattr("soup_cli.cans.run._run_subprocess", lambda argv, **kw: 0)
    real_run = subprocess.run

    def fake_run(args, *a, **kw):
        if isinstance(args, list) and len(args) >= 3 and args[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="# fake freeze\n"
            )
        return real_run(args, *a, **kw)

    monkeypatch.setattr("soup_cli.cans.run.subprocess.run", fake_run)

    def leftovers() -> list[Path]:
        return list(systmp.glob("soup-can-run-*"))

    return leftovers


@pytest.fixture()
def spy_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs) -> str:
        res = real_mkdtemp(*args, **kwargs)
        calls.append(res)
        return res

    monkeypatch.setattr(tempfile, "mkdtemp", _spy)
    return calls


def test_capture_env_to_outside_cwd_creates_no_tempdir(
    tmp_path: Path, run_env: Callable[[], list[Path]], spy_mkdtemp: list[str]
) -> None:
    can_path = _make_can(tmp_path / "test.can")
    outside_env = tmp_path.parent / "escape_env.txt"

    with pytest.raises(ValueError, match="outside cwd"):
        run_can(str(can_path.name), yes=True, capture_env_to=str(outside_env))

    assert run_env() == []
    assert len(spy_mkdtemp) == 0


def test_capture_env_to_missing_directory_cleans_up(
    tmp_path: Path, run_env: Callable[[], list[Path]]
) -> None:
    can_path = _make_can(tmp_path / "test.can")

    with pytest.raises(FileNotFoundError):
        run_can(str(can_path.name), yes=True, capture_env_to="missing_dir/env.txt")

    assert run_env() == []


def test_deploy_failure_cleans_up(
    tmp_path: Path, run_env: Callable[[], list[Path]]
) -> None:
    can_path = _make_can(
        tmp_path / "test.can",
        deploy_targets=[{"kind": "ollama", "name": "test-ollama"}],
    )

    with pytest.raises(ValueError, match="requires a \\*\\.gguf"):
        run_can(str(can_path.name), yes=True, deploy=True)

    assert run_env() == []


def test_training_subprocess_error_cleans_up(
    tmp_path: Path, run_env: Callable[[], list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    can_path = _make_can(tmp_path / "test.can")

    def _fail_subprocess(*args, **kwargs):
        raise RuntimeError("simulated training crash")

    monkeypatch.setattr("soup_cli.cans.run._run_subprocess", _fail_subprocess)

    with pytest.raises(RuntimeError, match="simulated training crash"):
        run_can(str(can_path.name), yes=True)

    assert run_env() == []


def test_success_preserves_extract_dir(
    tmp_path: Path, run_env: Callable[[], list[Path]]
) -> None:
    can_path = _make_can(tmp_path / "test.can")

    result = run_can(str(can_path.name), yes=True)

    assert isinstance(result, CanRunResult)
    assert result.train_returncode == 0
    assert result.extract_dir.exists()
    assert run_env() == [result.extract_dir]


def test_nonzero_train_returncode_preserves_extract_dir(
    tmp_path: Path, run_env: Callable[[], list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    can_path = _make_can(tmp_path / "test.can")
    monkeypatch.setattr("soup_cli.cans.run._run_subprocess", lambda argv, **kw: 3)

    result = run_can(str(can_path.name), yes=True)

    assert isinstance(result, CanRunResult)
    assert result.train_returncode == 3
    assert result.extract_dir.exists()
    assert run_env() == [result.extract_dir]


def test_user_supplied_extract_dir_not_cleaned_on_failure(
    tmp_path: Path, run_env: Callable[[], list[Path]]
) -> None:
    can_path = _make_can(tmp_path / "test.can")
    user_extract = tmp_path / "custom_extract"

    with pytest.raises(FileNotFoundError):
        run_can(
            str(can_path.name),
            yes=True,
            extract_dir=str(user_extract),
            capture_env_to="missing_dir/env.txt",
        )

    assert user_extract.exists()
    assert run_env() == []


def test_extraction_failure_cleans_up(
    tmp_path: Path, run_env: Callable[[], list[Path]]
) -> None:
    can_path = _make_can(tmp_path / "corrupt.can", with_symlink=True)

    with pytest.raises((tarfile.TarError, ValueError), match="bad_link"):
        run_can(str(can_path.name), yes=True)

    assert run_env() == []
