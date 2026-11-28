"""Artifact staging refuses corrupt, partial, stale and unexpectedly sourced bytes."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from benchmarks.harness.streaming_controls import fetch_artifacts as module


@pytest.fixture
def case(tmp_path, monkeypatch):
    payload = b"frozen artifact"
    digest = hashlib.sha256(payload).hexdigest()
    lock = tmp_path / "runtime.lock"
    lock.write_text("unchanged lock\n")
    manifest = {
        "files": [
            {
                "filename": "fixture.whl",
                "bytes": len(payload),
                "sha256": digest,
                "url": "https://files.pythonhosted.org/packages/fixture.whl",
            }
        ],
        "python": {
            "name": "python.tar.gz",
            "size": len(payload),
            "digest": "sha256:" + digest,
            "browser_download_url": "https://github.com/astral-sh/python-build-standalone/releases/download/x/a",
        },
        "lock_sha256": {"runtime.lock": hashlib.sha256(lock.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("--output") + 1]).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, "200", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    return manifest_path, tmp_path / "offline", tmp_path / "receipt.json", calls, payload


def execute(case):
    return module.fetch(*case[:3], time.time() + 60)


def test_downloads_identical_bytes_once_then_refuses_resume(case):
    result = execute(case)
    assert result["status"] == "PASS_ALL_ARTIFACTS_SHA_READBACK"
    assert result["verified_bytes"] == 2 * len(case[4])
    assert (case[1] / "wheels/fixture.whl").read_bytes() == case[4]
    assert (case[1] / "python.tar.gz").read_bytes() == case[4]
    assert not list(case[1].rglob("*.partial"))
    for command, kwargs in case[3]:
        assert command[:2] == ["curl", "--disable"]
        assert command[command.index("--retry") + 1] == "0"
        assert command[command.index("--proto-redir") + 1] == "=https"
        assert 0 < kwargs["timeout"] <= 60
        assert "--continue-at" not in command
    with pytest.raises(FileExistsError, match="^Fresh destination and receipt required; no retry$"):
        execute(case)
    assert len(case[3]) == 2


@pytest.mark.parametrize("fault", ["short", "long", "hash", "http", "exit", "timeout"])
def test_no_installable_file_on_failure(case, monkeypatch, fault):
    def run(command, **kwargs):
        case[3].append(command)
        target = Path(command[command.index("--output") + 1])
        content = {"short": case[4][:-1], "long": case[4] + b"x", "hash": b"x" * len(case[4])}.get(
            fault, case[4]
        )
        target.write_bytes(content)
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(
            command, 22 if fault == "exit" else 0, "206" if fault == "http" else "200", "failure"
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
        execute(case)
    assert len(case[3]) == 1
    assert json.loads(case[2].read_text())["status"] == "BLOCKED"
    assert not (case[1] / "wheels/fixture.whl").exists()
    assert not (case[1] / "python.tar.gz").exists()


@pytest.mark.parametrize(
    "fault,message",
    [
        ("source", "Unexpected artifact source"),
        ("path", "Unsafe artifact filename"),
        ("duplicate", "Duplicate artifact"),
        ("length", "Invalid artifact length"),
        ("lock", "Lock SHA mismatch"),
        ("cap", "Artifact traffic bound exceeded"),
    ],
)
def test_bad_inventory_refused_before_network_or_mkdir(case, fault, message):
    manifest = json.loads(case[0].read_text())
    row = manifest["files"][0]
    if fault == "source":
        row["url"] = "https://files.pythonhosted.org.evil.example/fixture.whl"
    elif fault == "path":
        row["filename"] = "../escape.whl"
    elif fault == "duplicate":
        manifest["files"].append(dict(row))
    elif fault == "length":
        row["bytes"] = True
    elif fault == "cap":
        row["bytes"] = 3_400_000_001
    else:
        (case[0].parent / "runtime.lock").write_text("changed")
    case[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"^{message}$"):
        execute(case)
    assert not case[3] and not case[1].exists()


@pytest.mark.requires_symlink
def test_expired_deadline_and_symlink_destination(case):
    with pytest.raises(ValueError, match="^Require an unexpired deadline within 1200 seconds$"):
        module.fetch(*case[:3], time.time() - 1)
    target = case[1].with_name("actual")
    target.mkdir()
    case[1].symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="^Symlink destination refused$"):
        execute(case)
    assert not case[3] and not list(target.iterdir())


def test_real_inventory_is_exactly_the_existing_frozen_set():
    root = Path(__file__).parents[1] / "benchmarks/harness/streaming_controls"
    manifest = json.loads((root / "offline_artifacts.json").read_text())
    rows = module.inventory(manifest)
    assert len(rows) == 109
    assert sum(row["bytes"] for row in rows) == 3_288_520_713
    assert all(
        row["sha256"] == original["sha256"]
        for row, original in zip(rows[:-1], manifest["files"], strict=True)
    )


def test_deadline_includes_readback_and_prevents_next_download(case, monkeypatch):
    clock = [100.0]
    original = module.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        clock[0] = 111.0
        return result

    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(TimeoutError, match="^Artifact readback deadline expired$"):
        module.fetch(*case[:3], 110.0)
    assert len(case[3]) == 1
    assert json.loads(case[2].read_text())["verified_bytes"] == 0
    assert not (case[1] / "wheels/fixture.whl").exists()


def test_late_failure_preserves_prior_verified_evidence_without_retry(case, monkeypatch):
    original = module.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if len(case[3]) == 2:
            return subprocess.CompletedProcess(command, 22, "404", "missing")
        return result

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(
        RuntimeError, match=r"^Artifact transfer failed: python\.tar\.gz; exit=22; http=404$"
    ):
        execute(case)
    receipt = json.loads(case[2].read_text())
    assert receipt["status"] == "BLOCKED"
    assert receipt["verified_bytes"] == len(case[4])
    assert len(receipt["files"]) == 1 and len(case[3]) == 2
    assert (case[1] / "wheels/fixture.whl").read_bytes() == case[4]
    assert not (case[1] / "python.tar.gz").exists()
