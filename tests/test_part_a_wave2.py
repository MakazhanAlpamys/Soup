"""Part A wave 2 — #34 soup can run / publish (v0.33.0).

Covers:
  - Schema bump: CAN_FORMAT_VERSION 1 → 2 with both supported.
  - DeployTarget validation (kind enum, path traversal, null-byte rejection).
  - cans.run.capture_env: smoke + cwd containment.
  - cans.run.run_can: confirmation gate, extract-dir containment, train
    subprocess invocation (mocked), env_capture, deploy targets.
  - cans.publish.publish_can: repo_id validation, token resolution,
    commit-message sanitization, HfApi mocking.
  - CLI: `soup can run` confirmation panel, `soup can publish` smoke.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers — build a minimal valid .can on disk
# ---------------------------------------------------------------------------


def _build_can(
    out_path: Path,
    *,
    name: str = "demo-recipe",
    can_format_version: int = 2,
    deploy_targets: list[dict] | None = None,
) -> Path:
    """Construct a minimal valid .can tarball at ``out_path``."""
    manifest = {
        "can_format_version": can_format_version,
        "name": name,
        "author": "tester",
        "created_at": "2026-04-27T12:00:00",
        "base_hash": "0" * 64,
        "description": "test can",
        "tags": [],
    }
    if deploy_targets is not None:
        manifest["deploy_targets"] = deploy_targets

    config = {
        "base": "test/model",
        "task": "sft",
        "data": {"train": "data.jsonl", "format": "alpaca"},
        "training": {"epochs": 1, "lr": 0.0001, "batch_size": 1},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, mode="w:gz") as tar:
        for fname, payload in (
            ("manifest.yaml", yaml.safe_dump(manifest)),
            ("config.yaml", yaml.safe_dump(config)),
            ("data_ref.yaml", yaml.safe_dump({"kind": "local"})),
        ):
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name=fname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out_path


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestCanFormatVersionBump:
    def test_v1_still_loads(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=1,
            name="x", author="a", created_at="2026-01-01",
            base_hash="0" * 64,
        )
        assert m.can_format_version == 1

    def test_v2_loads(self):
        from soup_cli.cans.schema import Manifest

        m = Manifest(
            can_format_version=2,
            name="x", author="a", created_at="2026-01-01",
            base_hash="0" * 64,
        )
        assert m.can_format_version == 2

    def test_v3_rejected(self):
        from soup_cli.cans.schema import Manifest

        with pytest.raises(Exception, match="unknown can_format_version"):
            Manifest(
                can_format_version=3,
                name="x", author="a", created_at="2026-01-01",
                base_hash="0" * 64,
            )

    def test_pack_writes_v2(self, tmp_path, monkeypatch):
        from soup_cli.cans.schema import CAN_FORMAT_VERSION

        assert CAN_FORMAT_VERSION == 2


class TestDeployTargetValidation:
    def test_known_kinds_accepted(self):
        from soup_cli.cans.schema import DeployTarget

        for kind in ("ollama", "gguf", "vllm"):
            DeployTarget(kind=kind, name="x")

    def test_unknown_kind_rejected(self):
        from soup_cli.cans.schema import DeployTarget

        with pytest.raises(Exception):
            DeployTarget(kind="docker", name="x")

    def test_path_traversal_rejected(self):
        from soup_cli.cans.schema import DeployTarget

        with pytest.raises(ValueError, match="\\.\\."):
            DeployTarget(kind="gguf", path="../../etc/passwd")

    def test_absolute_path_rejected(self):
        from soup_cli.cans.schema import DeployTarget

        with pytest.raises(ValueError, match="must be relative"):
            DeployTarget(kind="gguf", path="/etc/passwd")

    def test_null_byte_in_name_rejected(self):
        from soup_cli.cans.schema import DeployTarget

        with pytest.raises(ValueError, match="null bytes"):
            DeployTarget(kind="ollama", name="bad\x00name")

    def test_null_byte_in_path_rejected(self):
        from soup_cli.cans.schema import DeployTarget

        with pytest.raises(ValueError, match="null byte"):
            DeployTarget(kind="gguf", path="x\x00.gguf")

    def test_manifest_with_deploy_targets(self):
        from soup_cli.cans.schema import DeployTarget, Manifest

        m = Manifest(
            can_format_version=2,
            name="x", author="a", created_at="2026-01-01",
            base_hash="0" * 64,
            deploy_targets=[
                DeployTarget(kind="ollama", name="my-model"),
                DeployTarget(kind="gguf", path="model.gguf"),
            ],
        )
        assert len(m.deploy_targets) == 2
        assert m.deploy_targets[0].kind == "ollama"


# ---------------------------------------------------------------------------
# capture_env
# ---------------------------------------------------------------------------


class TestCaptureEnv:
    def test_capture_env_writes_python_version(self, tmp_path):
        from soup_cli.cans.run import capture_env

        out = capture_env(tmp_path / "env.txt")
        text = out.read_text(encoding="utf-8")
        assert "python=" in text
        assert "platform=" in text

    def test_capture_env_handles_pip_failure(self, tmp_path, monkeypatch):
        """If pip freeze times out, env.txt is still written."""
        from soup_cli.cans import run as run_mod

        def _boom(*args, **kwargs):
            raise OSError("pip not on PATH")

        monkeypatch.setattr(run_mod.subprocess, "run", _boom)
        out = run_mod.capture_env(tmp_path / "env.txt")
        text = out.read_text(encoding="utf-8")
        assert "pip freeze failed" in text


# ---------------------------------------------------------------------------
# run_can — orchestrator
# ---------------------------------------------------------------------------


class TestRunCan:
    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.can"
        with pytest.raises(ValueError, match="outside cwd"):
            run_can(str(outside), yes=True)

    def test_missing_can(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            run_can(str(tmp_path / "missing.can"), yes=True)

    def test_requires_yes_or_callback(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        with pytest.raises(ValueError, match="requires --yes"):
            run_can(str(can))

    def test_callback_can_decline(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        with pytest.raises(ValueError, match="declined"):
            run_can(str(can), confirm_callback=lambda _m: False)

    def test_train_invoked_via_subprocess(self, tmp_path, monkeypatch):
        from soup_cli.cans import run as run_mod

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")

        captured: dict = {}

        def _fake_run(argv, **kwargs):
            captured["argv"] = argv
            return MagicMock(returncode=0)

        monkeypatch.setattr(run_mod.subprocess, "run", _fake_run)
        result = run_mod.run_can(str(can), yes=True)
        assert result.train_returncode == 0
        assert "train" in captured["argv"]
        assert "--config" in captured["argv"]
        assert "--yes" in captured["argv"]

    def test_extract_dir_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        outside = str(tmp_path.parent / "extract")
        with pytest.raises(ValueError, match="outside cwd"):
            run_can(str(can), yes=True, extract_dir=outside)

    def test_env_capture_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cans.run import run_can

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        outside = str(tmp_path.parent / "env.txt")
        # We never reach the env capture step because the validation runs
        # before subprocess; but to test the validation we need to mock
        # subprocess.
        from soup_cli.cans import run as run_mod
        monkeypatch.setattr(run_mod.subprocess, "run",
                            lambda *a, **k: MagicMock(returncode=0))
        with pytest.raises(ValueError, match="outside cwd"):
            run_can(str(can), yes=True, capture_env_to=outside)


# ---------------------------------------------------------------------------
# publish_can
# ---------------------------------------------------------------------------


class TestPublishCan:
    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cans.publish import publish_can

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside cwd"):
            publish_can(
                str(tmp_path.parent / "x.can"),
                repo_id="me/repo", token="tok",
            )

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        from soup_cli.cans.publish import publish_can

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            publish_can(
                str(tmp_path / "missing.can"),
                repo_id="me/repo", token="tok",
            )

    def test_invalid_repo_id_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cans.publish import publish_can

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        with pytest.raises(ValueError):
            publish_can(str(can), repo_id="not a valid repo!", token="t")

    def test_no_token_raises(self, tmp_path, monkeypatch):
        from soup_cli.cans.publish import publish_can

        monkeypatch.chdir(tmp_path)
        # Force resolve_token to return empty
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(
            "soup_cli.cans.publish.resolve_token", lambda: "",
        )
        can = _build_can(tmp_path / "r.can")
        with pytest.raises(ValueError, match="no HF token"):
            publish_can(str(can), repo_id="me/repo")

    def test_commit_message_sanitized(self, tmp_path, monkeypatch):
        from soup_cli.cans.publish import _sanitize_commit_message

        # First-line only
        result = _sanitize_commit_message("first\nsecond\nthird")
        assert result == "first"

        # 200-char cap
        long_msg = "x" * 500
        capped = _sanitize_commit_message(long_msg)
        assert len(capped) <= 200

        # Empty / None falls back
        assert "can-format-v1" in _sanitize_commit_message(None)

    def test_happy_path_uploads_with_mocked_hf(self, tmp_path, monkeypatch):
        from soup_cli.cans import publish as publish_mod

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")

        fake_api = MagicMock()
        fake_api.create_repo = MagicMock(return_value=None)
        fake_api.upload_file = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=MagicMock(return_value=fake_api))},
        ):
            url = publish_mod.publish_can(
                str(can), repo_id="me/test-can", token="tok",
            )
        assert "huggingface.co/datasets/me/test-can" in url
        fake_api.create_repo.assert_called_once()
        fake_api.upload_file.assert_called_once()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCanCLIPublishAndRun:
    def test_can_run_without_yes_shows_panel_and_exits(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        can = _build_can(tmp_path / "r.can")
        result = runner.invoke(app, ["can", "run", str(can)])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "confirm" in result.output.lower() or "--yes" in result.output

    def test_can_publish_missing_file(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["can", "publish", str(tmp_path / "no.can"),
             "--hf-hub", "me/r"],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
