"""Tests for v0.34.0 Part D — crash reporter."""

from __future__ import annotations

import json

import pytest

from soup_cli.utils.crash import (
    CRASH_FORMAT_VERSION,
    MAX_BUNDLE_BYTES,
    build_crash_bundle,
    classify_crash,
    redact_secrets,
    write_crash_bundle,
)


class TestClassify:
    def test_oom(self):
        assert classify_crash("CUDA out of memory") == "oom"
        assert classify_crash("torch.cuda.OutOfMemoryError: …") == "oom"

    def test_nan(self):
        assert classify_crash("loss is NaN at step 100") == "nan"
        assert classify_crash("infinite loss detected") == "nan"

    def test_cuda(self):
        assert classify_crash("CUDA error: device-side assert triggered") == "cuda"

    def test_dataloader(self):
        assert classify_crash("DataLoader worker exited unexpectedly") == "dataloader"

    def test_nccl(self):
        assert classify_crash("NCCL error encountered") == "nccl"

    def test_other(self):
        assert classify_crash("ZeroDivisionError") == "other"

    def test_non_string(self):
        assert classify_crash(None) == "other"


class TestRedact:
    def test_hf_token(self):
        out = redact_secrets("token=hf_abcdef0123456789012345 broken")
        assert "hf_abcdef" not in out
        assert "<redacted>" in out

    def test_openai_key(self):
        out = redact_secrets("OPENAI_KEY=sk-abc1234567890123456789")
        assert "sk-abc" not in out
        assert "<redacted>" in out

    def test_bearer_token(self):
        out = redact_secrets("Authorization: Bearer abc12345678901234567890")
        assert "<redacted>" in out

    def test_no_change_when_clean(self):
        clean = "everything fine here"
        assert redact_secrets(clean) == clean


class TestBuildBundle:
    def test_minimal(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        assert bundle["format_version"] == CRASH_FORMAT_VERSION
        assert bundle["error"]["type"] == "RuntimeError"
        assert bundle["error"]["message"] == "boom"
        assert "traceback" in bundle["error"]
        assert "gpu_state" in bundle
        assert "environment" in bundle

    def test_classifies_kind(self):
        try:
            raise RuntimeError("CUDA out of memory; tried to allocate 80GB")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        assert bundle["error"]["kind"] == "oom"

    def test_redacts_secrets_in_error_message(self):
        try:
            raise RuntimeError("auth failed with hf_abcdef0123456789012345token")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        assert "hf_abcdef" not in bundle["error"]["message"]

    def test_metrics_tail_capped(self):
        rows = [{"step": i, "loss": 1.0 / (i + 1)} for i in range(200)]
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc, metrics=rows)
        assert len(bundle["metrics_tail"]) == 50
        assert bundle["metrics_tail"][-1]["step"] == 199

    def test_config_serialised(self):
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc, config={"base": "x", "lr": 1e-4})
        assert bundle["config"]["base"] == "x"

    def test_config_secrets_redacted(self):
        # H1: hf_TOKEN-shaped value inside config must be redacted.
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(
                error=exc,
                config={"hub": {"token": "hf_abcdef0123456789012345"}},
            )
        assert "hf_abcdef" not in str(bundle["config"])
        assert "<redacted>" in str(bundle["config"])

    def test_config_unserialisable_handled(self):
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc, config={"set": {1, 2, 3}})
        # Falls back to either serialised-via-default-str or unserialisable marker.
        assert "config" in bundle

    def test_output_dir_basename_only(self):
        # Output dir like /home/alice/.cache/x must not leak the path.
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(
                error=exc, output_dir="/home/alice/.cache/models/run_x",
            )
        assert "alice" not in str(bundle.get("output_dir") or "")
        assert bundle["output_dir"] == "run_x"


class TestWriteBundle:
    def test_writes_to_target_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "out"
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        path = write_crash_bundle(bundle, target_dir=target)
        assert path.exists()
        assert path.suffix == ".crash"
        assert path.parent == target.resolve()
        # Re-parse
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["error"]["type"] == "RuntimeError"

    def test_default_dir_is_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        path = write_crash_bundle(bundle)
        assert (tmp_path / ".soup-crashes") in path.parents

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside-crashes"
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        with pytest.raises(ValueError, match="not under cwd"):
            write_crash_bundle(bundle, target_dir=outside)

    def test_oversize_truncated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            bundle = build_crash_bundle(error=exc)
        # Stuff a giant string
        bundle["junk"] = "x" * (MAX_BUNDLE_BYTES + 100)
        path = write_crash_bundle(bundle, target_dir=tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("_truncated") is True
