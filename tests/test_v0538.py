"""v0.53.8 — Remote data + Hubs + Trackers (wave 2).

Tests cover:
* #85 — fsspec live loaders (data/loader.py)
* #130 — Live hub download/upload dispatcher (utils/hubs.py)
* #89 — `[trackers]` extra + missing-dep advisory
* #90 — PostHog telemetry network (best-effort, silent-fail)
* #93 — package-data migration (soup_cli/data/_fixtures/)
* #69 — HF Space SDK auto-pick from requirements.txt
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ----------------------------------------------------------------------
# #69 — HF Space SDK auto-pick
# ----------------------------------------------------------------------


class TestDetectSpaceSdk:
    def _import(self):
        from soup_cli.utils.hf_space import detect_space_sdk

        return detect_space_sdk

    def test_none_defaults_to_gradio(self):
        assert self._import()(None) == "gradio"

    def test_empty_defaults_to_gradio(self):
        assert self._import()("") == "gradio"

    def test_streamlit_picked(self):
        assert self._import()("streamlit>=1.0\nrequests") == "streamlit"

    def test_gradio_picked(self):
        assert self._import()("gradio==4.0") == "gradio"

    def test_streamlit_wins_when_first(self):
        # First match wins
        assert self._import()("streamlit\ngradio") == "streamlit"

    def test_comment_ignored(self):
        assert self._import()("# streamlit\ngradio") == "gradio"

    def test_no_match_defaults_gradio(self):
        assert self._import()("torch\nnumpy") == "gradio"

    def test_non_string_defaults_gradio(self):
        assert self._import()(123) == "gradio"  # type: ignore[arg-type]

    def test_with_extras(self):
        assert self._import()("streamlit[all]>=1.0") == "streamlit"

    def test_uppercase_normalised(self):
        assert self._import()("STREAMLIT==1.0") == "streamlit"

    def test_oversize_input_degrades_to_default(self):
        # security-review LOW — size cap defends against pathological input
        huge = "streamlit\n" + ("x" * (260 * 1024))
        assert self._import()(huge) == "gradio"


class TestIsSupportedSpaceSdk:
    def test_known(self):
        from soup_cli.utils.hf_space import is_supported_space_sdk

        assert is_supported_space_sdk("gradio")
        assert is_supported_space_sdk("streamlit")
        assert is_supported_space_sdk("docker")
        assert is_supported_space_sdk("static")

    def test_unknown(self):
        from soup_cli.utils.hf_space import is_supported_space_sdk

        assert not is_supported_space_sdk("flask")
        assert not is_supported_space_sdk("")
        assert not is_supported_space_sdk(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# #130 — Hub download / upload dispatcher
# ----------------------------------------------------------------------


class TestDownloadRepoValidation:
    def test_unknown_hub_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="not supported"):
            download_repo("evil_hub", "owner/repo", local_dir="./snap_v0538")

    def test_empty_repo_id_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="non-empty"):
            download_repo("hf", "", local_dir="./snap_v0538")

    def test_null_byte_repo_id_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="null bytes"):
            download_repo("hf", "owner\x00/repo", local_dir="./snap_v0538")

    def test_bool_repo_id_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(TypeError, match="bool"):
            download_repo("hf", True, local_dir="./snap_v0538")  # type: ignore[arg-type]

    def test_leading_slash_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="path separator"):
            download_repo("hf", "/etc/passwd", local_dir="./snap_v0538")

    def test_dotdot_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match=".."):
            download_repo("hf", "../escape", local_dir="./snap_v0538")

    def test_control_char_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="control"):
            download_repo("hf", "owner/repo\n", local_dir="./snap_v0538")

    def test_oversize_repo_id_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="too long"):
            download_repo("hf", "a" * 250, local_dir="./snap_v0538")

    def test_empty_local_dir_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="local_dir"):
            download_repo("hf", "owner/repo", local_dir="")

    def test_local_dir_outside_cwd_raises(self, tmp_path):
        # security-review HIGH — containment check on local_dir
        from soup_cli.utils.hubs import download_repo

        # tmp_path is a sibling of cwd, not under it
        out = str(tmp_path / "snap")
        with pytest.raises(ValueError, match="under the current working"):
            download_repo("hf", "owner/repo", local_dir=out)

    def test_local_dir_bool_rejected(self):
        # security-review LOW — bool before str check
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(TypeError, match="bool"):
            download_repo("hf", "owner/repo", local_dir=True)  # type: ignore[arg-type]

    def test_bad_repo_type_raises(self):
        from soup_cli.utils.hubs import download_repo

        with pytest.raises(ValueError, match="repo_type"):
            download_repo("hf", "owner/repo", local_dir="./snap_v0538", repo_type="evil")


class TestDownloadRepoLazyImport:
    def test_modelscope_missing_friendly_error(self, monkeypatch):
        from soup_cli.utils import hubs

        # Block modelscope import even if installed
        monkeypatch.setitem(sys.modules, "modelscope", None)
        with pytest.raises(ImportError, match="modelscope"):
            hubs.download_repo("modelscope", "owner/repo", local_dir="./snap_v0538")

    def test_modelers_missing_friendly_error(self, monkeypatch):
        from soup_cli.utils import hubs

        monkeypatch.setitem(sys.modules, "openmind_hub", None)
        with pytest.raises(ImportError, match="openmind-hub"):
            hubs.download_repo("modelers", "owner/repo", local_dir="./snap_v0538")

    def test_hf_dispatch_calls_snapshot_download(self):
        with patch(
            "huggingface_hub.snapshot_download", return_value="/local/snap"
        ) as mocked:
            from soup_cli.utils.hubs import download_repo

            result = download_repo("hf", "owner/repo", local_dir="./snap_v0538")
            assert result == "/local/snap"
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            assert kwargs["repo_id"] == "owner/repo"
            assert kwargs["local_dir"] == "./snap_v0538"


class TestUploadRepoValidation:
    def test_unknown_hub_raises(self):
        from soup_cli.utils.hubs import upload_repo

        with pytest.raises(ValueError, match="not supported"):
            upload_repo("evil", "o/r", folder_path="./folder_v0538")

    def test_empty_folder_raises(self):
        from soup_cli.utils.hubs import upload_repo

        with pytest.raises(ValueError, match="folder_path"):
            upload_repo("hf", "o/r", folder_path="")

    def test_folder_path_outside_cwd_raises(self, tmp_path):
        # security-review HIGH — containment check on folder_path
        from soup_cli.utils.hubs import upload_repo

        out = str(tmp_path / "out")
        with pytest.raises(ValueError, match="under the current working"):
            upload_repo("hf", "o/r", folder_path=out)

    def test_empty_commit_raises(self):
        from soup_cli.utils.hubs import upload_repo

        with pytest.raises(ValueError, match="commit_message"):
            upload_repo("hf", "o/r", folder_path="./folder_v0538", commit_message="")

    def test_commit_message_truncated(self):
        # First line + 200 char cap (mirrors v0.29.0 push policy)
        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            from soup_cli.utils.hubs import upload_repo

            long_msg = "line1\nline2" + "x" * 500
            upload_repo("hf", "o/r", folder_path="./folder_v0538", commit_message=long_msg)
            sent = mock_api.upload_folder.call_args.kwargs["commit_message"]
            assert "\n" not in sent
            assert len(sent) <= 200


# ----------------------------------------------------------------------
# #90 — PostHog telemetry network
# ----------------------------------------------------------------------


class TestSendTelemetryPayload:
    def test_disabled_short_circuits(self, monkeypatch):
        monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload({"command": "train"}) is False

    def test_empty_payload_returns_false(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload({}) is False

    def test_non_dict_returns_false(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload("not a dict") is False  # type: ignore[arg-type]

    def test_http_endpoint_rejected(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert (
            send_telemetry_payload(
                {"command": "train"}, endpoint="http://evil.example/i/"
            )
            is False
        )

    def test_private_ip_endpoint_rejected(self, monkeypatch):
        # security-review MEDIUM — SSRF via private IP override
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert (
            send_telemetry_payload(
                {"command": "train"}, endpoint="https://10.0.0.1/i/"
            )
            is False
        )

    def test_link_local_endpoint_rejected(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert (
            send_telemetry_payload(
                {"command": "train"}, endpoint="https://169.254.169.254/i/"
            )
            is False
        )

    def test_bool_timeout_rejected(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload({"command": "train"}, timeout=True) is False  # type: ignore[arg-type]

    def test_negative_timeout_rejected(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload({"command": "train"}, timeout=-1) is False

    def test_httpx_missing_returns_false(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        # Force ImportError on the lazy import
        monkeypatch.setitem(sys.modules, "httpx", None)
        from soup_cli.utils.trackers import send_telemetry_payload

        assert send_telemetry_payload({"command": "train"}) is False

    def test_happy_path_2xx(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")

        fake_httpx = MagicMock()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_httpx.post.return_value = fake_resp
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        from soup_cli.utils.trackers import send_telemetry_payload

        result = send_telemetry_payload({"command": "train", "soup_version": "x"})
        assert result is True
        # HTTPS-only + 1s timeout
        call = fake_httpx.post.call_args
        assert call.args[0].startswith("https://")
        assert call.kwargs["timeout"] == 1.0

    def test_network_exception_swallowed(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")

        fake_httpx = MagicMock()
        fake_httpx.post.side_effect = RuntimeError("network down")
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        from soup_cli.utils.trackers import send_telemetry_payload

        # Must never raise
        assert send_telemetry_payload({"command": "train"}) is False


# ----------------------------------------------------------------------
# #89 — Tracker missing-dep advisory
# ----------------------------------------------------------------------


class TestTrackerMissingDepMessage:
    def test_wandb_returns_none(self):
        from soup_cli.utils.trackers import tracker_missing_dep_message

        # Legacy backend — never advise
        assert tracker_missing_dep_message("wandb") is None

    def test_tensorboard_returns_none(self):
        from soup_cli.utils.trackers import tracker_missing_dep_message

        assert tracker_missing_dep_message("tensorboard") is None

    def test_unknown_returns_none(self):
        from soup_cli.utils.trackers import tracker_missing_dep_message

        assert tracker_missing_dep_message("evil") is None

    def test_non_string_returns_none(self):
        from soup_cli.utils.trackers import tracker_missing_dep_message

        assert tracker_missing_dep_message(123) is None  # type: ignore[arg-type]

    def test_missing_mlflow_advisory(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", None)
        from soup_cli.utils.trackers import tracker_missing_dep_message

        msg = tracker_missing_dep_message("mlflow")
        assert msg is not None
        assert "mlflow" in msg
        assert "soup-cli[trackers]" in msg

    def test_present_mlflow_returns_none(self, monkeypatch):
        # Pretend mlflow is installed
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        from soup_cli.utils.trackers import tracker_missing_dep_message

        assert tracker_missing_dep_message("mlflow") is None


# ----------------------------------------------------------------------
# #85 — fsspec live remote loader
# ----------------------------------------------------------------------


class TestRemoteLoader:
    def test_looks_like_remote_uri(self):
        from soup_cli.data.loader import _looks_like_remote_uri

        assert _looks_like_remote_uri("s3://bucket/path")
        assert _looks_like_remote_uri("gs://bucket/file.jsonl")
        assert _looks_like_remote_uri("oci://bkt/x")
        assert not _looks_like_remote_uri("local.jsonl")
        assert not _looks_like_remote_uri("owner/dataset")
        assert not _looks_like_remote_uri("")
        assert not _looks_like_remote_uri(None)  # type: ignore[arg-type]

    def test_fsspec_missing_raises_friendly(self, monkeypatch, tmp_path):
        # Force fsspec ImportError
        monkeypatch.setitem(sys.modules, "fsspec", None)
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.loader import _load_remote_dataset

        cfg = DataConfig(train="s3://my-bucket/data.jsonl", format="alpaca", val_split=0)
        with pytest.raises(ImportError):
            _load_remote_dataset("s3://my-bucket/data.jsonl", cfg)

    def test_invalid_remote_uri_rejected(self, monkeypatch):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.loader import _load_remote_dataset

        cfg = DataConfig(train="s3://bucket/x", format="alpaca", val_split=0)
        # Userinfo embedded URI should be rejected by validate_remote_uri
        # BEFORE fsspec is even imported.
        with pytest.raises(ValueError):
            _load_remote_dataset("s3://user:pw@bucket/x", cfg)

    def test_non_streaming_reads_jsonl(self, monkeypatch, tmp_path):
        from soup_cli.config.schema import DataConfig

        # Build a fake fsspec that yields two JSONL rows
        rows = [
            '{"instruction": "hi", "output": "hello"}',
            '{"instruction": "bye", "output": "later"}',
        ]

        class FakeFile:
            def __enter__(self):
                return iter([r + "\n" for r in rows])

            def __exit__(self, *a):
                return False

        fake_fsspec = MagicMock()
        fake_fsspec.open.return_value = FakeFile()
        monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)

        from soup_cli.data.loader import _load_remote_dataset

        cfg = DataConfig(
            train="s3://bucket/data.jsonl", format="alpaca", val_split=0
        )
        result = _load_remote_dataset("s3://bucket/data.jsonl", cfg)
        assert "train" in result
        assert len(result["train"]) == 2


# ----------------------------------------------------------------------
# #93 — Package-data fixture migration
# ----------------------------------------------------------------------


class TestPackageDataFixtures:
    def test_fixtures_dir_exists(self):
        import soup_cli

        pkg = Path(soup_cli.__file__).parent
        fixtures = pkg / "data" / "_fixtures"
        assert fixtures.is_dir(), f"missing package-data fixtures dir: {fixtures}"

    def test_all_known_bundles_present(self):
        import soup_cli

        pkg = Path(soup_cli.__file__).parent
        fixtures = pkg / "data" / "_fixtures"
        expected = {
            "alpaca_tiny.jsonl",
            "chat_preferences.jsonl",
            "dpo_sample.jsonl",
            "reasoning_math.jsonl",
        }
        present = {p.name for p in fixtures.glob("*.jsonl")}
        assert expected.issubset(present)

    def test_bundle_source_prefers_package_data(self):
        from soup_cli.utils.demo_bundles import _bundle_source_path, get_bundle

        bundle = get_bundle("alpaca_demo")
        src = _bundle_source_path(bundle)
        # Should resolve under soup_cli/data/_fixtures (not examples/data/).
        assert os.sep + "_fixtures" + os.sep in src, src

    def test_bundle_content_valid_jsonl(self):
        from soup_cli.utils.demo_bundles import _bundle_source_path, get_bundle

        for name in ("alpaca_demo", "sharegpt_demo", "dpo_demo", "grpo_demo"):
            bundle = get_bundle(name)
            src = _bundle_source_path(bundle)
            with open(src, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        json.loads(line)


# ----------------------------------------------------------------------
# #130 wiring smoke (CLI flag plumbing)
# ----------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    import re as _re

    # Strip ANSI escape sequences AND Rich-wrapping whitespace so we can
    # robustly assert on terminal-rendered Typer help output.
    out = _re.sub(r"\x1b\[[0-9;]*m", "", text)
    return _re.sub(r"\s+", " ", out)


class TestDataDownloadHubFlag:
    def test_help_lists_hub_flag(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        result = CliRunner().invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        assert "--hub" in _strip_ansi(result.output)

    def test_unknown_hub_rejected(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        result = CliRunner().invoke(
            app, ["download", "ds", "--hub", "evilcorp"]
        )
        assert result.exit_code != 0
        clean = _strip_ansi(result.output)
        assert "evilcorp" in clean or "not supported" in clean

    def test_modelscope_hub_advisory(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        result = CliRunner().invoke(
            app, ["download", "ds", "--hub", "modelscope"]
        )
        assert result.exit_code != 0
        clean = _strip_ansi(result.output)
        assert "modelscope" in clean
        assert "v0.53.9" in clean or "download_repo" in clean


# ----------------------------------------------------------------------
# pyproject extras
# ----------------------------------------------------------------------


def _repo_root() -> Path:
    """Resolve the repo root regardless of pytest cwd (CI quirk).

    Tests are run from various cwds across the matrix; ``pyproject.toml``
    sits next to the ``tests/`` folder, so derive from this file's path.
    """
    return Path(__file__).resolve().parent.parent


class TestPyprojectExtras:
    def test_trackers_extra_present(self):
        text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        assert "trackers = [" in text
        assert "mlflow" in text
        assert "swanlab" in text
        assert "trackio" in text

    def test_remote_extra_present(self):
        text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        assert "remote = [" in text
        assert "fsspec" in text

    def test_force_include_package_data(self):
        text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        assert "_fixtures" in text


# ----------------------------------------------------------------------
# Version bump
# ----------------------------------------------------------------------


class TestVersionBump:
    def test_init_version(self):
        import soup_cli

        # v0.53.8.1 is the PyPI-recoverable patch (force-include duplicate
        # fix); accept either as the shipped string.
        assert soup_cli.__version__.startswith("0.53.8")

    def test_pyproject_version(self):
        text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "0.53.8' in text
