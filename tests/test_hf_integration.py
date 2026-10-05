"""Tests for HF Hub Deep Integration (v0.29.0)."""

from __future__ import annotations

import json
import re
import sys
import types
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

# Strip ANSI colour / style escape sequences before substring assertions —
# macOS/Windows Typer runs inject them even under pytest, and they split
# tokens like ``--push-as`` into ``-`` + ``-push-as`` across escape groups.
# Same pattern as ``test_eval_gate.py`` after the 899ad8e fix.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


# ---------------------------------------------------------------------------
# Part A: utils/hf.py — token / endpoint / repo_id resolution
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_env_hf_token_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() == "env-token"

    def test_cached_token_new_location(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        token_file = tmp_path / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("cached-token\n")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() == "cached-token"

    def test_cached_token_legacy_location(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        token_file = tmp_path / ".huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("legacy-token")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() == "legacy-token"

    def test_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        from soup_cli.utils.hf import resolve_token

        assert resolve_token(explicit="explicit-token") == "explicit-token"

    def test_huggingface_hub_token_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hh-token")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() == "hh-token"

    def test_no_token_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() is None


class TestResolveEndpoint:
    def test_default_endpoint(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        from soup_cli.utils.hf import resolve_endpoint

        assert resolve_endpoint() == "https://huggingface.co"

    def test_custom_endpoint_env(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf.internal.example.com")
        from soup_cli.utils.hf import resolve_endpoint

        assert resolve_endpoint() == "https://hf.internal.example.com"

    def test_endpoint_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf.internal.example.com/")
        from soup_cli.utils.hf import resolve_endpoint

        assert resolve_endpoint() == "https://hf.internal.example.com"

    def test_endpoint_rejects_non_https_remote(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "http://evil.example.com")
        from soup_cli.utils.hf import resolve_endpoint

        with pytest.raises(ValueError, match="HTTPS"):
            resolve_endpoint()

    def test_endpoint_allows_localhost_http(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8080")
        from soup_cli.utils.hf import resolve_endpoint

        assert resolve_endpoint() == "http://localhost:8080"

    def test_endpoint_rejects_null_byte(self, monkeypatch):
        # OS-level setenv rejects null bytes on macOS/Windows before
        # ``resolve_endpoint`` ever runs, so patch the dict directly.
        monkeypatch.setattr(
            "soup_cli.utils.hf.os.environ",
            {"HF_ENDPOINT": "https://hf.example.com\x00"},
        )
        from soup_cli.utils.hf import resolve_endpoint

        with pytest.raises(ValueError):
            resolve_endpoint()

    def test_endpoint_rejects_bad_scheme(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "file:///etc/passwd")
        from soup_cli.utils.hf import resolve_endpoint

        with pytest.raises(ValueError):
            resolve_endpoint()


class TestValidateRepoId:
    def test_valid_user_repo(self):
        from soup_cli.utils.hf import validate_repo_id

        validate_repo_id("user/my-model")

    def test_valid_repo_only(self):
        from soup_cli.utils.hf import validate_repo_id

        validate_repo_id("my-model")

    def test_rejects_empty(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("")

    def test_rejects_slash_at_start(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("/my-model")

    def test_rejects_double_slash(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("user//model")

    def test_rejects_path_traversal(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("../../secrets")

    def test_rejects_null_byte(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("user/model\x00")

    def test_rejects_whitespace(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("user/my model")

    def test_rejects_too_long(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("user/" + ("a" * 200))

    def test_accepts_dots_underscores_hyphens(self):
        from soup_cli.utils.hf import validate_repo_id

        validate_repo_id("user/my_model.v2-beta")


class TestGetHfApi:
    def test_requires_huggingface_hub(self, monkeypatch):
        from soup_cli.utils.hf import get_hf_api

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError):
            get_hf_api()

    def test_passes_token_and_endpoint(self, monkeypatch):
        from soup_cli.utils import hf

        fake_hub = types.ModuleType("huggingface_hub")

        calls = {}

        class FakeApi:
            def __init__(self, token=None, endpoint=None):
                calls["token"] = token
                calls["endpoint"] = endpoint

        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        hf.get_hf_api(token="t1", endpoint="https://e.example.com")
        assert calls == {"token": "t1", "endpoint": "https://e.example.com"}


# ---------------------------------------------------------------------------
# Part B: Auto-push checkpoints (--push-as flag)
# ---------------------------------------------------------------------------


class TestPushAsCLIFlag:
    def test_train_shows_push_as_flag_in_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert "--push-as" in _plain(result.output)

    def test_push_as_rejects_invalid_repo(self, tmp_path, monkeypatch):
        cfg = tmp_path / "soup.yaml"
        cfg.write_text("base: meta-llama/Llama-3.1-8B\ntask: sft\ndata:\n  train: data.jsonl\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(cfg),
                "--push-as",
                "../../escape",
                "--dry-run",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "push-as" in result.output.lower()


class TestHFPushCallback:
    def test_callback_import(self):
        from soup_cli.monitoring.hf_push import HFPushCallback  # noqa: F401

    def test_callback_uploads_checkpoint_on_save(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_config.json").write_text("{}")

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)

        callback = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir=str(tmp_path),
        )

        args = types.SimpleNamespace(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=100, epoch=1)
        control = types.SimpleNamespace()
        callback.on_save(args, state, control)

        fake_api.create_repo.assert_called_once_with(
            repo_id="user/my-model",
            private=False,
            exist_ok=True,
        )
        assert fake_api.upload_folder.called
        kwargs = fake_api.upload_folder.call_args.kwargs
        assert kwargs["repo_id"] == "user/my-model"
        assert kwargs["folder_path"] == str(ckpt_dir)
        assert kwargs["revision"] == "checkpoint-100"
        assert "checkpoint-100" in kwargs["commit_message"]

    def test_callback_swallows_upload_errors(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        ckpt_dir = tmp_path / "checkpoint-50"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_config.json").write_text("{}")

        fake_api = MagicMock()
        fake_api.upload_folder.side_effect = RuntimeError("network")
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)

        callback = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir=str(tmp_path),
        )

        args = types.SimpleNamespace(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=50, epoch=1)
        control = types.SimpleNamespace()
        callback.on_save(args, state, control)

    def test_callback_skips_missing_checkpoint(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)

        callback = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir=str(tmp_path),
        )

        args = types.SimpleNamespace(output_dir=str(tmp_path))
        state = types.SimpleNamespace(global_step=200, epoch=2)
        control = types.SimpleNamespace()
        callback.on_save(args, state, control)

        assert not fake_api.upload_folder.called


class TestResolveLatestRevision:
    def test_returns_latest_step(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import resolve_latest_checkpoint_revision

        fake_api = MagicMock()
        fake_api.list_repo_refs.return_value = types.SimpleNamespace(
            branches=[
                types.SimpleNamespace(name="main"),
                types.SimpleNamespace(name="checkpoint-50"),
                types.SimpleNamespace(name="checkpoint-150"),
                types.SimpleNamespace(name="checkpoint-100"),
            ]
        )
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)
        result = resolve_latest_checkpoint_revision("user/my-model", token="t1")
        assert result == "checkpoint-150"

    def test_returns_none_when_no_checkpoints(self, monkeypatch):
        from soup_cli.monitoring.hf_push import resolve_latest_checkpoint_revision

        fake_api = MagicMock()
        fake_api.list_repo_refs.return_value = types.SimpleNamespace(
            branches=[types.SimpleNamespace(name="main")]
        )
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)
        result = resolve_latest_checkpoint_revision("user/my-model", token="t1")
        assert result is None

    def test_returns_none_on_api_error(self, monkeypatch):
        from soup_cli.monitoring.hf_push import resolve_latest_checkpoint_revision

        fake_api = MagicMock()
        fake_api.list_repo_refs.side_effect = RuntimeError("no such repo")
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)
        result = resolve_latest_checkpoint_revision("user/my-model", token="t1")
        assert result is None


# ---------------------------------------------------------------------------
# Part C: Model Card v2
# ---------------------------------------------------------------------------


class TestModelCardV2:
    def test_generate_enhanced_card_basic(self, tmp_path):
        from soup_cli.commands.push import generate_model_card_v2

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
                    "r": 32,
                    "lora_alpha": 64,
                }
            )
        )
        card = generate_model_card_v2(adapter_dir, repo_id="user/my-model")
        assert "my-model" in card
        assert "meta-llama/Llama-3.1-8B" in card
        assert "soup-cli" in card
        assert "LoRA" in card

    def test_generate_full_model_card(self, tmp_path):
        from soup_cli.commands.push import generate_model_card_v2

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        card = generate_model_card_v2(model_dir, repo_id="user/full-model")
        assert "full-model" in card
        assert "soup-cli" in card

    def test_card_includes_training_config_when_present(self, tmp_path):
        from soup_cli.commands.push import generate_model_card_v2

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "training_config.yaml").write_text(
            "base: meta-llama/Llama-3.1-8B\ntask: sft\ntraining:\n  epochs: 3\n  lr: 0.0002\n"
        )
        card = generate_model_card_v2(adapter_dir, repo_id="user/my-model")
        assert "sft" in card or "Training" in card
        assert "0.0002" in card or "lr" in card

    def test_card_includes_eval_scorecard_when_registry_available(self, tmp_path):
        from soup_cli.commands.push import generate_model_card_v2

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        eval_data = {
            "mmlu": 0.612,
            "gsm8k": 0.812,
        }
        card = generate_model_card_v2(
            adapter_dir,
            repo_id="user/my-model",
            eval_scorecard=eval_data,
        )
        assert "mmlu" in card.lower()
        assert "0.612" in card or "61.2" in card

    def test_card_safe_against_malformed_config(self, tmp_path):
        from soup_cli.commands.push import generate_model_card_v2

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("not json")
        card = generate_model_card_v2(adapter_dir, repo_id="user/broken")
        assert "broken" in card


# ---------------------------------------------------------------------------
# Part D: HF Collections
# ---------------------------------------------------------------------------


class TestCollections:
    def test_push_shows_collection_flag_in_help(self):
        result = runner.invoke(app, ["push", "--help"])
        assert "--collection" in _plain(result.output)

    def test_add_to_collection_calls_api(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        add_to_collection(
            collection_slug="user/collection-abc123",
            repo_id="user/my-model",
            token="t1",
        )
        assert fake_api.add_collection_item.called
        kwargs = fake_api.add_collection_item.call_args.kwargs
        assert kwargs["collection_slug"] == "user/collection-abc123"
        assert kwargs["item_id"] == "user/my-model"
        assert kwargs["item_type"] == "model"

    def test_add_to_collection_rejects_invalid_slug(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        with pytest.raises(ValueError):
            add_to_collection(
                collection_slug="../../escape",
                repo_id="user/my-model",
                token="t1",
            )

    def test_add_to_collection_survives_duplicate(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        fake_api.add_collection_item.side_effect = RuntimeError("already exists")
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        add_to_collection(
            collection_slug="user/collection-abc123",
            repo_id="user/my-model",
            token="t1",
            ignore_duplicate=True,
        )


# ---------------------------------------------------------------------------
# Part E: HF Datasets write
# ---------------------------------------------------------------------------


class TestDataPush:
    def test_push_subcommand_exists(self):
        result = runner.invoke(app, ["data", "push", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--hf-dataset" in _plain(result.output)

    def test_push_rejects_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data",
                "push",
                "--input",
                "nonexistent.jsonl",
                "--hf-dataset",
                "user/my-dataset",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_push_rejects_invalid_dataset_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.jsonl").write_text('{"text":"a"}\n')
        result = runner.invoke(
            app,
            [
                "data",
                "push",
                "--input",
                "data.jsonl",
                "--hf-dataset",
                "../../escape",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "hf-dataset" in result.output.lower() or "repo" in result.output.lower()

    def test_push_rejects_path_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.jsonl"
        outside.write_text('{"text":"a"}\n')
        try:
            result = runner.invoke(
                app,
                [
                    "data",
                    "push",
                    "--input",
                    str(outside),
                    "--hf-dataset",
                    "user/my-dataset",
                ],
            )
            assert result.exit_code == 1, (result.output, repr(result.exception))
            assert "current working directory" in result.output.lower()
        finally:
            outside.unlink(missing_ok=True)

    def test_push_requires_token(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no-creds")
        (tmp_path / "data.jsonl").write_text('{"text":"a"}\n')
        result = runner.invoke(
            app,
            [
                "data",
                "push",
                "--input",
                "data.jsonl",
                "--hf-dataset",
                "user/my-dataset",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "token" in result.output.lower()

    def test_push_dataset_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        (tmp_path / "data.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n')

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)

        result = runner.invoke(
            app,
            [
                "data",
                "push",
                "--input",
                "data.jsonl",
                "--hf-dataset",
                "user/my-dataset",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert fake_api.create_repo.called
        assert fake_api.upload_file.called or fake_api.upload_folder.called


# ---------------------------------------------------------------------------
# Part F: HF Spaces auto-deploy
# ---------------------------------------------------------------------------


class TestDeployHfSpace:
    def test_hf_space_command_registered(self):
        result = runner.invoke(app, ["deploy", "--help"])
        assert "hf-space" in _plain(result.output)

    def test_hf_space_help_shows_flags(self):
        result = runner.invoke(app, ["deploy", "hf-space", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = _plain(result.output)
        assert "--model" in plain
        assert "--template" in plain

    def test_hf_space_rejects_invalid_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        result = runner.invoke(
            app,
            [
                "deploy",
                "hf-space",
                "--model",
                "user/my-model",
                "--space",
                "../../escape",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "space" in result.output.lower()

    def test_hf_space_rejects_unknown_template(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        result = runner.invoke(
            app,
            [
                "deploy",
                "hf-space",
                "--model",
                "user/my-model",
                "--space",
                "user/my-space",
                "--template",
                "rocket-launcher",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "template" in result.output.lower()

    def test_hf_space_requires_token(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no-creds")
        result = runner.invoke(
            app,
            [
                "deploy",
                "hf-space",
                "--model",
                "user/my-model",
                "--space",
                "user/my-space",
                "--template",
                "gradio-chat",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "token" in result.output.lower()

    def test_hf_space_happy_path_gradio_chat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)

        result = runner.invoke(
            app,
            [
                "deploy",
                "hf-space",
                "--model",
                "user/my-model",
                "--space",
                "user/my-space",
                "--template",
                "gradio-chat",
                "--yes",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert fake_api.create_repo.called

    def test_render_gradio_chat_template(self):
        from soup_cli.commands.deploy import render_space_template

        rendered = render_space_template("gradio-chat", model_repo="user/my-model")
        assert "gradio" in rendered["app.py"].lower()
        assert "user/my-model" in rendered["app.py"]
        assert "requirements.txt" in rendered
        assert "README.md" in rendered

    def test_render_streamlit_chat_template(self):
        from soup_cli.commands.deploy import render_space_template

        rendered = render_space_template("streamlit-chat", model_repo="user/my-model")
        assert "streamlit" in rendered["app.py"].lower()
        assert "user/my-model" in rendered["app.py"]

    def test_render_unknown_template_raises(self):
        from soup_cli.commands.deploy import render_space_template

        with pytest.raises(ValueError):
            render_space_template("bogus", model_repo="user/my-model")

    def test_readme_sets_sdk_to_match_template(self):
        from soup_cli.commands.deploy import render_space_template

        gradio = render_space_template("gradio-chat", model_repo="user/my-model")
        assert "sdk: gradio" in gradio["README.md"]
        streamlit = render_space_template("streamlit-chat", model_repo="user/my-model")
        assert "sdk: streamlit" in streamlit["README.md"]

    def test_template_escapes_model_repo_id(self):
        from soup_cli.commands.deploy import render_space_template

        with pytest.raises(ValueError):
            render_space_template("gradio-chat", model_repo="../../escape")


# ---------------------------------------------------------------------------
# Auto-resume from HF (Part B)
# ---------------------------------------------------------------------------


class TestAutoResume:
    def test_train_help_shows_hf_resume(self):
        result = runner.invoke(app, ["train", "--help"])
        assert "--hf-resume" in _plain(result.output)

    def test_prepare_hf_resume_downloads_latest(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import prepare_hf_resume

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "runs" / "experiment"
        out_dir.mkdir(parents=True)

        called = {}

        def fake_download(repo_id, revision, local_dir, token, endpoint):
            called["repo_id"] = repo_id
            called["revision"] = revision
            called["local_dir"] = local_dir
            return local_dir

        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.resolve_latest_checkpoint_revision",
            lambda repo_id, token=None, endpoint=None: "checkpoint-300",
        )
        monkeypatch.setattr("soup_cli.monitoring.hf_push._download_checkpoint", fake_download)

        result = prepare_hf_resume(
            repo_id="user/my-model",
            output_dir=str(out_dir),
            token="t1",
        )
        assert result is not None
        assert called["revision"] == "checkpoint-300"

    def test_prepare_hf_resume_no_checkpoint_returns_none(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import prepare_hf_resume

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.resolve_latest_checkpoint_revision",
            lambda repo_id, token=None, endpoint=None: None,
        )
        result = prepare_hf_resume(
            repo_id="user/my-model",
            output_dir=str(out_dir),
            token="t1",
        )
        assert result is None

    def test_prepare_hf_resume_rejects_output_dir_outside_cwd(self, monkeypatch, tmp_path):
        import pytest

        from soup_cli.monitoring.hf_push import prepare_hf_resume

        cwd = tmp_path / "project"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        with pytest.raises(ValueError, match="under the current"):
            prepare_hf_resume(
                repo_id="user/my-model",
                output_dir=str(outside),
                token="t1",
            )


# ---------------------------------------------------------------------------
# Integration: utils/hf consolidated behavior
# ---------------------------------------------------------------------------


class TestHfUtilsModule:
    def test_module_exports(self):
        from soup_cli.utils import hf

        assert hasattr(hf, "resolve_token")
        assert hasattr(hf, "resolve_endpoint")
        assert hasattr(hf, "validate_repo_id")
        assert hasattr(hf, "get_hf_api")
        assert hasattr(hf, "add_to_collection")

    def test_token_with_whitespace_stripped(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        token_file = tmp_path / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("  padded-token  \n")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() == "padded-token"


# ---------------------------------------------------------------------------
# Extra coverage from TDD review: collection-slug negatives, callback factory,
# on_train_begin, short-circuit after repo failure, private-IP SSRF, edge
# cases around resolve_token.
# ---------------------------------------------------------------------------


class TestValidateCollectionSlug:
    def test_accepts_valid(self):
        from soup_cli.utils.hf import validate_collection_slug

        validate_collection_slug("user/my-collection-abc12345")

    def test_rejects_empty(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("")

    def test_rejects_whitespace(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("user/my coll")

    def test_rejects_null_byte(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("user/my-coll\x00")

    def test_rejects_path_traversal(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("../../escape")

    def test_rejects_too_long(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("user/" + ("a" * 300))

    def test_rejects_missing_slash(self):
        from soup_cli.utils.hf import validate_collection_slug

        with pytest.raises(ValueError):
            validate_collection_slug("onlyname")


class TestAddToCollectionExtras:
    def test_raises_on_duplicate_when_ignore_false(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        fake_api.add_collection_item.side_effect = RuntimeError("already exists")
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        with pytest.raises(RuntimeError):
            add_to_collection(
                collection_slug="user/my-coll-abc12345",
                repo_id="user/my-model",
                token="t1",
                ignore_duplicate=False,
            )

    def test_raises_on_unrelated_error(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        fake_api.add_collection_item.side_effect = RuntimeError("auth required")
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        with pytest.raises(RuntimeError):
            add_to_collection(
                collection_slug="user/my-coll-abc12345",
                repo_id="user/my-model",
                token="t1",
            )

    def test_rejects_invalid_item_type(self, monkeypatch):
        from soup_cli.utils.hf import add_to_collection

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        with pytest.raises(ValueError, match="item_type"):
            add_to_collection(
                collection_slug="user/my-coll-abc12345",
                repo_id="user/my-model",
                token="t1",
                item_type="notebook",
            )


class TestBuildPushCallback:
    def test_returns_none_when_no_token(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import build_push_callback

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no-creds")
        cb = build_push_callback(repo_id="user/my-model", output_dir=str(tmp_path))
        assert cb is None

    def test_returns_none_on_bad_endpoint(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import build_push_callback

        monkeypatch.setenv("HF_TOKEN", "t1")
        monkeypatch.setenv("HF_ENDPOINT", "http://evil.example.com")
        cb = build_push_callback(repo_id="user/my-model", output_dir=str(tmp_path))
        assert cb is None

    def test_happy_path_returns_callback(self, monkeypatch, tmp_path):
        from soup_cli.monitoring.hf_push import HFPushCallback, build_push_callback

        monkeypatch.setenv("HF_TOKEN", "t1")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        cb = build_push_callback(repo_id="user/my-model", output_dir=str(tmp_path))
        assert isinstance(cb, HFPushCallback)
        assert cb.repo_id == "user/my-model"


class TestCallbackLifecycle:
    def _fake_api(self, monkeypatch):
        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.monitoring.hf_push.get_hf_api", lambda **_: fake_api)
        return fake_api

    def test_on_train_begin_creates_repo(self, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        fake_api = self._fake_api(monkeypatch)
        cb = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir="/tmp",
        )
        cb.on_train_begin(
            args=types.SimpleNamespace(output_dir="/tmp"),
            state=types.SimpleNamespace(),
            control=types.SimpleNamespace(),
        )
        fake_api.create_repo.assert_called_once_with(
            repo_id="user/my-model",
            private=False,
            exist_ok=True,
        )

    def test_on_train_begin_swallows_failure(self, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        fake_api = self._fake_api(monkeypatch)
        fake_api.create_repo.side_effect = RuntimeError("auth")
        cb = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir="/tmp",
        )
        cb.on_train_begin(
            args=types.SimpleNamespace(output_dir="/tmp"),
            state=types.SimpleNamespace(),
            control=types.SimpleNamespace(),
        )
        assert cb._repo_failed is True
        assert cb._repo_created is False

    def test_create_repo_called_once_across_saves(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        fake_api = self._fake_api(monkeypatch)
        for step in (50, 100, 150):
            ckpt = tmp_path / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "adapter_config.json").write_text("{}")

        cb = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir=str(tmp_path),
        )
        for step in (50, 100, 150):
            cb.on_save(
                args=types.SimpleNamespace(output_dir=str(tmp_path)),
                state=types.SimpleNamespace(global_step=step, epoch=1),
                control=types.SimpleNamespace(),
            )
        assert fake_api.create_repo.call_count == 1
        assert fake_api.upload_folder.call_count == 3

    def test_no_upload_after_repo_creation_fails(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.hf_push import HFPushCallback

        fake_api = self._fake_api(monkeypatch)
        fake_api.create_repo.side_effect = RuntimeError("auth forbidden")

        ckpt = tmp_path / "checkpoint-10"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}")

        cb = HFPushCallback(
            repo_id="user/my-model",
            token="t1",
            output_dir=str(tmp_path),
        )
        cb.on_train_begin(
            args=types.SimpleNamespace(output_dir=str(tmp_path)),
            state=types.SimpleNamespace(),
            control=types.SimpleNamespace(),
        )
        cb.on_save(
            args=types.SimpleNamespace(output_dir=str(tmp_path)),
            state=types.SimpleNamespace(global_step=10, epoch=1),
            control=types.SimpleNamespace(),
        )
        assert fake_api.upload_folder.call_count == 0


class TestPrivateIPSSRF:
    @pytest.mark.parametrize(
        "host",
        [
            "http://10.0.0.1",
            "http://192.168.1.1",
            "http://172.16.0.1",
            "http://169.254.169.254",
        ],
    )
    def test_private_ip_http_rejected(self, monkeypatch, host):
        monkeypatch.setenv("HF_ENDPOINT", host)
        from soup_cli.utils.hf import resolve_endpoint

        with pytest.raises(ValueError):
            resolve_endpoint()

    def test_zero_address_rejected(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "http://0.0.0.0:8080")
        from soup_cli.utils.hf import resolve_endpoint

        with pytest.raises(ValueError, match="0.0.0.0"):
            resolve_endpoint()

    def test_127_allowed(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:8080")
        from soup_cli.utils.hf import resolve_endpoint

        assert resolve_endpoint() == "http://127.0.0.1:8080"


class TestResolveTokenEdgeCases:
    def test_whitespace_env_falls_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "   ")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no-creds")
        from soup_cli.utils.hf import resolve_token

        assert resolve_token() is None

    def test_empty_explicit_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-env")
        from soup_cli.utils.hf import resolve_token

        assert resolve_token(explicit="") == "from-env"

    def test_explicit_non_printable_raises(self):
        from soup_cli.utils.hf import resolve_token

        with pytest.raises(ValueError):
            resolve_token(explicit="abc\x01\x02")


class TestValidateRepoIdBoundaries:
    def test_accepts_max_length_component(self):
        from soup_cli.utils.hf import validate_repo_id

        validate_repo_id("u/" + ("a" * 96))

    def test_rejects_oversized_component(self):
        from soup_cli.utils.hf import validate_repo_id

        with pytest.raises(ValueError):
            validate_repo_id("u/" + ("a" * 97))


class TestEvalScorecardEdgeCases:
    def test_non_numeric_score(self):
        from soup_cli.commands.push import _render_eval_scorecard

        rendered = _render_eval_scorecard({"gsm8k": "N/A"})
        assert "N/A" in rendered
        assert "gsm8k" in rendered

    def test_escapes_pipe_in_task_name(self):
        from soup_cli.commands.push import _render_eval_scorecard

        rendered = _render_eval_scorecard({"math|injection": 0.5})
        # The pipe must be neutralised to avoid breaking the markdown table.
        for line in rendered.splitlines():
            if "0.500" in line:
                # Table row — exactly 3 unescaped pipes: leading, middle, trailing.
                assert line.count("|") == 3

    def test_escapes_markdown_injection(self):
        from soup_cli.commands.push import _render_eval_scorecard

        rendered = _render_eval_scorecard({"task\n| fake | 0.99": 0.5})
        assert "fake" in rendered  # literal preserved as text
        # The injection newline / pipe must be neutralised.
        assert "\n| fake" not in rendered


class TestHfResumeRequiresPushAs:
    def test_hf_resume_alone_exits_one(self, tmp_path, monkeypatch):
        cfg = tmp_path / "soup.yaml"
        cfg.write_text("base: meta-llama/Llama-3.1-8B\ntask: sft\ndata:\n  train: data.jsonl\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["train", "--config", str(cfg), "--hf-resume", "--dry-run"],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "push-as" in result.output.lower()


class TestReLoRAResumeRefusal:
    def _write_config(self, path):
        path.write_text(
            "base: meta-llama/Llama-3.1-8B\n"
            "task: sft\n"
            "data:\n"
            "  train: data.jsonl\n"
            "training:\n"
            "  quantization: none\n"
            "  relora_steps: 1\n",
            encoding="utf-8",
        )

    def test_local_resume_is_rejected_before_checkpoint_resolution(self, tmp_path, monkeypatch):
        cfg = tmp_path / "soup.yaml"
        self._write_config(cfg)
        monkeypatch.setattr(
            "soup_cli.commands.train._resolve_resume_or_exit",
            lambda *_args, **_kwargs: pytest.fail("checkpoint resolution should not run"),
        )
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(cfg),
                "--resume",
                str(tmp_path / "checkpoint"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "cannot resume" in result.output.lower()
        assert "relora_steps" in result.output

    def test_hf_resume_is_rejected_before_download(self, tmp_path, monkeypatch):
        cfg = tmp_path / "soup.yaml"
        self._write_config(cfg)
        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.prepare_hf_resume",
            lambda *_args, **_kwargs: pytest.fail("HF resume should not run"),
        )
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(cfg),
                "--push-as",
                "user/model",
                "--hf-resume",
                "--dry-run",
            ],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "cannot resume" in result.output.lower()
        assert "relora_steps" in result.output


class TestDataPushHappyPathExtras:
    def test_create_repo_called_with_dataset_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")
        (tmp_path / "data.jsonl").write_text('{"text":"a"}\n')

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)
        monkeypatch.setattr(
            "soup_cli.commands.data.get_hf_api", lambda **_: fake_api, raising=False
        )

        result = runner.invoke(
            app,
            [
                "data",
                "push",
                "--input",
                "data.jsonl",
                "--hf-dataset",
                "user/my-dataset",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        fake_api.create_repo.assert_called_once_with(
            repo_id="user/my-dataset",
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )


class TestHfSpaceHappyPathExtras:
    def test_create_repo_called_with_space_type_and_sdk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "t1")

        fake_api = MagicMock()
        monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: fake_api)

        result = runner.invoke(
            app,
            [
                "deploy",
                "hf-space",
                "--model",
                "user/my-model",
                "--space",
                "user/my-space",
                "--template",
                "gradio-chat",
                "--yes",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        fake_api.create_repo.assert_called_once_with(
            repo_id="user/my-space",
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
