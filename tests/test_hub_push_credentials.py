"""soup push: each hub authenticates with its own credential."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    # No cached HF login and no hub tokens unless a test sets one.
    empty = tmp_path / "home"
    empty.mkdir()
    for var in ("HOME", "USERPROFILE", "HF_HOME"):
        monkeypatch.setenv(var, str(empty))
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "MODELSCOPE_API_TOKEN", "MODELERS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return out


@pytest.fixture
def recorded(monkeypatch):
    calls: list[dict] = []

    def _fake_upload_repo(hub, repo, **kwargs):
        calls.append({"hub": hub, "repo": repo, **kwargs})

    monkeypatch.setattr("soup_cli.utils.hubs.upload_repo", _fake_upload_repo)
    return calls


def _push(*extra):
    from soup_cli.cli import app

    return runner.invoke(app, ["push", "--model", "out", "--repo", "user/m", *extra])


class TestHubTokenEnvVar:
    @pytest.mark.parametrize(
        ("hub", "var"),
        [("hf", "HF_TOKEN"), ("modelscope", "MODELSCOPE_API_TOKEN"),
         ("modelers", "MODELERS_TOKEN"), ("ModelScope", "MODELSCOPE_API_TOKEN")],
    )
    def test_maps_each_hub(self, hub, var):
        from soup_cli.utils.hubs import hub_token_env_var

        assert hub_token_env_var(hub) == var

    def test_unknown_hub_rejected(self):
        from soup_cli.utils.hubs import hub_token_env_var

        with pytest.raises(ValueError):
            hub_token_env_var("evilhub")

    def test_resolve_refuses_hf(self):
        from soup_cli.utils.hubs import resolve_hub_token

        with pytest.raises(ValueError):
            resolve_hub_token("hf")

    def test_resolve_ignores_hf_env(self, monkeypatch):
        from soup_cli.utils.hubs import resolve_hub_token

        monkeypatch.setenv("HF_TOKEN", "hf_SENTINEL")
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        assert resolve_hub_token("modelscope") is None


class TestNonHfPush:
    @pytest.mark.parametrize(("hub", "var"), [("modelscope", "MODELSCOPE_API_TOKEN"),
                                              ("modelers", "MODELERS_TOKEN")])
    def test_hf_env_token_never_reaches_other_hub(self, model_dir, recorded, monkeypatch,
                                                  hub, var):
        monkeypatch.setenv("HF_TOKEN", "hf_SENTINEL")
        monkeypatch.setenv(var, "hub_OWN")
        result = _push("--hub", hub)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert recorded and recorded[0]["token"] == "hub_OWN"
        assert "hf_SENTINEL" not in repr(recorded)

    @pytest.mark.parametrize(("hub", "var"), [("modelscope", "MODELSCOPE_API_TOKEN"),
                                              ("modelers", "MODELERS_TOKEN")])
    def test_only_hf_credential_is_refused_naming_hub_variable(self, model_dir, recorded,
                                                               monkeypatch, hub, var):
        monkeypatch.setenv("HF_TOKEN", "hf_SENTINEL")
        result = _push("--hub", hub)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert var in _clean(result.output)
        assert "hf_SENTINEL" not in result.output
        assert recorded == []

    def test_hf_cached_login_is_not_read(self, model_dir, recorded, monkeypatch):
        import soup_cli.utils.hf as hf_mod

        def _boom(*args, **kwargs):
            raise AssertionError("HF token resolution must not run for a non-HF hub")

        monkeypatch.setattr(hf_mod, "resolve_token", _boom)
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "hub_OWN")
        result = _push("--hub", "modelscope")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert recorded[0]["token"] == "hub_OWN"

    def test_command_line_token_is_used(self, model_dir, recorded, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_SENTINEL")
        result = _push("--hub", "modelscope", "--token", "ms_CLI")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert recorded[0]["token"] == "ms_CLI"

    def test_no_credentials_at_all_names_variable(self, model_dir, recorded):
        result = _push("--hub", "modelers")
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "MODELERS_TOKEN" in _clean(result.output)
        assert "HuggingFace" not in _clean(result.output)
        assert recorded == []


class TestHfPushUnchanged:
    def test_hf_hub_still_requires_hf_token(self, model_dir, recorded):
        result = _push()
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "No HuggingFace token found" in _clean(result.output)
