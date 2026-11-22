"""soup push: each hub authenticates with its own credential."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.utils import hf
from tests.conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(strip_ansi(text).split())


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


@pytest.fixture
def hf_api_tokens(monkeypatch):
    """Record the token handed to the HF client and stop before any network call."""
    seen: list = []

    def _fake_get_hf_api(token=None, endpoint=None):
        seen.append(token)
        raise ImportError("stop before upload")

    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", _fake_get_hf_api)
    return seen


class TestHfTokenDeprecation:
    """--token also reads HF_TOKEN via envvar; only a typed flag is deprecated (#2792)."""

    _DEPRECATION = "--token is deprecated"

    def test_hf_token_from_env_prints_no_deprecation(self, model_dir, hf_api_tokens,
                                                     monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
        result = _push()
        assert self._DEPRECATION not in _clean(result.output)
        # The env token is still the one used: resolution is unchanged.
        assert hf_api_tokens == ["hf_FROM_ENV"]

    @pytest.mark.parametrize(
        ("installed_version", "command"),
        [("0.33.5", "huggingface-cli login"), ("0.34.0", "hf auth login")],
    )
    def test_hf_token_typed_on_command_line_still_warns(
        self, model_dir, hf_api_tokens, monkeypatch, installed_version, command,
    ):
        monkeypatch.setattr(hf, "version", lambda _: installed_version, raising=False)
        result = _push("--token", "hf_TYPED")
        out = _clean(result.output)
        assert self._DEPRECATION in out
        assert f"Use HF_TOKEN env var or run '{command}'." in out
        assert hf_api_tokens == ["hf_TYPED"]

    def test_hf_token_typed_with_env_set_still_warns(self, model_dir, hf_api_tokens,
                                                     monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
        result = _push("--token", "hf_TYPED")
        assert self._DEPRECATION in _clean(result.output)
        assert hf_api_tokens == ["hf_TYPED"]

    @pytest.mark.parametrize(
        ("installed_version", "command"),
        [("0.33.5", "huggingface-cli login"), ("0.34.0", "hf auth login")],
    )
    def test_hf_missing_token_hint_does_not_suggest_deprecated_flag(
        self, model_dir, hf_api_tokens, monkeypatch, installed_version, command,
    ):
        monkeypatch.setattr(hf, "version", lambda _: installed_version, raising=False)
        result = _push()
        out = _clean(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "No HuggingFace token found" in out
        assert "HF_TOKEN" in out
        assert command in out
        assert "--token" not in out
        assert hf_api_tokens == []

    def test_non_hf_missing_token_hint_keeps_token_flag(self, model_dir, recorded):
        # On other hubs --token is the supported way, not a deprecated one.
        result = _push("--hub", "modelscope")
        out = _clean(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "--token YOUR_TOKEN" in out
        assert "MODELSCOPE_API_TOKEN" in out
        assert recorded == []
