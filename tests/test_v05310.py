"""v0.53.11 — Quick wins + packaging + UX wiring.

Covers seven closed issues:

* #150 ``[mix]`` extra + ``describe_default_optimizer`` advisory.
* #113 ``[data-pro]`` extras + lazy ``langdetect`` / Presidio routing.
* #154 ``SOUP_POSTHOG_KEY`` / ``SOUP_POSTHOG_ENDPOINT`` env override.
* #152 Multi-command ``--hub`` dispatch (chat / serve / infer / merge /
  export / push).
* #153 ``soup data download --hub <non-hf>`` live SDK lift.
* #155 Web UI Tool Outputs panel JS / HTML wiring.
* #156 SFT trainer-side ``record_call`` wire-up in
  ``monitoring/callback.SoupTrainerCallback``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strip_ansi(text: str) -> str:
    out = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", " ", out)


# ----------------------------------------------------------------------
# #150 — `[mix]` extra + scikit-optimize advisory
# ----------------------------------------------------------------------


class TestMixExtra:
    def test_pyproject_lists_mix_extra(self):
        body = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        # ``mix`` extra MUST be declared and bundle scikit-optimize.
        assert re.search(r"^mix\s*=\s*\[", body, re.MULTILINE), (
            "[mix] extra missing from pyproject"
        )
        mix_block = re.search(
            r"^mix\s*=\s*\[(.*?)\]", body, re.MULTILINE | re.DOTALL
        )
        assert mix_block is not None
        assert "scikit-optimize" in mix_block.group(1)

    def test_describe_default_optimizer_returns_label(self):
        from soup_cli.utils.data_mix import describe_default_optimizer

        label = describe_default_optimizer()
        assert label in ("scikit-optimize", "dirichlet-fallback")

    def test_describe_dirichlet_fallback_when_skopt_missing(self):
        # Force find_spec to report skopt missing.
        with patch("importlib.util.find_spec", return_value=None):
            from soup_cli.utils.data_mix import describe_default_optimizer

            assert describe_default_optimizer() == "dirichlet-fallback"


# ----------------------------------------------------------------------
# #113 — `[data-pro]` extras + langdetect / Presidio routing
# ----------------------------------------------------------------------


class TestDataProExtra:
    def test_pyproject_lists_data_pro(self):
        body = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r"^data-pro\s*=\s*\[", body, re.MULTILINE), (
            "[data-pro] extra missing from pyproject"
        )
        block = re.search(
            r"^data-pro\s*=\s*\[(.*?)\]", body, re.MULTILINE | re.DOTALL
        )
        assert block is not None
        assert "langdetect" in block.group(1)
        assert "presidio-analyzer" in block.group(1)

    def test_detect_language_falls_back_to_heuristic_without_langdetect(self):
        # Force the lazy import inside _langdetect_fast to fail by removing
        # any ``langdetect`` from sys.modules + blocking re-import.
        from soup_cli.utils import data_score

        with patch.object(
            data_score, "_langdetect_fast", return_value=None
        ):
            # Heuristic still picks 'en' on an English sentence.
            assert data_score.detect_language(
                "the quick brown fox jumps over the lazy dog"
            ) == "en"

    def test_detect_language_uses_langdetect_when_available(self):
        from soup_cli.utils import data_score

        with patch.object(data_score, "_langdetect_fast", return_value="ja"):
            assert data_score.detect_language(
                "the quick brown fox jumps over the lazy dog"
            ) == "ja"

    def test_detect_pii_falls_back_to_regex_without_presidio(self):
        from soup_cli.utils import data_score

        with patch.object(data_score, "_presidio_pii", return_value=None):
            hits = data_score.detect_pii(
                "Email me at user@example.com or 555-867-5309"
            )
            kinds = {h["kind"] for h in hits}
            assert "email" in kinds

    def test_detect_pii_uses_presidio_when_available(self):
        from soup_cli.utils import data_score

        presidio_hits = [{"kind": "email", "snippet": "user@example.com"}]
        with patch.object(
            data_score, "_presidio_pii", return_value=presidio_hits
        ):
            assert data_score.detect_pii("anything") == presidio_hits


# ----------------------------------------------------------------------
# #154 — PostHog env override
# ----------------------------------------------------------------------


class TestPostHogEnvOverride:
    def test_resolve_uses_default_key_when_env_unset(self):
        from soup_cli.utils.trackers import _POSTHOG_DEFAULT_KEY, _resolve_posthog_target

        # No env override; endpoint omitted to use the sentinel default.
        resolved = _resolve_posthog_target(None, env={})
        assert resolved is not None
        key, _ = resolved
        assert key == _POSTHOG_DEFAULT_KEY

    def test_resolve_env_key_overrides_default(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_KEY": "phc_user_project"}
        resolved = _resolve_posthog_target(None, env=env)
        assert resolved is not None
        key, _ = resolved
        assert key == "phc_user_project"

    def test_resolve_explicit_arg_wins_over_env(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_KEY": "phc_env"}
        resolved = _resolve_posthog_target("phc_caller", env=env)
        assert resolved is not None
        key, _ = resolved
        assert key == "phc_caller"

    def test_resolve_env_endpoint_overrides_default(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_ENDPOINT": "https://eu.i.posthog.com/i/v0/e/"}
        resolved = _resolve_posthog_target(None, env=env)
        assert resolved is not None
        _, endpoint = resolved
        assert endpoint == "https://eu.i.posthog.com/i/v0/e/"

    def test_resolve_rejects_http_endpoint(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_ENDPOINT": "http://attacker.example.com/"}
        assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_explicit_endpoint_locks_against_env(self):
        # code-review HIGH fix: a caller passing the default URL string
        # explicitly should NOT be silently overridden by the env var.
        from soup_cli.utils.trackers import _POSTHOG_ENDPOINT, _resolve_posthog_target

        env = {"SOUP_POSTHOG_ENDPOINT": "https://eu.i.posthog.com/i/v0/e/"}
        resolved = _resolve_posthog_target(None, endpoint=_POSTHOG_ENDPOINT, env=env)
        assert resolved is not None
        _, endpoint = resolved
        assert endpoint == _POSTHOG_ENDPOINT

    def test_resolve_rejects_control_char_key(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_KEY": "phc\nAuthorization: bypass"}
        assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_rejects_null_byte_key(self):
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_KEY": "phc\x00bypass"}
        assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_rejects_oversize_key(self):
        # tdd-review MEDIUM #3: 257-char SOUP_POSTHOG_KEY must reject.
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_KEY": "x" * 257}
        assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_rejects_explicit_empty_key(self):
        # tdd-review MEDIUM #4: explicit empty-string `api_key` must reject.
        from soup_cli.utils.trackers import _resolve_posthog_target

        assert _resolve_posthog_target("", env={}) is None


# ----------------------------------------------------------------------
# #152 — Multi-command `--hub` dispatch
# ----------------------------------------------------------------------


class TestHubPrefetchHelper:
    def test_hf_short_circuits_no_download(self):
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        model_out, base_out = apply_hub_to_cli_model(
            "meta-llama/Llama-3.1-8B", None, "hf"
        )
        assert model_out == "meta-llama/Llama-3.1-8B"
        assert base_out is None

    def test_none_hub_passes_through(self):
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        # None or empty hub is treated as no-op.
        assert apply_hub_to_cli_model("foo", None, "") == ("foo", None)

    def test_existing_local_path_not_rewritten(self, tmp_path, monkeypatch):
        # A real local dir means the user already merged + saved; the hub
        # flag should not force a re-download.
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "merged"
        local.mkdir()
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        model_out, base_out = apply_hub_to_cli_model(
            str(local), None, "modelscope"
        )
        assert model_out == str(local)

    def test_non_existent_base_invokes_prefetch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # patch prefetch to avoid network
        from soup_cli.utils import hubs

        with patch.object(
            hubs, "prefetch_model_from_hub", return_value=str(tmp_path / "snap")
        ) as p:
            model_out, base_out = hubs.apply_hub_to_cli_model(
                "./adapter", "some/repo", "modelscope"
            )
            p.assert_called_once()
            assert base_out == str(tmp_path / "snap")

    def test_apply_hub_to_cli_model_both_none(self):
        # tdd-review LOW #7: neither model nor base set should no-op.
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        assert apply_hub_to_cli_model(None, None, "modelscope") == (None, None)

    def test_prefetch_unknown_hub_raises(self):
        # Unknown hub name must propagate validation error.
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        with pytest.raises(ValueError):
            apply_hub_to_cli_model(None, "some/repo", "evilhub")

    def test_prefetch_outside_cwd_cache_root_raises(self, tmp_path, monkeypatch):
        # tdd-review MEDIUM #5: cache_root outside cwd must raise.
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.hubs import prefetch_model_from_hub

        outside = tmp_path.parent / "outside_cache"
        with pytest.raises(ValueError, match="escapes"):
            prefetch_model_from_hub(
                "some/repo", "modelscope", cache_root=str(outside)
            )


class TestCliHubFlags:
    @pytest.mark.parametrize(
        "cmd_module,cmd_name",
        [
            ("soup_cli.commands.chat", "chat"),
            ("soup_cli.commands.serve", "serve"),
            ("soup_cli.commands.infer", "infer"),
            ("soup_cli.commands.merge", "merge"),
            ("soup_cli.commands.export", "export"),
            ("soup_cli.commands.push", "push"),
        ],
    )
    def test_hub_flag_present_in_signature(self, cmd_module, cmd_name):
        # Re-importing fresh because typer-decorated commands wrap signatures.
        module = __import__(cmd_module, fromlist=[cmd_name])
        fn = getattr(module, cmd_name)
        # Typer commands are wrapped — pull the raw function.
        target = getattr(fn, "__wrapped__", fn)
        import inspect

        sig = inspect.signature(target)
        assert "hub" in sig.parameters, (
            f"{cmd_module}.{cmd_name} missing --hub keyword"
        )

    def test_apply_hub_helper_imported_in_non_push_commands(self):
        # tdd-review HIGH #1: every command except push must use the
        # shared helper so a future refactor cannot silently drop it.
        for mod in ("chat", "serve", "infer", "merge", "export"):
            src = (REPO_ROOT / f"soup_cli/commands/{mod}.py").read_text(
                encoding="utf-8"
            )
            assert "apply_hub_to_cli_model" in src, (
                f"{mod}.py missing apply_hub_to_cli_model helper import"
            )

    def test_push_uses_upload_repo_path(self):
        # push.py does NOT call apply_hub_to_cli_model (it's an upload
        # surface, not a download). It must call upload_repo + validate_hub_name.
        src = (REPO_ROOT / "soup_cli/commands/push.py").read_text(
            encoding="utf-8"
        )
        assert "upload_repo" in src
        assert "validate_hub_name" in src


# ----------------------------------------------------------------------
# #153 — `soup data download --hub <non-hf>` live SDK
# ----------------------------------------------------------------------


class TestDataDownloadNonHfLive:
    def test_modelers_unknown_sdk_friendly_error(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        # No openmind_hub installed; expect ImportError advisory.
        result = CliRunner().invoke(
            app, ["download", "dummy/ds", "--hub", "modelers"]
        )
        # exit_code 1 = friendly ImportError advisory; 2 = validation reject.
        assert result.exit_code in (1, 2)
        out = _strip_ansi(result.output)
        # No longer surfaces "wait for v0.53.9"; should mention modelers or
        # the missing SDK pip-install hint.
        assert "v0.53.9" not in out

    def test_modelscope_unknown_sdk_friendly_error(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        if "modelscope" in sys.modules:
            pytest.skip("modelscope is installed; live branch tested separately")
        result = CliRunner().invoke(
            app, ["download", "dummy/ds", "--hub", "modelscope"]
        )
        assert result.exit_code in (1, 2)
        out = _strip_ansi(result.output)
        # Friendly advisory mentions the SDK or `pip install`.
        assert "modelscope" in out

    def test_data_download_no_longer_advises_v0_53_9(self):
        # tdd-review LOW #9: a future regression of the advisory text
        # "wait for v0.53.9" must be caught at source-grep time, since
        # v0.53.11 lifted that advisory to live SDK dispatch.
        src = (REPO_ROOT / "soup_cli/commands/data.py").read_text(
            encoding="utf-8"
        )
        assert "wait for v0.53.9" not in src


# ----------------------------------------------------------------------
# #155 — Web UI Tool Outputs panel
# ----------------------------------------------------------------------


class TestWebUiToolOutputsPanel:
    def test_index_html_has_tools_nav_entry(self):
        html = (REPO_ROOT / "soup_cli/ui/static/index.html").read_text(
            encoding="utf-8"
        )
        assert 'data-page="tools"' in html
        assert "Tool Outputs" in html
        assert 'id="page-tools"' in html

    def test_app_js_has_load_tool_outputs(self):
        js = (REPO_ROOT / "soup_cli/ui/static/app.js").read_text(encoding="utf-8")
        assert "loadToolOutputs" in js
        assert "/api/tool-outputs" in js
        # XSS-safe — uses textContent or DOM API, NOT innerHTML for records.
        assert "td.textContent" in js


# ----------------------------------------------------------------------
# #156 — SFT callback `record_call` wire-up
# ----------------------------------------------------------------------


class TestCallbackToolBuffer:
    def test_on_step_end_method_exists(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback

        assert hasattr(SoupTrainerCallback, "on_step_end")

    def test_on_step_end_no_op_when_inputs_absent(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.utils.tool_outputs import (
            get_global_tool_buffer,
            reset_global_tool_buffer,
        )

        reset_global_tool_buffer()
        cb = SoupTrainerCallback.__new__(SoupTrainerCallback)
        # State is dataclass-shaped; only global_step is read.
        from types import SimpleNamespace

        cb.on_step_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
        )
        # Buffer should remain empty.
        assert len(list(get_global_tool_buffer().snapshot(limit=10))) == 0

    def test_on_step_end_records_when_tool_calls_present(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.utils.tool_outputs import (
            get_global_tool_buffer,
            reset_global_tool_buffer,
        )

        reset_global_tool_buffer()
        cb = SoupTrainerCallback.__new__(SoupTrainerCallback)
        from types import SimpleNamespace

        cb.on_step_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=42),
            control=SimpleNamespace(),
            inputs={"tool_calls": [{"name": "f"}, {"name": "g"}]},
        )
        records = list(get_global_tool_buffer().snapshot(limit=10))
        assert len(records) == 1
        assert "step 42" in records[0].output_preview
        assert records[0].success is True
        reset_global_tool_buffer()

    def test_on_step_end_empty_tool_calls_emits_no_record(self):
        # tdd-review MEDIUM #6: empty list short-circuits.
        from types import SimpleNamespace

        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.utils.tool_outputs import (
            get_global_tool_buffer,
            reset_global_tool_buffer,
        )

        reset_global_tool_buffer()
        cb = SoupTrainerCallback.__new__(SoupTrainerCallback)
        cb.on_step_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            inputs={"tool_calls": []},
        )
        assert len(list(get_global_tool_buffer().snapshot(limit=10))) == 0

    def test_on_step_end_bool_tool_calls_emits_no_record(self):
        # tdd-review HIGH #2: ``tool_calls=True`` (bool) is a falsy-on-list
        # short-circuit AND rejected by the numeric-branch bool guard.
        from types import SimpleNamespace

        from soup_cli.monitoring.callback import SoupTrainerCallback
        from soup_cli.utils.tool_outputs import (
            get_global_tool_buffer,
            reset_global_tool_buffer,
        )

        reset_global_tool_buffer()
        cb = SoupTrainerCallback.__new__(SoupTrainerCallback)
        # ``True`` is truthy (bypasses ``not tool_calls`` guard); the
        # bool-rejection branch must keep ``count=0`` and skip recording.
        cb.on_step_end(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            inputs={"tool_calls": True},
        )
        assert len(list(get_global_tool_buffer().snapshot(limit=10))) == 0


# ----------------------------------------------------------------------
# Version bump sanity check
# ----------------------------------------------------------------------


class TestVersionBump:
    def test_init_py_pin(self):
        from soup_cli import __version__

        # v0.54.0+: forward-compatible — assert the package is at least the
        # version this test ships with.
        major, minor, patch = (int(x) for x in __version__.split("."))
        assert (major, minor, patch) >= (0, 53, 11)

    def test_pyproject_pin(self):
        body = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', body, re.MULTILINE)
        assert match is not None, "pyproject.toml version line missing"
        major, minor, patch = (int(g) for g in match.groups())
        assert (major, minor, patch) >= (0, 53, 11), (
            f"pyproject.toml version pin behind 0.53.11: {match.group(0)}"
        )
