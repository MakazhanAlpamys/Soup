"""Tests for v0.40.2 Part A — originally scheduled issues (#36, #50, #51)."""
from __future__ import annotations

import re
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _plain(s: str) -> str:
    """Strip ANSI escape sequences AND any whitespace inserted by Rich's
    width-aware option-name wrapping (e.g. ``-\\n-template-dir`` on narrow
    CI terminals).
    """
    no_ansi = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)
    return re.sub(r"\s+", "", no_ansi)


# ---------------------------------------------------------------------------
# #50 — `--hf-resume` prefer local newer
# ---------------------------------------------------------------------------


class TestHFResumePreferLocal:
    def test_find_highest_local_checkpoint_empty(self, tmp_path):
        from soup_cli.monitoring.hf_push import _find_highest_local_checkpoint

        assert _find_highest_local_checkpoint(str(tmp_path)) is None

    def test_find_highest_local_checkpoint_picks_max(self, tmp_path):
        from soup_cli.monitoring.hf_push import _find_highest_local_checkpoint

        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-50").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        # Spurious dirs ignored
        (tmp_path / "garbage").mkdir()
        (tmp_path / "checkpoint-bad").mkdir()
        # Files (not dirs) ignored
        (tmp_path / "checkpoint-999").write_text("not a dir")

        assert _find_highest_local_checkpoint(str(tmp_path)) == 200

    def test_find_highest_local_checkpoint_missing_dir(self, tmp_path):
        from soup_cli.monitoring.hf_push import _find_highest_local_checkpoint

        assert _find_highest_local_checkpoint(str(tmp_path / "nope")) is None

    def test_prepare_hf_resume_skips_download_when_local_equal(
        self, tmp_path, monkeypatch
    ):
        """If local checkpoint-N == remote checkpoint-N, skip download."""
        from soup_cli.monitoring.hf_push import prepare_hf_resume

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        (out_dir / "checkpoint-300").mkdir()

        download_called = {"called": False}

        def fake_download(*args, **kwargs):
            download_called["called"] = True
            return kwargs.get("local_dir", "")

        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.resolve_latest_checkpoint_revision",
            lambda repo_id, token=None, endpoint=None: "checkpoint-300",
        )
        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push._download_checkpoint", fake_download
        )

        result = prepare_hf_resume(
            repo_id="user/my-model", output_dir=str(out_dir), token="t1"
        )
        assert download_called["called"] is False
        assert result is not None
        assert "checkpoint-300" in result

    def test_prepare_hf_resume_skips_download_when_local_newer(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.monitoring.hf_push import prepare_hf_resume

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        (out_dir / "checkpoint-500").mkdir()

        download_called = {"called": False}

        def fake_download(*args, **kwargs):
            download_called["called"] = True
            return kwargs.get("local_dir", "")

        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.resolve_latest_checkpoint_revision",
            lambda repo_id, token=None, endpoint=None: "checkpoint-300",
        )
        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push._download_checkpoint", fake_download
        )

        result = prepare_hf_resume(
            repo_id="user/my-model", output_dir=str(out_dir), token="t1"
        )
        assert download_called["called"] is False
        assert result is not None
        assert "checkpoint-500" in result

    def test_prepare_hf_resume_downloads_when_remote_newer(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.monitoring.hf_push import prepare_hf_resume

        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "runs"
        out_dir.mkdir()
        (out_dir / "checkpoint-100").mkdir()

        download_called = {"called": False, "revision": None}

        def fake_download(repo_id, revision, local_dir, token, endpoint):
            download_called["called"] = True
            download_called["revision"] = revision
            return local_dir

        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push.resolve_latest_checkpoint_revision",
            lambda repo_id, token=None, endpoint=None: "checkpoint-500",
        )
        monkeypatch.setattr(
            "soup_cli.monitoring.hf_push._download_checkpoint", fake_download
        )

        result = prepare_hf_resume(
            repo_id="user/my-model", output_dir=str(out_dir), token="t1"
        )
        assert download_called["called"] is True
        assert download_called["revision"] == "checkpoint-500"
        assert result is not None


# ---------------------------------------------------------------------------
# #51 — `soup deploy hf-space --template-dir`
# ---------------------------------------------------------------------------


class TestHfSpaceCustomTemplate:
    def test_render_custom_template_dir(self, tmp_path, monkeypatch):
        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "mytpl"
        tdir.mkdir()
        (tdir / "app.py").write_text("MODEL = '{MODEL_REPO}'\n")
        (tdir / "README.md").write_text("# Space for {MODEL_REPO}\n")
        (tdir / "requirements.txt").write_text("gradio\n")

        rendered = render_custom_template_dir(
            str(tdir), model_repo="user/my-model"
        )
        assert rendered["app.py"] == "MODEL = 'user/my-model'\n"
        assert rendered["README.md"] == "# Space for user/my-model\n"
        assert rendered["requirements.txt"] == "gradio\n"

    def test_render_custom_template_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.hf_space import render_custom_template_dir

        cwd = tmp_path / "project"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "app.py").write_text("x")
        (outside / "README.md").write_text("x")

        with pytest.raises(ValueError, match="under the current"):
            render_custom_template_dir(str(outside), model_repo="user/my-model")

    def test_render_custom_template_rejects_invalid_repo_id(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "tpl"
        tdir.mkdir()
        (tdir / "app.py").write_text("x")
        (tdir / "README.md").write_text("x")

        with pytest.raises(ValueError):
            render_custom_template_dir(str(tdir), model_repo="bad..repo")

    def test_render_custom_template_requires_app_py(self, tmp_path, monkeypatch):
        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "tpl"
        tdir.mkdir()
        (tdir / "README.md").write_text("x")

        with pytest.raises(FileNotFoundError, match="app.py"):
            render_custom_template_dir(str(tdir), model_repo="user/my-model")

    def test_render_custom_template_requires_readme(self, tmp_path, monkeypatch):
        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "tpl"
        tdir.mkdir()
        (tdir / "app.py").write_text("x")

        with pytest.raises(FileNotFoundError, match="README.md"):
            render_custom_template_dir(str(tdir), model_repo="user/my-model")

    def test_render_custom_template_size_cap(self, tmp_path, monkeypatch):
        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "tpl"
        tdir.mkdir()
        # 256KB+1 byte
        big = "a" * (256 * 1024 + 1)
        (tdir / "app.py").write_text(big)
        (tdir / "README.md").write_text("x")

        with pytest.raises(ValueError, match="256"):
            render_custom_template_dir(str(tdir), model_repo="user/my-model")

    def test_render_custom_template_rejects_symlink(self, tmp_path, monkeypatch):
        """Symlinked app.py must be rejected — TOCTOU defence."""
        import os
        import sys

        if sys.platform == "win32":
            pytest.skip("symlinks require admin on Windows")

        from soup_cli.utils.hf_space import render_custom_template_dir

        monkeypatch.chdir(tmp_path)
        tdir = tmp_path / "tpl"
        tdir.mkdir()
        # Real app.py outside, symlink inside
        evil_target = tmp_path / "evil_app.py"
        evil_target.write_text("evil\n")
        os.symlink(str(evil_target), str(tdir / "app.py"))
        (tdir / "README.md").write_text("x")

        with pytest.raises(ValueError, match="symlink"):
            render_custom_template_dir(str(tdir), model_repo="user/my-model")

    def test_deploy_hf_space_help_shows_template_dir(self):
        result = runner.invoke(app, ["deploy", "hf-space", "--help"])
        # Rich wraps int option names across lines on narrow CI terminals;
        # strip ANSI + whitespace to match the option name regardless.
        assert "--template-dir" in _plain(result.output)


# ---------------------------------------------------------------------------
# #36 — Eval-gate dashboard row
# ---------------------------------------------------------------------------


class TestEvalGateDashboardRow:
    def test_format_gate_row_disabled(self):
        from soup_cli.monitoring.display import format_gate_row

        # No state -> empty string
        assert format_gate_row(None) == ""

    def test_format_gate_row_pass(self):
        from soup_cli.monitoring.display import format_gate_row

        state = {
            "tasks": [
                {"name": "helpfulness", "score": 7.8, "passed": True},
            ],
            "overall_passed": True,
            "action": None,
        }
        out = format_gate_row(state)
        assert "Gate" in out
        assert "helpfulness" in out
        assert "7.8" in out

    def test_format_gate_row_regression_stop(self):
        from soup_cli.monitoring.display import format_gate_row

        state = {
            "tasks": [
                {
                    "name": "math",
                    "score": 0.82,
                    "passed": False,
                    "delta": -0.06,
                    "baseline": 0.88,
                },
            ],
            "overall_passed": False,
            "action": "stop",
        }
        out = format_gate_row(state)
        assert "math" in out
        assert "0.82" in out
        assert "STOP" in out.upper() or "stop" in out

    def test_format_gate_row_warn_action(self):
        from soup_cli.monitoring.display import format_gate_row

        state = {
            "tasks": [{"name": "t", "score": 0.5, "passed": False, "delta": -0.1}],
            "overall_passed": False,
            "action": "warn",
        }
        out = format_gate_row(state)
        assert "WARN" in out

    def test_format_gate_row_multi_task(self):
        from soup_cli.monitoring.display import format_gate_row

        state = {
            "tasks": [
                {"name": "helpfulness", "score": 7.8, "passed": True},
                {"name": "math", "score": 0.82, "passed": False, "delta": -0.06},
            ],
            "overall_passed": False,
            "action": "stop",
        }
        out = format_gate_row(state)
        assert "helpfulness" in out
        assert "math" in out
        assert "|" in out  # separator
        assert "STOP" in out

    def test_format_gate_row_passed_missing_field_renders_neutral(self):
        from soup_cli.monitoring.display import format_gate_row

        # `passed` absent — should not silently render as red ✗
        state = {
            "tasks": [{"name": "t", "score": 0.5}],
            "overall_passed": True,
            "action": None,
        }
        out = format_gate_row(state)
        # No action suffix (None), and the task row renders without
        # falsely claiming pass.
        assert "STOP" not in out
        assert "WARN" not in out

    def test_format_gate_row_renders_via_console(self):
        from soup_cli.monitoring.display import format_gate_row

        state = {
            "tasks": [{"name": "t", "score": 1.0, "passed": True}],
            "overall_passed": True,
            "action": None,
        }
        buf = StringIO()
        Console(file=buf, force_terminal=False, no_color=True, width=120).print(
            format_gate_row(state)
        )
        assert "t" in buf.getvalue()
