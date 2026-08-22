"""Regression tests for issue #91: the live Aider Polyglot runner."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


def _result(path: Path, *, passed: bool, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tests_outcomes": [passed], **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestAiderResultParser:
    def test_aggregates_upstream_per_exercise_json(self, tmp_path):
        from soup_cli.eval.aider_polyglot import parse_aider_results

        _result(
            tmp_path / "python/exercises/practice/clock/.aider.results.json",
            passed=True,
            num_error_outputs=2,
            test_timeouts=0,
        )
        _result(
            tmp_path / "rust/exercises/practice/clock/.aider.results.json",
            passed=False,
            num_error_outputs=1,
            test_timeouts=1,
        )

        row = parse_aider_results(tmp_path, model="openai/example")

        assert row["model"] == "openai/example"
        assert row["task"] == "aider_polyglot"
        assert row["score"] == 0.5
        assert row["errors"] == 1
        assert row["details"]["completed_tests"] == 2
        assert row["details"]["error_outputs"] == 3
        assert row["details"]["test_timeouts"] == 1

    def test_rejects_missing_results(self, tmp_path):
        from soup_cli.eval.aider_polyglot import AiderEvalError, parse_aider_results

        with pytest.raises(AiderEvalError, match="No Aider result files"):
            parse_aider_results(tmp_path, model="openai/example")

    def test_rejects_malformed_json(self, tmp_path):
        from soup_cli.eval.aider_polyglot import AiderEvalError, parse_aider_results

        result = tmp_path / "python/exercises/practice/clock/.aider.results.json"
        result.parent.mkdir(parents=True)
        result.write_text("{broken", encoding="utf-8")

        with pytest.raises(AiderEvalError, match="Malformed Aider result JSON"):
            parse_aider_results(tmp_path, model="openai/example")

    def test_rejects_oversized_json(self, tmp_path, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        result = tmp_path / "python/exercises/practice/clock/.aider.results.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(aider_eval, "MAX_RESULT_FILE_BYTES", 1)

        with pytest.raises(aider_eval.AiderEvalError, match="too large"):
            aider_eval.parse_aider_results(tmp_path, model="openai/example")

    def test_rejects_symlinked_result(self, tmp_path):
        from soup_cli.eval.aider_polyglot import AiderEvalError, parse_aider_results

        outside = tmp_path.parent / "outside-aider-result.json"
        outside.write_text('{"tests_outcomes": [true]}', encoding="utf-8")
        result = tmp_path / "python/exercises/practice/clock/.aider.results.json"
        result.parent.mkdir(parents=True)
        try:
            result.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks are unavailable on this platform")

        with pytest.raises(AiderEvalError, match="symlink"):
            parse_aider_results(tmp_path, model="openai/example")


class TestAiderDockerRunner:
    def test_missing_docker_has_friendly_error(self, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.setattr(aider_eval.shutil, "which", lambda _name: None)

        with pytest.raises(aider_eval.AiderEvalError, match="Docker CLI was not found"):
            aider_eval.preflight_docker("aider-benchmark")

    def test_daemon_failure_has_friendly_error(self, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.setattr(aider_eval.shutil, "which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            aider_eval.subprocess,
            "run",
            lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="", stderr="not running"),
        )

        with pytest.raises(aider_eval.AiderEvalError, match="daemon is unavailable"):
            aider_eval.preflight_docker("aider-benchmark")

    def test_missing_image_has_setup_hint(self, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        calls = iter(
            [
                SimpleNamespace(returncode=0, stdout="29.0", stderr=""),
                SimpleNamespace(returncode=1, stdout="", stderr="No such image"),
            ]
        )
        monkeypatch.setattr(aider_eval.shutil, "which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(aider_eval.subprocess, "run", lambda *_a, **_kw: next(calls))

        with pytest.raises(aider_eval.AiderEvalError, match="docker_build.sh"):
            aider_eval.preflight_docker("aider-benchmark")

    def test_preflight_timeout_is_friendly(self, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.setattr(aider_eval.shutil, "which", lambda _name: "/usr/bin/docker")

        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("docker info", 10)

        monkeypatch.setattr(aider_eval.subprocess, "run", timeout)
        with pytest.raises(aider_eval.AiderEvalError, match="timed out"):
            aider_eval.preflight_docker("aider-benchmark")

    def test_build_command_uses_fixed_argv_and_read_only_corpus(self, tmp_path, monkeypatch):
        from soup_cli.eval.aider_polyglot import build_docker_command

        corpus = tmp_path / "polyglot"
        output = tmp_path / "results"
        corpus.mkdir()
        output.mkdir()
        monkeypatch.setenv("OPENAI_API_KEY", "secret-never-in-argv")

        command = build_docker_command(
            docker="/usr/bin/docker",
            image="aider-benchmark",
            model="openai/example; touch /tmp/pwned",
            exercises_dir=corpus,
            output_dir=output,
            threads=2,
            num_tests=3,
        )

        assert isinstance(command, list)
        assert command[-1] == "3"
        assert "openai/example; touch /tmp/pwned" in command
        assert not any("secret-never-in-argv" in arg for arg in command)
        mount = command[command.index("--mount") + 1]
        assert "readonly" in mount
        assert "type=bind" in mount


class TestAiderCLI:
    def test_aider_optional_extra_is_declared(self):
        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")

        assert 'aider = ["aider-chat>=' in text

    def test_help_lists_required_flags(self):
        result = runner.invoke(app, ["eval", "aider", "--help"])
        output = _plain(result.output)

        assert result.exit_code == 0
        assert "--model" in output
        assert "--output" in output
        assert "--exercises-dir" in output
        assert "--run-id" in output

    def test_output_escape_is_rejected_before_docker(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        corpus = tmp_path / "polyglot"
        corpus.mkdir()

        with patch("soup_cli.eval.aider_polyglot.preflight_docker") as preflight:
            result = runner.invoke(
                app,
                [
                    "eval",
                    "aider",
                    "--model",
                    "openai/example",
                    "--output",
                    str(tmp_path.parent / "escape"),
                    "--exercises-dir",
                    str(corpus),
                ],
            )

        assert result.exit_code == 1
        assert "must stay under the current working" in result.output
        assert "directory" in result.output
        preflight.assert_not_called()

    def test_success_writes_soup_json_and_tracks_run(self, tmp_path, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.chdir(tmp_path)
        corpus = tmp_path / "polyglot"
        corpus.mkdir()
        output = tmp_path / "results"
        db_path = tmp_path / "experiments.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(db_path))

        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        tracker.start_run({}, "cpu", "test", {}, run_id="run_aider_test")
        tracker.close()

        monkeypatch.setattr(aider_eval, "preflight_docker", lambda _image: "/usr/bin/docker")

        def fake_run(command, **kwargs):
            assert kwargs["shell"] is False
            assert kwargs["timeout"] == 600
            _result(
                output / "python/exercises/practice/clock/.aider.results.json",
                passed=True,
            )
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(aider_eval.subprocess, "run", fake_run)

        result = runner.invoke(
            app,
            [
                "eval",
                "aider",
                "--model",
                "openai/example",
                "--output",
                "results",
                "--exercises-dir",
                "polyglot",
                "--run-id",
                "run_aider_test",
                "--timeout",
                "600",
                "--num-tests",
                "1",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((output / "soup_result.json").read_text(encoding="utf-8"))
        assert payload["task"] == "aider_polyglot"
        assert payload["score"] == 1.0
        tracker = ExperimentTracker()
        saved = tracker.get_eval_results(run_id="run_aider_test")
        assert len(saved) == 1
        assert saved[0]["benchmark"] == "aider_polyglot"
        tracker.close()

    def test_subprocess_failure_does_not_emit_summary(self, tmp_path, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.chdir(tmp_path)
        corpus = tmp_path / "polyglot"
        corpus.mkdir()
        monkeypatch.setattr(aider_eval, "preflight_docker", lambda _image: "/usr/bin/docker")
        monkeypatch.setattr(
            aider_eval.subprocess,
            "run",
            lambda *_a, **_kw: SimpleNamespace(returncode=17),
        )

        result = runner.invoke(
            app,
            [
                "eval",
                "aider",
                "--model",
                "openai/example",
                "--output",
                "results",
                "--exercises-dir",
                "polyglot",
            ],
        )

        assert result.exit_code == 1
        assert "exited with status 17" in result.output
        assert not (tmp_path / "results/soup_result.json").exists()

    def test_subprocess_timeout_is_friendly(self, tmp_path, monkeypatch):
        import soup_cli.eval.aider_polyglot as aider_eval

        monkeypatch.chdir(tmp_path)
        corpus = tmp_path / "polyglot"
        corpus.mkdir()
        monkeypatch.setattr(aider_eval, "preflight_docker", lambda _image: "/usr/bin/docker")

        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("docker run", 60)

        monkeypatch.setattr(aider_eval.subprocess, "run", timeout)
        result = runner.invoke(
            app,
            [
                "eval",
                "aider",
                "--model",
                "openai/example",
                "--output",
                "results",
                "--exercises-dir",
                "polyglot",
                "--timeout",
                "60",
            ],
        )

        assert result.exit_code == 1
        assert "timed out after 60 seconds" in result.output
