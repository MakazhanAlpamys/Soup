"""Tests for CLI commands."""

from soup_cli import __version__
from soup_cli.cli import app
from tests.conftest import make_cli_runner

runner = make_cli_runner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_full():
    result = runner.invoke(app, ["version", "--full"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "Python" in result.output


def test_version_full_short_flag():
    result = runner.invoke(app, ["version", "-f"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Fine-tune" in result.output


def test_init_unknown_template():
    result = runner.invoke(app, ["init", "--template", "nonexistent"])
    assert result.exit_code == 1


def test_train_missing_config():
    result = runner.invoke(app, ["train", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1


def test_chat_missing_model():
    result = runner.invoke(app, ["chat", "--model", "nonexistent_path"])
    assert result.exit_code == 1


def test_push_missing_model():
    result = runner.invoke(
        app, ["push", "--model", "nonexistent_path", "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_push_not_a_directory(tmp_path):
    # Create a file (not a directory)
    fake_file = tmp_path / "model.bin"
    fake_file.write_text("not a model")
    result = runner.invoke(
        app, ["push", "--model", str(fake_file), "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_push_invalid_model_dir(tmp_path):
    # Create an empty directory (no adapter_config.json or config.json)
    model_dir = tmp_path / "empty_model"
    model_dir.mkdir()
    result = runner.invoke(
        app, ["push", "--model", str(model_dir), "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_help_shows_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "push" in result.output
    assert "train" in result.output
    assert "init" in result.output
    assert "export" in result.output
    assert "merge" in result.output


def test_version_json():
    import json
    import platform

    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["version"] == __version__
    assert data["python"] == platform.python_version()
    assert data["platform"] == platform.system().lower()


def test_version_full_json():
    import json
    import platform

    result = runner.invoke(app, ["version", "--full", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["version"] == __version__
    assert data["python"] == platform.python_version()
    assert "platform" in data


def test_version_full_json_with_stderr_warning_succeeds(monkeypatch):
    """Simulate torch first-import stderr warning and verify stdout still parses cleanly (#1551)."""
    import json
    import sys

    orig_import = __import__

    def import_hook(name, *args, **kwargs):
        if name == "torch":
            sys.stderr.write(
                "W... triton not found; flop counting will not work for triton kernels\n"
            )
            # Click 8.1 snapshots the separate stderr stream without flushing it.
            sys.stderr.flush()
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_hook)
    result = runner.invoke(app, ["version", "--full", "--json"])
    assert result.exit_code == 0
    assert "triton not found" in result.stderr
    data = json.loads(result.stdout)
    assert data["version"] == __version__


def test_make_cli_runner_keeps_stderr_out_of_stdout_and_forwards_kwargs():
    """Separate the streams on every Click and pass extra kwargs through (#1551)."""
    import os

    import typer

    probe = typer.Typer()

    @probe.command()
    def emit():
        typer.echo("to-stdout")
        typer.echo("to-stderr", err=True)
        typer.echo(os.environ.get("SOUP_RUNNER_PROBE", "unset"))

    result = make_cli_runner(env={"SOUP_RUNNER_PROBE": "seen"}).invoke(probe, [])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.stdout == "to-stdout\nseen\n"
    assert "to-stderr" in result.stderr
    assert "to-stderr" not in result.stdout
