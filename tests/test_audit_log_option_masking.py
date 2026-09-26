"""Audit records never contain values of credential options."""

from __future__ import annotations

import json

import click
import pytest
import typer


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(path))
    monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
    import soup_cli.cli as cli

    monkeypatch.setattr(cli, "_audit_disabled", False, raising=False)
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _emit(argv):
    import soup_cli.cli as cli

    # argv[0] is the program name, exactly as run() passes sys.argv.
    cli._emit_audit_event(["soup", *argv], 0)


@pytest.mark.parametrize(
    "argv, secret",
    [
        (
            ["serve", "--model", "./out", "--tool-auth-token", "Zq3v9Xk2LmP8rT4wYb6N"],
            "Zq3v9Xk2LmP8rT4wYb6N",
        ),
        (
            ["serve", "--model", "./out", "--tool-auth-token=Zq3v9Xk2LmP8rT4wYb6N"],
            "Zq3v9Xk2LmP8rT4wYb6N",
        ),
        (["ui", "--auth-token", "u1Bx7dK9mQ2wE5rT8yU3iO6pA"], "u1Bx7dK9mQ2wE5rT8yU3iO6pA"),
        (
            ["mcp", "serve", "--auth-token", "m9Kd2Lq8Wz5Xc3Vb7Nm1As4D"],
            "m9Kd2Lq8Wz5Xc3Vb7Nm1As4D",
        ),
        (["data", "generate", "--api-key", "gsk_abcdefghijklmnop"], "gsk_abcdefghijklmnop"),
        (["push", "--model", "out", "--repo", "u/m", "-t", "plainvalue123"], "plainvalue123"),
        (
            ["--log-level", "debug", "serve", "--tool-auth-token", "Gl0b4lOpt1onsF1rst9"],
            "Gl0b4lOpt1onsF1rst9",
        ),
    ],
)
def test_secret_option_values_not_written(audit_path, argv, secret):
    _emit(argv)
    text = audit_path.read_text(encoding="utf-8")
    assert secret not in text
    assert "<redacted>" in text


def test_non_secret_options_kept(audit_path):
    _emit(["serve", "--model", "./out", "--max-tokens", "512"])
    record = _records(audit_path)[-1]
    assert "512" in record["args"]


def test_unresolvable_command_still_masks(audit_path):
    """An unknown command falls back to the app-wide set, never to raw args."""
    _emit(["no-such-command", "--auth-token", "F4llb4ckV4lue77"])
    text = audit_path.read_text(encoding="utf-8")
    assert "F4llb4ckV4lue77" not in text
    assert "<redacted>" in text


def test_mask_secret_args_forms():
    from soup_cli.utils.argv_redaction import mask_secret_args

    opts = frozenset({"--auth-token", "-a"})
    assert mask_secret_args(["--auth-token", "x", "--port", "1"], opts) == (
        "--auth-token", "<redacted>", "--port", "1",
    )
    assert mask_secret_args(["--auth-token=x"], opts) == ("--auth-token=<redacted>",)
    assert mask_secret_args(["-a", "x"], opts) == ("-a", "<redacted>")
    assert mask_secret_args(["-ax"], opts) == ("-a<redacted>",)
    assert mask_secret_args(["--auth-token"], opts) == ("--auth-token",)


def _all_commands(cmd, path=()):
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _all_commands(sub, path + (name,))


def test_every_secret_looking_option_is_masked(audit_path):
    """Ratchet: a new credential option anywhere in the app is masked automatically."""
    from soup_cli.cli import app
    from soup_cli.utils.argv_redaction import NOT_SECRET_OPTIONS, SECRET_NAME_RE

    root = typer.main.get_command(app)
    checked = 0
    for path, cmd in _all_commands(root):
        for param in getattr(cmd, "params", []):
            if not isinstance(param, click.Option) or param.is_flag:
                continue
            longs = [o for o in param.opts if o.startswith("--")]
            if not any(SECRET_NAME_RE.search(o[2:]) for o in longs):
                continue
            if any(o in NOT_SECRET_OPTIONS for o in longs):
                continue
            for opt in param.opts:
                value = f"SENTINEL{checked:04d}value"
                _emit([*path, opt, value])
                assert value not in audit_path.read_text(encoding="utf-8"), (path, opt)
                checked += 1
    assert checked >= 5, checked


def test_not_secret_list_only_names_real_options():
    from soup_cli.cli import app
    from soup_cli.utils.argv_redaction import NOT_SECRET_OPTIONS

    root = typer.main.get_command(app)
    names = {
        o
        for _path, cmd in _all_commands(root)
        for p in getattr(cmd, "params", [])
        if isinstance(p, click.Option)
        for o in p.opts
    }
    assert NOT_SECRET_OPTIONS <= names, NOT_SECRET_OPTIONS - names
