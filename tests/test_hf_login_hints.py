"""Hugging Face login hints match the installed CLI (#2923)."""

import ast
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.cans import publish
from soup_cli.cli import app
from soup_cli.utils import hf
from tests.conftest import make_cli_runner, strip_ansi

_LEGACY_LOGIN = "huggingface-cli login"
_MODERN_LOGIN = "hf auth login"
_SOURCE = Path(__file__).resolve().parents[1] / "src" / "soup_cli"


@pytest.mark.parametrize(
    ("installed_version", "command"),
    [("0.33.5", _LEGACY_LOGIN), ("0.34.0", _MODERN_LOGIN), ("2.4.0", _MODERN_LOGIN)],
)
@pytest.mark.parametrize(
    "args",
    [
        ["data", "push", "--input", "dataset.jsonl", "--hf-dataset", "user/dataset"],
        ["deploy", "hf-space", "--model", "user/model", "--space", "user/space"],
        ["can", "publish", "recipe.can", "--hf-hub", "user/recipe"],
    ],
    ids=["data-push", "deploy-hf-space", "can-publish"],
)
def test_hf_upload_login_hint(args, installed_version, command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dataset.jsonl").write_text('{"text": "example"}\n', encoding="utf-8")
    (tmp_path / "recipe.can").touch()
    monkeypatch.setattr(hf, "version", lambda _: installed_version, raising=False)
    monkeypatch.setattr(hf, "resolve_token", lambda: None)
    monkeypatch.setattr(publish, "resolve_token", lambda: None)
    monkeypatch.setattr(publish, "inspect_can", lambda _: SimpleNamespace(description=""))
    monkeypatch.setattr(publish, "credential_paths", lambda *_: [])

    result = make_cli_runner().invoke(app, args)

    assert result.exit_code == 1, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "HF_TOKEN" in output
    assert command in output
    other_command = _MODERN_LOGIN if command == _LEGACY_LOGIN else _LEGACY_LOGIN
    assert other_command not in output


@pytest.mark.parametrize(
    "installed_version", ["missing", None, "unknown"], ids=["missing", "version-absent", "invalid"]
)
def test_login_hint_falls_back_to_modern_command(installed_version, monkeypatch):
    def installed_version_of(package):
        assert package.replace("_", "-") == "huggingface-hub"
        if installed_version == "missing":
            raise PackageNotFoundError(package)
        return installed_version

    monkeypatch.setattr(hf, "version", installed_version_of, raising=False)

    assert hf.hf_login_command() == _MODERN_LOGIN


def test_only_the_hf_login_helper_contains_the_legacy_command():
    helper_path = _SOURCE / "utils" / "hf.py"
    unexpected = []

    for path in sorted(_SOURCE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if path == helper_path:
            helper = next(
                (node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "hf_login_command"),
                None,
            )
            if helper is not None:
                lines = source.splitlines()
                del lines[helper.lineno - 1:helper.end_lineno]
                source = "\n".join(lines)
        if _LEGACY_LOGIN in source:
            unexpected.append(str(path.relative_to(_SOURCE)))

    assert not unexpected, f"Use hf_login_command() in: {', '.join(unexpected)}"
