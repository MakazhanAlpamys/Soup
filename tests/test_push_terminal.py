"""soup push prints paths, repo ids and Hub errors as typed, not as Rich markup."""

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import hf as hf_utils
from tests.conftest import strip_ansi

runner = CliRunner()

# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    ":warning:",
    marks=pytest.mark.skipif(sys.platform == "win32", reason="':' is not valid in a Windows name"),
)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture
def model_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


class _FailingApi:
    def __init__(self, message: str) -> None:
        self.message = message

    def create_repo(self, **kwargs) -> None:
        return None

    def upload_folder(self, **kwargs) -> None:
        raise RuntimeError(self.message)


def _push(*args: str):
    return runner.invoke(app, ["push", *args])


@pytest.mark.parametrize("name", ["[red]missing", "[/]missing", "gone:warning:"])
def test_push_missing_model_is_named_literally(model_dir, name):
    result = _push("--model", name, "--repo", "user/model")

    assert result.exit_code == 1
    assert f"Model path not found: {name}" in _plain(result)


def test_push_missing_model_strips_controls_from_the_display(model_dir):
    result = _push("--model", "[bold]gone\x9b\u202e", "--repo", "user/model")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Model path not found: [bold]gone" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("name", ["[red]weights.bin", "[/].bin", COLON_NAME])
def test_push_file_instead_of_directory_is_named_literally(model_dir, name):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")

    result = _push("--model", name, "--repo", "user/model")

    assert result.exit_code == 1
    assert f"Expected a directory, got a file: {name}" in _plain(result)


@pytest.mark.parametrize("repo", ["[red]x", "user/[/]", "bad:warning:"])
def test_push_invalid_repo_is_named_literally(model_dir, repo):
    result = _push("--model", str(model_dir), "--repo", repo)

    assert result.exit_code == 1
    text = _plain(result)
    assert "Invalid --repo:" in text
    assert repr(repo) in text


def test_push_invalid_hub_is_named_literally(model_dir):
    result = _push("--model", str(model_dir), "--repo", "user/model", "--hub", "[red]x")

    assert result.exit_code == 2
    assert "[red]x" in _plain(result)


@pytest.mark.parametrize("message", ["bad [/] value", "[red]denied[/red]", "stop:warning:"])
def test_push_upload_failure_is_shown_as_typed(model_dir, monkeypatch, message):
    monkeypatch.setattr(hf_utils, "get_hf_api", lambda **kwargs: _FailingApi(message))

    result = _push("--model", str(model_dir), "--repo", "user/model")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"Upload failed: {message}" in _plain(result)


def test_push_upload_failure_strips_controls(model_dir, monkeypatch):
    monkeypatch.setattr(
        hf_utils, "get_hf_api", lambda **kwargs: _FailingApi("[bold]boom\x9b\u202e")
    )

    result = _push("--model", str(model_dir), "--repo", "user/model")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Upload failed: [bold]boom" in text
    assert "\x9b" not in text and "\u202e" not in text


class _WorkingApi(_FailingApi):
    def upload_folder(self, **kwargs) -> None:
        return None

    def upload_file(self, **kwargs) -> None:
        return None


def test_push_invalid_collection_slug_is_named_literally(model_dir, monkeypatch):
    monkeypatch.setattr(hf_utils, "get_hf_api", lambda **kwargs: _WorkingApi(""))
    monkeypatch.setattr(hf_utils, "add_to_collection", lambda **kwargs: None)

    result = _push("--model", str(model_dir), "--repo", "user/model", "--collection", "[red]x")

    assert result.exit_code == 1
    assert "Invalid --collection slug:" in _plain(result)
    assert repr("[red]x") in _plain(result)


def test_push_collection_failure_is_shown_as_typed(model_dir, monkeypatch):
    def _refuse(**kwargs):
        raise RuntimeError("no such [/] collection")

    monkeypatch.setattr(hf_utils, "get_hf_api", lambda **kwargs: _WorkingApi(""))
    monkeypatch.setattr(hf_utils, "add_to_collection", _refuse)

    result = _push(
        "--model", str(model_dir), "--repo", "user/model", "--collection", "user/slug-abc123"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Could not add to collection: no such [/] collection" in _plain(result)
