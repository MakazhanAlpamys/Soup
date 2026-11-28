"""An upload of a training output folder leaves the frozen-prefix cache behind."""

import fnmatch
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.frozen_prefix_cache import CACHE_DIRNAME, UPLOAD_IGNORE_PATTERNS

runner = CliRunner()


def ignored(relative_path):
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in UPLOAD_IGNORE_PATTERNS)


def test_ignore_patterns_cover_the_folder_and_its_entries():
    assert ignored(CACHE_DIRNAME)
    assert ignored(f"{CACHE_DIRNAME}/0123abcd.safetensors")
    assert not ignored("adapter_model.safetensors")
    assert not ignored("checkpoint-100/adapter_model.safetensors")


def test_push_command_passes_the_ignore_patterns(tmp_path: Path, monkeypatch):
    from soup_cli.utils import hf as hf_utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}", encoding="utf-8")
    uploads = []

    class Api:
        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

        def upload_file(self, **kwargs):
            pass

    monkeypatch.setattr(hf_utils, "get_hf_api", lambda **kwargs: Api())
    result = runner.invoke(app, ["push", "--model", "model", "--repo", "user/model"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(uploads) == 1
    assert uploads[0]["ignore_patterns"] == UPLOAD_IGNORE_PATTERNS


def test_upload_repo_on_the_hf_hub_passes_the_ignore_patterns(tmp_path: Path, monkeypatch):
    from soup_cli.utils.hubs import upload_repo

    uploads = []

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()

    upload_repo("hf", "user/model", folder_path="out", commit_message="x")

    assert uploads[0]["ignore_patterns"] == UPLOAD_IGNORE_PATTERNS


@pytest.mark.parametrize("hub", ["modelscope", "modelers"])
def test_hubs_without_an_ignore_list_refuse_a_folder_holding_the_cache(
    tmp_path: Path, monkeypatch, hub
):
    from soup_cli.utils.hubs import upload_repo

    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / CACHE_DIRNAME).mkdir(parents=True)

    with pytest.raises(ValueError, match="holds the frozen-prefix activation cache"):
        upload_repo(hub, "user/model", folder_path="out", commit_message="x")
