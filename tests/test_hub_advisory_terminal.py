"""The shared non-HF prefetch helper prints literal, relative advisories."""

import os
import sys
from io import StringIO

import pytest
from rich.console import Console

from soup_cli.utils import hubs

SPELLINGS = [
    ("x[red]n", "x[red]n"),
    ("x\\[n", "x\\[n"),
    ("x\x9b\u202en", "xn"),
    (":warning:", ":warning:"),
]


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_fetch_receipt_keeps_original_repo_and_returned_path(tmp_path, monkeypatch, value, shown):
    monkeypatch.chdir(tmp_path)
    repo_id = f"owner/model-{value}"
    local_path = str(tmp_path / f"snapshot-{value}")
    calls = []

    def download(hub, name, *, local_dir):
        calls.append((hub, name, local_dir))
        return local_path

    monkeypatch.setattr(hubs, "download_repo", download)
    stream = StringIO()
    console = Console(
        file=stream,
        width=300,
        height=100,
        force_terminal=True,
        no_color=False,
        color_system="standard",
    )

    assert hubs.prefetch_model_from_hub(repo_id, "modelscope", console=console) == local_path

    assert len(calls) == 1
    assert calls[0][:2] == ("modelscope", repo_id)
    assert os.path.commonpath([calls[0][2], os.getcwd()]) == os.getcwd()
    display_path = os.path.relpath(tmp_path / f"snapshot-{shown}")
    assert stream.getvalue() == (
        f"\x1b[2mFetched owner/model-{shown} from hub=modelscope → {display_path}\x1b[0m\n"
    )


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_cache_hit_shows_literal_relative_path_without_fetch(tmp_path, monkeypatch, value, shown):
    if sys.platform == "win32" and ":" in value:
        pytest.skip("colon is not a valid Windows filename character")
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / f"cache-{value}"
    snapshot = cache / "owner__model"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        pytest.fail("cache hit should not download")

    monkeypatch.setattr(hubs, "download_repo", unexpected_download)
    stream = StringIO()
    console = Console(
        file=stream,
        width=300,
        height=100,
        force_terminal=True,
        no_color=False,
        color_system="standard",
    )

    actual = hubs.prefetch_model_from_hub(
        "owner/model", "modelscope", cache_root=str(cache), console=console
    )

    assert actual == os.path.realpath(snapshot)
    display_path = os.path.relpath(tmp_path / f"cache-{shown}" / "owner__model")
    assert stream.getvalue() == f"\x1b[2mUsing cached snapshot at {display_path}\x1b[0m\n"
