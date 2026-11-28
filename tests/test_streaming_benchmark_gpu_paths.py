"""Exercise the real shard path guard with the benchmark's directory relationship."""

import os
import sys
import tempfile

import pytest

from benchmarks.harness.streaming_controls.gpu_tests import pytest_arguments


def test_cuda_fixture_path_is_within_cwd_without_expanding_allowed_roots(tmp_path, monkeypatch):
    from soup_cli.utils.layer_shard import _validate_out_dir

    owned = tmp_path / "owned_data"
    repo = owned / "repo"
    repo.mkdir(parents=True)
    home = tmp_path / "home"
    system_tmp = tmp_path / "system_tmp"
    home.mkdir()
    system_tmp.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(system_tmp))

    # A sibling of CWD outside TMPDIR is outside all three admitted roots.
    with pytest.raises(
        ValueError, match=r"^shard output directory must be under \$HOME, \$CWD or \$TMPDIR; got "
    ):
        _validate_out_dir(str(owned / "test_tmp" / "case" / "shards"))

    args = pytest_arguments(repo, ["tests/test_cuda.py"], repo / "results.xml")
    assert args[-1] == "tests/test_cuda.py"
    argument = next(item for item in args if item.startswith("--basetemp="))
    base = argument.split("=", 1)[1]
    shards = os.path.join(base, "case", "shards")
    assert _validate_out_dir(shards) == os.path.realpath(shards)
    assert os.path.commonpath([os.path.realpath(repo), os.path.realpath(shards)]) == (
        os.path.realpath(repo)
    )

    # The fix must not admit unrelated storage paths.
    with pytest.raises(
        ValueError, match=r"^shard output directory must be under \$HOME, \$CWD or \$TMPDIR; got "
    ):
        _validate_out_dir(str(tmp_path / "unrelated" / "shards"))


def test_bad_prepared_path_refused_before_download(tmp_path, monkeypatch):
    from benchmarks.harness.streaming_controls import representative_workload

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(repo))
    monkeypatch.setenv("USERPROFILE", str(repo))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(repo))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--snapshot",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "prepared"),
            "--download-deadline-unix-s",
            "1",
        ],
    )
    monkeypatch.setattr(
        representative_workload,
        "fetch_snapshot",
        lambda *args: pytest.fail("invalid output must be rejected before any download"),
    )
    with pytest.raises(ValueError, match="^prepared shards must be under HOME, CWD or TMPDIR$"):
        representative_workload.main()
    assert not (tmp_path / "model").exists()


def test_correct_prepared_path_reaches_preparation(tmp_path, monkeypatch):
    from benchmarks.harness.streaming_controls import representative_workload

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(repo))
    monkeypatch.setenv("USERPROFILE", str(repo))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(repo))
    output = repo / ".prepared_streaming"
    snapshot = tmp_path / "model"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--snapshot", str(snapshot), "--output", str(output)]
    )
    calls = []
    monkeypatch.setattr(
        representative_workload, "prepare", lambda source, dest: calls.append((source, dest))
    )
    representative_workload.main()
    assert calls == [(snapshot, output)]
    assert not output.exists(), "admission itself must not create the prepared output"


def test_preparation_refuses_existing_output(tmp_path):
    from benchmarks.harness.streaming_controls.representative_workload import (
        validate_preparation_destination,
    )

    with pytest.raises(
        FileExistsError, match="^prepared output must be new; no retry or overwrite$"
    ):
        validate_preparation_destination(tmp_path)


@pytest.mark.requires_symlink
def test_preparation_resolves_symlinked_parent_before_admission(tmp_path, monkeypatch):
    from benchmarks.harness.streaming_controls.representative_workload import (
        validate_preparation_destination,
    )

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "redirect").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(repo))
    monkeypatch.setenv("USERPROFILE", str(repo))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(repo))
    with pytest.raises(ValueError, match="^prepared shards must be under HOME, CWD or TMPDIR$"):
        validate_preparation_destination(repo / "redirect" / "prepared")
    assert not (outside / "prepared").exists()
