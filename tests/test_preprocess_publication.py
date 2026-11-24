"""Preprocessing publishes complete caches and preserves the previous usable output."""

import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from tests.test_issue1038_preprocess_to_pretokenized import _gate, _preprocess


def _invoke():
    from soup_cli.cli import app

    return CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])


@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    cache = _preprocess(tmp_path, monkeypatch, "data.jsonl")
    previous = (_gate(tmp_path, "data.jsonl", cache)[0].to_list(),
                (cache / "metadata.json").read_bytes())
    new_row = {"messages": [{"role": "user", "content": "France ?"},
                            {"role": "assistant", "content": "What ."}]}
    (tmp_path / "data.jsonl").write_text((json.dumps(new_row) + "\n") * 3)
    return cache, previous


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("step", ["staging", "arrow", "metadata_open", "metadata_write",
                                 "validation", "install"])
def test_publication_failure_preserves_prior_cache(
    tmp_path, monkeypatch, publication_case, existing, step,
):
    import datasets

    cache, previous = publication_case
    if not existing:
        shutil.rmtree(cache)
    real_mkdtemp = tempfile.mkdtemp
    real_save = datasets.Dataset.save_to_disk
    real_load = datasets.load_from_disk
    real_open = open
    real_dump = json.dump
    real_replace = os.replace

    def create(*args, **kwargs):
        if str(kwargs.get("prefix", "")).startswith(".tmp_"):
            raise OSError("injected staging failure")
        return real_mkdtemp(*args, **kwargs)

    def save(dataset, path, *args, **kwargs):
        real_save(dataset, path, *args, **kwargs)
        raise OSError("injected arrow failure")

    def open_metadata(path, mode="r", *args, **kwargs):
        if Path(path).name == "metadata.json" and "w" in mode:
            raise OSError("injected metadata_open failure")
        return real_open(path, mode, *args, **kwargs)

    def dump(value, handle, *args, **kwargs):
        if Path(handle.name).name == "metadata.json":
            handle.write("{")
            raise OSError("injected metadata_write failure")
        return real_dump(value, handle, *args, **kwargs)

    def load(path, *args, **kwargs):
        if Path(path).name.startswith(".tmp_"):
            raise OSError("injected validation failure")
        return real_load(path, *args, **kwargs)

    def replace(source, destination):
        if Path(source).name.startswith(".tmp_") and Path(destination) == cache:
            raise OSError("injected install failure")
        return real_replace(source, destination)

    hooks = {
        "staging": (tempfile, "mkdtemp", create),
        "arrow": (datasets.Dataset, "save_to_disk", save),
        "metadata_open": (__import__("builtins"), "open", open_metadata),
        "metadata_write": (json, "dump", dump),
        "validation": (datasets, "load_from_disk", load),
        "install": (os, "replace", replace),
    }
    with monkeypatch.context() as fault:
        fault.setattr(*hooks[step])
        result = _invoke()
    assert result.exit_code == 1, result.output
    assert f"injected {step} failure" in str(result.exception)
    if existing:
        assert (cache / "metadata.json").read_bytes() == previous[1]
        assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
        for changes in ({"max_length": 256}, {"base": "different/base"}):
            with pytest.raises(ValueError, match="cache hash mismatch"):
                _gate(tmp_path, "data.jsonl", cache, **changes)
    else:
        assert not cache.exists()
    assert not list(cache.parent.glob(".tmp_*"))
    assert not list(cache.parent.glob(".backup_*"))


def test_successful_rebuild_publishes_new_rows_and_complete_metadata(
    tmp_path, monkeypatch, publication_case,
):
    cache, previous = publication_case
    real_replace = os.replace
    publications = []

    def replace(source, destination):
        if Path(source).name.startswith(".tmp_") and Path(destination) == cache:
            from datasets import load_from_disk

            metadata = json.loads((Path(source) / "metadata.json").read_text())
            rows = load_from_disk(source, keep_in_memory=True).to_list()
            assert metadata["row_count"] == len(rows) == 3
            assert metadata["cache_key"] == cache.name
            publications.append(rows)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    result = _invoke()
    assert result.exit_code == 0, result.output
    current = _gate(tmp_path, "data.jsonl", cache)[0].to_list()
    assert publications == [current]
    assert len(current) == 3 and current != previous[0]
    assert not list(cache.parent.glob(".backup_*"))
    assert not list(cache.parent.glob(".tmp_*"))


def test_external_metadata_free_dataset_keeps_its_advisory(tmp_path, publication_case):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.utils.pretokenized import maybe_load_pretokenized

    cache, _ = publication_case
    (cache / "metadata.json").unlink()
    cfg = load_config_from_string(
        "base: different/base\ndata:\n  train: data.jsonl\n  format: pre_tokenized\n"
        f"  tokenized_path: {os.path.relpath(cache, tmp_path)}\n"
        "  max_length: 256\n  val_split: 0.0\n"
    )
    captured = MagicMock()
    rows, _ = maybe_load_pretokenized(cfg.data, cfg.base, captured, task="sft")
    assert len(rows) == 4
    assert "no metadata.json" in str(captured.print.call_args_list)


@pytest.mark.parametrize("step", ["backup_create", "backup_reservation", "backup_rename"])
def test_backup_failure_leaves_prior_cache_usable(tmp_path, monkeypatch, publication_case, step):
    cache, previous = publication_case
    real_create, real_rmdir, real_replace = tempfile.mkdtemp, Path.rmdir, os.replace

    def create(*args, **kwargs):
        if str(kwargs.get("prefix", "")).startswith(".backup_"):
            raise OSError("injected backup_create")
        return real_create(*args, **kwargs)

    def rmdir(path, *args, **kwargs):
        if path.name.startswith(".backup_"):
            raise OSError("injected backup_reservation")
        return real_rmdir(path, *args, **kwargs)

    def replace(source, destination):
        if Path(source) == cache:
            raise OSError("injected backup_rename")
        return real_replace(source, destination)

    hooks = {"backup_create": (tempfile, "mkdtemp", create),
             "backup_reservation": (Path, "rmdir", rmdir),
             "backup_rename": (os, "replace", replace)}
    with monkeypatch.context() as fault:
        fault.setattr(*hooks[step])
        result = _invoke()
    assert result.exit_code == 1
    assert f"injected {step}" in str(result.exception)
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert (cache / "metadata.json").read_bytes() == previous[1]
    assert not list(cache.parent.glob(".tmp_*"))


@pytest.mark.parametrize("corruption", ["metadata", "rows", "features"])
def test_staged_validation_refuses_corruption(
    tmp_path, monkeypatch, publication_case, corruption,
):
    import datasets

    cache, previous = publication_case
    real_dump, real_load = json.dump, datasets.load_from_disk

    def dump(value, handle, *args, **kwargs):
        if Path(handle.name).name == "metadata.json":
            value = {**value, "cache_key": "wrong-key"}
        return real_dump(value, handle, *args, **kwargs)

    def load(path, *args, **kwargs):
        loaded = real_load(path, *args, **kwargs)
        if Path(path).name.startswith(".tmp_"):
            assert kwargs.get("keep_in_memory") is True
            if corruption == "rows":
                loaded = loaded.select([0])
            else:
                loaded = loaded.remove_columns("labels")
        return loaded

    with monkeypatch.context() as fault:
        if corruption == "metadata":
            fault.setattr(json, "dump", dump)
        else:
            fault.setattr(datasets, "load_from_disk", load)
        result = _invoke()
    assert result.exit_code == 1
    assert "preprocess staged" in str(result.exception)
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert (cache / "metadata.json").read_bytes() == previous[1]


def test_failed_rollback_retains_and_reports_usable_backup(
    tmp_path, monkeypatch, publication_case,
):
    cache, previous = publication_case
    real_replace = os.replace

    def replace(source, destination):
        if Path(destination) == cache:
            raise OSError("injected install/restore failure")
        return real_replace(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", replace)
        result = _invoke()
    assert result.exit_code == 1 and not cache.exists()
    backup, = cache.parent.glob(".backup_*")
    assert str(backup) in result.output
    assert "Could not restore" in result.output
    assert (backup / "metadata.json").read_bytes() == previous[1]
    assert _gate(tmp_path, "data.jsonl", backup)[0].to_list() == previous[0]
    assert not list(cache.parent.glob(".tmp_*"))


def test_failed_backup_cleanup_keeps_new_cache_and_reports_backup(
    tmp_path, monkeypatch, publication_case,
):
    cache, previous = publication_case
    real_remove = shutil.rmtree

    def remove(path, *args, **kwargs):
        if Path(path).name.startswith(".backup_"):
            raise OSError("injected backup cleanup failure")
        return real_remove(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(shutil, "rmtree", remove)
        result = _invoke()
    assert result.exit_code == 0, result.output
    assert len(_gate(tmp_path, "data.jsonl", cache)[0]) == 3
    backup, = cache.parent.glob(".backup_*")
    assert str(backup) in result.output and "New cache is complete" in result.output
    assert _gate(tmp_path, "data.jsonl", backup)[0].to_list() == previous[0]
    assert not list(cache.parent.glob(".tmp_*"))


def test_cleanup_does_not_touch_other_runs(tmp_path, monkeypatch, publication_case):
    import datasets

    cache, previous = publication_case
    stale = [cache.parent / (prefix + cache.name) for prefix in (".tmp_", ".backup_")]
    for path in stale:
        path.mkdir()
        (path / "keep").write_text("other run")
    with monkeypatch.context() as fault:
        fault.setattr(datasets.Dataset, "save_to_disk", MagicMock(side_effect=OSError("save")))
        result = _invoke()
    assert result.exit_code == 1
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    for path in stale:
        assert (path / "keep").read_text() == "other run"


@pytest.mark.parametrize("overwrite", [False, True])
def test_two_writers_stage_separately_and_serialize_commit(
    tmp_path, monkeypatch, publication_case, overwrite,
):
    from datasets import Dataset, load_from_disk

    from soup_cli.commands import data

    cache, previous = publication_case
    metadata = json.loads(previous[1])
    first_rows = previous[0][:1]
    second_rows = previous[0][:2]
    first_at_install, second_at_lock, release_first = Event(), Event(), Event()
    real_replace, real_lock = os.replace, data._preprocess_publish_lock
    real_save = Dataset.save_to_disk
    staging_paths = []

    def save(dataset, path, *args, **kwargs):
        staging_paths.append(Path(path))
        return real_save(dataset, path, *args, **kwargs)

    @contextmanager
    def lock(target):
        if first_at_install.is_set():
            second_at_lock.set()
        with real_lock(target):
            yield

    def replace(source, destination):
        if Path(source).name.startswith(".tmp_") and not first_at_install.is_set():
            first_at_install.set()
            assert release_first.wait(5), "first writer was not released"
        return real_replace(source, destination)

    monkeypatch.setattr(data, "_preprocess_publish_lock", lock)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Dataset, "save_to_disk", save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(data._publish_preprocess_cache, Dataset.from_list(first_rows), cache,
                            {**metadata, "row_count": 1}, overwrite=True)
        try:
            assert first_at_install.wait(5), "first writer did not reach install"
            second = pool.submit(data._publish_preprocess_cache, Dataset.from_list(second_rows),
                                 cache, {**metadata, "row_count": 2}, overwrite=overwrite)
            assert second_at_lock.wait(5), "second writer did not reach commit lock"
            assert not second.done()
        finally:
            release_first.set()
        assert first.result(timeout=5) is True
        assert second.result(timeout=5) is overwrite
    assert len(staging_paths) == 2 and staging_paths[0] != staging_paths[1]
    expected_rows = second_rows if overwrite else first_rows
    assert load_from_disk(str(cache)).to_list() == expected_rows
    assert json.loads((cache / "metadata.json").read_text())["row_count"] == len(expected_rows)
    assert not list(cache.parent.glob(".tmp_*"))
    assert not list(cache.parent.glob(".backup_*"))
    assert cache.with_name(".lock_" + cache.name).is_file()


def test_lock_timeout_leaves_cache_and_releases_waiter_descriptor(
    tmp_path, monkeypatch, publication_case,
):
    from soup_cli.commands import data

    cache, previous = publication_case
    with data._preprocess_publish_lock(cache):
        with monkeypatch.context() as clock:
            clock.setattr(data, "monotonic", MagicMock(side_effect=[0, 31]))
            result = _invoke()
        assert result.exit_code == 1
        assert isinstance(result.exception, TimeoutError)
        assert "publication is busy; retry" in str(result.exception)
    # A timed-out waiter must not prevent a later writer from taking the same lock.
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert _invoke().exit_code == 0


@pytest.mark.requires_symlink
@pytest.mark.parametrize("name", ["target", "lock"])
def test_publication_refuses_symlinks_without_touching_destination(
    tmp_path, monkeypatch, publication_case, name,
):
    cache, previous = publication_case
    other = tmp_path / "other"
    if name == "target":
        cache.rename(other)
        cache.symlink_to(other, target_is_directory=True)
    else:
        other.write_bytes(b"do not touch")
        cache.with_name(".lock_" + cache.name).unlink()
        cache.with_name(".lock_" + cache.name).symlink_to(other)
    result = _invoke()
    assert result.exit_code == 1
    assert "symlink" in str(result.exception).lower() or "symbolic link" in str(
        result.exception,
    ).lower()
    if name == "target":
        assert (other / "metadata.json").read_bytes() == previous[1]
    else:
        assert other.read_bytes() == b"do not touch"
        assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert not list(cache.parent.glob(".tmp_*"))


def test_skip_existing_target_keeps_output_and_exit_zero(tmp_path, publication_case):
    from soup_cli.cli import app

    cache, previous = publication_case
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml"])
    assert result.exit_code == 0 and "Target already exists" in result.output
    assert (cache / "metadata.json").read_bytes() == previous[1]
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]


def test_file_target_is_preserved(tmp_path, publication_case):
    cache, _ = publication_case
    shutil.rmtree(cache)
    cache.write_bytes(b"not a cache directory")
    result = _invoke()
    assert result.exit_code == 1
    assert "target must be a directory" in str(result.exception)
    assert cache.read_bytes() == b"not a cache directory"


def test_unexpected_lock_error_preserves_cache(tmp_path, monkeypatch, publication_case):
    import errno

    cache, previous = publication_case
    if os.name == "nt":
        import msvcrt

        module, name = msvcrt, "locking"
    else:
        import fcntl

        module, name = fcntl, "flock"
    with monkeypatch.context() as fault:
        fault.setattr(module, name, MagicMock(side_effect=OSError(errno.EIO, "lock I/O")))
        result = _invoke()
    assert result.exit_code == 1
    assert isinstance(result.exception, OSError) and result.exception.errno == errno.EIO
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert not list(cache.parent.glob(".tmp_*"))


def test_keyboard_interrupt_during_install_restores_previous_cache(
    tmp_path, monkeypatch, publication_case,
):
    cache, previous = publication_case
    real_replace = os.replace

    def replace(source, destination):
        if Path(source).name.startswith(".tmp_"):
            raise KeyboardInterrupt
        return real_replace(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", replace)
        result = _invoke()
    assert result.exit_code == 130
    assert (cache / "metadata.json").read_bytes() == previous[1]
    assert _gate(tmp_path, "data.jsonl", cache)[0].to_list() == previous[0]
    assert not list(cache.parent.glob(".backup_*"))


def _process_writer(root, cache, rows, metadata, phase, reached, release, output):
    """Spawned worker uses real OS locks and Arrow, with events around the commit boundary."""
    from datasets import Dataset

    from soup_cli.commands import data

    os.chdir(root)
    real_replace, real_lock = os.replace, data._preprocess_publish_lock

    def replace(source, destination):
        if Path(source).name.startswith(".tmp_") and phase == "first":
            reached.set()
            assert release.wait(15), "parent did not release first writer"
        return real_replace(source, destination)

    @contextmanager
    def lock(target):
        if phase == "second":
            reached.set()
        with real_lock(target):
            yield

    data._preprocess_publish_lock = lock
    os.replace = replace
    try:
        result = data._publish_preprocess_cache(
            Dataset.from_list(rows), Path(cache), metadata, overwrite=True,
        )
        output.put((phase, result, None))
    except BaseException as exc:
        output.put((phase, False, repr(exc)))
        raise


def test_two_processes_publish_one_complete_generation(tmp_path, publication_case):
    from datasets import load_from_disk

    cache, previous = publication_case
    metadata = json.loads(previous[1])
    context = multiprocessing.get_context("spawn")
    first_at_install, second_at_lock, release_first = (
        context.Event(), context.Event(), context.Event()
    )
    output = context.Queue()
    first = context.Process(target=_process_writer, args=(
        str(tmp_path), str(cache), previous[0][:1], {**metadata, "row_count": 1}, "first",
        first_at_install, release_first, output,
    ))
    second = context.Process(target=_process_writer, args=(
        str(tmp_path), str(cache), previous[0][:2], {**metadata, "row_count": 2}, "second",
        second_at_lock, release_first, output,
    ))
    first.start()
    try:
        assert first_at_install.wait(15), "first process did not reach install"
        second.start()
        assert second_at_lock.wait(15), "second process did not reach commit lock"
        # Second must remain alive while the first holds the lock, not fail at the empty target.
        second.join(timeout=0.2)
        assert second.is_alive()
        release_first.set()
        first.join(timeout=15)
        second.join(timeout=15)
        assert first.exitcode == second.exitcode == 0
        assert {output.get(timeout=2), output.get(timeout=2)} == {
            ("first", True, None), ("second", True, None),
        }
    finally:
        release_first.set()
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
        output.close()
        output.join_thread()
    assert load_from_disk(str(cache)).to_list() == previous[0][:2]
    assert json.loads((cache / "metadata.json").read_text())["row_count"] == 2
    assert not list(cache.parent.glob(".tmp_*"))
    assert not list(cache.parent.glob(".backup_*"))
