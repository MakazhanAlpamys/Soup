"""``loop.yaml`` is on disk before it replaces the old one.

``os.replace`` without an ``fsync`` first can survive a power loss as a zero-length file: the
rename is durable and the data is not, and ``read_state`` then refuses the file ("not valid
JSON") so the loop will not start. The temporary file is flushed and fsynced before the
rename (and the directory after it, on POSIX).
"""

from __future__ import annotations

import os

import pytest

import soup_cli.utils.loop_state as loop_state
from soup_cli.utils.loop_state import LoopState, write_state


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _state() -> LoopState:
    return LoopState(served_model="base", eval_suite="e.json", baseline="b")


def test_the_file_is_fsynced_before_it_is_renamed_over_loop_yaml(cwd, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)

    write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

    assert events[:2] == ["fsync", "replace"], events


def test_the_data_fsynced_is_the_whole_file(cwd, monkeypatch):
    sizes = []
    real_fsync = os.fsync

    def fsync(fd):
        sizes.append(os.fstat(fd).st_size)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    target = str(cwd / ".soup" / "loop.yaml")
    write_state(_state(), target)

    assert sizes and sizes[0] == os.path.getsize(target) > 0


@pytest.mark.skipif(os.name != "posix", reason="a directory can only be fsynced on POSIX")
def test_the_directory_is_fsynced_after_the_rename(cwd, monkeypatch):
    opened = []
    real_fsync = os.fsync

    def fsync(fd):
        opened.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

    import stat

    assert any(stat.S_ISDIR(mode) for mode in opened)


def test_a_failed_fsync_leaves_the_old_file_and_no_temp_file(cwd, monkeypatch):
    target = str(cwd / ".soup" / "loop.yaml")
    write_state(_state(), target)
    before = open(target, "rb").read()

    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(loop_state.os, "fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        write_state(
            LoopState(served_model="base", eval_suite="other.json", baseline="b"), target
        )
    monkeypatch.undo()

    assert open(target, "rb").read() == before
    assert sorted(os.listdir(cwd / ".soup")) == ["loop.yaml"]
