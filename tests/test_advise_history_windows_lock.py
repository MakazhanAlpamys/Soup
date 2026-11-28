"""The Windows advise-history lock must not fall back to an unlocked append (#2569).

The old Windows path used ``msvcrt.LK_LOCK`` and, on ``OSError``, closed the
lock file and wrote anyway. Append-mode writes are not atomic across handles on
Windows, so a busy lock could lose a verdict. The path now retries
``LK_NBLCK`` until a deadline and raises ``TimeoutError`` naming the history
file; POSIX ``flock`` behaviour is unchanged.

These tests patch ``os.name`` and ``msvcrt``, so they run on every OS. They
never claim to reproduce the original Windows CI failure locally.
"""

from __future__ import annotations

import os
import sys
import time
import types

import pytest

from soup_cli.utils import advise_history

WINDOWS = "nt"


def _install_fake_windows(monkeypatch, *, lock_fails: bool, clock: dict):
    calls: list[tuple[str, int]] = []

    def locking(fd, mode, length):
        calls.append(("lock", mode))
        if lock_fails:
            raise OSError("lock busy")
        return None

    fake = types.SimpleNamespace(
        LK_LOCK=1,  # the old constant, so the pre-fix path also runs
        LK_NBLCK=2,
        LK_UNLCK=3,
        locking=locking,
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(os, "name", WINDOWS)
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda sec: clock.__setitem__("t", clock["t"] + sec))
    return calls


def test_a_busy_lock_raises_and_writes_nothing(tmp_path, monkeypatch):
    target = tmp_path / "history.jsonl"
    clock = {"t": 0.0}
    with monkeypatch.context() as patch:
        _install_fake_windows(patch, lock_fails=True, clock=clock)
        with pytest.raises(TimeoutError) as excinfo:
            advise_history._append_with_lock(str(target), '{"a": 1}\n')

    message = str(excinfo.value)
    assert str(target) in message
    assert not target.exists(), "no verdict may be written without the lock"
    assert clock["t"] >= advise_history._NT_LOCK_DEADLINE_SEC


def test_an_available_lock_appends_and_unlocks(tmp_path, monkeypatch):
    target = tmp_path / "history.jsonl"
    clock = {"t": 0.0}
    with monkeypatch.context() as patch:
        calls = _install_fake_windows(patch, lock_fails=False, clock=clock)
        advise_history._append_with_lock(str(target), '{"a": 1}\n')

    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'
    assert calls == [("lock", 2), ("lock", 3)]
    assert clock["t"] == 0.0, "an available lock does not wait"


def test_no_unrelated_warning_is_needed_for_the_fake_clock(tmp_path, monkeypatch):
    """The retry loop is bounded, so a permanently busy lock still returns."""
    target = tmp_path / "history.jsonl"
    clock = {"t": 0.0}
    with monkeypatch.context() as patch:
        calls = _install_fake_windows(patch, lock_fails=True, clock=clock)
        with pytest.raises(TimeoutError):
            advise_history._append_with_lock(str(target), "x\n")
        attempts = len(calls)

    # Bounded, not infinite: the deadline is reached in a fixed number of tries.
    assert 100 < attempts < 1000
    assert clock["t"] >= advise_history._NT_LOCK_DEADLINE_SEC
