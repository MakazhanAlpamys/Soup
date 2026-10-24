"""Issue #2178: the atomic writers stage the temporary file next to the target.

``os.replace`` is only atomic when the temporary file lives on the same
filesystem as the target, which is what ``dir=parent`` on ``tempfile.mkstemp``
buys. Nothing pinned it: dropping ``dir=parent`` left every test green while
the writers silently lost their crash guarantee. Each test below fails when
its writer's ``dir=parent`` is removed.
"""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest

from soup_cli.utils import paths


WRITERS = {
    "atomic_write_text": lambda p: paths.atomic_write_text("hello", p),
    "atomic_write_lines": lambda p: paths.atomic_write_lines(["hello"], p),
    "atomic_write_bytes": lambda p: paths.atomic_write_bytes(b"hello", p),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_atomic_writer_stages_the_temp_file_in_the_targets_parent(writer, tmp_path, monkeypatch):
    target = tmp_path / "out" / "artifact.bin"
    monkeypatch.chdir(tmp_path)

    calls: list[dict] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        calls.append(kwargs)
        return real_mkstemp(*args, **kwargs)

    with mock.patch.object(paths.tempfile, "mkstemp", side_effect=recording_mkstemp):
        WRITERS[writer](str(target))

    assert calls, f"{writer} never called mkstemp"
    expected_parent = os.path.dirname(os.path.abspath(target))
    assert calls[0].get("dir") == expected_parent, (
        f"{writer} staged its temporary file outside the target's parent; "
        "os.replace would no longer be atomic"
    )


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_atomic_writer_writes_the_target(writer, tmp_path, monkeypatch):
    target = tmp_path / "out" / "artifact.bin"
    monkeypatch.chdir(tmp_path)
    WRITERS[writer](str(target))
    assert target.exists()
