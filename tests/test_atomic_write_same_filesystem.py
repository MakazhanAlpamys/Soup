"""Atomic writers must work when the system temp directory is on another volume.

The filesystem boundary is simulated at os.replace: writes still use real files,
but moving a staged file from another directory raises EXDEV. This pins a case
that a single-volume developer machine otherwise cannot distinguish.
"""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path

import pytest

from soup_cli.utils import paths


@pytest.mark.parametrize("writer", ["text", "lines", "bytes"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "replace"])
def test_atomic_writer_works_with_system_temp_on_another_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str, existing: bool,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
    target = work / "reports" / "nested" / "artifact"
    target.parent.mkdir(parents=True)
    previous = b"last good artifact\n"
    if existing:
        target.write_bytes(previous)
    payload = "new caf\u00e9\nsecond row\n"
    expected = (
        payload.encode("utf-8")
        if writer == "bytes"
        else payload.replace("\n", os.linesep).encode("utf-8")
    )
    replacements = []
    real_replace = os.replace

    def replace_on_same_volume(source: str, destination: str) -> None:
        staged = Path(source)
        output = Path(destination)
        if os.path.realpath(staged.parent) != os.path.realpath(output.parent):
            raise OSError(errno.EXDEV, "staging and destination are on different volumes")
        assert os.path.realpath(staged) != os.path.realpath(target)
        assert staged.read_bytes() == expected
        assert target.exists() is existing
        if existing:
            assert target.read_bytes() == previous
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_on_same_volume)
    output = os.path.relpath(target, work)
    if writer == "text":
        result = paths.atomic_write_text(payload, output)
    elif writer == "lines":
        result = paths.atomic_write_lines(iter(payload.splitlines(keepends=True)), output)
    else:
        result = paths.atomic_write_bytes(expected, output)

    assert result == os.path.realpath(target)
    assert target.read_bytes() == expected
    assert len(replacements) == 1
    assert list(system_temp.iterdir()) == []
    assert list(target.parent.iterdir()) == [target]
