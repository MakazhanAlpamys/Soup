"""Keep atomic writes on the destination's filesystem (#2178)."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from soup_cli.utils.paths import atomic_write_bytes, atomic_write_lines, atomic_write_text


@pytest.mark.parametrize(
    ("writer", "content"),
    [
        pytest.param(atomic_write_text, "hello world", id="atomic_write_text"),
        pytest.param(atomic_write_lines, ["hello", " ", "world"], id="atomic_write_lines"),
        pytest.param(atomic_write_bytes, b"hello world", id="atomic_write_bytes"),
    ],
)
def test_atomic_writer_stages_in_destination_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[..., str],
    content: str | list[str] | bytes,
) -> None:
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "nested" / "output.txt"

    with patch("soup_cli.utils.paths.tempfile.mkstemp", wraps=tempfile.mkstemp) as mkstemp:
        writer(content, str(destination))

    mkstemp.assert_called_once()
    assert mkstemp.call_args.kwargs.get("dir") == str(destination.parent)
    assert destination.read_bytes() == b"hello world"
