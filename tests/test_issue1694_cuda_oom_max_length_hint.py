"""Tests for Issue #1694: CUDA out-of-memory error hints include data.max_length.

Pins that both 'CUDA out of memory' and 'OutOfMemoryError' suggest lowering
data.max_length alongside adjusting batch size / gradient accumulation.
"""

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from soup_cli.utils.errors import ERROR_MAP, format_friendly_error


def test_cuda_oom_error_suggests_data_max_length():
    """RuntimeError with 'CUDA out of memory' mentions data.max_length."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "data.max_length" in output
    assert "--batch-size" in output
    assert "GPU ran out of memory during training." in output


def test_out_of_memory_error_type_suggests_data_max_length():
    """OutOfMemoryError exception mentions data.max_length."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)

    class OutOfMemoryError(RuntimeError):
        pass

    with patch("soup_cli.utils.errors.console", test_console):
        exc = OutOfMemoryError("simulated out of memory error")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "data.max_length" in output
    assert "--batch-size" in output
    assert "GPU ran out of memory." in output


def test_error_map_entries_pin_data_max_length():
    """ERROR_MAP entries for CUDA OOM and OutOfMemoryError explicitly pin data.max_length."""
    oom_entries = [
        (pat, msg, fix)
        for pat, msg, fix in ERROR_MAP
        if pat in ("CUDA out of memory", "OutOfMemoryError")
    ]
    assert len(oom_entries) == 2, f"Expected 2 OOM entries in ERROR_MAP, found {len(oom_entries)}"
    for pat, _msg, fix in oom_entries:
        assert (
            "data.max_length" in fix
        ), f"ERROR_MAP pattern {pat!r} fix suggestion does not name data.max_length: {fix!r}"
