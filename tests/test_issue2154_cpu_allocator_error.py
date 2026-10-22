"""#2154: a refused CPU allocation gets a readable message, and CUDA keeps the GPU one."""

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from soup_cli.utils.errors import format_friendly_error
from tests.conftest import strip_ansi


def _shown(exc: Exception) -> str:
    buf = StringIO()
    with patch("soup_cli.utils.errors.console", Console(file=buf, stderr=False, width=200)):
        format_friendly_error(exc, verbose=False)
    return " ".join(strip_ansi(buf.getvalue()).split())


def test_a_refused_cpu_allocation_says_what_to_do():
    shown = _shown(
        RuntimeError(
            "DefaultCPUAllocator: not enough memory: you tried to allocate 22548578304 bytes. "
            "Buy new RAM!"
        )
    )

    assert "Error: The CPU allocator could not reserve memory for a tensor." in shown
    assert (
        "Free memory by closing other programs, add swap (Linux) or enlarge the page file "
        "(Windows), or use a smaller model."
    ) in shown


def test_a_cuda_out_of_memory_message_keeps_the_gpu_entry():
    shown = _shown(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))

    assert "GPU ran out of memory during training." in shown
    assert "CPU allocator" not in shown
