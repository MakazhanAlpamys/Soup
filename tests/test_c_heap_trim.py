"""Tests for C-heap memory reclamation utility."""

from soup_cli.utils.c_heap_trim import reclaim_c_heap_memory


def test_reclaim_c_heap_memory_executes():
    # Verify execution without errors on all platforms
    result = reclaim_c_heap_memory()
    assert isinstance(result, bool)
