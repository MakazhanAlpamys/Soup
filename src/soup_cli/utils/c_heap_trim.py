"""C-heap memory reclamation and allocator trim utility for Soup."""

import ctypes
import gc
import logging
import platform

logger = logging.getLogger("soup_cli.utils.c_heap_trim")


def reclaim_c_heap_memory() -> bool:
    """Force garbage collection and release unmapped C-heap memory using malloc_trim(0).

    Returns:
        bool: True if libc.malloc_trim succeeded, False otherwise.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            result = libc.malloc_trim(0)
            logger.debug(f"libc.malloc_trim(0) returned {result}")
            return bool(result)
        except Exception as exc:
            logger.debug(f"libc.malloc_trim not available: {exc}")
            return False

    return False
