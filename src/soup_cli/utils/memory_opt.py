"""
Low-RAM Memory Manager and PyTorch Allocator Optimizations for Soup.
Author: Muhtar Jaksilikov
"""

import gc
import ctypes
import os
import platform
import logging

logger = logging.getLogger("soup_cli.memory")

def setup_low_ram_environment():
    """Configure environment variables for <= 2GB RAM budget execution."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "100000")
    os.environ.setdefault("PYTHONMALLOC", "malloc")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

def purge_memory():
    """Force garbage collection and release system C-heap memory using malloc_trim."""
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
            libc.malloc_trim(0)
        except Exception:
            pass

class LowRAMContext:
    """Context manager to ensure low memory footprints during heavy LLM ops."""
    def __enter__(self):
        setup_low_ram_environment()
        purge_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        purge_memory()
