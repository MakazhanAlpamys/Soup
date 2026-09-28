"""
Sub-2GB RAM Benchmark & Memory Allocator Diagnostics Utility for Soup.
Author: Muhtar Jaksilikov
"""

import time
import os
import gc
import psutil
import platform
import ctypes
from typing import Dict, Any

class RAMBenchmark:
    """Diagnostic tool to profile system RAM budget, GC efficiency, and PyTorch memory allocations."""

    @staticmethod
    def get_system_memory_info() -> Dict[str, Any]:
        """Collect current process RSS memory, total system RAM, and OS virtual memory statistics."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        vm = psutil.virtual_memory()

        return {
            "pid": os.getpid(),
            "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "process_vsz_mb": round(mem_info.vms / (1024 * 1024), 2),
            "system_total_gb": round(vm.total / (1024**3), 2),
            "system_used_gb": round(vm.used / (1024**3), 2),
            "system_available_gb": round(vm.available / (1024**3), 2),
            "system_percent_used": vm.percent,
            "within_2gb_budget": (mem_info.rss / (1024 * 1024)) <= 2048,
        }

    @staticmethod
    def run_malloc_trim_benchmark() -> Dict[str, Any]:
        """Test C-heap memory reclamation using libc.malloc_trim(0)."""
        process = psutil.Process(os.getpid())
        before_rss = process.memory_info().rss / (1024 * 1024)

        temp_data = [bytearray(1024 * 1024) for _ in range(50)]
        during_rss = process.memory_info().rss / (1024 * 1024)

        del temp_data
        gc.collect()

        after_gc_rss = process.memory_info().rss / (1024 * 1024)

        trim_supported = False
        if platform.system() == "Linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                trim_supported = True
            except Exception:
                pass

        after_trim_rss = process.memory_info().rss / (1024 * 1024)

        return {
            "alloc_size_mb": 50,
            "before_rss_mb": round(before_rss, 2),
            "peak_rss_mb": round(during_rss, 2),
            "after_gc_rss_mb": round(after_gc_rss, 2),
            "after_trim_rss_mb": round(after_trim_rss, 2),
            "freed_by_trim_mb": round(after_gc_rss - after_trim_rss, 2),
            "trim_supported": trim_supported,
        }

    @staticmethod
    def run_full_diagnostics() -> Dict[str, Any]:
        """Run complete low-RAM health diagnostics for Soup."""
        start_time = time.time()
        sys_info = RAMBenchmark.get_system_memory_info()
        trim_info = RAMBenchmark.run_malloc_trim_benchmark()
        elapsed = time.time() - start_time

        return {
            "status": "PASS" if sys_info["within_2gb_budget"] else "WARNING",
            "execution_time_sec": round(elapsed, 4),
            "memory": sys_info,
            "trim_benchmark": trim_info,
        }
