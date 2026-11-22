"""Process-wide benchmark hooks, loaded by Python at startup from PYTHONPATH.

``orchestrate.py`` puts this folder on ``PYTHONPATH`` for EVERY arm alike (Soup streaming
arms, the Hugging Face / DeepSpeed arms, the old-revision arm), so both hooks below act the
same way on all of them. Both are inert unless their environment variable is set, and
neither imports torch or transformers at interpreter start-up: they install an import hook
that fires after the library is imported.

1. ``SOUP_BENCH_VRAM_CAP_GB=<gb>``  hard VRAM budget.
   After ``torch.cuda`` has initialised for the first time, call
   ``torch.cuda.set_per_process_memory_fraction(gb / device_total)`` on every visible
   device, before the allocation that triggered the initialisation. The cap is applied
   AFTER the original initialisation returns, never before it: the caching allocator reads
   ``PYTORCH_CUDA_ALLOC_CONF`` when the CUDA context is created, and ``soup train`` exports
   its ``expandable_segments:True`` hint just before that moment (src/soup_cli/utils/gpu.py:
   239-253, called at src/soup_cli/commands/train.py:1611). Initialising CUDA here at
   import time would make that hint a no-op and silently change what arm S1 measures.
   The cap limits the PyTorch caching allocator (reserved memory); the CUDA context itself
   (a few hundred MB) and pinned host memory are outside it.

2. ``SOUP_BENCH_TRACE_OUT=<file>``  passive per-step recorder.
   Appends ``bench_trace``'s callback class to ``transformers.trainer.DEFAULT_CALLBACKS``,
   so a ``soup train`` run (which this harness does not modify) is recorded by the same
   code that records ``hf_baseline.py``. The callback only reads trainer events.

A ``sitecustomize`` further down ``sys.path`` (some images ship one) is still executed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOOKS: dict[str, list] = {}


def _note(message: str) -> None:
    sys.stderr.write(f"[ds_compare] {message}\n")
    sys.stderr.flush()


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run registered callbacks right after a named module finished importing."""

    def __init__(self) -> None:
        self._resolving: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _HOOKS or fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._resolving.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module, _original=original_exec, _name=fullname):
            _original(module)
            for hook in _HOOKS.pop(_name, []):
                try:
                    hook(module)
                except Exception as exc:  # noqa: BLE001 - a hook must never break the import
                    _note(f"hook for {_name} failed: {type(exc).__name__}: {exc}")

        loader.exec_module = exec_module
        return spec


def _when_imported(name: str, hook) -> None:
    module = sys.modules.get(name)
    if module is not None:
        hook(module)
        return
    if not any(isinstance(finder, _PostImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder())
    _HOOKS.setdefault(name, []).append(hook)


# --- hook 1: VRAM cap -----------------------------------------------------------------


def _install_vram_cap(cuda_module) -> None:
    cap_gb = float(os.environ["SOUP_BENCH_VRAM_CAP_GB"])
    if cap_gb <= 0:
        raise ValueError("SOUP_BENCH_VRAM_CAP_GB must be positive")
    original = cuda_module._lazy_init
    state = {"done": False, "busy": False}

    def _lazy_init_then_cap():
        original()
        if state["done"] or state["busy"] or not cuda_module.is_initialized():
            return
        state["busy"] = True
        try:
            for index in range(cuda_module.device_count()):
                total = cuda_module.get_device_properties(index).total_memory
                fraction = min(1.0, cap_gb * 1024**3 / total)
                cuda_module.set_per_process_memory_fraction(fraction, index)
                _note(
                    f"VRAM cap applied: {cap_gb:g} GB = fraction {fraction:.4f} of "
                    f"{total / 1024**3:.2f} GB on cuda:{index} (pid {os.getpid()})"
                )
            state["done"] = True
        finally:
            state["busy"] = False

    cuda_module._lazy_init = _lazy_init_then_cap
    memory_module = getattr(cuda_module, "memory", None)
    if memory_module is not None and getattr(memory_module, "_lazy_init", None) is original:
        memory_module._lazy_init = _lazy_init_then_cap


# --- hook 2: passive trace callback --------------------------------------------------


def _install_trace_callback(trainer_module) -> None:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from bench_trace import make_callback_class

    callback_class = make_callback_class()
    trainer_module.DEFAULT_CALLBACKS.append(callback_class)


def _install() -> None:
    if os.environ.get("SOUP_BENCH_VRAM_CAP_GB"):
        try:
            float(os.environ["SOUP_BENCH_VRAM_CAP_GB"])
        except ValueError:
            _note("SOUP_BENCH_VRAM_CAP_GB is not a number; no cap will be applied")
        else:
            _when_imported("torch.cuda", _install_vram_cap)
    if os.environ.get("SOUP_BENCH_TRACE_OUT"):
        _when_imported("transformers.trainer", _install_trace_callback)


def _chain_next_sitecustomize() -> None:
    """Execute the next ``sitecustomize`` on sys.path, if there is one."""
    others = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", others)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        _note(f"the next sitecustomize failed: {type(exc).__name__}: {exc}")


try:
    _install()
except Exception as exc:  # noqa: BLE001 - never stop the interpreter from starting
    _note(f"sitecustomize hooks not installed: {type(exc).__name__}: {exc}")
_chain_next_sitecustomize()
