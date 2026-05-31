"""Activation offloading — move stored activations to CPU or disk.

During the backward pass, cached activations are the main VRAM cost. Offloading
them to RAM (CPU) or a scratch file (disk) trades IO/PCIe bandwidth for VRAM
headroom — useful for single-GPU large-batch training on small VRAM.

Implemented as a context manager that installs + removes the offload hooks
for the duration of ``trainer.train()``. The hooks wrap ``torch.utils.hooks``
/ saved-tensor hooks and move saved tensors to the target device.

Unlike DeepSpeed ZeRO offload (which is partitioning-aware), this is a
*per-tensor* offload that composes with any backend (DDP, FSDP, or single-GPU).
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import Any, Callable, Generator, Optional, Tuple

HookPair = Tuple[Callable[[Any], Any], Callable[[Any], Any]]


def validate_offload_config(
    target: Optional[str],
    backend: str,
    device: str,
    save_dir: Optional[str] = None,
) -> list[str]:
    """Validate ``activation_offloading`` config.

    Args:
        target: None / 'cpu' / 'disk'.
        backend: transformers / unsloth / mlx.
        device: cuda / cpu / mps.
        save_dir: Optional. Required when target='disk' — caller must supply
            a containment-checked scratch directory.

    Returns:
        List of error messages. Empty if valid or disabled.
    """
    errors: list[str] = []

    if target is None:
        return errors

    if backend == "unsloth":
        errors.append(
            "Activation offloading is not compatible with the unsloth backend. "
            "Unsloth manages its own memory. Use backend: transformers."
        )
        return errors

    if backend == "mlx":
        errors.append(
            "Activation offloading is not supported on the mlx backend. "
            "Use backend: transformers."
        )
        return errors

    if device != "cuda":
        errors.append(
            "Activation offloading requires CUDA training. "
            f"Current device: {device}."
        )

    if target == "disk" and not save_dir:
        errors.append(
            "activation_offloading='disk' requires a save_dir "
            "(scratch directory for offloaded activation tensors)."
        )

    return errors


@contextlib.contextmanager
def offload_context(
    target: Optional[str], save_dir: Optional[str] = None,
) -> Generator[None, None, None]:
    """Install saved-tensor hooks for activation offloading; remove on exit.

    Args:
        target: 'cpu' → offload to RAM; 'disk' → offload to scratch files;
            None → no-op.
        save_dir: Directory for disk-mode scratch files. Required for 'disk'.

    Yields:
        Nothing — use as a ``with`` block around ``trainer.train()``.

    Note:
        When ``torch`` is not installed the hooks are silently skipped; this
        keeps the context manager safe to enter from CI / CLI --help paths.
    """
    if target is None:
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    created_files: list[str] = []

    if target == "cpu":
        pack_hook, unpack_hook = _make_cpu_hooks(torch)
    elif target == "disk":
        if not save_dir:
            raise ValueError(
                "activation_offloading='disk' requires save_dir"
            )
        pack_hook, unpack_hook = _make_disk_hooks(torch, save_dir, created_files)
    else:
        raise ValueError(
            f"Unknown activation_offloading target: {target!r}. "
            "Expected None, 'cpu', or 'disk'."
        )

    # saved_tensors_hooks is the public API (torch>=1.11). Older torch: no-op.
    if not hasattr(torch.autograd.graph, "saved_tensors_hooks"):
        yield
        return

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            yield
    finally:
        # Best-effort cleanup for disk mode: remove any leftover scratch files
        # (e.g. from a crashed backward pass). Per-file OSErrors are swallowed
        # so cleanup is never itself a crash source.
        if target == "disk":
            import os
            for path in created_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _make_cpu_hooks(torch_module: ModuleType) -> HookPair:
    """Saved-tensor hooks: pack → move to CPU, unpack → move back."""

    def pack(tensor: Any) -> Any:
        if tensor.device.type == "cuda":
            return ("cuda", tensor.device, tensor.detach().cpu())
        return ("keep", None, tensor)

    def unpack(payload: Any) -> Any:
        kind, original_device, stored = payload
        if kind == "cuda":
            return stored.to(original_device, non_blocking=True)
        return stored

    return pack, unpack


def _make_disk_hooks(
    torch_module: ModuleType,
    save_dir: str,
    created_files: list[str],
) -> HookPair:
    """Saved-tensor hooks: pack → save to disk, unpack → reload.

    ``created_files`` accumulates paths for best-effort cleanup at context exit.
    The scratch fd is held open until ``torch.save`` returns to close the
    TOCTOU window between ``mkstemp`` and ``save``.
    """
    import os
    import tempfile

    os.makedirs(save_dir, exist_ok=True)

    def pack(tensor: Any) -> Any:
        if tensor.device.type != "cuda":
            return ("keep", None, None, tensor)
        fd, path = tempfile.mkstemp(dir=save_dir, suffix=".pt")
        created_files.append(path)
        # Keep fd open until torch.save flushes to close the TOCTOU gap
        # between mkstemp and a subsequent open-by-path.
        try:
            with os.fdopen(fd, "wb") as file_obj:
                torch_module.save(tensor.detach().cpu(), file_obj)
        except Exception:
            # Hook must be best-effort; fall back to keeping tensor in VRAM.
            try:
                os.unlink(path)
            except OSError:
                pass
            try:
                created_files.remove(path)
            except ValueError:
                pass
            return ("keep", None, None, tensor)
        return ("disk", tensor.device, path, None)

    def unpack(payload: Any) -> Any:
        kind, original_device, path, stored = payload
        if kind == "disk":
            try:
                loaded = torch_module.load(
                    path, map_location="cpu", weights_only=True,
                )
            except FileNotFoundError:
                # File already gone (e.g. GC'd by crash + cleanup); return a
                # sentinel None so autograd surfaces a clear error. This
                # should not happen in a healthy run.
                raise
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                try:
                    created_files.remove(path)
                except ValueError:
                    pass
            return loaded.to(original_device, non_blocking=True)
        return stored

    return pack, unpack
