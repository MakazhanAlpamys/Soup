# SPDX-License-Identifier: Apache-2.0
"""The one door to the GPU half of ``training.packed_lora``.

The CLI, the config and the CPU test suite import only the pure modules (``policy``, ``budget``,
``support``, ``layout``). Everything that imports ``triton`` (the kernels and the layers built on
them) is reached through :func:`load`, which says why it cannot when Triton is missing.
"""

from __future__ import annotations

import importlib
import importlib.util


class KernelsUnavailableError(RuntimeError):
    """The Triton half cannot be imported here; the message is the reason."""


def triton_available() -> bool:
    """Whether the ``triton`` package can be found (without importing it)."""
    return importlib.util.find_spec("triton") is not None


def load():
    """Import and return the GPU half: a namespace-like module ``runtime`` with ``PackedRun``.

    Raises :class:`KernelsUnavailableError` with the reason when Triton, torch or bitsandbytes
    is missing."""
    if not triton_available():
        raise KernelsUnavailableError(
            "the triton package is not installed (it comes with the Linux CUDA build of torch)"
        )
    try:
        return importlib.import_module("soup_cli.kernels.packed_lora.runtime")
    except ImportError as exc:
        raise KernelsUnavailableError(f"{exc.name or 'a dependency'} could not be imported: {exc}"
                                      ) from exc
