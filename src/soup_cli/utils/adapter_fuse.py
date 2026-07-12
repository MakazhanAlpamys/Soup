"""Shared LoRA -> dense fuse (v0.71.33).

Extracted from ``commands/shrink.py`` (v0.71.29) so both ``soup shrink`` (fuse
the distill-heal adapter back into the pruned model) and ``soup draft`` (a
speculative-decoding draft must be loadable standalone as ``assistant_model=``,
so the distilled adapter has to be merged into a dense checkpoint) go through
ONE implementation instead of two copies that can drift.

Heavy imports (torch / transformers / peft) stay inside the functions.
"""

from __future__ import annotations

import os

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink


def release_cuda() -> None:
    """Return the CUDA caching-allocator pool to the driver (best-effort)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fuse_adapter_into(*, base_dir: str, adapter_dir: str, trc: bool = False) -> None:
    """Merge a LoRA adapter into ``base_dir`` (a dense model), atomically.

    The merged model is written to a sibling temp dir and only then swapped in
    for ``base_dir``. An in-place ``save_pretrained`` over the just-loaded
    ``base_dir`` fails on Windows (error 1224 — the source ``.safetensors`` is
    still memory-mapped by the loaded weights), so the temp-dir swap is the
    cross-platform-safe path.

    ``base_dir`` is re-validated immediately before the swap: the training
    subprocess that produced ``adapter_dir`` may have run for hours, so the
    containment check done at command entry is stale by now (TOCTOU).
    """
    import gc
    import shutil
    import tempfile

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    enforce_under_cwd_and_no_symlink(base_dir, "fuse base dir")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, trust_remote_code=trc, torch_dtype="auto"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=trc)

    parent = os.path.dirname(os.path.abspath(base_dir)) or "."
    staging = tempfile.mkdtemp(prefix=".fuse_", dir=parent)
    try:
        merged.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
    finally:
        # Drop every reference so Windows releases the base_dir mmap before we
        # remove it; otherwise rmtree(base_dir) also hits error 1224.
        del merged, base, tokenizer
        gc.collect()
        release_cuda()
    shutil.rmtree(base_dir)
    os.replace(staging, base_dir)
