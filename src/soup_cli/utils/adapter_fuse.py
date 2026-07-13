"""Shared LoRA -> dense merge (v0.71.33).

Extracted from ``commands/shrink.py`` (v0.71.29) so both ``soup shrink`` (fuse
the distill-heal adapter back into the pruned model) and ``soup draft`` (a
speculative-decoding draft must be loadable standalone as ``assistant_model=``,
so the distilled adapter has to be merged into a dense checkpoint) go through
ONE implementation instead of two copies that can drift.

:func:`merge_adapter_to_dense` is the general form — base model + adapter ->
dense model at an arbitrary destination. :func:`fuse_adapter_into` is the
in-place special case (destination == the base directory) that ``soup shrink``
uses. Heavy imports (torch / transformers / peft) stay inside the functions.
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


def merge_adapter_to_dense(
    *, base_model: str, adapter_dir: str, out_dir: str, trc: bool = False
) -> None:
    """Merge a LoRA adapter into ``base_model`` and write a DENSE model to ``out_dir``.

    ``base_model`` may be a local directory OR a hub id — PEFT only ever writes
    ``adapter_config.json`` + ``adapter_model.safetensors``, so the base weights
    always have to come from somewhere else.

    The merged model is written to a sibling temp dir and only then swapped in
    for ``out_dir``. An in-place ``save_pretrained`` over a just-loaded model
    directory fails on Windows (error 1224 — the source ``.safetensors`` is
    still memory-mapped by the loaded weights), so the temp-dir swap is the
    cross-platform-safe path, and it also makes the destination replacement
    atomic.

    ``out_dir`` is re-validated immediately before the swap: the training
    subprocess that produced ``adapter_dir`` may have run for hours, so the
    containment check done at command entry is stale by now (TOCTOU).
    """
    import gc
    import shutil
    import tempfile

    # Cheap containment check BEFORE the heavy (and, on a core-only install,
    # ImportError-raising) transformers/peft imports, so a bad path reports the
    # actual problem instead of a confusing missing-dependency error.
    enforce_under_cwd_and_no_symlink(out_dir, "fused output dir")

    # Refuse pickle / PyTorch-classic weights in the adapter dir before PEFT
    # torch.load's them — the adapter dir may have been produced (or swapped)
    # by an untrusted process. Shallow scan: PEFT loads adapter_model.* from
    # the TOP LEVEL, while a training output dir also holds the HF Trainer's
    # own checkpoint-N/optimizer.pt pickles, which are not the threat (a
    # recursive scan would make Soup refuse its own trainer's output).
    from soup_cli.utils.strict_safetensors import assert_safe_top_level_weights

    assert_safe_top_level_weights(adapter_dir)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=trc, torch_dtype="auto"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    # The trainer saves the tokenizer alongside the adapter; fall back to the
    # base model's own tokenizer if it didn't.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=trc)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trc)

    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".fuse_", dir=parent)
    try:
        try:
            merged.save_pretrained(staging)
            tokenizer.save_pretrained(staging)
        finally:
            # Drop every reference so Windows releases the out_dir mmap before
            # we remove it; otherwise rmtree(out_dir) also hits error 1224.
            del merged, base, tokenizer
            gc.collect()
            release_cuda()
        # Re-validate the swap target IMMEDIATELY before the destructive
        # rmtree/replace — the model load + merge + save above took minutes, a
        # real window in which a junction/symlink could be planted at out_dir
        # (the entry-time check is now stale). enforce_* refuses symlinks AND
        # Windows reparse points, so rmtree cannot be redirected outside cwd.
        enforce_under_cwd_and_no_symlink(out_dir, "fused output dir")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
    except BaseException:
        # A half-written staging dir (disk full, interrupted save) must not be
        # orphaned next to the model — it would silently accumulate a full
        # model's worth of bytes per failed run.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def fuse_adapter_into(*, base_dir: str, adapter_dir: str, trc: bool = False) -> None:
    """Merge a LoRA adapter into ``base_dir`` in place (``soup shrink``'s case).

    ``base_dir`` is BOTH the base weights and the destination: the healed model
    replaces the pruned one. Thin wrapper over :func:`merge_adapter_to_dense`.
    """
    merge_adapter_to_dense(
        base_model=base_dir, adapter_dir=adapter_dir, out_dir=base_dir, trc=trc
    )
