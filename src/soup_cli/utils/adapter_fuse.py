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
    cross-platform-safe path. A pre-existing ``out_dir`` is renamed aside
    rather than deleted, and only removed once the staged model has taken
    its place; if the final swap itself then fails (disk full, a lock held
    on the staging dir), the renamed-aside copy is put back before the error
    propagates. If that restore also fails, neither artifact is deleted:
    both the previous model (at the renamed-aside path) and the newly merged
    model (at the staging path) are left on disk and named in the raised
    error, so a failure here can never destroy the last complete copy —
    worst case it leaves two, at paths the caller can recover by hand.

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
    backup: str | None = None
    keep_staging = False
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
        # rename/replace — the model load + merge + save above took minutes, a
        # real window in which a junction/symlink could be planted at out_dir
        # (the entry-time check is now stale). enforce_* refuses symlinks AND
        # Windows reparse points, so the rename cannot be redirected outside cwd.
        enforce_under_cwd_and_no_symlink(out_dir, "fused output dir")
        if os.path.isdir(out_dir):
            # Rename the previous model ASIDE instead of deleting it. The
            # actual swap below can still fail (AV/indexer holding the
            # staging dir, disk full), and until it succeeds the old model
            # is the only good copy there is.
            backup = os.path.join(
                parent, f".{os.path.basename(os.path.normpath(out_dir))}.old-{os.getpid()}"
            )
            try:
                os.replace(out_dir, backup)
            except BaseException as backup_exc:
                # Renaming the previous model aside failed (the same AV /
                # indexer lock the swap below guards against), so out_dir was
                # never touched and still holds the old model. Staging is now
                # the only copy of the new merge: keep it and name it.
                keep_staging = True
                raise RuntimeError(
                    f"failed to move the previous model at '{out_dir}' aside "
                    f"({backup_exc!r}); it is unchanged, and the newly merged "
                    f"model is intact at '{staging}' — move it into place "
                    f"manually"
                ) from backup_exc
        try:
            os.replace(staging, out_dir)
        except BaseException as swap_exc:
            # The swap itself failed: put the previous model straight back
            # before anything else runs, so this failure never leaves
            # neither copy in place.
            if backup is not None:
                try:
                    os.replace(backup, out_dir)
                except BaseException as restore_exc:
                    # The restore ALSO failed: out_dir now holds neither
                    # copy. Do not guess which of staging/backup to keep —
                    # keep both on disk and name them in the error, so the
                    # outer handler below must not rmtree(staging) either.
                    keep_staging = True
                    raise RuntimeError(
                        f"failed to swap the merged model into '{out_dir}' "
                        f"({swap_exc!r}), and restoring the previous model "
                        f"from '{backup}' also failed ({restore_exc!r}); "
                        f"the previous model is intact at '{backup}' and the "
                        f"newly merged model is intact at '{staging}' — move "
                        f"one of them into place manually"
                    ) from restore_exc
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        # A half-written staging dir (disk full, interrupted save) must not be
        # orphaned next to the model — it would silently accumulate a full
        # model's worth of bytes per failed run. But if the swap AND the
        # restore both failed above, staging is the only copy of the new
        # model left anywhere: keep it, and let the RuntimeError above name
        # it instead of deleting it here.
        if not keep_staging:
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
