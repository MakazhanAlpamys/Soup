"""Verify the B1 controller record required for MLX shard admission."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from soup_cli.autodistill._admission_io import load_canonical_artifact
from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    ShardManifest,
    _require_sha256,
    canonical_sha256,
)
from soup_cli.autodistill.mlx_worker import (
    _MAX_CONTROL_BYTES,
    _REQUEST_NAME,
    _RESULT_NAME,
    _WORKER_RECEIPT_NAME,
    MlxTeacherCaptureRequest,
    MlxTeacherCaptureResult,
    MlxTeacherWorkerReceipt,
)


def verify_mlx_capture_record(
    *,
    capture_root: Path,
    plan: AutoDistillPlan,
    manifest: ShardManifest,
    available_manifest_sha256: str,
    expected_request_sha256: str | None,
) -> None:
    """Bind the controller's post-exit record to the trusted dispatch and shard."""
    if expected_request_sha256 is None:
        raise ArtifactCorruptionError("MLX capture expected request sha256 is required")
    try:
        _require_sha256(expected_request_sha256, "expected MLX capture request sha256")
    except ValueError as exc:
        raise ArtifactCorruptionError(str(exc)) from exc

    # The validated manifest ID is a single component. The reader applies the
    # contract's path rules to the full relative path, including Windows aliases.
    transaction_id = manifest.transaction_id
    control_dir = f".workers/{transaction_id}"
    result, _ = load_canonical_artifact(
        capture_root,
        f"{control_dir}/{_RESULT_NAME}",
        MlxTeacherCaptureResult,
        max_bytes=_MAX_CONTROL_BYTES,
    )
    request, request_bytes = load_canonical_artifact(
        capture_root,
        f"{control_dir}/{_REQUEST_NAME}",
        MlxTeacherCaptureRequest,
        max_bytes=_MAX_CONTROL_BYTES,
    )
    if hashlib.sha256(request_bytes).hexdigest() != expected_request_sha256:
        raise ArtifactCorruptionError("MLX capture request sha256 mismatch")
    if request.shard_id != manifest.shard_id or request.transaction_id != transaction_id:
        raise ArtifactCorruptionError(
            "MLX capture request identifies a different shard transaction"
        )
    # The controller records the real publication root when it dispatches the worker.
    if os.path.normcase(request.publication_root) != os.path.normcase(capture_root):
        raise ArtifactCorruptionError("MLX capture request publication root mismatch")
    plan_sha256 = canonical_sha256(plan)
    if canonical_sha256(request.plan) != plan_sha256:
        raise ArtifactCorruptionError("MLX capture request plan sha256 mismatch")

    receipt, receipt_bytes = load_canonical_artifact(
        capture_root,
        f"{control_dir}/{_WORKER_RECEIPT_NAME}",
        MlxTeacherWorkerReceipt,
        max_bytes=_MAX_CONTROL_BYTES,
    )
    if result.worker_receipt_sha256 != hashlib.sha256(receipt_bytes).hexdigest():
        raise ArtifactCorruptionError("MLX post-exit result worker receipt sha256 mismatch")

    expected_receipt = {
        "worker_nonce": request.worker_nonce,
        "plan_sha256": plan_sha256,
        "teacher_fingerprint_sha256": canonical_sha256(plan.teacher),
        "tokenizer_fingerprint_sha256": canonical_sha256(plan.tokenizer),
        "dataset_fingerprint_sha256": canonical_sha256(plan.dataset),
        "available_manifest_sha256": available_manifest_sha256,
        "row_count": manifest.row_count,
        "token_count": manifest.token_count,
        "mlx_lm_version": plan.capture.backend_version,
        "inference_dtype": plan.capture.dtype,
        "quantization": plan.capture.quantization,
    }
    for field, expected in expected_receipt.items():
        if getattr(receipt, field) != expected:
            raise ArtifactCorruptionError(f"MLX capture worker receipt {field} mismatch")

    # On Windows the launcher and worker may have different PIDs.
    if result.available_manifest_sha256 != available_manifest_sha256:
        raise ArtifactCorruptionError("MLX post-exit result available manifest sha256 mismatch")
    if (result.row_count, result.token_count) != (manifest.row_count, manifest.token_count):
        raise ArtifactCorruptionError("MLX post-exit result row or token count mismatch")
