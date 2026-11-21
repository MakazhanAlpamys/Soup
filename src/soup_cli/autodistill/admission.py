"""Read-only admission of one published, same-tokenizer teacher capture."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

from soup_cli.autodistill._admission_capture import verify_mlx_capture_record
from soup_cli.autodistill._admission_io import (
    load_canonical_artifact,
    normalize_capture_root,
    read_bounded_artifact,
)
from soup_cli.autodistill.capture import _validate_teacher_expert_rows
from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    _require_artifact_id,
    _require_sha256,
    canonical_json_bytes,
    canonical_sha256,
    verify_payload_bytes,
)
from soup_cli.autodistill.publisher import _PAYLOAD_NAME, _STATES, _manifest_name

_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_CAPTURE_ROWS = 100_000


@dataclass(frozen=True)
class AdmittedCaptureShard:
    """Verified manifest and rows parsed from the exact verified payload bytes."""

    manifest: ShardManifest
    rows: tuple[CaptureToken, ...]


def _verify_plan(root: Path, expected: AutoDistillPlan) -> str:
    observed, data = load_canonical_artifact(
        root, "plan.json", AutoDistillPlan, max_bytes=_MAX_MANIFEST_BYTES,
    )
    for field in ("teacher", "student", "tokenizer", "dataset"):
        if getattr(observed, field) != getattr(expected, field):
            raise ArtifactCorruptionError(f"plan.json {field} fingerprint does not match")
    observed_sha256 = hashlib.sha256(memoryview(data)[:-1]).hexdigest()
    if observed_sha256 != canonical_sha256(expected):
        raise ArtifactCorruptionError("plan.json does not match the expected plan")
    return observed_sha256


def _verify_chain(
    root: Path,
    directory: str,
    available: ShardManifest,
    expected_plan_sha256: str,
) -> None:
    previous_sha256 = None
    for state in _STATES:
        filename = _manifest_name(state)
        if state == "available":
            manifest = available
        else:
            manifest, _ = load_canonical_artifact(
                root, f"{directory}/{filename}", ShardManifest,
                max_bytes=_MAX_MANIFEST_BYTES,
            )
        if manifest.state != state:
            raise ArtifactCorruptionError(f"{filename} does not record the {state} state")
        if (
            manifest.shard_id != available.shard_id
            or manifest.transaction_id != available.transaction_id
        ):
            raise ArtifactCorruptionError(
                f"{filename} shard or transaction identity does not match"
            )
        if manifest.plan_sha256 != expected_plan_sha256:
            raise ArtifactCorruptionError(f"{filename} plan SHA-256 does not match")
        if manifest.previous_manifest_sha256 != previous_sha256:
            raise ArtifactCorruptionError(f"{filename} manifest hash chain is broken")
        if manifest.payloads != available.payloads:
            raise ArtifactCorruptionError(f"{filename} payload commitments changed")
        previous_sha256 = canonical_sha256(manifest)


def _read_rows(
    payload: bytes, manifest: ShardManifest, plan: AutoDistillPlan,
) -> tuple[CaptureToken, ...]:
    rows: list[CaptureToken] = []
    for position, line in enumerate(io.BytesIO(payload), start=1):
        if position > manifest.row_count:
            raise ArtifactCorruptionError("capture.jsonl row count exceeds the manifest count")
        try:
            row = CaptureToken.model_validate_json(line)
            if line != canonical_json_bytes(row) + b"\n":
                raise ValueError("row is not canonical JSONL")
        except (TypeError, ValueError) as exc:
            raise ArtifactCorruptionError(f"capture.jsonl row {position}: {exc}") from exc
        rows.append(row)
    if len(rows) != manifest.row_count or len(rows) != manifest.token_count:
        raise ArtifactCorruptionError("capture.jsonl row or token count does not match manifest")
    try:
        _validate_teacher_expert_rows(rows, plan)
    except (TypeError, ValueError) as exc:
        raise ArtifactCorruptionError(f"capture.jsonl: {exc}") from exc
    return tuple(rows)


def admit_teacher_expert_shard(
    *,
    capture_root: str | os.PathLike[str],
    expected_plan: AutoDistillPlan,
    shard_id: str,
    expected_artifact_sha256: str,
    expected_request_sha256: str | None = None,
) -> AdmittedCaptureShard:
    """Admit a publisher shard only after checking its complete immutable record.

    The shard ID is the directory locator. The caller pins the actual identity
    with the canonical available manifest hash and the expected plan. An MLX
    caller also pins the exact bytes of the dispatched B1 request (including LF).
    """
    if not isinstance(expected_plan, AutoDistillPlan):
        raise TypeError("expected_plan must be an AutoDistillPlan")
    _require_artifact_id(shard_id, "shard_id")
    _require_sha256(expected_artifact_sha256, "expected_artifact_sha256")
    root = normalize_capture_root(capture_root)
    directory = f"shards/{shard_id}"
    available, available_bytes = load_canonical_artifact(
        root, f"{directory}/manifest.available.json", ShardManifest,
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if available.state != "available" or os.path.lexists(
        root / directory / "manifest.quarantined.json"
    ):
        raise ArtifactCorruptionError("shard publication state is not available for admission")
    if available.shard_id != shard_id:
        raise ArtifactCorruptionError("available manifest shard ID does not match its directory")
    if hashlib.sha256(memoryview(available_bytes)[:-1]).hexdigest() != expected_artifact_sha256:
        raise ArtifactCorruptionError("available manifest SHA-256 does not match expected artifact")
    _verify_chain(root, directory, available, _verify_plan(root, expected_plan))
    if tuple(entry.path for entry in available.payloads) != (_PAYLOAD_NAME,):
        raise ArtifactCorruptionError("payload membership must be exactly capture.jsonl")
    if not 0 < available.row_count <= _MAX_CAPTURE_ROWS:
        raise ArtifactCorruptionError(f"capture row count is outside the {_MAX_CAPTURE_ROWS} limit")
    if available.row_count > expected_plan.capture.planned_token_count:
        raise ArtifactCorruptionError("capture row count exceeds the plan's token count")
    if available.payloads[0].bytes > _MAX_PAYLOAD_BYTES:
        raise ArtifactCorruptionError(f"capture.jsonl exceeds the {_MAX_PAYLOAD_BYTES}-byte limit")
    if expected_plan.capture.backend == "mlx":
        verify_mlx_capture_record(
            capture_root=root,
            plan=expected_plan,
            manifest=available,
            available_manifest_sha256=expected_artifact_sha256,
            expected_request_sha256=expected_request_sha256,
        )
    payload = read_bounded_artifact(
        root, f"{directory}/{_PAYLOAD_NAME}", max_bytes=_MAX_PAYLOAD_BYTES,
    )
    verify_payload_bytes(available, {_PAYLOAD_NAME: payload})
    return AdmittedCaptureShard(available, _read_rows(payload, available, expected_plan))
