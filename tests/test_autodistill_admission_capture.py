"""Admission of a shard produced by the B1 controller and its actual child process."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from soup_cli.autodistill.admission import admit_teacher_expert_shard
from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    canonical_json_bytes,
)
from soup_cli.autodistill.mlx_worker import (
    _MAX_CONTROL_BYTES,
    MlxTeacherCaptureResult,
    run_mlx_teacher_capture_process,
)
from tests.test_issue580_autodistill_mlx_worker import (
    _local_plan,
    _runtime_environment,
    _write_fake_mlx_runtime,
)

_SHARD_ID = "admission-shard"
_TRANSACTION_ID = "admission-transaction"


@dataclass(frozen=True)
class PublishedMlxCapture:
    root: Path
    plan: AutoDistillPlan
    result: MlxTeacherCaptureResult
    request_sha256: str

    @property
    def controls(self) -> Path:
        return self.root / ".workers" / _TRANSACTION_ID

    def admit(self, request_sha256: str | None):
        return admit_teacher_expert_shard(
            capture_root=self.root,
            expected_plan=self.plan,
            shard_id=_SHARD_ID,
            expected_artifact_sha256=self.result.available_manifest_sha256,
            expected_request_sha256=request_sha256,
        )


@pytest.fixture
def published_mlx_capture(monkeypatch, tmp_path) -> PublishedMlxCapture:
    plan, teacher_root, tokenizer_root, dataset_root = _local_plan(tmp_path)
    _runtime_environment(monkeypatch, tmp_path, _write_fake_mlx_runtime(tmp_path))
    capture_root = tmp_path / "capture"
    result = run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=teacher_root,
        tokenizer_root=tokenizer_root,
        dataset_root=dataset_root,
        publication_root=capture_root,
        shard_id=_SHARD_ID,
        transaction_id=_TRANSACTION_ID,
        timeout_seconds=30,
    )
    request = capture_root / ".workers" / _TRANSACTION_ID / "request.json"
    return PublishedMlxCapture(
        capture_root, plan, result, hashlib.sha256(request.read_bytes()).hexdigest()
    )


def _write_control(path: Path, data: dict) -> str:
    raw = canonical_json_bytes(data) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _mutate_control(path: Path, **changes: object) -> str:
    return _write_control(path, json.loads(path.read_bytes()) | changes)


def test_admission_accepts_controller_record_without_loading_models(published_mlx_capture):
    capture = published_mlx_capture
    recorded = {path: path.read_bytes() for path in capture.root.rglob("*") if path.is_file()}
    admitted = capture.admit(capture.request_sha256)

    assert [row.position for row in admitted.rows] == [0, 1, 2]
    assert admitted.manifest.shard_id == _SHARD_ID
    assert recorded == {
        path: path.read_bytes() for path in capture.root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("control", ["result.json", "request.json", "worker-receipt.json"])
def test_admission_refuses_missing_mlx_control_record(published_mlx_capture, control):
    capture = published_mlx_capture
    (capture.controls / control).unlink()

    with pytest.raises(ArtifactCorruptionError, match=control.replace(".", r"\.")):
        capture.admit(capture.request_sha256)


@pytest.mark.parametrize("request_sha256", [None, "not-a-digest", "0" * 64])
def test_admission_requires_exact_trusted_request_sha256(published_mlx_capture, request_sha256):
    capture = published_mlx_capture

    with pytest.raises(ArtifactCorruptionError, match="request sha256"):
        capture.admit(request_sha256)


def test_admission_hashes_exact_controller_request_bytes(published_mlx_capture):
    capture = published_mlx_capture
    _mutate_control(capture.controls / "request.json", example_end=1)

    with pytest.raises(ArtifactCorruptionError, match="request sha256 mismatch"):
        capture.admit(capture.request_sha256)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("shard_id", "other-shard", "different shard transaction"),
        ("transaction_id", "other-transaction", "different shard transaction"),
        ("publication_root", "other-root", "publication root mismatch"),
    ],
)
def test_admission_binds_request_to_publication(published_mlx_capture, field, value, error):
    capture = published_mlx_capture
    changed_request_sha256 = _mutate_control(capture.controls / "request.json", **{field: value})

    with pytest.raises(ArtifactCorruptionError, match=error):
        capture.admit(changed_request_sha256)


def test_admission_binds_request_to_expected_plan(published_mlx_capture):
    capture = published_mlx_capture
    changed_plan = capture.plan.model_dump(mode="json", by_alias=True) | {"run_id": "other-run"}
    changed_request_sha256 = _mutate_control(capture.controls / "request.json", plan=changed_plan)

    with pytest.raises(ArtifactCorruptionError, match="request plan sha256 mismatch"):
        capture.admit(changed_request_sha256)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("worker_receipt_sha256", "0" * 64, "worker receipt sha256 mismatch"),
        ("available_manifest_sha256", "0" * 64, "available manifest sha256 mismatch"),
        ("row_count", 4, "row or token count mismatch"),
        ("token_count", 4, "row or token count mismatch"),
        ("worker_exit_code", 1, "worker_exit_code"),
        ("worker_exit_confirmed", False, "worker_exit_confirmed"),
        ("student_loaded", True, "student_loaded"),
        ("worker_exit_code", False, "not canonical"),
        ("worker_exit_confirmed", 1, "not canonical"),
        ("student_loaded", 0, "not canonical"),
        ("row_count", True, "not canonical"),
    ],
)
def test_admission_checks_post_exit_result(published_mlx_capture, field, value, error):
    capture = published_mlx_capture
    _mutate_control(capture.controls / "result.json", **{field: value})

    with pytest.raises(ArtifactCorruptionError, match=error):
        capture.admit(capture.request_sha256)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_nonce", "0" * 32),
        ("plan_sha256", "0" * 64),
        ("teacher_fingerprint_sha256", "0" * 64),
        ("tokenizer_fingerprint_sha256", "0" * 64),
        ("dataset_fingerprint_sha256", "0" * 64),
        ("available_manifest_sha256", "0" * 64),
        ("row_count", 4),
        ("token_count", 4),
        ("mlx_lm_version", "other-version"),
        ("inference_dtype", "float16"),
        ("quantization", "other-quantization"),
    ],
)
def test_admission_binds_receipt_to_dispatch_and_manifest(published_mlx_capture, field, value):
    capture = published_mlx_capture
    receipt_sha256 = _mutate_control(capture.controls / "worker-receipt.json", **{field: value})
    _mutate_control(capture.controls / "result.json", worker_receipt_sha256=receipt_sha256)

    with pytest.raises(ArtifactCorruptionError, match=f"worker receipt {field} mismatch"):
        capture.admit(capture.request_sha256)


def test_admission_does_not_match_launcher_and_worker_pids(published_mlx_capture):
    capture = published_mlx_capture
    _mutate_control(capture.controls / "result.json", worker_pid=capture.result.worker_pid + 1)

    assert len(capture.admit(capture.request_sha256).rows) == 3


def test_admission_rejects_noncanonical_b1_control_bytes(published_mlx_capture):
    capture = published_mlx_capture
    result_path = capture.controls / "result.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")

    with pytest.raises(ArtifactCorruptionError, match="result.json.*not canonical"):
        capture.admit(capture.request_sha256)


def test_admission_rejects_one_byte_over_b1_control_limit(published_mlx_capture):
    capture = published_mlx_capture
    with (capture.controls / "result.json").open("r+b") as result:
        result.truncate(_MAX_CONTROL_BYTES + 1)

    with pytest.raises(ArtifactCorruptionError, match="result.json.*8388608-byte limit"):
        capture.admit(capture.request_sha256)


@pytest.mark.requires_symlink
def test_admission_refuses_symlinked_b1_transaction_directory(published_mlx_capture):
    capture = published_mlx_capture
    original_directory = capture.controls.rename(capture.controls.with_name("original"))
    capture.controls.symlink_to(original_directory, target_is_directory=True)

    with pytest.raises(ArtifactCorruptionError, match="result.json"):
        capture.admit(capture.request_sha256)
