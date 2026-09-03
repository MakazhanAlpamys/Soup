"""Spawn-isolated MLX teacher capture for AutoDistill Milestone B1.

The controller and its request/receipt schemas are import-light.  MLX and
MLX-LM are imported only inside the child worker after immutable local inputs
have been verified.  The request intentionally has no student model field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from soup_cli.autodistill.capture import build_teacher_expert_capture_token
from soup_cli.autodistill.contract import (
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    canonical_json_bytes,
    canonical_sha256,
)
from soup_cli.autodistill.fingerprints import (
    verified_dataset_bytes,
    verify_teacher_fingerprint,
    verify_tokenizer_file_fingerprint,
    verify_tokenizer_fingerprint,
)
from soup_cli.autodistill.publisher import CaptureShardPublisher

MLX_TEACHER_REQUEST_SCHEMA = "soup.autodistill.mlx-teacher-request.v1"
MLX_TEACHER_WORKER_RECEIPT_SCHEMA = "soup.autodistill.mlx-teacher-worker-receipt.v1"
MLX_TEACHER_RESULT_SCHEMA = "soup.autodistill.mlx-teacher-result.v1"
TOKENIZED_TEACHER_EXAMPLE_SCHEMA = "soup.autodistill.tokenized-teacher-example.v1"
_REQUEST_NAME = "request.json"
_WORKER_RECEIPT_NAME = "worker-receipt.json"
_RESULT_NAME = "result.json"
_ERROR_NAME = "error.json"
_MAX_CONTROL_BYTES = 8 * 1024 * 1024
_MAX_CAPTURE_ROWS = 1_000_000
_ARTIFACT_ID_PATTERN = r"^[A-Za-z0-9][-A-Za-z0-9_.:]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TokenizedTeacherExample(_FrozenModel):
    """Bound JSONL row interpreted by the B1 MLX capture worker."""

    schema_id: Literal["soup.autodistill.tokenized-teacher-example.v1"] = Field(
        alias="schema"
    )
    example_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    prompt_token_ids: tuple[int, ...] = Field(min_length=1)
    target_token_ids: tuple[int, ...] = Field(min_length=1)

    @field_validator("prompt_token_ids", "target_token_ids", mode="before")
    @classmethod
    def _token_ids_are_integers(cls, value: object, info) -> object:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError(f"{info.field_name} must be a token-id sequence")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        ):
            raise ValueError(f"{info.field_name} must contain non-negative integers")
        return value


class MlxTeacherCaptureRequest(_FrozenModel):
    schema_id: Literal["soup.autodistill.mlx-teacher-request.v1"] = Field(alias="schema")
    plan: AutoDistillPlan
    teacher_root: str = Field(min_length=1)
    tokenizer_root: str = Field(min_length=1)
    dataset_root: str = Field(min_length=1)
    publication_root: str = Field(min_length=1)
    shard_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    transaction_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    example_start: int = Field(ge=0)
    example_end: int | None = Field(default=None, gt=0)

    @field_validator("example_start", "example_end", mode="before")
    @classmethod
    def _example_bounds_not_bool(cls, value: object, info) -> object:
        if isinstance(value, bool):
            raise TypeError(f"{info.field_name} must be an integer, not bool")
        return value

    @model_validator(mode="after")
    def _mlx_only(self) -> MlxTeacherCaptureRequest:
        if self.plan.capture.backend != "mlx":
            raise ValueError("MLX teacher worker requires capture.backend=mlx")
        if self.example_end is not None and self.example_end <= self.example_start:
            raise ValueError("example_end must be greater than example_start")
        return self


class MlxTeacherWorkerReceipt(_FrozenModel):
    schema_id: Literal["soup.autodistill.mlx-teacher-worker-receipt.v1"] = Field(
        alias="schema"
    )
    worker_pid: int = Field(gt=0)
    student_loaded: Literal[False]
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    teacher_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    tokenizer_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    available_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    row_count: int = Field(gt=0)
    token_count: int = Field(gt=0)
    mlx_version: str = Field(min_length=1, max_length=128)
    mlx_lm_version: str = Field(min_length=1, max_length=128)
    inference_dtype: Literal["float16", "bfloat16", "float32"]
    floating_parameter_dtypes: tuple[
        Literal["float16", "bfloat16", "float32"], ...
    ] = Field(min_length=1)
    quantization: str = Field(min_length=1, max_length=128)


class MlxTeacherCaptureResult(_FrozenModel):
    schema_id: Literal["soup.autodistill.mlx-teacher-result.v1"] = Field(alias="schema")
    worker_pid: int = Field(gt=0)
    worker_exit_code: Literal[0]
    worker_exit_confirmed: Literal[True]
    student_loaded: Literal[False]
    worker_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    available_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    row_count: int = Field(gt=0)
    token_count: int = Field(gt=0)


def _contained_path(root: Path, name: str) -> Path:
    if not name or name != os.path.basename(name) or "\x00" in name:
        raise ValueError("worker artifact name must be a portable basename")
    root_real = os.path.realpath(root)
    candidate = os.path.abspath(os.path.join(root_real, name))
    candidate_real = os.path.realpath(candidate)
    try:
        contained = os.path.commonpath((root_real, candidate_real)) == root_real
    except ValueError as exc:
        raise ValueError("worker artifact escapes its root") from exc
    if not contained:
        raise ValueError("worker artifact escapes its root")
    path = Path(candidate)
    if path.is_symlink():
        raise ValueError("worker artifact must not be a symlink")
    return path


def _atomic_write(path: Path, payload: bytes, *, overwrite_identical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite_identical and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite worker artifact {path.name!r}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"worker artifact appeared during write: {path.name!r}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_control(path: Path, model_type: type[_FrozenModel]) -> _FrozenModel:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"worker artifact {path.name!r} is unavailable")
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_CONTROL_BYTES:
        raise ValueError(f"worker artifact {path.name!r} has an invalid size")
    payload = path.read_bytes()
    model = model_type.model_validate_json(payload)
    if payload != canonical_json_bytes(model) + b"\n":
        raise ValueError(f"worker artifact {path.name!r} is not canonical")
    return model


def _load_bound_examples(request: MlxTeacherCaptureRequest) -> tuple[TokenizedTeacherExample, ...]:
    payload = verified_dataset_bytes(request.plan, dataset_root=request.dataset_root)
    examples = tuple(
        TokenizedTeacherExample.model_validate(json.loads(line))
        for line in payload.splitlines()
    )
    if not examples:
        raise ValueError("teacher capture dataset must not be empty")
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("teacher capture example_id values must be unique")
    token_count = sum(len(example.target_token_ids) for example in examples)
    if token_count != request.plan.capture.planned_token_count:
        raise ValueError("dataset target-token count does not match the plan")
    end = len(examples) if request.example_end is None else request.example_end
    if request.example_start >= len(examples) or end > len(examples):
        raise ValueError("requested example range is outside the bound dataset")
    selected = examples[request.example_start:end]
    selected_token_count = sum(len(example.target_token_ids) for example in selected)
    if selected_token_count > _MAX_CAPTURE_ROWS:
        raise ValueError("teacher capture exceeds the B1 row safety cap")
    vocab_size = request.plan.capture.vocab_size
    for example in examples:
        all_ids = (*example.prompt_token_ids, *example.target_token_ids)
        if any(token_id >= vocab_size for token_id in all_ids):
            raise ValueError("teacher capture dataset contains an id outside the vocabulary")
    return selected


def _context_for_position(
    example: TokenizedTeacherExample,
    position: int,
    request: MlxTeacherCaptureRequest,
) -> tuple[int, ...]:
    context = example.prompt_token_ids + example.target_token_ids[:position]
    maximum = request.plan.capture.max_sequence_length
    if len(context) <= maximum:
        return context
    truncation = request.plan.capture.truncation
    if truncation == "none":
        raise ValueError("teacher context exceeds max_sequence_length")
    if truncation == "left":
        return context[-maximum:]
    return context[:maximum]


def _logit_row(output: object, *, context_length: int, row_index: int, vocab_size: int):
    logits = getattr(output, "logits", output)
    shape = getattr(logits, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError("MLX teacher must return rank-3 causal logits")
    if shape[0] != 1 or shape[1] != context_length or shape[2] != vocab_size:
        raise ValueError("MLX teacher logits shape does not match the capture input")
    return logits[0, row_index, :]


def _quantization_descriptor(teacher_root: str) -> str:
    config_path = Path(teacher_root) / "config.json"
    config = json.loads(config_path.read_bytes())
    if not isinstance(config, dict):
        raise ValueError("teacher config.json must contain a JSON object")
    active = {
        key: config[key]
        for key in ("quantization", "quantization_config")
        if config.get(key) is not None
    }
    if not active:
        return "none"
    return f"config-sha256:{hashlib.sha256(canonical_json_bytes(active)).hexdigest()}"


def _parameter_dtype(value: object) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    name = str(dtype).removeprefix("mlx.core.")
    if name in {"float16", "bfloat16", "float32"}:
        return name
    return None


def _floating_parameter_dtypes(parameters: object) -> set[str]:
    if isinstance(parameters, dict):
        values = parameters.values()
    elif isinstance(parameters, (list, tuple)):
        values = parameters
    else:
        dtype = _parameter_dtype(parameters)
        return {dtype} if dtype is not None else set()
    observed: set[str] = set()
    for value in values:
        observed.update(_floating_parameter_dtypes(value))
    return observed


def _verify_declared_quantization(request: MlxTeacherCaptureRequest) -> str:
    quantization = _quantization_descriptor(request.teacher_root)
    if quantization != request.plan.capture.quantization:
        raise ValueError("teacher config quantization does not match capture.quantization")
    return quantization


def _verify_capture_runtime_dtype(
    request: MlxTeacherCaptureRequest,
    model: object,
    *,
    quantization: str,
) -> tuple[str, tuple[str, ...]]:
    parameters_method = getattr(model, "parameters", None)
    if not callable(parameters_method):
        raise ValueError("MLX teacher does not expose parameters for dtype verification")
    dtypes = _floating_parameter_dtypes(parameters_method())
    expected = request.plan.capture.dtype
    allowed = {expected} if quantization == "none" else {expected, "float32"}
    if expected not in dtypes or not dtypes <= allowed:
        rendered = ", ".join(sorted(dtypes)) or "none"
        raise ValueError(
            f"loaded teacher floating parameter dtypes ({rendered}) do not match {expected}"
        )
    return expected, tuple(sorted(dtypes))


def _capture_examples_with_mlx(
    request: MlxTeacherCaptureRequest,
    examples: tuple[TokenizedTeacherExample, ...],
) -> tuple[tuple[CaptureToken, ...], str, str, str, tuple[str, ...], str]:
    import mlx
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.utils import load_tokenizer

    mlx_version = _package_version("mlx", mlx)
    mlx_lm_version = _package_version("mlx-lm", mlx_lm)
    if mlx_lm_version != request.plan.capture.backend_version:
        raise ValueError("installed MLX-LM version does not match capture.backend_version")
    quantization = _verify_declared_quantization(request)
    model, teacher_tokenizer = load(request.teacher_root)
    inference_dtype, floating_parameter_dtypes = _verify_capture_runtime_dtype(
        request,
        model,
        quantization=quantization,
    )
    tokenizer = load_tokenizer(request.tokenizer_root)
    verify_tokenizer_fingerprint(
        request.plan,
        tokenizer_root=request.tokenizer_root,
        chat_template=getattr(tokenizer, "chat_template", "") or "",
        renderer=f"mlx-lm@{mlx_lm_version}",
    )
    captures: list[CaptureToken] = []
    try:
        for example in examples:
            final_context = example.prompt_token_ids + example.target_token_ids[:-1]
            can_use_one_forward = len(final_context) <= request.plan.capture.max_sequence_length
            if can_use_one_forward:
                inputs = mx.array([final_context])
                output = model(inputs)
                for position, target in enumerate(example.target_token_ids):
                    context = _context_for_position(example, position, request)
                    row = _logit_row(
                        output,
                        context_length=len(final_context),
                        row_index=len(example.prompt_token_ids) - 1 + position,
                        vocab_size=request.plan.capture.vocab_size,
                    )
                    row = mx.astype(row, mx.float32)
                    mx.eval(row)
                    captures.append(
                        build_teacher_expert_capture_token(
                            example_id=example.example_id,
                            position=position,
                            context_token_ids=context,
                            target_token_id=target,
                            teacher_logits=row.tolist(),
                            vocab_size=request.plan.capture.vocab_size,
                            probability_policy=request.plan.probability_policy,
                        )
                    )
            else:
                for position, target in enumerate(example.target_token_ids):
                    context = _context_for_position(example, position, request)
                    inputs = mx.array([context])
                    output = model(inputs)
                    row = _logit_row(
                        output,
                        context_length=len(context),
                        row_index=-1,
                        vocab_size=request.plan.capture.vocab_size,
                    )
                    row = mx.astype(row, mx.float32)
                    mx.eval(row)
                    captures.append(
                        build_teacher_expert_capture_token(
                            example_id=example.example_id,
                            position=position,
                            context_token_ids=context,
                            target_token_id=target,
                            teacher_logits=row.tolist(),
                            vocab_size=request.plan.capture.vocab_size,
                            probability_policy=request.plan.probability_policy,
                        )
                    )
    finally:
        del model
        del tokenizer
        del teacher_tokenizer
        mx.clear_cache()
    return (
        tuple(captures),
        mlx_version,
        mlx_lm_version,
        inference_dtype,
        floating_parameter_dtypes,
        quantization,
    )


def _package_version(distribution: str, module: object) -> str:
    module_version = getattr(module, "__version__", None)
    if isinstance(module_version, str) and module_version:
        return module_version
    try:
        return version(distribution)
    except PackageNotFoundError:
        raise ValueError(f"cannot determine {distribution} version") from None


def _run_worker(request_path: Path) -> None:
    request = _read_control(request_path, MlxTeacherCaptureRequest)
    assert isinstance(request, MlxTeacherCaptureRequest)
    worker_root = request_path.parent
    try:
        verify_teacher_fingerprint(request.plan, teacher_root=request.teacher_root)
        verify_tokenizer_file_fingerprint(
            request.plan,
            tokenizer_root=request.tokenizer_root,
        )
        examples = _load_bound_examples(request)
        (
            captures,
            mlx_version,
            mlx_lm_version,
            inference_dtype,
            floating_parameter_dtypes,
            quantization,
        ) = _capture_examples_with_mlx(request, examples)
        manifest = CaptureShardPublisher(
            root=request.publication_root,
            plan=request.plan,
            shard_id=request.shard_id,
            transaction_id=request.transaction_id,
        ).publish(captures)
        receipt = MlxTeacherWorkerReceipt(
            schema=MLX_TEACHER_WORKER_RECEIPT_SCHEMA,
            worker_pid=os.getpid(),
            student_loaded=False,
            plan_sha256=canonical_sha256(request.plan),
            teacher_fingerprint_sha256=canonical_sha256(request.plan.teacher),
            tokenizer_fingerprint_sha256=canonical_sha256(request.plan.tokenizer),
            dataset_fingerprint_sha256=canonical_sha256(request.plan.dataset),
            available_manifest_sha256=canonical_sha256(manifest),
            row_count=manifest.row_count,
            token_count=manifest.token_count,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            inference_dtype=inference_dtype,
            floating_parameter_dtypes=floating_parameter_dtypes,
            quantization=quantization,
        )
        _atomic_write(
            _contained_path(worker_root, _WORKER_RECEIPT_NAME),
            canonical_json_bytes(receipt) + b"\n",
        )
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "message": str(exc).replace("\x00", "")[:4096],
        }
        try:
            _atomic_write(
                _contained_path(worker_root, _ERROR_NAME),
                canonical_json_bytes(error) + b"\n",
            )
        except Exception:
            pass
        raise


def run_mlx_teacher_capture_process(
    *,
    plan: AutoDistillPlan,
    teacher_root: str | os.PathLike[str],
    tokenizer_root: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
    shard_id: str,
    transaction_id: str,
    example_start: int = 0,
    example_end: int | None = None,
    python_executable: str | os.PathLike[str] = sys.executable,
    timeout_seconds: float = 3600.0,
) -> MlxTeacherCaptureResult:
    """Run capture in a fresh child and commit the result only after child exit."""

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a finite positive number")
    if not 0.0 < float(timeout_seconds) <= 86_400.0:
        raise ValueError("timeout_seconds must be in (0, 86400]")
    publication = Path(os.path.realpath(publication_root))
    request = MlxTeacherCaptureRequest(
        schema=MLX_TEACHER_REQUEST_SCHEMA,
        plan=plan,
        teacher_root=os.path.realpath(teacher_root),
        tokenizer_root=os.path.realpath(tokenizer_root),
        dataset_root=os.path.realpath(dataset_root),
        publication_root=os.path.realpath(publication),
        shard_id=shard_id,
        transaction_id=transaction_id,
        example_start=example_start,
        example_end=example_end,
    )
    worker_root = publication / ".workers" / transaction_id
    if worker_root.exists():
        raise FileExistsError("worker transaction already exists")
    worker_root.mkdir(parents=True)
    request_path = _contained_path(worker_root, _REQUEST_NAME)
    _atomic_write(request_path, canonical_json_bytes(request) + b"\n")
    command = [os.fspath(python_executable), "-m", __name__, "--request", os.fspath(request_path)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        process.wait(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise TimeoutError("MLX teacher worker exceeded timeout and was killed") from exc
    if process.returncode != 0:
        error_path = _contained_path(worker_root, _ERROR_NAME)
        detail = ""
        if error_path.is_file() and error_path.stat().st_size <= _MAX_CONTROL_BYTES:
            detail = error_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            f"MLX teacher worker exited with {process.returncode}: {detail[:4096]}"
        )
    receipt = _read_control(
        _contained_path(worker_root, _WORKER_RECEIPT_NAME),
        MlxTeacherWorkerReceipt,
    )
    assert isinstance(receipt, MlxTeacherWorkerReceipt)
    if receipt.worker_pid != process.pid:
        raise ValueError("worker receipt PID does not match the exited child")
    manifest_path = publication / "shards" / shard_id / "manifest.available.json"
    manifest = ShardManifest.model_validate_json(manifest_path.read_bytes())
    if canonical_sha256(manifest) != receipt.available_manifest_sha256:
        raise ValueError("worker receipt does not match the available shard")
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    result = MlxTeacherCaptureResult(
        schema=MLX_TEACHER_RESULT_SCHEMA,
        worker_pid=process.pid,
        worker_exit_code=0,
        worker_exit_confirmed=True,
        student_loaded=False,
        worker_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        available_manifest_sha256=receipt.available_manifest_sha256,
        row_count=receipt.row_count,
        token_count=receipt.token_count,
    )
    _atomic_write(
        _contained_path(worker_root, _RESULT_NAME),
        canonical_json_bytes(result) + b"\n",
    )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    _run_worker(Path(arguments.request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
