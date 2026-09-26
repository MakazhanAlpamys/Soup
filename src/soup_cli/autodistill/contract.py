"""Versioned, model-free AutoDistill artifact contract for issue #580.

Milestone A deliberately stops at artifacts, deterministic arithmetic, and state
machines.  This module does not load a teacher, a student, or a tokenizer and is
independent from :mod:`soup_cli.config.schema`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AUTODISTILL_PLAN_SCHEMA = "soup.autodistill.plan.v1"
CAPTURE_TOKEN_SCHEMA = "soup.autodistill.capture-token.v1"
SHARD_MANIFEST_SCHEMA = "soup.autodistill.shard-manifest.v1"
CONSUMPTION_EVENT_SCHEMA = "soup.autodistill.consumption-event.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][-A-Za-z0-9_.:]{0,127}$")
_PROBABILITY_TOLERANCE = 1e-9

ExampleState = Literal[
    "proposed",
    "probed",
    "captured",
    "verified",
    "admitted",
    "rejected",
    "quarantined",
]
ShardState = Literal["staging", "complete", "verified", "available", "quarantined"]
ConsumptionState = Literal["available", "reserved", "committed"]
ConsumptionView = Literal["teacher_expert", "student_rollout"]
ResumeDecision = Literal[
    "resume_staging",
    "verify_then_publish",
    "publish",
    "reuse",
    "quarantine",
    "refuse",
]


class ArtifactCorruptionError(ValueError):
    """Raised when artifact bytes do not match their committed manifest."""


class _FrozenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_artifact_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be a portable artifact identifier")
    return value


def _require_positive_int(value: object, field: str) -> object:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    return value


def _require_non_negative_int(value: object, field: str) -> object:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    return value


def _require_finite_positive(value: object, field: str) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return value


def _require_safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("artifact path must be a non-empty string without null bytes")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError("artifact path must be relative")
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        raise ValueError("artifact path must not be drive-absolute")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact path must be normalized and may not contain '..'")
    return normalized


def _validate_file_sequence(files: tuple[FileDigest, ...], field: str) -> tuple[FileDigest, ...]:
    paths = [entry.path for entry in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{field} must be unique and sorted by path")
    return files


class FileDigest(_FrozenArtifact):
    """Exact bytes committed under a portable relative path."""

    path: str
    bytes: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return _require_safe_relative_path(value)

    @field_validator("bytes", mode="before")
    @classmethod
    def _bytes_not_bool(cls, value: object) -> object:
        return _require_non_negative_int(value, "bytes")

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _require_sha256(value, "sha256")


class PayloadDigest(FileDigest):
    """One immutable shard payload plus its logical cardinality."""

    rows: int = Field(ge=0)
    tokens: int = Field(ge=0)

    @field_validator("rows", "tokens", mode="before")
    @classmethod
    def _counts_not_bool(cls, value: object, info) -> object:
        return _require_non_negative_int(value, info.field_name)


class ModelFingerprint(_FrozenArtifact):
    """Content-bound model identity; a repository name alone is never sufficient."""

    model_id: str = Field(min_length=1, max_length=512)
    revision: str
    config_sha256: str
    weights: tuple[FileDigest, ...] = Field(min_length=1)

    @field_validator("model_id")
    @classmethod
    def _clean_model_id(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("model_id contains a control character")
        return value

    @field_validator("revision")
    @classmethod
    def _immutable_revision(cls, value: str) -> str:
        if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
            raise ValueError("revision must be an immutable 40-64 character hex id")
        return value

    @field_validator("config_sha256")
    @classmethod
    def _config_digest(cls, value: str) -> str:
        return _require_sha256(value, "config_sha256")

    @field_validator("weights")
    @classmethod
    def _ordered_weights(cls, value: tuple[FileDigest, ...]) -> tuple[FileDigest, ...]:
        return _validate_file_sequence(value, "weights")


class TokenizerFingerprint(_FrozenArtifact):
    """Shared tokenizer and rendering identity for the v1 capture boundary."""

    tokenizer_id: str = Field(min_length=1, max_length=512)
    revision: str
    vocab_size: int = Field(gt=0)
    files: tuple[FileDigest, ...] = Field(min_length=1)
    chat_template_sha256: str
    renderer: str = Field(min_length=1, max_length=256)

    @field_validator("revision")
    @classmethod
    def _immutable_revision(cls, value: str) -> str:
        if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
            raise ValueError("revision must be an immutable 40-64 character hex id")
        return value

    @field_validator("vocab_size", mode="before")
    @classmethod
    def _vocab_not_bool(cls, value: object) -> object:
        return _require_positive_int(value, "vocab_size")

    @field_validator("files")
    @classmethod
    def _ordered_files(cls, value: tuple[FileDigest, ...]) -> tuple[FileDigest, ...]:
        return _validate_file_sequence(value, "tokenizer files")

    @field_validator("chat_template_sha256")
    @classmethod
    def _template_digest(cls, value: str) -> str:
        return _require_sha256(value, "chat_template_sha256")


class DatasetFingerprint(_FrozenArtifact):
    """Ordered source bytes plus the canonicalized training-text digest."""

    normalization: Literal["soup-jsonl-c14n-v1"]
    normalized_sha256: str
    rows: int = Field(ge=0)
    source_files: tuple[FileDigest, ...] = Field(min_length=1)

    @field_validator("normalized_sha256")
    @classmethod
    def _normalized_digest(cls, value: str) -> str:
        return _require_sha256(value, "normalized_sha256")

    @field_validator("rows", mode="before")
    @classmethod
    def _rows_not_bool(cls, value: object) -> object:
        return _require_non_negative_int(value, "rows")

    @field_validator("source_files")
    @classmethod
    def _ordered_files(cls, value: tuple[FileDigest, ...]) -> tuple[FileDigest, ...]:
        return _validate_file_sequence(value, "source_files")


class CaptureSpec(_FrozenArtifact):
    """Teacher capture parameters that affect reusable bytes."""

    planned_token_count: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    max_forced_tokens_per_position: int = Field(ge=0)
    backend: Literal["transformers", "mlx", "vllm"]
    backend_version: str = Field(min_length=1, max_length=128)
    dtype: Literal["float16", "bfloat16", "float32"]
    quantization: str = Field(min_length=1, max_length=128)
    max_sequence_length: int = Field(gt=0)
    truncation: Literal["left", "right", "none"]

    @field_validator(
        "planned_token_count",
        "vocab_size",
        "max_forced_tokens_per_position",
        "max_sequence_length",
        mode="before",
    )
    @classmethod
    def _integers_not_bool(cls, value: object, info) -> object:
        return _require_non_negative_int(value, info.field_name)


class ProbabilityPolicy(_FrozenArtifact):
    """Explicit top-k union forced-token plus residual-tail policy.

    No storage width or ``top_k`` value has a default: choosing either is a
    scientific/operational decision, not a format-level quality claim.
    """

    name: Literal["topk_union_forced_tail.v1"]
    top_k: int = Field(gt=0)
    forced_token_sources: tuple[Literal["target", "student_sample"], ...]
    token_id_bytes: Literal[4, 8]
    log_probability_bytes: Literal[2, 4, 8]
    tail_mass_bytes: Literal[4, 8]
    entropy_bytes: Literal[4, 8]
    temperature: float = Field(gt=0.0)
    renormalize_selected: Literal[False]

    @field_validator("top_k", mode="before")
    @classmethod
    def _top_k_not_bool(cls, value: object) -> object:
        return _require_positive_int(value, "top_k")

    @field_validator(
        "token_id_bytes",
        "log_probability_bytes",
        "tail_mass_bytes",
        "entropy_bytes",
        mode="before",
    )
    @classmethod
    def _widths_not_bool(cls, value: object, info) -> object:
        return _require_positive_int(value, info.field_name)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature_finite(cls, value: object) -> object:
        return _require_finite_positive(value, "temperature")

    @field_validator("forced_token_sources")
    @classmethod
    def _forced_sources_complete(
        cls,
        value: tuple[Literal["target", "student_sample"], ...],
    ) -> tuple[Literal["target", "student_sample"], ...]:
        expected = ("target", "student_sample")
        if value != expected:
            raise ValueError(f"forced_token_sources must be exactly {expected!r}")
        return value


class ConsumptionPolicy(_FrozenArtifact):
    """Replay contract fixed before any cache consumer exists."""

    teacher_expert_replay: Literal["explicit"]
    student_rollout_replay: Literal["forbidden"]
    reservation_recovery: Literal["release_if_checkpoint_absent"]
    commit_requires_checkpoint_sha256: Literal[True]


class ThroughputProfile(_FrozenArtifact):
    """Previously measured end-to-end throughput; never measured during planning."""

    profile_sha256: str
    teacher_fingerprint_sha256: str
    hardware_fingerprint_sha256: str
    backend: Literal["transformers", "mlx", "vllm"]
    backend_version: str = Field(min_length=1, max_length=128)
    dtype: Literal["float16", "bfloat16", "float32"]
    quantization: str = Field(min_length=1, max_length=128)
    sequence_length_min: int = Field(gt=0)
    sequence_length_max: int = Field(gt=0)
    tokens_per_second_min: float = Field(gt=0.0)
    tokens_per_second_max: float = Field(gt=0.0)

    @field_validator(
        "profile_sha256",
        "teacher_fingerprint_sha256",
        "hardware_fingerprint_sha256",
    )
    @classmethod
    def _profile_digests(cls, value: str, info) -> str:
        return _require_sha256(value, info.field_name)

    @field_validator("sequence_length_min", "sequence_length_max", mode="before")
    @classmethod
    def _lengths_not_bool(cls, value: object, info) -> object:
        return _require_positive_int(value, info.field_name)

    @field_validator("tokens_per_second_min", "tokens_per_second_max", mode="before")
    @classmethod
    def _throughput_finite(cls, value: object, info) -> object:
        return _require_finite_positive(value, info.field_name)

    @model_validator(mode="after")
    def _ordered_range(self) -> ThroughputProfile:
        if self.sequence_length_min > self.sequence_length_max:
            raise ValueError("sequence_length_min must not exceed max")
        if self.tokens_per_second_min > self.tokens_per_second_max:
            raise ValueError("tokens_per_second_min must not exceed max")
        return self


class RuntimeEstimate(_FrozenArtifact):
    status: Literal["unknown", "profiled"]
    seconds_min: float | None
    seconds_max: float | None
    profile_sha256: str | None

    @model_validator(mode="after")
    def _consistent_status(self) -> RuntimeEstimate:
        if self.status == "unknown":
            if any(
                value is not None
                for value in (self.seconds_min, self.seconds_max, self.profile_sha256)
            ):
                raise ValueError("unknown runtime must not invent a range or profile")
            return self
        if self.seconds_min is None or self.seconds_max is None or self.profile_sha256 is None:
            raise ValueError("profiled runtime requires a range and profile_sha256")
        _require_finite_positive(self.seconds_min, "seconds_min")
        _require_finite_positive(self.seconds_max, "seconds_max")
        _require_sha256(self.profile_sha256, "profile_sha256")
        if self.seconds_min > self.seconds_max:
            raise ValueError("seconds_min must not exceed seconds_max")
        return self


class PlanEstimate(_FrozenArtifact):
    token_count: int = Field(gt=0)
    dense_payload_bytes: int = Field(gt=0)
    sparse_payload_bytes_upper_bound: int = Field(gt=0)
    container_metadata_included: Literal[False]
    runtime: RuntimeEstimate

    @field_validator(
        "token_count",
        "dense_payload_bytes",
        "sparse_payload_bytes_upper_bound",
        mode="before",
    )
    @classmethod
    def _integers_not_bool(cls, value: object, info) -> object:
        return _require_positive_int(value, info.field_name)


def build_plan_estimate(
    *,
    token_count: int,
    vocab_size: int,
    top_k: int,
    max_forced_tokens_per_position: int,
    token_id_bytes: int,
    log_probability_bytes: int,
    tail_mass_bytes: int,
    entropy_bytes: int,
    throughput_profile: ThroughputProfile | None = None,
) -> PlanEstimate:
    """Return exact payload arithmetic and an optional cached-profile runtime range.

    Container/index/JSON metadata is deliberately excluded and declared as such.
    The sparse estimate is an upper bound because forced ids may already be in
    top-k.  The function performs arithmetic only; it never probes or loads a model.
    """

    positive = {
        "token_count": token_count,
        "vocab_size": vocab_size,
        "top_k": top_k,
        "token_id_bytes": token_id_bytes,
        "log_probability_bytes": log_probability_bytes,
        "tail_mass_bytes": tail_mass_bytes,
        "entropy_bytes": entropy_bytes,
    }
    for field, value in positive.items():
        _require_positive_int(value, field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    _require_non_negative_int(
        max_forced_tokens_per_position,
        "max_forced_tokens_per_position",
    )
    if not isinstance(max_forced_tokens_per_position, int) or max_forced_tokens_per_position < 0:
        raise ValueError("max_forced_tokens_per_position must be a non-negative integer")
    if top_k > vocab_size:
        raise ValueError("top_k must not exceed vocab_size")

    selected_upper_bound = min(vocab_size, top_k + max_forced_tokens_per_position)
    dense_bytes = token_count * vocab_size * log_probability_bytes
    sparse_bytes = token_count * (
        selected_upper_bound * (token_id_bytes + log_probability_bytes)
        + tail_mass_bytes
        + entropy_bytes
    )
    if throughput_profile is None:
        runtime = RuntimeEstimate(
            status="unknown",
            seconds_min=None,
            seconds_max=None,
            profile_sha256=None,
        )
    else:
        if not isinstance(throughput_profile, ThroughputProfile):
            raise TypeError("throughput_profile must be ThroughputProfile or None")
        runtime = RuntimeEstimate(
            status="profiled",
            seconds_min=token_count / throughput_profile.tokens_per_second_max,
            seconds_max=token_count / throughput_profile.tokens_per_second_min,
            profile_sha256=throughput_profile.profile_sha256,
        )
    return PlanEstimate(
        token_count=token_count,
        dense_payload_bytes=dense_bytes,
        sparse_payload_bytes_upper_bound=sparse_bytes,
        container_metadata_included=False,
        runtime=runtime,
    )


class AutoDistillPlan(_FrozenArtifact):
    """Complete v1 plan artifact; validation requires no ML runtime."""

    schema_id: Literal["soup.autodistill.plan.v1"] = Field(alias="schema")
    run_id: str
    capture_boundary: Literal["same_tokenizer"]
    teacher: ModelFingerprint
    student: ModelFingerprint
    tokenizer: TokenizerFingerprint
    dataset: DatasetFingerprint
    capture: CaptureSpec
    probability_policy: ProbabilityPolicy
    consumption_policy: ConsumptionPolicy
    throughput_profile: ThroughputProfile | None
    estimate: PlanEstimate

    @field_validator("run_id")
    @classmethod
    def _valid_run_id(cls, value: str) -> str:
        return _require_artifact_id(value, "run_id")

    @model_validator(mode="after")
    def _consistent_plan(self) -> AutoDistillPlan:
        if self.capture.vocab_size != self.tokenizer.vocab_size:
            raise ValueError("capture vocab_size does not match tokenizer fingerprint")
        if self.probability_policy.top_k > self.capture.vocab_size:
            raise ValueError("top_k must not exceed capture vocab_size")
        if self.throughput_profile is not None:
            profile = self.throughput_profile
            if profile.teacher_fingerprint_sha256 != canonical_sha256(self.teacher):
                raise ValueError("throughput profile belongs to a different teacher")
            for field in ("backend", "backend_version", "dtype", "quantization"):
                if getattr(profile, field) != getattr(self.capture, field):
                    raise ValueError(f"throughput profile {field} does not match capture")
            if not (
                profile.sequence_length_min
                <= self.capture.max_sequence_length
                <= profile.sequence_length_max
            ):
                raise ValueError("capture sequence length is outside throughput profile range")
        expected = build_plan_estimate(
            token_count=self.capture.planned_token_count,
            vocab_size=self.capture.vocab_size,
            top_k=self.probability_policy.top_k,
            max_forced_tokens_per_position=self.capture.max_forced_tokens_per_position,
            token_id_bytes=self.probability_policy.token_id_bytes,
            log_probability_bytes=self.probability_policy.log_probability_bytes,
            tail_mass_bytes=self.probability_policy.tail_mass_bytes,
            entropy_bytes=self.probability_policy.entropy_bytes,
            throughput_profile=self.throughput_profile,
        )
        if self.estimate != expected:
            raise ValueError("estimate does not match the explicit plan inputs")
        return self


class CaptureToken(_FrozenArtifact):
    """One teacher probability row under the v1 missing-probability policy."""

    schema_id: Literal["soup.autodistill.capture-token.v1"] = Field(alias="schema")
    example_id: str
    trajectory_kind: ConsumptionView
    position: int = Field(ge=0)
    vocab_size: int = Field(gt=0)
    context_token_ids: tuple[int, ...]
    target_token_id: int | None
    student_sampled_token_id: int | None
    top_k_token_ids: tuple[int, ...] = Field(min_length=1)
    forced_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...] = Field(min_length=1)
    teacher_log_probabilities: tuple[float, ...] = Field(min_length=1)
    tail_mass: float = Field(ge=0.0, le=1.0)
    teacher_entropy: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)

    @field_validator("example_id")
    @classmethod
    def _valid_example_id(cls, value: str) -> str:
        return _require_artifact_id(value, "example_id")

    @field_validator("position", "vocab_size", mode="before")
    @classmethod
    def _integers_not_bool(cls, value: object, info) -> object:
        return _require_non_negative_int(value, info.field_name)

    @field_validator("tail_mass", "teacher_entropy", mode="before")
    @classmethod
    def _finite_non_negative(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{info.field_name} must be a finite number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{info.field_name} must be finite and non-negative")
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature_finite(cls, value: object) -> object:
        return _require_finite_positive(value, "temperature")

    @model_validator(mode="after")
    def _probability_contract(self) -> CaptureToken:
        for field, token_ids in (("context_token_ids", self.context_token_ids),):
            if any(isinstance(token_id, bool) for token_id in token_ids):
                raise ValueError(f"{field} must contain integer token ids, not bool")
            if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
                raise ValueError(f"{field} contains an id outside the vocabulary")
        for field, token_ids in (
            ("top_k_token_ids", self.top_k_token_ids),
            ("forced_token_ids", self.forced_token_ids),
            ("selected_token_ids", self.selected_token_ids),
        ):
            if any(isinstance(token_id, bool) for token_id in token_ids):
                raise ValueError(f"{field} must contain integer token ids, not bool")
            if tuple(sorted(token_ids)) != token_ids or len(set(token_ids)) != len(token_ids):
                raise ValueError(f"{field} must be unique and sorted")
            if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
                raise ValueError(f"{field} contains an id outside the vocabulary")
        for field, token_id in (
            ("target_token_id", self.target_token_id),
            ("student_sampled_token_id", self.student_sampled_token_id),
        ):
            if token_id is not None:
                if isinstance(token_id, bool) or token_id < 0 or token_id >= self.vocab_size:
                    raise ValueError(f"{field} must be a token id inside the vocabulary")
        if self.trajectory_kind == "teacher_expert":
            if self.target_token_id is None:
                raise ValueError("teacher_expert rows require target_token_id")
            if self.target_token_id not in self.forced_token_ids:
                raise ValueError("target_token_id must be forced into the selected set")
        if self.trajectory_kind == "student_rollout":
            if self.student_sampled_token_id is None:
                raise ValueError("student_rollout rows require student_sampled_token_id")
            if self.student_sampled_token_id not in self.forced_token_ids:
                raise ValueError("student_sampled_token_id must be forced into the selected set")
        expected_selected = tuple(sorted(set(self.top_k_token_ids) | set(self.forced_token_ids)))
        if self.selected_token_ids != expected_selected:
            raise ValueError("selected_token_ids must equal top-k union forced ids")
        if len(self.teacher_log_probabilities) != len(self.selected_token_ids):
            raise ValueError("one teacher log-probability is required per selected token")
        if any(
            not math.isfinite(value) or value > 0.0
            for value in self.teacher_log_probabilities
        ):
            raise ValueError("teacher_log_probabilities must be finite and <= 0")
        selected_mass = math.fsum(math.exp(value) for value in self.teacher_log_probabilities)
        if len(self.selected_token_ids) < self.vocab_size and self.tail_mass <= 0.0:
            raise ValueError("tail mass must be positive when selected ids omit vocabulary entries")
        if not math.isclose(
            selected_mass + self.tail_mass,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("selected probability mass plus tail_mass must equal one")
        return self


class ShardManifest(_FrozenArtifact):
    """Transactional commitment to one or more immutable shard payloads."""

    schema_id: Literal["soup.autodistill.shard-manifest.v1"] = Field(alias="schema")
    shard_id: str
    transaction_id: str
    state: ShardState
    plan_sha256: str
    previous_manifest_sha256: str | None
    row_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    payloads: tuple[PayloadDigest, ...] = Field(min_length=1)

    @field_validator("shard_id", "transaction_id")
    @classmethod
    def _valid_ids(cls, value: str, info) -> str:
        return _require_artifact_id(value, info.field_name)

    @field_validator("plan_sha256")
    @classmethod
    def _plan_digest(cls, value: str) -> str:
        return _require_sha256(value, "plan_sha256")

    @field_validator("previous_manifest_sha256")
    @classmethod
    def _previous_manifest_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_sha256(value, "previous_manifest_sha256")

    @field_validator("row_count", "token_count", mode="before")
    @classmethod
    def _counts_not_bool(cls, value: object, info) -> object:
        return _require_non_negative_int(value, info.field_name)

    @field_validator("payloads")
    @classmethod
    def _ordered_payloads(cls, value: tuple[PayloadDigest, ...]) -> tuple[PayloadDigest, ...]:
        paths = [entry.path for entry in value]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("payloads must be unique and sorted by path")
        return value

    @model_validator(mode="after")
    def _matching_counts(self) -> ShardManifest:
        if self.state == "staging" and self.previous_manifest_sha256 is not None:
            raise ValueError("staging is the first manifest and has no predecessor")
        if self.state != "staging" and self.previous_manifest_sha256 is None:
            raise ValueError("committed shard states require previous_manifest_sha256")
        if sum(payload.rows for payload in self.payloads) != self.row_count:
            raise ValueError("payload row counts do not match row_count")
        if sum(payload.tokens for payload in self.payloads) != self.token_count:
            raise ValueError("payload token counts do not match token_count")
        return self


def canonical_json_bytes(value: BaseModel | object) -> bytes:
    """Return UTF-8 canonical JSON bytes for hashing and manifest references."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def canonicalize_jsonl_bytes(data: bytes) -> bytes:
    """Canonicalize ordered JSONL rows for the ``soup-jsonl-c14n-v1`` digest.

    UTF-8 BOM and newline style are normalized. Row order and Unicode codepoints
    are preserved. Blank lines, duplicate object keys, and non-object rows fail
    closed so two parsers cannot silently fingerprint different logical data.
    """

    if not isinstance(data, bytes):
        raise TypeError("JSONL source must be bytes")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("JSONL source must be valid UTF-8") from exc

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    canonical_rows: list[bytes] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {index}")
        try:
            row = json.loads(line, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL row at line {index}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at line {index} must be an object")
        canonical_rows.append(canonical_json_bytes(row))
    return b"".join(row + b"\n" for row in canonical_rows)


def canonical_sha256(value: BaseModel | object) -> str:
    """Hash canonical JSON bytes with SHA-256."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def coarse_tail_forward_kl(
    *,
    teacher_log_probabilities: Sequence[float],
    student_log_probabilities: Sequence[float],
    teacher_tail_mass: float,
    student_tail_mass: float,
) -> float:
    """Forward KL on selected ids plus one coarse residual-tail bucket.

    This is exactly dense forward KL when selected ids cover the vocabulary.
    For ``k < vocab`` it is explicitly a coarse-grained approximation, not a
    reconstruction of the token-level distribution inside the tail.
    """

    if len(teacher_log_probabilities) != len(student_log_probabilities):
        raise ValueError("teacher and student selected distributions must align")
    if not teacher_log_probabilities:
        raise ValueError("at least one selected probability is required")
    for field, values in (
        ("teacher_log_probabilities", teacher_log_probabilities),
        ("student_log_probabilities", student_log_probabilities),
    ):
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value > 0.0
            for value in values
        ):
            raise ValueError(f"{field} must contain finite log-probabilities <= 0")
    for field, tail in (
        ("teacher_tail_mass", teacher_tail_mass),
        ("student_tail_mass", student_tail_mass),
    ):
        if isinstance(tail, bool) or not isinstance(tail, (int, float)):
            raise TypeError(f"{field} must be a finite probability")
        if not math.isfinite(float(tail)) or not 0.0 <= float(tail) <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")

    teacher_selected_mass = math.fsum(math.exp(value) for value in teacher_log_probabilities)
    student_selected_mass = math.fsum(math.exp(value) for value in student_log_probabilities)
    if not math.isclose(
        teacher_selected_mass + teacher_tail_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("teacher selected mass plus tail must equal one")
    if not math.isclose(
        student_selected_mass + student_tail_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("student selected mass plus tail must equal one")

    divergence = math.fsum(
        math.exp(teacher_log) * (teacher_log - student_log)
        for teacher_log, student_log in zip(
            teacher_log_probabilities,
            student_log_probabilities,
        )
    )
    if teacher_tail_mass > 0.0:
        if student_tail_mass == 0.0:
            return math.inf
        divergence += teacher_tail_mass * math.log(teacher_tail_mass / student_tail_mass)
    return divergence


def verify_payload_bytes(manifest: ShardManifest, payloads: Mapping[str, bytes]) -> None:
    """Verify exact payload membership, byte counts, and SHA-256 digests."""

    if not isinstance(manifest, ShardManifest):
        raise TypeError("manifest must be ShardManifest")
    if not isinstance(payloads, Mapping):
        raise TypeError("payloads must be a mapping of relative path to bytes")
    expected = {entry.path: entry for entry in manifest.payloads}
    if set(payloads) != set(expected):
        raise ArtifactCorruptionError("payload membership does not match manifest")
    for path, entry in expected.items():
        data = payloads[path]
        if not isinstance(data, bytes):
            raise TypeError(f"payload {path!r} must be bytes")
        if len(data) != entry.bytes:
            raise ArtifactCorruptionError(f"payload {path!r} byte count mismatch")
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise ArtifactCorruptionError(f"payload {path!r} sha256 mismatch")


_EXAMPLE_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"probed", "quarantined"}),
    "probed": frozenset({"captured", "rejected", "quarantined"}),
    "captured": frozenset({"verified", "quarantined"}),
    "verified": frozenset({"admitted", "rejected", "quarantined"}),
    "admitted": frozenset(),
    "rejected": frozenset(),
    "quarantined": frozenset(),
}
_SHARD_TRANSITIONS: dict[str, frozenset[str]] = {
    "staging": frozenset({"complete", "quarantined"}),
    "complete": frozenset({"verified", "quarantined"}),
    "verified": frozenset({"available", "quarantined"}),
    "available": frozenset({"quarantined"}),
    "quarantined": frozenset(),
}


def _ensure_transition(
    machine: str,
    transitions: Mapping[str, frozenset[str]],
    current: str,
    target: str,
) -> str:
    if current not in transitions or target not in transitions:
        raise ValueError(f"unknown {machine} state")
    if target not in transitions[current]:
        raise ValueError(f"invalid {machine} transition: {current} -> {target}")
    return target


def ensure_example_transition(current: ExampleState, target: ExampleState) -> ExampleState:
    """Validate one immutable example-ledger transition."""

    return _ensure_transition("example", _EXAMPLE_TRANSITIONS, current, target)  # type: ignore[return-value]


def ensure_shard_transition(current: ShardState, target: ShardState) -> ShardState:
    """Validate one transactional shard transition."""

    return _ensure_transition("shard", _SHARD_TRANSITIONS, current, target)  # type: ignore[return-value]


def ensure_consumption_transition(
    *,
    view: ConsumptionView,
    current: ConsumptionState,
    target: ConsumptionState,
    checkpoint_sha256: str | None = None,
    replay_of: str | None = None,
) -> ConsumptionState:
    """Validate reserve/commit/release semantics for immutable source artifacts."""

    if view not in {"teacher_expert", "student_rollout"}:
        raise ValueError("unknown consumption view")
    if current not in {"available", "reserved", "committed"}:
        raise ValueError("unknown current consumption state")
    if target not in {"available", "reserved", "committed"}:
        raise ValueError("unknown target consumption state")

    if current == "available" and target == "reserved":
        if checkpoint_sha256 is not None:
            raise ValueError("reservation must not already claim a checkpoint")
        if replay_of is not None:
            raise ValueError("initial reservation must not set replay_of")
        return target
    if current == "reserved" and target == "available":
        if checkpoint_sha256 is not None or replay_of is not None:
            raise ValueError("released reservations must not claim a checkpoint or replay")
        return target
    if current == "reserved" and target == "committed":
        if checkpoint_sha256 is None:
            raise ValueError("commit requires checkpoint_sha256")
        _require_sha256(checkpoint_sha256, "checkpoint_sha256")
        if replay_of is not None:
            raise ValueError("replay_of belongs on the replay reservation event")
        return target
    if current == "committed" and target == "reserved":
        if view == "student_rollout":
            raise ValueError("student_rollout replay is forbidden")
        if replay_of is None:
            raise ValueError("teacher_expert replay requires replay_of")
        _require_sha256(replay_of, "replay_of")
        if checkpoint_sha256 is not None:
            raise ValueError("reservation must not already claim a checkpoint")
        return target
    raise ValueError(f"invalid consumption transition: {current} -> {target}")


class ConsumptionEvent(_FrozenArtifact):
    """One append-only reservation, release, commit, or explicit replay event."""

    schema_id: Literal["soup.autodistill.consumption-event.v1"] = Field(alias="schema")
    event_id: str
    sequence: int = Field(ge=0)
    artifact_sha256: str
    view: ConsumptionView
    from_state: ConsumptionState = Field(alias="from")
    to_state: ConsumptionState = Field(alias="to")
    run_id: str
    reservation_id: str
    checkpoint_sha256: str | None
    replay_of: str | None

    @field_validator("event_id", "run_id", "reservation_id")
    @classmethod
    def _valid_ids(cls, value: str, info) -> str:
        return _require_artifact_id(value, info.field_name)

    @field_validator("sequence", mode="before")
    @classmethod
    def _sequence_not_bool(cls, value: object) -> object:
        return _require_non_negative_int(value, "sequence")

    @field_validator("artifact_sha256")
    @classmethod
    def _artifact_digest(cls, value: str) -> str:
        return _require_sha256(value, "artifact_sha256")

    @model_validator(mode="after")
    def _valid_transition(self) -> ConsumptionEvent:
        ensure_consumption_transition(
            view=self.view,
            current=self.from_state,
            target=self.to_state,
            checkpoint_sha256=self.checkpoint_sha256,
            replay_of=self.replay_of,
        )
        return self


def validate_consumption_ledger(events: Sequence[ConsumptionEvent]) -> ConsumptionState:
    """Validate a complete append-only ledger and return its final state."""

    if not events:
        return "available"
    if any(not isinstance(event, ConsumptionEvent) for event in events):
        raise TypeError("ledger entries must be ConsumptionEvent instances")
    first = events[0]
    if first.sequence != 0 or first.from_state != "available":
        raise ValueError("consumption ledger must start at sequence 0 from available")
    artifact_sha256 = first.artifact_sha256
    view = first.view
    for index, event in enumerate(events):
        if event.sequence != index:
            raise ValueError("consumption ledger sequence must be contiguous")
        if event.artifact_sha256 != artifact_sha256 or event.view != view:
            raise ValueError("consumption ledger must not mix artifacts or views")
        if index > 0:
            previous = events[index - 1]
            if event.from_state != previous.to_state:
                raise ValueError("consumption ledger state chain is broken")
            if event.from_state == "reserved" and event.reservation_id != previous.reservation_id:
                raise ValueError("release/commit must match the active reservation")
            if event.from_state == "committed" and event.reservation_id == previous.reservation_id:
                raise ValueError("an explicit replay requires a fresh reservation_id")
            if event.from_state == "committed" and event.replay_of != canonical_sha256(previous):
                raise ValueError("replay_of must identify the prior committed event")
    return events[-1].to_state


def decide_resume(
    *,
    state: ShardState,
    payloads_valid: bool,
    fingerprints_match: bool,
) -> ResumeDecision:
    """Return the only allowed next action after an interrupted capture."""

    if state not in _SHARD_TRANSITIONS:
        raise ValueError("unknown shard state")
    if not isinstance(payloads_valid, bool) or not isinstance(fingerprints_match, bool):
        raise TypeError("resume evidence must be boolean")
    if state == "quarantined":
        return "refuse"
    if not payloads_valid or not fingerprints_match:
        return "quarantine"
    decisions: dict[str, ResumeDecision] = {
        "staging": "resume_staging",
        "complete": "verify_then_publish",
        "verified": "publish",
        "available": "reuse",
    }
    return decisions[state]
