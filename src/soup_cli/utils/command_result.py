"""Bounded result produced by the CLI's command-result v1 preview."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from soup_cli import __version__
from soup_cli.data.formats import VALID_FORMATS
from soup_cli.utils.paths import is_under
from soup_cli.utils.terminal import strip_control

MAX_COMMAND_RESULT_BYTES = 262_144
_LIST_LIMIT = 32
_LIST_FIELDS = ("reasons", "warnings", "artifacts")
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)$"
)
_JSON_POINTER_PATTERN = r"^(?:/(?:[^~/]|~[01])*)+$"

_Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.]*$", max_length=96)]


def _is_relative_reference(path: str) -> bool:
    return not (
        PureWindowsPath(path).drive
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    )


def _single_line_text(value: str) -> str:
    if strip_control(value, single_line=True) != value:
        raise ValueError("text must not contain controls or bidi formatting")
    return value


_Text = Annotated[str, Field(min_length=1, max_length=1024), AfterValidator(_single_line_text)]
_Fraction = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
_NonnegativeFloat = Annotated[FiniteFloat, Field(ge=0.0)]
_PositiveFloat = Annotated[FiniteFloat, Field(gt=0.0)]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_LocationName = Annotated[str, Field(pattern=r"^(?:[A-Za-z_][A-Za-z0-9_]*|<key>)$", max_length=96)]
_PRODUCER_POINTER = re.compile(
    r"^/(?:reasons|warnings|artifacts|reasons/(?:0|[1-9][0-9]*)/details/errors)$"
)
_Phase = Literal[
    "preflight", "validate", "evaluate", "plan", "execute", "launch", "export", "finalize",
    "transport",
]


def _allow_reader_extensions(schema: dict[str, Any]) -> None:
    schema.pop("additionalProperties", None)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra=_allow_reader_extensions,
    )


class _ConfigError(_StrictModel):
    loc: list[_LocationName | NonNegativeInt] = Field(max_length=16)
    type: _Name



class ResultDetails(_StrictModel):
    """Typed details for data validation and training preflight."""

    actual: _Fraction | None = None
    minimum: _Fraction | None = None
    skipped: NonNegativeInt | None = None
    total: NonNegativeInt | None = None
    category: Literal[
        "unreadable", "yaml_parse", "yaml_limit", "wrong_root", "validation",
    ] | None = None
    errors: list[_ConfigError] | None = Field(default=None, max_length=_LIST_LIMIT)
    required_gb: _NonnegativeFloat | None = None
    available_gb: _NonnegativeFloat | None = None

    @model_serializer(mode="wrap")
    def _supplied_values(self, serialize: SerializerFunctionWrapHandler) -> dict[str, Any]:
        explicit_nullable = self.model_fields_set & {"required_gb", "available_gb"}
        return {
            key: value for key, value in serialize(self).items()
            if value is not None or key in explicit_nullable
        }


class ResultReason(_StrictModel):
    code: _Name
    message: _Text
    fix: _Text
    details: ResultDetails | None = None


class ArtifactRef(_StrictModel):
    """Wire reference to an artifact already verified under the reporting root."""

    kind: _Name
    relative_path: _Text
    schema_version: PositiveInt | None = None
    ref: _Text | None = None

    @field_validator("relative_path")
    @classmethod
    def _canonical_relative_path(cls, path: str) -> str:
        if not _is_relative_reference(path):
            raise ValueError("artifact path must be a canonical slash-separated relative path")
        return path

    def resolve_under(self, root: Path) -> Path:
        """Resolve this path for a producer to verify under its reporting root."""
        candidate = Path(os.path.realpath(root / self.relative_path))
        if not is_under(candidate, root):
            raise ValueError("artifact path must resolve under the reporting root")
        return candidate


class DataValidateResult(_StrictModel):
    format: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$", max_length=96)
    total: NonNegativeInt
    valid: NonNegativeInt
    invalid: NonNegativeInt
    valid_fraction: _Fraction | None

    @field_validator("format")
    @classmethod
    def _supported_format(cls, value: str) -> str:
        if value not in VALID_FORMATS:
            raise ValueError("format must be a supported dataset format")
        return value

    @model_validator(mode="after")
    def _consistent_row_counts(self) -> Self:
        if self.valid + self.invalid != self.total:
            raise ValueError("valid and invalid counts must add up to total")
        expected = self.valid / self.total if self.total else None
        if self.valid_fraction != expected:
            raise ValueError("valid_fraction must reflect the row counts, or be null for zero rows")
        return self


class PreflightVram(_StrictModel):
    estimated_peak_gb: _NonnegativeFloat | None = None
    available_gb: _NonnegativeFloat | None = None
    margin_fraction: _Fraction | None = None
    required_gb: _NonnegativeFloat | None = None


class PreflightDisk(_StrictModel):
    estimated_output_gb: _NonnegativeFloat | None = None
    estimated_checkpoints_gb: _NonnegativeFloat | None = None


class PreflightEstimateBasis(_StrictModel):
    vram: Literal["analytical_model", "unavailable"] = "unavailable"
    disk: Literal["checkpoint_projection", "unavailable"] = "unavailable"
    wall_time: Literal["local_timing_profile", "unavailable"] = "unavailable"


class PreflightResolved(_StrictModel):
    base_model: _Text | None = None
    task: _Name | None = None
    backend: _Name | None = None
    device_name: _Text | None = None
    max_steps: PositiveInt | None = None
    batch_size_per_device: PositiveInt | None = None
    effective_batch_size: PositiveInt | None = None
    context_length: PositiveInt | None = None
    adapter_rank: PositiveInt | None = None
    epochs: _PositiveFloat | None = None

    @field_validator("base_model")
    @classmethod
    def _model_reference(cls, value: str | None) -> str | None:
        if value is not None and not _is_relative_reference(value):
            raise ValueError("model reference must be local-relative or a model identifier")
        return value


class PreflightInputDigests(_StrictModel):
    config_sha256: _Sha256 | None = None
    data_sha256: _Sha256 | None = None


class TrainPreflightData(_StrictModel):
    """Measurements and estimates from a reached local training preflight."""

    dry_run: Literal[True]
    training_rows: NonNegativeInt | None = None
    validation_rows: NonNegativeInt | None = None
    vram: PreflightVram = Field(default_factory=PreflightVram)
    disk: PreflightDisk = Field(default_factory=PreflightDisk)
    estimated_wall_time_s: _PositiveFloat | None = None
    estimate_basis: PreflightEstimateBasis = Field(default_factory=PreflightEstimateBasis)
    resolved: PreflightResolved = Field(default_factory=PreflightResolved)
    input_digests: PreflightInputDigests = Field(default_factory=PreflightInputDigests)

    @field_validator("dry_run", mode="before")
    @classmethod
    def _explicit_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("dry_run must be true")
        return value

    @model_validator(mode="after")
    def _consistent_estimates(self) -> Self:
        vram = self.vram
        priced = (vram.estimated_peak_gb, vram.required_gb, vram.margin_fraction)
        if self.estimate_basis.vram == "analytical_model":
            if any(value is None for value in priced):
                raise ValueError("analytical VRAM requires a peak, margin and required capacity")
        elif any(value is not None for value in priced):
            raise ValueError("VRAM estimates require an analytical basis")
        disk = self.disk
        has_disk_estimate = (
            disk.estimated_output_gb is not None or disk.estimated_checkpoints_gb is not None
        )
        if has_disk_estimate != (self.estimate_basis.disk == "checkpoint_projection"):
            raise ValueError("disk estimates require a checkpoint projection basis")
        if (self.estimated_wall_time_s is not None) != (
            self.estimate_basis.wall_time == "local_timing_profile"
        ):
            raise ValueError("wall time estimates require a local timing profile")
        return self


def _cap_fields(
    document: dict[str, Any], names: tuple[str, ...], prefix: str, omissions: dict[str, int],
) -> dict[str, Any]:
    trimmed = document.copy()
    for name in names:
        rows = document.get(name)
        if isinstance(rows, list) and len(rows) > _LIST_LIMIT:
            trimmed[name] = rows[:_LIST_LIMIT]
            omissions[f"{prefix}/{name}"] = len(rows) - _LIST_LIMIT
    return trimmed


class CommandResult(_StrictModel):
    """Command result v1. Producers validate strictly. Reader schemas allow optional additions."""

    schema_id: Literal["soup.command_result"] = Field(
        alias="schema", json_schema_extra={"type": "string"}
    )
    schema_version: Literal[1] = Field(json_schema_extra={"type": "integer"})
    soup_version: str = Field(min_length=1, max_length=96)
    command: _Name
    phase: _Phase
    status: Literal["ok", "blocked", "refused", "failed", "cancelled"]
    exit_code: int | None = None
    run_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)
    started_at: datetime | None = Field(None, json_schema_extra={"pattern": _UTC_TIMESTAMP.pattern})
    duration_s: FiniteFloat | None = Field(None, ge=0.0)
    decision: Literal["valid", "invalid", "pass", "fail", "ship", "dont_ship"] | None = None
    reasons: list[ResultReason] = Field(default_factory=list, max_length=_LIST_LIMIT)
    warnings: list[ResultReason] = Field(default_factory=list, max_length=_LIST_LIMIT)
    data: DataValidateResult | TrainPreflightData | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list, max_length=_LIST_LIMIT)
    truncated: dict[str, PositiveInt] = Field(
        default_factory=dict,
        json_schema_extra={"propertyNames": {"pattern": _JSON_POINTER_PATTERN}},
    )
    child: None = None

    @model_validator(mode="before")
    @classmethod
    def _cap_lists(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        omissions: dict[str, int] = {}
        capped = _cap_fields(value, _LIST_FIELDS, "", omissions)
        reasons = capped.get("reasons")
        if isinstance(reasons, list):
            reasons = reasons.copy()
            for index, reason in enumerate(reasons):
                if isinstance(reason, dict) and isinstance(reason.get("details"), dict):
                    details = _cap_fields(
                        reason["details"], ("errors",), f"/reasons/{index}/details",
                        omissions,
                    )
                    reasons[index] = {**reason, "details": details}
            capped["reasons"] = reasons
        if not omissions:
            return value
        counts = value.get("truncated", {})
        if not isinstance(counts, dict):
            raise ValueError("truncated must contain omission counts")
        counts = counts.copy()
        for pointer, count in omissions.items():
            if pointer in counts and (type(counts[pointer]) is not int or counts[pointer] <= 0):
                raise ValueError("truncated counts must be positive integers")
            counts[pointer] = counts.get(pointer, 0) + count
        capped["truncated"] = counts
        return capped

    @field_validator("truncated")
    @classmethod
    def _known_omission_lists(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(_PRODUCER_POINTER.fullmatch(pointer) is None for pointer in counts):
            raise ValueError("truncated must name a supported result list")
        return counts

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("started_at", mode="before")
    @classmethod
    def _parse_wire_timestamp(cls, value: object, info: ValidationInfo) -> object:
        # The model's list-cap validator exposes decoded JSON to strict
        # fields. Parse this one wire string while keeping Python inputs strict.
        if info.mode != "json" or not isinstance(value, str):
            return value
        if _UTC_TIMESTAMP.fullmatch(value) is None:
            raise ValueError("started_at must be an RFC 3339 UTC timestamp")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("started_at must be a valid RFC 3339 UTC timestamp") from None

    @field_validator("started_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() != timedelta(0):
            raise ValueError("started_at must be an aware UTC time")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _consistent_outcome(self) -> Self:
        if self.status == "ok" and self.reasons:
            raise ValueError("an ok command result cannot have refusal reasons")
        if self.phase == "launch" and self.status == "ok" and self.run_id is None:
            raise ValueError("an ok launch result requires a run_id")
        if isinstance(self.data, DataValidateResult) and self.command != "data.validate":
            raise ValueError("the preview payload belongs to data.validate")
        if isinstance(self.data, TrainPreflightData) and (self.command, self.phase) != (
            "train", "preflight",
        ):
            raise ValueError("the training payload belongs to train preflight")
        return self


def _json_line(document: dict[str, Any]) -> bytes:
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def _bounded_line(result: CommandResult) -> bytes:
    validated = CommandResult.model_validate(result.model_dump(by_alias=True, warnings=False))
    document = validated.model_dump(mode="json", by_alias=True)
    payload = _json_line(document)
    for name, minimum in (("warnings", 0), ("artifacts", 0), ("reasons", 1)):
        entries = document[name]
        pointer = f"/{name}"
        while len(payload) > MAX_COMMAND_RESULT_BYTES and len(entries) > minimum:
            entries.pop()
            counts = document["truncated"]
            counts[pointer] = counts.get(pointer, 0) + 1
            payload = _json_line(document)
    if len(payload) > MAX_COMMAND_RESULT_BYTES:
        raise ValueError("command result exceeds the wire size limit")
    return payload


def encode_command_result(result: CommandResult) -> bytes:
    """Return one bounded UTF-8 JSON document followed by LF."""

    try:
        return _bounded_line(result)
    except (TypeError, ValueError, OverflowError, PydanticSerializationError):
        # Keep only a recognized producer identity and the historical exit code.
        if result.command == "train" and result.phase == "preflight":
            command, phase = "train", "preflight"
        else:
            command, phase = "data.validate", "validate"
        fallback = CommandResult(
            schema="soup.command_result",
            schema_version=1,
            soup_version=__version__,
            command=command,
            phase=phase,
            status="failed",
            exit_code=result.exit_code if type(result.exit_code) is int else None,
            reasons=[
                ResultReason(
                    code="soup.cli.result_serialization_failed",
                    message="The command result could not be encoded.",
                    fix="Check the local diagnostic and retry if appropriate.",
                )
            ],
        )
        return _json_line(fallback.model_dump(mode="json", by_alias=True))
