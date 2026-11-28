"""Versioned public status snapshots for durable training runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self, get_args

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from soup_cli.utils.command_result import (
    _UTC_TIMESTAMP,
    MAX_COMMAND_RESULT_BYTES,
    ResultReason,
    _is_relative_reference,
    _json_line,
    _single_line_text,
    _StrictModel,
    _Text,
)

ActiveRunState = Literal["launching", "running", "finalizing"]
TerminalRunState = Literal["done", "failed", "cancelled", "orphaned"]
RunState = Literal[ActiveRunState, TerminalRunState]
ACTIVE_RUN_STATES = frozenset(get_args(ActiveRunState))
TERMINAL_RUN_STATES = frozenset(get_args(TerminalRunState))
HEARTBEAT_INTERVAL_S = 15
STALE_HEARTBEAT_S = 60


class RunStatus(_StrictModel):
    """Frozen producer snapshot. The reader schema allows optional additions."""

    model_config = ConfigDict(frozen=True)

    schema_id: Literal["soup.run_status"] = Field(
        alias="schema", json_schema_extra={"type": "string"},
    )
    schema_version: Literal[1] = Field(json_schema_extra={"type": "integer"})
    soup_version: Annotated[str, AfterValidator(_single_line_text)] = Field(
        min_length=1, max_length=96,
    )
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)
    command: Literal["train", None] = Field(
        None, json_schema_extra={"type": ["string", "null"]},
    )
    started_by: Literal["cli", "mcp"]
    heartbeat_interval_s: PositiveInt | None = None
    state: RunState
    started_at: datetime = Field(json_schema_extra={"pattern": _UTC_TIMESTAMP.pattern})
    updated_at: datetime = Field(json_schema_extra={"pattern": _UTC_TIMESTAMP.pattern})
    finished_at: datetime | None = Field(
        None, json_schema_extra={"pattern": _UTC_TIMESTAMP.pattern},
    )
    step: NonNegativeInt | None = None
    total_steps: NonNegativeInt | None = None
    last_loss: FiniteFloat | None = None
    eta_s: FiniteFloat | None = Field(None, ge=0.0)
    log_relative_path: _Text | None = Field(
        None, description="Canonical slash-separated log path relative to the run root.",
    )
    exit_code: int | None = None
    terminal_reason: ResultReason | None = None
    truncated: dict[str, PositiveInt] = Field(default_factory=dict, max_length=0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("started_at", "updated_at", "finished_at", mode="before")
    @classmethod
    def _wire_timestamp(cls, value: object, info: ValidationInfo) -> object:
        if info.mode != "json" or value is None:
            return value
        if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
            raise ValueError("timestamps must be RFC 3339 UTC strings")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("timestamps must be valid RFC 3339 UTC times") from None

    @field_validator("started_at", "updated_at", "finished_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() != timedelta(0):
            raise ValueError("timestamps must be aware UTC times")
        return value.astimezone(UTC)

    @field_validator("log_relative_path")
    @classmethod
    def _relative_log(cls, value: str | None) -> str | None:
        if value is not None and (":" in value or not _is_relative_reference(value)):
            raise ValueError("log path must be a canonical slash-separated relative path")
        return value

    @model_validator(mode="after")
    def _consistent_state(self) -> Self:
        if self.updated_at < self.started_at:
            raise ValueError("updated_at must be at or after started_at")
        if self.step is not None and self.total_steps is not None and self.step > self.total_steps:
            raise ValueError("step must not exceed total_steps")
        if self.state in ACTIVE_RUN_STATES:
            if any(
                value is not None
                for value in (self.finished_at, self.exit_code, self.terminal_reason)
            ):
                raise ValueError(
                    "active runs cannot have a finish time, exit code or terminal reason"
                )
            return self
        if self.finished_at is None or not self.started_at <= self.finished_at <= self.updated_at:
            raise ValueError("terminal runs require an ordered finished_at")
        if self.state == "done":
            if self.terminal_reason is not None or self.exit_code not in (None, 0):
                raise ValueError("done runs cannot have a failure reason or nonzero exit code")
        elif self.terminal_reason is None:
            raise ValueError("failed, cancelled and orphaned runs require a terminal reason")
        return self


def _check_wire_size(size: int) -> None:
    if size > MAX_COMMAND_RESULT_BYTES:
        raise ValueError(f"run status exceeds the {MAX_COMMAND_RESULT_BYTES}-byte wire size limit")


def encode_run_status(status: RunStatus) -> bytes:
    """Serialize one validated, bounded UTF-8 snapshot followed by LF."""

    validated = RunStatus.model_validate(status.model_dump(by_alias=True, warnings=False))
    payload = _json_line(validated.model_dump(mode="json", by_alias=True))
    _check_wire_size(len(payload))
    return payload


def decode_run_status(payload: bytes | str) -> RunStatus:
    """Read one snapshot, rejecting oversize data before parsing JSON."""

    if not isinstance(payload, (bytes, str)):
        raise TypeError("run status payload must be bytes or text")
    _check_wire_size(len(payload))
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
        _check_wire_size(len(payload))
    return RunStatus.model_validate_json(payload)
