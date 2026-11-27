"""Wire contract for durable training status receipts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from importlib import resources

import pytest
from pydantic import ValidationError

from soup_cli.utils import run_status
from soup_cli.utils.command_result import MAX_COMMAND_RESULT_BYTES, ResultReason
from soup_cli.utils.run_status import (
    ACTIVE_RUN_STATES,
    TERMINAL_RUN_STATES,
    RunStatus,
    decode_run_status,
    encode_run_status,
)

_NOW = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(seconds=15)
_REASON = ResultReason(
    code="soup.train.failed", message="The training run failed.", fix="Check the diagnostic.",
)
_MINIMAL = {
    "schema": "soup.run_status",
    "schema_version": 1,
    "soup_version": "0.78.1",
    "run_id": "run_01",
    "started_by": "cli",
    "state": "running",
    "started_at": _NOW,
    "updated_at": _LATER,
}


def test_active_wire_includes_null_defaults() -> None:
    status = RunStatus(**_MINIMAL)
    payload = encode_run_status(status)
    assert payload.count(b"\n") == 1 and payload.endswith(b"\n")
    assert json.loads(payload) == {
        **_MINIMAL,
        "started_at": "2026-11-24T09:00:00Z",
        "updated_at": "2026-11-24T09:00:15Z",
        "command": None,
        "heartbeat_interval_s": None,
        "finished_at": None,
        "step": None,
        "total_steps": None,
        "last_loss": None,
        "eta_s": None,
        "log_relative_path": None,
        "exit_code": None,
        "terminal_reason": None,
        "truncated": {},
    }
    assert decode_run_status(payload) == decode_run_status(payload.decode()) == status


def test_state_groups_cover_only_the_seven_public_states() -> None:
    assert ACTIVE_RUN_STATES == {"launching", "running", "finalizing"}
    assert TERMINAL_RUN_STATES == {"done", "failed", "cancelled", "orphaned"}


@pytest.mark.parametrize(
    ("state", "exit_code", "reason"),
    [
        ("launching", None, None),
        ("running", None, None),
        ("finalizing", None, None),
        ("done", 0, None),
        ("done", None, None),
        ("failed", 1, _REASON),
        ("cancelled", -2, _REASON),
        ("orphaned", None, _REASON),
    ],
)
def test_active_and_terminal_status_roundtrip(state, exit_code, reason) -> None:
    status = RunStatus(**{
        **_MINIMAL,
        "state": state,
        "started_by": "mcp",
        "command": "train",
        "heartbeat_interval_s": 15,
        "finished_at": _LATER if state in TERMINAL_RUN_STATES else None,
        "step": 5,
        "total_steps": 10,
        "last_loss": 1.0,
        "eta_s": 0.0,
        "log_relative_path": "runs/run_01/train.log",
        "exit_code": exit_code,
        "terminal_reason": reason,
    })
    assert decode_run_status(encode_run_status(status)) == status
    if reason is not None:
        assert status.terminal_reason == reason


@pytest.mark.parametrize("name", ["schema", "schema_version", "soup_version"])
def test_wire_identity_is_required(name: str) -> None:
    data = json.loads(encode_run_status(RunStatus(**_MINIMAL)))
    del data[name]
    with pytest.raises(ValidationError, match=name):
        decode_run_status(json.dumps(data))


@pytest.mark.parametrize(
    "fields",
    [
        {"schema": "soup.command_result"},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": "1"},
        {"schema_version": 2},
        {"run_id": "../another-run"},
        {"started_by": "watcher"},
        {"state": "completed"},
        {"command": "export"},
        {"step": True},
        {"step": -1},
        {"step": "1"},
        {"step": 5, "total_steps": 4},
        {"total_steps": -1},
        {"last_loss": float("nan")},
        {"last_loss": float("inf")},
        {"last_loss": True},
        {"eta_s": -1.0},
        {"eta_s": float("-inf")},
        {"heartbeat_interval_s": 0},
        {"heartbeat_interval_s": False},
        {"finished_at": _LATER},
        {"exit_code": 0},
        {"terminal_reason": _REASON},
        {"updated_at": _NOW - timedelta(seconds=1)},
        {"started_at": _NOW.replace(tzinfo=None)},
        {"started_at": _NOW.astimezone(timezone(timedelta(hours=1)))},
        {"started_at": "2026-11-24T09:00:00Z"},
        {"owner": "private"},
        {"run_status_owner": "private"},
        {"worker_pid": 123},
        {"token": "private"},
        {"future_field": {}},
        {"truncated": {"/unknown": 1}},
    ],
)
def test_run_status_rejects_invalid_and_private_fields(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunStatus(**{**_MINIMAL, **fields})


@pytest.mark.parametrize(
    "update",
    [
        {"state": "done"},
        {"state": "done", "finished_at": _LATER, "exit_code": 1},
        {"state": "done", "finished_at": _LATER, "terminal_reason": _REASON},
        {"state": "failed", "finished_at": _LATER},
        {"state": "cancelled", "finished_at": _LATER},
        {"state": "orphaned", "finished_at": _LATER},
        {"state": "done", "finished_at": _NOW - timedelta(seconds=1)},
        {"state": "done", "finished_at": _LATER + timedelta(seconds=1)},
        {"state": "failed", "finished_at": _LATER, "exit_code": True, "terminal_reason": _REASON},
    ],
)
def test_terminal_status_rejects_incomplete_or_inconsistent_outcome(update) -> None:
    with pytest.raises(ValidationError):
        RunStatus(**{**_MINIMAL, **update})


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        1_764_000_000,
        "1764000000",
        "2026-11-24T09:00:00",
        "2026-11-24T09:00:00+01:00",
        "2026-11-24T09:00:00-00:00",
        "2026-11-24 09:00:00Z",
        "2026-02-30T09:00:00Z",
    ],
)
def test_wire_decode_rejects_non_rfc3339_and_non_utc_time(bad_timestamp) -> None:
    data = json.loads(encode_run_status(RunStatus(**_MINIMAL)))
    for field in ("started_at", "updated_at", "finished_at"):
        with pytest.raises(ValidationError, match=field):
            decode_run_status(json.dumps({**data, field: bad_timestamp}))


def test_wire_decode_preserves_utc_precision_and_normalizes_offset_spelling() -> None:
    data = json.loads(encode_run_status(RunStatus(**_MINIMAL)))
    data["updated_at"] = "2026-11-24T09:00:15.123456+00:00"
    status = decode_run_status(json.dumps(data))
    assert status.updated_at == _LATER.replace(microsecond=123456)
    assert json.loads(encode_run_status(status))["updated_at"] == "2026-11-24T09:00:15.123456Z"


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/train.log", "C:/train.log", "../train.log", "runs/./train.log",
        "runs//train.log", "runs\\train.log", "runs/train.log/",
        "runs/../train.log", "runs/log:stream",
        "runs/log\x00", "runs/log\u202e", "runs/log\u2028", "runs/log\x1b",
    ],
)
def test_log_reference_must_be_canonical_and_relative(value: str) -> None:
    with pytest.raises(ValidationError, match="log_relative_path"):
        RunStatus(**_MINIMAL, log_relative_path=value)


def test_public_fields_cannot_be_reassigned() -> None:
    status = RunStatus(**_MINIMAL)
    with pytest.raises(ValidationError, match="frozen"):
        status.step = 9
    with pytest.raises(ValidationError, match="step"):
        encode_run_status(status.model_copy(update={"step": -1}))


def test_wire_limit_counts_utf8_bytes_and_the_encoder_newline(monkeypatch) -> None:
    status = RunStatus(**{**_MINIMAL, "soup_version": "\U0001f642"})
    payload = encode_run_status(status)
    text = payload.decode()
    assert len(text) < len(payload)
    assert run_status.MAX_COMMAND_RESULT_BYTES == MAX_COMMAND_RESULT_BYTES == 262_144
    monkeypatch.setattr(run_status, "MAX_COMMAND_RESULT_BYTES", len(payload))
    assert encode_run_status(status) == payload
    assert decode_run_status(payload) == decode_run_status(text) == status
    monkeypatch.setattr(run_status, "MAX_COMMAND_RESULT_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="exceeds"):
        encode_run_status(status)
    for value in (payload, text, b"!" * len(payload)):
        with pytest.raises(ValueError, match="exceeds"):
            decode_run_status(value)


def test_reader_schema_extensions_and_version_bound() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = resources.files("soup_cli.schemas").joinpath("run-status-v1.json")
    schema = json.loads(schema_path.read_text("utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    document = json.loads(encode_run_status(RunStatus(**{
        **_MINIMAL,
        "soup_version": "v" * 96,
        "state": "failed",
        "finished_at": _LATER,
        "terminal_reason": _REASON,
    })))
    validator.validate(document)
    assert len(decode_run_status(json.dumps(document)).soup_version) == 96
    document["future_field"] = {}
    document["terminal_reason"]["future_field"] = []
    validator.validate(document)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        decode_run_status(json.dumps(document))
    document["soup_version"] += "v"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)
    with pytest.raises(ValidationError, match="soup_version"):
        RunStatus(**{**_MINIMAL, "soup_version": document["soup_version"]})
