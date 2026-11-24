"""Contract, generated schema, and installed-wheel checks for the result preview."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli import __version__
from soup_cli.data.formats import VALID_FORMATS
from soup_cli.utils.command_result import (
    MAX_COMMAND_RESULT_BYTES,
    ArtifactRef,
    CommandResult,
    DataValidateResult,
    ResultReason,
    encode_command_result,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = "command-result-v1.json"
_WHEEL_SCHEMA = f"soup_cli/schemas/{_SCHEMA}"
_NOW = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)
_IDENTITY = {"schema": "soup.command_result", "schema_version": 1, "soup_version": __version__}


def _warning(
    *, code: str = "soup.data.rows_skipped", message: str = "Dataset rows were skipped.",
    fix: str = "Check the dataset format.",
) -> ResultReason:
    return ResultReason(code=code, message=message, fix=fix)


def _blocked() -> CommandResult:
    return CommandResult(
        **_IDENTITY,
        command="data.validate",
        phase="validate",
        status="blocked",
        exit_code=2,
        started_at=_NOW,
        duration_s=0.25,
        decision="invalid",
        reasons=[
            ResultReason(
                code="soup.data.valid_fraction_below_minimum",
                message="The valid row fraction is below the requested minimum.",
                fix="Repair malformed rows and rerun validation.",
                details={"actual": 0.75, "minimum": 0.9},
            ),
        ],
        warnings=[
            ResultReason(
                code="soup.data.rows_skipped",
                message="Dataset rows were skipped.",
                fix="Check the dataset format.",
                details={"skipped": 1, "total": 4},
            ),
        ],
        data=DataValidateResult(format="chatml", total=4, valid=3, invalid=1, valid_fraction=0.75),
    )


def test_command_result_schema_snapshot() -> None:
    check = subprocess.run(
        [sys.executable, "scripts/gen_result_schemas.py", "--check"],
        cwd=_ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    assert check.returncode == 0, check.stderr
    schema = json.loads(resources.files("soup_cli.schemas").joinpath(_SCHEMA).read_text("utf-8"))
    assert schema == CommandResult.model_json_schema(by_alias=True)
    assert schema["properties"]["schema"]["const"] == "soup.command_result"
    assert "schema_id" not in schema["properties"]
    assert "launch" in schema["properties"]["phase"]["enum"]
    assert schema["properties"]["child"]["type"] == "null"
    assert schema["properties"]["reasons"]["maxItems"] == 32
    assert set(_IDENTITY) <= set(schema["required"])
    assert all("default" not in schema["properties"][name] for name in _IDENTITY)
    properties = schema["properties"]
    assert properties["schema"]["type"] == "string"
    assert properties["schema_version"]["type"] == "integer"
    assert properties["started_at"]["pattern"].endswith(r"(?:Z|\+00:00)$")
    assert "additionalProperties" not in schema
    for name in ("ResultReason", "ResultDetails", "DataValidateResult", "ArtifactRef"):
        assert "additionalProperties" not in schema["$defs"][name]
    data_format = schema["$defs"]["DataValidateResult"]["properties"]["format"]
    assert "pattern" in data_format and "enum" not in data_format
    truncated = properties["truncated"]
    assert "pattern" in truncated["propertyNames"] and "maxProperties" not in truncated
    assert truncated["additionalProperties"] == {"exclusiveMinimum": 0, "type": "integer"}


def test_reader_schema_accepts_additive_fields_and_checks_truncation() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(resources.files("soup_cli.schemas").joinpath(_SCHEMA).read_text("utf-8"))
    reader = jsonschema.Draft202012Validator(schema)
    document = json.loads(encode_command_result(_blocked()))
    document["future_field"] = {}
    document["data"]["future_field"] = []
    document["data"]["format"] = "future-format"
    document["reasons"][0]["future_field"] = True
    document["reasons"][0]["details"]["future_field"] = "value"
    document["artifacts"] = [{"kind": "export", "relative_path": "exports/model", "future": None}]
    document["truncated"] = dict.fromkeys(
        ("/reasons", "/warnings", "/artifacts", "/data/tasks", "/future/a~0b~1c"), 1,
    )
    reader.validate(document)
    for pointers in (
        {"": 1}, {"/data/tasks": 0}, {"/data/tasks": "1"}, {"data/tasks": 1}, {"/a~2": 1},
    ):
        with pytest.raises(jsonschema.ValidationError):
            reader.validate({**document, "truncated": pointers})
    with pytest.raises(jsonschema.ValidationError):
        reader.validate({**document, "data": {**document["data"], "format": "Future format"}})


def test_command_result_producer_remains_strict() -> None:
    document = json.loads(encode_command_result(_blocked()))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CommandResult.model_validate({**document, "started_at": _NOW, "future_field": {}})


def test_command_result_wire_names_defaults_and_typed_details() -> None:
    minimal = CommandResult(
        **_IDENTITY, command="data.validate", phase="validate", status="ok", exit_code=0,
    )
    assert json.loads(encode_command_result(minimal)) == {
        "schema": "soup.command_result",
        "schema_version": 1,
        "soup_version": __version__,
        "command": "data.validate",
        "phase": "validate",
        "status": "ok",
        "exit_code": 0,
        "run_id": None,
        "started_at": None,
        "duration_s": None,
        "decision": None,
        "reasons": [],
        "warnings": [],
        "data": None,
        "artifacts": [],
        "truncated": {},
        "child": None,
    }
    payload = encode_command_result(_blocked())
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    decoded = json.loads(payload)
    assert decoded["started_at"] == "2026-11-24T09:00:00Z"
    assert decoded["reasons"][0]["details"] == {"actual": 0.75, "minimum": 0.9}
    assert decoded["warnings"][0]["details"] == {"skipped": 1, "total": 4}
    assert CommandResult.model_validate_json(payload) == _blocked()


@pytest.mark.parametrize("name", _IDENTITY)
def test_command_result_rejects_missing_wire_identity(name: str) -> None:
    document = json.loads(encode_command_result(_blocked()))
    del document[name]
    with pytest.raises(ValidationError) as error:
        CommandResult.model_validate_json(json.dumps(document))
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [((name,), "missing")]


@pytest.mark.parametrize(
    "update",
    [
        {"duration_s": float("nan")},
        {"duration_s": float("inf")},
        {"duration_s": -1.0},
        {"started_at": datetime(2026, 11, 24, 9, 0)},
        {"started_at": _NOW.astimezone(timezone(timedelta(hours=1)))},
        {"started_at": "2026-11-24T09:00:00Z"},
        {"exit_code": True},
        {"child": {}},
    ],
)
def test_command_result_rejects_invalid_scalars(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CommandResult(**_IDENTITY, command="data.validate", phase="validate", status="ok", **update)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"status": "ok"}, "an ok command result cannot have refusal reasons"),
        (
            {"phase": "launch", "status": "ok", "reasons": []},
            "an ok launch result requires a run_id",
        ),
        ({"command": "train"}, "the preview payload belongs to data.validate"),
    ],
)
def test_command_result_rejects_inconsistent_outcomes(
    update: dict[str, object], message: str,
) -> None:
    fields = _blocked().model_dump(by_alias=True)
    with pytest.raises(ValidationError, match=message):
        CommandResult.model_validate({**fields, **update})


@pytest.mark.parametrize(
    "value",
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
def test_command_result_rejects_invalid_json_timestamps(value: object) -> None:
    document = json.loads(encode_command_result(_blocked()))
    document["started_at"] = value
    with pytest.raises(ValidationError, match="started_at"):
        CommandResult.model_validate_json(json.dumps(document))


def test_command_result_accepts_explicit_json_utc_offset() -> None:
    document = json.loads(encode_command_result(_blocked()))
    document["started_at"] = "2026-11-24T09:00:00.123456+00:00"
    result = CommandResult.model_validate_json(json.dumps(document))
    assert result.started_at == _NOW.replace(microsecond=123456)
    assert json.loads(encode_command_result(result))["started_at"] == "2026-11-24T09:00:00.123456Z"


@pytest.mark.parametrize(
    "details",
    [
        {"actual": float("nan")},
        {"minimum": 1.1},
        {"skipped": True},
        {"skipped": -1},
        {"secret": 1},
    ],
)
def test_reason_details_accept_only_known_finite_values(details: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ResultReason(
            code="soup.data.rows_skipped", message="Rows skipped.", fix="Check.", details=details,
        )


def test_empty_validation_retains_null_fraction() -> None:
    data = DataValidateResult(format="chatml", total=0, valid=0, invalid=0, valid_fraction=None)
    assert data.valid_fraction is None
    with pytest.raises(ValidationError):
        DataValidateResult(format="chatml", total=0, valid=0, invalid=0, valid_fraction=0.0)


@pytest.mark.parametrize("name", VALID_FORMATS)
def test_data_validation_receipt_accepts_each_supported_format(name: str) -> None:
    counts = DataValidateResult(format=name, total=1, valid=1, invalid=0, valid_fraction=1.0)
    document = json.loads(encode_command_result(_blocked()))
    document["data"] = counts.model_dump()
    assert CommandResult.model_validate_json(json.dumps(document)).data == counts


def test_data_validation_receipt_rejects_an_unknown_format() -> None:
    with pytest.raises(ValidationError, match="format"):
        DataValidateResult(format="unknown", total=0, valid=0, invalid=0, valid_fraction=None)


@pytest.mark.parametrize("text", ["a\nb", "a\x1bb", "a\x9bb", "a\u202eb"])
def test_result_reason_rejects_controls_and_bidi(text: str) -> None:
    with pytest.raises(ValidationError, match="controls or bidi"):
        _warning(message=text)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute", "../outside", "runs/../other", "C:/outside", "a//b", "a/b\tc",
        "a/b\x1bc", "a/b\x7fc", "a/b\x9bc", "a/b\u061cc", "a/b\u2028c", "a/b\u2029c",
        "a/b\u202ec", "a/b\u2067c",
    ],
)
def test_artifact_path_requires_canonical_relative_shape(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(kind="export", relative_path=path)


def test_artifact_path_resolves_under_the_reporting_root(tmp_path: Path) -> None:
    artifact = ArtifactRef(kind="export", relative_path="exports/model with spaces")
    path = tmp_path / artifact.relative_path
    path.parent.mkdir()
    path.touch()
    assert artifact.resolve_under(tmp_path).samefile(path)


@pytest.mark.requires_symlink
def test_artifact_path_rejects_a_link_to_another_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    other = tmp_path / "reports-other"
    other.mkdir()
    (other / "model").touch()
    (root / "export").symlink_to(other, target_is_directory=True)
    artifact = ArtifactRef(kind="export", relative_path="export/model")
    with pytest.raises(ValueError, match="reporting root"):
        artifact.resolve_under(root)


def test_result_byte_limit_preserves_decision_primary_reason_and_omission_counts() -> None:
    primary = _blocked().reasons[0]
    text = "\U0001f642" * 1024
    reason = _warning(code="soup.data.additional_reason", message=text, fix=text)
    warning = _warning(message=text, fix=text)
    artifact = ArtifactRef(kind="export", relative_path="exports/model with spaces")
    result = CommandResult(
        **_IDENTITY,
        command="data.validate",
        phase="validate",
        status="blocked",
        exit_code=2,
        decision="invalid",
        data=_blocked().data,
        reasons=[primary, *([reason] * 35)],
        warnings=[warning] * 35,
        artifacts=[artifact] * 34,
        truncated={"/warnings": 2},
    )
    encoded = encode_command_result(result)
    decoded = json.loads(encoded)
    assert len(encoded) <= MAX_COMMAND_RESULT_BYTES == 262_144
    assert encoded.endswith(b"\n") and encoded.count(b"\n") == 1
    assert decoded["status"] == "blocked" and decoded["exit_code"] == 2
    assert decoded["decision"] == "invalid" and decoded["data"]["total"] == 4
    assert decoded["reasons"][0]["code"] == primary.code
    for key, count in (("reasons", 36), ("warnings", 37), ("artifacts", 34)):
        assert len(decoded[key]) <= 32
        assert decoded["truncated"][f"/{key}"] == count - len(decoded[key])
    assert len(decoded["reasons"]) + len(decoded["warnings"]) < 64


def test_unserializable_result_falls_back_and_preserves_exit() -> None:
    broken = _blocked().model_copy(update={"duration_s": float("inf")})
    response = json.loads(encode_command_result(broken))
    assert response["status"] == "failed" and response["exit_code"] == 2
    assert response["command"] == "data.validate" and response["phase"] == "validate"
    assert response["reasons"][0]["code"] == "soup.cli.result_serialization_failed"
    assert len(json.dumps(response)) < 1024


@pytest.mark.skipif(importlib.util.find_spec("hatchling") is None, reason="needs local Hatchling")
def test_installed_wheel_command_result_schema(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    build = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(wheel_dir)],
        cwd=_ROOT, capture_output=True, text=True, timeout=600, check=False,
    )
    assert build.returncode == 0, build.stderr
    (wheel,) = wheel_dir.glob("soup_cli-*.whl")
    with zipfile.ZipFile(wheel) as bundle:
        assert bundle.namelist().count(_WHEEL_SCHEMA) == 1
        assert "soup_cli/schemas/run-status-v1.json" not in bundle.namelist()

    install_dir = tmp_path / "installed"
    install = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
            "--disable-pip-version-check", "--target", str(install_dir), str(wheel),
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False,
    )
    assert install.returncode == 0, install.stderr
    sample_path = tmp_path / "sample.json"
    sample_path.write_bytes(encode_command_result(_blocked()))
    check = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import json, pathlib, sys\n"
            "from importlib.resources import files\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import soup_cli\n"
            "from soup_cli.utils.command_result import CommandResult\n"
            "package = pathlib.Path(sys.argv[1]) / 'soup_cli'\n"
            "assert pathlib.Path(soup_cli.__file__).parent == package\n"
            f"resource = files('soup_cli.schemas').joinpath({_SCHEMA!r})\n"
            "schema = json.loads(resource.read_text('utf-8'))\n"
            "assert schema == CommandResult.model_json_schema(by_alias=True)\n"
            "sample = pathlib.Path(sys.argv[2]).read_bytes()\n"
            "result = CommandResult.model_validate_json(sample)\n"
            "assert result.exit_code == 2 and result.data.total == 4\n",
            str(install_dir), str(sample_path),
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert check.returncode == 0, check.stderr


def _train_preflight_data() -> dict:
    return {
        "dry_run": True,
        "training_rows": 4,
        "validation_rows": 1,
        "vram": {
            "estimated_peak_gb": 8.0, "available_gb": 10.0,
            "margin_fraction": 0.1, "required_gb": 8.8,
        },
        "disk": {"estimated_output_gb": None, "estimated_checkpoints_gb": None},
        "estimated_wall_time_s": None,
        "estimate_basis": {
            "vram": "analytical_model", "disk": "unavailable", "wall_time": "unavailable",
        },
        "resolved": {
            "base_model": "org/model", "task": "sft", "backend": "transformers",
            "device_name": "Test CUDA", "max_steps": None, "batch_size_per_device": 2,
            "effective_batch_size": 8, "context_length": 512, "adapter_rank": 8, "epochs": 2.0,
        },
        "input_digests": {"config_sha256": None, "data_sha256": "a" * 64},
    }


def test_train_preflight_wire_roundtrip() -> None:
    data = _train_preflight_data()
    model = CommandResult(
        **_IDENTITY, command="train", phase="preflight", status="ok", exit_code=0, data=data,
    )
    wire = encode_command_result(model)
    assert json.loads(wire)["data"] == data
    assert CommandResult.model_validate_json(wire) == model
    schema = CommandResult.model_json_schema(by_alias=True)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(schema).validate(json.loads(wire))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("training_rows",), True),
        (("vram", "estimated_peak_gb"), float("inf")),
        (("estimate_basis", "vram"), "unavailable"),
        (("resolved", "base_model"), "/private/model"),
        (("resolved", "base_model"), "C:/private/model"),
        (("resolved", "base_model"), "../model"),
        (("resolved", "adapter_rank"), 0),
        (("input_digests", "data_sha256"), "A" * 64),
    ],
)
def test_train_preflight_wire_rejects_invalid_fields(field: tuple[str, ...], value: object) -> None:
    data = _train_preflight_data()
    target = data if len(field) == 1 else data[field[0]]
    target[field[-1]] = value
    with pytest.raises(ValidationError):
        CommandResult(
            **_IDENTITY, command="train", phase="preflight", status="ok", exit_code=0, data=data,
        )


def test_train_identifier_survives_serializer_fallback() -> None:
    result = CommandResult(
        **_IDENTITY, command="train", phase="preflight", status="refused", exit_code=7,
    )
    broken = result.model_copy(update={"duration_s": float("inf")})
    wire = json.loads(encode_command_result(broken))
    assert (wire["command"], wire["phase"], wire["status"], wire["exit_code"]) == (
        "train", "preflight", "failed", 7,
    )
    assert wire["reasons"][0]["code"] == "soup.cli.result_serialization_failed"


def test_train_config_error_details_are_bounded_with_pointer() -> None:
    errors = [
        {"loc": ["training", "batch_size", index], "type": "float_type"}
        for index in range(33)
    ]
    result = CommandResult(
        **_IDENTITY, command="train", phase="preflight", status="refused", exit_code=1,
        reasons=[{
            "code": "soup.config.invalid",
            "message": "The configuration could not be loaded.",
            "fix": "Check the fields listed in the result.",
            "details": {"errors": errors},
        }],
    )
    data = json.loads(encode_command_result(result))
    assert data["reasons"][0]["details"]["errors"] == errors[:32]
    assert data["truncated"] == {"/reasons/0/details/errors": 1}
