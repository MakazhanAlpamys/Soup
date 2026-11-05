"""The `soup loop` data formats: packaged JSON Schemas and their example files.

Every example line must validate against its schema, the examples must refer to
each other consistently (feedback -> conversation, progress/version -> iteration),
and each schema must refuse the malformed records a producer is most likely to
write. See docs/loop-formats.md.
"""

from __future__ import annotations

import copy
import json
from importlib import resources

import pytest

from soup_cli.utils import loop_formats

jsonschema = pytest.importorskip("jsonschema")


def _validator(name: str):
    schema = loop_formats.load_schema(name)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )


def _errors(name: str, record: dict) -> list[str]:
    return [error.message for error in _validator(name).iter_errors(record)]


def _example(name: str, index: int = 0) -> dict:
    return copy.deepcopy(loop_formats.load_examples(name)[index])


# ---------------------------------------------------------------------------
# The package data itself
# ---------------------------------------------------------------------------


def test_every_format_ships_a_schema_and_an_example():
    for name in loop_formats.FORMATS:
        assert loop_formats.schema_resource(name).is_file(), name
        assert loop_formats.example_resource(name).is_file(), name


def test_no_schema_file_without_a_format_name():
    package = resources.files("soup_cli.schemas.loop")
    shipped = sorted(
        entry.name.removesuffix(".schema.json")
        for entry in package.iterdir()
        if entry.name.endswith(".schema.json")
    )
    assert shipped == sorted(loop_formats.FORMATS)


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_schema_is_a_valid_draft_2020_12_schema(name):
    schema = loop_formats.load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_schema_pins_the_current_schema_version(name):
    schema = loop_formats.load_schema(name)
    assert schema["properties"]["schema_version"] == {"const": loop_formats.SCHEMA_VERSION}
    assert "schema_version" in schema["required"]


def test_unknown_format_name_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown loop format 'traces'"):
        loop_formats.load_schema("traces")


def test_default_layout_keeps_every_log_under_soup_loops():
    for name, path in loop_formats.DEFAULT_LAYOUT.items():
        assert path.startswith(".soup-loops/"), name
    assert loop_formats.DEFAULT_LAYOUT["conversation"] == ".soup-loops/traces"


# ---------------------------------------------------------------------------
# Examples validate and agree with each other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_every_example_line_validates(name):
    examples = loop_formats.load_examples(name)
    assert len(examples) >= 3, f"{name}: an example file should show several cases"
    for line_no, record in enumerate(examples, start=1):
        assert _errors(name, record) == [], f"{name} example line {line_no}"


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_example_files_are_one_json_object_per_line(name):
    text = loop_formats.example_resource(name).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_feedback_examples_point_at_example_answers():
    answers = {record["id"] for record in loop_formats.load_examples("conversation")}
    for record in loop_formats.load_examples("feedback"):
        assert record["response_id"] in answers, record["id"]


def test_feedback_examples_cover_every_kind_and_a_cleared_thumb():
    feedback = loop_formats.load_examples("feedback")
    assert {record["kind"] for record in feedback} == {"thumb", "rating", "edit", "comment"}
    assert any(r["kind"] == "thumb" and r["value"] is None for r in feedback)


def test_iteration_examples_start_with_a_seed_and_reference_known_parents():
    iterations = loop_formats.load_examples("iteration")
    seed = iterations[0]
    assert seed["parent_iteration_id"] is None
    assert seed["from_version"] is None
    assert seed["to_version"] == 1
    assert seed["outcome"] == "promoted"
    known = {record["iteration_id"] for record in iterations}
    for record in iterations[1:]:
        assert record["parent_iteration_id"] in known


def test_iteration_examples_show_latest_line_wins_progression():
    lines = [
        r
        for r in loop_formats.load_examples("iteration")
        if r["iteration_id"] == "iter-20261105T120000-1b2c3d4e"
    ]
    assert [r["outcome"] for r in lines] == ["running", "canary", "promoted"]


def test_version_and_progress_examples_reference_example_iterations():
    known = {record["iteration_id"] for record in loop_formats.load_examples("iteration")}
    for name in ("version", "progress"):
        for record in loop_formats.load_examples(name):
            if record.get("iteration_id") is not None:
                assert record["iteration_id"] in known, (name, record)


def test_version_examples_cover_initial_promote_and_a_user_rollback():
    versions = loop_formats.load_examples("version")
    assert [r["cause"] for r in versions] == ["initial", "promote", "rollback"]
    assert versions[-1]["actor"] == "user"
    # The active version is the last line's to_version.
    assert versions[-1]["to_version"] == 1


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_unknown_fields_are_accepted(name):
    record = _example(name)
    record["field_from_a_future_version"] = {"anything": [1, 2, 3]}
    assert _errors(name, record) == []


@pytest.mark.parametrize("name", loop_formats.FORMATS)
def test_extension_namespace_must_be_an_object(name):
    record = _example(name)
    record["x_zero"] = {"job_id": "job-1"}
    assert _errors(name, record) == []
    record["x_zero"] = "job-1"
    assert _errors(name, record), "x_<app> must hold an object"


# ---------------------------------------------------------------------------
# Records a schema must refuse
# ---------------------------------------------------------------------------


def _set(path: str, value):
    """Return a mutation that sets a dotted path (``_DELETE`` removes it)."""

    def apply(record: dict) -> dict:
        target = record
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        if value is _DELETE:
            target.pop(keys[-1], None)
        else:
            target[keys[-1]] = value
        return record

    return apply


_DELETE = object()

# (format, example line, mutation, what is wrong)
REFUSED = [
    ("conversation", 0, _set("id", _DELETE), "no answer id"),
    ("conversation", 0, _set("id", "3F2B8C1E-5D4A-4F6B-9C2D-1A7E8B9C0D11"), "uppercase uuid"),
    ("conversation", 0, _set("id", "chatcmpl-1a2b3c4d"), "id is not a uuid"),
    ("conversation", 0, _set("ts", "2026-11-05T16:00:00+05:00"), "ts not UTC"),
    ("conversation", 0, _set("ts", "2026-11-05 11:00:00"), "ts not ISO 8601"),
    ("conversation", 0, _set("ts", 1762340400.0), "ts as epoch seconds (version 0)"),
    ("conversation", 0, _set("schema_version", 2), "future schema version"),
    ("conversation", 0, _set("tokens", 35), "tokens as a bare int (version 0)"),
    ("conversation", 0, _set("tokens.completion", _DELETE), "tokens without completion"),
    ("conversation", 0, _set("tokens.prompt", -1), "negative token count"),
    ("conversation", 0, _set("latency_ms", -5), "negative latency"),
    ("conversation", 0, _set("canary_bucket", "blue"), "unknown canary bucket"),
    ("conversation", 0, _set("messages", []), "no messages"),
    ("conversation", 0, _set("messages", [{"role": "user"}]), "message without content"),
    ("conversation", 0, _set("version", 0), "version 0"),
    ("conversation", 0, _set("source", "Soup Serve"), "source not a lowercase name"),
    ("feedback", 0, _set("value", 2), "thumb value 2"),
    ("feedback", 0, _set("value", True), "thumb value true"),
    ("feedback", 0, _set("value", _DELETE), "thumb without value"),
    ("feedback", 0, _set("kind", "like"), "unknown kind"),
    ("feedback", 0, _set("response_id", "answer-1"), "response_id not a uuid"),
    ("feedback", 0, _set("response_id", _DELETE), "feedback not tied to an answer"),
    ("feedback", 3, _set("value", 6), "rating above 5"),
    ("feedback", 3, _set("value", 0), "rating below 1"),
    ("feedback", 3, _set("value", 3.5), "fractional rating"),
    ("feedback", 1, _set("edited_text", _DELETE), "edit without edited_text"),
    ("feedback", 1, _set("edited_text", ""), "edit with empty text"),
    ("feedback", 5, _set("comment", _DELETE), "comment without text"),
    ("feedback", 5, _set("comment", None), "comment with null text"),
    ("iteration", 1, _set("iteration_id", "iter/escape"), "path separator in id"),
    ("iteration", 1, _set("iteration_id", "iter..up"), "dot-dot in id"),
    ("iteration", 1, _set("outcome", "done"), "unknown outcome"),
    ("iteration", 1, _set("outcome", _DELETE), "no outcome"),
    ("iteration", 2, _set("verdict.decision", "PASS"), "unknown verdict decision"),
    ("iteration", 2, _set("verdict.method", "vibes"), "unknown verdict method"),
    ("iteration", 2, _set("verdict.baseline.version", _DELETE), "version baseline, no number"),
    ("iteration", 2, _set("verdict.baseline.kind", "prod"), "unknown baseline kind"),
    ("iteration", 2, _set("canary.traffic_pct", 101), "traffic above 100%"),
    ("iteration", 2, _set("canary.verdict", "GOOD"), "unknown canary verdict"),
    ("iteration", 1, _set("dataset.sha256", "abc123"), "short sha256"),
    ("iteration", 1, _set("train.base", _DELETE), "training without its start point"),
    ("iteration", 1, _set("data.feedback.edit", -1), "negative feedback count"),
    ("version", 0, _set("from_version", 1), "initial with a previous version"),
    ("version", 2, _set("from_version", None), "rollback without a previous version"),
    ("version", 1, _set("cause", "upgrade"), "unknown cause"),
    ("version", 2, _set("actor", "admin"), "unknown actor"),
    ("version", 1, _set("to_version", None), "no target version"),
    ("progress", 0, _set("event", "finish"), "unknown event"),
    ("progress", 0, _set("stage", "Train"), "stage not lowercase"),
    ("progress", 0, _set("stage", ""), "empty stage"),
    ("progress", 3, _set("done", -1), "negative done"),
    ("progress", 0, _set("iteration_id", _DELETE), "iteration_id key missing"),
    ("progress", 0, _set("ts", "2026-11-05T12:00:00"), "ts without a zone"),
]


@pytest.mark.parametrize(
    ("name", "index", "mutate", "why"),
    REFUSED,
    ids=[f"{name}-{why}" for name, _, _, why in REFUSED],
)
def test_schema_refuses_malformed_record(name, index, mutate, why):
    record = mutate(_example(name, index))
    assert _errors(name, record), f"{name} accepted a record with {why}"


def test_open_stage_vocabulary_accepts_both_producers():
    for stage in ("harvest", "verdict", "promote", "data", "merge", "ship", "export"):
        record = _example("progress")
        record["stage"] = stage
        assert _errors("progress", record) == [], stage


def test_version_0_trace_line_is_not_a_version_1_record():
    """The line soup serve --trace-log writes today is version 0 (docs/loop-formats.md)."""
    legacy = {
        "ts": 1762340400.0,
        "prompt": "How do I reset my router?",
        "response": "Unplug it.",
        "latency_ms": 412.5,
        "tokens": 4,
    }
    assert _errors("conversation", legacy)
