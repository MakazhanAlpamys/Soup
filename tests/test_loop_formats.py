"""The `soup loop` data formats: packaged JSON Schemas and their example files.

Every example line must validate against its schema, the examples must refer to
each other consistently (feedback -> conversation, progress/version -> iteration),
and each schema must refuse the malformed records a producer is most likely to
write. See docs/loop-formats.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime
from importlib import resources
from pathlib import Path

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


def test_paraphrase_is_a_format_with_its_own_log():
    assert len(loop_formats.FORMATS) == 6
    assert loop_formats.DEFAULT_LAYOUT["paraphrase"] == ".soup-loops/paraphrases.jsonl"
    assert loop_formats.FORMATS.index("feedback") < loop_formats.FORMATS.index("paraphrase")
    assert loop_formats.FORMATS.index("paraphrase") < loop_formats.FORMATS.index("iteration")


def test_the_docs_describe_every_format_and_its_default_path():
    docs_path = Path(__file__).resolve().parent.parent / "docs" / "loop-formats.md"
    docs = docs_path.read_text(encoding="utf-8")
    for name, path in loop_formats.DEFAULT_LAYOUT.items():
        assert f"\n## {name}\n" in docs, f"docs/loop-formats.md has no section for {name}"
        assert path in docs, f"docs/loop-formats.md does not name {path}"


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


QUALITY_FIELDS = {
    "verdict",
    "rated",
    "negative",
    "share",
    "difference",
    "z",
    "p_value",
    "min_rated",
    "threshold",
    "timed_out",
    "decided_on",
}


def _quality_examples() -> list[dict]:
    return [
        record["canary"]["quality"]
        for record in loop_formats.load_examples("iteration")
        if isinstance(record.get("canary"), dict) and "quality" in record["canary"]
    ]


def test_iteration_examples_show_canary_quality_evidence():
    examples = _quality_examples()
    assert len(examples) >= 2
    for quality in examples:
        assert set(quality) == QUALITY_FIELDS
    # one that is still waiting (null shares) and one that was decided
    assert {quality["verdict"] for quality in examples} >= {"UNKNOWN", "OK"}


def test_iteration_example_quality_validates_on_its_own_line():
    lines = [
        (index, record)
        for index, record in enumerate(loop_formats.load_examples("iteration"))
        if isinstance(record.get("canary"), dict) and "quality" in record["canary"]
    ]
    assert lines
    for index, record in lines:
        assert _errors("iteration", record) == [], f"iteration example line {index + 1}"


def test_iteration_line_without_quality_still_validates():
    for index, record in enumerate(loop_formats.load_examples("iteration")):
        stripped = copy.deepcopy(record)
        if isinstance(stripped.get("canary"), dict):
            stripped["canary"].pop("quality", None)
        assert _errors("iteration", stripped) == [], f"iteration example line {index + 1}"
    # the pre-quality shape, written out: samples and verdict only
    legacy = _example("iteration", 3)
    legacy["canary"] = {"traffic_pct": 5, "samples": {"stable": 412, "canary": 31}, "verdict": "OK"}
    assert _errors("iteration", legacy) == []


def test_iteration_example_quality_agrees_with_its_own_counts():
    for quality in _quality_examples():
        for bucket in ("stable", "canary"):
            rated, negative = quality["rated"][bucket], quality["negative"][bucket]
            assert negative <= rated, bucket
            if rated == 0:
                assert quality["share"][bucket] is None, bucket
            else:
                assert quality["share"][bucket] == pytest.approx(negative / rated, abs=1e-4)
        shares = quality["share"]
        if shares["stable"] is None or shares["canary"] is None:
            assert quality["difference"] is None
            assert quality["z"] is None
            assert quality["p_value"] is None
        else:
            assert quality["difference"] == pytest.approx(
                shares["canary"] - shares["stable"], abs=1e-4
            )
            assert (quality["z"] > 0) == (quality["difference"] > 0)
        assert quality["rated"]["canary"] <= _canary_samples()["canary"]


def _canary_samples() -> dict:
    # the samples of the last canary line (the one that carries the OK evidence)
    for record in reversed(loop_formats.load_examples("iteration")):
        if isinstance(record.get("canary"), dict) and record["canary"].get("verdict") == "OK":
            return record["canary"]["samples"]
    raise AssertionError("no OK canary example")


@pytest.mark.parametrize("field", sorted(QUALITY_FIELDS))
def test_quality_without_any_one_field_is_refused(field):
    record = _example("iteration", 3)
    del record["canary"]["quality"][field]
    assert _errors("iteration", record), f"quality accepted without {field}"


@pytest.mark.parametrize(
    ("verdict", "timed_out", "decided_on"),
    [
        ("OK", False, "errors_and_quality"),
        ("MAJOR", False, "errors_and_quality"),
        ("UNKNOWN", False, "errors_and_quality"),
        ("INSUFFICIENT", True, "errors_only"),
        ("INSUFFICIENT", True, "errors_and_quality"),
    ],
)
def test_quality_accepts_every_verdict_and_decision_basis(verdict, timed_out, decided_on):
    record = _example("iteration", 3)
    record["canary"]["quality"].update(
        {"verdict": verdict, "timed_out": timed_out, "decided_on": decided_on}
    )
    assert _errors("iteration", record) == []


def test_quality_accepts_the_edges_of_every_range():
    record = _example("iteration", 3)
    record["canary"]["quality"].update(
        {
            "rated": {"stable": 0, "canary": 0},
            "negative": {"stable": 0, "canary": 0},
            "share": {"stable": 1, "canary": 0},
            "difference": -1,
            "z": -3.2,
            "p_value": 1,
            "min_rated": 1,
            "threshold": 0,
        }
    )
    assert _errors("iteration", record) == []
    record["canary"]["quality"].update({"difference": 1, "p_value": 0, "threshold": 1})
    assert _errors("iteration", record) == []


def test_quality_with_nulls_where_the_test_is_undefined_validates():
    record = _example("iteration", 3)
    record["canary"]["quality"].update(
        {"share": {"stable": 0.0, "canary": None}, "difference": None, "z": None, "p_value": None}
    )
    assert _errors("iteration", record) == []


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _answers() -> dict[str, dict]:
    return {record["id"]: record for record in loop_formats.load_examples("conversation")}


def _edited_answer_ids(until: str | None = None) -> set[str]:
    return {
        record["response_id"]
        for record in loop_formats.load_examples("feedback")
        if record["kind"] == "edit"
        and record["edited_text"] is not None
        and (until is None or _utc(record["ts"]) <= _utc(until))
    }


def _current_paraphrases(until: str | None = None) -> dict[str, dict]:
    """The row per response_id a reader uses: the last one in file order.

    ``until`` keeps only the rows created at or before that moment.
    """
    current: dict[str, dict] = {}
    for row in loop_formats.load_examples("paraphrase"):
        if until is None or _utc(row["created_at"]) <= _utc(until):
            current[row["response_id"]] = row
    return current


def test_paraphrase_examples_point_at_edited_example_answers():
    rows = loop_formats.load_examples("paraphrase")
    edited = _edited_answer_ids()
    for row in rows:
        assert row["response_id"] in edited, row["response_id"]


def test_paraphrase_examples_hash_the_prompt_of_their_answer():
    answers = _answers()
    for row in loop_formats.load_examples("paraphrase"):
        expected = loop_formats.prompt_hash(answers[row["response_id"]]["messages"])
        assert row["prompt_hash"] == expected, row["response_id"]


def test_paraphrase_examples_show_a_plain_row_an_extension_and_the_optional_fields():
    rows = loop_formats.load_examples("paraphrase")
    assert any("x_zero" in row for row in rows)
    assert any("dropped" in row and "dropped_reasons" in row for row in rows)
    assert any(
        not {"x_zero", "dropped", "dropped_reasons"} & set(row) for row in rows
    ), "no plain row"


def test_paraphrase_examples_show_that_the_last_row_per_answer_wins():
    rows = loop_formats.load_examples("paraphrase")
    ids = [row["response_id"] for row in rows]
    repeated = sorted({response_id for response_id in ids if ids.count(response_id) > 1})
    assert repeated, "no answer has a regenerated row"
    for response_id in repeated:
        created = [_utc(row["created_at"]) for row in rows if row["response_id"] == response_id]
        assert created == sorted(created), "append order is time order"
        first = next(row for row in rows if row["response_id"] == response_id)
        assert _current_paraphrases()[response_id] is not first
        assert _current_paraphrases()[response_id]["paraphrases"] != first["paraphrases"]


def _sign_test_p(successes: int, trials: int) -> float | None:
    """One-sided sign test: the chance of at least ``successes`` out of ``trials`` at p = 1/2."""
    if trials == 0:
        return None
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2**trials


def _evidence_lines() -> list[tuple[int, dict]]:
    return [
        (index, record)
        for index, record in enumerate(loop_formats.load_examples("iteration"))
        if isinstance(record.get("verdict"), dict) and "corrections" in record["verdict"]
    ]


CORRECTIONS_FIELDS = {
    "fixed",
    "broken",
    "ties",
    "n_edits",
    "n_paraphrases",
    "p_value",
    "scoring",
    "outcome",
}
HELD_OUT_FIELDS = {"better", "worse", "ties", "n", "p_value", "scoring", "outcome"}
SCORING_FIELDS = {"method", "threshold", "template"}
STYLE_FIELDS = {"preferred_new", "preferred_old", "n", "p_value"}


def test_iteration_examples_show_verdict_evidence_on_every_verdict_line():
    lines = _evidence_lines()
    with_verdict = [
        record
        for record in loop_formats.load_examples("iteration")
        if isinstance(record.get("verdict"), dict)
    ]
    assert len(lines) >= 2
    assert len(lines) == len(with_verdict), "the verdict is the same on every line that has one"
    for _, record in lines:
        verdict = record["verdict"]
        assert set(verdict["corrections"]) == CORRECTIONS_FIELDS
        assert set(verdict["held_out"]) >= HELD_OUT_FIELDS
        assert set(verdict["corrections"]["scoring"]) == SCORING_FIELDS
        assert set(verdict["held_out"]["scoring"]) == SCORING_FIELDS


def test_iteration_example_evidence_validates_on_its_own_line():
    for index, record in _evidence_lines():
        assert _errors("iteration", record) == [], f"iteration example line {index + 1}"


def test_iteration_line_without_verdict_evidence_still_validates():
    for index, record in enumerate(loop_formats.load_examples("iteration")):
        stripped = copy.deepcopy(record)
        if isinstance(stripped.get("verdict"), dict):
            stripped["verdict"].pop("corrections", None)
            stripped["verdict"].pop("held_out", None)
        assert _errors("iteration", stripped) == [], f"iteration example line {index + 1}"
    # evidence on one side only is valid too: each object is optional on its own
    only_corrections = _example("iteration", 2)
    del only_corrections["verdict"]["held_out"]
    assert _errors("iteration", only_corrections) == []
    only_held_out = _example("iteration", 2)
    del only_held_out["verdict"]["corrections"]
    assert _errors("iteration", only_held_out) == []


def test_iteration_example_corrections_agree_with_the_example_data():
    for _, record in _evidence_lines():
        corrections = record["verdict"]["corrections"]
        fixed, broken = corrections["fixed"], corrections["broken"]
        assert fixed + broken + corrections["ties"] == corrections["n_edits"]
        expected_p = _sign_test_p(fixed, fixed + broken)
        assert corrections["p_value"] == pytest.approx(expected_p, abs=1e-4)
        if fixed + broken < 5:
            assert corrections["outcome"] in {"insufficient", "not_applicable"}
        if corrections["n_edits"] == 0:
            assert corrections["outcome"] == "not_applicable"
        # the edits of the training set are the edits the harvest read, and the
        # paraphrases are those that existed when the harvest ran
        cursor = record["data"]["until"]
        assert corrections["n_edits"] == record["data"]["feedback"]["edit"]
        assert corrections["n_edits"] == len(_edited_answer_ids(cursor))
        paraphrased = _current_paraphrases(cursor)
        assert corrections["n_paraphrases"] == sum(
            len(paraphrased[response_id]["paraphrases"])
            for response_id in _edited_answer_ids(cursor)
        )


def test_iteration_example_held_out_agrees_with_its_own_counts():
    for _, record in _evidence_lines():
        held_out = record["verdict"]["held_out"]
        better, worse = held_out["better"], held_out["worse"]
        assert better + worse + held_out["ties"] == held_out["n"]
        if held_out["n"] < 5:
            assert held_out["outcome"] == "not_measured"
        if better + worse == 0:
            assert held_out["p_value"] is None
        else:
            assert held_out["p_value"] == pytest.approx(
                _sign_test_p(worse, better + worse), abs=1e-4
            )


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

CORR = "verdict.corrections"
HELD = "verdict.held_out"


def _style(**changes):
    """A valid held_out.style object with some fields changed (``_DELETE`` removes one)."""
    style = {"preferred_new": 3, "preferred_old": 1, "n": 5, "p_value": 0.1875}
    style.update(changes)
    return {key: value for key, value in style.items() if value is not _DELETE}


# (format, example line, mutation, what is wrong)
REFUSED = [
    ("iteration", 0, _set("dataset", {"sft_rows": -1}), "negative sft row count"),
    ("iteration", 0, _set("dataset", {"sft_rows": "3"}), "sft row count as a string"),
    ("iteration", 0, _set("dataset", {"sft_rows": True}), "sft row count as a bool"),
    ("iteration", 0, _set("dataset", {"graduated": -1}), "negative graduated count"),
    ("iteration", 0, _set("dataset", {"graduated": 1.5}), "graduated count not an int"),
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
    ("iteration", 3, _set("canary.quality.verdict", "GOOD"), "unknown quality verdict"),
    ("iteration", 3, _set("canary.quality.verdict", None), "null quality verdict"),
    ("iteration", 3, _set("canary.quality.verdict", _DELETE), "quality without a verdict"),
    ("iteration", 3, _set("canary.quality.share.canary", 1.2), "negative share above 1"),
    ("iteration", 3, _set("canary.quality.share.stable", -0.1), "negative share below 0"),
    ("iteration", 3, _set("canary.quality.share.stable", "0.15"), "share as a string"),
    ("iteration", 3, _set("canary.quality.share.canary", _DELETE), "share without a bucket"),
    ("iteration", 3, _set("canary.quality.negative.canary", -1), "negative count below 0"),
    ("iteration", 3, _set("canary.quality.rated.stable", -5), "rated count below 0"),
    ("iteration", 3, _set("canary.quality.rated.canary", 31.5), "fractional rated count"),
    ("iteration", 3, _set("canary.quality.rated.canary", True), "rated count as a bool"),
    ("iteration", 3, _set("canary.quality.rated.canary", _DELETE), "rated without a bucket"),
    ("iteration", 3, _set("canary.quality.rated.extra", 1), "rated with a third bucket"),
    ("iteration", 3, _set("canary.quality.difference", 1.5), "difference above 1"),
    ("iteration", 3, _set("canary.quality.difference", -1.5), "difference below -1"),
    ("iteration", 3, _set("canary.quality.difference", _DELETE), "quality without difference"),
    ("iteration", 3, _set("canary.quality.z", "0.63"), "z as a string"),
    ("iteration", 3, _set("canary.quality.z", _DELETE), "quality without z"),
    ("iteration", 3, _set("canary.quality.p_value", 1.5), "p-value above 1"),
    ("iteration", 3, _set("canary.quality.p_value", -0.01), "p-value below 0"),
    ("iteration", 3, _set("canary.quality.min_rated", 0), "min_rated below 1"),
    ("iteration", 3, _set("canary.quality.min_rated", 30.5), "fractional min_rated"),
    ("iteration", 3, _set("canary.quality.threshold", 1.5), "threshold above 1"),
    ("iteration", 3, _set("canary.quality.threshold", -0.05), "threshold below 0"),
    ("iteration", 3, _set("canary.quality.timed_out", 0), "timed_out as an int"),
    ("iteration", 3, _set("canary.quality.timed_out", _DELETE), "quality without timed_out"),
    ("iteration", 3, _set("canary.quality.decided_on", "quality_only"), "unknown decided_on"),
    ("iteration", 3, _set("canary.quality.decided_on", _DELETE), "quality without decided_on"),
    ("iteration", 3, _set("canary.quality.unexpected", 1), "unknown field inside quality"),
    ("iteration", 3, _set("canary.quality", "OK"), "quality as a bare string"),
    ("iteration", 3, _set("canary.quality", None), "quality as null"),
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
    # paraphrase (example 0 is a plain row, 1 carries x_zero, 2 carries the dropped fields)
    ("paraphrase", 0, _set("schema_version", 2), "future schema version"),
    ("paraphrase", 0, _set("response_id", "answer-1"), "response_id not a uuid"),
    ("paraphrase", 0, _set("response_id", "3F2B8C1E-5D4A-4F6B-9C2D-1A7E8B9C0D11"), "uppercase id"),
    ("paraphrase", 0, _set("prompt_hash", "abc123"), "short prompt_hash"),
    ("paraphrase", 0, _set("prompt_hash", "A1" * 32), "uppercase hex prompt_hash"),
    ("paraphrase", 0, _set("prompt_hash", "a1" * 31 + "zz"), "non-hex prompt_hash"),
    ("paraphrase", 0, _set("prompt_hash", "sha256:" + "a1" * 32), "prefixed prompt_hash"),
    ("paraphrase", 0, _set("prompt_hash", "a1" * 33), "prompt_hash longer than sha256"),
    ("paraphrase", 0, _set("prompt_hash", None), "null prompt_hash"),
    ("paraphrase", 0, _set("paraphrases", []), "an empty paraphrases array"),
    ("paraphrase", 0, _set("paraphrases", [""]), "an empty string paraphrase"),
    ("paraphrase", 0, _set("paraphrases", ["fine", ""]), "an empty paraphrase after a good one"),
    ("paraphrase", 0, _set("paraphrases", "How do I reset it?"), "paraphrases as a bare string"),
    ("paraphrase", 0, _set("paraphrases", [7]), "a paraphrase that is not a string"),
    ("paraphrase", 0, _set("paraphrases", None), "null paraphrases"),
    ("paraphrase", 0, _set("created_at", "2026-11-05T16:00:00+05:00"), "created_at not UTC"),
    ("paraphrase", 0, _set("created_at", "2026-11-05T11:03:00"), "created_at without a zone"),
    ("paraphrase", 0, _set("created_at", "2026-11-05 11:03:00"), "created_at not ISO 8601"),
    ("paraphrase", 0, _set("created_at", 1762340400.0), "created_at as epoch seconds"),
    ("paraphrase", 0, _set("generator", "SmolLM2"), "generator as a bare string"),
    ("paraphrase", 0, _set("generator", None), "null generator"),
    ("paraphrase", 0, _set("generator.model", ""), "empty generator model"),
    ("paraphrase", 0, _set("generator.model", 3), "generator model as a number"),
    ("paraphrase", 0, _set("generator.template", "v1"), "template as a bare string"),
    ("paraphrase", 0, _set("generator.template", None), "null template"),
    ("paraphrase", 0, _set("generator.template.id", ""), "empty template id"),
    ("paraphrase", 0, _set("generator.template.version", ""), "empty template version"),
    ("paraphrase", 0, _set("generator.template.version", 1), "template version as a number"),
    ("paraphrase", 0, _set("generator.template.version", _DELETE), "template without a version"),
    ("paraphrase", 2, _set("dropped", -1), "negative dropped count"),
    ("paraphrase", 2, _set("dropped", 1.5), "fractional dropped count"),
    ("paraphrase", 2, _set("dropped", True), "dropped count as a bool"),
    ("paraphrase", 2, _set("dropped", "1"), "dropped count as a string"),
    ("paraphrase", 2, _set("dropped", None), "null dropped count"),
    ("paraphrase", 2, _set("dropped_reasons", ["boring"]), "an unknown drop reason"),
    ("paraphrase", 2, _set("dropped_reasons", ["leaks_answer", "boring"]), "one unknown reason"),
    ("paraphrase", 2, _set("dropped_reasons", "leaks_answer"), "dropped_reasons as a string"),
    ("paraphrase", 2, _set("dropped_reasons", None), "null dropped_reasons"),
    # verdict evidence (iteration example 2 carries both objects)
    ("iteration", 2, _set(f"{CORR}.outcome", "maybe"), "unknown corrections outcome"),
    ("iteration", 2, _set(f"{CORR}.outcome", "weak"), "a held_out outcome as corrections"),
    ("iteration", 2, _set(f"{CORR}.outcome", "not-applicable"), "misspelled not_applicable"),
    ("iteration", 2, _set(f"{CORR}.outcome", None), "null corrections outcome"),
    ("iteration", 2, _set(f"{CORR}.fixed", -1), "negative fixed count"),
    ("iteration", 2, _set(f"{CORR}.broken", 1.5), "fractional broken count"),
    ("iteration", 2, _set(f"{CORR}.ties", "2"), "ties as a string"),
    ("iteration", 2, _set(f"{CORR}.n_edits", True), "n_edits as a bool"),
    ("iteration", 2, _set(f"{CORR}.n_paraphrases", -3), "negative n_paraphrases"),
    ("iteration", 2, _set(f"{CORR}.p_value", 1.5), "corrections p-value above 1"),
    ("iteration", 2, _set(f"{CORR}.p_value", -0.01), "corrections p-value below 0"),
    ("iteration", 2, _set(f"{CORR}.p_value", "0.04"), "corrections p-value as a string"),
    ("iteration", 2, _set(f"{CORR}.scoring.method", "content_recall"), "content_recall"),
    ("iteration", 2, _set(f"{CORR}.scoring.method", "vibes"), "unknown scoring method"),
    ("iteration", 2, _set(f"{CORR}.scoring.method", None), "null scoring method"),
    ("iteration", 2, _set(f"{CORR}.scoring.threshold", 1.5), "scoring threshold above 1"),
    ("iteration", 2, _set(f"{CORR}.scoring.threshold", -0.5), "scoring threshold below 0"),
    ("iteration", 2, _set(f"{CORR}.scoring.template", {"id": "j"}), "template without version"),
    ("iteration", 2, _set(f"{CORR}.scoring.template", {"version": "1"}), "template without id"),
    ("iteration", 2, _set(f"{CORR}.scoring.template", "j@1"), "template as a bare string"),
    ("iteration", 2, _set(f"{CORR}.scoring.template", {"id": "", "version": "1"}), "empty id"),
    ("iteration", 2, _set(f"{CORR}.unexpected", 1), "unknown field inside corrections"),
    ("iteration", 2, _set(f"{CORR}.scoring.unexpected", 1), "unknown field inside scoring"),
    ("iteration", 2, _set(CORR, "pass"), "corrections as a bare string"),
    ("iteration", 2, _set(CORR, None), "corrections as null"),
    ("iteration", 2, _set(f"{HELD}.outcome", "maybe"), "unknown held_out outcome"),
    ("iteration", 2, _set(f"{HELD}.outcome", "insufficient"), "a corrections outcome as held_out"),
    ("iteration", 2, _set(f"{HELD}.outcome", None), "null held_out outcome"),
    ("iteration", 2, _set(f"{HELD}.better", -1), "negative better count"),
    ("iteration", 2, _set(f"{HELD}.worse", 2.5), "fractional worse count"),
    ("iteration", 2, _set(f"{HELD}.ties", True), "held_out ties as a bool"),
    ("iteration", 2, _set(f"{HELD}.n", -1), "negative held_out n"),
    ("iteration", 2, _set(f"{HELD}.p_value", 2), "held_out p-value above 1"),
    ("iteration", 2, _set(f"{HELD}.p_value", -0.1), "held_out p-value below 0"),
    ("iteration", 2, _set(f"{HELD}.scoring", _DELETE), "held_out without scoring"),
    ("iteration", 2, _set(f"{HELD}.scoring", None), "held_out scoring as null"),
    ("iteration", 2, _set(f"{HELD}.scoring.method", "content_recall"), "content_recall"),
    ("iteration", 2, _set(f"{HELD}.unexpected", 1), "unknown field inside held_out"),
    ("iteration", 2, _set(HELD, "pass"), "held_out as a bare string"),
    ("iteration", 2, _set(HELD, None), "held_out as null"),
    ("iteration", 2, _set(f"{HELD}.style", _style(preferred_new=-1)), "negative style count"),
    ("iteration", 2, _set(f"{HELD}.style", _style(n=1.5)), "fractional style n"),
    ("iteration", 2, _set(f"{HELD}.style", _style(p_value=1.5)), "style p-value above 1"),
    ("iteration", 2, _set(f"{HELD}.style", _style(p_value=_DELETE)), "style without p_value"),
    ("iteration", 2, _set(f"{HELD}.style", _style(unexpected=1)), "unknown field inside style"),
    ("iteration", 2, _set(f"{HELD}.style", "new"), "style as a bare string"),
    ("iteration", 2, _set(f"{HELD}.style", None), "style as null"),
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
    """The line soup serve --trace-log wrote before version 1 is version 0 (loop-formats.md)."""
    legacy = {
        "ts": 1762340400.0,
        "prompt": "How do I reset my router?",
        "response": "Unplug it.",
        "latency_ms": 412.5,
        "tokens": 4,
    }
    assert _errors("conversation", legacy)


def test_a_dataset_reports_the_rows_of_all_three_datasets():
    record = _example("iteration")
    record["dataset"] = {"path": ".soup-loops/pairs/sft-1.jsonl", "sha256": None,
                         "dpo_pairs": 4, "kto_rows": 9, "sft_rows": 3, "graduated": 2}
    assert _errors("iteration", record) == []


# ---------------------------------------------------------------------------
# iteration: verdict.corrections and verdict.held_out
# ---------------------------------------------------------------------------


_JUDGE = {"id": "judge", "version": "2"}


def _scoring(**changes) -> dict:
    scoring = {"method": "added_recall", "threshold": 0.5, "template": None}
    scoring.update(changes)
    return scoring


@pytest.mark.parametrize("field", sorted(CORRECTIONS_FIELDS))
def test_corrections_without_any_one_field_is_refused(field):
    record = _example("iteration", 2)
    del record["verdict"]["corrections"][field]
    assert _errors("iteration", record), f"corrections accepted without {field}"


@pytest.mark.parametrize("field", sorted(HELD_OUT_FIELDS))
def test_held_out_without_any_one_required_field_is_refused(field):
    record = _example("iteration", 2)
    del record["verdict"]["held_out"][field]
    assert _errors("iteration", record), f"held_out accepted without {field}"


@pytest.mark.parametrize("field", sorted(SCORING_FIELDS))
@pytest.mark.parametrize("where", ["corrections", "held_out"])
def test_scoring_without_any_one_field_is_refused(where, field):
    record = _example("iteration", 2)
    del record["verdict"][where]["scoring"][field]
    assert _errors("iteration", record), f"{where}.scoring accepted without {field}"


@pytest.mark.parametrize("field", sorted(STYLE_FIELDS))
def test_style_without_any_one_field_is_refused(field):
    record = _example("iteration", 2)
    record["verdict"]["held_out"]["style"] = _style(**{field: _DELETE})
    assert _errors("iteration", record), f"style accepted without {field}"


@pytest.mark.parametrize("outcome", ["pass", "fail", "insufficient", "not_applicable"])
def test_corrections_accepts_every_outcome(outcome):
    record = _example("iteration", 2)
    record["verdict"]["corrections"]["outcome"] = outcome
    assert _errors("iteration", record) == []


def test_corrections_not_applicable_when_the_training_set_held_no_edits():
    record = _example("iteration", 2)
    record["verdict"]["corrections"].update(
        {
            "fixed": 0,
            "broken": 0,
            "ties": 0,
            "n_edits": 0,
            "n_paraphrases": 0,
            "p_value": None,
            "outcome": "not_applicable",
        }
    )
    assert _errors("iteration", record) == []


def test_corrections_with_a_measured_result_validates():
    record = _example("iteration", 2)
    record["verdict"]["corrections"] = {
        "fixed": 7,
        "broken": 1,
        "ties": 2,
        "n_edits": 10,
        "n_paraphrases": 30,
        "p_value": 0.0352,
        "scoring": _scoring(),
        "outcome": "pass",
    }
    assert _errors("iteration", record) == []


@pytest.mark.parametrize("outcome", ["pass", "fail", "weak", "not_measured"])
def test_held_out_accepts_every_outcome(outcome):
    record = _example("iteration", 2)
    record["verdict"]["held_out"]["outcome"] = outcome
    assert _errors("iteration", record) == []


def test_held_out_with_a_measured_result_validates():
    record = _example("iteration", 2)
    record["verdict"]["held_out"] = {
        "better": 3,
        "worse": 2,
        "ties": 7,
        "n": 12,
        "p_value": 0.8125,
        "scoring": _scoring(),
        "outcome": "pass",
    }
    assert _errors("iteration", record) == []


def test_held_out_style_is_optional_and_never_required_by_the_outcome():
    record = _example("iteration", 2)
    record["verdict"]["held_out"].pop("style", None)
    assert _errors("iteration", record) == []
    record["verdict"]["held_out"]["style"] = _style()
    assert _errors("iteration", record) == []
    record["verdict"]["held_out"]["style"] = _style(p_value=None)
    assert _errors("iteration", record) == []
    record["verdict"]["held_out"].update({"outcome": "not_measured", "n": 0})
    assert _errors("iteration", record) == []


@pytest.mark.parametrize(
    "scoring",
    [
        {"method": "added_recall", "threshold": 0.5, "template": None},
        {"method": "reference_judge", "threshold": None, "template": _JUDGE},
        {"method": "reference_judge", "threshold": 0.7, "template": _JUDGE},
        {"method": "added_recall", "threshold": 0, "template": None},
        {"method": "added_recall", "threshold": 1, "template": None},
    ],
)
@pytest.mark.parametrize("where", ["corrections", "held_out"])
def test_scoring_accepts_both_methods_with_or_without_a_template(where, scoring):
    record = _example("iteration", 2)
    record["verdict"][where]["scoring"] = scoring
    assert _errors("iteration", record) == []


def test_evidence_counts_and_p_values_accept_their_edges():
    record = _example("iteration", 2)
    for where in ("corrections", "held_out"):
        for p_value in (0, 1, 0.5, None):
            record["verdict"][where]["p_value"] = p_value
            assert _errors("iteration", record) == [], (where, p_value)
    record["verdict"]["corrections"].update(
        {"fixed": 0, "broken": 0, "ties": 0, "n_edits": 0, "n_paraphrases": 0}
    )
    record["verdict"]["held_out"].update({"better": 0, "worse": 0, "ties": 0, "n": 0})
    assert _errors("iteration", record) == []


def test_verdict_evidence_stays_optional_next_to_a_no_verdict_decision():
    record = _example("iteration", 2)
    record["verdict"] = {"decision": "SKIPPED", "method": "ship", "reason": "nothing to compare"}
    assert _errors("iteration", record) == []


# ---------------------------------------------------------------------------
# paraphrase
# ---------------------------------------------------------------------------

PARAPHRASE_FIELDS = [
    "schema_version",
    "response_id",
    "prompt_hash",
    "paraphrases",
    "generator",
    "created_at",
]


@pytest.mark.parametrize("field", PARAPHRASE_FIELDS)
def test_paraphrase_without_any_one_required_field_is_refused(field):
    record = _example("paraphrase")
    del record[field]
    assert _errors("paraphrase", record), f"paraphrase accepted without {field}"


@pytest.mark.parametrize(
    "path",
    [
        "generator.model",
        "generator.template",
        "generator.template.id",
        "generator.template.version",
    ],
)
def test_paraphrase_generator_without_any_one_field_is_refused(path):
    record = _set(path, _DELETE)(_example("paraphrase"))
    assert _errors("paraphrase", record), f"paraphrase accepted without {path}"


def test_paraphrase_optional_fields_may_be_absent_or_present():
    plain = _example("paraphrase", 0)
    assert "dropped" not in plain
    assert _errors("paraphrase", plain) == []
    plain.update({"dropped": 0, "dropped_reasons": []})
    assert _errors("paraphrase", plain) == []
    plain.update(
        {"dropped": 2, "dropped_reasons": ["same_as_prompt", "leaks_answer", "same_as_prompt"]}
    )
    assert _errors("paraphrase", plain) == []


def test_paraphrase_that_lost_some_candidates_keeps_at_least_one():
    record = _example("paraphrase", 2)
    record["paraphrases"] = record["paraphrases"][:1]
    record["dropped"] = 3
    assert _errors("paraphrase", record) == []
    record["paraphrases"] = []
    assert _errors("paraphrase", record), "nothing survived: no record is written"


def test_paraphrase_accepts_non_ascii_text_and_long_lists():
    record = _example("paraphrase")
    record["paraphrases"] = [
        f"Comment r{chr(0xE9)}initialiser mon routeur ?",
        chr(0x8DEF) + chr(0x7531),
    ]
    record["paraphrases"] += [f"variant {number}" for number in range(50)]
    assert _errors("paraphrase", record) == []


def test_paraphrase_generator_may_carry_extra_descriptors():
    record = _example("paraphrase")
    record["generator"]["temperature"] = 0.7
    assert _errors("paraphrase", record) == []


def test_paraphrase_extension_is_kept_apart_from_the_core_fields():
    record = _example("paraphrase", 1)
    assert isinstance(record["x_zero"], dict)
    assert _errors("paraphrase", record) == []
    record["x_other_app"] = {"anything": 1}
    assert _errors("paraphrase", record) == []
    record["x_zero"] = []
    assert _errors("paraphrase", record)


# ---------------------------------------------------------------------------
# prompt_hash
# ---------------------------------------------------------------------------

HELLO = [{"role": "user", "content": "hi"}]
# sha256 of the UTF-8 bytes  [{"content":"hi","role":"user"}]
HELLO_HASH = "4e79873118cd9be7a1f0308b9cd772950c5410c74ca3fe1ba2626cba009a9237"
# sha256 of the UTF-8 bytes  [{"content":"caf<e acute>","role":"user"}]
# (the letter itself, U+00E9, not the six characters of an escape)
CAFE_HASH = "ddbb783f977bcf3b98d96a928454f041d03d8c9c43800049e04efba4fc294c1e"


def test_prompt_hash_is_sha256_of_the_sorted_compact_json():
    assert loop_formats.prompt_hash(HELLO) == HELLO_HASH
    assert HELLO_HASH == hashlib.sha256(b'[{"content":"hi","role":"user"}]').hexdigest()


def test_prompt_hash_writes_non_ascii_text_as_utf8_not_as_escapes():
    messages = [{"role": "user", "content": "caf" + chr(0xE9)}]
    assert loop_formats.prompt_hash(messages) == CAFE_HASH


def test_prompt_hash_validates_against_the_paraphrase_schema():
    record = _example("paraphrase")
    record["prompt_hash"] = loop_formats.prompt_hash(HELLO)
    assert _errors("paraphrase", record) == []
    assert len(record["prompt_hash"]) == 64


def test_prompt_hash_is_stable_for_equal_messages():
    first = loop_formats.prompt_hash(HELLO)
    assert loop_formats.prompt_hash(copy.deepcopy(HELLO)) == first
    assert loop_formats.prompt_hash(tuple(HELLO)) == first
    # key order inside a message does not matter: keys are sorted
    assert loop_formats.prompt_hash([{"content": "hi", "role": "user"}]) == first


def test_prompt_hash_changes_when_one_character_changes():
    base = [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "How do I reset my router?"},
    ]
    seen = {loop_formats.prompt_hash(base)}
    for index, message in enumerate(base):
        content = message["content"]
        for position in (0, len(content) // 2, len(content) - 1):
            edited = copy.deepcopy(base)
            replacement = "X" if content[position] != "X" else "Y"
            edited[index]["content"] = content[:position] + replacement + content[position + 1:]
            digest = loop_formats.prompt_hash(edited)
            assert digest not in seen, (index, position)
            seen.add(digest)


def test_prompt_hash_depends_on_role_order_and_length_of_the_prompt():
    system = {"role": "system", "content": "Be brief."}
    user = {"role": "user", "content": "Reset?"}
    assert loop_formats.prompt_hash([system, user]) != loop_formats.prompt_hash([user, system])
    assert loop_formats.prompt_hash([user]) != loop_formats.prompt_hash([system, user])
    as_system = {"role": "system", "content": "Reset?"}
    assert loop_formats.prompt_hash([as_system]) != loop_formats.prompt_hash([user])
    # a trailing space is a different prompt: nothing is trimmed or case-folded
    spaced = {"role": "user", "content": "Reset? "}
    lowered = {"role": "user", "content": "reset?"}
    assert loop_formats.prompt_hash([spaced]) != loop_formats.prompt_hash([user])
    assert loop_formats.prompt_hash([lowered]) != loop_formats.prompt_hash([user])


def test_prompt_hash_does_not_change_its_input():
    messages = copy.deepcopy(HELLO)
    loop_formats.prompt_hash(messages)
    assert messages == HELLO


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([], "at least one message"),
        ("How do I reset my router?", "list of messages"),
        (None, "list of messages"),
        (["How do I reset my router?"], "must be a mapping"),
        ([{"role": "user", "content": "hi"}, 3], "must be a mapping"),
        ({"role": "user", "content": "hi"}, "list of messages"),
    ],
)
def test_prompt_hash_refuses_what_is_not_a_list_of_messages(messages, message):
    with pytest.raises(ValueError, match=message):
        loop_formats.prompt_hash(messages)
