"""The judge-row JSON Schema and its example file, in the style of the loop formats.

The schema is the contract a reader (Soup Zero vendors it with a strict checker) holds the
sidecar to, so three things are pinned here: every row a real ``soup ship`` pairwise run writes
(real prompt, real reply parser, only the judge's HTTP call scripted) validates against it; the
shipped example file validates and shows every case; and the schema refuses the rows a producer is
most likely to get wrong. The scenario is the one of ``test_ship_judge_rows.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from pathlib import Path

import pytest

from soup_cli.utils import ship_judge_rows
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED
from tests.test_ship_judge_rows import PAIRWISE, read_rows, run_ship

jsonschema = pytest.importorskip("jsonschema")

TIE_REASONS = ("real_tie", "order_disagreement", "unreadable_reply")


def _validator():
    return jsonschema.Draft202012Validator(
        ship_judge_rows.load_row_schema(),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def _errors(row: dict) -> list:
    return [error.message for error in _validator().iter_errors(row)]


def _example(index: int = 0) -> dict:
    return copy.deepcopy(ship_judge_rows.load_row_examples()[index])


def _example_where(**fields) -> dict:
    for row in ship_judge_rows.load_row_examples():
        if all(row[key] == value for key, value in fields.items()):
            return copy.deepcopy(row)
    raise AssertionError(f"no example row with {fields}")


@pytest.fixture
def real_rows(tmp_path, monkeypatch):
    """Every row of one pairwise run under ``--gate loop-v2``: both legs, every kind of tie."""
    result = run_ship(
        tmp_path,
        monkeypatch,
        *PAIRWISE,
        "--gate",
        "loop-v2",
        "--held-out",
        "held.jsonl",
        "--output",
        "out.json",
    )
    assert result.exit_code == EXIT_GATE_FAILED, (result.output, repr(result.exception))
    return read_rows(tmp_path / "out.judge-rows.jsonl")


def _check_invariants(row: dict, where: str) -> None:
    """What the fields of a row must say together, written out without the implementation."""
    firsts = (row["base_first"], row["tuned_first"])
    if firsts == ("tuned", "tuned"):
        assert row["result"] == "tuned", where
    elif firsts == ("base", "base"):
        assert row["result"] == "base", where
    else:
        assert row["result"] == "tie", where
    assert (row["tie_reason"] is None) == (row["result"] != "tie"), where
    assert row["score"] == {"tuned": 1.0, "tie": 0.5, "base": 0.0}[row["result"]], where
    if row["tie_reason"] == "real_tie":
        assert firsts == ("none", "none"), where  # both orders said tie
    if row["tie_reason"] == "unreadable_reply":
        assert "none" in firsts, where  # a reply that picked nothing
    if row["tie_reason"] == "order_disagreement":
        assert firsts != ("none", "none"), where  # a pick somewhere, and not the same one twice
    expected = hashlib.sha256(row["prompt"].strip().casefold().encode("utf-8")).hexdigest()
    assert row["prompt_hash"] == expected[:12], where
    if row["id_source"] == "prompt_sha256":
        assert row["id"] == hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(), where


# ---------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------


def test_the_schema_and_the_example_ship_as_package_data():
    package = resources.files("soup_cli.schemas.ship")
    assert ship_judge_rows.row_schema_resource().is_file()
    assert ship_judge_rows.row_example_resource().is_file()
    assert (package / "judge_row.schema.json").is_file()
    assert (package / "examples" / "judge_row.jsonl").is_file()


def test_the_schema_is_a_valid_draft_2020_12_schema_with_its_own_id():
    schema = ship_judge_rows.load_row_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "urn:soup-cli:ship:judge-row:2"
    assert schema["type"] == "object"


def test_the_schema_pins_the_version_the_writer_writes_and_asks_for_it_on_every_row():
    schema = ship_judge_rows.load_row_schema()
    assert schema["properties"]["schema_version"]["const"] == ship_judge_rows.SCHEMA_VERSION
    assert "schema_version" in schema["required"]
    assert str(ship_judge_rows.SCHEMA_VERSION) in schema["$id"]


def test_the_schema_says_each_row_is_self_describing_and_what_the_file_holds():
    description = ship_judge_rows.load_row_schema()["description"]
    assert "every line carries its own `schema_version`" in description
    assert "no header line" in description
    assert "a reader that meets a higher version it does not know skips the line" in description
    assert "<FILE stem>.judge-rows.jsonl" in description
    assert "`judge_rows` field" in description
    assert "prompts and both models' answers in full" in description  # a privacy note


def test_the_prompt_hash_is_described_as_the_12_hex_key_and_not_the_64_hex_messages_hash():
    description = ship_judge_rows.load_row_schema()["properties"]["prompt_hash"]["description"]
    assert (
        "`prompt_hash` is the 12-hex key sha256(strip().casefold()) of the row's last user "
        "message, and is not the 64-hex hash of a whole messages list used elsewhere"
    ) in description
    assert ship_judge_rows.load_row_schema()["properties"]["prompt_hash"]["pattern"] == (
        "^[0-9a-f]{12}$"
    )


def test_the_schema_describes_every_tie_reason_and_the_null_of_a_row_that_is_not_a_tie():
    prop = ship_judge_rows.load_row_schema()["properties"]["tie_reason"]
    for reason in TIE_REASONS:
        assert reason in prop["description"]
    assert prop["oneOf"] == [{"enum": list(TIE_REASONS)}, {"type": "null"}]
    assert "null on a row that is not a tie" in prop["description"]


def test_the_schema_says_what_the_cap_flags_and_the_judge_model_are():
    properties = ship_judge_rows.load_row_schema()["properties"]
    cap = properties["base_truncated"]["description"]
    assert "256 new tokens" in cap and "512" in cap
    assert "never guessed from the length of the text" in cap
    assert "model part of the URL" in properties["judge_model"]["description"]


# ---------------------------------------------------------------------------
# What validates
# ---------------------------------------------------------------------------


def test_every_row_a_real_pairwise_run_writes_validates(real_rows):
    assert len(real_rows) == 8  # five task prompts and three held-out edits
    for row in real_rows:
        assert _errors(row) == [], (row["leg"], row["index"])
        _check_invariants(row, f"{row['leg']} {row['index']}")


def test_the_real_run_shows_every_leg_tie_reason_and_cap_flag(real_rows):
    assert {row["leg"] for row in real_rows} == {"task_eval", "held_out"}
    assert {row["tie_reason"] for row in real_rows} == {None, *TIE_REASONS}
    assert {row["base_truncated"] for row in real_rows} == {True, False}
    assert {row["tuned_truncated"] for row in real_rows} == {True, False}
    assert {row["id_source"] for row in real_rows} == {"edit_id", "id", "prompt_sha256"}


def test_the_example_file_is_one_json_object_per_line():
    text = ship_judge_rows.row_example_resource().read_text(encoding="utf-8")
    lines = text.splitlines()
    assert text.endswith("\n") and all(line.strip() for line in lines)
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_every_example_row_validates_and_agrees_with_itself():
    examples = ship_judge_rows.load_row_examples()
    assert len(examples) >= 6, "the example file should show every case"
    for number, row in enumerate(examples, start=1):
        assert _errors(row) == [], f"example line {number}"
        _check_invariants(row, f"example line {number}")


def test_the_examples_show_every_case_a_reader_has_to_handle():
    examples = ship_judge_rows.load_row_examples()
    assert {row["tie_reason"] for row in examples} == {None, *TIE_REASONS}
    assert {row["result"] for row in examples} == {"base", "tuned", "tie"}
    assert {row["leg"] for row in examples} == {"task_eval", "held_out"}
    assert {row["id_source"] for row in examples} == {"edit_id", "id", "prompt_sha256"}
    assert {row["tuned_truncated"] for row in examples} == {True, False}
    assert any(key.startswith("x_") for row in examples for key in row)
    assert {row["schema_version"] for row in examples} == {ship_judge_rows.SCHEMA_VERSION}
    # a prompt whose hash ignores its case and outer spaces
    assert any(row["prompt"] != row["prompt"].strip() for row in examples)


def test_the_example_rows_are_what_the_writer_writes():
    # one writer, one example: the keys are the schema's required keys, nothing more
    required = set(ship_judge_rows.load_row_schema()["required"])
    for row in ship_judge_rows.load_row_examples():
        assert required <= set(row)
        assert set(row) - required <= {"x_zero"}


def test_a_row_is_written_with_exactly_the_keys_the_schema_requires(real_rows):
    required = set(ship_judge_rows.load_row_schema()["required"])
    for row in real_rows:
        assert set(row) == required


def test_unknown_fields_and_extension_namespaces_are_accepted():
    row = _example()
    row["field_from_a_future_version"] = {"anything": [1, 2, 3]}
    row["x_zero"] = {"job_id": "job-1"}
    assert _errors(row) == []
    row["x_zero"] = "job-1"
    assert _errors(row), "x_<app> must hold an object"


def test_a_row_judged_in_one_order_and_an_unreported_cap_validate():
    row = _example_where(result="tuned")
    row["tuned_first"] = None
    row["base_truncated"] = None
    row["tuned_truncated"] = None
    assert _errors(row) == []


# ---------------------------------------------------------------------------
# What the schema refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ship_judge_rows.load_row_schema()["required"])
def test_a_row_without_any_one_required_field_is_refused(key):
    row = _example()
    del row[key]
    assert any(key in message for message in _errors(row)), key


def _tie(**changes):
    row = _example_where(tie_reason="real_tie")
    row.update(changes)
    return row


def _not_tie(**changes):
    row = _example_where(result="tuned")
    row.update(changes)
    return row


REFUSED = {
    "tie without a reason": _tie(tie_reason=None),
    "tie with an unknown reason": _tie(tie_reason="draw"),
    "tie with a reason in the wrong case": _tie(tie_reason="Real_Tie"),
    "tuned win with a reason": _not_tie(tie_reason="real_tie"),
    "base win with a reason": {**_example_where(result="base"), "tie_reason": "order_disagreement"},
    "tie scored as a win": _tie(score=1.0),
    "tie scored as a loss": _tie(score=0),
    "tuned win scored as a tie": _not_tie(score=0.5),
    "tuned win scored as a loss": _not_tie(score=0.0),
    "base win scored as a win": {**_example_where(result="base"), "score": 1.0},
    "score outside the three values": _not_tie(score=0.75),
    "result not one of three": _not_tie(result="draw"),
    "unknown leg": _not_tie(leg="train"),
    "unknown id source": _not_tie(id_source="uuid"),
    "negative index": _not_tie(index=-1),
    "index as a string": _not_tie(index="0"),
    "fractional index": _not_tie(index=0.5),
    "empty id": _not_tie(id=""),
    "id over 256 characters": _not_tie(id="x" * 257),
    "empty judge model": _not_tie(judge_model=""),
    "judge model as a number": _not_tie(judge_model=7),
    "cap flag as a string": _not_tie(base_truncated="yes"),
    "cap flag as a number": _not_tie(tuned_truncated=1),
    "pick not base, tuned or none": _not_tie(base_first="A"),
    "second pick not base, tuned or none": _not_tie(tuned_first="B"),
    "answer as a number": _not_tie(base_answer=4),
    "prompt missing its text": _not_tie(prompt=None),
    "version 1": _not_tie(schema_version=1),
    "version 3": _not_tie(schema_version=3),
    "version as a string": _not_tie(schema_version="2"),
    # the hash is the 12-hex key, not the 64-hex hash of a whole messages list
    "64-hex prompt hash": _not_tie(prompt_hash=hashlib.sha256(b"x").hexdigest()),
    "11-hex prompt hash": _not_tie(prompt_hash="0123456789a"),
    "13-hex prompt hash": _not_tie(prompt_hash="0123456789abc"),
    "upper-case prompt hash": _not_tie(prompt_hash="0123456789AB"),
    "non-hex prompt hash": _not_tie(prompt_hash="0123456789ag"),
    "prompt hash as a number": _not_tie(prompt_hash=123456789012),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_the_schema_refuses_a_row_a_producer_is_likely_to_get_wrong(case):
    assert _errors(REFUSED[case]), case


def test_each_refused_row_differs_from_a_valid_one_in_one_thing_only():
    # guard of the table above: its starting rows are valid, so a refusal is the change's doing
    assert _errors(_tie()) == [] and _errors(_not_tie()) == []
    assert _errors(_example_where(result="base")) == []


# ---------------------------------------------------------------------------
# The docs
# ---------------------------------------------------------------------------


def test_the_docs_name_the_file_the_schema_the_pointer_and_every_new_field():
    docs = (Path(__file__).resolve().parent.parent / "docs" / "evaluation.md").read_text(
        encoding="utf-8"
    )
    for needle in (
        ".judge-rows.jsonl",
        "soup_cli/schemas/ship/judge_row.schema.json",
        "soup_cli/schemas/ship/examples/judge_row.jsonl",
        "judge_rows",
        "tie_reason",
        "base_truncated",
        "tuned_truncated",
        "judge_model",
        "prompt_hash",
        *TIE_REASONS,
    ):
        assert needle in docs, f"docs/evaluation.md does not mention {needle}"
