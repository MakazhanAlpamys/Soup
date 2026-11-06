"""Paraphrases of the prompts of corrected answers, which the loop's verdict asks the new version.

The loop's verdict ("loop-v2") checks whether a new model version learned the corrections people
made by asking it paraphrases of each corrected prompt. This PR creates and stores them:

- ``loop_paraphrase.paraphrase_edit`` asks a chat endpoint for two paraphrases of an edit's prompt
  (its last user message), drops one that is the prompt again (``same_as_prompt``) or holds the
  words the correction added (``leaks_answer``), asks once more when fewer than two survive, and
  returns the ``paraphrase`` record of docs/loop-formats.md, or ``None`` (then no line is written);
- the pre-wired train stage writes the records for every edit it trains on that has none yet, before
  training, and an edit is asked once: later iterations reuse its record;
- the paraphraser is ``loop.yaml``'s ``paraphraser_endpoint`` / ``paraphraser_model`` (``soup loop
  init --paraphraser-endpoint / --paraphraser-model``), else the loop's own serve endpoint, which
  answers the call without recording it when it carries ``X-Soup-Internal: loop-check`` and the
  tool token. Paraphrases are never training data.

The method is Soup Zero's (``zero_server/loop/corrections.py``); the filter cases below are its.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_paraphrase as lp
import soup_cli.utils.loop_stages as ls
from soup_cli.utils import loop_formats
from soup_cli.utils.loop_feedback import EditedAnswer
from soup_cli.utils.loop_state import LoopState, read_state, write_state

jsonschema = pytest.importorskip("jsonschema")

NOW = "2026-11-06T09:00:00+00:00"
PROMPT = "How do I reset my router?"
EDIT_TEXT = "Hold the reset button for ten seconds, then reconnect to the default network."
ORIGINAL = "Unplug it and plug it back in."
GOOD = "1. What steps reset my router?\n2. My router needs a reset, what do I do?"
GOOD_LIST = ["What steps reset my router?", "My router needs a reset, what do I do?"]
ENDPOINT = "http://127.0.0.1:8000"
TOKEN = "para-" + "t" * 27
PARAPHRASES_FILE = os.path.join(".soup-loops", "paraphrases.jsonl")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def make_edit(prompt=PROMPT, edited=EDIT_TEXT, original=ORIGINAL, system=None) -> EditedAnswer:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return EditedAnswer(str(uuid.uuid4()), tuple(messages), original, edited)


def added_of(edit: EditedAnswer) -> frozenset:
    return lp.added_words(edit.edited, edit.original)


class FakeChat:
    """A chat endpoint: ``replies`` in order (empty text once they run out), or a function."""

    def __init__(self, replies=None, *, error=None):
        self.replies = replies if replies is not None else [GOOD]
        self.error = error
        self.bodies: list = []

    def __call__(self, body):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        if callable(self.replies):
            return self.replies(body)
        index = len(self.bodies) - 1
        return self.replies[index] if index < len(self.replies) else ""


def schema_errors(record: dict) -> list:
    validator = jsonschema.Draft202012Validator(
        loop_formats.load_schema("paraphrase"),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )
    return [error.message for error in validator.iter_errors(record)]


def read_lines(path=PARAPHRASES_FILE) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG", ls.TOOL_AUTH_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    # (the free-VRAM measurement of quantization "auto" adds a note of its own)
    monkeypatch.setattr(ls, "resolve_sft_quantization", lambda requested, base: ("none", None))
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


# ---------------------------------------------------------------------------
# The template and the request
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_the_template_is_loop_paraphrase_version_1(self):
        assert dict(lp.TEMPLATE) == {"id": "loop-paraphrase", "version": "1"}

    def test_the_template_text_is_soup_zeros(self):
        assert lp.TEMPLATE_TEXT == (
            "Rewrite the question below in {n} different ways. Keep its meaning exactly and do "
            "not answer it. Reply with a numbered list only, one rewrite per line, like:\n"
            "1. ...\n2. ...\n\nQuestion: {question}"
        )

    def test_one_request_asks_for_two_paraphrases_as_a_numbered_list_at_temperature_07(self):
        chat = FakeChat([GOOD])
        edit = make_edit()

        found = lp.paraphrase(chat, "m", PROMPT, added_of(edit))

        assert found.kept == GOOD_LIST and found.dropped == 0 and found.reasons == []
        assert len(chat.bodies) == 1  # one request for both
        body = chat.bodies[0]
        assert body["model"] == "m" and body["temperature"] == 0.7 and body["stream"] is False
        assert body["max_tokens"] == lp.MAX_REPLY_TOKENS
        (message,) = body["messages"]
        assert message["role"] == "user"
        assert "2 different ways" in message["content"] and "numbered list" in message["content"]
        assert message["content"].endswith("Question: " + PROMPT)

    def test_the_question_is_not_read_as_a_format_string(self):
        body = lp.paraphrase_body("m", "What does {n} mean in {0}? {question}")

        text = body["messages"][0]["content"]
        assert text.endswith("Question: What does {n} mean in {0}? {question}")

    def test_two_paraphrases_per_edit_one_retry_and_two_attempts(self):
        assert (lp.PARAPHRASES_WANTED, lp.PARAPHRASE_ATTEMPTS) == (2, 2)

    def test_the_internal_header_strings_are_the_ones_soup_serve_honours(self):
        import soup_cli.commands.serve as serve

        assert (lp.INTERNAL_HEADER, lp.INTERNAL_LOOP_CHECK) == (
            serve._INTERNAL_HEADER, serve._INTERNAL_LOOP_CHECK
        )
        assert (lp.INTERNAL_HEADER, lp.INTERNAL_LOOP_CHECK) == ("X-Soup-Internal", "loop-check")


# ---------------------------------------------------------------------------
# Reading a reply, comparing questions
# ---------------------------------------------------------------------------


class TestParsing:
    def test_a_numbered_list_is_read_item_by_item_and_the_talk_around_it_is_ignored(self):
        text = (
            "Sure! Here you go:\n1. How can I reset my router?\n"
            '2) "What is the way to reset a router?"\n\n- Router reset: how?\nHope that helps.'
        )

        assert lp.parse_numbered_list(text) == [
            "How can I reset my router?", "What is the way to reset a router?",
            "Router reset: how?",
        ]
        assert lp.parse_numbered_list("") == [] and lp.parse_numbered_list("no list here") == []

    @pytest.mark.parametrize(
        "same",
        [
            "How do I reset my router", "How  do I reset my router ?",
            "how do i reset\tmy router?!", "  How do I reset my router?\n",
            "HOW, DO I RESET MY ROUTER!!", "“How do I reset my router?”",
        ],
    )
    def test_spacing_case_and_punctuation_alone_do_not_make_another_question(self, same):
        assert lp.paraphrase_key(same) == lp.paraphrase_key(PROMPT)

    @pytest.mark.parametrize("other", ["How can I reset my router?", "Reset my router: how?"])
    def test_a_real_rewording_is_another_question(self, other):
        assert lp.paraphrase_key(other) != lp.paraphrase_key(PROMPT)

    def test_symbols_are_punctuation_too(self):
        assert lp.paraphrase_key("Cost in $ + tax?") == lp.paraphrase_key("cost in   tax")

    def test_the_thinking_of_a_reasoning_model_is_not_read_as_a_paraphrase(self):
        assert lp.strip_think("<think>1. A draft</think>1. Real one") == "1. Real one"
        assert lp.strip_think("1. A draft</think>1. Real one") == "1. Real one"
        assert lp.strip_think("1. Real one<think>1. cut off") == "1. Real one"
        assert lp.strip_think("") == ""


class TestAddedWords:
    def test_the_added_words_are_what_the_edit_has_and_the_original_lacked(self):
        added = lp.added_words(EDIT_TEXT, ORIGINAL)

        assert {"hold", "reset", "button", "ten", "seconds"} <= added
        assert "the" not in added and "then" not in added  # stop words
        assert "plug" not in added  # only in the original

    def test_an_edit_that_adds_fewer_than_two_words_is_scored_on_all_its_words(self):
        assert lp.added_words("Paris is the capital", "Paris is the capital city") == frozenset(
            {"paris", "capital"}
        )

    def test_negations_and_numbers_are_never_stop_words(self):
        assert {"not", "no", "never", "42"} <= lp.content_words("not no never 42")
        assert lp.content_words("The cat is on a mat") == frozenset({"cat", "mat"})

    def test_half_of_the_words_in_the_text_is_a_hit(self):
        words = frozenset({"hold", "reset", "button", "ten"})

        assert lp.is_hit("hold the reset", words) is True  # 2 of 4 is exactly half
        assert lp.is_hit("hold the reset", frozenset({"hold", "reset", "button"})) is True
        assert lp.is_hit("hold", words) is False  # 1 of 4
        assert lp.is_hit("hold", frozenset({"hold", "reset", "button"})) is False  # 1 of 3
        assert lp.is_hit("<think>hold reset</think>fine", words) is False  # thinking is removed
        assert lp.is_hit("anything at all", frozenset()) is False  # nothing to find


# ---------------------------------------------------------------------------
# The sanity filter, with Soup Zero's cases
# ---------------------------------------------------------------------------


class TestSanityFilter:
    def test_a_paraphrase_that_normalises_to_the_prompt_is_dropped_as_same_as_prompt(self):
        edit = make_edit()
        first = (
            "1.   how do i reset my router?  \n2. What steps reset my router?\n"
            "3. HOW DO I RESET MY ROUTER?"
        )
        second = "1. My router needs a reset, what do I do?"  # one survived: asked once more

        found = lp.paraphrase(FakeChat([first, second]), "m", PROMPT, added_of(edit))

        assert found.kept == GOOD_LIST
        assert found.dropped == 2 and found.reasons == ["same_as_prompt"]
        assert found.counts == {"same_as_prompt": 2}

    def test_spacing_and_punctuation_alone_do_not_make_a_paraphrase_different(self):
        reply = (
            "1. How do I reset my router\n2. How  do I reset my router ?\n"
            "3. HOW, DO I RESET MY ROUTER!!\n4. “How do I reset my router?”\n"
            "5. What steps reset my router?\n6. My router needs a reset, what do I do?"
        )
        chat = FakeChat([reply])

        found = lp.paraphrase(chat, "m", PROMPT, frozenset())

        assert found.kept == GOOD_LIST
        assert found.dropped == 4 and found.reasons == ["same_as_prompt"]
        assert len(chat.bodies) == 1  # two real rewordings survived: no second request

    def test_a_real_rewording_passes(self):
        reply = "1. How can I reset my router?\n2. Router reset: how do I do it?"

        found = lp.paraphrase(FakeChat([reply]), "m", PROMPT, frozenset())

        assert found.kept == ["How can I reset my router?", "Router reset: how do I do it?"]
        assert found.dropped == 0

    def test_a_paraphrase_with_half_of_the_added_words_is_dropped_as_leaking_the_answer(self):
        edit = make_edit()
        added = added_of(edit)
        leaky = "How do I hold the reset button for ten seconds on my router?"
        assert lp.is_hit(leaky, added)
        reply = (
            f"1. {leaky}\n2. What steps reset my router?\n"
            "3. My router needs a reset, what do I do?"
        )

        found = lp.paraphrase(FakeChat([reply]), "m", PROMPT, added)

        assert found.kept == GOOD_LIST
        assert found.dropped == 1 and found.reasons == ["leaks_answer"]
        assert found.counts == {"leaks_answer": 1}
        assert not lp.is_hit("What steps reset my router?", added)  # a fair rewording is kept

    def test_the_two_reasons_are_counted_apart_and_named_in_the_order_they_first_happened(self):
        edit = make_edit()
        reply = (
            "1. How do I hold the reset button for ten seconds?\n2. how do i reset my router\n"
            "3. Hold the reset button ten seconds, how?\n4. What steps reset my router?\n"
            "5. My router needs a reset, what do I do?"
        )

        found = lp.paraphrase(FakeChat([reply]), "m", PROMPT, added_of(edit))

        assert found.kept == GOOD_LIST
        assert found.dropped == 3
        assert found.reasons == ["leaks_answer", "same_as_prompt"]
        assert found.counts == {"leaks_answer": 2, "same_as_prompt": 1}

    def test_a_second_request_is_made_once_when_fewer_than_two_survive_and_never_a_third(self):
        edit = make_edit()
        chat = FakeChat([
            "1. How do I reset my router?\n2. What steps reset my router?",
            "1. My router needs a reset, what do I do?\n2. How do I reset my router?",
        ])

        found = lp.paraphrase(chat, "m", PROMPT, added_of(edit))

        assert len(chat.bodies) == 2 and chat.bodies[0] == chat.bodies[1]  # the same request
        assert found.kept == GOOD_LIST and found.dropped == 1  # it stops at two

        nothing = FakeChat(["1. How do I reset my router?", "nope", "never asked"])
        assert lp.paraphrase(nothing, "m", PROMPT, added_of(edit)).kept == []
        assert len(nothing.bodies) == 2

    def test_two_survivors_need_no_second_request_and_no_more_than_two_are_kept(self):
        chat = FakeChat(["1. What steps reset my router?\n2. My router needs a reset, what do I do?"
                         "\n3. A third one."])

        found = lp.paraphrase(chat, "m", PROMPT, frozenset())

        assert len(found.kept) == 2 and len(chat.bodies) == 1

    def test_a_rewording_said_twice_is_one_rewording_and_is_not_counted_as_dropped(self):
        reply = (
            "<think>1. A draft I am not sure of</think>1. What steps reset my router?\n"
            "2. what steps reset my router?"
        )

        found = lp.paraphrase(FakeChat([reply, "1. What steps reset my router?"]), "m", PROMPT,
                              frozenset())

        assert found.kept == ["What steps reset my router?"] and found.dropped == 0

    def test_a_paraphraser_that_cannot_be_asked_raises(self):
        chat = FakeChat(error=lp.ParaphraserUnavailableError("down"))

        with pytest.raises(lp.ParaphraserUnavailableError, match="down"):
            lp.paraphrase(chat, "m", PROMPT, frozenset())


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_the_record_validates_against_the_paraphrase_schema(self):
        edit = make_edit()

        record = lp.paraphrase_edit(edit, FakeChat([GOOD]), model="qwen2.5:7b", created_at=NOW)

        assert schema_errors(record) == []
        assert record["schema_version"] == 1 and record["response_id"] == edit.id
        assert record["paraphrases"] == GOOD_LIST and record["created_at"] == NOW
        assert record["generator"] == {
            "model": "qwen2.5:7b", "template": {"id": "loop-paraphrase", "version": "1"},
        }
        assert "dropped" not in record and "dropped_reasons" not in record

    def test_the_prompt_hash_is_the_hash_of_all_the_messages_of_the_prompt(self):
        edit = make_edit(system="Be brief.")

        record = lp.paraphrase_edit(edit, FakeChat([GOOD]), model="m", created_at=NOW)

        assert record["prompt_hash"] == loop_formats.prompt_hash(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": PROMPT}]
        )
        assert record["prompt_hash"] != loop_formats.prompt_hash(
            [{"role": "user", "content": PROMPT}]
        )

    def test_dropped_paraphrases_are_counted_and_explained_in_the_record(self):
        reply = (
            "1. how do i reset my router?\n2. What steps reset my router?\n"
            "3. My router needs a reset, what do I do?"
        )

        record = lp.paraphrase_edit(make_edit(), FakeChat([reply]), model="m", created_at=NOW)

        assert record["paraphrases"] == GOOD_LIST
        assert record["dropped"] == 1 and record["dropped_reasons"] == ["same_as_prompt"]
        assert schema_errors(record) == []

    def test_both_reasons_validate(self):
        reply = (
            "1. How do I hold the reset button for ten seconds?\n2. how do i reset my router\n"
            "3. What steps reset my router?\n4. My router needs a reset, what do I do?"
        )

        record = lp.paraphrase_edit(make_edit(), FakeChat([reply]), model="m", created_at=NOW)

        assert record["dropped"] == 2
        assert record["dropped_reasons"] == ["leaks_answer", "same_as_prompt"]
        assert schema_errors(record) == []

    def test_when_nothing_survives_it_returns_none(self):
        run = lp.ParaphraseRun()

        record = lp.paraphrase_edit(
            make_edit(), FakeChat(["1. How do I reset my router?"]), model="m", run=run
        )

        assert record is None
        assert (run.asked, run.empty, run.dropped, run.counts) == (
            1, 1, 1, {"same_as_prompt": 1},
        )

    def test_a_prompt_with_several_turns_is_paraphrased_by_its_last_user_message(self):
        messages = (
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": PROMPT},
        )
        edit = EditedAnswer(str(uuid.uuid4()), messages, ORIGINAL, EDIT_TEXT)
        chat = FakeChat([GOOD])

        record = lp.paraphrase_edit(edit, chat, model="m", created_at=NOW)

        assert chat.bodies[0]["messages"][0]["content"].endswith("Question: " + PROMPT)
        assert record["prompt_hash"] == loop_formats.prompt_hash(messages)

    def test_a_prompt_without_a_user_message_is_not_asked(self):
        edit = EditedAnswer(
            str(uuid.uuid4()), ({"role": "system", "content": "Be brief."},), ORIGINAL, EDIT_TEXT
        )
        chat, run = FakeChat([GOOD]), lp.ParaphraseRun()

        assert lp.paraphrase_edit(edit, chat, model="m", run=run) is None
        assert chat.bodies == [] and (run.unusable, run.asked) == (1, 0)

    def test_the_last_user_message_helper(self):
        assert lp.last_user_message([{"role": "user", "content": "a"}]) == "a"
        assert lp.last_user_message([{"role": "assistant", "content": "a"}]) is None
        assert lp.last_user_message([{"role": "user", "content": "  "}]) is None


# ---------------------------------------------------------------------------
# Reading paraphrases.jsonl
# ---------------------------------------------------------------------------


def row(edit: EditedAnswer, paraphrases=None, **fields) -> dict:
    line = lp.paraphrase_edit(edit, FakeChat([GOOD]), model="m", created_at=NOW)
    if paraphrases is not None:
        line["paraphrases"] = paraphrases
    line.update(fields)
    return line


def write_jsonl(path, lines) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        for line in lines:
            data = line if isinstance(line, bytes) else json.dumps(line).encode("utf-8")
            handle.write(data + b"\n")


class TestReader:
    def test_a_missing_file_is_an_empty_log(self, cwd):
        log = lp.read_paraphrases()

        assert log.latest == {} and log.stats.skipped == 0 and log.stats.lines == 0

    def test_the_last_line_per_response_id_wins(self, cwd):
        edit, other = make_edit(), make_edit("Where is Paris?", "In France, on the Seine.", "no")
        write_jsonl(PARAPHRASES_FILE, [
            row(edit, ["old one", "old two"]), row(other), row(edit, ["new one"]),
        ])

        log = lp.read_paraphrases()

        assert log.latest[edit.id]["paraphrases"] == ["new one"]
        assert set(log.latest) == {edit.id, other.id}
        assert log.stats.skipped == 0

    def test_bad_lines_are_skipped_and_counted_and_the_good_ones_still_count(self, cwd):
        edit = make_edit()
        good = row(edit)
        write_jsonl(PARAPHRASES_FILE, [
            b"not json at all",
            b'["a list"]',
            {**good, "schema_version": 2},
            {**good, "schema_version": True},
            {**good, "response_id": "not-a-uuid"},
            {**good, "prompt_hash": "ABC"},
            {**good, "paraphrases": []},
            {**good, "paraphrases": ["fine", ""]},
            {**good, "paraphrases": "a string"},
            {**good, "generator": {"model": "m"}},
            {**good, "generator": {"model": "", "template": {"id": "i", "version": "1"}}},
            {k: v for k, v in good.items() if k != "created_at"},
            b"\xff\xfe not utf-8",
            good,
        ])

        log = lp.read_paraphrases()

        assert log.stats.skipped == 13 and log.stats.lines == 14
        assert list(log.latest) == [edit.id]

    def test_a_bad_last_line_leaves_the_earlier_good_one_in_force(self, cwd):
        edit = make_edit()
        write_jsonl(PARAPHRASES_FILE, [row(edit), {**row(edit), "paraphrases": []}])

        assert lp.read_paraphrases().latest[edit.id]["paraphrases"] == GOOD_LIST

    def test_unknown_fields_and_extensions_are_ignored(self, cwd):
        edit = make_edit()
        write_jsonl(PARAPHRASES_FILE, [{**row(edit), "x_zero": {"project_id": "demo"}, "extra": 1}])

        assert edit.id in lp.read_paraphrases().latest

    def test_a_line_over_the_cap_is_skipped_without_being_kept(self, cwd):
        edit, other = make_edit(), make_edit("Another question?", "Another correction text.", "x")
        huge = row(edit, ["y" * (1024 * 1024 + 10)])
        write_jsonl(PARAPHRASES_FILE, [row(other), huge, row(edit)])

        log = lp.read_paraphrases()

        assert log.stats.skipped == 1
        assert log.latest[edit.id]["paraphrases"] == GOOD_LIST  # the line after it still read
        assert other.id in log.latest

    def test_a_record_is_current_only_for_the_prompt_it_was_written_for(self, cwd):
        edit = make_edit()
        write_jsonl(PARAPHRASES_FILE, [row(edit)])
        log = lp.read_paraphrases()
        changed = EditedAnswer(
            edit.id, ({"role": "system", "content": "Be brief."}, *edit.messages),
            edit.original, edit.edited,
        )

        assert lp.current_paraphrases(log, [edit]) == {edit.id: GOOD_LIST}
        assert lp.current_paraphrases(log, [changed]) == {}
        assert lp.is_current(None, edit) is False

    def test_the_file_the_reader_and_the_writer_share_is_the_contract_layout(self):
        assert lp.PARAPHRASE_LOG == loop_formats.DEFAULT_LAYOUT["paraphrase"]
        assert lp.PARAPHRASE_LOG == ".soup-loops/paraphrases.jsonl"


# ---------------------------------------------------------------------------
# Written once, reused
# ---------------------------------------------------------------------------


def run_write(edits, chat, **kwargs):
    return lp.write_paraphrases(edits, chat, model="m", now=lambda: NOW, **kwargs)


class TestWriteOnce:
    def test_every_edit_gets_one_line_in_the_contract_format(self, cwd):
        edits = [make_edit(), make_edit("Where is Paris?", "In France, on the Seine.", "no")]
        chat = FakeChat(lambda body: GOOD if "router" in body["messages"][0]["content"]
                        else "1. Which city is Paris?\n2. Tell me where Paris lies.")

        run = run_write(edits, chat)

        assert (run.written, run.kept, run.reused, run.asked, run.stopped) == (2, 4, 0, 2, None)
        lines = read_lines()
        assert [line["response_id"] for line in lines] == [edit.id for edit in edits]
        assert all(schema_errors(line) == [] for line in lines)
        with open(PARAPHRASES_FILE, "rb") as handle:
            raw = handle.read()
        assert raw.endswith(b"\n") and raw.count(b"\n") == 2  # one line per record

    def test_the_lines_are_appended_with_the_shared_writer(self, cwd, monkeypatch):
        seen = []
        real = lp.append_json_line

        def spy(path, record):
            seen.append(path)
            real(path, record)

        monkeypatch.setattr(lp, "append_json_line", spy)

        run_write([make_edit(), make_edit("Another question?", "Another correction text.", "x")],
                  FakeChat(lambda body: GOOD))

        assert seen == [lp.PARAPHRASE_LOG] * 2

    def test_a_line_another_program_left_torn_does_not_swallow_the_new_one(self, cwd):
        os.makedirs(".soup-loops")
        with open(PARAPHRASES_FILE, "wb") as handle:
            handle.write(b'{"torn": ')

        run_write([make_edit()], FakeChat([GOOD]))

        log = lp.read_paraphrases()
        assert len(log.latest) == 1 and log.stats.skipped == 1

    def test_an_edit_is_asked_once_and_reused_by_the_next_iteration(self, cwd):
        edits = [make_edit(), make_edit("Where is Paris?", "In France, on the Seine.", "no")]
        first = FakeChat(lambda body: GOOD)
        run_write(edits, first)
        assert len(first.bodies) == 2

        again = FakeChat([GOOD])
        run = run_write(edits, again)

        assert again.bodies == []
        assert (run.reused, run.edits, run.written, run.asked) == (2, 0, 0, 0)
        assert len(read_lines()) == 2

        newer = make_edit("A brand new question?", "Some new correction text here.", "old")
        third = FakeChat(["1. A first rewording of it?\n2. Another rewording of it?"])
        run = run_write([*edits, newer], third)
        assert len(third.bodies) == 1 and (run.reused, run.written) == (2, 1)  # only the new one
        assert len(read_lines()) == 3

    def test_a_record_written_for_another_prompt_is_written_again_and_the_newest_wins(self, cwd):
        edit = make_edit()
        run_write([edit], FakeChat([GOOD]))
        changed = EditedAnswer(
            edit.id, ({"role": "system", "content": "Be brief."}, *edit.messages),
            edit.original, edit.edited,
        )

        rewrite = FakeChat(["1. Router reset steps?\n2. Resetting my router: how?"])
        run = run_write([changed], rewrite)

        assert run.written == 1
        assert lp.current_paraphrases(lp.read_paraphrases(), [changed]) == {
            edit.id: ["Router reset steps?", "Resetting my router: how?"],
        }
        assert lp.current_paraphrases(lp.read_paraphrases(), [edit]) == {}

    def test_when_no_paraphrase_survives_no_line_is_written(self, cwd):
        run = run_write([make_edit()], FakeChat(["1. How do I reset my router?"]))

        assert (run.written, run.empty, run.dropped) == (0, 1, 1)
        assert not os.path.exists(PARAPHRASES_FILE)

    def test_the_first_failure_to_ask_ends_the_pass_and_says_why(self, cwd):
        edits = [make_edit(), make_edit("Another question here?", "Another corrected text.", "x")]
        chat = FakeChat(error=lp.ParaphraserUnavailableError("no response (ConnectError)"))

        run = run_write(edits, chat)

        assert len(chat.bodies) == 1  # the endpoint is down for the rest as well
        assert run.written == 0 and run.stopped == "no response (ConnectError)"
        assert not os.path.exists(PARAPHRASES_FILE)

    def test_a_file_that_cannot_be_written_ends_the_pass_without_raising(self, cwd, monkeypatch):
        def refuse(path, record):
            raise OSError("disk full")

        monkeypatch.setattr(lp, "append_json_line", refuse)
        edits = [make_edit(), make_edit("Another question here?", "Another corrected text.", "x")]
        chat = FakeChat(lambda body: GOOD)

        run = run_write(edits, chat)

        assert len(chat.bodies) == 1  # the file will not take the next one either
        assert run.written == 0 and run.kept == 0
        assert run.stopped == "paraphrases.jsonl not written (OSError)"

    def test_an_edit_with_no_user_message_is_skipped_not_fatal(self, cwd):
        broken = EditedAnswer(str(uuid.uuid4()), ({"role": "system", "content": "x"},), "a", "b")

        run = run_write([broken, make_edit()], FakeChat([GOOD]))

        assert (run.written, run.unusable) == (1, 1)


# ---------------------------------------------------------------------------
# The notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_the_summary_names_new_paraphrases_edits_and_drops(self):
        run = lp.ParaphraseRun(
            edits=6, asked=6, written=6, kept=12, dropped=3,
            counts={"leaks_answer": 2, "same_as_prompt": 1},
        )

        assert lp.run_notes(run) == [
            "paraphrases: 12 new for 6 edits, 3 dropped (same_as_prompt 1, leaks_answer 2)"
        ]

    def test_a_single_edit_and_no_drops(self):
        run = lp.ParaphraseRun(edits=1, asked=1, written=1, kept=2)

        assert lp.run_notes(run) == ["paraphrases: 2 new for 1 edit"]

    def test_edits_left_without_paraphrases_are_counted(self):
        run = lp.ParaphraseRun(edits=3, asked=2, written=1, kept=2, empty=1, unusable=1)

        assert lp.run_notes(run) == [
            "paraphrases: 2 new for 1 edit, 2 edits left without paraphrases"
        ]

    def test_a_pass_that_stopped_says_why_and_how_many_edits_it_did_not_ask(self):
        run = lp.ParaphraseRun(
            edits=5, asked=1, written=1, kept=2, stopped="no response (ConnectError)"
        )

        (note,) = lp.run_notes(run)

        assert note == (
            "paraphrases: 2 new for 1 edit. Stopped, 4 edits not asked: no response (ConnectError)"
        )

    def test_nothing_to_do_is_no_note(self):
        assert lp.run_notes(lp.ParaphraseRun(reused=4)) == []

    def test_more_than_half_dropped_adds_the_warning(self):
        # 3 edits asked for 6 paraphrases: 4 dropped is more than half
        run = lp.ParaphraseRun(
            edits=3, asked=3, written=3, kept=6, dropped=4, counts={"leaks_answer": 4}
        )

        notes = lp.run_notes(run)

        assert notes[-1] == (
            "paraphrases mostly dropped: a small served model paraphrases poorly; "
            "set soup loop init --paraphraser-model"
        )
        assert len(notes) == 2 and notes[-1] == lp.MOSTLY_DROPPED

    def test_exactly_half_dropped_is_not_mostly(self):
        run = lp.ParaphraseRun(
            edits=3, asked=3, written=3, kept=6, dropped=3, counts={"leaks_answer": 3}
        )

        assert lp.MOSTLY_DROPPED not in lp.run_notes(run)

    def test_a_pass_through_a_poor_paraphraser_warns_and_a_good_one_does_not(self, cwd):
        poor = lp.write_paraphrases(
            [make_edit()], FakeChat(["1. How do I reset my router?\n2. how do i reset my router"]),
            model="m", now=lambda: NOW,
        )
        good = lp.write_paraphrases([make_edit()], FakeChat([GOOD]), model="m", now=lambda: NOW)

        assert lp.MOSTLY_DROPPED in lp.run_notes(poor)
        assert lp.MOSTLY_DROPPED not in lp.run_notes(good)


# ---------------------------------------------------------------------------
# Who answers: the endpoint and its validation
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000", "http://localhost:11434", "http://[::1]:8000",
            "https://192.168.1.20:8080", "https://10.0.0.5:8000/v1", "http://127.0.0.1:8000/",
        ],
    )
    def test_loopback_and_lan_addresses_are_accepted(self, url):
        assert lp.checked_endpoint(url) == url.rstrip("/")

    @pytest.mark.parametrize(
        ("url", "message"),
        [
            ("https://api.example.com/v1", "loopback/LAN"),
            ("http://example.com", "HTTPS"),
            ("https://8.8.8.8", "loopback/LAN"),
            ("ftp://127.0.0.1:8000", "http/https"),
            ("file:///etc/passwd", "http/https"),
            ("http://user:secret@127.0.0.1:8000", "credentials"),
            ("http://127.0.0.1:8000/?key=1", "query"),
            ("http://0.0.0.0:8000", "0.0.0.0"),
            ("http://192.168.1.20:8080", "HTTPS"),  # plain http only to this machine
            ("http://169.254.169.254", "HTTPS"),
            ("", "non-empty"),
            ("http://127.0.0.1:8000\n", "control characters"),
        ],
    )
    def test_anything_else_is_refused_with_the_reason(self, url, message):
        with pytest.raises(ValueError, match=message):
            lp.checked_endpoint(url, "paraphraser_endpoint")

    def test_the_refusal_names_the_setting_and_never_the_secret(self):
        with pytest.raises(ValueError) as caught:
            lp.checked_endpoint("http://user:secret@127.0.0.1:8000", "paraphraser_endpoint")

        assert "paraphraser_endpoint" in str(caught.value) and "secret" not in str(caught.value)

    def test_a_non_string_is_refused(self):
        with pytest.raises(ValueError):
            lp.checked_endpoint(8000)

    @pytest.mark.parametrize(
        ("base", "url"),
        [
            ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/chat/completions"),
            ("http://127.0.0.1:8000/", "http://127.0.0.1:8000/v1/chat/completions"),
            ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/chat/completions"),
            ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/v1/chat/completions"),
            ("https://10.0.0.5:8000/api", "https://10.0.0.5:8000/api/v1/chat/completions"),
        ],
    )
    def test_the_chat_url_follows_the_openai_base_url_convention(self, base, url):
        assert lp.chat_url(base) == url


class TestResolve:
    def test_the_endpoint_in_loop_yaml_wins(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        state = _state(
            paraphraser_endpoint="http://127.0.0.1:11434", paraphraser_model="qwen2.5:7b"
        )

        paraphraser, why = lp.resolve_paraphraser(state)

        assert why == ""
        assert paraphraser == lp.Paraphraser("http://127.0.0.1:11434", "qwen2.5:7b", internal=False)

    def test_without_one_the_loops_own_serve_endpoint_is_used_the_internal_way(
        self, cwd, monkeypatch
    ):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

        paraphraser, why = lp.resolve_paraphraser(_state())

        assert why == "" and paraphraser == lp.Paraphraser(ENDPOINT, "./model", internal=True)

    def test_the_model_recorded_for_the_serve_endpoint_is_the_stable_side(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

        paraphraser, _ = lp.resolve_paraphraser(_state(production_name="loop-train-ab12"))

        assert paraphraser.model == "loop-train-ab12"

    def test_neither_reachable_is_no_paraphraser_and_says_why(self, cwd):
        paraphraser, why = lp.resolve_paraphraser(_state())

        assert paraphraser is None
        assert "no paraphraser_endpoint" in why and "SOUP_LOOP_SERVE_ENDPOINT" in why

    def test_the_serve_endpoint_needs_the_tool_token(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)

        paraphraser, why = lp.resolve_paraphraser(_state())

        assert paraphraser is None and ls.TOOL_AUTH_TOKEN_ENV in why and "tool token" in why

    def test_a_serve_endpoint_that_is_not_local_is_refused(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://serve.example.com")
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

        paraphraser, why = lp.resolve_paraphraser(_state())

        assert paraphraser is None and "loopback/LAN" in why

    def test_a_hand_edited_loop_yaml_with_a_public_endpoint_is_refused_at_use(self, cwd):
        state = _state(paraphraser_endpoint="https://api.example.com", paraphraser_model="m")

        paraphraser, why = lp.resolve_paraphraser(state)

        assert paraphraser is None and "paraphraser_endpoint" in why and "loopback/LAN" in why

    @pytest.mark.parametrize(
        "fields",
        [{"paraphraser_endpoint": "http://127.0.0.1:11434"}, {"paraphraser_model": "m"}],
    )
    def test_half_a_pair_in_loop_yaml_is_no_paraphraser(self, cwd, monkeypatch, fields):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

        paraphraser, why = lp.resolve_paraphraser(_state(**fields))

        assert paraphraser is None and "only one of paraphraser_endpoint" in why

    @pytest.mark.parametrize("named", ["http://127.0.0.1:8000", "http://localhost:8000/",
                                       "http://[::1]:8000"])
    def test_the_loops_own_server_named_as_the_paraphraser_is_still_called_the_internal_way(
        self, cwd, monkeypatch, named
    ):
        # Its answers would otherwise be logged, and could come back as training data.
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        state = _state(paraphraser_endpoint=named, paraphraser_model="whatever")

        paraphraser, _ = lp.resolve_paraphraser(state)

        assert paraphraser.internal is True and paraphraser.endpoint == ENDPOINT
        assert paraphraser.model == "whatever"

    def test_the_loops_own_server_named_without_a_token_is_no_paraphraser(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        state = _state(paraphraser_endpoint=ENDPOINT, paraphraser_model="m")

        paraphraser, why = lp.resolve_paraphraser(state)

        assert paraphraser is None and "tool token" in why

    def test_another_server_is_not_called_the_internal_way(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        state = _state(paraphraser_endpoint="http://127.0.0.1:8001", paraphraser_model="m")

        paraphraser, _ = lp.resolve_paraphraser(state)

        assert paraphraser.internal is False

    def test_same_service_compares_scheme_host_and_port(self):
        assert lp._same_service("http://localhost:8000", "http://127.0.0.1:8000")
        assert lp._same_service("http://127.0.0.1", "http://127.0.0.1:80")
        assert not lp._same_service("http://127.0.0.1:8000", "http://127.0.0.1:8001")
        assert not lp._same_service("http://127.0.0.1:8000", "https://127.0.0.1:8000")
        assert not lp._same_service("http://10.0.0.5:8000", "http://127.0.0.1:8000")


_REPLY = object()


class TestStep:
    def test_edits_that_all_have_a_record_need_no_paraphraser_and_make_no_note(self, cwd):
        edits = [make_edit(), make_edit("Another question here?", "Another corrected text.", "x")]
        run_write(edits, FakeChat(lambda body: GOOD))

        # (no endpoint, no token: the step must not even look for them)
        assert lp.paraphrase_edits(_state(), edits) == []

    def test_no_edits_is_no_note(self, cwd):
        assert lp.paraphrase_edits(_state(), []) == []

    def test_only_the_edits_without_a_record_are_counted_when_nobody_can_answer(self, cwd):
        have = make_edit()
        lack = make_edit("Another question here?", "Another corrected text.", "x")
        run_write([have], FakeChat([GOOD]))

        (note,) = lp.paraphrase_edits(_state(), [have, lack])

        assert note.startswith("paraphrases: none written for 1 edit: ")


class TestTransport:
    def captured(self, monkeypatch, status=200, payload=_REPLY):
        calls = []

        def fake(url, headers, body):
            calls.append({"url": url, "headers": dict(headers), "body": body})
            if payload is _REPLY:
                payload_out = {"choices": [{"message": {"role": "assistant", "content": GOOD}}]}
            else:
                payload_out = payload
            return status, payload_out

        monkeypatch.setattr(lp, "_post_chat", fake)
        return calls

    def test_the_internal_call_carries_the_header_and_the_tool_token(self, cwd, monkeypatch):
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        calls = self.captured(monkeypatch)
        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        assert chat({"model": "m"}) == GOOD

        assert calls[0]["url"] == ENDPOINT + "/v1/chat/completions"
        assert calls[0]["headers"] == {
            "X-Soup-Internal": "loop-check", "Authorization": f"Bearer {TOKEN}",
        }

    def test_a_call_to_another_endpoint_carries_no_credential_and_no_header(self, cwd, monkeypatch):
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        calls = self.captured(monkeypatch)
        chat = lp.make_chat(lp.Paraphraser("http://127.0.0.1:11434/v1", "qwen", internal=False))

        chat({"model": "qwen"})

        assert calls[0]["headers"] == {}
        assert calls[0]["url"] == "http://127.0.0.1:11434/v1/chat/completions"

    def test_the_http_call_is_capped_and_never_reads_a_proxy_for_plain_http(self, cwd, monkeypatch):
        seen = {}

        def post(url, **kwargs):
            import httpx

            seen.update(kwargs, url=url)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": GOOD}}]},
                headers={"X-Soup-Internal": "loop-check"},
            )

        monkeypatch.setattr("soup_cli.utils.http_response.post_capped", post)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        assert chat({"model": "m", "messages": []}) == GOOD

        assert seen["max_bytes"] == lp.REPLY_LIMIT_BYTES == 64 * 1024
        assert seen["trust_env"] is False and seen["timeout"] == lp.CHAT_TIMEOUT_SEC
        assert "127.0.0.1" not in seen["endpoint_label"] and TOKEN not in seen["endpoint_label"]
        assert seen["headers"]["X-Soup-Internal"] == "loop-check"

    def stub_post_capped(self, monkeypatch, headers=None):
        import httpx

        def post(url, **kwargs):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": GOOD}}]}, headers=headers or {}
            )

        monkeypatch.setattr("soup_cli.utils.http_response.post_capped", post)

    def test_an_internal_call_the_server_did_not_echo_is_not_accepted(self, cwd, monkeypatch):
        # A soup serve older than the header ignores it and records the call: say so, stop.
        self.stub_post_capped(monkeypatch)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        with pytest.raises(lp.ParaphraserUnavailableError) as caught:
            chat({"model": "m"})

        assert "did not confirm the internal call" in str(caught.value)
        assert "restart soup serve" in str(caught.value)

    def test_a_wrong_echo_is_not_a_confirmation(self, cwd, monkeypatch):
        self.stub_post_capped(monkeypatch, {"X-Soup-Internal": "something-else"})
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        with pytest.raises(lp.ParaphraserUnavailableError, match="did not confirm"):
            chat({"model": "m"})

    def test_an_echoed_header_confirms_the_internal_call(self, cwd, monkeypatch):
        self.stub_post_capped(monkeypatch, {"x-soup-internal": "loop-check"})
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)

        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        assert chat({"model": "m"}) == GOOD

    def test_another_endpoint_is_not_asked_to_echo_anything(self, cwd, monkeypatch):
        self.stub_post_capped(monkeypatch)
        chat = lp.make_chat(lp.Paraphraser("http://127.0.0.1:11434", "m", internal=False))

        assert chat({"model": "m"}) == GOOD

    def test_a_server_that_does_not_echo_stops_the_pass_and_writes_nothing(self, cwd, monkeypatch):
        self.stub_post_capped(monkeypatch)
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        edits = [make_edit(), make_edit("Another question here?", "Another corrected text.", "x")]

        (note,) = lp.paraphrase_edits(_state(), edits, now=lambda: NOW)

        assert "Stopped, 2 edits not asked: the serve endpoint did not confirm" in note
        assert not os.path.exists(PARAPHRASES_FILE)

    @pytest.mark.parametrize(
        ("status", "payload", "text"),
        [
            (0, "ConnectError", "no response from the paraphraser endpoint (ConnectError)"),
            (0, "ResponseTooLargeError", "no response"),
            (500, {"detail": "Internal server error"}, "HTTP 500: Internal server error"),
            (404, None, "HTTP 404"),
            (200, {"choices": []}, "holds no message"),
            (200, {"choices": [{"message": {"content": None}}]}, "holds no message"),
            (200, None, "holds no message"),
        ],
    )
    def test_a_failed_call_raises_with_the_reason(self, cwd, monkeypatch, status, payload, text):
        self.captured(monkeypatch, status, payload)
        chat = lp.make_chat(lp.Paraphraser("http://127.0.0.1:11434", "m"))

        with pytest.raises(lp.ParaphraserUnavailableError) as caught:
            chat({"model": "m"})

        assert text in str(caught.value)

    def test_a_refused_internal_call_names_the_token_setting(self, cwd, monkeypatch):
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        self.captured(monkeypatch, 401, {"detail": "Invalid or missing bearer token"})
        chat = lp.make_chat(lp.Paraphraser(ENDPOINT, "m", internal=True))

        with pytest.raises(lp.ParaphraserUnavailableError) as caught:
            chat({"model": "m"})

        message = str(caught.value)
        assert "HTTP 401" in message and "SOUP_TOOL_AUTH_TOKEN" in message
        assert TOKEN not in message

    def test_a_detail_with_control_characters_is_cleaned(self, cwd, monkeypatch):
        self.captured(monkeypatch, 500, {"detail": "bad\x1b[31m\nthing" + "x" * 500})
        chat = lp.make_chat(lp.Paraphraser("http://127.0.0.1:11434", "m"))

        with pytest.raises(lp.ParaphraserUnavailableError) as caught:
            chat({"model": "m"})

        message = str(caught.value)
        assert "\x1b" not in message and "\n" not in message and len(message) < 300


# ---------------------------------------------------------------------------
# The train stage: when it runs, what it notes, what it never does
# ---------------------------------------------------------------------------


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(1_700_000_000.0 + offset, timezone.utc).isoformat()


TRACES = os.path.join(".soup-loops", "traces", "conversations.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")


def conversation(number, prompt=PROMPT, response=ORIGINAL, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return {
        "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
        "ts": iso(number), "model": "base", "version": 1, "adapter": None, "canary_bucket": None,
        "messages": messages, "response": response, "finish_reason": "stop", "source": "soup-serve",
    }


def edit_line(number, text=EDIT_TEXT):
    return {
        "schema_version": 1, "id": uid(5000 + number), "response_id": uid(number),
        "ts": iso(100 + number), "kind": "edit", "edited_text": text, "source": "playground",
    }


def thumb_line(number, value):
    return {
        "schema_version": 1, "id": uid(6000 + number), "response_id": uid(number),
        "ts": iso(100 + number), "kind": "thumb", "value": value, "source": "playground",
    }


def corrections(count=2, *, response=ORIGINAL):
    """``count`` answers, each corrected by a person, to different prompts."""
    prompts = [PROMPT, "Where is the capital of France?", "How many legs has a spider?"]
    texts = [EDIT_TEXT, "The capital of France is Paris, on the Seine.", "Eight legs, always."]
    write_jsonl(TRACES, [
        conversation(number + 1, prompts[number], response) for number in range(count)
    ])
    write_jsonl(FEEDBACK, [edit_line(number + 1, texts[number]) for number in range(count)])


class Trainer:
    """A ``soup train`` stand-in that remembers what it trained on."""

    def __init__(self, on_call=None):
        self.on_call = on_call
        self.calls = 0
        self.trained_on: list = []
        self.configs: list = []

    def __call__(self, argv, **kwargs):
        self.calls += 1
        if self.on_call:
            self.on_call()
        with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.configs.append(config)
        with open(config["data"]["train"], encoding="utf-8") as handle:
            self.trained_on.append([json.loads(line) for line in handle if line.strip()])
        return SimpleNamespace(returncode=0, stderr=b"")


class Transport:
    """Stands in for ``loop_paraphrase._post_chat`` and remembers every call."""

    def __init__(self, replies=None, status=200, order=None):
        self.replies = replies if replies is not None else [GOOD]
        self.status = status
        self.order = order
        self.calls: list = []

    def __call__(self, url, headers, body):
        self.calls.append({"url": url, "headers": dict(headers), "body": body})
        if self.order is not None:
            self.order.append("paraphrase")
        if self.status != 200:
            return self.status, "ConnectError" if self.status == 0 else {"detail": "refused"}
        reply = self.replies(body) if callable(self.replies) else self.replies[
            min(len(self.calls) - 1, len(self.replies) - 1)
        ]
        return 200, {"choices": [{"message": {"role": "assistant", "content": reply}}]}


@pytest.fixture
def serving(cwd, monkeypatch):
    """The loop's own serve endpoint and its tool token are configured."""
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
    return cwd


def train(state, monkeypatch, trainer=None, transport=None):
    trainer = trainer or Trainer()
    monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
    if transport is not None:
        monkeypatch.setattr(lp, "_post_chat", transport)
    ctx = ls.harvest_from_traces(state)
    return ls.train_dpo_from_pairs(state, ctx), trainer


class TestHarvestEdits:
    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_the_harvest_lists_the_corrected_answers_of_the_dataset(self, cwd, method):
        corrections(2)

        ctx = ls.harvest_from_traces(_state(train_method=method, min_new_edits=1))

        edits = ctx["edited_answers"]
        assert [edit.id for edit in edits] == [uid(1), uid(2)]
        assert edits[0].messages == ({"role": "user", "content": PROMPT},)
        assert (edits[0].original, edits[0].edited) == (ORIGINAL, EDIT_TEXT)

    def test_a_dpo_pair_needs_an_answer_to_reject_so_a_blank_original_is_not_listed(self, cwd):
        corrections(1, response="  ")

        dpo = ls.harvest_from_traces(_state(train_method="dpo"))
        sft = ls.harvest_from_traces(_state(train_method="sft", min_new_edits=1))

        assert dpo["edited_answers"] == []
        assert [edit.id for edit in sft["edited_answers"]] == [uid(1)]

    def test_an_answer_that_was_only_rated_is_not_an_edit(self, cwd):
        write_jsonl(TRACES, [conversation(1)])
        write_jsonl(FEEDBACK, [thumb_line(1, 1)])

        ctx = ls.harvest_from_traces(_state(train_method="sft", min_new_edits=1))

        assert ctx["edited_answers"] == []

    def test_an_edit_identical_to_the_answer_changes_nothing_and_is_not_listed(self, cwd):
        write_jsonl(TRACES, [conversation(1)])
        write_jsonl(FEEDBACK, [edit_line(1, ORIGINAL + "  ")])

        assert ls.harvest_from_traces(_state(train_method="sft"))["edited_answers"] == []


class TestTrainStage:
    def test_paraphrases_are_written_before_training_starts(self, serving, monkeypatch):
        corrections(2)
        order: list = []
        trainer = Trainer(on_call=lambda: order.append("train"))
        transport = Transport(order=order)

        out, _ = train(_state(train_method="sft", min_new_edits=1), monkeypatch, trainer, transport)

        assert out["skipped"] is False and out["run_id"], out
        assert order == ["paraphrase", "paraphrase", "train"]
        lines = read_lines()
        assert [line["response_id"] for line in lines] == [uid(1), uid(2)]
        assert all(schema_errors(line) == [] for line in lines)

    def test_the_note_counts_new_paraphrases_edits_and_drops(self, serving, monkeypatch):
        corrections(2)
        replies = [
            "1. how do i reset my router?\n2. What steps reset my router?\n"
            "3. My router needs a reset, what do I do?",
            "1. Which city is the capital there?\n2. What is the name of that country's main city?",
        ]

        out, _ = train(_state(train_method="sft", min_new_edits=1), monkeypatch,
                       transport=Transport(replies))

        assert out["notes"] == "paraphrases: 4 new for 2 edits, 1 dropped (same_as_prompt 1)"

    def test_the_call_goes_to_the_loops_own_serve_endpoint_the_internal_way(
        self, serving, monkeypatch
    ):
        corrections(1)
        transport = Transport()

        train(_state(train_method="sft", min_new_edits=1), monkeypatch, transport=transport)

        (call,) = transport.calls[:1]
        assert call["url"] == ENDPOINT + "/v1/chat/completions"
        assert call["headers"] == {
            "X-Soup-Internal": "loop-check", "Authorization": f"Bearer {TOKEN}",
        }
        assert call["body"]["messages"][0]["content"].endswith("Question: " + PROMPT)
        assert read_lines()[0]["generator"]["model"] == "./model"

    def test_a_configured_paraphraser_is_asked_instead_and_named_in_the_record(
        self, serving, monkeypatch
    ):
        corrections(1)
        transport = Transport()
        state = _state(
            train_method="sft", min_new_edits=1, paraphraser_model="qwen2.5:7b",
            paraphraser_endpoint="http://127.0.0.1:11434/v1",
        )

        train(state, monkeypatch, transport=transport)

        call = transport.calls[0]
        assert call["url"] == "http://127.0.0.1:11434/v1/chat/completions"
        assert call["headers"] == {}  # no token, no internal header for another server
        assert call["body"]["model"] == "qwen2.5:7b"
        assert read_lines()[0]["generator"]["model"] == "qwen2.5:7b"

    def test_an_edit_is_paraphrased_once_and_later_iterations_reuse_it(self, serving, monkeypatch):
        corrections(2)
        state = _state(train_method="sft", min_new_edits=1)
        first = Transport()
        train(state, monkeypatch, transport=first)
        assert len(first.calls) == 2

        second = Transport()
        out, _ = train(state, monkeypatch, transport=second)

        assert second.calls == [] and len(read_lines()) == 2
        assert "paraphrases" not in out.get("notes", "")  # nothing was done: no note

    def test_only_the_new_edit_is_paraphrased_the_next_time(self, serving, monkeypatch):
        corrections(1)
        state = _state(train_method="sft", min_new_edits=1)
        train(state, monkeypatch, transport=Transport())
        corrections(2)  # a second correction arrives

        again = Transport([
            "1. Which city is the capital there?\n"
            "2. What is the name of that country's main city?"
        ])
        out, _ = train(state, monkeypatch, transport=again)

        assert len(again.calls) == 1
        assert "Where is the capital" in again.calls[0]["body"]["messages"][0]["content"]
        assert [line["response_id"] for line in read_lines()] == [uid(1), uid(2)]
        assert out["notes"] == "paraphrases: 2 new for 1 edit"

    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_paraphrases_never_enter_any_dataset(self, serving, monkeypatch, method):
        corrections(2)
        state = _state(train_method=method, min_new_edits=1)

        _, trainer = train(state, monkeypatch, transport=Transport())

        written = lp.read_paraphrases().latest
        assert len(written) == 2
        paraphrases = [text for record in written.values() for text in record["paraphrases"]]
        assert paraphrases
        datasets = [
            path.read_text(encoding="utf-8") for path in Path(".soup-loops", "pairs").iterdir()
        ]
        trained = [json.dumps(rows) for rows in trainer.trained_on]
        assert datasets and trained
        for text in [*datasets, *trained]:
            assert "Rewrite the question below" not in text
            for paraphrase in paraphrases:
                assert paraphrase not in text, (method, paraphrase)

    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_the_harvest_gives_the_same_rows_with_and_without_paraphrases_on_disk(
        self, serving, monkeypatch, method
    ):
        corrections(2)
        state = _state(train_method=method, min_new_edits=1)
        before = ls.harvest_from_traces(state)
        with open(before["pairs_path"], encoding="utf-8") as handle:
            rows_before = handle.read()

        train(state, monkeypatch, transport=Transport())
        after = ls.harvest_from_traces(state)
        with open(after["pairs_path"], encoding="utf-8") as handle:
            rows_after = handle.read()

        assert lp.read_paraphrases().latest  # they are on disk now
        assert rows_before == rows_after
        assert (before["pairs_harvested"], before["new_pairs"]) == (
            after["pairs_harvested"], after["new_pairs"]
        )

    def test_the_paraphrase_file_is_not_a_conversation_log_of_the_trace_directory(
        self, serving, monkeypatch
    ):
        corrections(1)
        train(_state(train_method="sft", min_new_edits=1), monkeypatch, transport=Transport())

        from soup_cli.utils.loop_feedback import conversation_files

        assert os.path.exists(PARAPHRASES_FILE)
        assert all("paraphrases" not in path for path in conversation_files(".soup-loops/traces"))

    def test_without_a_paraphraser_nothing_is_written_and_the_note_names_the_flags(
        self, cwd, monkeypatch
    ):
        corrections(2)

        out, trainer = train(_state(train_method="sft", min_new_edits=1), monkeypatch)

        assert out["skipped"] is False and trainer.calls == 1  # training goes on
        assert not os.path.exists(PARAPHRASES_FILE)
        assert out["notes"].startswith("paraphrases: none written for 2 edits: ")
        assert "--paraphraser-endpoint" in out["notes"] and "--paraphraser-model" in out["notes"]
        assert "SOUP_LOOP_SERVE_ENDPOINT" in out["notes"] and ls.TOOL_AUTH_TOKEN_ENV in out["notes"]

    def test_an_unreachable_paraphraser_is_a_note_and_training_goes_on(self, serving, monkeypatch):
        corrections(2)
        transport = Transport(status=0)

        out, trainer = train(_state(train_method="sft", min_new_edits=1), monkeypatch,
                             transport=transport)

        assert trainer.calls == 1 and out["skipped"] is False
        assert len(transport.calls) == 1  # the first failure ends the pass
        assert "Stopped, 2 edits not asked" in out["notes"]
        assert "no response from the paraphraser endpoint (ConnectError)" in out["notes"]
        assert not os.path.exists(PARAPHRASES_FILE)

    def test_a_refusing_serve_endpoint_names_the_token_setting(self, serving, monkeypatch):
        corrections(1)

        out, _ = train(_state(train_method="sft", min_new_edits=1), monkeypatch,
                       transport=Transport(status=401))

        assert "HTTP 401" in out["notes"] and ls.TOOL_AUTH_TOKEN_ENV in out["notes"]

    def test_a_bug_in_the_step_is_a_note_not_a_failed_iteration(self, serving, monkeypatch):
        corrections(1)

        def boom(*args, **kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr(lp, "paraphrase_edits", boom)

        out, trainer = train(_state(train_method="sft", min_new_edits=1), monkeypatch)

        assert trainer.calls == 1 and "paraphrases: not written (RuntimeError)" in out["notes"]

    def test_a_poor_paraphraser_adds_the_warning_to_the_notes(self, serving, monkeypatch):
        corrections(2)
        def echo(body):
            question = body["messages"][0]["content"].split("Question: ", 1)[1]
            return f"1. {question}\n2. {question.lower()}"

        out, _ = train(_state(train_method="sft", min_new_edits=1), monkeypatch,
                       transport=Transport(echo))

        assert "paraphrases mostly dropped: a small served model paraphrases poorly; " \
               "set soup loop init --paraphraser-model" in out["notes"]
        assert "left without paraphrases" in out["notes"]

    def test_a_cycle_that_does_not_train_asks_nothing(self, serving, monkeypatch):
        corrections(1)
        transport = Transport()
        state = _state(train_method="sft", min_new_edits=5, min_new_rated=30)

        out, trainer = train(state, monkeypatch, transport=transport)

        assert out["skipped"] is True and "not_enough_new_data" in out["notes"]
        assert transport.calls == [] and trainer.calls == 0
        assert not os.path.exists(PARAPHRASES_FILE)

    def test_a_context_without_edits_asks_nothing(self, serving, monkeypatch):
        # a custom harvest_fn does not know about edited_answers
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", Trainer())
        monkeypatch.setattr(lp, "_post_chat", Transport())
        os.makedirs(".soup-loops/pairs")
        path = ".soup-loops/pairs/sft-x.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"messages": [{"role": "user", "content": "a"},
                                                  {"role": "assistant", "content": "b"}]}) + "\n")

        out = ls.train_dpo_from_pairs(
            _state(train_method="sft"), {"pairs_path": path, "pairs_harvested": 1, "new_pairs": 1}
        )

        assert out["skipped"] is False and "paraphrases" not in out.get("notes", "")
        assert not os.path.exists(PARAPHRASES_FILE)

    def test_the_paraphrase_note_follows_the_quantization_note(self, serving, monkeypatch):
        corrections(1)
        monkeypatch.setattr(
            "soup_cli.utils.loop_stages.resolve_sft_quantization",
            lambda requested, base: ("4bit", "sft quantization auto: 4bit (stub)"),
        )

        out, _ = train(_state(train_method="sft", min_new_edits=1), monkeypatch,
                       transport=Transport())

        assert out["notes"] == (
            "sft quantization auto: 4bit (stub); paraphrases: 2 new for 1 edit"
        )


# ---------------------------------------------------------------------------
# soup serve: the internal header
# ---------------------------------------------------------------------------


class _AdapterModel:
    def __init__(self) -> None:
        self.current = None

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous, self.current = self.current, None
        try:
            yield
        finally:
            self.current = previous


@pytest.fixture
def served(cwd, monkeypatch):
    """A real serve app: trace log, thumbs database, tool token, and a canary at 100% of keyed
    traffic, so any keyed request that is not kept out of the canary lands in it."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import soup_cli.commands.serve as serve
    from soup_cli.monitoring.trace_logger import TraceLogWriter
    from soup_cli.utils import local_rl

    state_path = os.path.abspath(os.path.join(".soup", "loop.yaml"))
    stats_path = os.path.abspath(os.path.join(".soup", "canary-stats.json"))
    write_state(
        LoopState(
            served_model="base", eval_suite="e", baseline="b", canary_active="candidate",
            canary_traffic_pct=100.0, canary_rollout_id="rollout-p",
        ),
        state_path,
    )
    local_rl.init_local_rl_db("thumbs.db")
    model = _AdapterModel()
    observed: list = []

    def generate(current_model, tokenizer, messages, **kwargs):
        observed.append(current_model.current)
        if messages[-1]["content"].startswith("Rewrite the question below"):
            return GOOD, 5, 7, "stop"
        return "an answer", 5, 7, "stop"

    monkeypatch.setattr(serve, "_generate_response", generate)

    def build(auth_token=TOKEN):
        app = serve._create_app(
            model_obj=model, tokenizer=object(), device="cpu", model_name="base",
            max_tokens_default=8, adapter_map={"candidate": "unused"},
            peft_adapter_names={"candidate"},
            trace_log_writer=TraceLogWriter(os.path.join(".soup-loops", "traces", "serve.jsonl")),
            record_thumbs_db="thumbs.db", auth_token=auth_token,
            canary_state_path=state_path, canary_stats_path=stats_path,
        )
        return TestClient(app, base_url=ENDPOINT), app

    client, app = build()

    def route_http(monkeypatch_):
        """Send the paraphrase step's HTTP into the app instead of the network."""
        import httpx

        seen: list = []

        def post(url, headers=None, json=None, timeout=None, trust_env=None, **kwargs):
            seen.append({"url": url, "headers": dict(headers or {}), "kwargs": kwargs})
            resp = client.post(urlparse(url).path, headers=headers or {}, json=json)
            echo = {lp.INTERNAL_HEADER: resp.headers[lp.INTERNAL_HEADER]} \
                if lp.INTERNAL_HEADER in resp.headers else {}
            return httpx.Response(
                resp.status_code, content=resp.content,
                headers={"content-type": resp.headers.get("content-type", ""), **echo},
            )

        monkeypatch_.setattr("soup_cli.utils.http_response.post_capped", post)
        return seen

    def trace_lines() -> list:
        path = os.path.join(".soup-loops", "traces", "serve.jsonl")
        return read_lines(path)

    def stats():
        from soup_cli.utils.canary_router import read_bucket_stats

        app.state.canary_outcomes.flush()
        return read_bucket_stats(
            stable="base", canary="candidate", rollout_id="rollout-p", path=stats_path
        )

    def thumbs() -> int:
        with sqlite3.connect("thumbs.db") as conn:
            return conn.execute("SELECT COUNT(*) FROM thumbs").fetchone()[0]

    return SimpleNamespace(
        client=client, app=app, observed=observed, route_http=route_http, trace_lines=trace_lines,
        stats=stats, thumbs=thumbs, build=build, serve=serve, model=model,
    )


INTERNAL = {"X-Soup-Internal": "loop-check", "Authorization": f"Bearer {TOKEN}"}


def chat(client, content="hello", headers=None, **extra):
    body = {"messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


class TestServeInternalHeader:
    def test_with_a_valid_token_the_answer_is_given_and_nothing_is_recorded(self, served):
        resp = chat(served.client, headers=INTERNAL)

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "an answer"
        assert served.trace_lines() == []
        assert not os.path.exists(os.path.join(".soup-loops", "traces", "serve.jsonl"))

    def test_an_honoured_call_says_so_in_the_response_and_a_refused_one_does_not(self, served):
        # What lets the loop tell a server that honours the header from an older one.
        honoured = chat(served.client, headers=INTERNAL)
        plain = chat(served.client, conversation_id="user-1")
        refused = chat(served.client, headers={"X-Soup-Internal": "loop-check"})
        unknown = chat(served.client, headers={**INTERNAL, "X-Soup-Internal": "nope"})

        assert honoured.headers["X-Soup-Internal"] == "loop-check"
        assert "X-Soup-Internal" not in plain.headers
        assert "X-Soup-Internal" not in refused.headers
        assert "X-Soup-Internal" not in unknown.headers

    def test_the_same_request_without_the_header_is_recorded_and_canaried(self, served):
        # The control: it is what makes the tests of the internal call mean something.
        resp = chat(served.client, conversation_id="user-1")

        assert resp.status_code == 200
        (line,) = served.trace_lines()
        assert line["canary_bucket"] == "canary" and line["adapter"] == "candidate"
        assert served.stats().canary_ok == 1
        assert served.observed == ["candidate"]

    def test_an_internal_call_is_never_routed_to_a_canary_bucket(self, served):
        for number in range(20):  # a keyed request, 20 different keys: all would be canary
            chat(served.client, conversation_id=f"user-{number}", headers=INTERNAL)
            chat(served.client, headers={**INTERNAL, "X-Conversation-Id": f"key-{number}"})

        assert served.observed == [None] * 40  # the stable side: no adapter selected
        assert "candidate" not in served.observed

    def test_an_internal_call_does_not_count_in_the_canary_stats(self, served):
        for number in range(10):
            chat(served.client, conversation_id=f"user-{number}", headers=INTERNAL)

        stats = served.stats()

        assert (stats.canary_ok, stats.canary_major, stats.stable_ok, stats.stable_major) == (
            0, 0, 0, 0
        )

    def test_an_internal_call_that_fails_does_not_count_either(self, served, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("cuda out of memory")

        monkeypatch.setattr(served.serve, "_generate_response", boom)

        resp = chat(served.client, conversation_id="user-1", headers=INTERNAL)

        assert resp.status_code == 500
        stats = served.stats()
        assert (stats.canary_major, stats.stable_major, stats.canary_ok, stats.stable_ok) == (
            0, 0, 0, 0
        )

    def test_an_internal_call_writes_no_thumbs(self, served):
        chat(served.client, headers=INTERNAL)

        assert served.thumbs() == 0

    def test_a_streamed_internal_call_is_not_recorded_or_canaried_either(self, served):
        resp = chat(served.client, conversation_id="user-1", headers=INTERNAL, stream=True)

        assert resp.status_code == 200 and resp.text.rstrip().endswith("data: [DONE]")
        assert resp.headers["X-Soup-Internal"] == "loop-check"
        assert served.trace_lines() == []
        stats = served.stats()
        assert (stats.canary_ok, stats.canary_major) == (0, 0)
        assert served.observed == [None]

    def test_the_header_without_any_token_is_refused_and_nothing_is_generated(self, served):
        resp = chat(served.client, headers={"X-Soup-Internal": "loop-check"})

        assert resp.status_code == 401
        assert "X-Soup-Internal is honoured only with a valid tool token" in resp.json()["detail"]
        assert resp.headers["www-authenticate"] == "Bearer"
        assert served.observed == [] and served.trace_lines() == []

    def test_the_header_with_a_wrong_token_is_refused(self, served):
        resp = chat(served.client, headers={
            "X-Soup-Internal": "loop-check", "Authorization": "Bearer not-the-token",
        })

        assert resp.status_code == 401
        assert "Invalid or missing bearer token" in resp.json()["detail"]
        assert "X-Soup-Internal" in resp.json()["detail"]
        assert served.observed == [] and served.trace_lines() == []

    def test_the_header_with_the_token_in_another_scheme_is_refused(self, served):
        resp = chat(served.client, headers={
            "X-Soup-Internal": "loop-check", "Authorization": f"Token {TOKEN}",
        })

        assert resp.status_code == 401 and served.observed == []

    def test_a_server_without_a_tool_token_refuses_the_header_whatever_is_sent(self, served):
        client, _ = served.build(auth_token=None)

        resp = chat(client, headers={
            "X-Soup-Internal": "loop-check", "Authorization": "Bearer anything",
        })

        assert resp.status_code == 401
        assert "Start soup serve with --tool-auth-token" in resp.json()["detail"]
        assert served.observed == []

    def test_an_unknown_value_is_refused_even_with_the_token(self, served):
        resp = chat(served.client, headers={**INTERNAL, "X-Soup-Internal": "skip-logging"})

        assert resp.status_code == 400
        assert "the only one is 'loop-check'" in resp.json()["detail"]
        assert served.observed == []

    def test_a_token_alone_changes_nothing_the_answer_is_still_recorded(self, served):
        chat(served.client, conversation_id="user-1", headers={"Authorization": f"Bearer {TOKEN}"})

        assert len(served.trace_lines()) == 1 and served.stats().canary_ok == 1

    def test_the_header_is_not_honoured_by_the_anthropic_route(self, served):
        resp = served.client.post(
            "/v1/messages", headers=INTERNAL,
            json={"model": "base", "max_tokens": 8,
                  "messages": [{"role": "user", "content": "hello"}]},
        )

        assert resp.status_code == 200, resp.text
        assert len(served.trace_lines()) == 1  # only /v1/chat/completions has the exception

    def test_a_paraphrase_call_through_serve_leaves_no_trace_line_and_no_canary_stat(
        self, served, monkeypatch
    ):
        # The hard rule: otherwise the paraphrases leak into the next training run.
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        seen = served.route_http(monkeypatch)
        edits = [make_edit(), make_edit("Where is Paris?", "In France, on the Seine.", "no")]

        notes = lp.paraphrase_edits(_state(served_model="base"), edits, now=lambda: NOW)

        assert notes == ["paraphrases: 4 new for 2 edits"]
        assert len(seen) == 2 and seen[0]["headers"]["X-Soup-Internal"] == "loop-check"
        assert seen[0]["kwargs"]["max_bytes"] == lp.REPLY_LIMIT_BYTES
        assert served.trace_lines() == []
        stats = served.stats()
        assert (stats.canary_ok, stats.canary_major, stats.stable_ok, stats.stable_major) == (
            0, 0, 0, 0
        )
        assert served.thumbs() == 0
        assert served.observed == [None, None]  # the stable side, never the canary
        lines = read_lines()
        assert [line["response_id"] for line in lines] == [edit.id for edit in edits]
        assert all(schema_errors(line) == [] for line in lines)
        assert lines[0]["generator"]["model"] == "base"

    def test_the_loop_check_does_not_make_the_next_harvest_see_anything(self, served, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
        served.route_http(monkeypatch)
        chat(served.client, "How do I reset my router?", conversation_id="user-1")  # a real user
        lp.paraphrase_edits(_state(served_model="base"), [make_edit()], now=lambda: NOW)

        from soup_cli.data.traces.parsers import parse_soup_serve

        traces = list(parse_soup_serve(os.path.join(".soup-loops", "traces")))

        assert [trace.prompt for trace in traces] == ["How do I reset my router?"]


# ---------------------------------------------------------------------------
# soup loop init --paraphraser-endpoint / --paraphraser-model
# ---------------------------------------------------------------------------


@pytest.fixture
def loop_cli(cwd):
    from soup_cli.commands.loop import app as loop_app

    def cli(*args):
        result = CliRunner().invoke(loop_app, list(args))
        return result, _clean(result.output)

    return cli


INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


class TestInitFlags:
    def test_both_flags_are_stored_in_loop_yaml(self, loop_cli):
        result, out = loop_cli(
            *INIT, "--paraphraser-endpoint", "http://127.0.0.1:11434/v1/",
            "--paraphraser-model", " qwen2.5:7b ",
        )

        assert result.exit_code == 0, (out, repr(result.exception))
        state = read_state()
        assert state.paraphraser_endpoint == "http://127.0.0.1:11434/v1"
        assert state.paraphraser_model == "qwen2.5:7b"
        assert "paraphraser: qwen2.5:7b at http://127.0.0.1:11434/v1" in out

    def test_without_them_nothing_is_stored_and_the_panel_is_unchanged(self, loop_cli):
        result, out = loop_cli(*INIT)

        assert result.exit_code == 0, (out, repr(result.exception))
        state = read_state()
        assert (state.paraphraser_endpoint, state.paraphraser_model) == (None, None)
        assert "paraphraser" not in out

    def test_status_shows_the_paraphraser_only_when_one_is_set(self, loop_cli):
        loop_cli(*INIT, "--paraphraser-endpoint", "https://192.168.1.20:8080",
                 "--paraphraser-model", "big-model")

        _, out = loop_cli("status")

        assert re.search(r"paraphraser\s*│\s*big-model at https://192\.168\.1\.20:8080", out), out
        loop_cli(*INIT, "--force")
        assert "paraphraser" not in loop_cli("status")[1]

    @pytest.mark.parametrize(
        ("flags", "message"),
        [
            (("--paraphraser-model", "m"), "--paraphraser-model needs --paraphraser-endpoint"),
            (("--paraphraser-endpoint", "http://127.0.0.1:8000"),
             "--paraphraser-endpoint needs --paraphraser-model"),
            (("--paraphraser-endpoint", "https://api.example.com/v1", "--paraphraser-model", "m"),
             "invalid --paraphraser-endpoint: --paraphraser-endpoint must be loopback/LAN"),
            (("--paraphraser-endpoint", "http://example.com", "--paraphraser-model", "m"),
             "invalid --paraphraser-endpoint"),
            (("--paraphraser-endpoint", "http://u:p@127.0.0.1:8000", "--paraphraser-model", "m"),
             "must not carry credentials"),
            (("--paraphraser-endpoint", "ftp://127.0.0.1", "--paraphraser-model", "m"),
             "http/https"),
            (("--paraphraser-endpoint", "http://127.0.0.1:8000", "--paraphraser-model", "   "),
             "invalid --paraphraser-model"),
            (("--paraphraser-endpoint", "http://127.0.0.1:8000",
              "--paraphraser-model", "a" * 257), "invalid --paraphraser-model"),
            (("--paraphraser-endpoint", "http://127.0.0.1:8000",
              "--paraphraser-model", "bad\x01name"), "invalid --paraphraser-model"),
        ],
    )
    def test_a_bad_pair_is_refused_naming_the_flag_and_nothing_is_written(
        self, loop_cli, cwd, flags, message
    ):
        result, out = loop_cli(*INIT, *flags)

        assert result.exit_code == 2, (out, repr(result.exception))
        assert message in out
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_the_secret_in_a_refused_url_is_not_echoed(self, loop_cli):
        _, out = loop_cli(*INIT, "--paraphraser-endpoint", "http://user:hunter2@127.0.0.1:8000",
                          "--paraphraser-model", "m")

        assert "hunter2" not in out


class TestLoopStateFields:
    def test_they_are_the_last_two_fields_and_default_to_none(self):
        names = list(LoopState.__dataclass_fields__)

        assert names[-2:] == ["paraphraser_endpoint", "paraphraser_model"]
        assert names[-3] == "sft_recipe"
        state = _state()
        assert (state.paraphraser_endpoint, state.paraphraser_model) == (None, None)

    def test_a_loop_yaml_from_before_them_loads_with_none(self, cwd):
        legacy = {"served_model": "m", "eval_suite": "e", "baseline": "b", "status": "stopped"}
        os.makedirs(".soup")
        with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        state = read_state()

        assert (state.paraphraser_endpoint, state.paraphraser_model) == (None, None)

    def test_they_survive_a_write_and_a_read(self, cwd):
        write_state(_state(paraphraser_endpoint="http://127.0.0.1:11434", paraphraser_model="m"))

        state = read_state()

        assert state.paraphraser_endpoint == "http://127.0.0.1:11434"
        assert state.paraphraser_model == "m"

    @pytest.mark.parametrize("field", ["paraphraser_endpoint", "paraphraser_model"])
    @pytest.mark.parametrize("value", ["", "a\x00b", "a\nb", "a\x7fb", "x" * 513])
    def test_a_bad_string_is_refused(self, field, value):
        with pytest.raises(ValueError):
            _state(**{field: value})

    @pytest.mark.parametrize("field", ["paraphraser_endpoint", "paraphraser_model"])
    @pytest.mark.parametrize("value", [1, True, ["x"], b"x"])
    def test_a_non_string_is_refused(self, field, value):
        with pytest.raises(TypeError):
            _state(**{field: value})

    def test_a_hand_edited_loop_yaml_with_a_bad_value_is_refused_on_read(self, cwd):
        os.makedirs(".soup")
        with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as handle:
            json.dump({"served_model": "m", "eval_suite": "e", "baseline": "b",
                       "paraphraser_model": 7}, handle)

        with pytest.raises(ValueError, match="invalid contents"):
            read_state()
