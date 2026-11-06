"""`soup loop` trains from feedback: three datasets, whatever method the loop uses.

People give feedback on logged answers: thumbs, ratings, edits, comments, written as
JSONL lines (the loop's data contract: a conversation record per answer in the trace
directory, a feedback record per action in the feedback log). One harvest turns them into

- ``dpo`` pairs: an edit is a preference (chosen = the edited text, rejected = the answer),
  next to the pairs the serve traces and the thumbs database already gave;
- ``kto`` rows: the answer as given, labelled good or bad by thumbs, ratings and edits, and
  an edit's text as a good one;
- ``sft`` rows: the edited text as the answer to the prompt, and the answers rated good.

``soup loop init --train-method dpo|kto|sft`` picks the one the loop trains on (sft for a new
loop; a loop.yaml without the field keeps dpo); ``--feedback-log`` says where the feedback
is. The rules of the contract (latest line per kind wins, null clears, comments accumulate,
unknown ids and unknown ``schema_version`` lines are ignored, a version 0 line has no id) are
read by ``soup_cli.utils.loop_feedback``, whoever wrote the file.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_feedback as lf
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState, init_state, read_state, write_state

T0 = 1_700_000_000.0
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_LINE_IDS = itertools.count(1000)


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "How do I reset my router?"},
]
ANSWER = "Unplug it."
EDIT = "Hold the reset button for ten seconds."


def conv(number, *, ts=0, messages=None, response=ANSWER, **extra):
    """A version 1 conversation record (the answer ``uid(number)``)."""
    line = {
        "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
        "ts": iso(ts), "model": "base", "version": 1, "adapter": None, "canary_bucket": None,
        "messages": messages if messages is not None else MESSAGES, "response": response,
        "finish_reason": "stop", "latency_ms": 12.5, "tokens": {"prompt": 5, "completion": 3},
        "options": {}, "source": "soup-serve",
    }
    line.update(extra)
    return line


def fb(number, kind, ts, **fields):
    """A version 1 feedback record about the answer ``uid(number)``."""
    line = {
        "schema_version": 1, "id": uid(next(_LINE_IDS)), "response_id": uid(number),
        "ts": iso(ts), "kind": kind, "source": "playground",
    }
    line.update(fields)
    return line


def thumb(number, ts, value):
    return fb(number, "thumb", ts, value=value)


def rating(number, ts, value):
    return fb(number, "rating", ts, value=value)


def edit(number, ts, text=EDIT):
    return fb(number, "edit", ts, edited_text=text)


def write_lines(path, lines, *, newline="\n") -> str:
    """Write dicts as JSON lines (strings and bytes verbatim) and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    chunks = []
    for line in lines:
        if isinstance(line, dict):
            line = json.dumps(line)
        if isinstance(line, str):
            line = line.encode("utf-8")
        chunks.append(line + newline.encode("ascii"))
    with open(path, "wb") as handle:
        handle.write(b"".join(chunks))
    return path


TRACES = os.path.join(".soup-loops", "traces", "conversations.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")


def logs(conversations, feedback):
    """Write both logs at the default places."""
    write_lines(TRACES, conversations)
    write_lines(FEEDBACK, feedback)


def harvest(trace_dir=os.path.join(".soup-loops", "traces"), feedback_path=FEEDBACK):
    return lf.harvest_feedback(trace_dir, feedback_path)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def rows(items):
    return [item.row for item in items]


def stamps(items):
    return [item.ts for item in items]


def kto_row(completion, label, prompt=None):
    return {
        "prompt": prompt if prompt is not None else MESSAGES, "completion": completion,
        "label": label, "source": "feedback",
    }


def sft_good_row(answer, prompt=None):
    return {
        "messages": [*(prompt if prompt is not None else MESSAGES),
                     {"role": "assistant", "content": answer}],
        "source": "feedback-good",
    }


# ---------------------------------------------------------------------------
# The rules of the contract
# ---------------------------------------------------------------------------


class TestRules:
    def test_the_latest_line_per_kind_wins_in_file_order(self, cwd):
        write_lines(FEEDBACK, [
            thumb(1, 1, 1), thumb(1, 2, -1),
            rating(1, 3, 5), rating(1, 4, 2),
            edit(1, 5, "first"), edit(1, 6, "second"),
        ])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert (record.thumb, record.thumb_ts) == (-1, T0 + 2)
        assert (record.rating, record.rating_ts) == (2, T0 + 4)
        assert (record.edit, record.edit_ts) == ("second", T0 + 6)

    def test_file_order_decides_not_ts(self, cwd):
        write_lines(FEEDBACK, [thumb(1, 50, 1), thumb(1, 10, -1)])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert (record.thumb, record.thumb_ts) == (-1, T0 + 10)

    @pytest.mark.parametrize("kind", ["thumb", "rating"])
    def test_a_null_value_clears_a_thumb_or_a_rating(self, cwd, kind):
        first = 1 if kind == "thumb" else 4
        write_lines(FEEDBACK, [fb(1, kind, 1, value=first), fb(1, kind, 2, value=None)])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert getattr(record, kind) is None
        assert getattr(record, f"{kind}_ts") is None

    def test_a_null_edited_text_withdraws_an_edit(self, cwd):
        write_lines(FEEDBACK, [edit(1, 1), fb(1, "edit", 2, edited_text=None)])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert record.edit is None and record.edit_ts is None

    def test_a_later_value_after_a_clear_counts_again(self, cwd):
        write_lines(FEEDBACK, [thumb(1, 1, 1), thumb(1, 2, None), thumb(1, 3, -1)])

        assert lf.read_feedback(FEEDBACK).by_response[uid(1)].thumb == -1

    def test_the_kinds_are_independent(self, cwd):
        """Clearing the thumb leaves the edit and the rating alone."""
        write_lines(FEEDBACK, [edit(1, 1), rating(1, 2, 4), thumb(1, 3, 1), thumb(1, 4, None)])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert (record.thumb, record.rating, record.edit) == (None, 4, EDIT)

    def test_comments_accumulate(self, cwd):
        write_lines(FEEDBACK, [
            fb(1, "comment", 1, comment="too short"),
            thumb(1, 2, 1),
            fb(1, "comment", 3, comment="but accurate"),
            fb(1, "comment", 4, comment="too short"),
        ])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert record.comments == ["too short", "but accurate", "too short"]
        assert record.thumb == 1
        assert record.lines == 4

    def test_unknown_fields_are_ignored(self, cwd):
        write_lines(FEEDBACK, [
            {**thumb(1, 1, 1), "x_zero": {"panel": "left"}, "brand_new": [1, 2]},
        ])
        write_lines(TRACES, [{**conv(1), "x_zero": {"a": 1}, "something_new": True}])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True)]

    def test_the_users_ref_and_comment_next_to_a_value_do_not_change_it(self, cwd):
        write_lines(FEEDBACK, [{**thumb(1, 1, -1), "user_ref": "local", "comment": "meh"}])

        record = lf.read_feedback(FEEDBACK).by_response[uid(1)]

        assert record.thumb == -1 and record.comments == []

    def test_feedback_about_an_unknown_answer_is_ignored_and_counted(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), thumb(99, 2, 1), edit(99, 3), thumb(98, 4, -1)])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True)]
        assert out.datasets.dpo == []
        assert rows(out.datasets.sft) == [sft_good_row(ANSWER)]  # the thumb up on answer 1
        assert out.datasets.unknown_response == 3

    @pytest.mark.parametrize("version", [2, 0, 99, "1", 1.5, True, False, None, [1]])
    def test_a_feedback_line_of_an_unknown_schema_version_is_skipped(self, cwd, version):
        logs([conv(1)], [{**thumb(1, 1, 1), "schema_version": version}, thumb(1, 2, -1)])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, False)]
        assert out.skipped_lines == 1

    def test_a_feedback_line_without_a_version_is_read_as_version_one(self, cwd):
        line = thumb(1, 1, 1)
        del line["schema_version"]
        logs([conv(1)], [line])

        assert rows(harvest().datasets.kto) == [kto_row(ANSWER, True)]

    @pytest.mark.parametrize("version", [2, "1", True, None, 1.0])
    def test_a_conversation_line_of_an_unknown_schema_version_is_skipped(self, cwd, version):
        logs([{**conv(1), "schema_version": version}], [thumb(1, 1, 1)])

        out = harvest()

        assert out.datasets.kto == [] and out.datasets.unknown_response == 1
        assert out.answers == 0

    def test_a_version_zero_line_has_no_id_so_feedback_cannot_point_at_it(self, cwd):
        legacy = {"ts": T0, "prompt": "How do I reset my router?", "response": ANSWER,
                  "latency_ms": 9, "tokens": 3}
        with_id = {**legacy, "id": uid(1)}  # still version 0: the id is not honoured
        logs([legacy, with_id, conv(2)], [thumb(1, 1, 1), thumb(2, 2, -1)])

        out = harvest()

        assert out.legacy_lines == 2
        assert out.skipped_lines == 0  # version 0 is valid; the serve-trace parser reads it
        assert out.answers == 1
        assert rows(out.datasets.kto) == [kto_row(ANSWER, False)]
        assert out.datasets.unknown_response == 1

    def test_a_missing_feedback_log_and_trace_directory_read_as_empty(self, cwd):
        out = lf.harvest_feedback(
            os.path.join("nowhere", "traces"), os.path.join("nowhere", "feedback.jsonl")
        )

        assert (out.datasets.dpo, out.datasets.kto, out.datasets.sft) == ([], [], [])
        assert (out.answers, out.feedback_lines, out.skipped_lines) == (0, 0, 0)
        assert lf.harvest_feedback(None, None).datasets.unknown_response == 0

    def test_a_directory_where_the_feedback_log_should_be_reads_as_empty(self, cwd):
        os.makedirs("fb.jsonl")

        assert lf.read_feedback("fb.jsonl").by_response == {}

    def test_only_jsonl_files_of_the_trace_directory_are_read(self, cwd):
        write_lines(os.path.join(".soup-loops", "traces", "a.jsonl"), [conv(1)])
        write_lines(os.path.join(".soup-loops", "traces", "b.jsonl"), [conv(2)])
        write_lines(os.path.join(".soup-loops", "traces", "notes.txt"), [conv(3)])
        os.makedirs(os.path.join(".soup-loops", "traces", "dir.jsonl"))
        write_lines(FEEDBACK, [thumb(1, 1, 1), thumb(2, 2, 1), thumb(3, 3, 1)])

        out = harvest()

        assert out.answers == 2
        assert out.datasets.unknown_response == 1

    def test_the_first_record_of_an_id_wins(self, cwd):
        logs([conv(1, response="first"), conv(1, response="second")], [thumb(1, 1, 1)])

        assert rows(harvest().datasets.kto) == [kto_row("first", True)]


class TestMalformedLines:
    """A log is written by other programs: a bad line is skipped and counted, never raised."""

    BAD_FEEDBACK = [
        "not json at all",
        "[1, 2, 3]",
        "42",
        '"a string"',
        "null",
        "{",
        b"\xff\xfe\x00 not utf-8",
        "[" * 50_000,
        json.dumps({"schema_version": 1, "kind": "thumb", "value": 1}),  # no response_id
        json.dumps({"response_id": 7, "kind": "thumb", "value": 1}),
        json.dumps({"response_id": "", "kind": "thumb", "value": 1}),
        json.dumps({"response_id": uid(1), "kind": "reaction", "value": 1}),
        json.dumps({"response_id": uid(1), "kind": None, "value": 1}),
        json.dumps({"response_id": uid(1), "kind": "thumb"}),  # no value: not a clear
        json.dumps({"response_id": uid(1), "kind": "thumb", "value": 5}),
        json.dumps({"response_id": uid(1), "kind": "thumb", "value": 0}),
        json.dumps({"response_id": uid(1), "kind": "thumb", "value": True}),
        json.dumps({"response_id": uid(1), "kind": "thumb", "value": "up"}),
        json.dumps({"response_id": uid(1), "kind": "thumb", "value": 1.0}),
        json.dumps({"response_id": uid(1), "kind": "rating", "value": 0}),
        json.dumps({"response_id": uid(1), "kind": "rating", "value": 6}),
        json.dumps({"response_id": uid(1), "kind": "rating", "value": 4.5}),
        json.dumps({"response_id": uid(1), "kind": "rating", "value": True}),
        json.dumps({"response_id": uid(1), "kind": "edit"}),  # no edited_text: not a withdrawal
        json.dumps({"response_id": uid(1), "kind": "edit", "edited_text": 5}),
        json.dumps({"response_id": uid(1), "kind": "edit", "edited_text": ""}),
        json.dumps({"response_id": uid(1), "kind": "edit", "edited_text": "  \n"}),
        json.dumps({"response_id": uid(1), "kind": "edit", "edited_text": ["a"]}),
        json.dumps({"response_id": uid(1), "kind": "comment"}),
        json.dumps({"response_id": uid(1), "kind": "comment", "comment": ""}),
        json.dumps({"response_id": uid(1), "kind": "comment", "comment": 7}),
    ]

    @pytest.mark.parametrize("line", BAD_FEEDBACK, ids=range(len(BAD_FEEDBACK)))
    def test_a_bad_feedback_line_is_skipped_and_the_good_ones_around_it_count(self, cwd, line):
        logs([conv(1)], [thumb(1, 1, 1), line, thumb(1, 3, -1)])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, False)]
        assert out.skipped_lines == 1
        assert out.feedback_lines == 3

    def test_a_bad_conversation_line_is_skipped(self, cwd):
        bad = [
            "garbage", "[]", b"\xff\xfe",
            {**conv(2), "messages": []},
            {**conv(3), "messages": "hi"},
            {**conv(4), "messages": [{"role": "user"}]},
            {**conv(5), "messages": [{"role": "user", "content": ["a part"]}]},
            {**conv(6), "messages": ["hi"]},
            {**conv(7), "response": None},
            {**conv(8), "response": 5},
            {**conv(9), "id": ""},
            {**conv(10), "id": 10},
        ]
        no_prompt = conv(11)
        del no_prompt["messages"]
        logs([conv(1), *bad, no_prompt], [thumb(1, 1, 1)])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True)]
        assert out.answers == 1
        assert out.skipped_lines >= 10

    def test_blank_lines_are_not_counted(self, cwd):
        logs([conv(1), "", "   "], [thumb(1, 1, 1), "", "\t"])

        out = harvest()

        assert out.skipped_lines == 0
        assert out.feedback_lines == 1

    def test_a_bom_and_windows_line_endings_are_read(self, cwd):
        write_lines(TRACES, [b"\xef\xbb\xbf" + json.dumps(conv(1)).encode(), conv(2)],
                    newline="\r\n")
        write_lines(FEEDBACK, [b"\xef\xbb\xbf" + json.dumps(thumb(1, 1, 1)).encode(),
                               thumb(2, 2, -1)], newline="\r\n")

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True), kto_row(ANSWER, False)]
        assert out.skipped_lines == 0

    @staticmethod
    def _padded(line, size):
        """``line`` with a filler field, so that its JSON is exactly ``size`` bytes."""
        line = {**line, "x_pad": ""}
        line["x_pad"] = "x" * (size - len(json.dumps(line)))
        assert len(json.dumps(line)) == size
        return line

    def test_a_line_over_the_cap_is_skipped_and_the_next_one_is_read_whole(
        self, cwd, monkeypatch
    ):
        answers = [conv(1), conv(2), conv(3)]
        cap = max(len(json.dumps(line)) for line in answers)
        monkeypatch.setattr(lf, "MAX_LINE_BYTES", cap)  # the longest conversation fits exactly
        huge = thumb(1, 1, 1)
        huge["comment"] = "x" * (3 * cap)  # several times the cap: drained in chunks
        logs(answers, [huge, thumb(2, 2, 1), thumb(3, 3, -1)])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True), kto_row(ANSWER, False)]
        assert out.skipped_lines == 1
        assert out.feedback_lines == 3

    def test_a_line_one_byte_over_the_cap_is_skipped_and_one_at_the_cap_is_read(
        self, cwd, monkeypatch
    ):
        answers = [conv(1), conv(2)]
        cap = max(len(json.dumps(line)) for line in answers)
        monkeypatch.setattr(lf, "MAX_LINE_BYTES", cap)
        over = self._padded(thumb(1, 1, -1), cap + 1)
        exact = self._padded(thumb(2, 2, 1), cap)
        logs(answers, [over, exact])

        out = harvest()

        assert rows(out.datasets.kto) == [kto_row(ANSWER, True)]
        assert out.skipped_lines == 1

    def test_a_file_over_the_cap_is_read_up_to_it(self, cwd, monkeypatch):
        answers = [conv(number, response=f"a{number}") for number in (1, 2, 3, 4)]
        width = max(len(json.dumps(line)) for line in answers)
        lines = [self._padded(thumb(number, number, 1), width) for number in (1, 2, 3, 4)]
        monkeypatch.setattr(lf, "MAX_FILE_BYTES", (width + 1) * 2)  # two lines of either log
        logs(answers, lines)

        out = harvest()

        assert [row["completion"] for row in rows(out.datasets.kto)] == ["a1", "a2"]
        assert out.truncated is True

    def test_a_file_exactly_at_the_cap_is_not_truncated(self, cwd, monkeypatch):
        line = thumb(1, 1, 1)
        monkeypatch.setattr(lf, "MAX_FILE_BYTES", len(json.dumps(line)) + 1)
        logs([conv(1)], [line])

        out = harvest()

        assert len(out.datasets.kto) == 1 and out.truncated is False

    def test_an_id_with_a_nul_or_a_huge_id_is_not_accepted(self, cwd):
        logs([conv(1)], [{**thumb(1, 1, 1), "response_id": "a\x00b"},
                         {**thumb(1, 2, 1), "response_id": "x" * 300}])

        assert harvest().skipped_lines == 2


# ---------------------------------------------------------------------------
# The three datasets
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_an_edit_is_a_preference_a_labelled_pair_and_an_sft_answer(self, cwd):
        logs([conv(1)], [edit(1, 5)])

        data = harvest().datasets

        assert rows(data.dpo) == [
            {"prompt": MESSAGES, "chosen": EDIT, "rejected": ANSWER, "source": "feedback-edit"}
        ]
        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert rows(data.sft) == [
            {"messages": [*MESSAGES, {"role": "assistant", "content": EDIT}],
             "source": "feedback-edit"}
        ]
        assert stamps(data.dpo) == [T0 + 5]
        assert stamps(data.kto) == [T0 + 5, T0 + 5]
        assert stamps(data.sft) == [T0 + 5]

    @pytest.mark.parametrize(("value", "label"), [(1, True), (-1, False)])
    def test_a_thumb_labels_the_answer_and_makes_no_pair(self, cwd, value, label):
        logs([conv(1)], [thumb(1, 7, value)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(ANSWER, label)]
        assert stamps(data.kto) == [T0 + 7]
        assert data.dpo == []
        # a thumb up is an answer rated good: an sft row; a thumb down alone gives none
        assert rows(data.sft) == ([sft_good_row(ANSWER)] if label else [])

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, False), (2, False), (3, None), (4, True), (5, True)],
    )
    def test_a_rating_of_one_or_two_is_bad_four_or_five_good_three_says_nothing(
        self, cwd, value, expected
    ):
        logs([conv(1)], [rating(1, 7, value)])

        data = harvest().datasets

        assert rows(data.kto) == ([] if expected is None else [kto_row(ANSWER, expected)])
        assert data.dpo == []
        assert rows(data.sft) == ([sft_good_row(ANSWER)] if expected else [])

    def test_the_newest_of_a_thumb_and_a_rating_labels_the_answer(self, cwd):
        logs([conv(1), conv(2)], [
            thumb(1, 1, 1), rating(1, 2, 1),   # thumb up, then a rating of 1: bad
            rating(2, 3, 5), thumb(2, 4, -1),  # rating of 5, then thumb down: bad
        ])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(ANSWER, False)]  # both answers collapse
        assert stamps(data.kto) == [T0 + 4]  # the newest line of the two copies

    def test_a_rating_of_three_does_not_override_an_earlier_thumb(self, cwd):
        logs([conv(1)], [thumb(1, 1, -1), rating(1, 2, 3)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 1]

    def test_a_thumb_after_an_edit_does_not_label_the_original_good_again(self, cwd):
        """The edit wins (tests/test_loop_sft_default.py has the whole rule)."""
        logs([conv(1)], [edit(1, 1), thumb(1, 2, 1)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 1, T0 + 1]
        assert len(data.dpo) == 1 and len(data.sft) == 1

    def test_an_edit_after_a_thumb_up_labels_the_original_bad(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), edit(1, 2)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 2, T0 + 2]

    def test_a_row_carries_the_ts_of_the_newest_line_it_came_from(self, cwd):
        logs([conv(1)], [edit(1, 3, "first try"), edit(1, 9, EDIT)])

        data = harvest().datasets

        assert rows(data.dpo)[0]["chosen"] == EDIT
        assert stamps(data.dpo) == stamps(data.sft) == [T0 + 9]
        assert stamps(data.kto) == [T0 + 9, T0 + 9]

    def test_a_cleared_signal_leaves_no_row(self, cwd):
        logs([conv(1), conv(2), conv(3)], [
            thumb(1, 1, 1), thumb(1, 2, None),
            rating(2, 3, 5), rating(2, 4, None),
            edit(3, 5), fb(3, "edit", 6, edited_text=None),
        ])

        data = harvest().datasets

        assert (data.dpo, data.kto, data.sft) == ([], [], [])

    def test_an_edit_that_changes_nothing_gives_no_row_but_a_thumb_still_does(self, cwd):
        logs([conv(1), conv(2)], [
            edit(1, 1, ANSWER), edit(1, 2, f"  {ANSWER}\n"),
            thumb(2, 3, 1), edit(2, 4, ANSWER),
        ])

        data = harvest().datasets

        assert data.dpo == []
        assert rows(data.kto) == [kto_row(ANSWER, True)]  # the thumb on answer 2
        assert rows(data.sft) == [sft_good_row(ANSWER)]  # an edit that changes nothing is no edit

    def test_the_prompt_is_the_whole_conversation_without_the_answer(self, cwd):
        history = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How do I reset my router?"},
        ]
        logs([conv(1, messages=history)], [edit(1, 1)])

        data = harvest().datasets

        assert rows(data.dpo)[0]["prompt"] == history
        assert rows(data.sft)[0]["messages"] == [*history, {"role": "assistant", "content": EDIT}]
        assert rows(data.kto)[0]["prompt"] == history

    def test_extra_keys_of_a_message_are_not_carried_over(self, cwd):
        messages = [{"role": "user", "content": "Q", "name": "ann", "x_zero": {"a": 1}}]
        logs([conv(1, messages=messages)], [thumb(1, 1, 1)])

        assert rows(harvest().datasets.kto)[0]["prompt"] == [{"role": "user", "content": "Q"}]

    def test_rows_do_not_share_their_prompt_objects(self, cwd):
        logs([conv(1)], [edit(1, 1)])

        data = harvest().datasets
        data.dpo[0].row["prompt"][0]["content"] = "changed"

        assert data.kto[0].row["prompt"][0]["content"] == "Be brief."
        assert data.sft[0].row["messages"][0]["content"] == "Be brief."

    def test_answers_that_agree_are_one_row_with_the_newest_ts(self, cwd):
        logs([conv(1), conv(2)], [edit(1, 4), edit(2, 8)])

        data = harvest().datasets

        assert len(data.dpo) == len(data.sft) == 1 and len(data.kto) == 2
        assert stamps(data.dpo) == stamps(data.sft) == [T0 + 8]
        assert stamps(data.kto) == [T0 + 8, T0 + 8]

    def test_the_same_answer_labelled_both_ways_by_two_people_keeps_both_rows(self, cwd):
        logs([conv(1), conv(2)], [thumb(1, 1, 1), thumb(2, 2, -1)])

        assert rows(harvest().datasets.kto) == [kto_row(ANSWER, True), kto_row(ANSWER, False)]

    def test_rows_follow_the_order_the_feedback_first_names_the_answers(self, cwd):
        logs([conv(1, response="a"), conv(2, response="b"), conv(3, response="c")], [
            thumb(3, 1, 1), thumb(1, 2, 1), thumb(2, 3, 1),
        ])

        completions = [row["completion"] for row in rows(harvest().datasets.kto)]

        assert completions == ["c", "a", "b"]

    def test_an_empty_original_answer_gives_no_pair_and_no_bad_row(self, cwd):
        logs([conv(1, response="")], [edit(1, 1), thumb(1, 2, -1)])

        data = harvest().datasets

        assert data.dpo == []
        assert rows(data.kto) == [kto_row(EDIT, True)]
        assert len(data.sft) == 1

    def test_a_line_without_a_usable_ts_gives_a_row_without_one(self, cwd):
        logs([conv(1)], [{**thumb(1, 1, 1), "ts": "yesterday"}])

        assert stamps(harvest().datasets.kto) == [None]

    def test_the_ts_may_be_a_number_or_an_iso_string_with_z(self, cwd):
        logs([conv(1), conv(2, response="two")], [
            {**thumb(1, 1, 1), "ts": T0 + 3},
            {**thumb(2, 1, 1), "ts": "2023-11-14T22:13:25Z"},
        ])

        assert stamps(harvest().datasets.kto) == [T0 + 3, T0 + 5]

    @pytest.mark.parametrize(("fmt", "dataset"), [("dpo", "dpo"), ("kto", "kto"),
                                                  ("chatml", "sft")])
    def test_every_row_is_one_soup_train_reads(self, cwd, fmt, dataset):
        from soup_cli.data.formats import format_to_messages

        logs([conv(1), conv(2)], [edit(1, 1), thumb(2, 2, -1)])

        found = rows(getattr(harvest().datasets, dataset))

        assert found
        for row in found:
            assert format_to_messages(row, fmt) is not None, row

    def test_non_ascii_text_survives_the_files(self, cwd):
        messages = [{"role": "user", "content": "Как сбросить роутер?"}]
        logs([conv(1, messages=messages, response="Выдерните.")],
             [edit(1, 1, "Нажмите reset — на 10 секунд.")])

        out = ls.harvest_from_traces(_state(train_method="sft", min_new_edits=1))

        (row,) = _file_rows(out["pairs_path"])
        assert row["messages"][-1]["content"] == "Нажмите reset — на 10 секунд."
        assert row["messages"][0]["content"] == "Как сбросить роутер?"


class TestIsNew:
    def test_everything_is_new_before_the_first_training_run(self):
        assert lf.is_new(None, None) is True
        assert lf.is_new(5.0, None) is True

    def test_a_row_is_new_when_its_feedback_is_later_than_the_cursor(self):
        assert lf.is_new(11.0, 10.0) is True
        assert lf.is_new(10.0, 10.0) is False
        assert lf.is_new(9.0, 10.0) is False

    def test_a_row_without_a_ts_is_not_new_once_there_is_a_cursor(self):
        assert lf.is_new(None, 10.0) is False

    def test_newest_ts_skips_rows_without_one(self):
        items = [lf.DatasetRow({}, None), lf.DatasetRow({}, 3.0), lf.DatasetRow({}, 2.0)]

        assert lf.newest_ts(items) == 3.0
        assert lf.newest_ts([lf.DatasetRow({}, None)]) is None
        assert lf.newest_ts([]) is None


# ---------------------------------------------------------------------------
# harvest_from_traces: the dataset of the chosen method
# ---------------------------------------------------------------------------


def _three_answers():
    """An edit, a thumb up and a thumb down: dpo 1, kto 4, sft 1."""
    logs([conv(1), conv(2, response="B"), conv(3, response="C")], [
        edit(1, 10), thumb(2, 20, 1), thumb(3, 30, -1),
    ])


def _file_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestHarvest:
    def test_dpo_is_the_default_and_writes_the_edit_pairs(self, cwd):
        _three_answers()

        out = ls.harvest_from_traces(_state())

        assert out["train_method"] == "dpo"
        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (1, 1, T0 + 10)
        assert (out["dpo_pairs"], out["kto_rows"], out["sft_rows"]) == (1, 4, 2)
        assert os.path.basename(out["pairs_path"]).startswith("pairs-")
        assert _file_rows(out["pairs_path"]) == [
            {"prompt": MESSAGES, "chosen": EDIT, "rejected": ANSWER, "source": "feedback-edit"}
        ]

    def test_kto_writes_every_labelled_row(self, cwd):
        _three_answers()

        out = ls.harvest_from_traces(_state(train_method="kto"))

        assert out["train_method"] == "kto"
        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (4, 4, T0 + 30)
        assert (out["dpo_pairs"], out["kto_rows"], out["sft_rows"]) == (1, 4, 2)
        assert os.path.basename(out["pairs_path"]).startswith("kto-")
        assert _file_rows(out["pairs_path"]) == [
            kto_row(EDIT, True), kto_row(ANSWER, False),
            kto_row("B", True), kto_row("C", False),
        ]

    def test_sft_writes_the_corrected_answers_and_the_answers_rated_good(self, cwd):
        _three_answers()

        # One new edit is enough here (the default is 8 edits or 30 rated answers).
        out = ls.harvest_from_traces(_state(train_method="sft", min_new_edits=1))

        assert out["train_method"] == "sft"
        # The cursor also covers the thumb down at +30, which gives no row but was seen.
        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (2, 2, T0 + 30)
        assert (out["dpo_pairs"], out["kto_rows"], out["sft_rows"]) == (1, 4, 2)
        assert os.path.basename(out["pairs_path"]).startswith("sft-")
        assert _file_rows(out["pairs_path"]) == [
            {"messages": [*MESSAGES, {"role": "assistant", "content": EDIT}],
             "source": "feedback-edit"},
            sft_good_row("B"),
        ]

    def test_the_old_keys_are_still_there(self, cwd):
        _three_answers()

        out = ls.harvest_from_traces(_state())

        assert {"pairs_harvested", "new_pairs", "cursor_ts", "pairs_path", "traces_collected",
                "thumbs_pairs"} <= set(out)
        assert out["thumbs_pairs"] == 0

    @pytest.mark.parametrize(
        ("method", "total", "new", "cursor"),
        [("dpo", 1, 0, T0 + 10), ("kto", 4, 2, T0 + 30)],
    )
    def test_new_rows_are_those_after_the_cursor_and_all_rows_train(
        self, cwd, method, total, new, cursor
    ):
        _three_answers()

        out = ls.harvest_from_traces(
            _state(train_method=method, harvest_cursor_ts=T0 + 15, min_new_pairs=1)
        )

        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (
            total, new, cursor
        )
        if new:
            assert len(_file_rows(out["pairs_path"])) == total  # not just the new ones
        else:
            assert out["pairs_path"] is None

    def test_sft_counts_new_rows_by_the_cursor_but_waits_for_edits_or_rated_answers(self, cwd):
        _three_answers()

        out = ls.harvest_from_traces(
            _state(train_method="sft", harvest_cursor_ts=T0 + 15, min_new_pairs=1)
        )

        # Two sft rows (the edit at +10, the thumb up at +20), one of them new.
        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (2, 1, T0 + 30)
        assert (out["new_edits"], out["new_rated"]) == (0, 2)
        # min_new_pairs is not the rule of an sft loop with a feedback log: 0 < 8, 2 < 30.
        assert out["pairs_path"] is None

    def test_a_row_at_exactly_the_cursor_is_not_new(self, cwd):
        _three_answers()

        out = ls.harvest_from_traces(_state(train_method="kto", harvest_cursor_ts=T0 + 20))

        assert (out["pairs_harvested"], out["new_pairs"]) == (4, 1)  # only the thumb at +30

    def test_the_dataset_file_waits_for_min_new_pairs_of_its_own_method(self, cwd):
        _three_answers()

        waiting = ls.harvest_from_traces(
            _state(train_method="kto", harvest_cursor_ts=T0 + 15, min_new_pairs=3)
        )
        assert (waiting["pairs_harvested"], waiting["new_pairs"]) == (4, 2)
        assert waiting["pairs_path"] is None
        assert not os.path.exists(os.path.join(".soup-loops", "pairs"))

        enough = ls.harvest_from_traces(
            _state(train_method="kto", harvest_cursor_ts=T0 + 5, min_new_pairs=3)
        )
        assert enough["new_pairs"] == 4
        assert len(_file_rows(enough["pairs_path"])) == 4

    def test_a_row_without_a_ts_is_new_only_while_there_is_no_cursor(self, cwd):
        logs([conv(1)], [{**thumb(1, 1, 1), "ts": "garbage"}])

        first = ls.harvest_from_traces(_state(train_method="kto"))
        later = ls.harvest_from_traces(_state(train_method="kto", harvest_cursor_ts=5.0))

        assert (first["pairs_harvested"], first["new_pairs"], first["cursor_ts"]) == (1, 1, None)
        assert (later["pairs_harvested"], later["new_pairs"], later["cursor_ts"]) == (1, 0, None)

    def test_no_rows_of_the_method_means_zero_everything(self, cwd):
        logs([conv(1)], [thumb(1, 1, -1)])  # a kto row, but no edit and no answer rated good

        out = ls.harvest_from_traces(_state(train_method="sft", harvest_cursor_ts=5.0))

        assert (out["pairs_harvested"], out["new_pairs"]) == (0, 0)
        assert out["cursor_ts"] is None and out["pairs_path"] is None
        assert (out["dpo_pairs"], out["kto_rows"], out["sft_rows"]) == (0, 1, 0)

    def test_nothing_at_all_still_reports_the_new_keys(self, cwd):
        out = ls.harvest_from_traces(_state())

        assert out["pairs_harvested"] == 0
        assert (out["dpo_pairs"], out["kto_rows"], out["sft_rows"]) == (0, 0, 0)
        assert out["feedback_unknown_response"] == 0
        assert out["feedback_lines_skipped"] == 0
        assert out["traces_collected"] == 0

    def test_feedback_about_unknown_answers_is_counted_in_the_report(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), thumb(50, 2, 1), edit(50, 3)])

        out = ls.harvest_from_traces(_state(train_method="kto"))

        assert out["feedback_unknown_response"] == 2
        assert out["pairs_harvested"] == 1

    def test_skipped_feedback_lines_are_counted_in_the_report(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), "junk", {**thumb(1, 2, 1), "schema_version": 7}])

        out = ls.harvest_from_traces(_state(train_method="kto"))

        assert out["feedback_lines_skipped"] == 2

    def test_conversation_records_count_as_traces(self, cwd):
        legacy = [{"id": "a", "prompt": "Q", "response": "good", "signal": "thumbs_up"},
                  {"id": "b", "prompt": "Q", "response": "bad", "signal": "thumbs_down"}]
        logs([*legacy, conv(1), conv(2), conv(3)], [thumb(1, 1, 1)])

        out = ls.harvest_from_traces(_state(train_method="kto"))

        assert out["traces_collected"] == 5

    def test_serve_traces_and_edits_make_one_dpo_dataset(self, cwd):
        legacy = [{"id": "a", "prompt": "Q", "response": "good", "signal": "thumbs_up", "ts": 1},
                  {"id": "b", "prompt": "Q", "response": "bad", "signal": "thumbs_down", "ts": 2}]
        logs([*legacy, conv(1)], [edit(1, 40)])

        out = ls.harvest_from_traces(_state())

        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (2, 2, T0 + 40)
        assert [row["source"] for row in _file_rows(out["pairs_path"])] == [
            "thumbs_up", "feedback-edit"
        ]

    def test_a_serve_trace_pair_and_the_same_edit_are_one_pair(self, cwd):
        legacy = [{"id": "a", "prompt": "Q", "response": "good", "signal": "thumbs_up"},
                  {"id": "b", "prompt": "Q", "response": "bad", "signal": "thumbs_down"}]
        asked = [{"role": "user", "content": "Q"}]
        logs([*legacy, conv(1, messages=asked, response="bad")], [edit(1, 40, "good")])

        out = ls.harvest_from_traces(_state())

        assert out["pairs_harvested"] == 1
        assert out["dpo_pairs"] == 1
        (row,) = _file_rows(out["pairs_path"])
        assert row["prompt"] == "Q" and row["source"] == "thumbs_up"

    def test_the_thumbs_database_joins_the_dpo_dataset_only(self, cwd):
        from soup_cli.utils import local_rl

        local_rl.init_local_rl_db("t.db")
        with contextlib.closing(sqlite3.connect("t.db")) as conn:
            conn.executemany(
                "INSERT INTO thumbs (ts, prompt, response, thumb) VALUES (?, ?, ?, ?)",
                [(T0 + 1, "P", "good", "up"), (T0 + 2, "P", "bad", "down")],
            )
            conn.commit()
        _three_answers()

        dpo = ls.harvest_from_traces(_state(thumbs_db="t.db"))
        kto = ls.harvest_from_traces(_state(thumbs_db="t.db", train_method="kto"))

        assert (dpo["pairs_harvested"], dpo["thumbs_pairs"], dpo["dpo_pairs"]) == (2, 1, 2)
        assert (kto["pairs_harvested"], kto["thumbs_pairs"], kto["dpo_pairs"]) == (4, 1, 2)

    def test_old_thumb_traces_feed_sft_but_not_kto(self, cwd):
        legacy = [{"id": "a", "prompt": "Q", "response": "good", "signal": "thumbs_up"},
                  {"id": "b", "prompt": "Q", "response": "bad", "signal": "thumbs_down"}]
        write_lines(TRACES, legacy)

        kto = ls.harvest_from_traces(_state(train_method="kto"))
        sft = ls.harvest_from_traces(_state(train_method="sft"))

        assert kto["pairs_harvested"] == 0 and kto["dpo_pairs"] == 1
        assert sft["pairs_harvested"] == 1 and sft["dpo_pairs"] == 1  # the thumbs-up answer only
        (row,) = _file_rows(sft["pairs_path"])
        assert row["messages"] == [{"role": "user", "content": "Q"},
                                   {"role": "assistant", "content": "good"}]


class TestFeedbackLogResolution:
    def test_the_default(self, cwd):
        assert ls.feedback_log_source(_state()) == (".soup-loops/feedback.jsonl", "default")

    def test_loop_yaml_replaces_the_default(self, cwd):
        assert ls.feedback_log_source(_state(feedback_log="fb.jsonl")) == (
            "fb.jsonl", "feedback_log"
        )

    def test_the_environment_beats_loop_yaml(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "env.jsonl")

        assert ls.feedback_log_source(_state(feedback_log="fb.jsonl")) == (
            "env.jsonl", "SOUP_LOOP_FEEDBACK_LOG"
        )

    def test_an_empty_environment_variable_is_not_a_choice(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "")

        assert ls.feedback_log_source(_state(feedback_log="fb.jsonl"))[1] == "feedback_log"

    def test_the_harvest_reads_the_resolved_log(self, cwd, monkeypatch):
        write_lines(TRACES, [conv(1)])
        write_lines("yaml.jsonl", [thumb(1, 1, 1)])
        write_lines("env.jsonl", [thumb(1, 1, -1)])
        write_lines(FEEDBACK, [edit(1, 1)])

        by_default = ls.harvest_from_traces(_state(train_method="sft"))
        by_yaml = ls.harvest_from_traces(_state(train_method="kto", feedback_log="yaml.jsonl"))
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "env.jsonl")
        by_env = ls.harvest_from_traces(_state(train_method="kto", feedback_log="yaml.jsonl"))

        assert by_default["pairs_harvested"] == 1
        assert _file_rows(by_yaml["pairs_path"]) == [kto_row(ANSWER, True)]
        assert _file_rows(by_env["pairs_path"]) == [kto_row(ANSWER, False)]

    def test_a_missing_log_harvests_nothing(self, cwd):
        write_lines(TRACES, [conv(1)])

        out = ls.harvest_from_traces(_state(train_method="kto", feedback_log="not-yet.jsonl"))

        assert out["pairs_harvested"] == 0

    def test_the_conversations_come_from_the_trace_directory_in_force(self, cwd):
        write_lines(os.path.join("logs", "c.jsonl"), [conv(1)])
        write_lines(FEEDBACK, [thumb(1, 1, 1)])

        elsewhere = ls.harvest_from_traces(_state(train_method="kto", trace_dir="logs"))
        default = ls.harvest_from_traces(_state(train_method="kto"))

        assert elsewhere["pairs_harvested"] == 1
        assert default["pairs_harvested"] == 0
        assert default["feedback_unknown_response"] == 1

    def test_notice_is_silent_for_a_file_and_for_a_missing_default(self, cwd):
        assert ls.feedback_log_notice(_state()) is None
        write_lines("fb.jsonl", [thumb(1, 1, 1)])
        assert ls.feedback_log_notice(_state(feedback_log="fb.jsonl")) is None

    def test_notice_warns_for_a_configured_log_that_is_not_there_yet(self, cwd, monkeypatch):
        notice = ls.feedback_log_notice(_state(feedback_log="fb.jsonl"))
        assert notice.startswith("feedback_log: no feedback log at 'fb.jsonl' yet")
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "env.jsonl")
        assert ls.feedback_log_notice(_state()).startswith(
            "SOUP_LOOP_FEEDBACK_LOG: no feedback log at 'env.jsonl' yet"
        )

    def test_notice_refuses_a_directory_naming_where_the_path_came_from(self, cwd, monkeypatch):
        os.makedirs("fb")
        with pytest.raises(ValueError, match=r"^feedback_log: 'fb' is a directory, not a file$"):
            ls.feedback_log_notice(_state(feedback_log="fb"))
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "fb")
        with pytest.raises(
            ValueError, match=r"^SOUP_LOOP_FEEDBACK_LOG: 'fb' is a directory, not a file$"
        ):
            ls.feedback_log_notice(_state())

    def test_notice_refuses_a_default_path_that_is_a_directory(self, cwd):
        os.makedirs(os.path.join(".soup-loops", "feedback.jsonl"))

        with pytest.raises(
            ValueError,
            match=r"^\.soup-loops/feedback\.jsonl \(the default\): .* is a directory, not a file$",
        ):
            ls.feedback_log_notice(_state())


# ---------------------------------------------------------------------------
# The train stage renders the config of the chosen method
# ---------------------------------------------------------------------------


class _Trainer:
    """Stand-in for the ``soup train`` subprocess: records the config and the rows."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.configs: list = []
        self.trained_on: list = []

    def __call__(self, argv, **kwargs):
        with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.configs.append(config)
        self.trained_on.append(_file_rows(config["data"]["train"]))
        if self.returncode == 0:
            os.makedirs(config["output"], exist_ok=True)
            with open(os.path.join(config["output"], "adapter_config.json"), "w") as handle:
                json.dump({"base_model_name_or_path": config["base"]}, handle)
        return SimpleNamespace(returncode=self.returncode, stderr=b"")


class TestTrainStage:
    @pytest.mark.parametrize(
        ("method", "task", "fmt", "training"),
        [
            ("dpo", "dpo", "dpo", {"epochs": 1, "batch_size": 1, "lr": 5.0e-5}),
            ("kto", "kto", "kto", {"epochs": 1, "batch_size": 2, "lr": 5.0e-5}),
            ("sft", "sft", "chatml", {
                "epochs": 8, "batch_size": 2, "lr": 5.0e-4,
                "gradient_accumulation_steps": 4, "seed": 42,
                "lora": {"r": 64, "alpha": 128, "target_modules": "all-linear"},
                # quantization auto with no measurable GPU (the test conftest): the safe side.
                "quantization": "4bit",
            }),
        ],
    )
    def test_the_config_names_the_task_the_format_and_the_safe_settings(
        self, cwd, method, task, fmt, training
    ):
        rendered = yaml.safe_load(
            ls._render_train_yaml(_state(train_method=method), "d.jsonl", "out")
        )

        assert rendered == {
            "base": "org/base",
            "task": task,
            "data": {"train": "d.jsonl", "format": fmt, "val_split": 0},
            "training": training,
            "output": "out",
        }

    def test_the_kto_config_is_one_soup_train_accepts(self, cwd):
        """TRL refuses a per-device KTO batch of 1 (#1420); the schema checks it on load."""
        from soup_cli.config.schema import SoupConfig

        rendered = yaml.safe_load(
            ls._render_train_yaml(_state(train_method="kto"), "d.jsonl", "out")
        )

        assert SoupConfig(**rendered).task == "kto"

    @pytest.mark.parametrize("method", ["dpo", "sft"])
    def test_the_other_configs_are_ones_soup_train_accepts(self, cwd, method):
        from soup_cli.config.schema import SoupConfig

        rendered = yaml.safe_load(
            ls._render_train_yaml(_state(train_method=method), "d.jsonl", "out")
        )

        assert SoupConfig(**rendered).task == method

    def test_the_dpo_renderer_stays_dpo_whatever_the_state_says(self, cwd):
        rendered = yaml.safe_load(
            ls._render_dpo_yaml(_state(train_method="kto"), "p.jsonl", "out")
        )

        assert (rendered["task"], rendered["data"]["format"]) == ("dpo", "dpo")

    def test_an_unknown_base_model_still_refuses(self, cwd):
        with pytest.raises(ValueError):
            ls._render_train_yaml(
                _state(served_model="registry://nope", base_model=None, train_method="kto"),
                "d.jsonl", "out",
            )

    @pytest.mark.parametrize(
        ("method", "task", "fmt", "count"),
        [("dpo", "dpo", "dpo", 1), ("kto", "kto", "kto", 4), ("sft", "sft", "chatml", 2)],
    )
    def test_harvest_then_train_runs_soup_train_on_the_chosen_dataset(
        self, cwd, monkeypatch, method, task, fmt, count
    ):
        _three_answers()
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        state = _state(train_method=method, min_new_edits=1)  # (only an sft loop reads it)

        out = ls.train_dpo_from_pairs(state, ls.harvest_from_traces(state))

        assert out["skipped"] is False and out["run_id"], out
        (config,) = trainer.configs
        assert (config["task"], config["data"]["format"], config["base"]) == (
            task, fmt, "org/base"
        )
        assert config["data"]["val_split"] == 0
        assert len(trainer.trained_on[0]) == count
        assert os.path.basename(config["data"]["train"]).startswith(
            {"dpo": "pairs-", "kto": "kto-", "sft": "sft-"}[method]
        )

    def test_too_few_new_rows_skip_with_the_methods_own_wording(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        ctx = {"pairs_harvested": 4, "new_pairs": 1, "pairs_path": "unused"}

        kto = ls.train_dpo_from_pairs(_state(train_method="kto", min_new_pairs=3), ctx)
        sft = ls.train_dpo_from_pairs(_state(train_method="sft", min_new_pairs=3), ctx)
        dpo = ls.train_dpo_from_pairs(_state(min_new_pairs=3), ctx)

        assert kto["notes"] == (
            "not_enough_new_data: 1 new kto rows since the last training run, "
            "training needs 3 (soup loop init --min-new-pairs)"
        )
        assert sft["notes"].startswith("not_enough_new_data: 1 new sft rows since")
        assert dpo["notes"].startswith("not_enough_new_data: 1 new preference pairs since")
        assert trainer.configs == []

    def test_no_rows_skip_with_the_methods_own_wording(self, cwd):
        ctx = {"pairs_harvested": 0, "new_pairs": 0}

        assert ls.train_dpo_from_pairs(_state(train_method="kto"), ctx)["notes"] == (
            "no_kto_rows: 0 valid kto rows harvested, training skipped"
        )
        assert ls.train_dpo_from_pairs(_state(train_method="sft"), ctx)["notes"] == (
            "no_sft_rows: 0 valid sft rows harvested, training skipped"
        )
        assert ls.train_dpo_from_pairs(_state(), ctx)["notes"] == (
            "no_preference_pairs: 0 valid pairs harvested, training skipped"
        )


class TestTheLoopOverFeedback:
    """The pre-wired stages through run_once: the attempt cursor, per method.

    A training run moves ``attempt_cursor_ts`` (the newest feedback it read); the data cursor
    ``harvest_cursor_ts`` follows the version in production and moves on a promotion
    (tests/test_loop_verdict_v2.py).
    """

    @staticmethod
    def _config():
        return WatchConfig(
            harvest_fn=ls.harvest_from_traces,
            train_fn=ls.train_dpo_from_pairs,
            gate_fn=lambda state, ctx: {"gate_verdict": "OK"},
            deploy_fn=lambda state, ctx: {"deployed": False, "canary_verdict": None},
            cost_fn=lambda state: 0.0,
        )

    def test_a_kto_loop_trains_on_news_then_waits_then_trains_on_everything(
        self, cwd, monkeypatch
    ):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        state = _state(train_method="kto")
        _three_answers()

        state, first, _ = run_once(state, self._config())
        assert first.run_id and first.pairs_harvested == 4
        assert state.attempt_cursor_ts == T0 + 30

        state, quiet, _ = run_once(state, self._config())
        assert quiet.run_id is None
        assert "not_enough_new_data: 0 new kto rows since the last training run" in quiet.notes
        assert state.attempt_cursor_ts == T0 + 30

        with open(FEEDBACK, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(thumb(2, 50, -1)) + "\n")  # B was bad after all
        state, third, _ = run_once(state, self._config())

        assert third.run_id
        labels = [(row["completion"], row["label"]) for row in trainer.trained_on[-1]]
        assert labels == [(EDIT, True), (ANSWER, False), ("B", False), ("C", False)]
        assert state.attempt_cursor_ts == T0 + 50
        assert len(trainer.trained_on) == 2

    def test_a_failed_training_leaves_the_cursors_so_the_feedback_is_tried_again(
        self, cwd, monkeypatch
    ):
        trainer = _Trainer(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        _three_answers()

        state, failed, _ = run_once(
            _state(train_method="sft", min_new_edits=1), self._config()
        )

        assert failed.run_id is None
        assert state.attempt_cursor_ts is None
        trainer.returncode = 0
        state, retry, _ = run_once(state, self._config())
        # (+30 is the thumb down, which gives no sft row but was seen by the run.)
        assert retry.run_id and state.attempt_cursor_ts == T0 + 30

    def test_min_new_pairs_counts_rows_of_the_chosen_method(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        logs([conv(1), conv(2, response="two")], [thumb(1, 1, 1), thumb(2, 2, -1)])

        state, waiting, _ = run_once(
            _state(train_method="kto", min_new_pairs=3), self._config()
        )
        assert waiting.run_id is None
        assert "not_enough_new_data: 2 new kto rows since the last training run" in waiting.notes
        assert state.attempt_cursor_ts is None
        assert trainer.configs == []

        with open(FEEDBACK, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(edit(1, 4)) + "\n")  # a correction adds a row
        state, trained, _ = run_once(state, self._config())

        assert trained.run_id
        assert len(trainer.trained_on[-1]) == 3
        assert state.attempt_cursor_ts == T0 + 4


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------


class TestLoopStateFields:
    def test_defaults_keep_dpo_and_the_default_log(self):
        state = _state()

        assert (state.feedback_log, state.train_method) == (None, "dpo")

    def test_the_fields_come_after_the_data_cursor_fields_in_order(self):
        names = list(LoopState.__dataclass_fields__)

        assert names[names.index("harvest_cursor_ts"):][:4] == [
            "harvest_cursor_ts", "min_new_pairs", "feedback_log", "train_method"
        ]

    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_the_methods(self, method):
        assert _state(train_method=method).train_method == method

    @pytest.mark.parametrize("value", ["ppo", "", "DPO", "kto ", None, 1, True, ["dpo"]])
    def test_any_other_method_is_refused_naming_the_field(self, value):
        message = r"train_method must be one of \['dpo', 'kto', 'sft'\]"
        with pytest.raises(ValueError, match=message):
            _state(train_method=value)

    @pytest.mark.parametrize("value", ["", "a\x00b", "x" * 600])
    def test_a_bad_feedback_log_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="feedback_log"):
            _state(feedback_log=value)

    def test_a_non_string_feedback_log_is_refused(self):
        with pytest.raises(TypeError, match="feedback_log must be str"):
            _state(feedback_log=5)

    def test_init_state_threads_both_fields_and_the_file_keeps_them(self, cwd):
        state, path = init_state(
            "m", "e", "b", feedback_log="fb.jsonl", train_method="kto",
            path=str(cwd / "loop.json"),
        )

        assert (state.feedback_log, state.train_method) == ("fb.jsonl", "kto")
        reread = read_state(path)
        assert (reread.feedback_log, reread.train_method) == ("fb.jsonl", "kto")

    def test_init_state_defaults(self, cwd):
        state, _ = init_state("m", "e", "b", path=str(cwd / "loop.json"))

        assert (state.feedback_log, state.train_method) == (None, "dpo")

    def test_a_loop_yaml_from_before_the_fields_still_loads(self, cwd):
        legacy = {
            "served_model": "m", "eval_suite": "e", "baseline": "b", "status": "running",
            "iteration_count": 3, "min_new_pairs": 2, "harvest_cursor_ts": 12.5,
            "created_at": "x", "updated_at": "y",
        }
        (cwd / "legacy.json").write_text(json.dumps(legacy), "utf-8")

        state = read_state(str(cwd / "legacy.json"))

        assert (state.feedback_log, state.train_method) == (None, "dpo")
        assert (state.iteration_count, state.min_new_pairs) == (3, 2)

    def test_an_invalid_value_in_the_file_is_reported_with_its_name(self, cwd):
        bad = {"served_model": "m", "eval_suite": "e", "baseline": "b", "train_method": "ppo"}
        (cwd / "bad.json").write_text(json.dumps(bad), "utf-8")

        with pytest.raises(ValueError, match="state file has invalid contents: train_method"):
            read_state(str(cwd / "bad.json"))

    def test_the_fields_round_trip_through_write_state(self, cwd):
        state, path = init_state("m", "e", "b", path=str(cwd / "loop.json"))
        write_state(replace(state, feedback_log="a.jsonl", train_method="sft"), path)

        reread = read_state(path)

        assert (reread.feedback_log, reread.train_method) == ("a.jsonl", "sft")


# ---------------------------------------------------------------------------
# soup loop init / status / watch
# ---------------------------------------------------------------------------


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


class TestCli:
    def test_the_defaults_are_sft_and_the_default_log(self, cwd):
        result, text = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert (state.train_method, state.feedback_log) == ("sft", None)
        assert "train_method: sft" in text
        assert "feedback_log: .soup-loops/feedback.jsonl (from default)" in text

    def test_the_flags_are_stored(self, cwd):
        write_lines("fb.jsonl", [thumb(1, 1, 1)])

        result, text = _cli(*_INIT, "--train-method", "kto", "--feedback-log", "fb.jsonl")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert (state.train_method, state.feedback_log) == ("kto", "fb.jsonl")
        assert "train_method: kto" in text
        assert "feedback_log: fb.jsonl (from --feedback-log)" in text

    @pytest.mark.parametrize(("typed", "stored"), [("SFT", "sft"), (" Kto ", "kto")])
    def test_the_method_is_case_and_space_insensitive(self, cwd, typed, stored):
        result, _ = _cli(*_INIT, "--train-method", typed)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().train_method == stored

    @pytest.mark.parametrize("value", ["ppo", "dpo,kto", "orpo", ""])
    def test_any_other_method_is_refused(self, cwd, value):
        result, text = _cli(*_INIT, "--train-method", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"--train-method must be one of dpo, kto, sft, got '{value}'" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_an_empty_feedback_log_is_refused(self, cwd):
        result, text = _cli(*_INIT, "--feedback-log", "  ")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--feedback-log must not be empty" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_a_directory_is_refused_naming_the_flag(self, cwd):
        os.makedirs("fb")

        result, text = _cli(*_INIT, "--feedback-log", "fb")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--feedback-log: fb is a directory, not a file" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_a_log_that_is_not_there_yet_is_a_warning(self, cwd):
        result, text = _cli(*_INIT, "--feedback-log", "fb.jsonl")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "no feedback log at fb.jsonl yet" in text
        assert read_state().feedback_log == "fb.jsonl"

    def test_help_names_both_flags(self, cwd):
        result, text = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--feedback-log" in text and "--train-method" in text

    def test_status_shows_the_method_and_where_the_log_comes_from(self, cwd, monkeypatch):
        write_lines("fb.jsonl", [thumb(1, 1, 1)])
        _cli(*_INIT, "--train-method", "kto", "--feedback-log", "fb.jsonl")

        result, text = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"train_method\W+kto", text), text
        assert re.search(r"feedback_log\W+fb\.jsonl \(from --feedback-log\)", text), text

        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "env.jsonl")
        _, text = _cli("status")
        assert re.search(r"feedback_log\W+env\.jsonl \(from SOUP_LOOP_FEEDBACK_LOG\)", text), text

    def test_status_of_a_default_loop_names_the_defaults(self, cwd):
        _cli(*_INIT)

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"train_method\W+sft", text), text
        assert re.search(r"feedback_log\W+\.soup-loops/feedback\.jsonl \(from default\)", text)

    def test_status_of_a_state_written_before_the_fields(self, cwd):
        _cli(*_INIT)
        path = os.path.join(".soup", "loop.yaml")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        del data["feedback_log"], data["train_method"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"train_method\W+dpo", text), text

    def test_watch_pre_wired_refuses_a_feedback_log_that_is_a_directory(self, cwd, monkeypatch):
        _cli(*_INIT)
        os.makedirs("fb")
        monkeypatch.setenv("SOUP_LOOP_FEEDBACK_LOG", "fb")

        result, text = _cli("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid feedback log: SOUP_LOOP_FEEDBACK_LOG: fb is a directory, not a file" in (
            text.replace("'", "")
        )
        assert read_state().iteration_count == 0

    def test_watch_pre_wired_names_the_loop_yaml_field(self, cwd):
        os.makedirs("fb")
        _cli(*_INIT)
        write_state(replace(read_state(), feedback_log="fb"))

        result, text = _cli("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid feedback log: feedback_log:" in text

    def test_watch_pre_wired_warns_for_a_configured_log_not_there_yet(self, cwd, monkeypatch):
        _cli(*_INIT, "--feedback-log", "fb.jsonl")
        monkeypatch.setattr("soup_cli.commands.loop.watch", lambda cfg: (read_state(), 0))

        result, text = _cli("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "feedback_log: no feedback log at 'fb.jsonl' yet" in text

    def test_watch_pre_wired_is_quiet_with_the_default_log_absent(self, cwd, monkeypatch):
        _cli(*_INIT)
        monkeypatch.setattr("soup_cli.commands.loop.watch", lambda cfg: (read_state(), 0))

        result, text = _cli("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "warning" not in text.lower()
