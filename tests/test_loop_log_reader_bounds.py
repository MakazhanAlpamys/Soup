"""What the loop's log readers refuse: a ``ts`` far in the future, and a line over 1 MiB.

* A record whose ``ts`` is more than 24 hours ahead of now is skipped with a counted
  warning. Left in, it would be the newest ``ts`` and freeze the data cursor: every later row
  would be "not new" and the loop would wait for ever. Now + 1 hour is accepted (a client
  whose clock runs ahead); year 9999 and now + 2 days are not.
* A JSONL line longer than ``MAX_LINE_BYTES`` is skipped and counted without being kept in
  memory; a line of exactly that size is read.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import pytest

from soup_cli.data.traces.parsers import (
    MAX_FUTURE_TS_SEC,
    parse_soup_serve,
    ts_is_in_the_future,
)
from soup_cli.utils import loop_feedback as lf

YEAR_9999 = 253402300799.0
DAY = 24 * 3600


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def feedback_line(number: int, ts) -> dict:
    return {
        "schema_version": 1, "id": uid(900 + number), "response_id": uid(number), "ts": ts,
        "kind": "thumb", "value": 1, "source": "app",
    }


def conversation_line(number: int, ts) -> dict:
    return {
        "schema_version": 1, "id": uid(number), "ts": ts, "model": "base",
        "messages": [{"role": "user", "content": f"q{number}"}], "response": f"a{number}",
        "source": "soup-serve",
    }


def write_jsonl(path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


class TestTheRule:
    def test_the_tolerance_is_24_hours(self):
        assert MAX_FUTURE_TS_SEC == 24 * 3600

    @pytest.mark.parametrize(
        ("offset", "future"),
        [(-3600, False), (0, False), (3600, False), (DAY - 60, False),
         (DAY + 60, True), (2 * DAY, True)],
    )
    def test_the_boundary(self, offset, future):
        now = 1_800_000_000.0

        assert ts_is_in_the_future(now + offset, now) is future

    def test_year_9999_is_in_the_future(self):
        assert ts_is_in_the_future(YEAR_9999) is True

    def test_no_ts_is_not_in_the_future(self):
        assert ts_is_in_the_future(None) is False


# ---------------------------------------------------------------------------
# The feedback log
# ---------------------------------------------------------------------------


class TestFeedbackLog:
    def test_a_future_line_is_skipped_counted_and_warned_about(self, tmp_path, caplog):
        now = time.time()
        path = tmp_path / "feedback.jsonl"
        write_jsonl(path, [
            feedback_line(1, iso(now - 60)),
            feedback_line(2, YEAR_9999),
            feedback_line(3, iso(now + 2 * DAY)),
            feedback_line(4, now + 3600),
        ])

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.loop_feedback"):
            log = lf.read_feedback(str(path))

        assert set(log.by_response) == {uid(1), uid(4)}
        assert log.stats.future_ts == 2
        assert log.stats.skipped == 2
        assert "2 feedback record(s) skipped" in caplog.text
        assert "24 h in the future" in caplog.text

    def test_the_cursor_follows_the_newest_real_row_not_the_year_9999_one(self, tmp_path):
        now = time.time()
        path = tmp_path / "feedback.jsonl"
        write_jsonl(path, [
            feedback_line(1, now - 120),
            feedback_line(2, YEAR_9999),
            feedback_line(3, now - 60),
        ])

        log = lf.read_feedback(str(path))
        cursor = max(record.thumb_ts for record in log.by_response.values())

        assert cursor == pytest.approx(now - 60, abs=1)
        assert lf.is_new(now - 30, cursor) is True  # a later row is still news

    def test_now_plus_one_hour_is_accepted(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        write_jsonl(path, [feedback_line(1, time.time() + 3600)])

        log = lf.read_feedback(str(path))

        assert set(log.by_response) == {uid(1)} and log.stats.future_ts == 0

    def test_no_warning_when_nothing_was_skipped(self, tmp_path, caplog):
        path = tmp_path / "feedback.jsonl"
        write_jsonl(path, [feedback_line(1, iso(time.time() - 5))])

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.loop_feedback"):
            lf.read_feedback(str(path))

        assert "skipped" not in caplog.text


# ---------------------------------------------------------------------------
# The conversation records and the serve-trace parser
# ---------------------------------------------------------------------------


class TestConversationRecords:
    def test_a_future_record_is_skipped_counted_and_warned_about(self, tmp_path, caplog):
        now = time.time()
        write_jsonl(tmp_path / "traces" / "serve.jsonl", [
            conversation_line(1, iso(now - 60)),
            conversation_line(2, YEAR_9999),
            conversation_line(3, iso(now + 2 * DAY)),
            conversation_line(4, iso(now + 3600)),
        ])

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.loop_feedback"):
            log = lf.read_conversations(str(tmp_path / "traces"))

        assert set(log.records) == {uid(1), uid(4)}
        assert log.stats.future_ts == 2
        assert "2 conversation record(s) skipped" in caplog.text


class TestServeTraceParser:
    def test_future_lines_are_skipped_with_a_counted_warning(self, tmp_path, caplog):
        now = time.time()
        write_jsonl(tmp_path / "traces" / "serve.jsonl", [
            conversation_line(1, iso(now - 60)),
            conversation_line(2, YEAR_9999),
            conversation_line(3, iso(now + 2 * DAY)),
            conversation_line(4, iso(now + 3600)),
        ])

        with caplog.at_level(logging.WARNING, logger="soup_cli.data.traces.parsers"):
            traces = list(parse_soup_serve(str(tmp_path / "traces")))

        assert [trace.trace_id for trace in traces] == [uid(1), uid(4)]
        assert "2 trace line(s) skipped" in caplog.text

    def test_the_newest_ts_of_the_kept_traces_is_a_real_one(self, tmp_path):
        now = time.time()
        write_jsonl(tmp_path / "traces" / "serve.jsonl", [
            conversation_line(1, iso(now - 60)),
            conversation_line(2, YEAR_9999),
        ])

        traces = list(parse_soup_serve(str(tmp_path / "traces")))

        assert max(trace.ts for trace in traces) == pytest.approx(now - 60, abs=1)


# ---------------------------------------------------------------------------
# The 1 MiB line cap
# ---------------------------------------------------------------------------


def line_of(size: int, number: int = 5) -> bytes:
    """A valid feedback line of exactly ``size`` bytes (without its newline)."""
    base = json.dumps(feedback_line(number, iso(time.time() - 5))).encode("utf-8")
    assert base.endswith(b"}") and size >= len(base)
    line = base[:-1] + b" " * (size - len(base)) + b"}"
    assert len(line) == size
    return line


class TestLineCap:
    def test_the_cap_is_one_mebibyte(self):
        assert lf.MAX_LINE_BYTES == 1024 * 1024

    def test_a_line_of_exactly_the_cap_is_read(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        path.write_bytes(line_of(lf.MAX_LINE_BYTES) + b"\n")

        log = lf.read_feedback(str(path))

        assert set(log.by_response) == {uid(5)}
        assert log.stats.skipped == 0

    def test_one_byte_over_is_skipped_and_counted_and_its_neighbours_are_read(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        good_before = json.dumps(feedback_line(1, iso(time.time() - 9))).encode("utf-8")
        good_after = json.dumps(feedback_line(2, iso(time.time() - 8))).encode("utf-8")
        path.write_bytes(
            good_before + b"\n" + line_of(lf.MAX_LINE_BYTES + 1) + b"\n" + good_after + b"\n"
        )

        log = lf.read_feedback(str(path))

        assert set(log.by_response) == {uid(1), uid(2)}
        assert log.stats.lines == 3
        assert log.stats.skipped == 1

    def test_a_huge_line_without_a_newline_at_the_end_of_the_file_is_skipped(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        path.write_bytes(line_of(lf.MAX_LINE_BYTES * 2))

        log = lf.read_feedback(str(path))

        assert log.by_response == {} and log.stats.skipped == 1
