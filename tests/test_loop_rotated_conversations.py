"""The loop reads the conversation records ``soup serve`` rotated away.

``soup serve --trace-log`` renames ``serve.jsonl`` to ``serve.jsonl.1`` at 100 MB (the older
``.1`` is deleted). The loop read only names ending ``.jsonl``, so feedback given after a
rotation on an answer in the rotated file became ``feedback_unknown_response`` and was dropped
from training and from the canary's statistics. Now ``<name>.jsonl.<n>`` is read too, the
highest ``n`` (the oldest) first, then the live file, with the same per-line and per-file caps.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

import soup_cli.utils.loop_feedback as lf

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc).timestamp()


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def conv(number: int, response: str = "Unplug it.") -> dict:
    return {
        "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
        "ts": iso(number), "model": "base", "version": 1, "adapter": None,
        "canary_bucket": None,
        "messages": [{"role": "user", "content": f"question {number}"}],
        "response": response, "finish_reason": "stop", "latency_ms": 1.0,
        "tokens": {"prompt": 1, "completion": 1}, "options": {}, "source": "soup-serve",
    }


def edit(number: int, text: str = "Hold the reset button.") -> dict:
    return {
        "schema_version": 1, "id": uid(1000 + number), "response_id": uid(number),
        "ts": iso(100 + number), "kind": "edit", "source": "playground", "edited_text": text,
    }


def write(path, *lines) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))
    return path


@pytest.fixture
def traces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return os.path.join(".soup-loops", "traces")


class TestConversationFiles:
    def test_a_rotated_file_comes_before_its_live_file_oldest_first(self, traces):
        for name in ("serve.jsonl", "serve.jsonl.1", "serve.jsonl.2", "serve.jsonl.10"):
            write(os.path.join(traces, name), conv(1))

        names = [os.path.basename(path) for path in lf.conversation_files(traces)]

        assert names == ["serve.jsonl.10", "serve.jsonl.2", "serve.jsonl.1", "serve.jsonl"]

    def test_each_log_keeps_its_own_group(self, traces):
        for name in ("b.jsonl", "a.jsonl.1", "a.jsonl", "b.jsonl.1"):
            write(os.path.join(traces, name), conv(1))

        names = [os.path.basename(path) for path in lf.conversation_files(traces)]

        assert names == ["a.jsonl.1", "a.jsonl", "b.jsonl.1", "b.jsonl"]

    @pytest.mark.parametrize(
        "name", ["serve.jsonl.bak", "serve.jsonl.1x", "serve.jsonl.-1",
                 "serve.jsonl.1.gz", "serve.json.1", "serve.1"],
    )
    def test_names_that_are_not_rotations_are_ignored(self, traces, name):
        write(os.path.join(traces, name), conv(1))

        assert lf.conversation_files(traces) == []

    def test_a_rotated_file_without_a_live_file_is_still_read(self, traces):
        write(os.path.join(traces, "serve.jsonl.1"), conv(1))

        assert [os.path.basename(p) for p in lf.conversation_files(traces)] == ["serve.jsonl.1"]


class TestFeedbackOnARotatedAnswer:
    def test_an_edit_for_an_answer_in_the_rotated_file_reaches_the_dataset(self, traces):
        write(os.path.join(traces, "serve.jsonl.1"), conv(1))
        write(os.path.join(traces, "serve.jsonl"), conv(2))
        feedback = write(os.path.join(".soup-loops", "feedback.jsonl"), edit(1), edit(2))

        out = lf.harvest_feedback(traces, feedback)

        assert out.datasets.unknown_response == 0
        assert out.answers == 2
        assert len(out.datasets.edits) == 2
        assert len(out.datasets.sft) == 2

    def test_without_the_rotated_file_the_edit_is_unknown(self, traces):
        write(os.path.join(traces, "serve.jsonl"), conv(2))
        feedback = write(os.path.join(".soup-loops", "feedback.jsonl"), edit(1))

        out = lf.harvest_feedback(traces, feedback)

        assert out.datasets.unknown_response == 1
