"""`soup loop` trains on sft by default: corrected answers and answers people rated good.

Decided from measured recipe experiments:

- ``soup loop init --train-method`` defaults to ``sft`` for a NEW loop; a ``loop.yaml`` that
  has no ``train_method`` (or says ``dpo``) keeps training dpo, and dpo / kto stay selectable;
- the sft rows are the real edits plus every unedited answer whose newest label is good
  (feedback log, ``--record-thumbs`` database, serve-trace ``thumbs_up``); a bad label alone
  gives no row;
- an edit wins over thumbs and ratings for dpo, kto and sft alike: the latest real edit
  decides, before or after any thumb;
- the sft config is the loop's sft recipe (tests/test_loop_sft_recipe.py): candidate B of the
  Soup Zero fit runs, lr 5e-4, 8 epochs, LoRA r 64 / alpha 128 on all-linear, quantization
  auto (4bit here: no CUDA device, unknown size; tests/test_loop_sft_quantization.py);
- with an sft loop and a feedback log, training starts at ``min_new_edits`` (8) new edits or
  ``min_new_rated`` (30) new rated answers, whichever comes first.
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
_LINE_IDS = itertools.count(5000)


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _panel_text(text: str) -> str:
    """Output with the panel's box characters removed, so a wrapped sentence reads whole."""
    return _clean(text.replace("│", " ").replace("|", " "))


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


def conv(number, *, ts=0, messages=None, response=ANSWER):
    return {
        "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
        "ts": iso(ts), "model": "base", "version": 1, "adapter": None, "canary_bucket": None,
        "messages": messages if messages is not None else MESSAGES, "response": response,
        "finish_reason": "stop", "source": "soup-serve",
    }


def fb(number, kind, ts, **fields):
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


def write_lines(path, lines) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))
    return path


def append_lines(path, lines) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))


TRACE_DIR = os.path.join(".soup-loops", "traces")
TRACES = os.path.join(TRACE_DIR, "conversations.jsonl")
SERVE_TRACES = os.path.join(TRACE_DIR, "serve.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")


def logs(conversations, feedback):
    write_lines(TRACES, conversations)
    write_lines(FEEDBACK, feedback)


def harvest():
    return lf.harvest_feedback(TRACE_DIR, FEEDBACK)


def rows(items):
    return [item.row for item in items]


def stamps(items):
    return [item.ts for item in items]


def kto_row(completion, label, prompt=None):
    return {
        "prompt": prompt if prompt is not None else MESSAGES, "completion": completion,
        "label": label, "source": "feedback",
    }


def sft_edit_row(text=EDIT, prompt=None):
    return {
        "messages": [*(prompt if prompt is not None else MESSAGES),
                     {"role": "assistant", "content": text}],
        "source": "feedback-edit",
    }


def sft_good_row(text=ANSWER, prompt=None):
    return {
        "messages": [*(prompt if prompt is not None else MESSAGES),
                     {"role": "assistant", "content": text}],
        "source": "feedback-good",
    }


def _file_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER", "_TRAIN_RUNNER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def _sft(**kwargs) -> LoopState:
    """An sft loop that trains on any news (one new edit or one new rated answer)."""
    kwargs.setdefault("min_new_edits", 1)
    kwargs.setdefault("min_new_rated", 1)
    return _state(train_method="sft", **kwargs)


def _db(rows_, path="t.db") -> str:
    """A thumbs database holding ``(ts offset, prompt, response, thumb)`` rows."""
    from soup_cli.utils import local_rl

    local_rl.init_local_rl_db(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.executemany(
            "INSERT INTO thumbs (ts, prompt, response, thumb) VALUES (?, ?, ?, ?)",
            [(T0 + offset, prompt, response, value) for offset, prompt, response, value in rows_],
        )
        conn.commit()
    return path


# ---------------------------------------------------------------------------
# An edit wins over thumbs and ratings, for every method
# ---------------------------------------------------------------------------


class TestEditWins:
    @pytest.mark.parametrize(
        "after",
        [thumb(1, 2, 1), rating(1, 2, 5), rating(1, 2, 4)],
        ids=["thumb up", "rating 5", "rating 4"],
    )
    def test_a_good_label_after_the_edit_does_not_make_the_original_good(self, cwd, after):
        logs([conv(1)], [edit(1, 1), after])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 1, T0 + 1]
        assert rows(data.sft) == [sft_edit_row()]
        assert rows(data.dpo) == [
            {"prompt": MESSAGES, "chosen": EDIT, "rejected": ANSWER, "source": "feedback-edit"}
        ]

    @pytest.mark.parametrize(
        "before",
        [thumb(1, 1, 1), rating(1, 1, 5)],
        ids=["thumb up", "rating 5"],
    )
    def test_a_good_label_before_the_edit_does_not_make_the_original_good(self, cwd, before):
        logs([conv(1)], [before, edit(1, 2)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert rows(data.sft) == [sft_edit_row()]

    @pytest.mark.parametrize(
        "other",
        [thumb(1, 2, -1), rating(1, 2, 1), rating(1, 2, 3)],
        ids=["thumb down", "rating 1", "rating 3"],
    )
    def test_a_bad_or_neutral_label_after_the_edit_changes_nothing(self, cwd, other):
        logs([conv(1)], [edit(1, 1), other])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 1, T0 + 1]
        assert rows(data.sft) == [sft_edit_row()]

    def test_the_latest_edit_decides(self, cwd):
        logs([conv(1)], [edit(1, 1, "first try"), thumb(1, 2, 1), edit(1, 3, EDIT)])

        data = harvest().datasets

        assert rows(data.sft) == [sft_edit_row(EDIT)]
        assert rows(data.kto) == [kto_row(EDIT, True), kto_row(ANSWER, False)]
        assert stamps(data.kto) == [T0 + 3, T0 + 3]

    def test_a_withdrawn_edit_hands_the_decision_back_to_the_thumbs(self, cwd):
        logs([conv(1)], [edit(1, 1), thumb(1, 2, 1), fb(1, "edit", 3, edited_text=None)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(ANSWER, True)]
        assert rows(data.sft) == [sft_good_row()]
        assert data.dpo == []

    def test_an_edit_equal_to_the_answer_is_no_edit_so_the_thumb_decides(self, cwd):
        logs([conv(1)], [edit(1, 1, f" {ANSWER}\n"), thumb(1, 2, 1)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(ANSWER, True)]
        assert rows(data.sft) == [sft_good_row()]
        assert data.dpo == []

    def test_an_edit_that_undoes_an_earlier_edit_is_no_edit(self, cwd):
        logs([conv(1)], [edit(1, 1), thumb(1, 2, 1), edit(1, 3, ANSWER)])

        data = harvest().datasets

        assert rows(data.sft) == [sft_good_row()]
        assert rows(data.kto) == [kto_row(ANSWER, True)]

    def test_an_empty_original_has_no_bad_row_and_no_good_row(self, cwd):
        logs([conv(1, response="")], [edit(1, 1), thumb(1, 2, 1)])

        data = harvest().datasets

        assert rows(data.kto) == [kto_row(EDIT, True)]
        assert rows(data.sft) == [sft_edit_row()]
        assert data.dpo == []

    def test_edit_then_thumbs_up_across_two_harvests(self, cwd):
        """Harvest 1 sees the edit; a thumbs-up on the same answer is appended; harvest 2
        still labels the original bad, adds no sft row for it and keeps the edit's rows."""
        logs([conv(1)], [edit(1, 1)])

        def run(method):
            return ls.harvest_from_traces(_state(train_method=method, min_new_edits=1))

        first = {method: run(method) for method in ("dpo", "kto", "sft")}
        append_lines(FEEDBACK, [thumb(1, 2, 1)])
        second = {method: run(method) for method in ("dpo", "kto", "sft")}

        for method, expected in {
            "dpo": [{"prompt": MESSAGES, "chosen": EDIT, "rejected": ANSWER,
                     "source": "feedback-edit"}],
            "kto": [kto_row(EDIT, True), kto_row(ANSWER, False)],
            "sft": [sft_edit_row()],
        }.items():
            assert _file_rows(first[method]["pairs_path"]) == expected, method
            assert _file_rows(second[method]["pairs_path"]) == expected, method
            assert second[method]["cursor_ts"] == first[method]["cursor_ts"] == T0 + 1, method
        assert (second["sft"]["kto_rows"], second["sft"]["sft_rows"]) == (2, 1)

    def test_a_good_label_after_the_edit_is_not_news_once_the_edit_was_trained_on(self, cwd):
        logs([conv(1)], [edit(1, 1), thumb(1, 2, 1)])

        out = ls.harvest_from_traces(_sft(harvest_cursor_ts=T0 + 1))

        assert (out["new_pairs"], out["new_edits"], out["new_rated"]) == (0, 0, 0)
        assert out["pairs_path"] is None

    def test_a_thumb_after_an_edit_does_not_make_a_kto_row_new(self, cwd):
        logs([conv(1)], [edit(1, 1), thumb(1, 2, 1)])

        out = ls.harvest_from_traces(_state(train_method="kto", harvest_cursor_ts=T0 + 1))

        assert (out["pairs_harvested"], out["new_pairs"]) == (2, 0)


# ---------------------------------------------------------------------------
# The sft rows from answers rated good
# ---------------------------------------------------------------------------


class TestGoodAnswersFromTheFeedbackLog:
    @pytest.mark.parametrize(
        "label",
        [thumb(1, 5, 1), rating(1, 5, 4), rating(1, 5, 5)],
        ids=["thumb up", "rating 4", "rating 5"],
    )
    def test_a_good_label_on_an_unedited_answer_is_an_sft_row(self, cwd, label):
        logs([conv(1)], [label])

        data = harvest().datasets

        assert rows(data.sft) == [sft_good_row()]
        assert stamps(data.sft) == [T0 + 5]

    @pytest.mark.parametrize(
        "label",
        [thumb(1, 5, -1), rating(1, 5, 1), rating(1, 5, 2), rating(1, 5, 3)],
        ids=["thumb down", "rating 1", "rating 2", "rating 3"],
    )
    def test_an_answer_that_was_only_rated_bad_or_neutral_gives_no_row(self, cwd, label):
        logs([conv(1)], [label])

        assert harvest().datasets.sft == []

    @pytest.mark.parametrize(
        ("lines", "good"),
        [
            ([thumb(1, 1, 1), rating(1, 2, 1)], False),    # up, then a rating of 1
            ([thumb(1, 1, -1), rating(1, 2, 5)], True),    # down, then a rating of 5
            ([rating(1, 1, 5), thumb(1, 2, -1)], False),   # 5, then thumb down
            ([rating(1, 1, 1), thumb(1, 2, 1)], True),     # 1, then thumb up
            ([thumb(1, 1, 1), rating(1, 2, 3)], True),     # a rating of 3 says nothing
            ([thumb(1, 1, -1), rating(1, 2, 3)], False),
            ([thumb(1, 9, 1), thumb(1, 2, -1)], False),    # file order, not ts
            ([thumb(1, 1, 1), thumb(1, 2, None)], False),  # cleared
            ([thumb(1, 1, 1), thumb(1, 2, None), thumb(1, 3, 1)], True),
        ],
        ids=range(9),
    )
    def test_the_newest_label_in_file_order_decides(self, cwd, lines, good):
        logs([conv(1)], lines)

        assert (len(harvest().datasets.sft) == 1) is good

    def test_the_prompt_is_the_whole_conversation_as_logged(self, cwd):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How do I reset my router?"},
        ]
        logs([conv(1, messages=history)], [thumb(1, 1, 1)])

        assert rows(harvest().datasets.sft) == [sft_good_row(prompt=history)]

    def test_an_empty_answer_is_no_training_target(self, cwd):
        logs([conv(1, response=" \n")], [thumb(1, 1, 1)])

        assert harvest().datasets.sft == []

    def test_the_same_prompt_and_answer_count_once_with_the_newest_ts(self, cwd):
        logs([conv(1), conv(2), conv(3, response="Other.")],
             [thumb(1, 4, 1), thumb(2, 9, 1), thumb(3, 6, 1)])

        data = harvest().datasets

        assert rows(data.sft) == [sft_good_row(), sft_good_row("Other.")]
        assert stamps(data.sft) == [T0 + 9, T0 + 6]

    def test_good_rows_and_edit_rows_come_in_the_order_the_feedback_names_the_answers(self, cwd):
        logs([conv(1, response="a"), conv(2, response="b"), conv(3, response="c")],
             [thumb(3, 1, 1), edit(1, 2), thumb(2, 3, 1)])

        completions = [row["messages"][-1]["content"] for row in rows(harvest().datasets.sft)]

        assert completions == ["c", EDIT, "b"]

    def test_a_good_answer_and_an_edit_to_the_same_text_are_one_row(self, cwd):
        logs([conv(1, response="a"), conv(2, response="b")], [thumb(1, 1, 1), edit(2, 2, "a")])

        data = harvest().datasets

        assert len(data.sft) == 1
        assert data.sft[0].row["messages"][-1]["content"] == "a"
        assert data.sft[0].ts == T0 + 2


# ---------------------------------------------------------------------------
# The sft rows from serve traces and the thumbs database
# ---------------------------------------------------------------------------


def _trace(ident, prompt, response, signal, ts=None):
    line = {"id": ident, "prompt": prompt, "response": response, "signal": signal}
    if ts is not None:
        line["ts"] = ts
    return line


def _user(text):
    return [{"role": "user", "content": text}]


def _good(text, prompt, source):
    return {"messages": [*_user(prompt), {"role": "assistant", "content": text}],
            "source": source}


class TestGoodAnswersFromTracesAndTheThumbsDatabase:
    def test_a_thumbs_up_trace_is_an_sft_row(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q", "good", "thumbs_up", T0 + 3)])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert out["pairs_harvested"] == 1 and out["cursor_ts"] == T0 + 3
        assert _file_rows(out["pairs_path"]) == [_good("good", "Q", "thumbs_up")]

    def test_a_thumbs_down_trace_alone_gives_no_row(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q", "bad", "thumbs_down", T0 + 3)])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert out["pairs_harvested"] == 0 and out["pairs_path"] is None

    @pytest.mark.parametrize(
        ("signals", "rows_"),
        [(("thumbs_up", "thumbs_down"), 0), (("thumbs_down", "thumbs_up"), 1),
         (("thumbs_up", "thumbs_up"), 1), (("thumbs_up", "none", "thumbs_down"), 0)],
    )
    def test_the_newest_signal_of_a_trace_answer_decides(self, cwd, signals, rows_):
        write_lines(SERVE_TRACES, [
            _trace(f"t{index}", "Q", "answer", signal, T0 + index)
            for index, signal in enumerate(signals)
        ])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert out["pairs_harvested"] == rows_

    def test_a_trace_without_a_signal_is_not_a_label(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q", "plain", "none")])

        assert ls.harvest_from_traces(_state(train_method="sft"))["pairs_harvested"] == 0

    def test_an_up_thumb_in_the_database_is_an_sft_row(self, cwd):
        _db([(3, "Q", "good", "up")])

        out = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))

        assert out["pairs_harvested"] == 1 and out["cursor_ts"] == T0 + 3
        assert _file_rows(out["pairs_path"]) == [_good("good", "Q", "soup-serve-thumbs")]

    def test_a_down_thumb_alone_gives_no_row(self, cwd):
        _db([(3, "Q", "bad", "down")])

        out = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))

        assert out["pairs_harvested"] == 0 and out["pairs_path"] is None

    @pytest.mark.parametrize(
        ("thumbs", "rows_"),
        [(("up", "down"), 0), (("down", "up"), 1), (("up", "up"), 1), (("down", "up", "down"), 0)],
    )
    def test_the_newest_label_per_prompt_and_answer_decides(self, cwd, thumbs, rows_):
        _db([(index, "Q", "answer", thumb_) for index, thumb_ in enumerate(thumbs, start=1)])

        out = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))

        assert out["pairs_harvested"] == rows_

    def test_the_labels_are_per_answer_not_per_prompt(self, cwd):
        _db([(1, "Q", "liked", "up"), (2, "Q", "disliked", "down"), (3, "R", "also", "up")])

        out = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))

        assert [row["messages"][-1]["content"] for row in _file_rows(out["pairs_path"])] == [
            "liked", "also"
        ]

    def test_the_same_answer_from_all_three_sources_is_one_row(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q", "same", "thumbs_up", T0 + 1)])
        _db([(2, "Q", "same", "up")])
        logs([conv(1, messages=_user("Q"), response="same")], [thumb(1, 7, 1)])

        out = ls.harvest_from_traces(_sft(thumbs_db="t.db"))

        assert out["pairs_harvested"] == 1 and out["sft_rows"] == 1
        assert out["cursor_ts"] == T0 + 7  # the newest of the three copies
        (row,) = _file_rows(out["pairs_path"])
        assert row["messages"] == [*_user("Q"), {"role": "assistant", "content": "same"}]

    def test_the_sources_add_up(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q1", "from a trace", "thumbs_up", T0 + 1)])
        _db([(2, "Q2", "from the database", "up")])
        logs([conv(1, response="from the log"), conv(2, response="liked")],
             [edit(1, 4, "corrected"), thumb(2, 5, 1)])

        out = ls.harvest_from_traces(_sft(thumbs_db="t.db"))

        assert [row["messages"][-1]["content"] for row in _file_rows(out["pairs_path"])] == [
            "from a trace", "from the database", "corrected", "liked"
        ]
        assert out["sft_rows"] == 4

    def test_a_database_thumb_cannot_make_an_edited_original_good_again(self, cwd):
        """The edit wins across sources too: the answer the person corrected is not trained on."""
        _db([(9, "How do I reset my router?", ANSWER, "up")])
        logs([conv(1, messages=[MESSAGES[1]])], [edit(1, 1)])

        out = ls.harvest_from_traces(_sft(thumbs_db="t.db"))

        assert _file_rows(out["pairs_path"]) == [
            sft_edit_row(prompt=[MESSAGES[1]])
        ]

    def test_a_thumbs_up_trace_cannot_make_an_edited_original_good_again(self, cwd):
        write_lines(SERVE_TRACES, [
            _trace("a", "How do I reset my router?", ANSWER, "thumbs_up", T0 + 9)
        ])
        logs([conv(1, messages=[MESSAGES[1]])], [edit(1, 1)])

        out = ls.harvest_from_traces(_sft())

        assert _file_rows(out["pairs_path"]) == [sft_edit_row(prompt=[MESSAGES[1]])]

    def test_the_rows_count_as_new_by_their_own_ts(self, cwd):
        write_lines(SERVE_TRACES, [_trace("a", "Q1", "old", "thumbs_up", T0 + 1)])
        _db([(5, "Q2", "new", "up")])

        out = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db",
                                            harvest_cursor_ts=T0 + 3))

        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (2, 1, T0 + 5)

    def test_without_a_feedback_log_min_new_pairs_counts_the_new_sft_rows(self, cwd):
        _db([(1, "Q1", "a", "up"), (2, "Q2", "b", "up")])

        waiting = ls.harvest_from_traces(
            _state(train_method="sft", thumbs_db="t.db", min_new_pairs=3)
        )
        enough = ls.harvest_from_traces(
            _state(train_method="sft", thumbs_db="t.db", min_new_pairs=2)
        )

        assert (waiting["new_pairs"], waiting["pairs_path"]) == (2, None)
        assert len(_file_rows(enough["pairs_path"])) == 2

    def test_the_other_methods_do_not_read_the_thumbs_as_sft_rows(self, cwd):
        _db([(1, "Q1", "a", "up")])

        out = ls.harvest_from_traces(_state(train_method="kto", thumbs_db="t.db"))

        assert out["pairs_harvested"] == 0
        assert out["sft_rows"] == 1  # reported with the other dataset sizes


class TestEveryRowIsOneSoupTrainReads:
    def test_rows_of_all_sources_pass_the_chatml_converter(self, cwd):
        from soup_cli.data.formats import format_to_messages

        write_lines(SERVE_TRACES, [_trace("a", "Q1", "from a trace", "thumbs_up", T0 + 1)])
        _db([(2, "Q2", "from the database", "up")])
        logs([conv(1), conv(2, response="liked")], [edit(1, 4), thumb(2, 5, 1)])

        out = ls.harvest_from_traces(_sft(thumbs_db="t.db"))

        found = _file_rows(out["pairs_path"])
        assert len(found) == 4
        for row in found:
            converted = format_to_messages(row, "chatml")
            assert converted is not None, row
            roles = [message["role"] for message in converted["messages"]]
            assert roles[-1] == "assistant" and roles.count("assistant") == 1, row

    def test_soup_train_loads_the_file_with_the_rendered_config(self, cwd):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.loader import load_dataset

        write_lines(SERVE_TRACES, [_trace("a", "Q1", "from a trace", "thumbs_up", T0 + 1)])
        _db([(2, "Q2", "from the database", "up")])
        logs([conv(1), conv(2, response="liked")], [edit(1, 4), thumb(2, 5, 1)])
        out = ls.harvest_from_traces(_sft(thumbs_db="t.db"))
        raw = yaml.safe_load(ls._render_train_yaml(_sft(), out["pairs_path"], "out"))

        loaded = load_dataset(DataConfig(**raw["data"]))["train"]

        assert len(loaded) == 4
        assert [row["messages"][-1]["content"] for row in loaded] == [
            "from a trace", "from the database", EDIT, "liked"
        ]


# ---------------------------------------------------------------------------
# The sft training config
# ---------------------------------------------------------------------------


class TestSftConfig:
    def test_the_config_is_candidate_b_lr_5e4_eight_epochs_and_lora_64_128(self, cwd):
        rendered = yaml.safe_load(ls._render_train_yaml(_state(train_method="sft"), "d.jsonl", "o"))

        assert rendered == {
            "base": "org/base",
            "task": "sft",
            "data": {"train": "d.jsonl", "format": "chatml", "val_split": 0},
            "training": {
                "epochs": 8, "batch_size": 1, "lr": 5.0e-4,
                "lora": {"r": 64, "alpha": 128, "target_modules": "all-linear"},
                "quantization": "4bit",
            },
            "output": "o",
        }

    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_every_rendered_config_is_one_soup_train_accepts(self, cwd, method):
        from soup_cli.config.schema import SoupConfig

        state = _state(train_method=method)
        rendered = yaml.safe_load(ls._render_train_yaml(state, "d.jsonl", "o"))

        config = SoupConfig(**rendered)

        assert config.task == method
        if method == "sft":
            assert (config.training.epochs, config.training.lr) == (8, 5.0e-4)
            assert (config.training.lora.r, config.training.lora.alpha) == (64, 128)
            assert config.training.lora.target_modules == "all-linear"
            assert config.data.val_split == 0

    @pytest.mark.parametrize(
        ("method", "batch"), [("dpo", 1), ("kto", 2)]
    )
    def test_the_dpo_and_kto_configs_are_unchanged(self, cwd, method, batch):
        state = _state(train_method=method)
        rendered = yaml.safe_load(ls._render_train_yaml(state, "d.jsonl", "o"))

        assert rendered["training"] == {"epochs": 1, "batch_size": batch, "lr": 5.0e-5}


# ---------------------------------------------------------------------------
# When an sft loop with a feedback log trains
# ---------------------------------------------------------------------------


def _edits(count, *, start=1, first=1, tag="fixed"):
    """``count`` corrected answers: conversation records and their edit lines."""
    conversations = [conv(first + index, response=f"{tag} a{index}") for index in range(count)]
    lines = [edit(first + index, start + index, f"{tag} {index}") for index in range(count)]
    return conversations, lines


def _likes(count, *, first=100, start=1):
    conversations = [conv(first + index, response=f"liked {index}") for index in range(count)]
    lines = [thumb(first + index, start + index, 1) for index in range(count)]
    return conversations, lines


def _both(*groups):
    conversations = [c for group_c, _ in groups for c in group_c]
    lines = [line for _, group_l in groups for line in group_l]
    logs(conversations, lines)


class TestSftTrigger:
    def test_the_defaults_are_eight_edits_or_thirty_rated_answers(self):
        state = _state()

        assert (state.min_new_edits, state.min_new_rated) == (8, 30)

    @pytest.mark.parametrize(("edits", "waits"), [(7, True), (8, False)])
    def test_eight_new_edits_start_training(self, cwd, edits, waits):
        _both(_edits(edits))

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert out["new_edits"] == edits and out["new_rated"] == edits
        assert (out["pairs_path"] is None) is waits

    @pytest.mark.parametrize(("likes", "waits"), [(29, True), (30, False)])
    def test_thirty_new_rated_answers_start_training(self, cwd, likes, waits):
        _both(_likes(likes))

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (0, likes)
        assert (out["pairs_path"] is None) is waits
        assert out["new_pairs"] == likes  # the rows are counted too, but do not decide

    def test_either_rule_alone_is_enough(self, cwd):
        _both(_edits(8), _likes(5, start=20))
        by_edits = ls.harvest_from_traces(_state(train_method="sft"))

        _both(_edits(3), _likes(27, start=20))
        by_rated = ls.harvest_from_traces(_state(train_method="sft"))

        assert (by_edits["new_edits"], by_edits["new_rated"]) == (8, 13)
        assert by_edits["pairs_path"] is not None
        assert (by_rated["new_edits"], by_rated["new_rated"]) == (3, 30)
        assert by_rated["pairs_path"] is not None

    def test_neither_rule_met_waits(self, cwd):
        _both(_edits(7), _likes(22, start=20))

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (7, 29)
        assert out["pairs_path"] is None
        assert not os.path.exists(os.path.join(".soup-loops", "pairs"))

    def test_the_thresholds_are_the_loop_states(self, cwd):
        _both(_edits(2), _likes(4, start=20))

        by_edits = ls.harvest_from_traces(_state(train_method="sft", min_new_edits=2))
        by_rated = ls.harvest_from_traces(_state(train_method="sft", min_new_rated=6))
        neither = ls.harvest_from_traces(
            _state(train_method="sft", min_new_edits=3, min_new_rated=7)
        )

        assert by_edits["pairs_path"] is not None
        assert by_rated["pairs_path"] is not None
        assert neither["pairs_path"] is None

    def test_min_new_pairs_does_not_apply_to_an_sft_loop_with_a_feedback_log(self, cwd):
        _both(_edits(2))

        out = ls.harvest_from_traces(_state(train_method="sft", min_new_pairs=1))

        assert out["new_pairs"] == 2 and out["pairs_path"] is None

    def test_only_news_after_the_cursor_counts(self, cwd):
        _both(_edits(8), _likes(30, start=20))

        old = ls.harvest_from_traces(_state(train_method="sft", harvest_cursor_ts=T0 + 100))
        fresh = ls.harvest_from_traces(_state(train_method="sft", harvest_cursor_ts=T0 + 7))

        assert (old["new_edits"], old["new_rated"], old["pairs_path"]) == (0, 0, None)
        # The edits are at +1 .. +8, the likes at +20 .. +49: one edit and every like are new.
        assert (fresh["new_edits"], fresh["new_rated"]) == (1, 31)
        assert fresh["pairs_path"] is not None

    def test_an_edit_at_exactly_the_cursor_is_not_new(self, cwd):
        _both(_edits(3))

        out = ls.harvest_from_traces(_sft(harvest_cursor_ts=T0 + 2))

        assert (out["new_edits"], out["new_rated"]) == (1, 1)  # only the edit at +3

    def test_a_row_without_a_ts_is_new_only_while_there_is_no_cursor(self, cwd):
        logs([conv(1)], [{**edit(1, 1), "ts": "garbage"}])

        first = ls.harvest_from_traces(_sft())
        later = ls.harvest_from_traces(_sft(harvest_cursor_ts=5.0))

        assert (first["new_edits"], first["new_rated"]) == (1, 1)
        assert (later["new_edits"], later["new_rated"]) == (0, 0)

    def test_every_label_counts_as_rated_and_only_a_real_edit_as_an_edit(self, cwd):
        logs(
            [conv(1), conv(2, response="b"), conv(3, response="c"), conv(4, response="d"),
             conv(5, response="e"), conv(6, response="f")],
            [thumb(1, 1, 1), thumb(2, 2, -1), rating(3, 3, 3), rating(4, 4, 1),
             edit(5, 5, "e"),      # an edit equal to the answer: no edit, not rated
             edit(6, 6, "fixed")],
        )

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (1, 5)

    def test_an_answer_counts_once_however_many_lines_it_has(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), thumb(1, 2, -1), rating(1, 3, 5), edit(1, 4),
                         edit(1, 5, "again")])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (1, 1)

    def test_a_cleared_label_is_not_rated(self, cwd):
        logs([conv(1)], [thumb(1, 1, 1), thumb(1, 2, None)])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (0, 0)

    def test_the_counts_are_reported_for_every_method(self, cwd):
        _both(_edits(2), _likes(3, start=20))

        for method in ("dpo", "kto", "sft"):
            out = ls.harvest_from_traces(_state(train_method=method))
            assert (out["new_edits"], out["new_rated"]) == (2, 5), method

    def test_without_any_feedback_the_counts_are_zero(self, cwd):
        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_edits"], out["new_rated"]) == (0, 0)

    def test_dpo_and_kto_keep_min_new_pairs(self, cwd):
        _both(_edits(2))

        dpo = ls.harvest_from_traces(_state(min_new_pairs=3))
        kto = ls.harvest_from_traces(_state(train_method="kto", min_new_pairs=5))
        kto_enough = ls.harvest_from_traces(_state(train_method="kto", min_new_pairs=4))

        assert dpo["pairs_path"] is None and kto["pairs_path"] is None
        assert kto_enough["pairs_path"] is not None

    def test_a_feedback_log_that_exists_decides_the_rule_even_when_empty(self, cwd):
        _db([(1, "Q1", "a", "up"), (2, "Q2", "b", "up"), (3, "Q3", "c", "up")])
        write_lines(FEEDBACK, [])

        with_log = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))
        os.remove(FEEDBACK)
        without_log = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))

        assert with_log["pairs_harvested"] == 3 and with_log["pairs_path"] is None
        assert without_log["pairs_path"] is not None  # min_new_pairs 1 applies

    def test_the_log_the_loop_reads_is_the_one_that_decides(self, cwd):
        _db([(1, "Q1", "a", "up")])
        write_lines("elsewhere.jsonl", [])

        default = ls.harvest_from_traces(_state(train_method="sft", thumbs_db="t.db"))
        named = ls.harvest_from_traces(
            _state(train_method="sft", thumbs_db="t.db", feedback_log="elsewhere.jsonl")
        )

        assert default["pairs_path"] is not None
        assert named["pairs_path"] is None

    def test_the_cursor_covers_every_rated_answer_not_only_those_with_a_row(self, cwd):
        logs([conv(1), conv(2, response="b")], [thumb(1, 1, 1), thumb(2, 9, -1)])

        out = ls.harvest_from_traces(_sft())

        assert out["pairs_harvested"] == 1
        assert out["cursor_ts"] == T0 + 9  # the thumb down at +9 was seen too

    def test_rated_answers_without_a_row_do_not_start_a_training_on_nothing(self, cwd):
        logs([conv(number, response=f"b{number}") for number in range(1, 31)],
             [thumb(number, number, -1) for number in range(1, 31)])

        out = ls.harvest_from_traces(_state(train_method="sft"))

        assert (out["new_rated"], out["pairs_harvested"], out["pairs_path"]) == (30, 0, None)
        note = ls.train_dpo_from_pairs(_state(train_method="sft"), out)["notes"]
        assert note == "no_sft_rows: 0 valid sft rows harvested, training skipped"


class _Trainer:
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


class TestSftTrainStage:
    def test_the_skip_note_names_both_counts(self, cwd):
        _both(_edits(3), _likes(9, start=20))
        state = _state(train_method="sft")

        out = ls.train_dpo_from_pairs(state, ls.harvest_from_traces(state))

        assert out["skipped"] is True and out["run_id"] is None
        assert out["notes"] == (
            "not_enough_new_data: 3 new edits (needs 8) and 12 new rated answers (needs 30) "
            "since the last training run"
        )

    def test_the_note_carries_the_loop_states_thresholds(self, cwd):
        _both(_edits(1))
        state = _state(train_method="sft", min_new_edits=4, min_new_rated=9)

        out = ls.train_dpo_from_pairs(state, ls.harvest_from_traces(state))

        assert out["notes"] == (
            "not_enough_new_data: 1 new edits (needs 4) and 1 new rated answers (needs 9) "
            "since the last training run"
        )

    @pytest.mark.parametrize(
        ("edits", "likes"), [(8, 0), (0, 30), (7, 23), (0, 31)],
        ids=["edits", "rated", "both", "more"],
    )
    def test_either_rule_trains(self, cwd, monkeypatch, edits, likes):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        _both(_edits(edits), _likes(likes, start=50))
        state = _state(train_method="sft")

        out = ls.train_dpo_from_pairs(state, ls.harvest_from_traces(state))

        assert out["skipped"] is False and out["run_id"], out
        (config,) = trainer.configs
        assert config["task"] == "sft" and config["training"]["epochs"] == 8
        assert len(trainer.trained_on[0]) == edits + likes

    def test_the_train_stage_applies_the_rule_the_harvest_reports(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        (cwd / "rows.jsonl").write_text('{"messages": []}\n', "utf-8")
        ctx = {"pairs_harvested": 4, "new_pairs": 4, "pairs_path": "rows.jsonl",
               "new_data_rule": "edits_or_rated", "new_edits": 7, "new_rated": 29}

        waits = ls.train_dpo_from_pairs(_state(train_method="sft"), ctx)
        by_edits = ls.train_dpo_from_pairs(
            _state(train_method="sft"), {**ctx, "new_edits": 8}
        )
        by_rated = ls.train_dpo_from_pairs(
            _state(train_method="sft"), {**ctx, "new_rated": 30}
        )

        assert waits["skipped"] is True
        assert "7 new edits (needs 8) and 29 new rated answers (needs 30)" in waits["notes"]
        assert by_edits["skipped"] is False and by_rated["skipped"] is False

    def test_a_harvest_that_reports_no_rule_keeps_min_new_pairs(self, cwd, monkeypatch):
        """A custom harvest_fn (new_pairs only) is judged as before."""
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", _Trainer())
        ctx = {"pairs_harvested": 4, "new_pairs": 1, "pairs_path": "unused"}

        out = ls.train_dpo_from_pairs(_state(train_method="sft", min_new_pairs=3), ctx)

        assert out["notes"] == (
            "not_enough_new_data: 1 new sft rows since the last training run, "
            "training needs 3 (soup loop init --min-new-pairs)"
        )

    def test_dpo_and_kto_notes_are_unchanged(self, cwd):
        ctx = {"pairs_harvested": 4, "new_pairs": 1, "pairs_path": "unused",
               "new_edits": 0, "new_rated": 0}

        dpo = ls.train_dpo_from_pairs(_state(min_new_pairs=3), ctx)
        kto = ls.train_dpo_from_pairs(_state(train_method="kto", min_new_pairs=3), ctx)

        assert dpo["notes"].startswith("not_enough_new_data: 1 new preference pairs since")
        assert kto["notes"].startswith("not_enough_new_data: 1 new kto rows since")


class TestSftLoop:
    """The pre-wired stages through run_once: the cursor of an sft loop."""

    @staticmethod
    def _config():
        return WatchConfig(
            harvest_fn=ls.harvest_from_traces,
            train_fn=ls.train_dpo_from_pairs,
            gate_fn=lambda state, ctx: {"gate_verdict": "OK"},
            deploy_fn=lambda state, ctx: {"deployed": False, "canary_verdict": None},
            cost_fn=lambda state: 0.0,
        )

    def test_it_waits_trains_on_the_threshold_then_waits_for_more_news(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        state = _state(train_method="sft")
        _both(_edits(3), _likes(9, start=20))

        state, waiting, _ = run_once(state, self._config())
        assert waiting.run_id is None
        assert "not_enough_new_data: 3 new edits (needs 8) and 12 new rated answers (needs 30)" in (
            waiting.notes
        )
        assert state.harvest_cursor_ts is None and trainer.configs == []

        conversations, lines = _edits(5, first=200, start=60, tag="more")
        append_lines(TRACES, conversations)
        append_lines(FEEDBACK, lines)
        state, trained, _ = run_once(state, self._config())

        assert trained.run_id
        assert len(trainer.trained_on[-1]) == 3 + 9 + 5  # everything, not only the news
        assert state.harvest_cursor_ts == T0 + 64

        state, quiet, _ = run_once(state, self._config())
        assert quiet.run_id is None
        assert "not_enough_new_data: 0 new edits (needs 8) and 0 new rated answers (needs 30)" in (
            quiet.notes
        )
        assert len(trainer.trained_on) == 1

    def test_ratings_without_a_row_are_consumed_by_a_run_not_counted_again(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        conversations, lines = _likes(30, start=1)
        bad_conversations = [conv(300 + n, response=f"bad {n}") for n in range(5)]
        bad_lines = [thumb(300 + n, 100 + n, -1) for n in range(5)]
        logs([*conversations, *bad_conversations], [*lines, *bad_lines])
        state = _state(train_method="sft")

        state, first, _ = run_once(state, self._config())

        assert first.run_id and state.harvest_cursor_ts == T0 + 104
        state, second, _ = run_once(state, self._config())
        assert second.run_id is None
        assert "0 new edits (needs 8) and 0 new rated answers (needs 30)" in second.notes

    def test_a_failed_training_leaves_the_news_to_be_counted_again(self, cwd, monkeypatch):
        trainer = _Trainer(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        _both(_edits(8))

        state, failed, _ = run_once(_state(train_method="sft"), self._config())

        assert failed.run_id is None and state.harvest_cursor_ts is None
        trainer.returncode = 0
        state, retry, _ = run_once(state, self._config())
        assert retry.run_id and state.harvest_cursor_ts == T0 + 8

    def test_a_thumbs_up_after_a_trained_edit_does_not_ask_for_a_new_run(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        _both(_edits(8))
        state, first, _ = run_once(_state(train_method="sft"), self._config())
        assert first.run_id
        append_lines(FEEDBACK, [thumb(1 + n, 50 + n, 1) for n in range(8)])

        state, second, _ = run_once(state, self._config())

        assert second.run_id is None
        assert "0 new edits (needs 8) and 0 new rated answers (needs 30)" in second.notes


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------


class TestLoopState:
    def test_the_dataclass_default_stays_dpo_so_an_old_loop_keeps_its_method(self):
        assert _state().train_method == "dpo"

    def test_the_two_fields_come_last_in_this_order(self):
        names = list(LoopState.__dataclass_fields__)

        # The sft recipe (tests/test_loop_sft_recipe.py) follows them.
        assert names[-3:-1] == ["min_new_edits", "min_new_rated"]
        assert names[-4] == "version_log_started"

    def test_a_loop_yaml_without_the_fields_loads_with_their_defaults(self, cwd):
        legacy = {"served_model": "m", "eval_suite": "e", "baseline": "b", "status": "running",
                  "min_new_pairs": 2}
        (cwd / "legacy.json").write_text(json.dumps(legacy), "utf-8")

        state = read_state(str(cwd / "legacy.json"))

        assert state.train_method == "dpo"  # not the new init default
        assert (state.min_new_edits, state.min_new_rated, state.min_new_pairs) == (8, 30, 2)

    def test_a_loop_yaml_keeps_the_method_it_names(self, cwd):
        for method in ("dpo", "kto", "sft"):
            data = {"served_model": "m", "eval_suite": "e", "baseline": "b",
                    "train_method": method}
            (cwd / f"{method}.json").write_text(json.dumps(data), "utf-8")

            assert read_state(str(cwd / f"{method}.json")).train_method == method

    @pytest.mark.parametrize("name", ["min_new_edits", "min_new_rated"])
    @pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "8", None, [8]])
    def test_a_bad_threshold_is_refused_naming_the_field(self, name, value):
        with pytest.raises(ValueError, match=rf"{name} must be a positive int"):
            _state(**{name: value})

    @pytest.mark.parametrize("name", ["min_new_edits", "min_new_rated"])
    def test_a_bad_threshold_in_the_file_is_reported_with_its_name(self, cwd, name):
        bad = {"served_model": "m", "eval_suite": "e", "baseline": "b", name: 0}
        (cwd / "bad.json").write_text(json.dumps(bad), "utf-8")

        with pytest.raises(ValueError, match=f"state file has invalid contents: {name}"):
            read_state(str(cwd / "bad.json"))

    def test_init_state_threads_the_thresholds_and_the_file_keeps_them(self, cwd):
        state, path = init_state(
            "m", "e", "b", min_new_edits=3, min_new_rated=12, path=str(cwd / "loop.json")
        )

        assert (state.min_new_edits, state.min_new_rated) == (3, 12)
        reread = read_state(path)
        assert (reread.min_new_edits, reread.min_new_rated) == (3, 12)

    def test_init_state_defaults(self, cwd):
        state, _ = init_state("m", "e", "b", path=str(cwd / "loop.json"))

        assert (state.min_new_edits, state.min_new_rated) == (8, 30)

    def test_the_thresholds_round_trip_through_write_state(self, cwd):
        state, path = init_state("m", "e", "b", path=str(cwd / "loop.json"))
        write_state(replace(state, min_new_edits=2, min_new_rated=5), path)

        reread = read_state(path)

        assert (reread.min_new_edits, reread.min_new_rated) == (2, 5)


# ---------------------------------------------------------------------------
# soup loop init / status
# ---------------------------------------------------------------------------


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output), _panel_text(result.output)


_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")
_SFT_SENTENCE = (
    "sft trains on corrected answers and on answers rated good; an answer that was only "
    "rated bad gives no training row"
)


class TestCli:
    def test_a_new_loop_trains_sft(self, cwd):
        result, text, panel = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert state.train_method == "sft"
        assert (state.min_new_edits, state.min_new_rated, state.min_new_pairs) == (8, 30, 1)
        assert "train_method: sft" in text
        assert _SFT_SENTENCE in panel

    @pytest.mark.parametrize("method", ["dpo", "kto"])
    def test_dpo_and_kto_stay_selectable(self, cwd, method):
        result, text, panel = _cli(*_INIT, "--train-method", method)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().train_method == method
        assert f"train_method: {method}" in text
        assert "sft trains on" not in panel

    def test_sft_can_be_named(self, cwd):
        result, _, panel = _cli(*_INIT, "--train-method", "SFT")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().train_method == "sft"
        assert _SFT_SENTENCE in panel

    def test_the_trigger_flags_are_stored(self, cwd):
        result, _, _ = _cli(*_INIT, "--min-new-edits", "3", "--min-new-rated", "12")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert (state.min_new_edits, state.min_new_rated) == (3, 12)

    @pytest.mark.parametrize("flag", ["--min-new-edits", "--min-new-rated"])
    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_a_threshold_below_one_is_refused(self, cwd, flag, value):
        result, text, _ = _cli(*_INIT, flag, value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"{flag} must be a positive int" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    @pytest.mark.parametrize("flag", ["--min-new-edits", "--min-new-rated"])
    def test_a_threshold_that_is_not_a_number_is_refused(self, cwd, flag):
        result, text, _ = _cli(*_INIT, flag, "many")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_help_names_the_flags_and_the_default(self, cwd):
        result, _, panel = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--min-new-edits" in panel and "--min-new-rated" in panel
        assert "sft (the default;" in panel
        assert "default: sft" in panel

    def test_status_shows_the_thresholds(self, cwd):
        _cli(*_INIT, "--min-new-edits", "3")

        result, text, _ = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"train_method\W+sft", text), text
        assert re.search(r"min_new_edits\W+3\b", text), text
        assert re.search(r"min_new_rated\W+30\b", text), text
        assert re.search(r"min_new_pairs\W+1\b", text), text

    def test_status_of_a_loop_written_before_the_fields_keeps_dpo(self, cwd):
        _cli(*_INIT)
        path = os.path.join(".soup", "loop.yaml")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        del data["train_method"], data["min_new_edits"], data["min_new_rated"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        result, text, _ = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"train_method\W+dpo", text), text
        assert re.search(r"min_new_edits\W+8\b", text), text
        assert re.search(r"min_new_rated\W+30\b", text), text
        assert read_state().train_method == "dpo"
