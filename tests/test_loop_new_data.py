"""`soup loop` retrains only when there is NEW feedback: the data cursor.

Before this, `soup loop watch --pre-wired` harvested every preference pair ever
recorded in each iteration. A promoted canary was followed by a cycle that
trained on the same feedback again and started another canary.

The cursor that a training run moves (`attempt_cursor_ts` in loop.yaml) is a TRIGGER only: the
newest feedback timestamp that a training run has already read. A pair is new when the newer of
its two rows is later than the later of that and the data cursor (`harvest_cursor_ts`, which
follows the version in production: tests/test_loop_verdict_v2.py). Training runs once at least
`min_new_pairs` (`soup loop init --min-new-pairs`, default 1) pairs are new, and it trains on ALL
accumulated pairs, not just the new ones: every iteration trains a fresh LoRA from
the base model, so a version trained on the new pairs alone would forget the last.
The attempt cursor advances after a training run, whatever the verdict was, and never
after a skip or a failed training.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState, init_state, read_state, write_state

T0 = 1_700_000_000.0  # 2023-11-14T22:13:20Z
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _db(path: str, rows) -> None:
    """Insert ``(ts, prompt, response, thumb)`` rows with the timestamps given."""
    from soup_cli.utils import local_rl

    if not os.path.exists(path):
        local_rl.init_local_rl_db(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.executemany(
            "INSERT INTO thumbs (ts, prompt, response, thumb) VALUES (?, ?, ?, ?)", rows
        )
        conn.commit()


def _pair_in_db(path: str, prompt: str, up_ts: float, down_ts: float, tag: str = "") -> None:
    _db(path, [(up_ts, prompt, f"good{tag}", "up"), (down_ts, prompt, f"bad{tag}", "down")])


def _trace_rows(prompt, up_ts=None, down_ts=None, tag="", good="good", bad="bad"):
    rows = [
        {"id": f"{prompt}-u{tag}", "prompt": prompt, "output": good, "signal": "thumbs_up"},
        {"id": f"{prompt}-d{tag}", "prompt": prompt, "output": bad, "signal": "thumbs_down"},
    ]
    if up_ts is not None:
        rows[0]["ts"] = up_ts
    if down_ts is not None:
        rows[1]["ts"] = down_ts
    return rows


def _write_traces(root, rows, name="serve.jsonl"):
    directory = root / "traces"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return str(directory)


def _read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _state(**kwargs) -> LoopState:
    # A registry ref, so the served model is never mistaken for a trace directory.
    return LoopState(served_model="registry://dummy", eval_suite="e", baseline="b", **kwargs)


# ---------------------------------------------------------------------------
# local_rl.harvest_dpo_pairs_since
# ---------------------------------------------------------------------------


class TestPairsSince:
    def test_with_no_cursor_every_pair_is_new_and_the_pairs_match_the_plain_harvest(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs, harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _pair_in_db("t.db", "B", T0 + 5, T0 + 3)

        pairs, new_pairs, newest = harvest_dpo_pairs_since("t.db", None)

        assert pairs == harvest_dpo_pairs("t.db")
        assert new_pairs == pairs
        assert newest == T0 + 5

    def test_a_pair_whose_rows_are_both_old_is_not_new(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _pair_in_db("t.db", "B", T0 + 10, T0 + 11)

        pairs, new_pairs, newest = harvest_dpo_pairs_since("t.db", T0 + 5)

        assert [p.prompt for p in pairs] == ["A", "B"]  # the training set keeps both
        assert [p.prompt for p in new_pairs] == ["B"]
        assert newest == T0 + 11

    def test_an_old_up_with_a_new_down_is_new(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 20)
        _pair_in_db("t.db", "B", T0 + 20, T0 + 1)

        _, new_pairs, newest = harvest_dpo_pairs_since("t.db", T0 + 5)

        assert [p.prompt for p in new_pairs] == ["A", "B"]
        assert newest == T0 + 20

    def test_a_new_row_that_replaces_an_old_one_makes_the_pair_new(self, cwd):
        """The pair is the LATEST up with the latest down: a newer up changes it."""
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _db("t.db", [(T0 + 30, "A", "better", "up")])

        pairs, new_pairs, newest = harvest_dpo_pairs_since("t.db", T0 + 5)

        assert [(p.chosen, p.rejected) for p in pairs] == [("better", "bad")]
        assert new_pairs == pairs
        assert newest == T0 + 30

    def test_a_row_at_exactly_the_cursor_is_not_new(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 5)

        pairs, new_pairs, newest = harvest_dpo_pairs_since("t.db", T0 + 5)

        assert len(pairs) == 1
        assert new_pairs == ()
        assert newest == T0 + 5

    def test_newest_ts_covers_the_rows_of_pairs_only(self, cwd):
        """An up with no down yet is not part of any pair, so it does not move the cursor."""
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _db("t.db", [(T0 + 999, "lonely", "answer", "up")])

        _, _, newest = harvest_dpo_pairs_since("t.db", None)

        assert newest == T0 + 2

    def test_a_pair_that_appears_later_from_a_row_older_than_the_cursor_is_still_new(self, cwd):
        """The up arrived before the cursor moved, the down after: the pair is new."""
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _db("t.db", [(T0 + 6, "later", "answer", "up")])
        assert harvest_dpo_pairs_since("t.db", T0 + 5)[1] == ()
        _db("t.db", [(T0 + 9, "later", "worse", "down")])

        _, new_pairs, _ = harvest_dpo_pairs_since("t.db", T0 + 5)

        assert [p.prompt for p in new_pairs] == ["later"]

    def test_no_pairs_gives_empty_results_and_no_newest_ts(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _db("t.db", [(T0 + 1, "A", "only up", "up")])

        assert harvest_dpo_pairs_since("t.db", None) == ((), (), None)
        assert harvest_dpo_pairs_since("t.db", T0) == ((), (), None)

    def test_an_int_cursor_is_accepted(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)

        assert len(harvest_dpo_pairs_since("t.db", int(T0))[1]) == 1

    def test_a_cursor_in_the_far_future_is_accepted_and_nothing_is_new(self, cwd):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)

        pairs, new_pairs, _ = harvest_dpo_pairs_since("t.db", 10**400)

        assert len(pairs) == 1 and new_pairs == ()

    @pytest.mark.parametrize("bad", [True, "5", [1], float("nan"), float("inf"), -float("inf")])
    def test_a_bad_cursor_is_refused(self, cwd, bad):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)

        with pytest.raises((TypeError, ValueError), match="since_ts"):
            harvest_dpo_pairs_since("t.db", bad)

    def test_the_path_is_checked_like_the_plain_harvest(self, cwd, tmp_path):
        from soup_cli.utils.local_rl import harvest_dpo_pairs_since

        with pytest.raises(ValueError, match="under cwd"):
            harvest_dpo_pairs_since(str(tmp_path.parent / "elsewhere.db"), None)
        with pytest.raises(FileNotFoundError, match="db_path not found"):
            harvest_dpo_pairs_since("missing.db", None)
        with pytest.raises(TypeError):
            harvest_dpo_pairs_since(True, None)  # type: ignore[arg-type]

    def test_the_plain_harvest_is_unchanged(self, cwd):
        from soup_cli.utils.local_rl import DpoPair, harvest_dpo_pairs

        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)

        assert harvest_dpo_pairs("t.db") == (DpoPair(prompt="A", chosen="good", rejected="bad"),)


# ---------------------------------------------------------------------------
# Trace.ts
# ---------------------------------------------------------------------------


def _ts_of(root, value, *, raw=None) -> object:
    """The ts parse_soup_serve gives a line whose ``ts`` is ``value`` (``raw`` = literal JSON)."""
    from soup_cli.data.traces.parsers import parse_soup_serve

    directory = root / "ts-case"
    directory.mkdir(exist_ok=True)
    body = raw if raw is not None else json.dumps(value)
    (directory / "s.jsonl").write_text(
        f'{{"id": "x", "prompt": "p", "response": "r", "ts": {body}}}\n', "utf-8"
    )
    (trace,) = list(parse_soup_serve(str(directory)))
    return trace.ts


class TestTraceTs:
    def test_the_field_defaults_to_none_and_keeps_positional_order(self):
        from soup_cli.data.traces.parsers import Trace

        trace = Trace("id", "prompt", "output", "thumbs_up", 2, "edit", {"k": 1})
        assert trace.ts is None
        assert (trace.signal, trace.regen_order, trace.edited_output) == ("thumbs_up", 2, "edit")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1700000000.5, 1700000000.5),
            (1700000000, 1700000000.0),
            (0, 0.0),
            ("2023-11-14T22:13:20Z", T0),
            ("2023-11-14T22:13:20+00:00", T0),
            ("2023-11-14T22:13:20.5Z", T0 + 0.5),
            ("2023-11-14T22:13:20.250000+00:00", T0 + 0.25),
        ],
    )
    def test_numbers_and_utc_iso_strings_are_epoch_seconds(self, tmp_path, value, expected):
        assert _ts_of(tmp_path, value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "2023-11-14T22:13:20+03:00",   # another offset: not converted
            "2023-11-14T22:13:20",         # no zone
            "2023-11-14",
            "2023-13-45T99:99:99Z",        # ends in a UTC suffix, still not a time
            "yesterdayZ",
            "+00:00",
            "yesterday",
            "",
            "1700000000",                  # a number in a string
            True,
            False,
            None,
            -5,
            1.7e12,                        # milliseconds
            [1700000000],
            {"t": 1},
        ],
    )
    def test_anything_else_is_none(self, tmp_path, value):
        assert _ts_of(tmp_path, value) is None

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1" + "0" * 400])
    def test_non_finite_or_unrepresentable_numbers_are_none(self, tmp_path, literal):
        assert _ts_of(tmp_path, None, raw=literal) is None

    def test_a_line_without_ts_has_none(self, tmp_path):
        from soup_cli.data.traces.parsers import parse_soup_serve

        directory = tmp_path / "no-ts"
        directory.mkdir()
        (directory / "s.jsonl").write_text('{"prompt": "p", "response": "r"}\n', "utf-8")
        assert [t.ts for t in parse_soup_serve(str(directory))] == [None]


# ---------------------------------------------------------------------------
# harvest_from_traces: all pairs train, only new ones trigger
# ---------------------------------------------------------------------------


class TestHarvest:
    def test_without_a_cursor_every_pair_is_new(self, cwd, monkeypatch):
        rows = _trace_rows("A", 10, 11) + _trace_rows("B", 20, 21)
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", _write_traces(cwd, rows))

        out = ls.harvest_from_traces(_state())

        assert out["pairs_harvested"] == 2, out
        assert out["new_pairs"] == 2, out
        assert out["cursor_ts"] == 21.0, out

    def test_all_pairs_are_written_but_only_pairs_past_the_cursor_count_as_new(
        self, cwd, monkeypatch
    ):
        trace_dir = _write_traces(cwd, _trace_rows("A", 10, 11) + _trace_rows("B", 20, 21))
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        out = ls.harvest_from_traces(_state(harvest_cursor_ts=15.0))

        assert out["pairs_harvested"] == 2, out
        assert out["new_pairs"] == 1, out
        assert out["cursor_ts"] == 21.0, out
        assert sorted(p["prompt"] for p in _read_jsonl(out["pairs_path"])) == ["A", "B"]

    def test_an_old_up_with_a_new_down_is_a_new_pair(self, cwd, monkeypatch):
        trace_dir = _write_traces(cwd, _trace_rows("C", 10, 30))
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        out = ls.harvest_from_traces(_state(harvest_cursor_ts=15.0))

        assert (out["pairs_harvested"], out["new_pairs"], out["cursor_ts"]) == (1, 1, 30.0)

    def test_a_pair_older_than_the_cursor_is_not_new(self, cwd, monkeypatch):
        trace_dir = _write_traces(cwd, _trace_rows("A", 10, 11))
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        out = ls.harvest_from_traces(_state(harvest_cursor_ts=11.0))  # == cursor: not newer

        assert (out["pairs_harvested"], out["new_pairs"]) == (1, 0)

    def test_traces_without_ts_are_new_only_while_there_is_no_cursor(self, cwd, monkeypatch):
        trace_dir = _write_traces(cwd, _trace_rows("A"))
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        first = ls.harvest_from_traces(_state())
        later = ls.harvest_from_traces(_state(harvest_cursor_ts=15.0))

        assert (first["pairs_harvested"], first["new_pairs"], first["cursor_ts"]) == (1, 1, None)
        # Used once, so not re-used forever: still trained on, no longer a reason to train.
        assert (later["pairs_harvested"], later["new_pairs"], later["cursor_ts"]) == (1, 0, None)

    def test_a_trace_without_ts_next_to_a_new_one_is_new_through_the_other(self, cwd, monkeypatch):
        trace_dir = _write_traces(cwd, _trace_rows("A", up_ts=None, down_ts=30))
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        out = ls.harvest_from_traces(_state(harvest_cursor_ts=15.0))

        assert (out["new_pairs"], out["cursor_ts"]) == (1, 30.0)

    def test_an_iso_timestamp_in_a_log_counts_like_a_number(self, cwd, monkeypatch):
        trace_dir = _write_traces(
            cwd, _trace_rows("A", "2023-11-14T22:13:20Z", "2023-11-14T22:13:21+00:00")
        )
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", trace_dir)

        out = ls.harvest_from_traces(_state(harvest_cursor_ts=T0))

        assert (out["new_pairs"], out["cursor_ts"]) == (1, T0 + 1)

    def test_unpaired_traces_do_not_move_the_cursor(self, cwd, monkeypatch):
        rows = _trace_rows("A", 10, 11) + [
            {"id": "z", "prompt": "Z", "output": "x", "signal": "thumbs_up", "ts": 999}
        ]
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", _write_traces(cwd, rows))

        out = ls.harvest_from_traces(_state())

        assert out["cursor_ts"] == 11.0

    def test_thumbs_pairs_use_the_cursor_too(self, cwd):
        _pair_in_db("t.db", "A", T0 + 1, T0 + 2)
        _pair_in_db("t.db", "B", T0 + 10, T0 + 11)

        out = ls.harvest_from_traces(_state(thumbs_db="t.db", harvest_cursor_ts=T0 + 5))

        assert out["thumbs_pairs"] == 2, out  # pairs READ from the database
        assert (out["pairs_harvested"], out["new_pairs"]) == (2, 1)
        assert out["cursor_ts"] == T0 + 11
        assert sorted(p["prompt"] for p in _read_jsonl(out["pairs_path"])) == ["A", "B"]

    def test_traces_and_thumbs_are_unioned_and_new_counts_each_pair_once(self, cwd, monkeypatch):
        # "dup" is in both sources: old in the log, new in the database.
        monkeypatch.setenv(
            "SOUP_LOOP_TRACE_DIR",
            _write_traces(cwd, _trace_rows("dup", 1, 2) + _trace_rows("old", 3, 4)),
        )
        _db("t.db", [(T0 + 20, "dup", "good", "up"), (T0 + 21, "dup", "bad", "down")])

        out = ls.harvest_from_traces(_state(thumbs_db="t.db", harvest_cursor_ts=T0))

        assert out["pairs_harvested"] == 2, out  # the duplicate is one pair
        assert out["new_pairs"] == 1, out
        assert out["cursor_ts"] == T0 + 21

    def test_a_pair_old_in_both_sources_is_old(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", _write_traces(cwd, _trace_rows("dup", 1, 2)))
        _db("t.db", [(2.0, "dup", "good", "up"), (3.0, "dup", "bad", "down")])

        out = ls.harvest_from_traces(_state(thumbs_db="t.db", harvest_cursor_ts=10.0))

        assert (out["pairs_harvested"], out["new_pairs"]) == (1, 0)

    def test_the_pairs_file_is_not_written_while_training_would_wait(self, cwd, monkeypatch):
        """Else every poll of a quiet loop would leave another copy of the same pairs behind."""
        monkeypatch.setenv(
            "SOUP_LOOP_TRACE_DIR",
            _write_traces(cwd, _trace_rows("A", 1, 2) + _trace_rows("B", 20, 21)),
        )

        waiting = ls.harvest_from_traces(_state(harvest_cursor_ts=10.0, min_new_pairs=2))
        assert (waiting["pairs_harvested"], waiting["new_pairs"]) == (2, 1)
        assert waiting["pairs_path"] is None
        assert not os.path.exists(os.path.join(".soup-loops", "pairs"))

        enough = ls.harvest_from_traces(_state(harvest_cursor_ts=0.0, min_new_pairs=2))
        assert enough["new_pairs"] == 2
        assert len(_read_jsonl(enough["pairs_path"])) == 2

    def test_nothing_harvested_reports_zero_new_pairs_and_no_cursor(self, cwd):
        out = ls.harvest_from_traces(_state(harvest_cursor_ts=5.0))

        assert out["pairs_harvested"] == 0
        assert out["new_pairs"] == 0
        assert out["cursor_ts"] is None
        assert out["pairs_path"] is None


# ---------------------------------------------------------------------------
# train_dpo_from_pairs: skip when too few pairs are new
# ---------------------------------------------------------------------------


class _Trainer:
    """Stand-in for the ``soup train`` subprocess: records the pairs it trains on."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls = 0
        self.trained_on: list = []

    def __call__(self, argv, **kwargs):
        self.calls += 1
        with open(argv[argv.index("--config") + 1], encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        self.trained_on.append(_read_jsonl(config["data"]["train"]))
        if self.returncode == 0:
            os.makedirs(config["output"], exist_ok=True)
            with open(os.path.join(config["output"], "adapter_config.json"), "w") as fh:
                json.dump({"base_model_name_or_path": config["base"]}, fh)
        return SimpleNamespace(returncode=self.returncode, stderr=b"")


class TestTrainSkip:
    def _ctx(self, cwd, new_pairs, pairs=2):
        path = cwd / "pairs.jsonl"
        row = {"prompt": "q", "chosen": "a", "rejected": "b", "source": "x"}
        path.write_text(json.dumps(row) + "\n", "utf-8")
        ctx = {"pairs_harvested": pairs, "pairs_path": str(path)}
        if new_pairs is not None:
            ctx["new_pairs"] = new_pairs
        return ctx

    def test_too_few_new_pairs_skip_with_the_reason(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)

        out = ls.train_dpo_from_pairs(_state(base_model="m", min_new_pairs=3), self._ctx(cwd, 1))

        assert out["skipped"] is True and out["run_id"] is None
        assert out["notes"] == (
            "not_enough_new_data: 1 new preference pairs since the last training run, "
            "training needs 3 (soup loop init --min-new-pairs)"
        )
        assert trainer.calls == 0

    def test_zero_new_pairs_skip_under_the_default(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)

        out = ls.train_dpo_from_pairs(_state(base_model="m"), self._ctx(cwd, 0))

        assert out["skipped"] is True
        assert out["notes"].startswith("not_enough_new_data: 0 new preference pairs")
        assert out["notes"].endswith("training needs 1 (soup loop init --min-new-pairs)")
        assert trainer.calls == 0

    def test_exactly_the_minimum_trains(self, cwd, monkeypatch):
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)

        out = ls.train_dpo_from_pairs(_state(base_model="m", min_new_pairs=3), self._ctx(cwd, 3))

        assert out["skipped"] is False and out["run_id"], out
        assert trainer.calls == 1

    def test_a_harvest_that_reports_no_new_pairs_treats_every_pair_as_new(self, cwd, monkeypatch):
        """A custom harvest_fn knows nothing of the cursor: what it hands over is all new."""
        trainer = _Trainer()
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)

        trained = ls.train_dpo_from_pairs(
            _state(base_model="m", min_new_pairs=2), self._ctx(cwd, None, pairs=2)
        )
        waiting = ls.train_dpo_from_pairs(
            _state(base_model="m", min_new_pairs=3), self._ctx(cwd, None, pairs=2)
        )

        assert trained["skipped"] is False, trained
        assert waiting["skipped"] is True
        assert waiting["notes"].startswith("not_enough_new_data: 2 new preference pairs")
        assert trainer.calls == 1

    def test_the_zero_pairs_note_is_unchanged(self, cwd):
        out = ls.train_dpo_from_pairs(
            _state(base_model="m", min_new_pairs=3), {"pairs_harvested": 0, "new_pairs": 0}
        )

        assert out["notes"] == "no_preference_pairs: 0 valid pairs harvested, training skipped"


# ---------------------------------------------------------------------------
# run_once: when the cursor moves
# ---------------------------------------------------------------------------


def _config(train, cursor=500.0, gate="OK", pairs=3):
    return WatchConfig(
        harvest_fn=lambda s: {"pairs_harvested": pairs, "new_pairs": pairs, "cursor_ts": cursor},
        train_fn=lambda s, c: train,
        gate_fn=lambda s, c: {"gate_verdict": gate},
        deploy_fn=lambda s, c: {"deployed": False, "canary_verdict": None},
        cost_fn=lambda s: 0.0,
    )


def _base_state(**kwargs) -> LoopState:
    return LoopState(served_model="m", eval_suite="e", baseline="b", **kwargs)


class TestCursorAdvance:
    """What a training run moves is ``attempt_cursor_ts``, the newest feedback it read.

    ``harvest_cursor_ts`` used to be this; it now follows production (a promotion moves it, a
    rollback returns it), and the trigger counts what is new after the later of the two.
    """

    def test_it_advances_after_a_training_run(self, cwd):
        new_state, _, _ = run_once(_base_state(), _config({"run_id": "r1", "skipped": False}))

        assert new_state.attempt_cursor_ts == 500.0
        # The data cursor follows the version in production, not the last training run: it
        # moves when the candidate is promoted (tests/test_loop_verdict_v2.py).
        assert new_state.harvest_cursor_ts is None

    @pytest.mark.parametrize("gate", ["OK", "MAJOR", "SKIPPED"])
    def test_the_verdict_does_not_matter(self, cwd, gate):
        """DON'T SHIP still consumed the data: retraining on it would loop forever."""
        new_state, record, _ = run_once(
            _base_state(), _config({"run_id": "r1", "skipped": False}, gate=gate)
        )

        assert record.gate_verdict == gate
        assert new_state.attempt_cursor_ts == 500.0

    @pytest.mark.parametrize(
        "train",
        [
            {"run_id": None, "skipped": True},                         # skipped (not enough data)
            {"run_id": None, "skipped": True, "notes": "x"},           # failed subprocess
            {"run_id": "r1", "skipped": True},                         # skipped wins
            {"run_id": None, "skipped": False},                        # no run to credit
            {"skipped": False},
        ],
    )
    def test_it_stays_when_training_did_not_run(self, cwd, train):
        new_state, _, _ = run_once(_base_state(), _config(train))

        assert new_state.attempt_cursor_ts is None

    def test_it_never_moves_back(self, cwd):
        new_state, _, _ = run_once(
            _base_state(attempt_cursor_ts=600.0), _config({"run_id": "r1", "skipped": False})
        )

        assert new_state.attempt_cursor_ts == 600.0

    def test_it_moves_forward_from_an_earlier_value(self, cwd):
        new_state, _, _ = run_once(
            _base_state(attempt_cursor_ts=100.0), _config({"run_id": "r1", "skipped": False})
        )

        assert new_state.attempt_cursor_ts == 500.0

    def test_a_harvest_without_a_cursor_leaves_it_alone(self, cwd):
        new_state, _, _ = run_once(
            _base_state(attempt_cursor_ts=100.0),
            _config({"run_id": "r1", "skipped": False}, cursor=None),
        )

        assert new_state.attempt_cursor_ts == 100.0

    @pytest.mark.parametrize("bad", ["500", float("nan"), float("inf"), -5.0, True, 1e300, [1]])
    def test_a_garbage_cursor_from_a_custom_harvest_is_ignored(self, cwd, bad):
        new_state, _, _ = run_once(
            _base_state(), _config({"run_id": "r1", "skipped": False}, cursor=bad)
        )

        assert new_state.attempt_cursor_ts is None

    def test_an_integer_cursor_is_stored_as_a_number(self, cwd):
        new_state, _, _ = run_once(
            _base_state(), _config({"run_id": "r1", "skipped": False}, cursor=500)
        )

        assert new_state.attempt_cursor_ts == 500.0

    def test_the_default_stub_stages_never_move_it(self, cwd):
        new_state, _, _ = run_once(_base_state(), WatchConfig(cost_fn=lambda s: 0.0))

        assert new_state.attempt_cursor_ts is None


class TestLifetimeCounters:
    """traces_collected / pairs_distilled count what a training run used."""

    @staticmethod
    def _config(train):
        return WatchConfig(
            harvest_fn=lambda s: {"pairs_harvested": 4, "traces_collected": 9},
            train_fn=lambda s, c: train,
            gate_fn=lambda s, c: {"gate_verdict": "OK"},
            deploy_fn=lambda s, c: {"deployed": False, "canary_verdict": None},
            cost_fn=lambda s: 0.0,
        )

    def test_a_training_run_adds_its_pairs_and_traces(self, cwd):
        new_state, _, _ = run_once(
            _base_state(traces_collected=1, pairs_distilled=2),
            self._config({"run_id": "r1", "skipped": False}),
        )

        assert (new_state.traces_collected, new_state.pairs_distilled) == (10, 6)

    @pytest.mark.parametrize(
        "train",
        [
            {"run_id": None, "skipped": True},   # waited for more feedback
            {"run_id": None, "skipped": True, "notes": "failed"},
            {"run_id": "r1", "skipped": True},
            {"run_id": None, "skipped": False},
        ],
    )
    def test_an_iteration_without_a_run_adds_nothing_but_still_counts_itself(self, cwd, train):
        new_state, record, _ = run_once(
            _base_state(traces_collected=1, pairs_distilled=2), self._config(train)
        )

        assert (new_state.traces_collected, new_state.pairs_distilled) == (1, 2)
        assert record.pairs_harvested == 4  # the record still says what the harvest saw
        assert (new_state.iteration_count, new_state.runs_today) == (1, 1)


# ---------------------------------------------------------------------------
# The pre-wired stages together (no server): the loop trains when there is news
# ---------------------------------------------------------------------------


class _Loop:
    def __init__(self, root, monkeypatch, **init) -> None:
        self.root = root
        self.trainer = _Trainer()
        self.decision = "SHIP"
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", self.trainer)
        monkeypatch.setattr(ls, "_SHIP_RUNNER", self._ship)
        self.state, _ = init_state(
            "./base", "tasks.jsonl", "b", pre_wired=True, thumbs_db="t.db", base_model="org/base",
            verify="ship", **init,
        )
        self.config = replace(
            ls.build_prewired_watch_config(max_iterations=1), cost_fn=lambda s: 0.0
        )

    def _ship(self, argv, **kwargs):
        with open(argv[argv.index("--output") + 1], "w", encoding="utf-8") as fh:
            json.dump({"decision": self.decision}, fh)
        return SimpleNamespace(returncode=0, stderr=b"")

    def cycle(self):
        self.state, record, _ = run_once(self.state, self.config)
        return record

    def pairs_files(self) -> int:
        directory = os.path.join(".soup-loops", "pairs")
        return len(os.listdir(directory)) if os.path.isdir(directory) else 0


@pytest.fixture
def loop(cwd, monkeypatch):
    return _Loop(cwd, monkeypatch)


class TestTheLoop:
    def test_new_feedback_on_a_new_prompt_retrains_on_old_and_new(self, loop):
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)
        first = loop.cycle()
        assert first.run_id and first.pairs_harvested == 1
        assert loop.state.attempt_cursor_ts == T0 + 2

        quiet = loop.cycle()
        assert quiet.run_id is None
        assert "not_enough_new_data: 0 new preference pairs since the last training run" in (
            quiet.notes
        )
        assert loop.trainer.calls == 1
        assert loop.state.attempt_cursor_ts == T0 + 2

        _pair_in_db("t.db", "Q2", T0 + 10, T0 + 11)
        third = loop.cycle()
        assert third.run_id and third.pairs_harvested == 2
        # The fresh LoRA trains on everything so far, not on the one new pair.
        assert sorted(row["prompt"] for row in loop.trainer.trained_on[-1]) == ["Q1", "Q2"]
        assert loop.state.attempt_cursor_ts == T0 + 11
        assert loop.cycle().run_id is None
        assert loop.trainer.calls == 2

    def test_a_quiet_cycle_leaves_no_new_pairs_file_behind(self, loop):
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)
        loop.cycle()
        assert loop.pairs_files() == 1

        for _ in range(3):
            loop.cycle()

        assert loop.pairs_files() == 1

    def test_a_new_up_on_an_old_prompt_retrains(self, loop):
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)
        loop.cycle()
        _db("t.db", [(T0 + 30, "Q1", "better", "up")])

        record = loop.cycle()

        assert record.run_id
        assert [(r["chosen"], r["rejected"]) for r in loop.trainer.trained_on[-1]] == [
            ("better", "bad")
        ]

    def test_dont_ship_still_consumed_the_data(self, loop):
        loop.decision = "DON'T SHIP"
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)

        first = loop.cycle()

        assert first.run_id and first.gate_verdict == "MAJOR"
        assert loop.state.attempt_cursor_ts == T0 + 2
        assert loop.cycle().run_id is None
        assert loop.trainer.calls == 1

    def test_a_failed_training_is_retried_on_the_same_data(self, loop):
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)
        loop.trainer.returncode = 1

        failed = loop.cycle()
        assert failed.run_id is None
        assert loop.state.attempt_cursor_ts is None

        loop.trainer.returncode = 0
        retry = loop.cycle()
        assert retry.run_id
        assert loop.trainer.calls == 2
        assert loop.state.attempt_cursor_ts == T0 + 2

    def test_feedback_accumulates_until_min_new_pairs(self, cwd, monkeypatch):
        loop = _Loop(cwd, monkeypatch, min_new_pairs=2)
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)

        waiting = loop.cycle()
        assert waiting.run_id is None
        assert (
            "not_enough_new_data: 1 new preference pairs since the last training run, "
            "training needs 2 (soup loop init --min-new-pairs)"
        ) in waiting.notes
        assert loop.state.attempt_cursor_ts is None

        _pair_in_db("t.db", "Q2", T0 + 3, T0 + 4)
        enough = loop.cycle()

        assert enough.run_id
        assert sorted(row["prompt"] for row in loop.trainer.trained_on[-1]) == ["Q1", "Q2"]
        assert loop.state.attempt_cursor_ts == T0 + 4

    def test_the_cursor_survives_a_restart(self, loop):
        _pair_in_db("t.db", "Q1", T0 + 1, T0 + 2)
        loop.cycle()
        write_state(loop.state)

        assert read_state().attempt_cursor_ts == T0 + 2


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------


class TestLoopStateFields:
    def test_defaults(self):
        state = _base_state()
        assert state.harvest_cursor_ts is None
        assert state.min_new_pairs == 1

    def test_the_fields_come_last(self):
        """Appended at the end when added; fields added after them (see
        tests/test_loop_feedback_datasets.py) follow, so nothing else sits between."""
        names = list(LoopState.__dataclass_fields__)
        tail = names[names.index("harvest_cursor_ts"):]
        assert tail[:2] == ["harvest_cursor_ts", "min_new_pairs"]
        assert names.index("min_new_pairs") > names.index("previous_production_adapter")

    @pytest.mark.parametrize("value", [0, 0.0, 0.5, 1700000000, 1700000000.25])
    def test_a_cursor_is_a_finite_non_negative_number(self, value):
        assert _base_state(harvest_cursor_ts=value).harvest_cursor_ts == value

    @pytest.mark.parametrize(
        "value", [True, False, "5", [1], -1, -0.5, float("nan"), float("inf"), 1e300]
    )
    def test_a_bad_cursor_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="harvest_cursor_ts"):
            _base_state(harvest_cursor_ts=value)

    @pytest.mark.parametrize("value", [1, 2, 50])
    def test_min_new_pairs_accepts_positive_ints(self, value):
        assert _base_state(min_new_pairs=value).min_new_pairs == value

    @pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, 1.5, "2", None])
    def test_a_bad_min_new_pairs_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="min_new_pairs"):
            _base_state(min_new_pairs=value)

    def test_init_state_threads_both_fields(self, cwd):
        state, path = init_state(
            "m", "e", "b", harvest_cursor_ts=T0, min_new_pairs=4, path=str(cwd / "loop.json")
        )

        assert (state.harvest_cursor_ts, state.min_new_pairs) == (T0, 4)
        reread = read_state(path)
        assert (reread.harvest_cursor_ts, reread.min_new_pairs) == (T0, 4)

    def test_init_state_defaults(self, cwd):
        state, _ = init_state("m", "e", "b", path=str(cwd / "loop.json"))

        assert (state.harvest_cursor_ts, state.min_new_pairs) == (None, 1)

    def test_a_loop_yaml_from_before_the_fields_still_loads(self, cwd):
        legacy = {
            "served_model": "m", "eval_suite": "e", "baseline": "b", "status": "running",
            "iteration_count": 3, "pairs_distilled": 7, "created_at": "x", "updated_at": "y",
        }
        (cwd / "legacy.json").write_text(json.dumps(legacy), "utf-8")

        state = read_state(str(cwd / "legacy.json"))

        assert state.iteration_count == 3
        assert (state.harvest_cursor_ts, state.min_new_pairs) == (None, 1)

    def test_an_invalid_value_in_the_file_is_reported_with_its_name(self, cwd):
        bad = {"served_model": "m", "eval_suite": "e", "baseline": "b", "min_new_pairs": 0}
        (cwd / "bad.json").write_text(json.dumps(bad), "utf-8")

        with pytest.raises(ValueError, match="state file has invalid contents: min_new_pairs"):
            read_state(str(cwd / "bad.json"))

    def test_a_cursor_round_trips_through_the_file(self, cwd):
        state, path = init_state("m", "e", "b", path=str(cwd / "loop.json"))
        write_state(replace(state, harvest_cursor_ts=T0 + 0.25), path)

        assert read_state(path).harvest_cursor_ts == T0 + 0.25
        assert math.isfinite(read_state(path).harvest_cursor_ts)


# ---------------------------------------------------------------------------
# soup loop init / status
# ---------------------------------------------------------------------------


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


class TestCli:
    def test_the_default_is_one(self, cwd):
        result, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert (state.min_new_pairs, state.harvest_cursor_ts) == (1, None)

    def test_the_flag_is_stored(self, cwd):
        result, _ = _cli(*_INIT, "--min-new-pairs", "3")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().min_new_pairs == 3

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_below_one_is_refused(self, cwd, value):
        result, text = _cli(*_INIT, "--min-new-pairs", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--min-new-pairs must be a positive int" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    @pytest.mark.parametrize("value", ["abc", "1.5", ""])
    def test_a_non_integer_is_refused(self, cwd, value):
        result, text = _cli(*_INIT, "--min-new-pairs", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert re.search(r"Invalid value for\W+--min-new-pairs", text), text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_help_names_the_flag(self, cwd):
        result, text = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--min-new-pairs" in text

    def test_status_before_any_training_says_the_start(self, cwd):
        _cli(*_INIT, "--min-new-pairs", "3")

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"new_data_since\W+the start", text), text
        assert re.search(r"min_new_pairs\W+3", text), text

    def test_status_shows_the_cursor_as_utc_iso(self, cwd):
        _cli(*_INIT)
        write_state(replace(read_state(), harvest_cursor_ts=1_700_000_000.9))

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"new_data_since\W+2023-11-14T22:13:20\+00:00", text), text
        assert re.search(r"min_new_pairs\W+1\b", text), text

    def test_status_of_a_state_written_before_the_fields(self, cwd):
        _cli(*_INIT)
        path = os.path.join(".soup", "loop.yaml")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        del data["harvest_cursor_ts"], data["min_new_pairs"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"new_data_since\W+the start", text), text
        assert re.search(r"min_new_pairs\W+1\b", text), text
