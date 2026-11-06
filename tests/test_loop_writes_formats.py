"""``soup loop`` writes the iteration, version and progress logs of docs/loop-formats.md.

``soup serve`` writes the conversation and feedback logs; the loop writes the other three:
``.soup-loops/iterations.jsonl`` (a snapshot of each iteration, a new line whenever its outcome
changes), ``.soup-loops/versions.jsonl`` (one line per change of the active version) and
``.soup-loops/progress.jsonl`` (stage events). This file covers the append writer, the
line builders, the version counter in ``loop.yaml``, the daemon and the CLI commands, and
one end-to-end scenario whose every line is checked against the packaged JSON Schemas.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid

import pytest

from soup_cli.utils import loop_formats, loop_log
from soup_cli.utils.loop_log import LoopLog, append_json_line

jsonschema = pytest.importorskip("jsonschema")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _validator(name: str):
    return jsonschema.Draft202012Validator(
        loop_formats.load_schema(name),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def _errors(name: str, record: dict) -> list:
    return [error.message for error in _validator(name).iter_errors(record)]


def _lines(path) -> list:
    """Every line of a JSONL file, each parsed as one JSON object."""
    with open(path, "rb") as fh:
        raw = fh.read()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM"
    assert raw.endswith(b"\n"), "the last line is not terminated"
    assert b"\r" not in raw, "CRLF line endings"
    records = [json.loads(line) for line in raw.decode("utf-8").split("\n")[:-1]]
    for record in records:
        assert isinstance(record, dict), record
    return records


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# The append writer: one write call per line, append mode, contained, no symlink
# ---------------------------------------------------------------------------


class TestAppendWriter:
    def test_a_line_is_one_json_object_ending_in_a_newline(self, cwd):
        append_json_line("log.jsonl", {"a": 1, "text": "héllo ☃"})

        raw = (cwd / "log.jsonl").read_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == 1
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in raw
        assert json.loads(raw.decode("utf-8")) == {"a": 1, "text": "héllo ☃"}

    def test_the_directory_is_created(self, cwd):
        append_json_line(os.path.join("deep", "er", "log.jsonl"), {"a": 1})

        assert (cwd / "deep" / "er" / "log.jsonl").is_file()

    def test_lines_accumulate_in_order(self, cwd):
        for number in range(5):
            append_json_line("log.jsonl", {"n": number})

        assert [record["n"] for record in _lines(cwd / "log.jsonl")] == [0, 1, 2, 3, 4]

    def test_a_line_is_written_by_one_write_call(self, cwd, monkeypatch):
        calls = []
        real_write = os.write

        def counting_write(fd, data):
            calls.append(bytes(data))
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", counting_write)

        append_json_line("log.jsonl", {"a": 1})
        append_json_line("log.jsonl", {"b": 2, "text": "x" * 5000})

        assert len(calls) == 2
        assert [json.loads(call) for call in calls] == [
            {"a": 1}, {"b": 2, "text": "x" * 5000}
        ]
        assert all(call.endswith(b"\n") and call.count(b"\n") == 1 for call in calls)

    def test_only_a_short_write_is_continued(self, cwd, monkeypatch):
        calls = []
        real_write = os.write

        def short_write(fd, data):
            calls.append(len(data))
            return real_write(fd, bytes(data)[:7])

        monkeypatch.setattr(os, "write", short_write)

        append_json_line("log.jsonl", {"key": "a value long enough to need several writes"})

        assert len(calls) > 1
        (record,) = _lines(cwd / "log.jsonl")
        assert record == {"key": "a value long enough to need several writes"}

    def test_a_write_that_stores_nothing_is_an_error_not_a_hang(self, cwd, monkeypatch):
        monkeypatch.setattr(os, "write", lambda fd, data: 0)

        with pytest.raises(OSError):
            append_json_line("log.jsonl", {"a": 1})

    def test_the_file_is_opened_in_append_mode_and_earlier_lines_survive(self, cwd):
        # Another writer (a desktop app) already wrote two lines: never rewritten, never
        # overwritten from offset 0.
        existing = b'{"from": "zero", "n": 1}\n{"from": "zero", "n": 2}\n'
        (cwd / "log.jsonl").write_bytes(existing)

        append_json_line("log.jsonl", {"from": "soup"})

        raw = (cwd / "log.jsonl").read_bytes()
        assert raw.startswith(existing)
        assert [record["from"] for record in _lines(cwd / "log.jsonl")] == [
            "zero", "zero", "soup"
        ]

    def test_a_line_another_writer_appends_just_before_our_write_is_not_overwritten(
        self, cwd, monkeypatch
    ):
        # The position this process last saw is the end of the file; a desktop app appends
        # a line before our write lands. Only append mode puts ours after it.
        (cwd / "log.jsonl").write_bytes(b'{"from": "soup", "n": 0}\n')
        real_write = os.write
        done = []

        def racing_write(fd, data):
            if not done:
                done.append(True)
                with open(cwd / "log.jsonl", "ab") as other:
                    other.write(b'{"from": "zero", "n": 1}\n')
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", racing_write)

        append_json_line("log.jsonl", {"from": "soup", "n": 2})

        assert [(r["from"], r["n"]) for r in _lines(cwd / "log.jsonl")] == [
            ("soup", 0), ("zero", 1), ("soup", 2)
        ]

    def test_another_writer_between_two_appends_loses_nothing(self, cwd):
        append_json_line("log.jsonl", {"from": "soup", "n": 1})
        with open(cwd / "log.jsonl", "ab") as other:
            other.write(b'{"from": "zero", "n": 2}\n')
        append_json_line("log.jsonl", {"from": "soup", "n": 3})

        assert [(r["from"], r["n"]) for r in _lines(cwd / "log.jsonl")] == [
            ("soup", 1), ("zero", 2), ("soup", 3)
        ]

    def test_a_torn_last_line_does_not_swallow_the_next_record(self, cwd):
        (cwd / "log.jsonl").write_bytes(b'{"ok": 1}\n{"torn": ')

        append_json_line("log.jsonl", {"next": True})

        raw = (cwd / "log.jsonl").read_bytes()
        lines = raw.split(b"\n")
        assert lines[0] == b'{"ok": 1}' and lines[1] == b'{"torn": '
        assert json.loads(lines[2]) == {"next": True}
        assert raw.endswith(b"\n")

    def test_a_record_that_is_not_json_is_refused_and_nothing_is_written(self, cwd):
        with pytest.raises(ValueError):
            append_json_line("log.jsonl", {"x": float("nan")})
        with pytest.raises(TypeError):
            append_json_line("log.jsonl", {"x": object()})
        with pytest.raises(TypeError):
            append_json_line("log.jsonl", ["not", "a", "mapping"])

        assert not (cwd / "log.jsonl").exists() or (cwd / "log.jsonl").read_bytes() == b""

    def test_a_lone_surrogate_is_escaped_not_a_crash(self, cwd):
        append_json_line("log.jsonl", {"text": "bad \ud800 text"})

        (record,) = _lines(cwd / "log.jsonl")
        assert record["text"] == "bad \ud800 text"

    def test_a_line_over_one_mebibyte_is_refused(self, cwd):
        with pytest.raises(ValueError, match="exceeds 1 MiB"):
            append_json_line("log.jsonl", {"text": "x" * (1024 * 1024 + 1)})

        assert not (cwd / "log.jsonl").exists()

    def test_a_path_outside_the_project_is_refused_by_name(self, cwd):
        outside = cwd.parent / "outside-the-project.jsonl"

        with pytest.raises(ValueError, match="must stay under cwd"):
            append_json_line(str(outside), {"a": 1})

        assert not outside.exists()

    def test_a_dotdot_path_is_refused(self, cwd):
        with pytest.raises(ValueError, match="must stay under cwd"):
            append_json_line(os.path.join("..", "escaped.jsonl"), {"a": 1})

    @pytest.mark.requires_symlink
    def test_a_symlinked_target_is_refused_and_not_written_through(self, cwd):
        victim = cwd / "victim.txt"
        victim.write_text("keep me\n", "utf-8")
        os.symlink(victim, cwd / "log.jsonl")

        with pytest.raises(ValueError, match="symlink"):
            append_json_line("log.jsonl", {"a": 1})

        assert victim.read_text("utf-8") == "keep me\n"

    @pytest.mark.requires_symlink
    def test_a_symlinked_directory_is_refused(self, cwd):
        real = cwd / "real"
        real.mkdir()
        os.symlink(real, cwd / "linked", target_is_directory=True)

        with pytest.raises((ValueError, OSError)):
            append_json_line(os.path.join("linked", "log.jsonl"), {"a": 1})

        assert not (real / "log.jsonl").exists()


# ---------------------------------------------------------------------------
# LoopLog: a failure is a note and a warning, never an exception
# ---------------------------------------------------------------------------


class TestLoopLog:
    def test_the_default_paths_are_the_contract_layout(self, cwd):
        log = LoopLog()

        assert log.path("iteration") == os.path.join(".soup-loops", "iterations.jsonl")
        assert log.path("version") == os.path.join(".soup-loops", "versions.jsonl")
        assert log.path("progress") == os.path.join(".soup-loops", "progress.jsonl")
        for kind, relative in loop_formats.DEFAULT_LAYOUT.items():
            if kind in ("iteration", "version", "progress"):
                assert log.path(kind).replace(os.sep, "/") == relative

    def test_a_base_directory_moves_all_three_files(self, cwd):
        log = LoopLog("iters")

        assert log.path("iteration") == os.path.join("iters", "iterations.jsonl")

    def test_each_kind_goes_to_its_own_file(self, cwd):
        log = LoopLog()

        assert log.iteration({"k": "iteration"}) and log.version({"k": "version"})
        assert log.progress({"k": "progress"})

        assert _lines(cwd / ".soup-loops" / "iterations.jsonl") == [{"k": "iteration"}]
        assert _lines(cwd / ".soup-loops" / "versions.jsonl") == [{"k": "version"}]
        assert _lines(cwd / ".soup-loops" / "progress.jsonl") == [{"k": "progress"}]

    def test_a_write_failure_is_returned_logged_and_noted_never_raised(self, cwd, caplog):
        # A directory where the file should be: opening it for append fails.
        os.makedirs(os.path.join(".soup-loops", "iterations.jsonl"))
        log = LoopLog()

        with caplog.at_level(logging.WARNING):
            written = log.iteration({"a": 1})

        assert written is False
        assert any("iterations.jsonl" in message for message in caplog.messages), caplog.messages
        (note,) = log.take_failures()
        assert note.startswith("iterations.jsonl not written (")
        assert str(cwd) not in note  # a class name, not a path

    def test_a_record_that_is_not_json_is_a_failure_note_too(self, cwd):
        log = LoopLog()

        assert log.version({"x": float("nan")}) is False

        assert log.take_failures() == ["versions.jsonl not written (ValueError)"]

    def test_a_path_outside_the_project_is_a_failure_note(self, cwd):
        log = LoopLog(str(cwd.parent / "elsewhere"))

        assert log.progress({"a": 1}) is False

        assert log.take_failures() == ["progress.jsonl not written (ValueError)"]

    def test_failures_are_taken_once(self, cwd):
        os.makedirs(os.path.join(".soup-loops", "progress.jsonl"))
        log = LoopLog()
        log.progress({"a": 1})

        assert len(log.take_failures()) == 1
        assert log.take_failures() == []

    def test_one_note_per_distinct_failure(self, cwd):
        os.makedirs(os.path.join(".soup-loops", "progress.jsonl"))
        log = LoopLog()
        for _ in range(3):
            log.progress({"a": 1})

        assert len(log.take_failures()) == 1

    def test_a_disabled_log_writes_nothing_and_fails_nothing(self, cwd):
        log = LoopLog(enabled=False)

        assert log.iteration({"a": 1}) is False
        assert log.version({"a": 1}) is False
        assert log.progress({"a": 1}) is False

        assert not (cwd / ".soup-loops").exists()
        assert log.take_failures() == []
        assert loop_log.NO_LOG.take_failures() == []

    def test_a_good_write_after_a_bad_one_still_lands(self, cwd):
        log = LoopLog()
        assert log.version({"x": float("nan")}) is False
        assert log.version({"x": 1}) is True

        assert _lines(cwd / ".soup-loops" / "versions.jsonl") == [{"x": 1}]


def test_line_ids_are_uuids_in_canonical_lowercase_form():
    line = loop_log.version_line(
        from_version=None, to_version=1, cause="initial", actor="user",
        iteration_id=None, served_as=None, reason=None,
    )

    assert str(uuid.UUID(line["id"])) == line["id"]
    assert _errors("version", line) == [], line


# ---------------------------------------------------------------------------
# loop.yaml: the version counter and the iteration a running canary belongs to
# ---------------------------------------------------------------------------

from soup_cli.utils.loop_state import LoopState, read_state, write_state  # noqa: E402

NEW_FIELDS = [
    "production_version",
    "previous_production_version",
    "latest_version",
    "production_iteration_id",
    "previous_production_iteration_id",
    "canary_iteration",
    "version_log_started",
]


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./base")
    kwargs.setdefault("eval_suite", "tasks.jsonl")
    kwargs.setdefault("baseline", "b")
    return LoopState(**kwargs)


def _snapshot(**fields) -> dict:
    return {"iteration_id": "iter-20261105T120000-1b2c3d4e", "outcome": "canary", **fields}


class TestVersionFields:
    def test_the_fields_come_last_in_this_order_with_these_defaults(self):
        names = list(LoopState.__dataclass_fields__)

        # The two trigger thresholds of the sft loop (tests/test_loop_sft_default.py) follow.
        assert names[-2:] == ["min_new_edits", "min_new_rated"]
        assert names[-len(NEW_FIELDS) - 2:-2] == NEW_FIELDS
        assert names.index("production_version") == names.index("canary_on_timeout") + 1
        state = _state()
        assert (state.production_version, state.latest_version) == (1, 1)
        assert state.previous_production_version is None
        assert state.production_iteration_id is None
        assert state.previous_production_iteration_id is None
        assert state.canary_iteration is None
        assert state.version_log_started is False

    @pytest.mark.parametrize("value", [1, 2, 99])
    def test_a_version_is_a_positive_integer(self, value):
        assert _state(production_version=value, latest_version=value).production_version == value

    @pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "2", None, [1]])
    def test_a_version_that_is_not_a_positive_integer_is_refused(self, value):
        with pytest.raises(ValueError, match="production_version must be a positive int"):
            _state(production_version=value)
        with pytest.raises(ValueError, match="latest_version must be a positive int"):
            _state(latest_version=value)

    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
    def test_a_previous_version_is_a_positive_integer_or_none(self, value):
        with pytest.raises(ValueError, match="previous_production_version must be"):
            _state(previous_production_version=value, latest_version=2)

    def test_the_latest_version_is_never_behind_the_production_version(self):
        with pytest.raises(ValueError, match="latest_version must be at least"):
            _state(production_version=3, latest_version=2)
        with pytest.raises(ValueError, match="latest_version must be at least"):
            _state(previous_production_version=3, latest_version=2)
        assert _state(production_version=1, latest_version=3).latest_version == 3

    @pytest.mark.parametrize(
        "name", ["production_iteration_id", "previous_production_iteration_id"]
    )
    @pytest.mark.parametrize("value", ["", "a/b", "..", "a..b", "x" * 129, "a\x00b", 5, "-lead"])
    def test_an_iteration_id_must_be_safe_as_a_directory_name(self, name, value):
        with pytest.raises((ValueError, TypeError)):
            _state(**{name: value})

    def test_the_contract_form_of_an_id_is_accepted(self):
        state = _state(production_iteration_id="iter-20261105T120000-1b2c3d4e")

        assert state.production_iteration_id == "iter-20261105T120000-1b2c3d4e"

    def test_the_flag_must_be_a_bool(self):
        with pytest.raises(ValueError, match="version_log_started must be bool"):
            _state(version_log_started=1)

    def test_a_running_canary_carries_its_iteration_snapshot(self):
        state = _state(canary_active="loop-train-ab", canary_iteration=_snapshot(to_version=2))

        assert state.canary_iteration["to_version"] == 2

    @pytest.mark.parametrize(
        "snapshot",
        [
            "text",
            [1],
            {},
            {"iteration_id": "iter-1"},  # no outcome
            {"iteration_id": "iter-1", "outcome": "promoted"},  # only a canary is carried
            {"iteration_id": "../x", "outcome": "canary"},
            {"iteration_id": "iter-1", "outcome": "canary", "n": float("nan")},
            {"iteration_id": "iter-1", "outcome": "canary", "n": object()},
            {"iteration_id": "iter-1", "outcome": "canary", 1: "int key"},
            {"iteration_id": "iter-1", "outcome": "canary", "pad": "x" * 40000},
        ],
    )
    def test_a_snapshot_that_is_not_plain_small_json_of_a_canary_is_refused(self, snapshot):
        with pytest.raises(ValueError, match="canary_iteration"):
            _state(canary_active="c", canary_iteration=snapshot)

    def test_everything_survives_a_write_and_a_read(self, cwd):
        state = _state(
            canary_active="loop-train-ab",
            canary_iteration=_snapshot(to_version=3, canary={"traffic_pct": 5.0}),
            production_name="loop-train-aa",
            production_version=2,
            previous_production_version=1,
            latest_version=2,
            production_iteration_id="iter-b",
            previous_production_iteration_id="iter-a",
            version_log_started=True,
        )
        os.makedirs(".soup")

        write_state(state, os.path.join(".soup", "loop.yaml"))
        again = read_state(os.path.join(".soup", "loop.yaml"))

        for name in NEW_FIELDS:
            assert getattr(again, name) == getattr(state, name), name


# ---------------------------------------------------------------------------
# The iteration line: built from the IterationRecord plus what the stages returned
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
from dataclasses import replace  # noqa: E402

from soup_cli.utils.loop_iteration import IterationRecord  # noqa: E402

ITER = "iter-20261105T120000-1b2c3d4e"
RUN = "loop-train-1b2c3d4e"


def _record(**kwargs) -> IterationRecord:
    base = {
        "iteration_id": ITER, "started_at": "2026-11-05T12:00:00+00:00",
        "finished_at": "2026-11-05T12:09:06+00:00", "pairs_harvested": 3, "run_id": RUN,
        "gate_verdict": "OK", "canary_verdict": None, "shipped": True, "rolled_back": False,
        "estimated_cost_usd": 0.0, "notes": "",
    }
    base.update(kwargs)
    return IterationRecord(**base)


def _stages(cwd, **override) -> dict:
    """What the pre-wired stages return for an iteration that reached a canary."""
    pairs = cwd / ".soup-loops" / "pairs" / "pairs-1b2c3d4e.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_bytes(b'{"prompt": "p"}\n')
    stages = {
        "harvest": {
            "pairs_harvested": 3, "new_pairs": 3, "cursor_ts": 1_793_880_000.5,
            "pairs_path": os.path.join(".soup-loops", "pairs", "pairs-1b2c3d4e.jsonl"),
            "traces_collected": 7, "dpo_pairs": 3, "kto_rows": 4, "sft_rows": 0,
        },
        "train": {
            "run_id": RUN, "skipped": False,
            "adapter_path": os.path.join(".soup-loops", "adapters", RUN),
        },
        "gate": {
            "gate_verdict": "OK", "verdict_method": "ship", "verdict_decision": "SHIP",
            "verdict_path": os.path.join(".soup-loops", "verdicts", RUN + ".json"),
            "verdict_baseline": {"kind": "base", "ref": "org/base"},
        },
        "deploy": {
            "deployed": True, "canary_verdict": None, "canary_active": RUN,
            "canary_adapter": os.path.join(".soup-loops", "adapters", RUN),
            "canary_traffic_pct": 20.0, "notes": f"canary {RUN}: 20% of keyed traffic",
        },
    }
    for name, fields in override.items():
        stages[name] = {**stages[name], **fields} if fields is not None else {}
    return stages


def _line(record, stages, *, state=None, new_state=None) -> dict:
    state = state or _state(base_model="org/base", verify="ship")
    new_state = new_state or replace(
        state, harvest_cursor_ts=stages["harvest"].get("cursor_ts"),
        canary_active=stages["deploy"].get("canary_active"), canary_traffic_pct=20.0,
    )
    line = loop_log.iteration_from_record(
        record, state=state, new_state=new_state, harvest=stages["harvest"],
        train=stages["train"], gate=stages["gate"], deploy=stages["deploy"],
    )
    assert _errors("iteration", line) == [], line
    return line


class TestIterationLine:
    def test_an_iteration_that_reached_a_canary(self, cwd):
        stages = _stages(cwd)
        digest = hashlib.sha256(b'{"prompt": "p"}\n').hexdigest()
        state = _state(
            base_model="org/base", verify="ship", production_version=1,
            production_iteration_id="iter-seed", latest_version=1,
        )

        line = _line(_record(), stages, state=state)

        assert line["schema_version"] == 1
        assert line["iteration_id"] == ITER
        assert line["parent_iteration_id"] == "iter-seed"
        assert (line["from_version"], line["to_version"]) == (1, 2)
        assert line["started_at"] == "2026-11-05T12:00:00+00:00"
        assert line["finished_at"] is None  # a canary is not a final outcome
        assert line["outcome"] == "canary"
        assert line["data"] == {"until": "2026-11-05T12:00:00.500000+00:00", "conversations": 7}
        assert line["dataset"] == {
            "path": ".soup-loops/pairs/pairs-1b2c3d4e.jsonl", "sha256": digest,
            "dpo_pairs": 3, "kto_rows": 4, "sft_rows": 0,
        }
        assert line["train"] == {
            "method": "dpo", "run_id": RUN, "base": "org/base", "root_base": "org/base"
        }
        assert line["verdict"] == {
            "decision": "SHIP", "method": "ship", "baseline": {"kind": "base", "ref": "org/base"},
            "path": f".soup-loops/verdicts/{RUN}.json", "reason": None,
        }
        assert line["canary"] == {"traffic_pct": 20.0, "verdict": "UNKNOWN"}
        assert line["artifacts"] == {
            "adapter": f".soup-loops/adapters/{RUN}", "merged": None, "gguf": None,
            "served_as": RUN,
        }
        assert line["estimated_cost_usd"] == 0.0
        assert line["reason"] == f"canary {RUN}: 20% of keyed traffic"

    def test_the_facts_come_from_the_record_it_is_given(self, cwd):
        stages = _stages(cwd)

        line = _line(
            _record(run_id="loop-train-other", started_at="2026-01-02T03:04:05+00:00",
                    estimated_cost_usd=1.5, iteration_id="iter-other"),
            stages,
        )

        assert line["iteration_id"] == "iter-other"
        assert line["train"]["run_id"] == "loop-train-other"
        assert line["started_at"] == "2026-01-02T03:04:05+00:00"
        assert line["estimated_cost_usd"] == 1.5

    def test_nothing_to_train_is_skipped_with_nulls_for_what_did_not_happen(self, cwd):
        stages = _stages(
            cwd,
            harvest={"pairs_path": None, "pairs_harvested": 0},
            train={"run_id": None, "skipped": True, "adapter_path": None,
                   "notes": "no_preference_pairs: 0 valid pairs harvested, training skipped"},
            gate={"gate_verdict": "SKIPPED", "verdict_method": "ship"},
            deploy={"deployed": False, "canary_active": None, "notes": None},
        )
        record = _record(
            run_id=None, gate_verdict="SKIPPED", shipped=False, pairs_harvested=0,
            finished_at="2026-11-05T12:00:03+00:00",
        )

        line = _line(record, stages)

        assert line["outcome"] == "skipped"
        assert line["finished_at"] == "2026-11-05T12:00:03+00:00"
        assert line["to_version"] is None and line["from_version"] == 1
        assert line["dataset"] is None and line["train"] is None
        assert line["verdict"] is None and line["canary"] is None
        assert line["artifacts"] is None
        assert line["reason"] == "no_preference_pairs: 0 valid pairs harvested, training skipped"

    def test_a_training_run_that_died_is_failed(self, cwd):
        stages = _stages(
            cwd,
            train={"run_id": None, "skipped": True, "adapter_path": None, "failed": True},
            gate={"gate_verdict": "SKIPPED"},
            deploy={"deployed": False, "canary_active": None, "notes": None},
        )

        line = _line(_record(run_id=None, gate_verdict="SKIPPED", shipped=False), stages)

        assert line["outcome"] == "failed" and line["to_version"] is None
        assert line["train"] is None
        assert line["reason"] == "a stage failed"

    def test_dont_ship_is_a_verdict_and_a_skipped_rollout(self, cwd):
        stages = _stages(
            cwd,
            gate={"gate_verdict": "MAJOR", "verdict_decision": "DON'T SHIP"},
            deploy={"deployed": False, "canary_active": None, "notes": None},
        )

        line = _line(_record(gate_verdict="MAJOR", shipped=False), stages)

        assert line["outcome"] == "skipped" and line["to_version"] is None
        assert line["verdict"]["decision"] == "DON'T SHIP"
        assert line["train"]["run_id"] == RUN  # it did train
        assert line["artifacts"]["served_as"] is None
        assert line["artifacts"]["adapter"] == f".soup-loops/adapters/{RUN}"
        assert line["canary"] is None
        assert "DON'T SHIP" in line["reason"]

    def test_a_verdict_that_could_not_run_is_an_error_verdict_with_its_reason(self, cwd):
        why = "verdict_skipped: soup ship wrote no verdict (exit 2)"
        stages = _stages(
            cwd,
            gate={"gate_verdict": "SKIPPED", "notes": why, "verdict_path": None},
            deploy={"deployed": False, "canary_active": None, "notes": None},
        )

        line = _line(_record(gate_verdict="SKIPPED", shipped=False), stages)

        assert line["outcome"] == "skipped"
        assert line["verdict"] == {
            "decision": "ERROR", "method": "ship",
            "baseline": {"kind": "base", "ref": "org/base"}, "path": None, "reason": why,
        }

    def test_a_skipped_verdict_without_an_error_is_skipped(self, cwd):
        stages = _stages(
            cwd, gate={"gate_verdict": "SKIPPED"},
            deploy={"deployed": False, "canary_active": None, "notes": None},
        )

        line = _line(_record(gate_verdict="SKIPPED", shipped=False), stages)

        assert line["verdict"]["decision"] == "SKIPPED" and line["verdict"]["reason"] is None

    def test_a_deploy_that_failed_after_a_good_verdict_is_failed(self, cwd):
        stages = _stages(
            cwd,
            deploy={"deployed": False, "canary_active": None, "failed": True,
                    "notes": "adapter load failed (HTTP 500)"},
        )

        line = _line(_record(shipped=False), stages)

        assert line["outcome"] == "failed" and line["to_version"] is None
        assert line["verdict"]["decision"] == "SHIP"
        assert line["canary"] is None
        assert line["reason"] == "adapter load failed (HTTP 500)"

    def test_a_deploy_that_was_not_attempted_is_skipped_not_failed(self, cwd):
        stages = _stages(
            cwd,
            deploy={"deployed": False, "canary_active": None,
                    "notes": "no SOUP_LOOP_SERVE_ENDPOINT configured"},
        )

        line = _line(_record(shipped=False), stages)

        assert line["outcome"] == "skipped"
        assert line["reason"] == "no SOUP_LOOP_SERVE_ENDPOINT configured"

    def test_the_baseline_of_a_ship_verdict_names_the_production_version(self, cwd):
        stages = _stages(
            cwd,
            gate={"verdict_baseline": {"kind": "version", "ref": ".soup-loops/adapters/old"}},
        )
        state = _state(
            base_model="org/base", verify="ship", production_name="old", production_version=2,
            latest_version=2,
        )

        line = _line(_record(), stages, state=state)

        assert line["verdict"]["baseline"] == {
            "kind": "version", "ref": ".soup-loops/adapters/old", "version": 2
        }
        assert (line["from_version"], line["to_version"]) == (2, 3)

    def test_the_eval_gate_has_no_baseline_the_format_can_name(self, cwd):
        stages = _stages(cwd, gate={"verdict_method": "eval_gate", "verdict_baseline": None})
        state = _state(base_model="org/base", verify="eval_gate")

        line = _line(_record(), stages, state=state)

        assert line["verdict"]["method"] == "eval_gate"
        assert line["verdict"]["decision"] == "SHIP"
        assert "baseline" not in line["verdict"] or line["verdict"]["baseline"] is None

    def test_stub_stages_that_report_little_still_give_a_valid_line(self, cwd):
        stages = {
            "harvest": {"pairs_harvested": 0, "pairs_path": None},
            "train": {"run_id": None, "skipped": True},
            "gate": {"gate_verdict": "SKIPPED"},
            "deploy": {"deployed": False, "canary_verdict": None},
        }

        line = _line(
            _record(run_id=None, gate_verdict="SKIPPED", shipped=False, pairs_harvested=0),
            stages,
        )

        assert line["outcome"] == "skipped" and line["data"] is None
        assert line["reason"] == "nothing to train"

    def test_counts_the_stages_did_not_report_are_left_out_not_made_up(self, cwd):
        stages = _stages(cwd, harvest={"dpo_pairs": None, "kto_rows": "many"})

        line = _line(_record(), stages)

        for key in ("dpo_pairs", "kto_rows"):
            assert key not in line["dataset"], key
        assert line["dataset"]["sft_rows"] == 0  # the one that was counted stays

    def test_the_sft_row_count_is_written_next_to_the_other_two(self, cwd):
        stages = _stages(cwd, harvest={"dpo_pairs": 0, "kto_rows": 2, "sft_rows": 5})

        line = _line(_record(), stages)  # (the schema knows dataset.sft_rows)

        assert {k: line["dataset"][k] for k in ("dpo_pairs", "kto_rows", "sft_rows")} == {
            "dpo_pairs": 0, "kto_rows": 2, "sft_rows": 5
        }

    def test_a_dataset_file_that_vanished_has_a_null_hash(self, cwd):
        stages = _stages(cwd, harvest={"pairs_path": ".soup-loops/pairs/gone.jsonl"})

        line = _line(_record(), stages)

        assert line["dataset"]["sha256"] is None
        assert line["dataset"]["path"] == ".soup-loops/pairs/gone.jsonl"

    def test_a_shipped_deploy_without_a_canary_is_not_called_promoted(self, cwd):
        stages = _stages(cwd, deploy={"deployed": True, "canary_active": None, "notes": None})

        line = _line(_record(), stages)

        assert line["outcome"] == "skipped" and line["to_version"] is None

    def test_a_long_reason_is_cut_to_the_schema_limit(self, cwd):
        stages = _stages(cwd, train={"notes": "n" * 9000, "run_id": None, "skipped": True})

        line = _line(_record(run_id=None, shipped=False, gate_verdict="SKIPPED"), stages)

        assert len(line["reason"]) == 4096


def test_paths_are_written_with_forward_slashes(monkeypatch):
    monkeypatch.setattr(os, "sep", "\\")

    assert loop_log.portable_path(".soup-loops\\adapters\\x") == ".soup-loops/adapters/x"
    assert loop_log.portable_path("") is None and loop_log.portable_path(None) is None


# ---------------------------------------------------------------------------
# The same iteration, later: promoted or withdrawn
# ---------------------------------------------------------------------------


def _canary_line(cwd) -> dict:
    return _line(_record(), _stages(cwd))


def _quality(canary_rated=31, stable_rated=40) -> dict:
    return {
        "verdict": "OK", "rated": {"stable": stable_rated, "canary": canary_rated},
        "negative": {"stable": 6, "canary": 4}, "share": {"stable": 0.15, "canary": 0.129},
        "difference": -0.021, "z": -0.2, "p_value": 0.58, "min_rated": 30, "threshold": 0.05,
        "timed_out": False, "decided_on": "errors_and_quality",
    }


class TestResolvedLine:
    def test_a_promotion_is_the_snapshot_with_its_new_outcome(self, cwd):
        snapshot = _canary_line(cwd)
        canary = loop_log.canary_result(
            traffic_pct=20.0, verdict="OK", stable_samples=240, canary_samples=60,
            quality=_quality(),
        )

        line = loop_log.resolved_line(
            snapshot, outcome="promoted", reason="canary X promoted to production",
            finished_at="2026-11-05T18:00:00+00:00", to_version=2, canary=canary,
        )

        assert _errors("iteration", line) == [], line
        assert line["iteration_id"] == ITER and line["outcome"] == "promoted"
        assert line["finished_at"] == "2026-11-05T18:00:00+00:00" and line["to_version"] == 2
        assert line["canary"] == {
            "traffic_pct": 20.0, "verdict": "OK", "samples": {"stable": 240, "canary": 60},
            "quality": _quality(),
        }
        for kept in ("dataset", "train", "verdict", "artifacts", "data", "started_at",
                     "parent_iteration_id", "from_version", "estimated_cost_usd"):
            assert line[kept] == snapshot[kept], kept
        assert snapshot["outcome"] == "canary" and snapshot["finished_at"] is None

    def test_a_withdrawal_produces_no_version(self, cwd):
        line = loop_log.resolved_line(
            _canary_line(cwd), outcome="rolled_back", reason="canary regression",
            finished_at="2026-11-05T18:00:00+00:00", to_version=None, canary=None,
        )

        assert _errors("iteration", line) == [], line
        assert line["outcome"] == "rolled_back" and line["to_version"] is None
        assert line["canary"]["traffic_pct"] == 20.0  # the start of the canary is kept

    def test_a_canary_result_leaves_out_what_was_not_measured(self):
        canary = loop_log.canary_result(
            traffic_pct=5.0, verdict=None, stable_samples=None, canary_samples=None, quality=None
        )

        assert canary == {"traffic_pct": 5.0}
        assert _errors("iteration", {
            "schema_version": 1, "iteration_id": ITER, "started_at": "2026-11-05T12:00:00Z",
            "outcome": "canary", "canary": canary,
        }) == []

    def test_the_quality_evidence_is_copied_not_shared(self):
        quality = _quality()

        canary = loop_log.canary_result(
            traffic_pct=5.0, verdict="OK", stable_samples=1, canary_samples=1, quality=quality
        )
        canary["quality"]["rated"]["canary"] = 999

        assert quality["rated"]["canary"] == 31


# ---------------------------------------------------------------------------
# The version counter
# ---------------------------------------------------------------------------


class TestVersionChanges:
    def test_a_promotion_takes_the_next_number_after_the_highest(self):
        state = _state(
            canary_active="c", canary_iteration=_snapshot(), production_version=1,
            latest_version=3, production_iteration_id="iter-seed",
        )

        changes = loop_log.promotion_changes(state)

        assert changes == {
            "production_version": 4, "previous_production_version": 1, "latest_version": 4,
            "production_iteration_id": "iter-20261105T120000-1b2c3d4e",
            "previous_production_iteration_id": "iter-seed", "canary_iteration": None,
        }
        assert replace(state, **changes).latest_version == 4

    def test_a_canary_no_iteration_started_is_promoted_without_an_iteration_id(self):
        state = _state(canary_active="c", production_iteration_id="iter-seed")

        changes = loop_log.promotion_changes(state)

        assert changes["production_iteration_id"] is None
        assert changes["production_version"] == 2

    def test_a_rollback_returns_to_the_remembered_version_and_its_iteration(self):
        state = _state(
            production_name="b", previous_production_name="a", production_version=3,
            previous_production_version=2, latest_version=3, production_iteration_id="iter-b",
            previous_production_iteration_id="iter-a",
        )

        changes = loop_log.rollback_changes(state)

        assert changes == {
            "production_version": 2, "previous_production_version": None,
            "production_iteration_id": "iter-a", "previous_production_iteration_id": None,
            "canary_iteration": None,
        }
        assert replace(state, **changes).production_version == 2

    def test_back_to_the_served_model_is_version_one(self):
        state = _state(
            production_name="a", production_version=2, previous_production_version=1,
            latest_version=2, production_iteration_id="iter-a",
            previous_production_iteration_id="iter-seed",
        )

        changes = loop_log.rollback_changes(state)

        assert changes["production_version"] == 1
        assert changes["production_iteration_id"] == "iter-seed"

    def test_a_second_rollback_goes_to_the_served_model_as_the_command_does(self):
        # After one rollback previous_production_name is None: the next goes to the served model.
        state = _state(
            production_name="a", production_version=2, previous_production_version=None,
            latest_version=3,
        )

        assert loop_log.rollback_changes(state)["production_version"] == 1

    def test_the_number_is_never_reused_after_a_rollback(self):
        state = _state(production_version=2, latest_version=2, production_name="a",
                       previous_production_version=1)
        back = replace(state, **loop_log.rollback_changes(state))
        assert (back.production_version, back.latest_version) == (1, 2)

        forward = replace(
            back, canary_active="b", **loop_log.promotion_changes(replace(back, canary_active="b"))
        )

        assert forward.production_version == 3 and forward.latest_version == 3


# ---------------------------------------------------------------------------
# The first version
# ---------------------------------------------------------------------------


class TestFirstVersion:
    def test_the_seed_is_an_iteration_and_an_initial_version_that_agree(self):
        state = _state(production_adapter="./adapters/mine", status="stopped")

        iteration, version = loop_log.seed_lines(state, actor="user")

        assert _errors("iteration", iteration) == [] and _errors("version", version) == []
        assert iteration["parent_iteration_id"] is None and iteration["from_version"] is None
        assert iteration["to_version"] == 1 and iteration["outcome"] == "promoted"
        assert iteration["reason"] == "initial deploy"
        assert iteration["finished_at"] == iteration["started_at"]
        assert iteration["artifacts"] == {
            "adapter": "./adapters/mine", "merged": None, "gguf": None, "served_as": None
        }
        assert (version["from_version"], version["to_version"]) == (None, 1)
        assert (version["cause"], version["actor"]) == ("initial", "user")
        assert version["iteration_id"] == iteration["iteration_id"]
        assert re.fullmatch(r"iter-\d{8}T\d{6}-[0-9a-f]{8}", iteration["iteration_id"])
        assert version["source"] == "soup-loop"

    def test_starting_the_log_writes_both_lines_and_marks_the_state(self, cwd):
        log = LoopLog()
        state = _state()

        started = loop_log.start_log(state, log, actor="user")

        assert started.version_log_started is True
        (iteration,) = _lines(cwd / ".soup-loops" / "iterations.jsonl")
        (version,) = _lines(cwd / ".soup-loops" / "versions.jsonl")
        assert started.production_iteration_id == iteration["iteration_id"]
        assert version["iteration_id"] == iteration["iteration_id"]
        assert version["cause"] == "initial"

    def test_a_started_log_is_not_started_again(self, cwd):
        log = LoopLog()
        started = loop_log.start_log(_state(), log, actor="user")

        again = loop_log.start_log(started, log, actor="loop")

        assert again is started
        assert len(_lines(cwd / ".soup-loops" / "versions.jsonl")) == 1

    def test_a_failed_write_leaves_the_state_to_try_again(self, cwd):
        os.makedirs(os.path.join(".soup-loops", "versions.jsonl"))
        log = LoopLog()

        started = loop_log.start_log(_state(), log, actor="loop")

        assert started.version_log_started is False and started.production_iteration_id is None
        (note,) = log.take_failures()
        assert note.startswith("versions.jsonl not written (")

    def test_a_loop_that_already_promoted_starts_its_log_at_its_current_version(self, cwd):
        state = _state(production_name="loop-train-aa", production_version=2,
                       previous_production_version=1, latest_version=2)

        iteration, version = loop_log.seed_lines(state, actor="loop")

        assert iteration["to_version"] == 2 and version["to_version"] == 2
        assert version["served_as"] == "loop-train-aa" and version["actor"] == "loop"
        assert _errors("version", version) == []


# ---------------------------------------------------------------------------
# Stage events: nothing is written for an iteration that did no work
# ---------------------------------------------------------------------------


class TestStageEvents:
    def test_events_are_held_until_there_is_work(self, cwd):
        log = LoopLog()
        events = loop_log.StageEvents(log, ITER)

        events.emit("harvest", "start")
        events.emit("harvest", "end", message="0 rows")

        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()

    def test_going_live_writes_what_was_held_in_order_then_everything_after(self, cwd):
        log = LoopLog()
        events = loop_log.StageEvents(log, ITER)
        events.emit("harvest", "start")
        events.emit("harvest", "end", message="3 rows")

        events.go_live()
        events.emit("train", "start")

        lines = _lines(cwd / ".soup-loops" / "progress.jsonl")
        assert [(line["stage"], line["event"]) for line in lines] == [
            ("harvest", "start"), ("harvest", "end"), ("train", "start")
        ]
        assert {line["iteration_id"] for line in lines} == {ITER}
        for line in lines:
            assert _errors("progress", line) == [], line

    def test_going_live_twice_writes_nothing_twice(self, cwd):
        log = LoopLog()
        events = loop_log.StageEvents(log, ITER)
        events.emit("harvest", "start")

        events.go_live()
        events.go_live()

        assert len(_lines(cwd / ".soup-loops" / "progress.jsonl")) == 1

    def test_the_time_of_a_held_event_is_when_it_happened(self, cwd, monkeypatch):
        log = LoopLog()
        stamps = iter(["2026-11-05T12:00:00+00:00", "2026-11-05T12:09:00+00:00"])
        monkeypatch.setattr(loop_log, "utc_now_iso", lambda: next(stamps))
        events = loop_log.StageEvents(log, ITER)
        events.emit("harvest", "start")
        events.emit("harvest", "end")

        events.go_live()

        assert [line["ts"] for line in _lines(cwd / ".soup-loops" / "progress.jsonl")] == [
            "2026-11-05T12:00:00+00:00", "2026-11-05T12:09:00+00:00"
        ]


def _write_raw(cwd, **fields) -> LoopState:
    """A loop.yaml as an older soup wrote it: none of the new fields."""
    body = {"served_model": "./base", "eval_suite": "t", "baseline": "b", "status": "stopped"}
    body.update(fields)
    os.makedirs(".soup", exist_ok=True)
    (cwd / ".soup" / "loop.yaml").write_text(json.dumps(body), "utf-8")
    return read_state(os.path.join(".soup", "loop.yaml"))


class TestOldLoopYaml:
    def test_a_loop_that_promoted_nothing_starts_at_version_one(self, cwd):
        state = _write_raw(cwd)

        assert (state.production_version, state.latest_version) == (1, 1)
        assert state.previous_production_version is None
        assert state.version_log_started is False

    def test_a_promoted_adapter_is_at_least_version_two(self, cwd):
        state = _write_raw(cwd, production_name="loop-train-aa", production_adapter="a")

        assert (state.production_version, state.latest_version) == (2, 2)
        assert state.previous_production_version == 1

    def test_a_remembered_previous_adapter_means_one_more_promotion(self, cwd):
        state = _write_raw(
            cwd, production_name="loop-train-bb", previous_production_name="loop-train-aa"
        )

        assert (state.production_version, state.latest_version) == (3, 3)
        assert state.previous_production_version == 2

    def test_a_file_that_has_the_counter_is_read_as_written(self, cwd):
        state = _write_raw(
            cwd, production_name="x", production_version=7, latest_version=9,
            previous_production_version=5,
        )

        assert (state.production_version, state.latest_version) == (7, 9)
        assert state.previous_production_version == 5


# ---------------------------------------------------------------------------
# The daemon: run_once appends the lines of a cycle (when it is given a log)
# ---------------------------------------------------------------------------

from soup_cli.utils.canary_router import (  # noqa: E402
    CanaryPolicy,
    QualityStats,
    record_bucket_outcome,
)
from soup_cli.utils.loop_daemon import WatchConfig, run_once, watch  # noqa: E402


def _read(cwd, kind) -> list:
    """A log of the project, every line checked against its schema."""
    name = loop_log.LOG_FILES[kind]
    path = cwd / ".soup-loops" / name
    if not path.exists():
        return []
    records = _lines(path)
    for record in records:
        assert _errors(kind, record) == [], record
    return records


def _last_wins(records) -> dict:
    """The latest iteration line per iteration_id, as a reader applies the contract."""
    latest = {}
    for record in records:
        latest[record["iteration_id"]] = record
    return latest


class Pipeline:
    """Stub stages that train and start a canary named after the cycle, like the pre-wired ones."""

    def __init__(self, cwd, *, traffic=5.0) -> None:
        self.cwd = cwd
        self.traffic = traffic
        self.cycles = 0
        self.train_runs = True
        self.promote_ok = True
        self.verdict = "OK"
        self.quality = None
        self.stats_path = str(cwd / "stats.json")

    def adapter(self) -> str:
        return f"loop-train-{self.cycles:08x}"

    def harvest(self, state):
        self.cycles += 1
        return {
            "pairs_harvested": 3, "new_pairs": 3, "cursor_ts": 1_793_880_000.0 + self.cycles,
            "pairs_path": ".soup-loops/pairs/pairs-x.jsonl", "traces_collected": 5,
            "dpo_pairs": 3, "kto_rows": 0,
        }

    def train(self, state, ctx):
        if not self.train_runs:
            return {"run_id": None, "skipped": True, "adapter_path": None, "failed": True}
        return {"run_id": self.adapter(), "skipped": False,
                "adapter_path": f".soup-loops/adapters/{self.adapter()}"}

    def gate(self, state, ctx):
        if ctx.get("skipped"):
            return {"gate_verdict": "SKIPPED"}
        return {"gate_verdict": self.verdict, "verdict_method": "eval_gate"}

    def deploy(self, state, ctx):
        if ctx.get("gate_verdict") != "OK":
            return {"deployed": False, "canary_verdict": None}
        return {
            "deployed": True, "canary_verdict": None, "canary_active": self.adapter(),
            "canary_adapter": f".soup-loops/adapters/{self.adapter()}",
            "canary_traffic_pct": self.traffic,
            "notes": f"canary {self.adapter()}: {self.traffic:g}% of keyed traffic",
        }

    def promote(self, state):
        return {"promoted": self.promote_ok, "notes": "" if self.promote_ok else "server said no"}

    def config(self, **kwargs) -> WatchConfig:
        return WatchConfig(
            poll_interval_sec=1.0, max_iterations=1, canary_stats_path=self.stats_path,
            harvest_fn=self.harvest, train_fn=self.train, gate_fn=self.gate,
            deploy_fn=self.deploy, promote_fn=self.promote, cost_fn=lambda state: 0.0,
            quality_fn=(lambda state: self.quality) if self.quality is not None else None,
            **kwargs,
        )

    def rate(self, state, *, ok=30, major=0, stable=None) -> None:
        """Answer requests of the running canary: ``ok`` fine, ``major`` failed, ``stable``
        answered by production (default: as many as the canary got)."""
        policy = CanaryPolicy(
            stable=state.stable_name(), canary=state.canary_active, traffic_pct=5.0
        )
        for count, good in ((ok, True), (major, False)):
            for _ in range(count):
                record_bucket_outcome(
                    policy, "canary", good, rollout_id=state.canary_rollout_id,
                    path=self.stats_path,
                )
        for _ in range(ok + major if stable is None else stable):
            record_bucket_outcome(
                policy, "stable", True, rollout_id=state.canary_rollout_id, path=self.stats_path
            )


def _running(**kwargs) -> LoopState:
    return _state(status="running", base_model="org/base", **kwargs)


def _cycle(state, pipeline, log=None, **kwargs):
    log = log or LoopLog()
    new_state, record, decision = run_once(state, pipeline.config(**kwargs), log=log)
    return new_state, record, decision


class TestIdlePoll:
    def test_an_iteration_with_nothing_to_train_is_one_skipped_line_and_no_progress(self, cwd):
        state = _running()

        new_state, record, decision = run_once(
            state, WatchConfig(poll_interval_sec=1.0, max_iterations=1), log=LoopLog()
        )

        assert decision.proceed is True
        iterations = _read(cwd, "iteration")
        seed, idle = iterations
        assert seed["outcome"] == "promoted" and seed["reason"] == "initial deploy"
        assert idle["iteration_id"] == record.iteration_id
        assert idle["outcome"] == "skipped" and idle["to_version"] is None
        assert idle["parent_iteration_id"] == seed["iteration_id"]
        assert (idle["from_version"],) == (1,)
        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()
        (initial,) = _read(cwd, "version")
        assert initial["cause"] == "initial" and initial["actor"] == "loop"
        assert new_state.version_log_started is True
        assert new_state.production_iteration_id == seed["iteration_id"]

    def test_without_a_log_run_once_writes_nothing_and_changes_no_version(self, cwd):
        state = _running()

        new_state, _, _ = run_once(state, WatchConfig(poll_interval_sec=1.0, max_iterations=1))

        assert not (cwd / ".soup-loops").exists()
        assert new_state.version_log_started is False
        assert new_state.production_iteration_id is None

    def test_the_first_version_is_written_once(self, cwd):
        pipeline = Pipeline(cwd)
        pipeline.train_runs = False
        state = _running()

        state, _, _ = _cycle(state, pipeline)
        state, _, _ = _cycle(state, pipeline)

        assert len(_read(cwd, "version")) == 1
        assert len([r for r in _read(cwd, "iteration") if r["reason"] == "initial deploy"]) == 1


class TestCanaryStarts:
    def test_the_stages_report_their_progress_in_order(self, cwd):
        pipeline = Pipeline(cwd, traffic=20.0)

        state, record, _ = _cycle(_running(), pipeline)

        progress = _read(cwd, "progress")
        assert [(line["stage"], line["event"]) for line in progress] == [
            ("harvest", "start"), ("harvest", "end"), ("train", "start"), ("train", "end"),
            ("verdict", "start"), ("verdict", "end"), ("deploy", "start"), ("deploy", "end"),
            ("canary", "start"),
        ]
        assert {line["iteration_id"] for line in progress} == {record.iteration_id}
        canary = progress[-1]
        assert (canary["done"], canary["total"]) == (0, 30)
        assert "20% of keyed traffic" in canary["message"]

    def test_the_iteration_line_is_a_canary_and_is_kept_with_the_running_canary(self, cwd):
        pipeline = Pipeline(cwd)

        state, record, _ = _cycle(_running(), pipeline)

        seed, line = _read(cwd, "iteration")
        assert line["outcome"] == "canary" and line["finished_at"] is None
        assert (line["from_version"], line["to_version"]) == (1, 2)
        assert line["parent_iteration_id"] == seed["iteration_id"]
        assert line["train"]["run_id"] == record.run_id == pipeline.adapter()
        assert line["artifacts"]["served_as"] == pipeline.adapter()
        assert state.canary_iteration == line
        assert state.canary_active == pipeline.adapter()
        # a canary is not a version: production and the counter do not move yet
        assert (state.production_version, state.latest_version) == (1, 1)
        assert len(_read(cwd, "version")) == 1

    def test_the_manifest_and_the_line_say_the_same(self, cwd):
        from soup_cli.utils.loop_iteration import read_iteration, write_iteration

        pipeline = Pipeline(cwd)
        _, record, _ = _cycle(_running(), pipeline)
        write_iteration(record)

        manifest = read_iteration(record.iteration_id)
        line = _last_wins(_read(cwd, "iteration"))[record.iteration_id]
        assert line["iteration_id"] == manifest.iteration_id
        assert line["started_at"] == manifest.started_at
        assert line["train"]["run_id"] == manifest.run_id
        assert line["estimated_cost_usd"] == manifest.estimated_cost_usd
        assert (line["outcome"] == "canary") == manifest.shipped
        assert line["verdict"]["decision"] == "SHIP"
        assert manifest.gate_verdict == "OK"

    def test_a_cycle_whose_training_died_is_a_failed_iteration(self, cwd):
        pipeline = Pipeline(cwd)
        pipeline.train_runs = False

        state, _, _ = _cycle(_running(), pipeline)

        line = _read(cwd, "iteration")[-1]
        assert line["outcome"] == "failed" and line["to_version"] is None
        assert state.canary_active is None and state.canary_iteration is None

    def test_dont_ship_is_skipped_with_its_verdict(self, cwd):
        pipeline = Pipeline(cwd)
        pipeline.verdict = "MAJOR"

        state, _, _ = _cycle(_running(), pipeline)

        line = _read(cwd, "iteration")[-1]
        assert line["outcome"] == "skipped"
        assert line["verdict"]["decision"] == "DON'T SHIP"
        assert state.canary_active is None
        stages = [(p["stage"], p["event"]) for p in _read(cwd, "progress")]
        assert ("deploy", "start") not in stages  # nothing is deployed after DON'T SHIP


class TestCanaryWaits:
    def _waiting(self, cwd, **kwargs):
        pipeline = Pipeline(cwd)
        state, _, _ = _cycle(_running(), pipeline)
        before = len(_read(cwd, "iteration"))
        return pipeline, state, before

    def test_a_waiting_cycle_writes_one_canary_progress_line_and_no_iteration(self, cwd):
        pipeline, state, before = self._waiting(cwd)
        pipeline.rate(state, ok=12)
        progress_before = len(_read(cwd, "progress"))

        state, record, decision = _cycle(state, pipeline)

        assert decision.proceed is False
        assert len(_read(cwd, "iteration")) == before  # nothing is recorded
        new = _read(cwd, "progress")[progress_before:]
        assert len(new) == 1
        line = new[0]
        assert (line["stage"], line["event"]) == ("canary", "progress")
        assert (line["done"], line["total"]) == (12, 30)
        assert line["iteration_id"] == state.canary_iteration["iteration_id"]

    def test_the_numbers_are_the_rated_answers_when_people_rate_them(self, cwd):
        pipeline, state, _ = self._waiting(cwd)
        pipeline.quality = QualityStats(
            stable_rated=40, stable_negative=2, canary_rated=7, canary_negative=1
        )
        progress_before = len(_read(cwd, "progress"))

        _cycle(state, pipeline)

        (line,) = _read(cwd, "progress")[progress_before:]
        assert (line["done"], line["total"]) == (7, 30)
        assert "7/30" in line["message"] and "40/30" in line["message"]

    def test_there_is_one_canary_line_per_cycle(self, cwd):
        pipeline, state, _ = self._waiting(cwd)
        for _ in range(3):
            state, _, _ = _cycle(state, pipeline)

        canary_lines = [
            p for p in _read(cwd, "progress")
            if p["stage"] == "canary" and p["event"] == "progress"
        ]
        assert len(canary_lines) == 3

    def test_a_canary_whose_statistics_cannot_be_read_is_reported_without_numbers(self, cwd):
        pipeline, state, _ = self._waiting(cwd)
        (cwd / "stats.json").write_text("not json", "utf-8")
        progress_before = len(_read(cwd, "progress"))

        _cycle(state, pipeline)

        (line,) = _read(cwd, "progress")[progress_before:]
        assert line["done"] is None and line["total"] is None


class TestCanaryResolved:
    def test_a_passing_canary_is_promoted_and_versioned(self, cwd):
        pipeline = Pipeline(cwd)
        state, first, _ = _cycle(_running(), pipeline)
        name = state.canary_active
        pipeline.rate(state, ok=40, stable=55)
        pipeline.train_runs = False  # the cycle that promotes trains nothing new

        state, second, _ = _cycle(state, pipeline)

        iterations = _read(cwd, "iteration")
        latest = _last_wins(iterations)
        assert latest[first.iteration_id]["outcome"] == "promoted"
        assert latest[first.iteration_id]["to_version"] == 2
        assert latest[first.iteration_id]["finished_at"] is not None
        assert latest[first.iteration_id]["canary"]["verdict"] == "OK"
        assert latest[first.iteration_id]["canary"]["samples"] == {"stable": 55, "canary": 40}
        versions = _read(cwd, "version")
        promote = versions[-1]
        assert (promote["cause"], promote["actor"]) == ("promote", "loop")
        assert (promote["from_version"], promote["to_version"]) == (1, 2)
        assert promote["iteration_id"] == first.iteration_id
        assert promote["served_as"] == name
        assert (state.production_version, state.latest_version) == (2, 2)
        assert state.production_iteration_id == first.iteration_id
        assert state.canary_iteration is None and state.canary_active is None
        # the iteration of this cycle follows the promotion and starts from the new version
        assert latest[second.iteration_id]["from_version"] == 2
        assert latest[second.iteration_id]["parent_iteration_id"] == first.iteration_id
        # the order in the file: the promotion first
        order = [line["iteration_id"] for line in iterations]
        assert order.index(first.iteration_id) < order.index(second.iteration_id)
        assert [i["outcome"] for i in iterations if i["iteration_id"] == first.iteration_id] == [
            "canary", "promoted"
        ]

    def test_the_promoted_line_carries_the_quality_evidence_of_the_record(self, cwd):
        pipeline = Pipeline(cwd)
        pipeline.quality = QualityStats(
            stable_rated=40, stable_negative=4, canary_rated=35, canary_negative=3
        )
        state, first, _ = _cycle(_running(), pipeline)
        pipeline.rate(state, ok=40)
        pipeline.train_runs = False

        state, second, _ = _cycle(state, pipeline)

        promoted = _last_wins(_read(cwd, "iteration"))[first.iteration_id]
        assert second.canary_quality is not None
        assert promoted["canary"]["quality"] == second.canary_quality
        assert promoted["canary"]["quality"]["rated"] == {"stable": 40, "canary": 35}

    def test_a_regressing_canary_is_rolled_back_and_the_version_does_not_change(self, cwd):
        pipeline = Pipeline(cwd)
        state, first, _ = _cycle(_running(), pipeline)
        pipeline.rate(state, ok=10, major=30)
        pipeline.train_runs = False

        state, _, _ = _cycle(state, pipeline)

        latest = _last_wins(_read(cwd, "iteration"))[first.iteration_id]
        assert latest["outcome"] == "rolled_back" and latest["to_version"] is None
        assert latest["canary"]["verdict"] == "MAJOR"
        assert "rolled back" in latest["reason"]
        assert [v["cause"] for v in _read(cwd, "version")] == ["initial"]
        assert (state.production_version, state.latest_version) == (1, 1)
        assert state.canary_iteration is None
        stages = [(p["stage"], p["event"]) for p in _read(cwd, "progress")]
        assert ("rollback", "end") in stages

    def test_a_refused_promotion_changes_nothing_but_says_so(self, cwd):
        pipeline = Pipeline(cwd)
        state, first, _ = _cycle(_running(), pipeline)
        pipeline.rate(state, ok=40)
        pipeline.promote_ok = False
        pipeline.train_runs = False
        iterations_before = len(_read(cwd, "iteration"))

        state, _, _ = _cycle(state, pipeline)

        assert state.canary_active is not None and state.canary_iteration is not None
        assert state.production_version == 1
        assert [v["cause"] for v in _read(cwd, "version")] == ["initial"]
        outcomes = [
            line["outcome"] for line in _read(cwd, "iteration")[iterations_before:]
            if line["iteration_id"] == first.iteration_id
        ]
        assert outcomes == []
        errors = [p for p in _read(cwd, "progress") if p["stage"] == "promote"]
        assert [(p["event"], p.get("error")) for p in errors][-2:] == [
            ("start", None), ("error", "promote_refused")
        ]

    def test_a_promotion_then_a_rollback_then_a_promotion_never_reuses_a_number(self, cwd):
        from dataclasses import replace as _replace

        pipeline = Pipeline(cwd)
        state, _, _ = _cycle(_running(), pipeline)
        pipeline.rate(state, ok=40)
        state, _, _ = _cycle(state, pipeline)  # promoted v2; this cycle also starts a canary
        assert state.production_version == 2 and state.canary_active
        changes = loop_log.rollback_changes(_replace(state, canary_active=None))
        state = _replace(state, canary_active=None, canary_rollout_id=None, **changes)
        assert state.production_version == 1 and state.latest_version == 2

        pipeline.train_runs = True
        state, _, _ = _cycle(state, pipeline)
        pipeline.rate(state, ok=40)
        state, _, _ = _cycle(state, pipeline)

        assert state.production_version == 3 and state.latest_version == 3
        promotes = [v for v in _read(cwd, "version") if v["cause"] == "promote"]
        assert [(v["from_version"], v["to_version"]) for v in promotes][-1] == (1, 3)

    def test_a_canary_started_by_hand_is_promoted_without_an_iteration(self, cwd):
        pipeline = Pipeline(cwd)
        pipeline.train_runs = False
        state = _running(
            canary_active="hand-made", canary_traffic_pct=5.0, canary_rollout_id="r1",
            version_log_started=True, production_iteration_id="iter-seed",
        )
        pipeline.rate(state, ok=40)

        state, _, _ = _cycle(state, pipeline)

        promote = _read(cwd, "version")[-1]
        assert promote["cause"] == "promote" and promote["iteration_id"] is None
        assert (promote["from_version"], promote["to_version"]) == (1, 2)
        assert state.production_version == 2 and state.production_iteration_id is None
        assert all(line["iteration_id"] != "hand-made" for line in _read(cwd, "iteration"))


class TestLogFailures:
    def test_a_log_that_cannot_be_written_is_a_note_and_the_cycle_goes_on(self, cwd, caplog):
        os.makedirs(os.path.join(".soup-loops", "iterations.jsonl"))
        pipeline = Pipeline(cwd)

        with caplog.at_level(logging.WARNING):
            state, record, decision = _cycle(_running(), pipeline)

        assert decision.proceed is True
        assert state.canary_active == pipeline.adapter()  # the loop did what it always does
        assert "iterations.jsonl not written (" in record.notes
        assert any("iterations.jsonl" in message for message in caplog.messages)
        assert _read(cwd, "version") == []  # the seed failed with it: tried again next time
        assert state.version_log_started is False
        # the other logs still work
        assert [p["stage"] for p in _read(cwd, "progress")][0] == "harvest"

    def test_the_note_is_added_once_per_failure_kind(self, cwd):
        os.makedirs(os.path.join(".soup-loops", "progress.jsonl"))
        pipeline = Pipeline(cwd)

        _, record, _ = _cycle(_running(), pipeline)

        assert record.notes.count("progress.jsonl not written (") == 1

    def test_the_lines_are_written_before_the_state_is_saved(self, cwd, monkeypatch):
        import soup_cli.utils.loop_daemon as daemon

        pipeline = Pipeline(cwd)
        pipeline.train_runs = False
        state = _running()
        os.makedirs(".soup")
        write_state(state)
        seen = {}

        real_write_state = daemon.write_state

        def checking(new_state, path=None):
            seen.setdefault("lines", len(_read(cwd, "iteration")))  # at the first save
            return real_write_state(new_state, path)

        monkeypatch.setattr(daemon, "write_state", checking)

        watch(pipeline.config())

        assert seen["lines"] == 2  # the seed and the iteration, before the first save


class TestWatch:
    def test_watch_writes_the_logs_and_the_manifest(self, cwd):
        from soup_cli.utils.loop_iteration import list_iterations

        pipeline = Pipeline(cwd)
        os.makedirs(".soup")
        write_state(_running())

        final, ran = watch(pipeline.config())

        assert ran == 1
        (manifest_id,) = list_iterations()
        assert _last_wins(_read(cwd, "iteration"))[manifest_id]["outcome"] == "canary"
        saved = read_state()
        assert saved.canary_iteration["iteration_id"] == manifest_id
        assert saved.version_log_started is True

    def test_logs_can_be_turned_off(self, cwd):
        pipeline = Pipeline(cwd)
        os.makedirs(".soup")
        write_state(_running())

        watch(pipeline.config(loop_logs=False))

        assert not list((cwd / ".soup-loops").glob("*.jsonl"))
        assert read_state().version_log_started is False

    def test_the_switch_must_be_a_bool(self):
        with pytest.raises(ValueError, match="loop_logs must be bool"):
            WatchConfig(loop_logs=1)

    def test_logs_follow_a_custom_iteration_directory(self, cwd):
        pipeline = Pipeline(cwd)
        os.makedirs(".soup")
        write_state(_running())

        watch(pipeline.config(iteration_dir="iters"))

        assert (cwd / "iters" / "iterations.jsonl").is_file()
        assert (cwd / "iters" / "versions.jsonl").is_file()
        assert not (cwd / ".soup-loops").exists()


# ---------------------------------------------------------------------------
# The stages say when they failed (a skip is not a failure)
# ---------------------------------------------------------------------------

import soup_cli.utils.loop_stages as stages_module  # noqa: E402


class TestStageFailureFlags:
    def _train(self, cwd, monkeypatch, runner, **ctx):
        (cwd / "pairs.jsonl").write_text(
            json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", "utf-8"
        )
        monkeypatch.setattr(stages_module, "_TRAIN_RUNNER", runner)
        return stages_module.train_dpo_from_pairs(
            _state(base_model="org/base"),
            {"pairs_path": "pairs.jsonl", "pairs_harvested": 1, **ctx},
        )

    def test_a_training_run_that_exits_nonzero_failed(self, cwd, monkeypatch):
        class Died:
            returncode = 1

        out = self._train(cwd, monkeypatch, lambda argv, **kw: Died())

        assert out["failed"] is True and out["skipped"] is True and out["run_id"] is None

    def test_a_training_run_that_could_not_start_failed(self, cwd, monkeypatch):
        def broken(argv, **kwargs):
            raise OSError("no such interpreter")

        out = self._train(cwd, monkeypatch, broken)

        assert out["failed"] is True and out["run_id"] is None

    def test_a_training_run_that_finished_did_not_fail(self, cwd, monkeypatch):
        class Fine:
            returncode = 0

        out = self._train(cwd, monkeypatch, lambda argv, **kw: Fine())

        assert "failed" not in out and out["skipped"] is False

    def test_nothing_to_train_is_a_skip_not_a_failure(self, cwd, monkeypatch):
        out = self._train(cwd, monkeypatch, lambda argv, **kw: None, pairs_harvested=0)

        assert out["skipped"] is True and "failed" not in out

    def _deploy(self, monkeypatch, status, detail=""):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        monkeypatch.setattr(
            stages_module, "_LOAD_POSTER", lambda endpoint, name, path: (status, detail)
        )
        return stages_module.deploy_to_canary(
            _state(), {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/loop-train-aa"}
        )

    def test_a_deploy_the_server_refused_failed(self, cwd, monkeypatch):
        out = self._deploy(monkeypatch, 500, "boom")

        assert out["deployed"] is False and out["failed"] is True and out["notes"]

    def test_a_deploy_that_started_a_canary_did_not_fail(self, cwd, monkeypatch):
        out = self._deploy(monkeypatch, 200)

        assert out["deployed"] is True and "failed" not in out

    def test_a_deploy_without_an_endpoint_was_not_attempted(self, cwd, monkeypatch):
        monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)

        out = stages_module.deploy_to_canary(
            _state(), {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/x"}
        )

        assert out["deployed"] is False and "failed" not in out

    def test_an_endpoint_the_guard_rejects_failed(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://example.com:8000")

        out = stages_module.deploy_to_canary(
            _state(), {"gate_verdict": "OK", "adapter_path": ".soup-loops/adapters/x"}
        )

        assert out["deployed"] is False and out["failed"] is True


# ---------------------------------------------------------------------------
# The commands: init writes the first version, canary and rollback write theirs
# ---------------------------------------------------------------------------

from typer.testing import CliRunner  # noqa: E402

from soup_cli.commands.loop import app as loop_app  # noqa: E402


def _cli(*args):
    result = CliRunner().invoke(loop_app, list(args))
    return result, _clean(result.output)


def _init(cwd, *flags):
    (cwd / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return _cli(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base",
        *flags,
    )


@pytest.fixture
def no_server(monkeypatch):
    """No serve endpoint: a rollback then changes loop.yaml only (the server is not told)."""
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)


class TestInit:
    def test_a_pre_wired_init_writes_the_first_version(self, cwd):
        result, out = _init(cwd, "--pre-wired")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "warning" not in out
        (seed,) = _read(cwd, "iteration")
        (initial,) = _read(cwd, "version")
        assert (seed["to_version"], seed["outcome"]) == (1, "promoted")
        assert (initial["cause"], initial["actor"], initial["to_version"]) == ("initial", "user", 1)
        assert initial["iteration_id"] == seed["iteration_id"]
        state = read_state()
        assert state.version_log_started is True
        assert state.production_iteration_id == seed["iteration_id"]

    def test_an_init_without_the_pre_wired_stages_leaves_it_to_the_first_watch_cycle(self, cwd):
        result, _ = _init(cwd)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not (cwd / ".soup-loops").exists()
        assert read_state().version_log_started is False

    def test_a_log_that_cannot_be_written_is_a_warning_and_the_loop_is_still_initialised(
        self, cwd
    ):
        os.makedirs(os.path.join(".soup-loops", "versions.jsonl"))

        result, out = _init(cwd, "--pre-wired")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "warning: versions.jsonl not written (" in out
        assert "soup loop watch tries again" in out
        assert read_state().version_log_started is False

    def test_a_forced_init_starts_the_version_log_again_without_touching_earlier_lines(
        self, cwd
    ):
        _init(cwd, "--pre-wired")
        before = (cwd / ".soup-loops" / "versions.jsonl").read_bytes()

        result, _ = _init(cwd, "--pre-wired", "--force")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        raw = (cwd / ".soup-loops" / "versions.jsonl").read_bytes()
        assert raw.startswith(before) and len(_read(cwd, "version")) == 2


def _canary_state(**kwargs) -> LoopState:
    return _state(
        status="running", base_model="org/base", canary_active="loop-train-aa",
        canary_traffic_pct=5.0, canary_rollout_id="r1", canary_started_ts=1_793_880_000.0,
        canary_adapter=".soup-loops/adapters/loop-train-aa", version_log_started=True,
        production_iteration_id="iter-20261105T090000-0a1b2c3d",
        canary_iteration={
            "schema_version": 1, "iteration_id": ITER, "outcome": "canary",
            "parent_iteration_id": "iter-20261105T090000-0a1b2c3d", "from_version": 1,
            "to_version": 2, "started_at": "2026-11-05T12:00:00+00:00", "finished_at": None,
            "canary": {"traffic_pct": 5.0, "verdict": "UNKNOWN"},
        },
        **kwargs,
    )


def _save(state: LoopState) -> None:
    os.makedirs(".soup", exist_ok=True)
    write_state(state)


class TestCanaryCommand:
    def test_a_canary_started_by_hand_writes_one_progress_line_and_no_version(self, cwd):
        _save(_state(status="running", version_log_started=True))

        result, out = _cli("canary", "hand-made", "--traffic", "10%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _read(cwd, "iteration") == [] and _read(cwd, "version") == []
        (line,) = _read(cwd, "progress")
        assert (line["stage"], line["event"], line["iteration_id"]) == ("canary", "start", None)
        assert (line["done"], line["total"]) == (0, 30)
        assert "10% of keyed traffic" in line["message"]
        assert read_state().canary_iteration is None

    def test_it_replaces_a_running_canary_and_that_iteration_is_closed(self, cwd):
        _save(_canary_state())

        result, _ = _cli("canary", "hand-made")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        (line,) = _read(cwd, "iteration")
        assert (line["iteration_id"], line["outcome"], line["to_version"]) == (
            ITER, "rolled_back", None
        )
        assert line["reason"] == "replaced by soup loop canary hand-made"
        assert line["finished_at"] is not None
        assert read_state().canary_active == "hand-made"
        assert read_state().canary_iteration is None

    def test_a_log_failure_does_not_stop_the_command(self, cwd):
        _save(_state(status="running"))
        os.makedirs(os.path.join(".soup-loops", "progress.jsonl"))

        result, out = _cli("canary", "hand-made")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set: hand-made" in out
        assert "warning: progress.jsonl not written (" in out
        assert read_state().canary_active == "hand-made"


class TestRollbackCommand:
    def test_going_back_a_version_is_a_user_rollback_line(self, cwd, no_server):
        _save(_state(
            status="running", production_name="loop-train-aa", production_adapter="a",
            production_version=2, previous_production_version=1, latest_version=2,
            version_log_started=True, production_iteration_id=ITER,
            previous_production_iteration_id="iter-seed",
        ))

        result, out = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        (line,) = _read(cwd, "version")
        assert (line["cause"], line["actor"]) == ("rollback", "user")
        assert (line["from_version"], line["to_version"]) == (2, 1)
        assert line["iteration_id"] == ITER  # the iteration whose version was given up
        assert line["served_as"] is None  # the served model
        assert line["source"] == "soup-loop"
        state = read_state()
        assert (state.production_version, state.latest_version) == (1, 2)
        assert state.production_iteration_id == "iter-seed"
        (progress,) = _read(cwd, "progress")
        assert (progress["stage"], progress["event"]) == ("rollback", "end")
        assert progress["iteration_id"] == ITER

    def test_a_rollback_to_an_earlier_adapter_names_it(self, cwd, no_server):
        _save(_state(
            status="running", production_name="loop-train-bb", production_adapter="b",
            previous_production_name="loop-train-aa", previous_production_adapter="a",
            production_version=3, previous_production_version=2, latest_version=3,
            version_log_started=True,
        ))

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        (line,) = _read(cwd, "version")
        assert (line["from_version"], line["to_version"]) == (3, 2)
        assert line["served_as"] == "loop-train-aa"

    def test_a_running_canary_is_withdrawn_with_it_and_its_iteration_closed(self, cwd, no_server):
        _save(replace(
            _canary_state(), production_name="loop-train-zz", production_adapter="z",
            production_version=2, previous_production_version=1, latest_version=2,
        ))

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        (iteration,) = _read(cwd, "iteration")
        assert (iteration["iteration_id"], iteration["outcome"]) == (ITER, "rolled_back")
        assert iteration["reason"] == "withdrawn by soup loop rollback"
        (version,) = _read(cwd, "version")
        assert version["cause"] == "rollback"
        assert read_state().canary_active is None
        assert read_state().canary_iteration is None

    def test_withdrawing_a_canary_alone_changes_no_version(self, cwd, no_server):
        _save(_canary_state())

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert _read(cwd, "version") == []
        (iteration,) = _read(cwd, "iteration")
        assert iteration["outcome"] == "rolled_back"
        (progress,) = _read(cwd, "progress")
        assert progress["stage"] == "rollback" and progress["iteration_id"] == ITER
        assert read_state().production_version == 1

    def test_an_older_loop_yaml_gets_its_first_version_before_the_rollback(self, cwd, no_server):
        _save(_state(
            status="running", production_name="loop-train-aa", production_adapter="a",
            production_version=2, previous_production_version=1, latest_version=2,
        ))

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        initial, rollback = _read(cwd, "version")
        assert (initial["cause"], initial["to_version"]) == ("initial", 2)
        assert (rollback["from_version"], rollback["to_version"]) == (2, 1)
        assert rollback["ts"] >= initial["ts"]

    def test_a_server_that_refuses_the_switch_writes_nothing(self, cwd, monkeypatch):
        import soup_cli.utils.loop_stages as ls

        monkeypatch.setattr(ls, "switch_served_adapter", lambda name: (False, "refused"))
        _save(_state(
            status="running", production_name="loop-train-aa", production_version=2,
            previous_production_version=1, latest_version=2, version_log_started=True,
        ))

        result, out = _cli("rollback")

        assert result.exit_code == 1 and "rollback failed" in out
        assert not (cwd / ".soup-loops").exists()
        assert read_state().production_version == 2

    def test_nothing_to_roll_back_writes_nothing(self, cwd):
        _save(_state(status="running"))

        result, out = _cli("rollback")

        assert result.exit_code == 2 and "nothing to roll back to" in out
        assert not (cwd / ".soup-loops").exists()

    def test_a_log_failure_does_not_stop_the_rollback(self, cwd, no_server):
        _save(_state(
            status="running", production_name="loop-train-aa", production_version=2,
            previous_production_version=1, latest_version=2, version_log_started=True,
        ))
        os.makedirs(os.path.join(".soup-loops", "versions.jsonl"))

        result, out = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "rolled back:" in out and "warning: versions.jsonl not written (" in out
        assert read_state().production_name is None


# ---------------------------------------------------------------------------
# End to end: serve writes the conversation and feedback logs, the loop reads them and
# writes the other three; every line of all five files validates against its schema
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import soup_cli.commands.serve as serve  # noqa: E402
import soup_cli.utils.loop_stages as ls  # noqa: E402
from soup_cli.utils.loop_iteration import list_iterations, read_iteration  # noqa: E402

TOKEN = "e2e-" + "t" * 28
ENDPOINT = "http://127.0.0.1:8000"
FIX = "Hold the reset button for ten seconds."


class _AdapterModel:
    """A PEFT-shaped stand-in: which adapter is selected, which are loaded."""

    def __init__(self) -> None:
        self.current = None
        self.loaded: set = set()

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous, self.current = self.current, None
        try:
            yield
        finally:
            self.current = previous


class _Done:
    def __init__(self) -> None:
        self.returncode = 0
        self.stderr = b""


@pytest.fixture
def served(cwd, monkeypatch):
    """A real serve app (trace and feedback logs), a pre-wired loop, stubbed train and ship."""
    import httpx

    from soup_cli.monitoring.feedback_log import FeedbackLogWriter
    from soup_cli.monitoring.trace_logger import TraceLogWriter

    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(cwd / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_FEEDBACK_LOG"):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    (cwd / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    os.makedirs(os.path.join(".soup-loops", "adapters"))

    model = _AdapterModel()
    failing: set = set()
    served_by: list = []

    def generate(current_model, *args, **kwargs):
        served_by.append(current_model.current)
        if current_model.current in failing:
            raise RuntimeError("canary adapter broke")
        return f"answer from {current_model.current or 'base'}", 3, 4, "stop"

    def load(model_obj, path, name):
        assert os.path.isfile(os.path.join(path, "adapter_config.json"))
        model_obj.loaded.add(name)
        return model_obj

    monkeypatch.setattr(serve, "_generate_response", generate)
    monkeypatch.setattr(serve, "_load_runtime_adapter", load)
    app = serve._create_app(
        model_obj=model, tokenizer=object(), device="cpu", model_name="base",
        max_tokens_default=8, adapter_map={}, peft_adapter_names=set(),
        trace_log_writer=TraceLogWriter(os.path.join(".soup-loops", "traces", "serve.jsonl")),
        feedback_log_writer=FeedbackLogWriter(os.path.join(".soup-loops", "feedback.jsonl")),
        auth_token=TOKEN,
        adapter_dir=os.path.realpath(os.path.join(".soup-loops", "adapters")),
    )
    client = TestClient(app, base_url=ENDPOINT)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def post(url, headers=None, json=None, timeout=None, trust_env=None, **_):
        resp = client.post(urlparse(url).path, headers=headers or {}, json=json)
        return httpx.Response(
            resp.status_code, content=resp.content,
            headers={"content-type": resp.headers.get("content-type", "")},
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("soup_cli.utils.http_response.post_capped", post)

    def train(argv, **kwargs):
        config = yaml.safe_load(open(argv[argv.index("--config") + 1], encoding="utf-8"))
        os.makedirs(config["output"], exist_ok=True)
        with open(os.path.join(config["output"], "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": config["base"]}, fh)
        return _Done()

    def ship(argv, **kwargs):
        with open(argv[argv.index("--output") + 1], "w", encoding="utf-8") as fh:
            json.dump({"decision": "SHIP"}, fh)
        return _Done()

    monkeypatch.setattr(ls, "_TRAIN_RUNNER", train)
    monkeypatch.setattr(ls, "_SHIP_RUNNER", ship)

    result, _ = _cli(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired",
        "--base-model", "org/base", "--canary-traffic", "20%", "--train-method", "dpo",
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))

    def cycle():
        config = ls.build_prewired_watch_config(max_iterations=1, poll_interval_sec=1.0)
        return watch(dataclasses.replace(config, cost_fn=lambda state: 0.0))

    def chat(content, conversation_id=None) -> dict:
        body = {"messages": [{"role": "user", "content": content}]}
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def feedback(response_id, kind, **fields) -> None:
        resp = client.post(
            "/v1/feedback", headers=headers,
            json={"response_id": response_id, "kind": kind, **fields},
        )
        assert resp.status_code == 200, resp.text

    def keyed_traffic(count=300) -> list:
        """Keyed chats; returns (response id, "canary" | "stable") per answered request."""
        answers = []
        for index in range(count):
            served_by.clear()
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_id": f"user-{index}",
            })
            if resp.status_code == 200:  # (a failing canary answers 500: no answer to rate)
                bucket = "stable" if served_by[-1] is None else "canary"
                answers.append((resp.json()["response_id"], bucket))
        app.state.canary_outcomes.flush()
        return answers

    return {
        "client": client, "model": model, "failing": failing, "cycle": cycle, "chat": chat,
        "feedback": feedback, "keyed_traffic": keyed_traffic, "headers": headers,
    }


def _thumbs(served, answers, *, limit=None, value=1) -> dict:
    """Thumbs on the answers of each bucket (``limit`` per bucket) through POST /v1/feedback."""
    given = {"canary": 0, "stable": 0}
    for response_id, bucket in answers:
        if limit is not None and given[bucket] >= limit:
            continue
        served["feedback"](response_id, "thumb", value=value)
        given[bucket] += 1
    return given


def _start_canary(served) -> str:
    """One answer, corrected through POST /v1/feedback: an iteration trains on it and canaries."""
    answer = served["chat"]("How do I reset my router?")
    served["feedback"](answer["response_id"], "edit", edited_text=FIX)
    served["cycle"]()
    state = read_state()
    assert state.canary_active and state.canary_active.startswith("loop-train-"), state
    return state.canary_active


def _all_five_files_validate(cwd) -> dict:
    """Every line of the five formats against its schema; returns the lines by format."""
    layout = {
        "conversation": cwd / ".soup-loops" / "traces" / "serve.jsonl",
        "feedback": cwd / ".soup-loops" / "feedback.jsonl",
        "iteration": cwd / ".soup-loops" / "iterations.jsonl",
        "version": cwd / ".soup-loops" / "versions.jsonl",
        "progress": cwd / ".soup-loops" / "progress.jsonl",
    }
    assert set(layout) == set(loop_formats.FORMATS)
    lines = {}
    for name, path in layout.items():
        assert path.is_file(), name
        lines[name] = _lines(path)  # no BOM, ends in a newline, one JSON object per line
        for number, record in enumerate(lines[name], start=1):
            assert _errors(name, record) == [], (name, number, record)
    return lines


def test_a_loop_that_trains_canaries_promotes_and_rolls_back_leaves_five_valid_logs(
    served, cwd
):
    # A person corrected one answer; the loop trains on it, verifies it and starts a canary.
    name = _start_canary(served)
    lines = _all_five_files_validate(cwd)
    first = read_state().canary_iteration["iteration_id"]
    assert [v["cause"] for v in lines["version"]] == ["initial"]
    seed = lines["iteration"][0]
    assert (seed["to_version"], seed["outcome"]) == (1, "promoted")
    assert seed["reason"] == "initial deploy"
    canary_line = _last_wins(lines["iteration"])[first]
    assert canary_line["outcome"] == "canary" and canary_line["parent_iteration_id"] == (
        seed["iteration_id"]
    )
    assert (canary_line["from_version"], canary_line["to_version"]) == (1, 2)
    assert canary_line["dataset"]["dpo_pairs"] == 1 and canary_line["train"]["run_id"] == name
    assert [(p["stage"], p["event"]) for p in lines["progress"]] == [
        ("harvest", "start"), ("harvest", "end"), ("train", "start"), ("train", "end"),
        ("verdict", "start"), ("verdict", "end"), ("deploy", "start"), ("deploy", "end"),
        ("canary", "start"),
    ]
    assert {p["iteration_id"] for p in lines["progress"]} == {first}

    # The canary gets keyed traffic; people rate 5 answers per bucket: not enough yet.
    answers = served["keyed_traffic"]()
    given = _thumbs(served, answers, limit=5)
    assert given == {"canary": 5, "stable": 5}
    served["cycle"]()
    lines = _all_five_files_validate(cwd)
    waits = [p for p in lines["progress"] if p["stage"] == "canary" and p["event"] == "progress"]
    assert len(waits) == 1  # one canary line for the waiting cycle
    assert (waits[0]["done"], waits[0]["total"], waits[0]["iteration_id"]) == (5, 30, first)
    assert len(list_iterations()) == 1  # a waiting canary starts no iteration
    assert [r["outcome"] for r in lines["iteration"] if r["iteration_id"] == first] == ["canary"]

    # More ratings arrive: the next cycle promotes it (and finds nothing new to train on).
    _thumbs(served, answers)
    served["cycle"]()
    lines = _all_five_files_validate(cwd)
    assert read_state().production_name == name
    latest = _last_wins(lines["iteration"])
    promoted = latest[first]
    assert promoted["outcome"] == "promoted" and promoted["to_version"] == 2
    assert [r["outcome"] for r in lines["iteration"] if r["iteration_id"] == first] == [
        "canary", "promoted"
    ]
    quality = promoted["canary"]["quality"]
    assert quality["verdict"] == "OK" and quality["decided_on"] == "errors_and_quality"
    assert quality["rated"]["canary"] >= 30 and quality["rated"]["stable"] >= 30
    assert promoted["canary"]["samples"]["canary"] >= 30
    promote = lines["version"][-1]
    assert (promote["cause"], promote["actor"]) == ("promote", "loop")
    assert (promote["from_version"], promote["to_version"]) == (1, 2)
    assert (promote["iteration_id"], promote["served_as"]) == (first, name)
    assert [i for i in latest if i not in (seed["iteration_id"], first)]  # this cycle's iteration

    # The manifest of the promoting cycle says the same thing the log does about its evidence.
    (second_id,) = [i for i in list_iterations() if i != first]
    manifest = read_iteration(second_id)
    assert manifest.canary_quality == promoted["canary"]["quality"]
    assert latest[second_id]["outcome"] == "skipped" and latest[second_id]["from_version"] == 2
    assert latest[second_id]["parent_iteration_id"] == first

    # The person says no: back to the served model. The iteration is not rewritten.
    count = len(lines["iteration"])
    result, text = _cli("rollback")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"rolled back: {name} -> the served model" in text
    lines = _all_five_files_validate(cwd)
    assert len(lines["iteration"]) == count
    rollback = lines["version"][-1]
    assert (rollback["cause"], rollback["actor"]) == ("rollback", "user")
    assert (rollback["from_version"], rollback["to_version"]) == (2, 1)
    assert rollback["iteration_id"] == first
    assert [v["to_version"] for v in lines["version"]] == [1, 2, 1]
    assert _last_wins(lines["iteration"])[first]["outcome"] == "promoted"
    assert read_state().production_version == 1 and read_state().latest_version == 2


def test_every_iteration_line_agrees_with_its_manifest(served, cwd):
    _start_canary(served)
    answers = served["keyed_traffic"]()
    _thumbs(served, answers)
    served["cycle"]()

    latest = _last_wins(_all_five_files_validate(cwd)["iteration"])
    for iteration_id in list_iterations():
        manifest = read_iteration(iteration_id)
        line = latest[iteration_id]
        assert line["started_at"] == manifest.started_at
        assert line["estimated_cost_usd"] == manifest.estimated_cost_usd
        assert (line["train"] or {}).get("run_id") == manifest.run_id
        if manifest.run_id is None:
            assert line["train"] is None
        if manifest.shipped:
            assert line["dataset"] is not None and line["outcome"] in ("canary", "promoted")
        if line["outcome"] in ("skipped", "failed"):
            assert line["finished_at"] == manifest.finished_at
    assert len(latest) == len(list_iterations()) + 1  # and the seed, which has no manifest


def test_a_regressing_canary_is_rolled_back_and_the_log_says_so(served, cwd):
    name = _start_canary(served)
    served["failing"].add(name)
    answers = served["keyed_traffic"]()
    assert len(answers) < 300  # the requests the broken canary got were not answered
    first = read_state().canary_iteration["iteration_id"]

    served["cycle"]()

    lines = _all_five_files_validate(cwd)
    withdrawn = _last_wins(lines["iteration"])[first]
    assert withdrawn["outcome"] == "rolled_back" and withdrawn["to_version"] is None
    assert withdrawn["canary"]["verdict"] == "MAJOR"
    assert withdrawn["canary"]["samples"]["canary"] > 0
    assert [v["cause"] for v in lines["version"]] == ["initial"]  # production never changed
    assert ("rollback", "end") in [(p["stage"], p["event"]) for p in lines["progress"]]
    assert read_state().production_version == 1 and read_state().canary_iteration is None
