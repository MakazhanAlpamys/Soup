"""The loop's verdict v2: the holdout split, graduation, the lineage cursor, the plan-time wait
and ``soup ship --gate loop-v2``.

The continual-learning loop used to judge a candidate by a task set and a general suite; it never
saw a correction a person made. The loop of the consuming application
decides this with two rules both projects share, and this file pins them in the pre-wired loop:

- **the holdout split.** A corrected answer whose prompt hashes under 30 is kept out of training in
  exactly ONE iteration (the first that trains), feeds that iteration's held-out check, and trains
  from the next iteration on. Thumbs and ratings are never held out;
- **the lineage cursor.** The data cursor (``harvest_cursor_ts``) follows the version in
  production: it moves when the candidate that used the rows is promoted, ``soup loop rollback``
  returns to the previous one. A candidate that is thrown away leaves its rows new;
- **the plan-time wait.** Fewer than 5 trainable corrections: the cycle trains nothing and says so;
- **the gate.** ``soup ship --gate loop-v2 --corrections FILE --held-out FILE`` decides, and a
  DON'T SHIP with ``canary_decides`` still deploys the candidate to a canary.

Real code throughout (harvest, train stage, verify stage, daemon, ``loop.yaml``); stubbed are the
``soup train`` and ``soup ship`` subprocesses (``_TRAIN_RUNNER`` / ``_SHIP_RUNNER``), the
paraphrases (written to ``paraphrases.jsonl`` as the train stage would) and the serve endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils import loop_formats, loop_gate, loop_holdout
from soup_cli.utils.canary_router import CanaryPolicy, record_bucket_outcome
from soup_cli.utils.loop_daemon import run_once
from soup_cli.utils.loop_feedback import EditedAnswer
from soup_cli.utils.loop_iteration import IterationRecord, read_iteration, write_iteration
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, init_state, read_state, write_state

jsonschema = pytest.importorskip("jsonschema")

T0 = 1_700_000_000.0
ENDPOINT = "http://127.0.0.1:8000"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TRACES = os.path.join(".soup-loops", "traces", "conversations.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")
PARAPHRASES = os.path.join(".soup-loops", "paraphrases.jsonl")
LEDGER = os.path.join(".soup-loops", "holdout.jsonl")
NOTE_4_OF_5 = "Waiting: 4/5 trainable corrections (2 more kept aside for checking) · 41 new ratings"
CANARY_DECIDES_NOTE = "No corrections to check - the canary's ratings decide"


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _bucket(text: str) -> int:
    """The split, derived here again from the rule so a change of the code cannot hide."""
    return int(hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest(), 16) % 100


def _prompts(held: bool, count: int, start: int = 0) -> list:
    found, number = [], start
    while len(found) < count:
        text = f"Question {number}: how do I configure feature {number}?"
        if (_bucket(text) < 30) == held:
            found.append(text)
        number += 1
    return found


def _prompt_in_bucket(bucket: int) -> str:
    number = 0
    while _bucket(f"Bucket probe {number}") != bucket:
        number += 1
    return f"Bucket probe {number}"


HELD = _prompts(True, 12)
FREE = _prompts(False, 24)


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def fixed_text(number: int) -> str:
    return f"Fix {number}: alpha{number} bravo{number} charlie{number}."


def wrong_text(number: int) -> str:
    return f"Wrong answer {number}."


# ---------------------------------------------------------------------------
# The world: files an application writes, stubs for the two subprocesses, the daemon
# ---------------------------------------------------------------------------


def _append(path: str, lines: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))


def _read(kind_or_path: str) -> list:
    path = {
        "iteration": os.path.join(".soup-loops", "iterations.jsonl"),
        "progress": os.path.join(".soup-loops", "progress.jsonl"),
        "version": os.path.join(".soup-loops", "versions.jsonl"),
    }.get(kind_or_path, kind_or_path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest() -> dict:
    """The latest iterations.jsonl line per iteration, as a reader applies the contract."""
    return {line["iteration_id"]: line for line in _read("iteration")}


def schema_errors(kind: str, line: dict) -> list:
    validator = jsonschema.Draft202012Validator(
        loop_formats.load_schema(kind),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )
    return [error.message for error in validator.iter_errors(line)]


def scoring() -> dict:
    return {"method": "added_recall", "threshold": 0.5, "template": None}


def gate_output(
    decision="SHIP", *, fixed=5, broken=0, ties=0, outcome="pass", held=(0, 0, 0),
    held_outcome="not_measured", canary_decides=False, reason=None, gate="loop-v2",
) -> dict:
    """What ``soup ship --gate loop-v2 --output`` writes (docs/evaluation.md)."""
    data = {
        "decision": decision, "failed_rule": None if decision == "SHIP" else "corrections",
        "task_win": {"mode": "metric", "base": 0.5, "tuned": 0.5, "won": False},
        "benchmark_deltas": [],
    }
    if gate is not None:
        better, worse, held_ties = held
        data.update({
            "gate": gate,
            "corrections": {
                "fixed": fixed, "broken": broken, "ties": ties, "n_edits": fixed + broken + ties,
                "n_paraphrases": 2 * (fixed + broken + ties), "p_value": 0.03125,
                "scoring": scoring(), "outcome": outcome,
            },
            "held_out": {
                "better": better, "worse": worse, "ties": held_ties,
                "n": better + worse + held_ties, "p_value": None, "scoring": scoring(),
                "outcome": held_outcome,
            },
            "canary_decides": canary_decides,
            "reason": reason,
        })
    return data


def _rows(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().split("\n") if line.strip()]


class Trainer:
    """A ``soup train`` stand-in that writes an adapter and remembers what it trained on."""

    def __init__(self) -> None:
        self.calls = 0
        self.returncode = 0
        self.trained_on: list = []

    def __call__(self, argv, **kwargs):
        self.calls += 1
        if self.returncode != 0:
            return SimpleNamespace(returncode=self.returncode, stderr=b"boom")
        with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.trained_on.append(_rows(config["data"]["train"]))
        os.makedirs(config["output"], exist_ok=True)
        with open(os.path.join(config["output"], "adapter_config.json"), "w") as handle:
            json.dump({"base_model_name_or_path": config["base"]}, handle)
        return SimpleNamespace(returncode=0, stderr=b"")

    def answers(self) -> list:
        """The assistant text of every row the last run trained on."""
        return [row["messages"][-1]["content"] for row in self.trained_on[-1]]


class Ship:
    """A ``soup ship`` stand-in: remembers the argv and the two files it was given."""

    def __init__(self) -> None:
        self.calls: list = []
        self.output = gate_output()
        self.corrections: list = []
        self.held_out: list = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        for flag, name in (("--corrections", "corrections"), ("--held-out", "held_out")):
            if flag in argv:
                setattr(self, name, _rows(argv[argv.index(flag) + 1]))
        out = argv[argv.index("--output") + 1]
        if self.output is not None:
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(self.output, handle)
        return SimpleNamespace(returncode=0, stderr=b"")

    @property
    def argv(self) -> list:
        return self.calls[-1]


class World:
    """A pre-wired sft loop (verify ship) over the files an application writes."""

    def __init__(self, root, monkeypatch, **init) -> None:
        self.root = root
        self.trainer = Trainer()
        self.ship = Ship()
        self.loaded: list = []
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", self.trainer)
        monkeypatch.setattr(ls, "_SHIP_RUNNER", self.ship)
        monkeypatch.setattr(ls, "_LOAD_POSTER", lambda endpoint, name, path: (200, ""))
        monkeypatch.setattr(ls, "resolve_sft_quantization", lambda requested, base: ("none", None))
        self.stats = str(root / "stats.json")
        init.setdefault("train_method", "sft")
        init.setdefault("min_new_edits", 1)
        init.setdefault("verify", "ship")
        self.state, _ = init_state(
            "./base", "tasks.jsonl", "b", pre_wired=True, base_model="org/base", **init
        )
        self.config = replace(
            ls.build_prewired_watch_config(max_iterations=1, poll_interval_sec=1.0),
            cost_fn=lambda state: 0.0,
            promote_fn=lambda state: {"promoted": True},
            quality_fn=None,
            canary_stats_path=self.stats,
        )
        self.records: list = []
        self.answered: set = set()

    # -- what an application writes ------------------------------------------------------

    def answer(self, number: int, prompt: str, response=None) -> None:
        if number in self.answered:
            return
        self.answered.add(number)
        _append(TRACES, [{
            "schema_version": 1, "id": uid(number), "conversation_id": f"chat-{number}",
            "ts": iso(number), "model": "base", "version": 1, "adapter": None,
            "canary_bucket": None, "messages": [{"role": "user", "content": prompt}],
            "response": response or wrong_text(number), "finish_reason": "stop",
            "source": "playground",
        }])

    def paraphrase(self, number: int, prompt: str) -> None:
        messages = [{"role": "user", "content": prompt}]
        _append(PARAPHRASES, [{
            "schema_version": 1, "response_id": uid(number),
            "prompt_hash": loop_formats.prompt_hash(messages),
            "paraphrases": [f"Reworded {number} one: {prompt}", f"Reworded {number} two: {prompt}"],
            "generator": {"model": "m", "template": {"id": "loop-paraphrase", "version": "1"}},
            "created_at": "2026-11-06T09:00:00+00:00",
        }])

    def edit(self, number: int, prompt: str, *, at=None, paraphrases=True) -> None:
        self.answer(number, prompt)
        _append(FEEDBACK, [{
            "schema_version": 1, "id": uid(5000 + number), "response_id": uid(number),
            "ts": iso(100 + number if at is None else at), "kind": "edit",
            "edited_text": fixed_text(number), "source": "playground",
        }])
        if paraphrases:
            self.paraphrase(number, prompt)

    def edits(self, numbers, prompts, **kwargs) -> None:
        for number, prompt in zip(numbers, prompts):
            self.edit(number, prompt, **kwargs)

    def thumbs(self, first: int, count: int, *, at=None) -> None:
        """``count`` answers to different prompts, each given a thumb up."""
        for number in range(first, first + count):
            self.answer(number, f"Rated question {number}", f"Fine answer {number}")
            _append(FEEDBACK, [{
                "schema_version": 1, "id": uid(7000 + number), "response_id": uid(number),
                "ts": iso(300 + number if at is None else at), "kind": "thumb", "value": 1,
                "source": "playground",
            }])

    # -- the daemon ----------------------------------------------------------------------

    def cycle(self, **kwargs):
        self.state, record, _ = run_once(self.state, self.config, log=LoopLog(), **kwargs)
        self.records.append(record)
        return record

    def promote(self):
        """Healthy answers from the running canary, then the cycle that promotes it."""
        policy = CanaryPolicy(
            stable=self.state.stable_name(), canary=self.state.canary_active, traffic_pct=5.0
        )
        for bucket in ("canary", "stable"):
            for _ in range(30):
                record_bucket_outcome(
                    policy, bucket, True, rollout_id=self.state.canary_rollout_id, path=self.stats
                )
        return self.cycle()

    def fail_canary(self):
        policy = CanaryPolicy(
            stable=self.state.stable_name(), canary=self.state.canary_active, traffic_pct=5.0
        )
        for _ in range(30):
            record_bucket_outcome(
                policy, "canary", False, rollout_id=self.state.canary_rollout_id, path=self.stats
            )
            record_bucket_outcome(
                policy, "stable", True, rollout_id=self.state.canary_rollout_id, path=self.stats
            )
        return self.cycle()

    def ledger(self) -> list:
        return _read(LEDGER)

    def harvest(self) -> dict:
        return dict(ls.harvest_from_traces(self.state))


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_FEEDBACK_LOG",
        ls.TOOL_AUTH_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


@pytest.fixture
def world(cwd, monkeypatch):
    return World(cwd, monkeypatch)


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


def _init_loop() -> None:
    _cli("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


def _flat(text: str) -> str:
    """A status table as plain words: the borders gone, a wrapped cell read as one line."""
    return " ".join(re.sub("[\u2502\u2503|]", " ", text).split())


# ---------------------------------------------------------------------------
# The split: sha256 of the last user message, casefolded
# ---------------------------------------------------------------------------


class TestSplit:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("How do I reset my password?", 87),
            ("  How do I RESET my password?  ", 87),  # outer whitespace and case do not matter
            ("Straße öffnen – wie?", 12),  # casefold: ß is "ss"
            ("Line one\nline two\twith tab", 5),  # inner whitespace is kept, not collapsed
            ("Привет, как "
             "сбросить "
             "пароль?", 83),
        ],
    )
    def test_the_pinned_fixtures(self, text, value):
        assert loop_holdout.holdout_bucket(text) == value
        assert loop_holdout.is_held_out(text) is (value < 30)

    def test_the_fixture_that_is_held_out_is_the_line_one_prompt(self):
        held = [
            text for text in (
                "How do I reset my password?", "Straße öffnen – wie?",
                "Line one\nline two\twith tab",
            )
            if loop_holdout.is_held_out(text)
        ]

        assert held == ["Straße öffnen – wie?", "Line one\nline two\twith tab"]

    def test_casefold_not_lower(self):
        """"Straße" lower-cases to itself but case-folds to "strasse": the two hash apart."""
        text = "Straße öffnen – wie?"
        lowered = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
        folded = hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()

        assert lowered != folded
        assert loop_holdout.holdout_bucket(text) == int(folded, 16) % 100 == 12
        assert int(lowered, 16) % 100 != 12

    def test_the_threshold_is_30_so_29_is_held_out_and_30_is_not(self):
        assert loop_holdout.HOLDOUT_PERCENT == 30
        assert loop_holdout.is_held_out(_prompt_in_bucket(29)) is True
        assert loop_holdout.is_held_out(_prompt_in_bucket(30)) is False
        assert loop_holdout.is_held_out(_prompt_in_bucket(0)) is True
        assert loop_holdout.is_held_out(_prompt_in_bucket(99)) is False

    def test_about_three_in_ten_prompts_are_held_out(self):
        held = sum(loop_holdout.is_held_out(f"prompt number {n}") for n in range(2000))

        assert 500 < held < 700

    def test_the_key_is_the_last_user_message(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": HELD[0]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": f"  {FREE[0].upper()}  "},
            {"role": "assistant", "content": "ok"},
        ]

        key = loop_holdout.holdout_key(messages)

        assert key == FREE[0].casefold()
        assert loop_holdout.is_held_out(key) is False
        assert loop_holdout.holdout_key(list(reversed(messages[1:4]))) == HELD[0].casefold()

    @pytest.mark.parametrize(
        "messages",
        [
            [],
            [{"role": "system", "content": "Be brief."}],
            [{"role": "assistant", "content": "hi"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": "   \n"}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": ["parts"]}],
            [{"role": "user"}],
            ["not a message"],
            None,
            "a string",
        ],
    )
    def test_no_user_message_with_text_means_no_key(self, messages):
        assert loop_holdout.holdout_key(messages) is None

    def test_an_edit_without_a_key_is_never_held_out(self):
        edit = EditedAnswer(uid(1), ({"role": "system", "content": "x"},), "a", "b", T0)

        plan = loop_holdout.plan_holdout([edit], None, loop_holdout.Ledger())

        assert plan.kept_aside == () and [e.id for e in plan.trainable] == [uid(1)]

    def test_the_ledger_stores_a_digest_of_the_key_not_the_prompt(self):
        key = loop_holdout.holdout_key([{"role": "user", "content": " Hello "}])

        assert loop_holdout.key_digest(key) == hashlib.sha256(b"hello").hexdigest()


# ---------------------------------------------------------------------------
# The plan, as a pure function of the edits, the cursor and the ledger
# ---------------------------------------------------------------------------


def make_edit(number, prompt, ts=None) -> EditedAnswer:
    return EditedAnswer(
        uid(number), ({"role": "user", "content": prompt},), wrong_text(number),
        fixed_text(number), T0 + 100 + number if ts is None else ts,
    )


def ledger_of(*rows) -> "loop_holdout.Ledger":
    ledger = loop_holdout.Ledger()
    for row in rows:
        entry = loop_holdout.LedgerEntry(**row)
        ledger.latest[entry.edit_id] = entry
        ledger.keys.add(entry.prompt_key)
    return ledger


def entry(number, prompt, event, until=None, iteration="iter-a") -> dict:
    return {
        "event": event, "edit_id": uid(number), "iteration_id": iteration, "until": until,
        "prompt_key": loop_holdout.key_digest(prompt.strip().casefold()),
    }


class TestPlan:
    def test_the_first_sighting_of_a_held_side_prompt_is_kept_aside(self):
        edits = [make_edit(1, FREE[0]), make_edit(2, HELD[0]), make_edit(3, HELD[0])]

        plan = loop_holdout.plan_holdout(edits, None, loop_holdout.Ledger())

        # every edit on the held prompt is kept aside together; none trains
        assert [e.id for e in plan.kept_aside] == [uid(2), uid(3)]
        assert [e.id for e in plan.trainable] == [uid(1)]
        assert plan.graduated == ()

    def test_a_prompt_seen_in_the_ledger_is_never_kept_aside_again(self):
        edits = [make_edit(2, HELD[0])]
        ledger = ledger_of(entry(2, HELD[0], "kept_aside", until=T0 + 102))

        plan = loop_holdout.plan_holdout(edits, None, ledger)

        assert plan.kept_aside == ()
        assert [e.id for e in plan.graduated] == [uid(2)] == [e.id for e in plan.trainable]

    def test_a_new_edit_on_a_prompt_that_was_seen_trains(self):
        ledger = ledger_of(entry(2, HELD[0], "kept_aside", until=T0 + 102))
        edits = [make_edit(2, HELD[0]), make_edit(9, HELD[0])]

        plan = loop_holdout.plan_holdout(edits, None, ledger)

        assert plan.kept_aside == ()
        assert [e.id for e in plan.trainable] == [uid(2), uid(9)]

    def test_an_edit_the_cursor_has_passed_is_not_trainable_unless_it_waits_to_graduate(self):
        old, waiting = make_edit(1, FREE[0]), make_edit(2, HELD[0])
        ledger = ledger_of(entry(2, HELD[0], "kept_aside", until=T0 + 102))

        plan = loop_holdout.plan_holdout([old, waiting], T0 + 200, ledger)

        assert plan.kept_aside == ()
        assert [e.id for e in plan.trainable] == [uid(2)]  # old trained already; this one waits
        assert [e.id for e in plan.graduated] == [uid(2)]

    def test_a_graduated_edit_the_cursor_has_passed_is_done(self):
        edit = make_edit(2, HELD[0])
        ledger = ledger_of(
            entry(2, HELD[0], "kept_aside", until=T0 + 102, iteration="iter-a"),
            entry(2, HELD[0], "graduated", until=T0 + 150, iteration="iter-b"),
        )

        done = loop_holdout.plan_holdout([edit], T0 + 150, ledger)
        not_promoted = loop_holdout.plan_holdout([edit], T0 + 120, ledger)

        assert done.trainable == () and done.graduated == ()
        # the iteration that graduated it was thrown away: the cursor is behind it
        assert [e.id for e in not_promoted.graduated] == [uid(2)]

    def test_an_edit_given_again_after_it_graduated_is_news_but_not_a_second_graduation(self):
        ledger = ledger_of(entry(2, HELD[0], "graduated", until=T0 + 150))
        edit = make_edit(2, HELD[0], ts=T0 + 400)

        plan = loop_holdout.plan_holdout([edit], T0 + 150, ledger)

        # The consuming application keys graduation by response id: the answer trained, so it is
        # not reported again
        assert [e.id for e in plan.trainable] == [uid(2)] and plan.graduated == ()

    def test_an_old_held_side_edit_with_no_ledger_row_trained_before_the_ledger(self):
        edit = make_edit(2, HELD[0])

        plan = loop_holdout.plan_holdout([edit], T0 + 500, loop_holdout.Ledger())

        assert plan.kept_aside == () and plan.trainable == () and plan.graduated == ()

    def test_with_no_cursor_nothing_is_old(self):
        edits = [make_edit(n, FREE[n]) for n in range(3)]

        plan = loop_holdout.plan_holdout(edits, None, loop_holdout.Ledger())

        assert [e.id for e in plan.new_trainable] == [e.id for e in edits]

    def test_an_edit_without_a_timestamp_is_new_only_while_there_is_no_cursor(self):
        edit = EditedAnswer(
            uid(1), ({"role": "user", "content": FREE[0]},), "a", "b", None
        )

        assert loop_holdout.plan_holdout([edit], None, loop_holdout.Ledger()).trainable
        assert not loop_holdout.plan_holdout([edit], T0, loop_holdout.Ledger()).trainable

    def test_trainable_is_each_edit_once_in_the_order_given(self):
        ledger = ledger_of(entry(3, HELD[0], "kept_aside", until=T0 + 103))
        edits = [make_edit(1, FREE[0]), make_edit(3, HELD[0]), make_edit(4, FREE[1])]

        plan = loop_holdout.plan_holdout(edits, None, ledger)

        # 3 is both new and graduating: counted once
        assert [e.id for e in plan.trainable] == [uid(1), uid(3), uid(4)]
        assert [e.id for e in plan.graduated] == [uid(3)]


class TestWaitingNote:
    def test_the_note_gives_both_counts_and_the_ratings(self):
        assert loop_holdout.waiting_note(4, 2, 41) == NOTE_4_OF_5

    def test_without_edits_kept_aside_the_parenthesis_is_left_out(self):
        note = loop_holdout.waiting_note(3, 0, 12)

        assert note == "Waiting: 3/5 trainable corrections · 12 new ratings"

    def test_one_rating_is_singular(self):
        note = loop_holdout.waiting_note(1, 0, 1)

        assert note == "Waiting: 1/5 trainable corrections · 1 new rating"

    @pytest.mark.parametrize("trainable", [0, 5, 6, 40])
    def test_no_note_at_zero_or_at_five_and_over(self, trainable):
        assert loop_holdout.waiting_note(trainable, 1, 50) is None

    @pytest.mark.parametrize("trainable", [1, 2, 3, 4])
    def test_a_note_from_one_to_four(self, trainable):
        assert loop_holdout.waiting_note(trainable, 0, 9).startswith(
            f"Waiting: {trainable}/5 trainable corrections"
        )

    def test_the_minimum_is_ships_discordant_minimum(self):
        assert loop_holdout.MIN_TRAINABLE_EDITS == 5


# ---------------------------------------------------------------------------
# The ledger: append-only, last row wins, written by the iteration that trains
# ---------------------------------------------------------------------------


class TestLedgerFile:
    def test_a_missing_file_is_an_empty_ledger(self, cwd):
        ledger = loop_holdout.read_ledger()

        assert ledger.latest == {} and ledger.keys == set()

    def test_the_last_row_per_edit_wins_and_every_key_is_remembered(self, cwd):
        rows = [
            {"schema_version": 1, "ts": iso(0), "iteration_id": "iter-a", "event": "kept_aside",
             "edit_id": uid(1), "prompt_key": "a" * 64, "until": T0 + 5},
            {"schema_version": 1, "ts": iso(1), "iteration_id": "iter-b", "event": "graduated",
             "edit_id": uid(1), "prompt_key": "a" * 64, "until": T0 + 9},
            {"schema_version": 1, "ts": iso(1), "iteration_id": "iter-b", "event": "kept_aside",
             "edit_id": uid(2), "prompt_key": "b" * 64, "until": None},
        ]
        _append(LEDGER, rows)

        ledger = loop_holdout.read_ledger()

        assert ledger.latest[uid(1)].event == "graduated"
        assert ledger.latest[uid(1)].iteration_id == "iter-b"
        assert ledger.latest[uid(1)].until == T0 + 9
        assert ledger.latest[uid(2)].until is None
        assert ledger.keys == {"a" * 64, "b" * 64}

    GOOD_ROW = {
        "schema_version": 1, "ts": "2026-11-06T09:00:00+00:00", "iteration_id": "iter-a",
        "event": "kept_aside", "edit_id": "x", "prompt_key": "a" * 64, "until": 5.0,
    }

    def test_the_row_the_bad_ones_are_made_from_is_read(self, cwd):
        _append(LEDGER, [self.GOOD_ROW])

        ledger = loop_holdout.read_ledger()

        assert ledger.latest["x"].until == 5.0 and ledger.stats.skipped == 0

    @pytest.mark.parametrize(
        "change",
        [
            {"schema_version": 2}, {"schema_version": True}, {"schema_version": "1"},
            {"event": "other"}, {"event": None}, {"edit_id": ""}, {"edit_id": 7},
            {"edit_id": "x" * 257}, {"edit_id": "a\x00b"},
            {"iteration_id": ""}, {"iteration_id": None}, {"iteration_id": "i" * 129},
            {"prompt_key": "short"}, {"prompt_key": "A" * 64}, {"prompt_key": "g" * 64},
            {"prompt_key": None}, {"until": "soon"}, {"until": True}, {"until": -1},
            {"until": 1e300}, {"until": [1]},
        ],
    )
    def test_a_row_with_one_bad_field_is_skipped_and_counted(self, cwd, change):
        _append(LEDGER, [{**self.GOOD_ROW, **change}])

        ledger = loop_holdout.read_ledger()

        assert ledger.latest == {} and ledger.stats.skipped == 1

    def test_a_row_without_until_or_ts_is_read(self, cwd):
        row = {key: value for key, value in self.GOOD_ROW.items() if key not in ("until", "ts")}
        _append(LEDGER, [row])

        ledger = loop_holdout.read_ledger()

        assert ledger.latest["x"].until is None

    def test_garbage_lines_are_skipped_not_raised(self, cwd):
        os.makedirs(".soup-loops")
        with open(LEDGER, "w", encoding="utf-8") as handle:
            handle.write("not json\n[1]\n\n")

        ledger = loop_holdout.read_ledger()

        assert ledger.latest == {} and ledger.stats.skipped == 2

    def test_recording_a_plan_appends_one_row_per_edit(self, cwd):
        plan = loop_holdout.plan_holdout(
            [make_edit(1, FREE[0]), make_edit(2, HELD[0]), make_edit(3, HELD[1])],
            None, ledger_of(entry(3, HELD[1], "kept_aside", until=T0 + 103)),
        )

        notes = loop_holdout.record_plan(
            plan, iteration_id="iter-20261106T090000-abcd1234", until=T0 + 103
        )

        assert notes == []
        rows = _read(LEDGER)
        assert [(row["event"], row["edit_id"]) for row in rows] == [
            ("kept_aside", uid(2)), ("graduated", uid(3)),
        ]
        for row in rows:
            assert row["schema_version"] == 1
            assert row["iteration_id"] == "iter-20261106T090000-abcd1234"
            assert row["until"] == T0 + 103
            assert re.fullmatch(r"[0-9a-f]{64}", row["prompt_key"])
            assert row["ts"].endswith("+00:00")

    def test_recording_never_rewrites_an_earlier_row(self, cwd):
        _append(LEDGER, [{"keep": "me"}])
        plan = loop_holdout.plan_holdout([make_edit(2, HELD[0])], None, loop_holdout.Ledger())

        loop_holdout.record_plan(plan, iteration_id="iter-a", until=None)
        loop_holdout.record_plan(plan, iteration_id="iter-b", until=None)

        rows = _read(LEDGER)
        assert rows[0] == {"keep": "me"} and len(rows) == 3
        assert [row["iteration_id"] for row in rows[1:]] == ["iter-a", "iter-b"]

    def test_a_ledger_that_cannot_be_written_is_a_note_not_an_error(self, cwd):
        os.makedirs(LEDGER)  # a directory where the file goes
        plan = loop_holdout.plan_holdout([make_edit(2, HELD[0])], None, loop_holdout.Ledger())

        notes = loop_holdout.record_plan(plan, iteration_id="iter-a", until=None)

        assert len(notes) == 1 and notes[0].startswith("holdout.jsonl not written")
        assert str(cwd) not in notes[0]


# ---------------------------------------------------------------------------
# The harvest: which rows train, the plan-time gate
# ---------------------------------------------------------------------------


class TestHarvestSplit:
    def test_a_held_side_edit_is_kept_out_of_the_dataset_and_the_plan_says_so(self, world):
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])

        out = world.harvest()

        plan = out["holdout"]
        assert [e.id for e in plan.kept_aside] == [uid(6)]
        assert len(plan.trainable) == 5
        with open(out["pairs_path"], encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        answers = [row["messages"][-1]["content"] for row in rows]
        assert answers == [fixed_text(n) for n in range(1, 6)]
        assert out["pairs_harvested"] == 5
        assert out["training_blocked"] is None and out["waiting"] is None
        # the paraphrase step asks about what trains, not about what is kept aside
        assert [e.id for e in out["edited_answers"]] == [uid(n) for n in range(1, 6)]

    @pytest.mark.parametrize("method", ["dpo", "kto", "sft"])
    def test_every_dataset_leaves_out_the_kept_aside_edit(self, cwd, monkeypatch, method):
        world = World(cwd, monkeypatch, train_method=method)
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])

        out = world.harvest()

        assert out["holdout"].kept_aside and out["pairs_harvested"] > 0
        text = open(out["pairs_path"], encoding="utf-8").read()
        assert HELD[0] not in text and fixed_text(6) not in text and wrong_text(6) not in text

    def test_the_kto_original_of_a_kept_aside_edit_is_left_out_too(self, cwd, monkeypatch):
        world = World(cwd, monkeypatch, train_method="kto")
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])

        text = open(world.harvest()["pairs_path"], encoding="utf-8").read()

        assert wrong_text(1) in text  # the others' originals are bad rows, as ever
        assert wrong_text(6) not in text

    def test_thumbs_and_ratings_on_a_held_side_prompt_are_never_held_out(self, world):
        world.answer(1, HELD[0], "A liked answer.")
        world.answer(2, HELD[1], "A rated answer.")
        _append(FEEDBACK, [
            {"schema_version": 1, "id": uid(901), "response_id": uid(1), "ts": iso(101),
             "kind": "thumb", "value": 1, "source": "playground"},
            {"schema_version": 1, "id": uid(902), "response_id": uid(2), "ts": iso(102),
             "kind": "rating", "value": 5, "source": "playground"},
        ])
        world.edits(range(3, 8), FREE[:5])

        out = world.harvest()

        assert out["holdout"].kept_aside == ()
        answers = [json.loads(line)["messages"][-1]["content"]
                   for line in open(out["pairs_path"], encoding="utf-8")]
        assert "A liked answer." in answers and "A rated answer." in answers

    def test_a_loop_that_does_not_ship_is_not_split(self, cwd, monkeypatch):
        world = World(cwd, monkeypatch, verify="eval_gate")
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])

        out = world.harvest()

        assert out["holdout"] is None and out["waiting"] is None
        assert out["pairs_harvested"] == 6
        assert [e.id for e in out["edited_answers"]] == [uid(n) for n in range(1, 7)]

    def test_new_edits_still_counts_every_new_edit_kept_aside_ones_too(self, world):
        world.edits(range(1, 4), FREE[:3])
        world.edit(4, HELD[0])

        out = world.harvest()

        assert out["new_edits"] == 4 and out["new_rated"] == 4


class TestPlanTimeGate:
    """Fewer than five trainable corrections: spend no GPU, say what is missing."""

    def test_zero_trainable_edits_train_anyway_when_the_trigger_fires(self, world):
        world.thumbs(1, 30)

        record = world.cycle()

        assert record.run_id and world.trainer.calls == 1
        assert world.state.waiting_note is None
        assert world.ship.corrections == [] and world.ship.held_out == []

    def test_zero_trainable_edits_and_no_trigger_wait_as_ever(self, world):
        world.thumbs(1, 3)

        record = world.cycle()

        assert record.run_id is None and world.trainer.calls == 0
        assert "not_enough_new_data" in record.notes and "Waiting" not in record.notes

    def test_six_new_edits_two_kept_aside_is_four_so_it_waits_with_the_exact_note(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.edits(range(5, 7), HELD[:2])
        world.thumbs(10, 35)  # 6 edits + 35 rated answers = 41 new ratings; 6 < 8 edits

        record = world.cycle()

        assert record.run_id is None and world.trainer.calls == 0
        assert NOTE_4_OF_5 in record.notes
        assert world.state.waiting_note == NOTE_4_OF_5
        assert world.ledger() == []  # nothing trained: nothing was kept aside for good

    def test_the_wait_is_a_progress_line_that_validates(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.edits(range(5, 7), HELD[:2])
        world.thumbs(10, 35)

        record = world.cycle()

        waiting = [line for line in _read("progress") if line["event"] == "progress"]
        (line,) = waiting
        assert (line["stage"], line["done"], line["total"]) == ("harvest", 4, 5)
        assert line["message"] == NOTE_4_OF_5
        assert line["iteration_id"] == record.iteration_id
        assert schema_errors("progress", line) == []

    def test_a_wait_that_has_not_changed_is_not_written_again(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.thumbs(10, 35)

        for _ in range(3):
            world.cycle()

        assert len([line for line in _read("progress") if line["event"] == "progress"]) == 1

    def test_a_wait_that_changed_is_written(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.thumbs(10, 35)
        world.cycle()
        world.thumbs(50, 1)

        world.cycle()

        messages = [line["message"] for line in _read("progress") if line["event"] == "progress"]
        assert messages == [
            "Waiting: 4/5 trainable corrections · 39 new ratings",
            "Waiting: 4/5 trainable corrections · 40 new ratings",
        ]

    @pytest.mark.parametrize(
        ("free", "trains"), [(1, False), (2, False), (4, False), (5, True), (6, True)]
    )
    def test_the_boundary_is_five_trainable_edits(self, world, free, trains):
        world.edits(range(1, free + 1), FREE[:free])
        world.thumbs(20, 30)  # the rated-answer trigger fires either way

        record = world.cycle()

        assert bool(record.run_id) is trains
        assert (world.trainer.calls == 1) is trains
        assert (world.state.waiting_note is None) is trains

    def test_edits_kept_aside_do_not_count_toward_the_five(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.edits(range(5, 8), HELD[:3])  # seven new edits, three kept aside: four trainable
        world.thumbs(20, 30)

        record = world.cycle()

        assert record.run_id is None
        assert "Waiting: 4/5 trainable corrections (3 more kept aside for checking)" in record.notes

    def test_five_trainable_with_edits_kept_aside_trains(self, world):
        world.edits(range(1, 6), FREE[:5])
        world.edits(range(6, 8), HELD[:2])
        world.thumbs(20, 30)

        record = world.cycle()

        assert record.run_id and len(world.ship.corrections) == 5
        assert len(world.ship.held_out) == 2

    def test_the_wait_is_cleared_once_the_loop_trains(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.thumbs(10, 35)
        world.cycle()
        assert world.state.waiting_note
        world.edit(40, FREE[10])

        record = world.cycle()

        assert record.run_id and world.state.waiting_note is None

    def test_waiting_edits_accumulate_and_are_all_trained_later(self, world):
        world.edits(range(1, 3), FREE[:2])
        world.thumbs(10, 35)
        world.cycle()
        world.edits(range(3, 6), FREE[2:5])  # three more: five in all

        record = world.cycle()

        assert record.run_id
        assert {fixed_text(n) for n in range(1, 6)} <= set(world.trainer.answers())
        assert [row["edit_id"] for row in world.ship.corrections] == [uid(n) for n in range(1, 6)]

    def test_an_edit_trigger_that_waits_says_the_same(self, world):
        world.edits(range(1, 4), FREE[:3])  # min_new_edits is 1 in this world

        record = world.cycle()

        assert record.run_id is None
        assert "Waiting: 3/5 trainable corrections · 3 new ratings" in record.notes

    def test_the_thresholds_are_untouched(self, cwd, monkeypatch):
        world = World(cwd, monkeypatch, min_new_edits=8)
        world.edits(range(1, 8), FREE[:7])

        record = world.cycle()

        assert record.run_id is None
        assert "not_enough_new_data: 7 new edits (needs 8)" in record.notes
        assert "Waiting" not in record.notes


# ---------------------------------------------------------------------------
# One iteration kept aside, then graduation, never back
# ---------------------------------------------------------------------------


def _first_iteration(world, free=5):
    world.edits(range(1, free + 1), FREE[:free])
    world.edit(6, HELD[0])
    return world.cycle()


class TestKeepAsideThenGraduation:
    def test_the_iteration_that_trains_keeps_the_edit_aside_and_checks_it(self, world):
        record = _first_iteration(world)

        assert record.run_id and world.trainer.answers() == [fixed_text(n) for n in range(1, 6)]
        assert world.ship.held_out == [{
            "edit_id": uid(6), "prompt": HELD[0], "expected": fixed_text(6),
            "original": wrong_text(6),
        }]
        (row,) = world.ledger()
        assert (row["event"], row["edit_id"], row["iteration_id"]) == (
            "kept_aside", uid(6), record.iteration_id
        )
        assert row["until"] == T0 + 106  # the newest feedback the iteration read

    def test_the_next_iteration_trains_it_and_counts_it_in_the_corrections(self, world):
        _first_iteration(world)
        world.promote()
        world.edits(range(7, 11), FREE[5:9])  # four new edits; the kept-aside one makes five

        record = world.cycle()

        assert record.run_id and fixed_text(6) in world.trainer.answers()
        assert [row["edit_id"] for row in world.ship.corrections] == [
            uid(6), uid(7), uid(8), uid(9), uid(10)
        ]
        assert world.ship.held_out == []  # nothing new is kept aside
        assert [(row["event"], row["edit_id"]) for row in world.ledger()[1:]] == [
            ("graduated", uid(6))
        ]

    def test_the_graduated_edits_are_counted_in_the_iteration_line(self, world):
        first = _first_iteration(world)
        world.promote()
        world.edits(range(7, 11), FREE[5:9])
        second = world.cycle()

        lines = {line["iteration_id"]: line for line in _read("iteration")}

        assert lines[first.iteration_id]["dataset"]["graduated"] == 0
        assert lines[second.iteration_id]["dataset"]["graduated"] == 1
        assert schema_errors("iteration", lines[second.iteration_id]) == []

    def test_it_never_goes_back(self, world):
        _first_iteration(world)
        world.promote()
        world.edits(range(7, 11), FREE[5:9])
        world.cycle()
        world.promote()
        world.edits(range(11, 16), FREE[9:14])

        third = world.cycle()

        assert third.run_id
        assert fixed_text(6) in world.trainer.answers()
        assert [row["edit_id"] for row in world.ship.corrections] == [uid(n) for n in range(11, 16)]
        assert world.ship.held_out == []
        line = _latest()[third.iteration_id]
        assert line["dataset"]["graduated"] == 0
        assert [row["edit_id"] for row in world.ledger()].count(uid(6)) == 2  # kept, graduated

    def test_a_second_edit_on_the_same_held_prompt_later_is_trained_not_kept_aside(self, world):
        _first_iteration(world)
        world.promote()
        world.edit(7, HELD[0])  # another answer to the prompt that was kept aside
        world.edits(range(8, 12), FREE[5:9])

        world.cycle()

        assert world.ship.held_out == []
        assert fixed_text(7) in world.trainer.answers()

    def test_two_edits_on_one_held_prompt_are_kept_aside_together_and_checked_once(self, world):
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])
        world.edit(7, HELD[0])
        # the same correction twice is one held-out row
        _append(FEEDBACK, [{
            "schema_version": 1, "id": uid(5999), "response_id": uid(7), "ts": iso(200),
            "kind": "edit", "edited_text": fixed_text(6), "source": "playground",
        }])

        world.cycle()

        assert [row["edit_id"] for row in world.ship.held_out] == [uid(6)]
        assert {row["edit_id"] for row in world.ledger()} == {uid(6), uid(7)}
        assert fixed_text(6) not in world.trainer.answers()
        assert fixed_text(7) not in world.trainer.answers()

    def test_a_training_run_that_failed_does_not_use_up_the_keep_aside(self, world):
        world.trainer.returncode = 1
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, HELD[0])

        failed = world.cycle()

        assert failed.run_id is None and world.ledger() == []
        world.trainer.returncode = 0
        retry = world.cycle()
        assert retry.run_id
        assert world.ship.held_out and world.ship.held_out[0]["edit_id"] == uid(6)

    def test_an_iteration_that_trains_nothing_does_not_use_up_the_keep_aside(self, world):
        world.edits(range(1, 4), FREE[:3])
        world.edit(6, HELD[0])
        assert world.cycle().run_id is None  # three trainable: it waits
        assert world.ledger() == []

        world.edits(range(7, 9), FREE[3:5])
        record = world.cycle()

        assert record.run_id
        assert [row["edit_id"] for row in world.ship.held_out] == [uid(6)]
        assert [row["event"] for row in world.ledger()] == ["kept_aside"]

    def test_a_rejected_candidate_still_used_up_the_keep_aside(self, world):
        world.ship.output = gate_output("DON'T SHIP", outcome="fail", fixed=1, broken=4)
        _first_iteration(world)
        assert world.state.canary_active is None

        world.edit(7, FREE[5])
        second = world.cycle()

        # the rejected candidate was checked against edit 6: it trains now, whatever the verdict was
        assert second.run_id and fixed_text(6) in world.trainer.answers()
        assert world.ship.held_out == []

    def test_the_ledger_survives_a_restart_of_the_watch(self, world):
        _first_iteration(world)
        write_state(world.state)
        world.state = read_state()
        world.promote()
        world.edits(range(7, 11), FREE[5:9])

        world.cycle()

        assert world.ship.held_out == [] and fixed_text(6) in world.trainer.answers()


# ---------------------------------------------------------------------------
# The lineage cursor
# ---------------------------------------------------------------------------


class TestLineageCursor:
    def test_training_does_not_move_the_cursor_only_the_attempt(self, world):
        _first_iteration(world)

        state = world.state
        assert state.harvest_cursor_ts is None
        assert state.attempt_cursor_ts == T0 + 106
        assert state.canary_cursor_ts == T0 + 106 and state.canary_active

    def test_promotion_moves_it_and_remembers_the_one_before(self, world):
        _first_iteration(world)

        world.promote()

        state = world.state
        assert state.harvest_cursor_ts == T0 + 106
        assert state.previous_harvest_cursor_ts is None
        assert state.canary_cursor_ts is None and state.canary_active is None
        assert state.production_name

    def test_a_second_promotion_remembers_the_first_cursor(self, world):
        _first_iteration(world)
        world.promote()
        world.edits(range(7, 12), FREE[5:10], at=250)

        world.cycle()
        world.promote()

        state = world.state
        assert state.harvest_cursor_ts == T0 + 250
        assert state.previous_harvest_cursor_ts == T0 + 106

    def test_a_dont_ship_leaves_its_rows_new(self, world):
        world.ship.output = gate_output("DON'T SHIP", outcome="fail", fixed=1, broken=4)
        world.edits(range(1, 6), FREE[:5])

        world.cycle()

        assert world.state.harvest_cursor_ts is None and world.state.canary_cursor_ts is None
        assert world.state.attempt_cursor_ts == T0 + 105
        # a new edit arrives: the five rejected ones are trainable again, with the new one
        world.edit(7, FREE[5])
        record = world.cycle()
        assert record.run_id
        ids = [row["edit_id"] for row in world.ship.corrections]
        assert ids == [uid(n) for n in (1, 2, 3, 4, 5, 7)]

    def test_nothing_new_since_the_rejected_run_means_no_retraining(self, world):
        world.ship.output = gate_output("DON'T SHIP", outcome="fail", fixed=1, broken=4)
        world.edits(range(1, 6), FREE[:5])
        world.cycle()

        for _ in range(3):
            world.cycle()

        assert world.trainer.calls == 1  # the rejected rows do not make it retrain by themselves

    def test_a_withdrawn_canary_leaves_its_rows_new(self, world):
        world.edits(range(1, 6), FREE[:5])
        world.cycle()
        assert world.state.canary_cursor_ts == T0 + 105

        world.fail_canary()

        state = world.state
        assert state.canary_active is None and state.canary_cursor_ts is None
        assert state.harvest_cursor_ts is None and state.production_name is None
        world.edit(7, FREE[5])
        world.cycle()
        ids = [row["edit_id"] for row in world.ship.corrections]
        assert ids == [uid(n) for n in (1, 2, 3, 4, 5, 7)]

    def test_a_failed_run_leaves_every_cursor(self, world):
        world.trainer.returncode = 1
        world.edits(range(1, 6), FREE[:5])

        world.cycle()

        state = world.state
        assert (state.harvest_cursor_ts, state.attempt_cursor_ts, state.canary_cursor_ts) == (
            None, None, None
        )

    def test_the_rows_of_a_promoted_candidate_are_not_new(self, world):
        world.edits(range(1, 6), FREE[:5])
        world.cycle()
        world.promote()

        world.edit(7, FREE[5])
        record = world.cycle()

        assert record.run_id is None
        assert "Waiting: 1/5 trainable corrections · 1 new rating" in record.notes
        assert world.state.waiting_note == "Waiting: 1/5 trainable corrections · 1 new rating"

    def test_the_trigger_counts_new_since_the_last_attempt_as_ever(self, world):
        world.ship.output = gate_output("DON'T SHIP", outcome="fail", fixed=1, broken=4)
        world.state = replace(world.state, min_new_edits=3)
        world.edits(range(1, 6), FREE[:5])
        world.cycle()

        world.edit(7, FREE[5])
        record = world.cycle()

        assert record.run_id is None
        assert "not_enough_new_data: 1 new edits (needs 3)" in record.notes

    def test_rollback_returns_to_the_cursor_of_the_previous_production(self, world, monkeypatch):
        world.edits(range(1, 6), FREE[:5])
        world.cycle()
        world.promote()
        first = world.state.harvest_cursor_ts
        world.edits(range(7, 12), FREE[5:10], at=250)
        world.cycle()
        world.promote()
        assert world.state.harvest_cursor_ts == T0 + 250
        write_state(world.state)
        monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT")  # (no server to tell)

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state()
        assert state.harvest_cursor_ts == first == T0 + 105
        assert state.previous_harvest_cursor_ts is None and state.canary_cursor_ts is None
        # and the rolled-back iteration's rows are new again
        world.state = state
        world.edit(13, FREE[11], at=300)
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
        record = world.cycle()
        assert record.run_id
        assert [row["edit_id"] for row in world.ship.corrections] == [
            uid(n) for n in (7, 8, 9, 10, 11, 13)
        ]

    def test_rollback_to_the_served_model_goes_back_to_no_cursor(self, world, monkeypatch):
        world.edits(range(1, 6), FREE[:5])
        world.cycle()
        world.promote()
        assert world.state.harvest_cursor_ts == T0 + 105
        write_state(world.state)
        monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT")

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().harvest_cursor_ts is None

    def test_rollback_of_a_loop_yaml_that_never_remembered_a_cursor_keeps_it(
        self, world, monkeypatch
    ):
        monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT")
        state = replace(
            world.state, production_name="loop-train-a", production_adapter="a",
            previous_production_name="loop-train-0", previous_production_adapter="0",
            production_version=3, previous_production_version=2, latest_version=3,
            harvest_cursor_ts=T0 + 50,
        )
        write_state(state)

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().harvest_cursor_ts == T0 + 50

    def test_rollback_withdraws_a_running_canary_with_its_cursor(self, world, monkeypatch):
        world.edits(range(1, 6), FREE[:5])
        world.cycle()
        write_state(world.state)
        monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT")

        result, _ = _cli("rollback")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state()
        assert state.canary_active is None and state.canary_cursor_ts is None
        assert state.harvest_cursor_ts is None

    def test_a_canary_started_by_hand_carries_no_cursor(self, world):
        world.state = replace(world.state, canary_cursor_ts=T0 + 9)
        write_state(world.state)

        result, _ = _cli("canary", "some-adapter", "--traffic", "10%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().canary_cursor_ts is None

    def test_promoting_a_canary_that_no_iteration_started_leaves_the_cursor(self, world):
        world.state = replace(world.state, harvest_cursor_ts=T0 + 7)
        write_state(world.state)
        _cli("canary", "some-adapter", "--traffic", "10%")
        world.state = read_state()

        world.promote()

        assert world.state.production_name == "some-adapter"
        assert world.state.harvest_cursor_ts == T0 + 7
        assert world.state.previous_harvest_cursor_ts == T0 + 7

    def test_data_until_is_what_the_iteration_read_up_to_and_idle_polls_report_production(
        self, world
    ):
        trained = _first_iteration(world)
        world.promote()
        idle = world.records[-1]

        lines = _latest()

        assert lines[trained.iteration_id]["data"]["until"] == iso(106)
        # (the promoting cycle trained nothing: it reports the data cursor, now at 106)
        assert world.state.harvest_cursor_ts == T0 + 106
        assert lines[idle.iteration_id]["data"]["until"] == iso(106)

    def test_the_cursor_never_moves_back_on_promotion(self, world):
        world.state = replace(
            world.state, harvest_cursor_ts=T0 + 900, canary_active="x", canary_cursor_ts=T0 + 5,
            canary_traffic_pct=5.0, canary_rollout_id="r", canary_adapter="x",
        )

        from soup_cli.utils.loop_holdout import promotion_cursor_changes

        assert promotion_cursor_changes(world.state)["harvest_cursor_ts"] == T0 + 900


class TestLoopStateFields:
    NEW = ["attempt_cursor_ts", "canary_cursor_ts", "previous_harvest_cursor_ts", "waiting_note"]

    def test_the_fields_come_last_in_this_order_and_default_to_none(self):
        names = list(LoopState.__dataclass_fields__)

        assert names[-len(self.NEW):] == self.NEW
        # the paraphraser fields (tests/test_loop_paraphrases.py) are the two before them
        assert names[-len(self.NEW) - 2: -len(self.NEW)] == [
            "paraphraser_endpoint", "paraphraser_model"
        ]
        state = LoopState(served_model="m", eval_suite="e", baseline="b")
        assert [getattr(state, name) for name in self.NEW] == [None] * 4

    def test_a_loop_yaml_from_before_them_loads_and_its_cursor_is_the_production_cursor(
        self, cwd
    ):
        legacy = {
            "served_model": "m", "eval_suite": "e", "baseline": "b", "status": "running",
            "harvest_cursor_ts": 1_700_000_050.5, "production_name": "loop-train-1",
            "canary_active": None, "iteration_count": 4,
        }
        os.makedirs(".soup")
        with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        state = read_state()

        assert state.harvest_cursor_ts == 1_700_000_050.5
        assert [getattr(state, name) for name in self.NEW] == [None] * 4

    def test_they_survive_a_write_and_a_read(self, cwd):
        state = LoopState(
            served_model="m", eval_suite="e", baseline="b", attempt_cursor_ts=5.0,
            canary_cursor_ts=4.5, previous_harvest_cursor_ts=3.0,
            waiting_note="Waiting: 4/5 trainable corrections · 41 new ratings",
        )
        write_state(state)

        again = read_state()

        assert (again.attempt_cursor_ts, again.canary_cursor_ts) == (5.0, 4.5)
        assert again.previous_harvest_cursor_ts == 3.0
        assert again.waiting_note == state.waiting_note

    @pytest.mark.parametrize("field", ["attempt_cursor_ts", "canary_cursor_ts",
                                       "previous_harvest_cursor_ts"])
    @pytest.mark.parametrize("value", [0, 0.0, 1_700_000_000.25])
    def test_a_cursor_is_epoch_seconds(self, field, value):
        state = LoopState(served_model="m", eval_suite="e", baseline="b", **{field: value})

        assert getattr(state, field) == value

    @pytest.mark.parametrize("field", ["attempt_cursor_ts", "canary_cursor_ts",
                                       "previous_harvest_cursor_ts"])
    @pytest.mark.parametrize(
        "value", [True, "5", [1], -1, float("nan"), float("inf"), 1e300]
    )
    def test_a_bad_cursor_is_refused_naming_the_field(self, field, value):
        with pytest.raises(ValueError, match=field):
            LoopState(served_model="m", eval_suite="e", baseline="b", **{field: value})

    @pytest.mark.parametrize("value", ["", "a\x00b", "a\nb", "a\x1b[31mb", "x" * 513, 5, True])
    def test_a_bad_waiting_note_is_refused(self, value):
        with pytest.raises((ValueError, TypeError)):
            LoopState(served_model="m", eval_suite="e", baseline="b", waiting_note=value)

    def test_a_bad_value_in_the_file_is_reported_with_its_name(self, cwd):
        bad = {"served_model": "m", "eval_suite": "e", "baseline": "b", "attempt_cursor_ts": -3}
        os.makedirs(".soup")
        with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as handle:
            json.dump(bad, handle)

        with pytest.raises(ValueError, match="attempt_cursor_ts"):
            read_state()


class TestStatus:
    def test_a_wait_is_a_row(self, cwd):
        _init_loop()
        write_state(replace(read_state(), waiting_note=NOTE_4_OF_5))

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"waiting\W+Waiting: 4/5 trainable corrections", _flat(text)), text
        assert NOTE_4_OF_5 in _flat(text)

    def test_no_wait_no_row(self, cwd):
        _init_loop()

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not re.search(r"\bwaiting\b", _flat(text)), text

    def test_the_note_is_shown_as_text_not_as_markup(self, cwd):
        _init_loop()
        write_state(replace(read_state(), waiting_note="Waiting: [bold]x[/bold]"))

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "[bold]x[/bold]" in _flat(text)

    def test_a_status_after_the_loop_waited_shows_what_the_cycle_said(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.edits(range(5, 7), HELD[:2])
        world.thumbs(10, 35)
        world.cycle()
        write_state(world.state)

        result, text = _cli("status")

        assert NOTE_4_OF_5 in _flat(text)


# ---------------------------------------------------------------------------
# The gate: the two files and the soup ship call
# ---------------------------------------------------------------------------


def ship_parse(text: str, keys: tuple) -> list:
    """The rules ``soup ship`` applies to a --corrections / --held-out file (ship_loop_gate.py)."""
    rows, seen = [], set()
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict), number
        assert set(row) == set(keys), (number, sorted(row))
        edit_id = row["edit_id"]
        assert isinstance(edit_id, str) and edit_id.strip() and len(edit_id) <= 256
        assert edit_id not in seen, f"duplicate edit_id {edit_id}"
        seen.add(edit_id)
        for key in keys[1:]:
            if key == "paraphrases":
                assert isinstance(row[key], list) and 1 <= len(row[key]) <= 20
                assert all(isinstance(p, str) and p.strip() and len(p) <= 20000 for p in row[key])
            elif key == "original":
                assert isinstance(row[key], str) and len(row[key]) <= 20000
            else:
                assert isinstance(row[key], str) and row[key].strip() and len(row[key]) <= 20000
        rows.append(row)
    assert len(rows) <= 10_000
    return rows


class TestGateFiles:
    def test_the_corrections_file_is_what_ship_expects(self, world):
        world.edits(range(1, 6), FREE[:5])

        record = world.cycle()

        argv = world.ship.argv
        path = argv[argv.index("--corrections") + 1]
        rows = ship_parse(open(path, encoding="utf-8").read(), (
            "edit_id", "expected", "original", "paraphrases"
        ))
        assert rows == [
            {
                "edit_id": uid(n), "expected": fixed_text(n), "original": wrong_text(n),
                "paraphrases": [
                    f"Reworded {n} one: {FREE[n - 1]}", f"Reworded {n} two: {FREE[n - 1]}"
                ],
            }
            for n in range(1, 6)
        ]
        assert record.run_id in os.path.basename(path)

    def test_the_held_out_file_is_what_ship_expects(self, world):
        _first_iteration(world)

        argv = world.ship.argv
        path = argv[argv.index("--held-out") + 1]
        rows = ship_parse(open(path, encoding="utf-8").read(), (
            "edit_id", "prompt", "expected", "original"
        ))
        assert rows == [{
            "edit_id": uid(6), "prompt": HELD[0], "expected": fixed_text(6),
            "original": wrong_text(6),
        }]

    def test_the_held_out_prompt_is_the_last_user_message_as_written_not_the_key(self, world):
        prompt = HELD[0].upper() + "  "
        # (a held-side prompt: the key is casefolded and stripped, the row is not)
        assert loop_holdout.is_held_out(prompt)
        world.edits(range(1, 6), FREE[:5])
        world.edit(6, prompt)

        world.cycle()

        assert world.ship.held_out[0]["prompt"] == prompt

    def test_both_files_are_passed_even_when_empty(self, world):
        world.thumbs(1, 30)

        world.cycle()

        argv = world.ship.argv
        for flag in ("--corrections", "--held-out"):
            path = argv[argv.index(flag) + 1]
            assert os.path.isfile(path) and os.path.getsize(path) == 0

    def test_the_files_are_under_verdicts_and_written_atomically(self, world, cwd):
        _first_iteration(world)

        directory = os.path.join(".soup-loops", "verdicts")
        names = sorted(os.listdir(directory))
        run_id = world.records[-1].run_id
        assert names == [
            f"{run_id}.corrections.jsonl", f"{run_id}.held-out.jsonl", f"{run_id}.json"
        ]
        assert not [name for name in os.listdir(directory) if name.endswith(".tmp")]

    def test_the_argv_is_the_gate_with_both_files(self, world):
        _first_iteration(world)

        run_id = world.records[-1].run_id
        directory = ".soup-loops/verdicts"
        assert [part.replace("\\", "/") for part in world.ship.argv] == [
            sys.executable.replace("\\", "/"), "-m", "soup_cli.cli", "ship",
            "--base", "org/base",
            "--adapter", f".soup-loops/adapters/{run_id}",
            "--task-eval", "tasks.jsonl",
            "--output", f"{directory}/{run_id}.json",
            "--gate", "loop-v2",
            "--corrections", f"{directory}/{run_id}.corrections.jsonl",
            "--held-out", f"{directory}/{run_id}.held-out.jsonl",
        ]

    def test_the_gate_is_kept_with_a_base_adapter(self, world):
        _first_iteration(world)
        world.promote()
        world.edits(range(7, 12), FREE[5:10])

        world.cycle()

        argv = world.ship.argv
        assert argv[argv.index("--base-adapter") + 1] == world.state.production_adapter
        assert argv[argv.index("--gate") + 1] == "loop-v2"
        assert "--corrections" in argv and "--held-out" in argv

    def test_a_loop_that_checks_with_the_eval_gate_does_not_call_ship(self, cwd, monkeypatch):
        world = World(cwd, monkeypatch, verify="eval_gate")
        monkeypatch.setattr(
            ls, "gate_against_baseline", lambda state, ctx: {"gate_verdict": "SKIPPED"}
        )
        world.edits(range(1, 6), FREE[:5])

        world.cycle()

        assert world.ship.calls == []

    def test_a_trained_edit_without_paraphrases_is_left_out_and_counted(self, world):
        world.edits(range(1, 5), FREE[:4])
        world.edit(5, FREE[4], paraphrases=False)

        record = world.cycle()

        assert record.run_id
        assert [row["edit_id"] for row in world.ship.corrections] == [uid(n) for n in range(1, 5)]
        assert (
            "1 of 5 trained edits have no paraphrases and were left out of the corrections check"
        ) in record.notes

    def test_a_stale_paraphrase_record_is_not_used(self, world):
        world.edits(range(1, 6), FREE[:5])
        # the record in force for edit 1 was written for another prompt
        _append(PARAPHRASES, [{
            "schema_version": 1, "response_id": uid(1),
            "prompt_hash": loop_formats.prompt_hash(
                [{"role": "user", "content": "Another prompt"}]
            ),
            "paraphrases": ["stale"], "generator": {
                "model": "m", "template": {"id": "loop-paraphrase", "version": "1"}},
            "created_at": "2026-11-06T09:00:00+00:00",
        }])

        world.cycle()

        assert uid(1) not in [row["edit_id"] for row in world.ship.corrections]

    def test_rows_ship_would_refuse_are_left_out_not_sent(self):
        good = make_edit(1, FREE[0])
        huge = EditedAnswer(uid(2), good.messages, "x", "y" * 20_001, T0)
        blank = EditedAnswer(uid(3), good.messages, "x", "   ", T0)
        paraphrases = {
            uid(1): ["a", "  ", "b"] + [f"p{n}" for n in range(30)],
            uid(2): ["a"], uid(3): ["a"],
        }

        files = loop_gate.build_gate_files([good, huge, blank], [huge, blank, good], paraphrases)

        (row,) = ship_parse(files.corrections, ("edit_id", "expected", "original", "paraphrases"))
        assert row["edit_id"] == uid(1)
        assert row["paraphrases"][:2] == ["a", "b"] and len(row["paraphrases"]) == 20
        (held,) = ship_parse(files.held_out, ("edit_id", "prompt", "expected", "original"))
        assert held["edit_id"] == uid(1)
        assert files.n_corrections == 1 and files.n_held_out == 1
        assert files.over_limit == 4 and files.without_paraphrases == 0

    def test_nothing_is_written_with_a_lone_surrogate_that_cannot_be_encoded(self):
        edit = EditedAnswer(uid(1), ({"role": "user", "content": "q"},), "o", "e\ud800", T0)

        files = loop_gate.build_gate_files([edit], [edit], {uid(1): ["p"]})

        files.corrections.encode("utf-8")  # plain ASCII: no UnicodeEncodeError
        files.held_out.encode("utf-8")
        assert json.loads(files.corrections.splitlines()[0])["expected"] == "e\ud800"

    def test_identical_held_out_edits_count_once(self):
        first = make_edit(1, HELD[0])
        second = EditedAnswer(uid(2), first.messages, "other", first.edited, T0)

        files = loop_gate.build_gate_files([], [first, second], {})

        assert files.n_held_out == 1


# ---------------------------------------------------------------------------
# What soup ship answered: the decision, the evidence, canary_decides
# ---------------------------------------------------------------------------


class TestShipResult:
    def test_ship_with_a_passing_gate_rolls_out_and_the_evidence_is_kept(self, world):
        world.ship.output = gate_output(
            fixed=5, held=(3, 1, 1), held_outcome="not_measured"
        )
        record = _first_iteration(world)

        assert record.gate_verdict == "OK" and record.shipped is True
        assert record.verdict_gate == "loop-v2"
        assert record.verdict_corrections == world.ship.output["corrections"]
        assert record.verdict_held_out == world.ship.output["held_out"]
        assert world.state.canary_active

    def test_the_line_and_the_manifest_carry_the_same_evidence(self, world):
        world.ship.output = gate_output()
        record = _first_iteration(world)
        write_iteration(record)

        line = _latest()[record.iteration_id]
        manifest = read_iteration(record.iteration_id)

        assert line["verdict"]["gate"] == "loop-v2" == manifest.verdict_gate
        assert line["verdict"]["corrections"] == world.ship.output["corrections"]
        assert line["verdict"]["corrections"] == manifest.verdict_corrections
        assert line["verdict"]["held_out"] == manifest.verdict_held_out
        assert line["verdict"]["decision"] == "SHIP" and line["verdict"]["method"] == "ship"
        assert line["outcome"] == "canary"
        assert manifest.graduated == 0 == line["dataset"]["graduated"]

    def test_every_line_the_loop_writes_validates_against_its_schema(self, world):
        _first_iteration(world)
        world.promote()
        world.edits(range(7, 11), FREE[5:9])
        world.cycle()
        world.promote()

        for kind in ("iteration", "progress", "version"):
            lines = _read(kind)
            assert lines
            for line in lines:
                assert schema_errors(kind, line) == [], (kind, line)

    def test_a_dont_ship_stops_and_says_why(self, world):
        world.ship.output = gate_output(
            "DON'T SHIP", outcome="insufficient", fixed=2, broken=1,
            reason="not enough corrections yet (3/5)",
        )
        record = _first_iteration(world)

        assert record.gate_verdict == "MAJOR" and record.shipped is False
        assert world.state.canary_active is None
        line = _latest()[record.iteration_id]
        assert line["outcome"] == "skipped"
        assert line["verdict"]["decision"] == "DON'T SHIP"
        assert line["verdict"]["reason"] == "Not enough corrections yet (3/5)"
        assert line["verdict"]["corrections"]["outcome"] == "insufficient"
        assert schema_errors("iteration", line) == []

    def test_canary_decides_deploys_the_candidate_to_a_canary(self, world):
        world.ship.output = gate_output(
            "DON'T SHIP", fixed=0, outcome="not_applicable", canary_decides=True,
            reason="no corrections to check - the canary's ratings decide",
        )
        world.thumbs(1, 30)

        record = world.cycle()

        assert record.run_id and record.gate_verdict == "MAJOR"
        assert record.shipped is True and world.state.canary_active
        assert CANARY_DECIDES_NOTE in record.notes
        line = _latest()[record.iteration_id]
        assert line["outcome"] == "canary"
        assert line["verdict"]["decision"] == "DON'T SHIP"
        assert line["verdict"]["reason"] == CANARY_DECIDES_NOTE
        assert line["verdict"]["corrections"]["outcome"] == "not_applicable"
        assert schema_errors("iteration", line) == []
        # the canary is judged like any other, and its cursor moves on promotion
        assert world.state.canary_cursor_ts is not None
        progress = [(line["stage"], line["event"]) for line in _read("progress")]
        assert ("deploy", "start") in progress and ("canary", "start") in progress

    def test_canary_decides_loads_the_adapter(self, world, monkeypatch):
        loads: list = []
        monkeypatch.setattr(
            ls, "_LOAD_POSTER", lambda endpoint, name, path: (loads.append(name), (200, ""))[1]
        )
        world.ship.output = gate_output(
            "DON'T SHIP", fixed=0, outcome="not_applicable", canary_decides=True
        )
        world.thumbs(1, 30)

        record = world.cycle()

        assert loads == [record.run_id]

    def test_a_dont_ship_without_canary_decides_loads_nothing(self, world, monkeypatch):
        loads: list = []
        monkeypatch.setattr(
            ls, "_LOAD_POSTER", lambda endpoint, name, path: (loads.append(name), (200, ""))[1]
        )
        world.ship.output = gate_output("DON'T SHIP", outcome="fail", fixed=0, broken=5)
        world.thumbs(1, 30)

        world.cycle()

        assert loads == [] and world.state.canary_active is None

    @pytest.mark.parametrize("claim", ["yes", 1, "true", None, [True]])
    def test_canary_decides_must_be_a_real_true(self, world, claim):
        world.ship.output = {**gate_output("DON'T SHIP", outcome="fail", fixed=0, broken=5),
                             "canary_decides": claim}
        world.thumbs(1, 30)

        world.cycle()

        assert world.state.canary_active is None

    def test_canary_decides_needs_the_loop_v2_gate(self, world):
        world.ship.output = {**gate_output("DON'T SHIP", canary_decides=True), "gate": "other"}
        world.thumbs(1, 30)

        world.cycle()

        assert world.state.canary_active is None

    def test_canary_decides_with_a_ship_decision_is_just_a_ship(self, world):
        world.ship.output = gate_output("SHIP", canary_decides=True)
        world.thumbs(1, 30)

        record = world.cycle()

        assert record.gate_verdict == "OK" and world.state.canary_active
        assert CANARY_DECIDES_NOTE not in record.notes

    def test_a_decision_without_the_gate_keeps_the_old_behaviour(self, world):
        world.ship.output = gate_output("DON'T SHIP", gate=None)
        world.thumbs(1, 30)

        record = world.cycle()

        assert record.gate_verdict == "MAJOR" and world.state.canary_active is None
        assert record.verdict_gate is None and record.verdict_corrections is None
        line = _latest()[record.iteration_id]
        assert "gate" not in line["verdict"] and "corrections" not in line["verdict"]

    @pytest.mark.parametrize(
        "damage",
        [
            lambda data: data["corrections"].pop("fixed"),
            lambda data: data["corrections"].update(extra=1),
            lambda data: data["corrections"].update(fixed=True),
            lambda data: data["corrections"].update(fixed=-1),
            lambda data: data["corrections"].update(outcome="great"),
            lambda data: data["corrections"].update(p_value=1.5),
            lambda data: data["corrections"].update(scoring={"method": "x"}),
            lambda data: data.update(corrections="all good"),
            lambda data: data["held_out"].update(outcome="pass!"),
            lambda data: data["held_out"].update(style={"preferred_new": 1}),
            lambda data: data.update(held_out=[1]),
        ],
    )
    def test_evidence_that_does_not_follow_the_format_is_left_out_not_trusted(
        self, world, damage
    ):
        output = gate_output()
        damage(output)
        world.ship.output = output
        world.edits(range(1, 6), FREE[:5])

        record = world.cycle()

        assert record.gate_verdict == "OK"  # the decision stands
        line = _latest()[record.iteration_id]
        assert schema_errors("iteration", line) == [], line["verdict"]

    def test_a_style_object_is_kept_when_it_follows_the_format(self, world):
        output = gate_output()
        output["held_out"]["style"] = {
            "preferred_new": 3, "preferred_old": 1, "n": 4, "p_value": 0.625,
        }
        world.ship.output = output
        record = _first_iteration(world)

        assert record.verdict_held_out["style"]["n"] == 4

    def test_no_verdict_file_is_a_skip_as_ever(self, world):
        world.ship.output = None
        world.edits(range(1, 6), FREE[:5])

        record = world.cycle()

        assert record.gate_verdict == "SKIPPED" and "soup ship wrote no verdict" in record.notes
        assert world.state.canary_active is None

    def test_files_that_cannot_be_written_skip_the_verdict(self, world, monkeypatch):
        original = ls.atomic_write_text

        def selective(text, path, **kwargs):  # (the pairs file and the train yaml still work)
            if "verdicts" in path:
                raise OSError("disk full")
            return original(text, path, **kwargs)

        monkeypatch.setattr(ls, "atomic_write_text", selective)
        world.edits(range(1, 6), FREE[:5])

        record = world.cycle()

        assert record.gate_verdict == "SKIPPED"
        assert "verdict_skipped: the gate files could not be written" in record.notes
        assert world.ship.calls == []

    def test_the_gate_words_are_the_contract_ones(self):
        assert loop_gate.GATE == "loop-v2"
        assert loop_gate.NO_CORRECTIONS_NOTE == CANARY_DECIDES_NOTE


class TestGateModule:
    def test_ship_output_without_the_gate_has_no_gate_result(self):
        assert loop_gate.read_gate_result(gate_output(gate=None), "SHIP") is None
        assert loop_gate.read_gate_result({"decision": "SHIP", "gate": "x"}, "SHIP") is None
        assert loop_gate.read_gate_result(["SHIP"], "SHIP") is None
        assert loop_gate.read_gate_result(None, "SHIP") is None

    def test_a_gate_result_carries_the_checks_and_the_reason_as_one_clean_line(self):
        data = gate_output(
            "DON'T SHIP", outcome="fail", fixed=1, broken=4,
            reason="the new version\x1b[31m did not\nlearn it  " + "x" * 5000,
        )

        found = loop_gate.read_gate_result(data, "DON'T SHIP")

        assert found.corrections == data["corrections"] and found.held_out == data["held_out"]
        assert found.canary_decides is False
        assert "\x1b" not in found.reason and "\n" not in found.reason
        assert found.reason.startswith("the new version [31m did not learn it xxx")
        assert len(found.reason) == 4096

    def test_a_reason_that_is_blank_or_not_text_is_none(self):
        for reason in ("", "  \n ", None, 5, ["x"]):
            data = {**gate_output("DON'T SHIP"), "reason": reason}
            assert loop_gate.read_gate_result(data, "DON'T SHIP").reason is None

    def test_canary_decides_needs_a_real_true_next_to_dont_ship(self):
        data = gate_output("DON'T SHIP", canary_decides=True)

        assert loop_gate.read_gate_result(data, "DON'T SHIP").canary_decides is True
        assert loop_gate.read_gate_result(data, "SHIP").canary_decides is False
        assert loop_gate.read_gate_result(data, None).canary_decides is False

    def test_the_checks_are_returned_as_copies(self):
        data = gate_output()

        found = loop_gate.read_gate_result(data, "SHIP")
        found.corrections["fixed"] = 99

        assert data["corrections"]["fixed"] == 5

    @pytest.mark.parametrize("p_value", [0, 0.5, 1, None])
    def test_p_values_at_their_edges_are_accepted(self, p_value):
        data = gate_output()
        data["corrections"]["p_value"] = p_value

        assert loop_gate.clean_corrections(data["corrections"])["p_value"] == p_value

    @pytest.mark.parametrize("p_value", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True])
    def test_p_values_out_of_range_are_refused(self, p_value):
        data = gate_output()
        data["corrections"]["p_value"] = p_value

        assert loop_gate.clean_corrections(data["corrections"]) is None

    def test_a_judge_scoring_with_its_template_is_accepted(self):
        data = gate_output()
        data["held_out"]["scoring"] = {
            "method": "reference_judge", "threshold": None,
            "template": {"id": "judge", "version": "1"},
        }

        assert loop_gate.clean_held_out(data["held_out"]) == data["held_out"]

    def test_a_bad_style_alone_is_dropped_and_the_check_beside_it_stays(self):
        data = gate_output(held=(3, 1, 1))
        data["held_out"]["style"] = {"preferred_new": "many"}

        found = loop_gate.clean_held_out(data["held_out"])

        assert "style" not in found and found["better"] == 3
        with pytest.raises(ValueError, match="style"):
            loop_gate.check_held_out(data["held_out"])


class TestIterationRecordEvidence:
    def _record(self, **kwargs) -> IterationRecord:
        return IterationRecord(
            iteration_id="iter-20261106T090000-abcd1234", started_at="2026-11-06T09:00:00+00:00",
            finished_at=None, pairs_harvested=1, run_id="r", gate_verdict="OK",
            canary_verdict=None, shipped=False, rolled_back=False, estimated_cost_usd=0.0,
            **kwargs,
        )

    def test_the_fields_are_appended_last_and_default_to_none(self):
        names = list(IterationRecord.__dataclass_fields__)

        assert names[-6:] == [
            "canary_quality", "verdict_gate", "verdict_corrections", "verdict_held_out",
            "verdict_reason", "graduated",
        ]
        record = self._record()
        assert (record.verdict_gate, record.verdict_corrections, record.graduated) == (
            None, None, None
        )

    def test_valid_evidence_is_accepted(self):
        output = gate_output()

        record = self._record(
            verdict_gate="loop-v2", verdict_corrections=output["corrections"],
            verdict_held_out=output["held_out"], verdict_reason="Because", graduated=2,
        )

        assert record.graduated == 2 and record.verdict_gate == "loop-v2"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"verdict_gate": "loop-v3"}, {"verdict_gate": 2},
            {"verdict_corrections": {"fixed": 1}}, {"verdict_corrections": "x"},
            {"verdict_held_out": {"better": 1}}, {"verdict_held_out": []},
            {"verdict_reason": 5}, {"verdict_reason": "x" * 4097}, {"verdict_reason": "a\x00"},
            {"graduated": -1}, {"graduated": True}, {"graduated": 1.5}, {"graduated": "2"},
        ],
    )
    def test_invalid_evidence_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            self._record(**kwargs)

    def test_an_old_manifest_without_the_fields_loads(self, cwd):
        record = self._record()
        write_iteration(record)
        path = os.path.join(".soup-loops", record.iteration_id, "iteration.json")
        data = json.load(open(path, encoding="utf-8"))
        for name in ("verdict_gate", "verdict_corrections", "verdict_held_out",
                     "verdict_reason", "graduated"):
            data.pop(name)
        json.dump(data, open(path, "w", encoding="utf-8"))

        assert read_iteration(record.iteration_id).verdict_gate is None


class TestSchema:
    def test_the_gate_is_an_optional_enum_in_the_verdict(self):
        schema = loop_formats.load_schema("iteration")
        verdict = schema["properties"]["verdict"]["oneOf"][0]

        assert verdict["properties"]["gate"]["enum"] == ["loop-v2"]
        assert "gate" not in verdict["required"]
