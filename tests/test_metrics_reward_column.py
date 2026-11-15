"""The per-step training reward of a reinforcement-style run reaches the metrics table.

``SoupTrainerCallback.on_log`` stored loss / lr / grad_norm / speed for every step and
dropped the reward that GRPO, PPO and online-DPO log, so a program that wants a reward
curve had to parse ``trainer_state.json``. Two nullable, add-only columns now carry it:
``metrics.reward`` (the batch's mean reward) and ``metrics.reward_std`` (GRPO's spread).

The contract, in the order of the sections below:

* the callback maps an explicit allowlist of trainer log keys
  (``experiment.reward_columns.REWARD_LOG_KEYS``) to the columns, and nothing else: DPO's
  implicit ``rewards/chosen`` ..., the KL-shaped ``objective/rlhf_reward`` and an
  evaluation's ``eval_reward`` are other numbers with "reward" in their name;
* NULL means "not measured": every non-RL run, every RL row that logged no reward, a
  reward that is not a number (a bool, a string), and a reward that is not finite;
* a step that logs a reward but no loss is still a row: ``loss`` is NULL (the previous
  step's loss is not carried under a step whose objective was not measured), the reward
  is set, and ``kind`` stays ``train`` (``tests/test_nullable_loss_contract.py``);
* the columns are appended after every existing column, an existing database gains them
  in place, and a database written by this version stays readable by 0.77.2's code;
* ``soup runs show`` / ``compare`` print a reward only for a run that has one, the live
  event and the HTTP metric API carry it as a nullable field, and ``benchmark:reward`` is
  not an evaluation benchmark.

Nothing here trains a model: the "trainer" is a dict of log entries, in the key set TRL
0.29.1 writes (read from its source; see the report of this change).
"""

from __future__ import annotations

import ast
import inspect
import re
import sqlite3
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.utils.eval_schedule import validation_metric

SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"
DOCS = Path(__file__).resolve().parent.parent / "docs"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()

_NAN = float("nan")
_INF = float("inf")

# The metrics table as 0.77.2 wrote it, in column order (CREATE TABLE of a fresh database).
_COLUMNS_0772 = [
    "id", "run_id", "step", "epoch", "loss", "val_loss", "val_reward", "lr", "grad_norm",
    "speed", "gpu_mem", "kind", "timestamp",
]
_NEW_COLUMNS = ["reward", "reward_std"]


def _plain_lines(text: str) -> str:
    """Strip colour escapes (they differ per OS) and collapse whitespace within each line."""
    return "\n".join(" ".join(line.split()) for line in _ANSI_RE.sub("", text).splitlines())


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


@pytest.fixture
def tracker(tmp_path):
    instance = ExperimentTracker(db_path=tmp_path / "experiments.db")
    yield instance
    instance.close()


def _start(tracker, run_id="run_reward", task="grpo"):
    return tracker.start_run(
        {"base": "test-model", "task": task}, device="cpu", device_name="CPU",
        gpu_info={}, experiment_name="reward-column", run_id=run_id,
    )


def _columns(path: Path, table: str = "metrics") -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# the logs a trainer writes (key sets read from the installed TRL 0.29.1 source)
# ---------------------------------------------------------------------------


def _grpo_log(reward, reward_std=0.5, *, loss=0.012, **extra):
    """trl GRPOTrainer.log, train mode: grpo_trainer.py:1940-1946, 2070-2071, 2262, 2302-2311
    plus the Trainer's own loss / grad_norm / learning_rate / epoch."""
    logs = {
        "loss": loss, "grad_norm": 0.4, "learning_rate": 5e-6, "epoch": 0.1,
        "reward": reward, "reward_std": reward_std,
        "rewards/accuracy_reward/mean": reward, "rewards/accuracy_reward/std": reward_std,
        "frac_reward_zero_std": 0.25, "completions/mean_length": 41.0,
        "kl": 0.001, "clip_ratio": 0.0, "entropy": 1.2,
    }
    logs.update(extra)
    return {key: value for key, value in logs.items() if value is not _OMIT}


_OMIT = object()


def _ppo_log(scores=0.8):
    """trl experimental PPOTrainer: ppo_trainer.py:884-902. No ``loss``, no
    ``learning_rate``, no ``reward``: the score is ``objective/scores`` and the lr is ``lr``."""
    return {
        "eps": 3, "objective/kl": 0.2, "objective/entropy": 4.1,
        "objective/non_score_reward": -0.05, "objective/rlhf_reward": scores - 0.05,
        "objective/scores": scores, "policy/approxkl_avg": 0.01, "policy/clipfrac_avg": 0.1,
        "loss/policy_avg": -0.02, "loss/value_avg": 0.3, "val/clipfrac_avg": 0.0,
        "policy/entropy_avg": 4.0, "val/ratio": 1.0, "val/ratio_var": 0.0,
        "val/num_eos_tokens": 12, "lr": 1e-5, "episode": 64,
    }


def _online_dpo_reward_log(scores=0.4):
    """trl experimental OnlineDPOTrainer with ``reward_funcs``: online_dpo_trainer.py:1409-1442,
    1488-1495. ``rewards/*`` are DPO's implicit rewards; ``objective/scores`` is the reward."""
    return {
        "loss": 0.69, "grad_norm": 1.1, "learning_rate": 5e-7,
        "objective/kl": 0.1, "objective/entropy": 3.0, "objective/non_score_reward": -0.02,
        "objective/rlhf_reward": scores - 0.02, "objective/scores_margin": 0.3,
        "objective/scores": scores,
        "rewards/chosen": 0.05, "rewards/rejected": -0.05, "rewards/margins": 0.1,
        "rewards/accuracies": 0.6, "logps/chosen": -40.0, "logps/rejected": -42.0,
        "val/contain_eos_token": 1.0, "beta": 0.1,
    }


def _sft_log(loss=2.5):
    return {"loss": loss, "grad_norm": 1.0, "learning_rate": 1e-4, "epoch": 0.1}


def _event(callback, step, **logs):
    callback.on_log(
        SimpleNamespace(), SimpleNamespace(global_step=step, epoch=1.0), SimpleNamespace(),
        logs=logs,
    )


@pytest.fixture
def session(tracker):
    from soup_cli.utils.train_event_buffer import reset_global_buffer

    reset_global_buffer()
    run_id = _start(tracker)
    display = MagicMock()
    callback = SoupTrainerCallback(display=display, tracker=tracker, run_id=run_id)
    yield callback, tracker, run_id, display
    reset_global_buffer()


def _rows(tracker, run_id):
    """Rows in the order they were written (``get_metrics`` orders by step)."""
    return sorted(tracker.get_metrics(run_id), key=lambda row: row["id"])


# ---------------------------------------------------------------------------
# the allowlist: one place, exact key names
# ---------------------------------------------------------------------------


def test_the_allowlist_is_exactly_these_keys():
    """Docs and the report quote this table; renaming or adding a key is a decision."""
    from soup_cli.experiment.reward_columns import REWARD_COLUMNS, REWARD_LOG_KEYS

    assert REWARD_LOG_KEYS == (
        ("reward", "reward"),
        ("objective/scores", "reward"),
        ("reward_std", "reward_std"),
    )
    assert REWARD_COLUMNS == ("reward", "reward_std")
    assert {column for _, column in REWARD_LOG_KEYS} == set(REWARD_COLUMNS)


_DECOY_KEYS = [
    # DPO-family implicit rewards: a function of the policy, not an environment score
    "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies",
    "reward_margin", "reward_accuracy", "reward_chosen", "reward_rejected",
    # per-function means and the KL-shaped objective
    "rewards/accuracy_reward/mean", "rewards/accuracy_reward/std", "reward/mean",
    "objective/rlhf_reward", "objective/non_score_reward", "objective/scores_margin",
    # the held-out evaluation: its own column (val_reward), never the training one
    "eval_reward", "eval_reward_std", "val_reward", "val_reward_std",
    # spellings that merely contain the word
    "Reward", "reward_fn_score", "mean_reward", "reward_model_score", "rewards",
    "frac_reward_zero_std",
]


@pytest.mark.parametrize("key", _DECOY_KEYS)
def test_a_key_that_merely_contains_reward_is_not_mapped(session, key):
    callback, tracker, run_id, _ = session
    _event(callback, 1, loss=1.0, learning_rate=1e-4, **{key: 7.5})
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] is None and row["reward_std"] is None, key
    # and it measures nothing by itself: no row for an event that carries only the decoy
    _event(callback, 2, **{key: 7.5})
    assert len(tracker.get_metrics(run_id)) == 1, key


def test_reward_wins_over_objective_scores_when_both_are_logged(session):
    callback, tracker, run_id, _ = session
    _event(callback, 1, loss=1.0, learning_rate=1e-4, **{"objective/scores": 9.0, "reward": 2.0})
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] == 2.0


def test_the_allowlist_has_one_reader_in_the_callback():
    """The callback goes through ``reward_from_logs``; it does not grow its own key test."""
    tree = ast.parse((SRC / "monitoring" / "callback.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "reward_from_logs" in names
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not {"reward", "reward_std", "objective/scores"} & literals, (
        "the callback must not spell the allowlisted keys itself"
    )


# ---------------------------------------------------------------------------
# what counts as a reward value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.75, 0.75), (0.0, 0.0), (-3.5, -3.5), (2, 2.0), (1e300, 1e300),
        (_NAN, None), (_INF, None), (-_INF, None),
        (True, None), (False, None), ("0.5", None), (None, None), ([0.5], None), ({}, None),
        (10 ** 400, None),   # an integer beyond float range is no finite number
    ],
)
def test_finite_reward_keeps_numbers_and_refuses_everything_else(value, expected):
    from soup_cli.experiment.reward_columns import finite_reward

    assert finite_reward(value) == expected
    if expected is not None:
        assert type(finite_reward(value)) is float


def test_finite_reward_accepts_a_numpy_scalar():
    np = pytest.importorskip("numpy")
    from soup_cli.experiment.reward_columns import finite_reward

    assert finite_reward(np.float32(0.5)) == 0.5
    assert finite_reward(np.float64(0.25)) == 0.25
    assert finite_reward(np.bool_(True)) is None


def test_reward_from_logs_says_which_columns_the_event_measured():
    from soup_cli.experiment.reward_columns import reward_from_logs

    assert reward_from_logs({}) == {}
    assert reward_from_logs(_sft_log()) == {}
    assert reward_from_logs(_grpo_log(0.75, 0.25)) == {"reward": 0.75, "reward_std": 0.25}
    assert reward_from_logs(_ppo_log(0.8)) == {"reward": 0.8}
    assert reward_from_logs(_online_dpo_reward_log(0.4)) == {"reward": 0.4}
    # present but unusable: measured (the step happened) and stored as NULL
    assert reward_from_logs({"reward": _NAN, "reward_std": _INF}) == {
        "reward": None, "reward_std": None,
    }
    # a bool or a string is not a measurement at all
    assert reward_from_logs({"reward": True, "reward_std": "0.5", "objective/scores": None}) == {}
    # a bool under the first key does not block a real number under the next
    assert reward_from_logs({"reward": True, "objective/scores": 0.5}) == {"reward": 0.5}


# ---------------------------------------------------------------------------
# the schema: appended, nullable, idempotent, readable by 0.77.2
# ---------------------------------------------------------------------------


def test_a_fresh_database_has_the_columns_after_every_existing_one(tmp_path):
    db = tmp_path / "fresh.db"
    ExperimentTracker(db_path=db).close()
    assert _columns(db) == [*_COLUMNS_0772, *_NEW_COLUMNS]


def _legacy_db(path: Path) -> None:
    """A database from before both columns: the LIVE schema with them stripped."""
    from soup_cli.experiment import tracker as tracker_module

    live = re.search(
        r'_SCHEMA_SQL = """(.*?)"""',
        Path(tracker_module.__file__).read_text(encoding="utf-8"), re.S,
    ).group(1)
    legacy = live.replace(
        "    timestamp TEXT NOT NULL,\n    reward REAL,\n    reward_std REAL\n);",
        "    timestamp TEXT NOT NULL\n);",
    )
    assert "reward_std" not in legacy and "\n    reward REAL" not in legacy, (
        "the fixture must genuinely predate both columns"
    )
    conn = sqlite3.connect(str(path))
    conn.executescript(legacy)
    conn.execute(
        "INSERT INTO runs (run_id, created_at, status, config_json, task, initial_loss,"
        " final_loss, total_steps) VALUES"
        " ('old-run', '2026-01-01T00:00:00', 'completed', '{}', 'grpo', 3.0, 1.0, 3)"
    )
    for step in (1, 2, 3):
        conn.execute(
            "INSERT INTO metrics (run_id, step, epoch, loss, lr, kind, timestamp)"
            " VALUES ('old-run', ?, ?, ?, 1e-4, 'train', '2026-01-01T00:00:00')",
            (step, step / 3, 4.0 - step),
        )
    conn.commit()
    conn.close()


def test_an_existing_database_gains_the_columns_in_place_and_old_rows_read_null(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    assert not set(_NEW_COLUMNS) & set(_columns(db))

    old = ExperimentTracker(db_path=db)
    try:
        assert _columns(db)[-2:] == _NEW_COLUMNS, "appended after the existing columns"
        rows = old.get_metrics("old-run")
        assert len(rows) == 3
        # nothing is backfilled: NULL says "written before the column existed"
        assert [(row["reward"], row["reward_std"]) for row in rows] == [(None, None)] * 3
        assert [row["loss"] for row in rows] == [3.0, 2.0, 1.0]   # existing data survives
        old.log_metrics("old-run", step=4, loss=0.5, lr=1e-4, reward=0.9, kind="train")
        assert old.get_reward_summary("old-run") == {"first": 0.9, "last": 0.9, "steps": 1}
    finally:
        old.close()


def test_the_migration_is_idempotent_and_the_second_open_issues_no_alter(tmp_path):
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    ExperimentTracker(db_path=db).close()
    again = ExperimentTracker(db_path=db)
    try:
        statements: list[str] = []
        conn = again._get_conn()
        conn.set_trace_callback(statements.append)
        again.init_db()
        conn.set_trace_callback(None)
        assert not [query for query in statements if "ALTER TABLE" in query.upper()]
        assert _columns(db)[-2:] == _NEW_COLUMNS
    finally:
        again.close()


def test_log_metrics_keeps_its_positional_signature_and_appends_the_new_arguments():
    names = list(inspect.signature(ExperimentTracker.log_metrics).parameters)
    assert names == [
        "self", "run_id", "step", "epoch", "loss", "lr", "grad_norm", "speed", "gpu_mem",
        "val_loss", "val_reward", "kind", "reward", "reward_std",
    ]


def test_a_database_written_by_this_version_is_readable_by_0772_statements(tmp_path):
    """0.77.2's reader is ``SELECT * ... ORDER BY step`` + ``dict(row)``, its writer an
    INSERT that names its twelve columns, its migration a per-column ``PRAGMA table_info``
    gate: none of them breaks on two extra nullable columns. (The real 0.77.2 module was
    run against such a file when this change was made: see the report.)"""
    db = tmp_path / "new.db"
    store = ExperimentTracker(db_path=db)
    run_id = _start(store)
    store.log_metrics(run_id, step=1, loss=0.1, lr=1e-4, reward=0.5, reward_std=0.1,
                      kind="train")
    store.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """INSERT INTO metrics
           (run_id, step, epoch, loss, val_loss, val_reward, lr, grad_norm, speed,
            gpu_mem, kind, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, 2, 0.0, 0.2, None, None, 1e-4, None, None, "", "train", "2026-01-01"),
    )
    rows = [dict(row) for row in conn.execute(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
    ).fetchall()]
    existing = {row[1] for row in conn.execute("PRAGMA table_info(metrics)").fetchall()}
    conn.close()
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["reward"] for row in rows] == [0.5, None]    # the old writer left it NULL
    assert set(_COLUMNS_0772) <= existing
    # every column 0.77.2 knows is still at its old position (a positional reader)
    assert _columns(db)[: len(_COLUMNS_0772)] == _COLUMNS_0772


# ---------------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------------


def test_a_reward_round_trips_and_a_row_without_one_reads_null(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=1, loss=0.3, reward=0.75, reward_std=0.25, kind="train")
    tracker.log_metrics(run_id, step=2, loss=0.2, kind="train")
    first, second = _rows(tracker, run_id)
    assert (first["reward"], first["reward_std"]) == (0.75, 0.25)
    assert (second["reward"], second["reward_std"]) == (None, None)


def test_a_measured_zero_and_a_negative_reward_stay_what_they_are(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=1, reward=0.0, reward_std=0.0, kind="train")
    tracker.log_metrics(run_id, step=2, reward=-2.5, kind="train")
    rows = _rows(tracker, run_id)
    assert [(row["reward"], row["reward_std"]) for row in rows] == [(0.0, 0.0), (-2.5, None)]


@pytest.mark.parametrize("bad", [True, False, "0.5", [0.5], _NAN, _INF, -_INF, 10 ** 400])
@pytest.mark.parametrize("column", ["reward", "reward_std"])
def test_the_tracker_stores_null_for_anything_that_is_not_a_finite_number(
    tracker, column, bad
):
    """Defence in depth for a writer other than the callback: SQLite would store a bool
    as 1.0, a numeric string as a number, and keeps an infinity."""
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=1, loss=0.3, kind="train", **{column: bad})
    (row,) = tracker.get_metrics(run_id)
    assert row[column] is None
    assert row["loss"] == 0.3, "the row is kept"


def test_an_integer_reward_is_stored_as_a_real(tracker):
    run_id = _start(tracker)
    tracker.log_metrics(run_id, step=1, reward=3, kind="train")
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] == 3.0 and isinstance(row["reward"], float)


def test_get_metric_series_reads_the_reward_like_the_loss(tracker):
    run_id = _start(tracker)
    for step, reward in ((1, 0.1), (2, None), (3, 0.5)):
        tracker.log_metrics(run_id, step=step, loss=1.0, reward=reward, kind="train")
    assert tracker.get_metric_series(run_id, "reward") == [0.1, 0.5]
    assert tracker.get_metric_series(run_id, "loss") == [1.0, 1.0, 1.0]


def test_get_reward_summary_is_none_without_a_reward_and_ordered_by_step(tracker):
    sft = _start(tracker, "run_sft", task="sft")
    tracker.log_metrics(sft, step=1, loss=1.0, kind="train")
    assert tracker.get_reward_summary(sft) is None
    assert tracker.get_reward_summary("run_that_does_not_exist") is None

    rl = _start(tracker, "run_rl")
    # written out of step order, with a NULL between: first/last follow the steps
    for step, reward in ((3, 0.9), (1, 0.1), (2, None), (4, 0.5)):
        tracker.log_metrics(rl, step=step, loss=1.0, reward=reward, kind="train")
    assert tracker.get_reward_summary(rl) == {"first": 0.1, "last": 0.5, "steps": 3}


def test_get_reward_summary_does_not_read_an_evaluation_benchmark_named_reward(tracker):
    """``get_metric_series`` falls back to ``eval_results``; a summary of the TRAINING
    reward must not."""
    run_id = _start(tracker, task="sft")
    tracker.log_metrics(run_id, step=1, loss=1.0, kind="train")
    tracker.save_eval_result("m", "reward", 0.9, {}, run_id=run_id)
    assert tracker.get_reward_summary(run_id) is None


# ---------------------------------------------------------------------------
# the callback
# ---------------------------------------------------------------------------


def test_a_grpo_stream_writes_one_row_per_step_with_the_reward(session):
    callback, tracker, run_id, display = session
    display.config = SimpleNamespace(task="grpo")   # the reward-collapse path runs too
    curve = [(0.10, 0.30), (0.25, 0.28), (0.40, 0.22), (0.55, 0.20)]
    for step, (reward, spread) in enumerate(curve, start=1):
        _event(callback, step * 10, **_grpo_log(reward, spread, loss=0.01 * step))
    rows = _rows(tracker, run_id)
    assert [row["step"] for row in rows] == [10, 20, 30, 40]
    assert [row["reward"] for row in rows] == [0.10, 0.25, 0.40, 0.55]
    assert [row["reward_std"] for row in rows] == [0.30, 0.28, 0.22, 0.20]
    assert [row["kind"] for row in rows] == ["train"] * 4
    # the loss contract is intact: the logged loss, the logged lr
    assert [row["loss"] for row in rows] == [0.01, 0.02, 0.03, 0.04]
    assert [row["lr"] for row in rows] == [5e-6] * 4
    assert tracker.get_metric_series(run_id, "reward") == [0.10, 0.25, 0.40, 0.55]
    assert tracker.get_reward_summary(run_id) == {"first": 0.10, "last": 0.55, "steps": 4}


def test_an_sft_stream_has_no_reward_anywhere(session):
    callback, tracker, run_id, _ = session
    for step in (1, 2, 3):
        _event(callback, step, **_sft_log(3.0 - step * 0.5))
    rows = _rows(tracker, run_id)
    assert len(rows) == 3
    assert [(row["reward"], row["reward_std"]) for row in rows] == [(None, None)] * 3
    assert tracker.get_reward_summary(run_id) is None


def test_a_ppo_step_records_objective_scores_as_the_reward_and_invents_nothing(session):
    """TRL 0.29.1's PPO logs no ``loss``, ``learning_rate`` or ``reward``: the callback
    stored no row for such a log at all. It now stores the reward, and leaves ``loss`` and
    ``lr`` NULL, not the callback's 0.0 start value for an lr nobody measured."""
    callback, tracker, run_id, display = session
    _event(callback, 1, **_ppo_log(0.8))
    _event(callback, 2, **_ppo_log(1.1))
    rows = _rows(tracker, run_id)
    assert [row["reward"] for row in rows] == [0.8, 1.1]
    assert [row["reward_std"] for row in rows] == [None, None]
    assert [row["loss"] for row in rows] == [None, None]
    assert [row["lr"] for row in rows] == [None, None]
    assert [row["kind"] for row in rows] == ["train", "train"]
    # the KL-shaped objective is not the reward
    assert 0.75 not in [row["reward"] for row in rows]
    assert display.update.called


def test_an_online_dpo_step_with_a_reward_model_records_objective_scores(session):
    callback, tracker, run_id, _ = session
    _event(callback, 5, **_online_dpo_reward_log(0.4))
    (row,) = tracker.get_metrics(run_id)
    assert (row["reward"], row["loss"], row["kind"]) == (0.4, 0.69, "train")
    # rewards/chosen, rewards/margins ... are DPO's implicit rewards and stay out
    assert row["reward_std"] is None


def test_a_pairwise_judge_run_has_no_reward_to_record(session):
    """OnlineDPOTrainer(judge=...) has no ``objective/scores``: a ranking is not a score."""
    callback, tracker, run_id, _ = session
    logs = _online_dpo_reward_log()
    for key in ("objective/scores", "objective/scores_margin", "objective/rlhf_reward"):
        del logs[key]
    _event(callback, 5, **logs)
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] is None and row["loss"] == 0.69


def test_a_step_with_a_reward_and_no_loss_is_a_row_with_a_null_loss(session):
    """The #2589 contract: NULL is "not measured". The previous step's loss is not copied
    under a step whose objective was not logged, and the row is still a ``train`` row."""
    callback, tracker, run_id, display = session
    _event(callback, 10, **_grpo_log(0.2, loss=0.5))
    _event(callback, 20, **_grpo_log(0.6, loss=_OMIT))
    first, second = _rows(tracker, run_id)
    assert (first["loss"], first["reward"]) == (0.5, 0.2)
    assert second["step"] == 20
    assert second["loss"] is None
    assert second["reward"] == 0.6
    assert second["kind"] == "train"
    assert second["lr"] == 5e-6, "the schedule's value is the event's own"
    # the panel still shows the last real loss (#541): only the record is quiet
    assert display.update.call_args.kwargs["loss"] == 0.5


def test_a_step_with_a_loss_and_no_reward_still_carries_nothing_new(session):
    """A non-RL step is exactly what it was: no reward column set, loss as logged."""
    callback, tracker, run_id, _ = session
    _event(callback, 10, **_grpo_log(0.2, loss=0.5))
    _event(callback, 20, **_sft_log(0.4))
    first, second = _rows(tracker, run_id)
    assert (second["loss"], second["reward"], second["reward_std"]) == (0.4, None, None)
    assert first["reward"] == 0.2, "no carrying forward of a reward either"


def test_an_event_without_a_reward_still_carries_the_last_loss_as_before(session):
    """Unchanged behaviour, pinned so the rule above stays scoped to reward-bearing events."""
    callback, tracker, run_id, _ = session
    _event(callback, 10, **_sft_log(2.0))
    _event(callback, 11, grad_norm=0.5)
    first, second = _rows(tracker, run_id)
    assert (second["loss"], second["grad_norm"], second["reward"]) == (2.0, 0.5, None)
    assert first["loss"] == 2.0


@pytest.mark.parametrize("bad", [_NAN, _INF, -_INF])
def test_a_non_finite_reward_is_null_and_the_row_is_kept(session, bad):
    callback, tracker, run_id, _ = session
    _event(callback, 1, **_grpo_log(bad, bad, loss=0.3))
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] is None and row["reward_std"] is None
    assert row["loss"] == 0.3 and row["kind"] == "train"


def test_a_nan_reward_std_beside_a_finite_reward_keeps_the_reward(session):
    """``rewards.std()`` of a one-sample batch is NaN in torch; the mean is still real."""
    callback, tracker, run_id, _ = session
    _event(callback, 1, **_grpo_log(0.7, _NAN))
    (row,) = tracker.get_metrics(run_id)
    assert (row["reward"], row["reward_std"]) == (0.7, None)


@pytest.mark.parametrize("bad", [True, False, "0.5", None])
def test_a_reward_that_is_not_a_number_is_not_a_measurement(session, bad):
    callback, tracker, run_id, _ = session
    _event(callback, 1, loss=1.0, learning_rate=1e-4, reward=bad)
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] is None
    _event(callback, 2, reward=bad)
    assert len(tracker.get_metrics(run_id)) == 1, "a bare non-number writes no row"


def test_a_real_zero_reward_is_a_measurement_on_its_own(session):
    callback, tracker, run_id, _ = session
    _event(callback, 3, **{"objective/scores": 0.0})
    (row,) = tracker.get_metrics(run_id)
    assert row["reward"] == 0.0 and row["loss"] is None


def test_an_evaluation_event_records_val_reward_and_no_training_reward(tracker):
    from soup_cli.utils.train_event_buffer import reset_global_buffer

    reset_global_buffer()
    run_id = _start(tracker)
    callback = SoupTrainerCallback(
        display=MagicMock(), tracker=tracker, run_id=run_id,
        val_metric=validation_metric("grpo"),
    )
    _event(callback, 10, **_grpo_log(0.3))
    # TRL prefixes every metric of an evaluation with ``eval_`` (grpo_trainer.py:2302-2309)
    _event(callback, 10, eval_loss=-0.13, eval_reward=15.25, eval_reward_std=2.5,
           **{"eval_rewards/accuracy_reward/mean": 15.25})
    train, evaluation = _rows(tracker, run_id)
    assert (train["kind"], train["reward"]) == ("train", 0.3)
    assert evaluation["kind"] == "eval"
    assert evaluation["val_reward"] == 15.25
    assert evaluation["reward"] is None and evaluation["reward_std"] is None
    reset_global_buffer()


def test_the_final_runtime_summary_row_has_no_reward(session):
    callback, tracker, run_id, _ = session
    _event(callback, 22, **_grpo_log(0.3))
    _event(callback, 22, train_runtime=3.38, train_loss=0.01, train_steps_per_second=6.5)
    train, summary = _rows(tracker, run_id)
    assert (train["kind"], train["reward"]) == ("train", 0.3)
    assert (summary["kind"], summary["reward"], summary["reward_std"]) == (
        "summary", None, None,
    )


# ---------------------------------------------------------------------------
# the live event
# ---------------------------------------------------------------------------


def test_the_live_event_carries_the_reward_as_a_nullable_field(session):
    from soup_cli.utils.sse_train_stream import to_payload
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, _, _, _ = session
    _event(callback, 10, **_grpo_log(0.25, 0.5, loss=0.1))
    _event(callback, 20, **_grpo_log(_NAN, _NAN, loss=0.2))
    first, second = get_global_buffer().snapshot()
    payload = to_payload(first)
    assert (payload["reward"], payload["reward_std"]) == (0.25, 0.5)
    assert payload["loss"] == 0.1 and payload["kind"] == "train"
    # a reward that is not finite is no value: the keys are omitted like any None field
    assert "reward" not in to_payload(second) and "reward_std" not in to_payload(second)


def test_an_sft_event_payload_has_exactly_the_keys_it_had(session):
    """Byte-identity of what the live stream sends for a run with no reward."""
    from soup_cli.utils.sse_train_stream import format_sse_frame, to_payload
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, _, _, _ = session
    _event(callback, 10, **_sft_log(2.5))
    (event,) = get_global_buffer().snapshot()
    payload = to_payload(event)
    assert set(payload) == {"type", "ts", "step", "epoch", "loss", "lr", "grad_norm", "kind"}
    assert "reward" not in format_sse_frame(event)


def test_a_reward_only_event_sends_the_events_own_null_loss(session):
    from soup_cli.utils.sse_train_stream import to_payload
    from soup_cli.utils.train_event_buffer import get_global_buffer

    callback, _, _, _ = session
    _event(callback, 10, **_grpo_log(0.2, loss=0.5))
    _event(callback, 20, **_grpo_log(0.6, loss=_OMIT))
    _, second = get_global_buffer().snapshot()
    payload = to_payload(second)
    assert payload["reward"] == 0.6
    assert "loss" not in payload, "the previous step's loss is not sent as this step's"


def test_the_train_event_rejects_nothing_new_and_defaults_the_fields_to_none():
    from soup_cli.utils.sse_train_stream import TrainEvent

    event = TrainEvent(type="metric", step=1, loss=1.0)
    assert event.reward is None and event.reward_std is None


# ---------------------------------------------------------------------------
# readers: soup runs show / compare
# ---------------------------------------------------------------------------

# The `soup runs show` / `soup runs compare` text of two SFT runs, captured from the code
# of origin/main (610e26d51) BEFORE this change, with a 96-column console and whitespace
# collapsed. Adding the reward must leave it byte-identical for a run with no reward.
_GOLDEN_SHOW_SFT = """\
╭──────────────────────────────────────── Run Details ─────────────────────────────────────────╮
│ Run ID: run_sft_a │
│ Name: golden │
│ Status: completed │
│ Date: 2026-01-02 03:04:05 │
│ │
│ Model: test-model │
│ Revision: unknown │
│ Identity: unknown │
│ Commit: unknown │
│ State: unknown │
│ Task: sft │
│ Device: CPU (cpu) │
│ GPU Memory: - │
│ │
│ Loss: 3.0000 -> 1.0000 │
│ Steps: 3 │
│ Duration: 1m │
│ Cost: — │
│ Output: - │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────── Config ───────────────────────────────────────────╮
│ { │
│ "base": "test-model", │
│ "task": "sft", │
│ "training": { │
│ "epochs": 1, │
│ "lr": 0.0001 │
│ } │
│ } │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯"""

_GOLDEN_COMPARE_SFT = """\
Run Comparison
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric ┃ run_sft_a ┃ run_sft_b ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Name │ golden │ golden │
│ Model │ test-model │ test-model │
│ Task │ sft │ sft │
│ Device │ CPU │ CPU │
│ Status │ completed │ completed │
│ Code commit │ unknown │ unknown │
│ Code state │ unknown │ unknown │
│ Dataset SHA-256 │ unknown │ unknown │
│ Initial Loss │ 3.0000 │ 3.0000 │
│ Final Loss │ 1.0000 │ 0.5000 │
│ Steps │ 3 │ 3 │
│ Duration │ 2m │ 2m │
│ Epochs │ 1 │ 1 │
│ Learning Rate │ 0.0001 │ 0.0001 │
│ Batch Size │ - │ - │
│ Quantization │ - │ - │
│ LoRA r │ - │ - │
│ LoRA alpha │ - │ - │
└─────────────────┴────────────┴────────────┘
No config differences.
No stored evaluation scores."""


def _golden_run(store, run_id, final, *, task="sft", rewards=None):
    """A completed run with fixed fields; ``rewards`` adds a reward to each of its 3 steps."""
    store.start_run(
        {"base": "test-model", "task": task, "training": {"epochs": 1, "lr": 1e-4}},
        device="cpu", device_name="CPU", gpu_info={}, experiment_name="golden", run_id=run_id,
    )
    for index, (step, loss) in enumerate(((1, 3.0), (2, 2.0), (3, final))):
        reward = rewards[index] if rewards else None
        store.log_metrics(run_id, step=step, epoch=step / 3, loss=loss, lr=1e-4,
                          grad_norm=0.5, speed=None, reward=reward, kind="train")
    store.finish_run(run_id, 3.0, final, 3, 90.0, "", loss_summary_kind="per_step")
    conn = store._get_conn()
    conn.execute("UPDATE runs SET created_at = '2026-01-02T03:04:05' WHERE run_id = ?", (run_id,))
    conn.commit()


def _run_cli(args, monkeypatch) -> str:
    import soup_cli.commands.runs as runs_module

    output = StringIO()
    monkeypatch.setattr(
        runs_module, "console",
        Console(file=output, width=96, force_terminal=False, color_system=None,
                legacy_windows=False),
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain_lines(output.getvalue())


def _same_text(actual: str, golden: str) -> None:
    """Compare two small texts line by line, printing only the first differing line."""
    actual_lines, golden_lines = actual.splitlines(), golden.splitlines()
    for index, (have, want) in enumerate(zip(actual_lines, golden_lines)):
        assert have == want, f"line {index + 1} differs: {have!r} != {want!r}"
    assert len(actual_lines) == len(golden_lines), (
        f"{len(actual_lines)} lines, golden has {len(golden_lines)}"
    )


@pytest.fixture
def sft_runs(tracker):
    _golden_run(tracker, "run_sft_a", 1.0)
    _golden_run(tracker, "run_sft_b", 0.5)
    tracker.close()


def test_soup_runs_show_of_an_sft_run_is_byte_identical_to_main(sft_runs, monkeypatch):
    text = _run_cli(["runs", "show", "run_sft_a", "--no-plot"], monkeypatch)
    _same_text(text, _GOLDEN_SHOW_SFT)
    assert "Reward" not in text


def test_soup_runs_compare_of_two_sft_runs_is_byte_identical_to_main(sft_runs, monkeypatch):
    text = _run_cli(["runs", "compare", "run_sft_a", "run_sft_b"], monkeypatch)
    _same_text(text, _GOLDEN_COMPARE_SFT)
    assert "Reward" not in text


def test_soup_runs_show_prints_the_reward_under_the_loss_for_a_run_that_has_one(
    tracker, monkeypatch
):
    _golden_run(tracker, "run_grpo", 1.0, task="grpo", rewards=[0.1, 0.4, 0.8])
    tracker.close()
    text = _run_cli(["runs", "show", "run_grpo", "--no-plot"], monkeypatch)
    lines = text.splitlines()
    loss = next(i for i, line in enumerate(lines) if "Loss: 3.0000 -> 1.0000" in line)
    assert lines[loss + 1] == "│ Reward: 0.1000 -> 0.8000 (3 steps) │", lines[loss : loss + 3]
    assert lines[loss + 2] == "│ Steps: 3 │"


def test_soup_runs_show_prints_a_single_reward_once(tracker, monkeypatch):
    _golden_run(tracker, "run_one", 1.0, task="grpo", rewards=[None, 0.5, None])
    tracker.close()
    text = _run_cli(["runs", "show", "run_one", "--no-plot"], monkeypatch)
    assert "│ Reward: 0.5000 (1 step) │" in text.splitlines()


def test_soup_runs_compare_adds_reward_rows_when_either_run_has_a_reward(tracker, monkeypatch):
    _golden_run(tracker, "run_grpo", 1.0, task="grpo", rewards=[0.1, 0.4, 0.8])
    _golden_run(tracker, "run_sft_b", 0.5)
    tracker.close()
    lines = _run_cli(["runs", "compare", "run_grpo", "run_sft_b"], monkeypatch).splitlines()
    final = next(i for i, line in enumerate(lines) if line.startswith("│ Final Loss"))
    assert lines[final + 1] == "│ Initial Reward │ 0.1000 │ - │"
    assert lines[final + 2] == "│ Final Reward │ 0.8000 │ - │"
    assert lines[final + 3].startswith("│ Steps")


# ---------------------------------------------------------------------------
# readers: the HTTP metric API and the evaluation gate
# ---------------------------------------------------------------------------


def test_the_http_metric_api_returns_reward_and_reward_std_nullable(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from soup_cli.ui.app import create_app, get_auth_token
    from tests.ui_helpers import ui_client

    db = tmp_path / "http.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db))
    store = ExperimentTracker(db_path=db)
    run_id = _start(store, "run_http_reward")
    store.log_metrics(run_id, step=1, loss=0.1, lr=1e-4, reward=0.5, reward_std=0.25,
                      kind="train")
    store.log_metrics(run_id, step=2, loss=0.1, lr=1e-4, kind="train")
    store.log_metrics(run_id, step=3, loss=0.1, lr=1e-4, reward=_NAN, kind="train")
    store.close()

    client = ui_client(create_app())
    headers = {"Authorization": f"Bearer {get_auth_token()}"}
    metrics = client.get(f"/api/runs/{run_id}/metrics", headers=headers).json()["metrics"]
    assert [(row["reward"], row["reward_std"]) for row in metrics] == [
        (0.5, 0.25), (None, None), (None, None),
    ]


def test_benchmark_reward_is_a_reserved_training_metric_not_an_eval_benchmark():
    """``soup eval against --metric benchmark:reward`` would otherwise gate two runs on their
    TRAINING rewards as if they were benchmark scores (#812, as for ``val_reward``)."""
    from soup_cli.utils.eval_gate_hook import resolve_gate_metric

    for name in ("reward", "reward_std"):
        with pytest.raises(ValueError, match=f"benchmark:{name} is a reserved training metric"):
            resolve_gate_metric(f"benchmark:{name}")


def test_every_per_step_column_of_the_metrics_table_is_a_reserved_training_metric(tmp_path):
    """The set is documented as "every column of the metrics table a run writes": a column
    added later must be added to it, and this is the test that notices."""
    from soup_cli.utils.eval_gate_hook import _RESERVED_TRAINING_METRICS

    db = tmp_path / "cols.db"
    ExperimentTracker(db_path=db).close()
    structural = {"id", "run_id", "step", "timestamp", "kind"}
    missing = {name for name in _columns(db) if name not in structural} - _RESERVED_TRAINING_METRICS
    # `epoch` is in the set too, so everything per-step is covered
    assert not missing, f"not reserved: {sorted(missing)}"


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------


def test_the_docs_say_what_the_columns_are_and_what_null_means():
    from soup_cli.experiment.reward_columns import REWARD_LOG_KEYS

    docs = (DOCS / "backends-and-ops.md").read_text(encoding="utf-8")
    for needle in ("`metrics.reward`", "`metrics.reward_std`", "`benchmark:reward`"):
        assert needle in docs, f"{needle} is not documented in docs/backends-and-ops.md"
    for key, _ in REWARD_LOG_KEYS:
        assert f"`{key}`" in docs, f"the logged key {key} is not documented"
    section = docs[docs.index("`metrics.reward`"):]
    section = section[: section.index("\n\n")]
    assert "NULL" in section
    # the decoys the docs promise are not recorded are named, so a reader does not look
    # for them in the column
    for decoy in ("rewards/chosen", "objective/rlhf_reward", "eval_reward"):
        assert decoy in docs


def test_the_module_docstring_matches_the_allowlist():
    from soup_cli.experiment import reward_columns

    doc = reward_columns.__doc__ or ""
    for key, column in reward_columns.REWARD_LOG_KEYS:
        assert f"``{key}``" in doc and f"``{column}``" in doc
