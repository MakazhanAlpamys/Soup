"""`soup why` and `soup sweep` read the validation loss Soup records after every epoch.

The fixture holds real tracker records: seed 0 of an epochs 1/2/3 sweep on a Kaggle T4
(`tests/fixtures/val_loss_t4_sweep.json`, provenance in its ``source`` field;
benchmarks/gate-why-sweep-val-loss-t4.md has the full measurement). Its validation losses:

* 1 epoch: 0.1648
* 2 epochs: 0.1701, 0.1764 (+3.7%, under the 5% rule)
* 3 epochs: 0.1745, 0.1814, 0.2250 (+29%)

On main, `soup why` reads "No anomalies detected" on all three, and `soup sweep` ranks the
3-epoch arm first, because its final training loss is the lowest.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from rich.console import Console

import soup_cli.commands.sweep as sweep_mod
from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import make_cli_runner, strip_ansi

pytestmark = pytest.mark.integration  # real SQLite tracker and the real CLI

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "val_loss_t4_sweep.json").read_text(encoding="utf-8")
)
RUNS = {entry["run"]["experiment_name"]: entry for entry in FIXTURE["runs"]}
ONE, TWO, THREE = RUNS["s0_1"], RUNS["s0_2"], RUNS["s0_3"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    # Wide enough that no finding wraps inside its table cell. The consoles are module
    # level, so COLUMNS would only count if it were set before the first import.
    monkeypatch.setattr("soup_cli.commands.why.console", Console(width=500))
    monkeypatch.setattr(sweep_mod, "console", Console(width=500))
    with closing(ExperimentTracker()):  # creates the schema
        pass
    return tmp_path / "runs.db"


def _load(db: Path, entry: dict) -> str:
    """Write one recorded run and its metric rows into the tracker database, as recorded."""
    run = entry["run"]
    with closing(sqlite3.connect(db)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
        kept = {key: value for key, value in run.items() if key in columns}
        conn.execute(
            f"INSERT INTO runs ({', '.join(kept)}) VALUES ({', '.join('?' * len(kept))})",
            list(kept.values()),
        )
        for row in entry["metrics"]:
            conn.execute(
                f"INSERT INTO metrics (run_id, {', '.join(row)}) "
                f"VALUES (?, {', '.join('?' * len(row))})",
                [run["run_id"], *row.values()],
            )
        conn.commit()
    return run["run_id"]


def _why(run_id: str) -> str:
    result = make_cli_runner().invoke(app, ["why", run_id])
    assert result.exit_code == 0, result.output
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


# --- soup why -----------------------------------------------------------------------------


def test_why_warns_when_validation_loss_rose_after_epoch_one(db):
    text = _why(_load(db, THREE))

    assert "val_loss_rose" in text
    assert "Validation loss was lowest at epoch 1 (0.1745) and ended 29% higher (0.2250)." in text
    # 11:25:51 (epoch 1 evaluation) to 12:31:19 (epoch 3 evaluation) in the record
    assert "The training after epoch 1 took 65 min." in text
    assert "Next time set training.epochs: 1." in text
    assert "The saved adapter is from the end of training, not from epoch 1." in text
    assert "No anomalies detected" not in text


def test_why_is_silent_on_the_one_epoch_run(db):
    text = _why(_load(db, ONE))

    assert "val_loss_rose" not in text
    assert "No anomalies detected" in text


def test_why_is_silent_on_a_rise_under_five_percent(db):
    """The 2-epoch run's validation loss rose 3.7%: under the rule, so no finding."""
    text = _why(_load(db, TWO))

    assert "val_loss_rose" not in text
    assert "No anomalies detected" in text


def test_why_leaves_the_recorded_run_unchanged(db):
    run_id = _load(db, THREE)
    with closing(ExperimentTracker()) as tracker:
        before = (tracker.get_run(run_id), tracker.get_metrics(run_id))

    _why(run_id)

    with closing(ExperimentTracker()) as tracker:
        assert (tracker.get_run(run_id), tracker.get_metrics(run_id)) == before


# --- soup sweep ---------------------------------------------------------------------------


@pytest.fixture
def sweep(db, tmp_path, monkeypatch):
    """Run the real `soup sweep` over the recorded arms.

    `_run_single` (config, data, training, tracker bookkeeping) is replaced by a stub that
    writes the arm's recorded tracker rows and returns the recorded final training loss and
    last validation loss. The sweep loop, the ranking and the summary are the real code.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: Qwen/Qwen2.5-1.5B-Instruct\ndata:\n  train: ./d.jsonl\n", encoding="utf-8"
    )

    def run(arms: list[dict], param: str = "training.epochs=1,2,3"):
        recorded = iter(arms)

        def replay(cfg, combo, name, path):
            entry = next(recorded)
            val_losses = [row["val_loss"] for row in entry["metrics"]
                          if row["val_loss"] is not None]
            return {"run_id": _load(db, entry), "final_loss": entry["run"]["final_loss"],
                    "val_loss": val_losses[-1] if val_losses else None, "duration": "-"}

        monkeypatch.setattr(sweep_mod, "_run_single", replay)
        result = make_cli_runner().invoke(
            app, ["sweep", "--config", "soup.yaml", "--param", param,
                  "--name", "s0", "--yes"],
        )
        assert result.exit_code == 0, result.output
        return " ".join(strip_ansi(result.output).replace("│", " ").split())

    return run


def test_last_val_loss_reads_the_final_evaluation(db):
    run_id = _load(db, THREE)
    with closing(ExperimentTracker()) as tracker:
        assert sweep_mod._last_val_loss(tracker, run_id) == pytest.approx(0.22501541674137115)


def test_sweep_ranks_the_recorded_arms_by_validation_loss(sweep):
    text = sweep([ONE, TWO, THREE])

    assert "Ranked by: last validation loss (lower is better)." in text
    assert "Best run: s0_1 (val loss: 0.1648)" in text
    assert text.index("s0_1 1 0.1364 0.1648 *") < text.index("s0_2 2 0.1059 0.1764")
    assert text.index("s0_2 2 0.1059 0.1764") < text.index("s0_3 3 0.0689 0.2250")


def test_sweep_falls_back_to_training_loss_when_an_arm_has_no_validation_loss(sweep):
    """The 3-epoch arm with its evaluation rows removed: the sweep cannot compare validation
    losses across all arms, so it ranks by final training loss and says so."""
    no_eval = {"run": THREE["run"],
               "metrics": [row for row in THREE["metrics"] if row["val_loss"] is None]}
    text = sweep([ONE, TWO, no_eval])

    assert "Ranked by: final training loss. Not every ranked run recorded a validation " \
           "loss." in text
    assert "Best run: s0_3 (loss: 0.0689)" in text


@pytest.mark.parametrize(
    ("param", "changed"),
    [("val_split=0.05,0.1,0.2", "data.val_split"),
     ("data.max_length=512,1024,2048", "data.max_length")],
)
def test_sweep_ranks_by_training_loss_when_it_changes_a_data_setting(sweep, param, changed):
    """Arms with different data settings hold out different rows, or tokenize them
    differently, so their validation losses are not comparable. Every arm has one here,
    and the sweep still ranks by final training loss and says why."""
    text = sweep([ONE, TWO, THREE], param)

    assert f"Ranked by: final training loss. The sweep changes {changed}, so the runs " \
           "measured validation loss on different rows." in text
    assert "Best run: s0_3 (loss: 0.0689)" in text


@pytest.mark.parametrize(
    "param",
    ["data.replay=./a.jsonl,./b.jsonl,./c.jsonl", "data.replay_ratio=0.1,0.2,0.3",
     "data.replay_seed=0,1,2"],
)
def test_sweep_over_a_replay_setting_keeps_validation_loss(sweep, param):
    """Replay rows are mixed into the training split only, after the split, so every arm
    holds out the same validation rows and the sweep still ranks by validation loss."""
    Path("soup.yaml").write_text(
        "base: Qwen/Qwen2.5-1.5B-Instruct\ndata:\n  train: ./d.jsonl\n  replay: ./a.jsonl\n",
        encoding="utf-8",
    )
    text = sweep([ONE, TWO, THREE], param)

    assert "Ranked by: last validation loss (lower is better)." in text
    assert "Best run: s0_1 (val loss: 0.1648)" in text


def test_a_diverged_arm_is_never_ranked_and_does_not_switch_the_metric():
    """#944 holds on validation loss too: an arm whose training loss diverged ranks last even
    with the lowest validation loss, and having none does not send the sweep back to
    training loss. Values are the seed-0 arms', with one diverged arm added."""
    console = Console(record=True, width=200)

    def arm(name, final, val):
        return {"name": name, "params": {}, "run_id": name, "final_loss": final,
                "val_loss": val, "duration": "-", "status": "completed"}

    for diverged in (arm("nan-run", float("nan"), 0.1), arm("inf-run", float("inf"), None)):
        results = [arm("s0_3", 0.0689, 0.2250), diverged, arm("s0_1", 0.1364, 0.1648)]
        assert sweep_mod._ranking_metric(results) == "val_loss"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sweep_mod, "console", console)
            sweep_mod._display_summary(results, {})
        text = console.export_text()
        assert "Best run: s0_1 (val loss: 0.1648)" in text
        assert text.index("s0_1") < text.index("s0_3") < text.index(diverged["name"])
