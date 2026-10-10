"""#1389: a grpo run's validation number is the held-out reward, recorded as a reward.

Since #1313 ``task: grpo`` + ``training.eval_steps`` evaluates the held-out prompts, and
the callback turned that evaluation's ``eval_loss`` into the run's ``val_loss``: the
``Val loss`` panel row, the tracker column, the ``TrainEvent`` field. On grpo that
number is TRL's policy objective at importance ratio 1 with group-normalised
advantages: near zero, often negative, unrelated to held-out quality. The number that
measures it, ``eval_reward``, reached only ``trainer_state.json``.

Now a per-task table (``VALIDATION_METRICS`` in ``utils/eval_schedule.py``) says which
evaluation entry is a task's validation number and what it is called. grpo records
``eval_reward`` as ``val_reward`` (panel row ``Val reward``, tracker column, event field),
and its ``eval_loss`` reaches no sink; every other task records ``eval_loss`` as
``val_loss`` exactly as before.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.display import TrainingDisplay
from soup_cli.utils.eval_schedule import (
    DEFAULT_VALIDATION_METRIC,
    EVAL_STEPS_UNSUPPORTED_TASKS,
    GENERATION_EVAL_TASKS,
    VALIDATION_METRICS,
    validation_metric,
)
from soup_cli.utils.sse_train_stream import TrainEvent, format_sse_frame, to_payload

_GRPO_EVAL_LOG = {"eval_loss": -0.1301, "eval_reward": 15.25, "eval_reward_std": 2.5}


class TestTheTable:
    def test_grpo_records_the_held_out_reward(self):
        spec = validation_metric("grpo")
        assert spec.log_key == "eval_reward"
        assert spec.field == "val_reward"
        assert spec.label == "Val reward"
        assert spec.direction == "higher is better"

    @pytest.mark.parametrize("task", ["sft", "dpo", "pretrain", "classifier", None, ""])
    def test_everything_else_records_the_loss(self, task):
        assert validation_metric(task) is DEFAULT_VALIDATION_METRIC
        assert DEFAULT_VALIDATION_METRIC.log_key == "eval_loss"
        assert DEFAULT_VALIDATION_METRIC.field == "val_loss"
        assert DEFAULT_VALIDATION_METRIC.direction == "lower is better"

    def test_every_generation_task_that_can_evaluate_has_a_row(self):
        """ppo and online_dpo refuse eval_steps today; the day one of them
        evaluates, it needs a row here or its policy objective becomes a loss."""
        evaluating = GENERATION_EVAL_TASKS - set(EVAL_STEPS_UNSUPPORTED_TASKS)
        assert evaluating == {"grpo"}
        assert evaluating <= set(VALIDATION_METRICS)

    def test_every_recorded_validation_field_is_a_reserved_training_metric(self):
        """#812 keeps `soup eval against --metric benchmark:<name>` off the metrics
        table's training columns; a new validation field is one of those columns, or
        `benchmark:val_reward` would compare two runs' training-time rewards as
        benchmark scores (review of #1592)."""
        from soup_cli.utils.eval_gate_hook import _RESERVED_TRAINING_METRICS

        for spec in (DEFAULT_VALIDATION_METRIC, *VALIDATION_METRICS.values()):
            assert spec.field in _RESERVED_TRAINING_METRICS, spec

    def test_every_field_in_the_table_has_a_sink_everywhere(self, tmp_path):
        """The field name is spelled in four places; the table must name a field
        that every sink knows, or the value is written to nothing."""
        db = tmp_path / "t.db"
        ExperimentTracker(db_path=str(db)).init_db()
        columns = {row[1] for row in sqlite3.connect(db).execute("PRAGMA table_info(metrics)")}
        display = TrainingDisplay(_display_config())
        for spec in (DEFAULT_VALIDATION_METRIC, *VALIDATION_METRICS.values()):
            assert spec.field in columns, spec
            assert spec.field in TrainEvent.__dataclass_fields__, spec
            assert hasattr(display, spec.field), spec


def _display_config():
    return SimpleNamespace(
        training=SimpleNamespace(epochs=1), experiment_name="x", base="org/m"
    )


class _Tracker:
    def __init__(self):
        self.rows = []

    def log_metrics(self, **row):
        self.rows.append(row)


def _callback(task, tracker, events, monkeypatch):
    from soup_cli.monitoring.callback import SoupTrainerCallback, soup_callback_kwargs
    from soup_cli.utils import train_event_buffer

    monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)
    display = TrainingDisplay(_display_config())
    callback = SoupTrainerCallback(
        display, tracker=tracker, run_id="run-1",
        **soup_callback_kwargs(SimpleNamespace(), task=task),
    )
    return callback, display


def _log(callback, step, logs):
    callback.on_log(
        args=None, state=SimpleNamespace(global_step=step, epoch=1.0), control=None, logs=logs
    )


class TestTheCallbackRoutesTheTasksMetric:
    def test_a_grpo_evaluation_reaches_every_sink_as_a_reward(self, monkeypatch):
        tracker, events = _Tracker(), []
        callback, display = _callback("grpo", tracker, events, monkeypatch)
        _log(callback, 1, dict(_GRPO_EVAL_LOG))

        assert display.val_reward == 15.25 and display.val_loss is None
        assert tracker.rows[-1]["val_reward"] == 15.25
        assert tracker.rows[-1].get("val_loss") is None
        assert events[-1].val_reward == 15.25 and events[-1].val_loss is None

    def test_the_policy_objective_is_stored_nowhere(self, monkeypatch):
        """``eval_loss`` is in the same log dict and must not leak into any sink
        under any name."""
        tracker, events = _Tracker(), []
        callback, display = _callback("grpo", tracker, events, monkeypatch)
        _log(callback, 1, dict(_GRPO_EVAL_LOG))
        recorded = {k: v for k, v in tracker.rows[-1].items() if v == -0.1301}
        assert recorded == {}
        assert -0.1301 not in to_payload(events[-1]).values()
        assert "Val loss" not in display._render().renderable

    def test_a_training_log_carries_the_reward_forward_on_the_panel_only(self, monkeypatch):
        tracker, events = _Tracker(), []
        callback, display = _callback("grpo", tracker, events, monkeypatch)
        _log(callback, 1, dict(_GRPO_EVAL_LOG))
        _log(callback, 2, {"loss": 0.5, "learning_rate": 1e-5, "grad_norm": 0.1})

        assert display.val_reward == 15.25, "sticky between evaluations"
        assert tracker.rows[-1]["val_reward"] is None, "not a measurement at step 2"
        assert events[-1].val_reward is None

    def test_the_default_task_still_records_the_loss(self, monkeypatch):
        """The control: an sft evaluation lands in ``val_loss`` as before, and the
        reward column stays empty."""
        tracker, events = _Tracker(), []
        callback, display = _callback("sft", tracker, events, monkeypatch)
        _log(callback, 1, {"eval_loss": 1.25})

        assert display.val_loss == 1.25 and display.val_reward is None
        assert tracker.rows[-1]["val_loss"] == 1.25
        assert tracker.rows[-1].get("val_reward") is None
        assert events[-1].val_loss == 1.25 and events[-1].val_reward is None

    def test_the_table_is_what_routes_it(self, monkeypatch):
        """The same grpo-shaped log through the default spec records the loss and
        ignores the reward: the routing is the task's row, not the log's keys."""
        tracker, events = _Tracker(), []
        callback, display = _callback(None, tracker, events, monkeypatch)
        _log(callback, 1, dict(_GRPO_EVAL_LOG))

        assert tracker.rows[-1]["val_loss"] == -0.1301
        assert tracker.rows[-1].get("val_reward") is None
        assert display.val_reward is None

    def test_grpo_passes_its_task_to_the_shared_kwargs(self):
        """The one trainer whose row differs must say so; the others take the
        default. Read from the source the way #802's ratchet reads the kwargs."""
        import inspect

        from soup_cli.trainer import grpo

        source = inspect.getsource(grpo)
        assert "task=self.config.task" in source

    def test_the_builder_forwards_the_task(self):
        """``build_soup_trainer_callback`` (#1546) is the only path a trainer has to
        the shared kwargs now, so the task must survive it."""
        from soup_cli.monitoring.callback import build_soup_trainer_callback

        def built(task):
            config = SimpleNamespace(training=SimpleNamespace(), eval=None)
            return build_soup_trainer_callback(
                TrainingDisplay(_display_config()), config=config, task=task
            )

        assert built("grpo")._val_metric == VALIDATION_METRICS["grpo"]
        assert built(None)._val_metric is DEFAULT_VALIDATION_METRIC

    def test_the_kwargs_carry_the_spec(self):
        from soup_cli.monitoring.callback import soup_callback_kwargs

        assert soup_callback_kwargs(SimpleNamespace())["val_metric"] is DEFAULT_VALIDATION_METRIC
        assert soup_callback_kwargs(SimpleNamespace(), task="grpo")["val_metric"] == (
            VALIDATION_METRICS["grpo"]
        )


class TestThePanel:
    def test_a_reward_row_is_labelled_as_a_reward(self):
        display = TrainingDisplay(_display_config())
        display.update(step=1, epoch=0.5, loss=0.4, lr=1e-5, val_reward=15.25)
        text = display._render().renderable
        assert "Val reward: 15.2500" in text
        assert "Val loss" not in text

    def test_a_loss_row_is_unchanged(self):
        display = TrainingDisplay(_display_config())
        display.update(step=1, epoch=0.5, loss=0.4, lr=1e-5, val_loss=1.25)
        text = display._render().renderable
        assert "Val loss: 1.2500" in text
        assert "Val reward" not in text

    def test_the_reward_row_stays_between_evaluations(self):
        """The panel's own stickiness, not the callback's carry-forward (the one
        mutation that survived the first round)."""
        display = TrainingDisplay(_display_config())
        display.update(step=1, epoch=0.5, loss=0.4, lr=1e-5, val_reward=15.25)
        display.update(step=2, epoch=0.6, loss=0.3, lr=1e-5)
        assert "Val reward: 15.2500" in display._render().renderable

    def test_nothing_is_shown_before_an_evaluation(self):
        display = TrainingDisplay(_display_config())
        display.update(step=1, epoch=0.5, loss=0.4, lr=1e-5)
        text = display._render().renderable
        assert "Val" not in text


def _legacy_db(path: Path) -> None:
    """A metrics table from before the ``val_loss`` column existed."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step INTEGER NOT NULL,
            epoch REAL, loss REAL, lr REAL, grad_norm REAL, speed REAL, gpu_mem TEXT,
            timestamp TEXT NOT NULL
        );
        INSERT INTO runs VALUES ('old', 'completed');
        INSERT INTO metrics (run_id, step, loss, timestamp) VALUES ('old', 1, 2.0, 't');
        """
    )
    conn.commit()
    conn.close()


def _columns(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(metrics)")}
    finally:
        conn.close()


class TestTheTracker:
    def test_a_fresh_db_has_the_column(self, tmp_path):
        db = tmp_path / "fresh.db"
        ExperimentTracker(db_path=str(db)).init_db()
        assert "val_reward" in _columns(db)

    def test_a_legacy_db_gains_it_and_the_old_row_reads_null(self, tmp_path):
        db = tmp_path / "experiments.db"
        _legacy_db(db)
        assert "val_reward" not in _columns(db)
        tracker = ExperimentTracker(db_path=str(db))
        tracker.init_db()
        assert "val_reward" in _columns(db)
        (row,) = tracker.get_metrics("old")
        assert row["loss"] == 2.0 and row["val_reward"] is None

    def test_the_second_migration_issues_no_alter(self, tmp_path):
        db = tmp_path / "experiments.db"
        _legacy_db(db)
        ExperimentTracker(db_path=str(db)).init_db()
        tracker = ExperimentTracker(db_path=str(db))
        conn = tracker._get_conn()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        tracker.init_db()
        conn.set_trace_callback(None)
        assert not [q for q in statements if "ALTER TABLE" in q.upper()]

    def test_the_reward_round_trips_and_the_loss_stays_null(self, tmp_path):
        tracker = ExperimentTracker(db_path=str(tmp_path / "t.db"))
        tracker.init_db()
        tracker.log_metrics("r", step=1, loss=0.5, val_reward=15.25)
        tracker.log_metrics("r", step=2, loss=0.4)
        rows = tracker.get_metrics("r")
        assert [r["val_reward"] for r in rows] == [15.25, None]
        assert [r["val_loss"] for r in rows] == [None, None]
        assert tracker.get_metric_series("r", "val_reward") == [15.25]


class TestTheEventStream:
    def test_the_reward_reaches_the_wire(self):
        event = TrainEvent(type="metric", step=3, val_reward=15.25)
        payload = to_payload(event)
        assert payload["val_reward"] == 15.25 and "val_loss" not in payload
        assert '"val_reward":15.25' in format_sse_frame(event)

    def test_absent_means_omitted(self):
        assert "val_reward" not in to_payload(TrainEvent(type="metric", step=3, loss=0.1))


# ==========================================================================
# the issue's reproduction: a real tiny grpo run on CPU, and the sft control
# ==========================================================================
def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _tiny_llama_dir(root: Path) -> str:
    """The fixture from tests/test_issue1223_evaluate_val_split.py, verbatim."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    words = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        words[word] = len(words)
    tok = Tokenizer(models.WordLevel(vocab=words, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="<pad>",
    ).save_pretrained(str(weights))
    return str(weights)


def _cfg(task: str, base: str, out: Path, **training_over):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    training = {
        "batch_size": 2, "gradient_accumulation_steps": 1, "quantization": "none",
        "epochs": 1, "logging_steps": 1, "save_steps": 1000,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if task == "grpo":
        # The issue's reward: completion length, so the held-out reward is a
        # number that varies and is visibly not the policy objective.
        training.update({"num_generations": 2, "reward_fn": "./reward.py", "eval_steps": 1})
    training.update(training_over)
    return load_config_from_string(yaml.safe_dump({
        "base": base, "task": task, "backend": "transformers", "modality": "text",
        "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
        "training": training, "output": str(out),
    }))


def _run(tmp_path, monkeypatch, task, rows):
    import importlib

    from soup_cli.utils import train_event_buffer

    monkeypatch.chdir(tmp_path)
    events = []
    monkeypatch.setattr(train_event_buffer, "push_train_event", events.append)
    base = _tiny_llama_dir(tmp_path)
    (tmp_path / "reward.py").write_text(
        "def reward_fn(completions, **kwargs):\n"
        "    return [float(len(c)) for c in completions]\n",
        encoding="utf-8",
    )
    cfg = _cfg(task, base, tmp_path / "out")
    module, cls_name = {
        "grpo": ("soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
        "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper"),
    }[task]
    wrapper = getattr(importlib.import_module(module), cls_name)(cfg, device="cpu")
    wrapper.setup({"train": rows(8), "val": rows(2)})
    tracker = ExperimentTracker(db_path=str(tmp_path / "experiments.db"))
    tracker.init_db()
    display = TrainingDisplay(cfg)
    wrapper.train(display=display, tracker=tracker, run_id="run-1")
    return wrapper, tracker.get_metrics("run-1"), display, events


def _prompt_rows(n):
    return [{"prompt": "hi"} for _ in range(n)]


def _chat_rows(n):
    return [{"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "good answer"}]} for _ in range(n)]


class TestARealGrpoRun:
    def test_the_tracker_holds_the_held_out_reward_and_no_val_loss(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper, metrics, display, events = _run(tmp_path, monkeypatch, "grpo", _prompt_rows)

        evals = [e for e in wrapper.trainer.state.log_history if "eval_reward" in e]
        assert evals, "the run never evaluated"
        # A length reward on a random model is a real, positive number, and it is
        # not the policy objective: the two series must be distinguishable, or
        # the equality below would hold for the wrong reason.
        assert all(e["eval_reward"] > 0 for e in evals), evals
        assert [e["eval_reward"] for e in evals] != [e["eval_loss"] for e in evals]
        recorded = [row["val_reward"] for row in metrics if row["val_reward"] is not None]
        assert recorded == pytest.approx([e["eval_reward"] for e in evals])
        assert all(row["val_loss"] is None for row in metrics), (
            "the policy objective was stored as a loss"
        )
        assert display.val_reward == pytest.approx(evals[-1]["eval_reward"])
        assert display.val_loss is None
        streamed = [e.val_reward for e in events if e.val_reward is not None]
        assert streamed == pytest.approx([e["eval_reward"] for e in evals])
        assert all(e.val_loss is None for e in events)

    def test_the_sft_control_is_unchanged(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper, metrics, display, events = _run(tmp_path, monkeypatch, "sft", _chat_rows)

        evals = [e for e in wrapper.trainer.state.log_history if "eval_loss" in e]
        assert evals
        recorded = [row["val_loss"] for row in metrics if row["val_loss"] is not None]
        assert recorded == pytest.approx([e["eval_loss"] for e in evals])
        assert all(row["val_reward"] is None for row in metrics)
        assert display.val_loss is not None and display.val_reward is None
        assert all(e.val_reward is None for e in events)


class TestTheDocs:
    def test_training_md_points_at_the_new_metric(self):
        docs = (Path(__file__).resolve().parents[1] / "docs" / "training.md").read_text(
            encoding="utf-8"
        )
        assert "`val_reward`" in docs and "Val reward" in docs
        assert "VALIDATION_METRICS" in docs
