"""#1223 — the validation split is evaluated, or it is not withheld.

``data.val_split: 0.1`` moved 10% of the rows out of training on every run, and
no trainer set ``eval_strategy``, whose Hugging Face default is ``"no"``. The
rows were withheld and never evaluated.

This file covers the config and loader half of the fix:

* ``training.eval_steps`` -- the new field, its bounds, and the footgun refusals
  (a value that nothing would read);
* the shared helper that decides the evaluation kwargs, the per-run loader
  config and the notice;
* the loader decision as ``soup train`` and ``soup sweep`` drive it: generation
  tasks (``grpo``, ``ppo``, ``online_dpo``) do not withhold rows unless
  ``training.eval_steps`` asks for evaluation, and everything else is unchanged.

The trainer half (every wrapper schedules evaluation, a real run records
``val_loss``) is ``test_issue1223_evaluate_val_split.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from tests.conftest import strip_ansi

_CHAT_ROWS = 20


def _plain(text: str | None) -> str:
    """ANSI-stripped and whitespace-collapsed: Rich colours and wraps CLI output."""
    return " ".join(strip_ansi(text).split())


def _load(yaml_text: str):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(yaml_text)


def _yaml(
    *,
    task: str = "sft",
    train: str = "chat.jsonl",
    backend: str = "transformers",
    data_extra: str = "",
    training_extra: str = "",
) -> str:
    extra = {
        "grpo": "  num_generations: 2\n  reward_fn: format\n",
        "ppo": "  reward_fn: format\n",
        "online_dpo": "  online_dpo_judge: 'ollama://m'\n",
        "unlearn": "  unlearn_method: npo\n",
    }.get(task, "")
    data_block = f"data:\n  train: {train}\n  max_length: 64\n{data_extra}"
    if task == "unlearn":
        data_block += "  forget_set: forget.jsonl\n"
    return (
        "base: some-org/tiny-model\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        f"{data_block}"
        "training:\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  quantization: none\n"
        f"{extra}"
        f"{training_extra}"
        "output: ./out\n"
    )


# ==========================================================================
# training.eval_steps -- the field
# ==========================================================================
class TestEvalStepsField:
    def test_unset_by_default(self):
        cfg = _load(_yaml())
        assert cfg.training.eval_steps is None

    @pytest.mark.parametrize("value", [1, 7, 500])
    def test_a_positive_int_is_accepted(self, value):
        cfg = _load(_yaml(training_extra=f"  eval_steps: {value}\n"))
        assert cfg.training.eval_steps == value

    @pytest.mark.parametrize("value", [0, -3])
    def test_zero_and_negatives_are_refused(self, value):
        with pytest.raises(ValueError, match="eval_steps") as info:
            _load(_yaml(training_extra=f"  eval_steps: {value}\n"))
        assert "greater than or equal to 1" in str(info.value)

    def test_a_yaml_bool_is_refused_rather_than_read_as_one(self):
        with pytest.raises(ValueError, match="eval_steps must be an int, not a bool"):
            _load(_yaml(training_extra="  eval_steps: true\n"))


class TestEvalStepsFootguns:
    """Every refusal names the setting, the reason, and the issue."""

    def test_nothing_to_evaluate_on_a_local_file_with_val_split_zero(self):
        with pytest.raises(ValueError) as info:
            _load(_yaml(data_extra="  val_split: 0\n", training_extra="  eval_steps: 5\n"))
        message = str(info.value)
        assert "training.eval_steps" in message
        assert "data.val_split is 0" in message
        assert "#1223" in message

    def test_nothing_to_evaluate_on_an_interleaved_list_of_local_files(self):
        with pytest.raises(ValueError, match="data.val_split is 0"):
            _load(
                _yaml(
                    train="[a.jsonl, b.jsonl]",
                    data_extra="  val_split: 0\n  interleave: concat\n",
                    training_extra="  eval_steps: 5\n",
                )
            )

    def test_a_hub_dataset_may_carry_its_own_validation_split(self):
        """The control. An HF-hub dataset's own ``validation`` split reaches the
        trainer whatever ``val_split`` says, so refusing here would refuse a
        config that evaluates. Whether the split exists is only known at load."""
        cfg = _load(
            _yaml(
                train="some-org/some-dataset",
                data_extra="  val_split: 0\n",
                training_extra="  eval_steps: 5\n",
            )
        )
        assert cfg.training.eval_steps == 5

    def test_a_pre_tokenized_cache_may_carry_its_own_validation_split(self):
        cfg = _load(
            _yaml(
                data_extra=(
                    "  val_split: 0\n  format: pre_tokenized\n  tokenized_path: ./cache\n"
                ),
                training_extra="  eval_steps: 5\n",
            )
        )
        assert cfg.training.eval_steps == 5

    def test_a_positive_val_split_is_enough(self):
        cfg = _load(_yaml(data_extra="  val_split: 0.2\n", training_extra="  eval_steps: 5\n"))
        assert cfg.training.eval_steps == 5

    @pytest.mark.parametrize(
        ("task", "reason"),
        [
            ("ppo", "never runs an evaluation pass"),
            ("online_dpo", "no evaluation step for prompt-only rows"),
            ("unlearn", "never evaluates a validation split"),
        ],
    )
    def test_tasks_without_an_evaluation_pass_refuse_it(self, task, reason):
        with pytest.raises(ValueError) as info:
            _load(_yaml(task=task, training_extra="  eval_steps: 5\n"))
        message = str(info.value)
        assert f"training.eval_steps is not supported for task='{task}'" in message
        assert reason in message
        assert "#1223" in message

    def test_grpo_accepts_it_as_the_opt_in(self):
        cfg = _load(_yaml(task="grpo", training_extra="  eval_steps: 5\n"))
        assert cfg.training.eval_steps == 5

    def test_mlx_refuses_it_because_mlx_lm_evaluates_on_its_own_cadence(self):
        with pytest.raises(ValueError) as info:
            _load(_yaml(backend="mlx", training_extra="  eval_steps: 5\n"))
        message = str(info.value)
        assert "training.eval_steps is not supported for backend='mlx'" in message
        assert "#1223" in message

    def test_mlx_without_it_still_loads(self):
        """The control for the refusal above: MLX itself is untouched."""
        cfg = _load(_yaml(backend="mlx"))
        assert cfg.training.eval_steps is None


# ==========================================================================
# the helper
# ==========================================================================
class _Rows:
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


class _Unsized:
    """An iterable dataset whose length is not known."""

    def __iter__(self):
        return iter(())


class TestTrainingEvalKwargs:
    def test_a_non_empty_split_evaluates_every_epoch_at_the_train_batch(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml())
        assert training_eval_kwargs(cfg, _Rows(2), batch_size=3) == {
            "per_device_eval_batch_size": 3,
            "eval_strategy": "epoch",
        }

    def test_eval_steps_switches_to_a_step_schedule(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml(training_extra="  eval_steps: 4\n"))
        assert training_eval_kwargs(cfg, _Rows(2), batch_size=1) == {
            "per_device_eval_batch_size": 1,
            "eval_strategy": "steps",
            "eval_steps": 4,
        }

    def test_a_generation_task_does_not_evaluate_unasked(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml(task="grpo"))
        kwargs = training_eval_kwargs(cfg, _Rows(2), batch_size=2)
        assert kwargs["eval_strategy"] == "no"
        assert kwargs["per_device_eval_batch_size"] == 2

    def test_a_generation_task_evaluates_when_asked(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml(task="grpo", training_extra="  eval_steps: 3\n"))
        kwargs = training_eval_kwargs(cfg, _Rows(2), batch_size=2)
        assert (kwargs["eval_strategy"], kwargs["eval_steps"]) == ("steps", 3)

    @pytest.mark.parametrize("dataset", [None, _Rows(0)], ids=["none", "empty"])
    def test_no_rows_means_no_evaluation(self, dataset):
        """HF raises on a step schedule with no eval_dataset, and divides by the
        row count of an empty one; neither is a reason to fail a run nobody
        asked to evaluate."""
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml())
        kwargs = training_eval_kwargs(cfg, dataset, batch_size=2)
        assert kwargs["eval_strategy"] == "no"

    def test_an_unsized_dataset_counts_as_rows(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml())
        assert training_eval_kwargs(cfg, _Unsized(), batch_size=2)["eval_strategy"] == "epoch"

    def test_eval_steps_with_no_rows_refuses_naming_the_empty_hub_split(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(
            _yaml(
                train="some-org/some-dataset",
                data_extra="  val_split: 0\n",
                training_extra="  eval_steps: 5\n",
            )
        )
        with pytest.raises(ValueError) as info:
            training_eval_kwargs(cfg, None, batch_size=2)
        message = str(info.value)
        assert "training.eval_steps=5 asks for evaluation" in message
        assert "carries no validation split of its own" in message
        assert "#1223" in message

    def test_eval_steps_with_every_row_dropped_says_so(self):
        from soup_cli.utils.eval_schedule import training_eval_kwargs

        cfg = _load(_yaml(training_extra="  eval_steps: 5\n"))
        with pytest.raises(ValueError, match="every validation row was dropped"):
            training_eval_kwargs(cfg, _Rows(0), batch_size=2)


class TestTheLoaderDecision:
    @pytest.mark.parametrize("task", ["sft", "dpo", "pretrain", "distill", "classifier"])
    def test_evaluating_tasks_keep_their_split(self, task):
        from soup_cli.utils.eval_schedule import (
            effective_val_split,
            holds_out_validation,
            loader_data_config,
        )

        training = "  teacher_model: some-org/teacher\n" if task == "distill" else ""
        training += "  num_labels: 2\n" if task == "classifier" else ""
        cfg = _load(_yaml(task=task, training_extra=training))
        assert holds_out_validation(cfg) is True
        assert loader_data_config(cfg) is cfg.data
        assert effective_val_split(cfg) == pytest.approx(0.1)

    @pytest.mark.parametrize("task", ["grpo", "ppo", "online_dpo"])
    def test_generation_tasks_withhold_nothing_by_default(self, task):
        from soup_cli.utils.eval_schedule import (
            effective_val_split,
            holds_out_validation,
            loader_data_config,
        )

        cfg = _load(_yaml(task=task))
        assert holds_out_validation(cfg) is False
        data_cfg = loader_data_config(cfg)
        assert data_cfg.val_split == 0.0
        assert effective_val_split(cfg) == 0.0
        # A copy, never the config itself: the recorded run config keeps the
        # value the user wrote.
        assert cfg.data.val_split == pytest.approx(0.1)
        assert data_cfg.train == cfg.data.train

    def test_grpo_with_eval_steps_holds_the_split_out(self):
        from soup_cli.utils.eval_schedule import holds_out_validation, loader_data_config

        cfg = _load(_yaml(task="grpo", training_extra="  eval_steps: 4\n"))
        assert holds_out_validation(cfg) is True
        assert loader_data_config(cfg) is cfg.data

    def test_val_split_zero_holds_nothing_out(self):
        from soup_cli.utils.eval_schedule import holds_out_validation, loader_data_config

        cfg = _load(_yaml(data_extra="  val_split: 0\n"))
        assert holds_out_validation(cfg) is False
        assert loader_data_config(cfg) is cfg.data

    def test_mlx_is_unchanged(self):
        """mlx-lm evaluates the split itself (#739), so it stays withheld there,
        whatever the task. MLX admits only sft today, so the generation-task
        case is built with ``model_copy`` (no re-validation): the branch has to
        hold the day MLX ships one."""
        from soup_cli.utils.eval_schedule import holds_out_validation, loader_data_config

        assert holds_out_validation(_load(_yaml(backend="mlx"))) is True
        mlx_grpo = _load(_yaml(task="grpo")).model_copy(update={"backend": "mlx"})
        assert holds_out_validation(mlx_grpo) is True
        assert loader_data_config(mlx_grpo) is mlx_grpo.data


class TestTheNotice:
    def test_grpo_names_the_ignored_setting_and_the_way_out(self):
        from soup_cli.utils.eval_schedule import validation_notice

        notice = validation_notice(_load(_yaml(task="grpo")))
        assert notice == (
            "data.val_split is ignored for task=grpo because evaluation means "
            "generation, so every row trains. Set training.eval_steps to evaluate "
            "on a held-out split."
        )

    @pytest.mark.parametrize("task", ["ppo", "online_dpo"])
    def test_tasks_that_refuse_eval_steps_are_not_told_to_set_it(self, task):
        from soup_cli.utils.eval_schedule import validation_notice

        notice = validation_notice(_load(_yaml(task=task)))
        assert notice is not None
        assert f"data.val_split is ignored for task={task}" in notice
        assert "eval_steps" not in notice

    def test_no_notice_when_nothing_is_ignored(self):
        from soup_cli.utils.eval_schedule import validation_notice

        assert validation_notice(_load(_yaml())) is None
        assert validation_notice(_load(_yaml(task="grpo", data_extra="  val_split: 0\n"))) is None
        assert (
            validation_notice(_load(_yaml(task="grpo", training_extra="  eval_steps: 4\n")))
            is None
        )

    def test_the_module_imports_without_torch(self):
        """``config/schema.py`` imports it, so it must stay import-light."""
        import subprocess
        import sys

        code = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['transformers'] = None;"
            "import soup_cli.utils.eval_schedule as s;"
            "print(sorted(s.GENERATION_EVAL_TASKS))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "['grpo', 'online_dpo', 'ppo']"


# ==========================================================================
# the loader decision, as `soup train --dry-run` drives it
# ==========================================================================
def _write_chat_rows(directory: Path, n: int = _CHAT_ROWS) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }
        for i in range(n)
    ]
    (directory / "chat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _dry_run(tmp_path, monkeypatch, yaml_text: str) -> str:
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_chat_rows(tmp_path)
    (tmp_path / "soup.yaml").write_text(yaml_text, encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run", "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


class TestDryRunCounts:
    @pytest.mark.parametrize("task", ["grpo", "ppo", "online_dpo"])
    def test_generation_tasks_train_on_every_row(self, tmp_path, monkeypatch, task):
        out = _dry_run(tmp_path, monkeypatch, _yaml(task=task, data_extra="  format: chatml\n"))
        assert f"Data OK: {_CHAT_ROWS} train samples" in out
        assert "Val:" not in out
        assert f"data.val_split is ignored for task={task}" in out

    def test_the_grpo_notice_offers_eval_steps(self, tmp_path, monkeypatch):
        out = _dry_run(tmp_path, monkeypatch, _yaml(task="grpo", data_extra="  format: chatml\n"))
        assert "Set training.eval_steps to evaluate on a held-out split." in out

    def test_grpo_with_eval_steps_holds_the_split_out(self, tmp_path, monkeypatch):
        out = _dry_run(
            tmp_path,
            monkeypatch,
            _yaml(
                task="grpo",
                data_extra="  format: chatml\n",
                training_extra="  eval_steps: 2\n",
            ),
        )
        assert f"Data OK: {_CHAT_ROWS - 2} train samples" in out
        assert "Val: 2 samples" in out
        assert "is ignored" not in out

    def test_control_sft_still_holds_the_split_out(self, tmp_path, monkeypatch):
        """Nothing changes for a task that now evaluates its split."""
        out = _dry_run(tmp_path, monkeypatch, _yaml(data_extra="  format: chatml\n"))
        assert f"Data OK: {_CHAT_ROWS - 2} train samples" in out
        assert "Val: 2 samples" in out
        assert "is ignored" not in out


class TestSweepAndCostFollowTheSameRule:
    """`soup sweep` and `soup cost` reason about the same split `soup train` uses."""

    @pytest.mark.parametrize(("task", "expected"), [("grpo", 0.0), ("sft", 0.1)])
    def test_sweep_hands_the_loader_the_run_split(self, task, expected):
        from soup_cli.commands.sweep import _run_single

        cfg = _load(_yaml(task=task, data_extra="  format: chatml\n"))
        wrapper = {
            "grpo": "soup_cli.trainer.grpo.GRPOTrainerWrapper",
            "sft": "soup_cli.trainer.sft.SFTTrainerWrapper",
        }[task]
        fake_result = {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 1,
            "duration_secs": 1.0,
            "output_dir": "./out",
            "duration": "0m",
        }
        with mock_patch(
            "soup_cli.data.loader.load_dataset", return_value={"train": [{"x": 1}]}
        ) as mock_load, mock_patch(
            "soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU")
        ), mock_patch(
            "soup_cli.utils.gpu.get_gpu_info",
            return_value={"memory_total": "0 MB", "memory_total_bytes": 0},
        ), mock_patch(
            "soup_cli.experiment.tracker.ExperimentTracker"
        ) as mock_tracker_cls, mock_patch(
            "soup_cli.monitoring.display.TrainingDisplay"
        ), mock_patch(f"{wrapper}.setup"), mock_patch(
            f"{wrapper}.train", return_value=fake_result
        ):
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-1"
            mock_tracker_cls.return_value = mock_tracker
            _run_single(cfg, {}, "sweep_1", None)

        data_cfg = mock_load.call_args.args[0]
        assert data_cfg.val_split == pytest.approx(expected)

    @pytest.mark.parametrize(("task", "rows"), [("grpo", 20), ("sft", 18)])
    def test_cost_counts_the_rows_that_train(self, tmp_path, monkeypatch, task, rows):
        from soup_cli.commands.cost import _get_dataset_size

        monkeypatch.chdir(tmp_path)
        _write_chat_rows(tmp_path)
        cfg = _load(_yaml(task=task, data_extra="  format: chatml\n"))
        assert _get_dataset_size(cfg) == (rows, False)


def test_the_plain_helper_collapses_what_rich_splits():
    """Guard for `_plain`: Rich wraps and colours per character on Linux CI."""
    raw = "\x1b[1mData\x1b[0m OK:\n  20 train\x1b[36m samples\x1b[0m"
    assert _plain(raw) == "Data OK: 20 train samples"
    assert re.search(r"\x1b", _plain(raw)) is None
