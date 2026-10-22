"""#1644 — Auto-eval after soup train follows the run's --trust-remote-code flag.

The automatic evaluation that `soup train` starts follows that run's
--trust-remote-code flag. Without the flag it stays off, as before.
There is no config field for it: the opt-in is given on the command line
only and never comes from a soup.yaml.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import EvalConfig, SoupConfig


class TestAutoEvalTrustRemoteCodeDirect:
    """Direct unit tests for _run_auto_eval_after_training."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_benchmark_only_receives_run_flag(self, monkeypatch, flag):
        import soup_cli.commands.eval as ce
        import soup_cli.commands.train as train_cmd

        benchmark_calls = []
        monkeypatch.setattr(
            ce, "benchmark", lambda **kw: benchmark_calls.append(kw)
        )
        monkeypatch.setattr(
            train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True
        )

        eval_config = EvalConfig(auto_eval=True, benchmarks=["mmlu"])
        train_cmd._run_auto_eval_after_training(
            eval_config, "out_dir", "run-1", trust_remote_code=flag
        )

        assert len(benchmark_calls) == 1
        assert benchmark_calls[0]["trust_remote_code"] is flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_custom_only_receives_run_flag(self, monkeypatch, flag):
        import soup_cli.commands.eval as ce
        import soup_cli.commands.train as train_cmd

        custom_calls = []
        monkeypatch.setattr(
            ce, "custom", lambda **kw: custom_calls.append(kw)
        )
        monkeypatch.setattr(
            train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True
        )

        eval_config = EvalConfig(auto_eval=True, custom_tasks="tasks.jsonl")
        train_cmd._run_auto_eval_after_training(
            eval_config, "out_dir", "run-1", trust_remote_code=flag
        )

        assert len(custom_calls) == 1
        assert custom_calls[0]["trust_remote_code"] is flag

    def test_flag_reaches_only_the_two_evaluation_loads(self, monkeypatch):
        """The flag must reach only the benchmark and custom loads of the trained

        output and its base, and no secondary callers (e.g. judge).
        """
        import soup_cli.commands.eval as ce
        import soup_cli.commands.train as train_cmd

        all_calls = []
        monkeypatch.setattr(
            ce, "benchmark", lambda **kw: all_calls.append(("benchmark", kw))
        )
        monkeypatch.setattr(
            ce, "custom", lambda **kw: all_calls.append(("custom", kw))
        )
        monkeypatch.setattr(
            train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True
        )

        eval_config = EvalConfig(
            auto_eval=True,
            benchmarks=["mmlu"],
            custom_tasks="tasks.jsonl",
            judge={"model": "gpt-4o-mini", "provider": "openai"},
        )
        train_cmd._run_auto_eval_after_training(
            eval_config, "out_dir", "run-1", trust_remote_code=True
        )

        assert len(all_calls) == 2
        names_and_flags = [(name, kw["trust_remote_code"]) for name, kw in all_calls]
        assert names_and_flags == [("benchmark", True), ("custom", True)]


class TestNoConfigFieldForTrustRemoteCode:
    """A soup.yaml cannot switch trust_remote_code on (top-level or under eval:)."""

    def test_schema_has_no_trust_remote_code_field(self):
        assert "trust_remote_code" not in SoupConfig.model_fields
        assert "trust_remote_code" not in EvalConfig.model_fields

    @pytest.mark.parametrize(
        ("yaml_content", "expected_key"),
        [
            (
                "base: org/model\ntask: sft\ndata: {train: data.jsonl}\ntrust_remote_code: true\n",
                "trust_remote_code",
            ),
            (
                (
                    "base: org/model\n"
                    "task: sft\n"
                    "data: {train: data.jsonl}\n"
                    "eval:\n"
                    "  auto_eval: true\n"
                    "  trust_remote_code: true\n"
                ),
                "eval.trust_remote_code",
            ),
        ],
    )
    def test_soup_yaml_refuses_trust_remote_code_key(self, yaml_content, expected_key):
        with pytest.raises(ValueError) as exc_info:
            load_config_from_string(yaml_content)
        assert f"unknown config key '{expected_key}'" in str(exc_info.value)


class TestSoupTrainCliAutoEvalTrust:
    """CLI-level tests verifying soup train passes --trust-remote-code to auto-eval."""

    @pytest.mark.parametrize(
        ("extra_args", "expected_trust"),
        [
            ([], False),
            (["--trust-remote-code"], True),
        ],
    )
    def test_soup_train_cli_passes_flag_to_auto_eval(
        self, tmp_path, monkeypatch, extra_args, expected_trust
    ):
        import soup_cli.commands.eval as ce
        import soup_cli.commands.train as train_cmd
        import soup_cli.trainer.dispatch as dispatch_mod

        # Prepare dummy files
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps({"instruction": "hello", "output": "world"}) + "\n",
            encoding="utf-8",
        )
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(
            json.dumps({"prompt": "2+2?", "expected": "4", "scoring": "exact"}) + "\n",
            encoding="utf-8",
        )
        config_file = tmp_path / "soup.yaml"
        config_file.write_text(
            "base: test-org/test-model\n"
            "task: sft\n"
            f"data: {{train: {data_file.name}, val_split: 0.0}}\n"
            "output: out\n"
            f"eval: {{auto_eval: true, benchmarks: [mmlu], custom_tasks: {tasks_file.name}}}\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "test.db"))

        # Stub trainer dispatch to avoid loading heavy model/cuda
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "total_steps": 1,
            "duration_secs": 0.5,
            "duration": "0.5s",
            "output_dir": str(tmp_path / "out"),
        }
        monkeypatch.setattr(
            dispatch_mod, "build_trainer", lambda cfg, **kw: mock_trainer
        )
        monkeypatch.setattr(
            train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True
        )

        # Record auto_eval calls
        eval_calls = []
        monkeypatch.setattr(
            ce, "benchmark", lambda **kw: eval_calls.append(("benchmark", kw))
        )
        monkeypatch.setattr(
            ce, "custom", lambda **kw: eval_calls.append(("custom", kw))
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["train", "--config", str(config_file), "--yes", *extra_args],
        )
        assert result.exit_code == 0, result.output

        assert len(eval_calls) == 2, eval_calls
        assert eval_calls[0][0] == "benchmark"
        assert eval_calls[0][1]["trust_remote_code"] is expected_trust
        assert eval_calls[1][0] == "custom"
        assert eval_calls[1][1]["trust_remote_code"] is expected_trust


def test_omitted_kwarg_stays_off(monkeypatch):
    import soup_cli.commands.eval as ce
    import soup_cli.commands.train as train_cmd

    calls = []
    monkeypatch.setattr(ce, "benchmark", lambda **kw: calls.append(("benchmark", kw)))
    monkeypatch.setattr(ce, "custom", lambda **kw: calls.append(("custom", kw)))
    monkeypatch.setattr(train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True)

    eval_config = EvalConfig(auto_eval=True, benchmarks=["mmlu"], custom_tasks="tasks.jsonl")
    train_cmd._run_auto_eval_after_training(eval_config, "out_dir", "run-1")

    assert [(name, kw["trust_remote_code"]) for name, kw in calls] == [
        ("benchmark", False),
        ("custom", False),
    ]


def test_flag_has_exactly_two_consumers():
    import soup_cli.commands.train as train_cmd

    source = textwrap.dedent(inspect.getsource(train_cmd._run_auto_eval_after_training))
    tree = ast.parse(source)

    consumers = sorted(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "trust_remote_code"
    )
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "trust_remote_code"
        and isinstance(node.ctx, ast.Load)
    ]

    assert consumers == ["benchmark", "custom"], consumers
    assert len(reads) == 2, len(reads)
    values = [
        ast.unparse(kw.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "trust_remote_code"
    ]
    assert values == ["trust_remote_code", "trust_remote_code"], values
