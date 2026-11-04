"""Tests for eval command (basic CLI validation, not full eval runs)."""

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_eval_benchmark_missing_model():
    """soup eval benchmark with nonexistent model path should fail."""
    result = runner.invoke(
        app, ["eval", "benchmark", "--model", "nonexistent_path", "--benchmarks", "mmlu"],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_eval_help():
    """soup eval --help should show subcommands."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.output.lower()
    assert "custom" in result.output.lower()
    assert "judge" in result.output.lower()
    assert "leaderboard" in result.output.lower()


# #2003: scoring bundles must stop before the first LM-consumer operation.
@pytest.mark.parametrize("artifact_kind", ["root", "payload", "recorded-base"])
@pytest.mark.parametrize(
    "function_name, argument, first_work_line, list_input",
    [
        ("benchmark", "model", "model_path = Path(model)", False),
        ("custom", "model", "from soup_cli.eval.custom import load_eval_tasks", False),
        ("judge", "target", "from soup_cli.eval.judge import (", False),
        ("human", "model_a", "from soup_cli.eval.human import (", False),
        ("human", "model_b", "from soup_cli.eval.human import (", False),
        ("gate_cmd", "model", "if not 0.0 <= regression_threshold <= 1.0:", False),
        ("quant_check_cmd", "before", "from soup_cli.eval.quant_check import (", False),
        ("quant_check_cmd", "after", "from soup_cli.eval.quant_check import (", False),
    ],
)
def test_reward_bundle_refused_before_work(
    tmp_path,
    function_name,
    argument,
    first_work_line,
    list_input,
    artifact_kind,
):
    from importlib import import_module

    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / "reward"
    root.mkdir()
    if artifact_kind == "payload":
        (root / "reward.safetensors").write_bytes(b"broken")
    else:
        (root / "reward_model.json").write_text("broken")
    if artifact_kind == "recorded-base":
        import json

        adapter = tmp_path / "lm-adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(root)})
        )
        root = adapter
    function = getattr(import_module("soup_cli.commands.eval"), function_name)
    value = [str(root)] if list_input else str(root)
    kwargs = {}
    kwargs[argument] = value
    assert_reward_guard(function, kwargs, first_work_line)


def test_reward_resolved_benchmark_base_refused(tmp_path, monkeypatch):
    import json

    from soup_cli.commands.eval import benchmark
    from soup_cli.utils import mlx_reward_bundle
    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / 'reward'
    root.mkdir()
    (root / 'reward_model.json').write_text('broken')
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text(
        json.dumps({'base_model_name_or_path': str(root)})
    )
    # Isolate the post-resolution seam from the independent preflight check.
    monkeypatch.setattr(mlx_reward_bundle, 'read_json', lambda _: {})
    assert_reward_guard(benchmark, {'model': str(adapter)}, 'if base_model:')


@pytest.mark.parametrize('field', ['output', 'base'])
def test_reward_auto_eval_paths_refused_before_work(tmp_path, monkeypatch, field):
    from types import SimpleNamespace

    from soup_cli.commands.eval import auto
    from soup_cli.config import loader
    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / 'reward'
    root.mkdir()
    (root / 'reward_model.json').write_text('broken')
    config = tmp_path / 'soup.yaml'
    config.write_text('base: ordinary')
    values = {'base': 'ordinary-base', 'output': 'ordinary-output'}
    values[field] = str(root)
    monkeypatch.setattr(loader, 'load_config', lambda _: SimpleNamespace(**values))
    assert_reward_guard(auto, {'config': str(config)}, 'output_dir = Path(soup_config.output)')
