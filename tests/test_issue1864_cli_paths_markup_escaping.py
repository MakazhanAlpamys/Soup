"""Tests for Issue #1864: model and output paths printed as raw Rich markup
in chat, diff, and eval human commands.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app
from soup_cli.commands import chat, diff
from soup_cli.commands import eval as eval_cmd

runner = CliRunner()
ESC = "\x1b]0;t\x07"
EVIL_PATH_STR = f"models[/x]/run-[red]7{ESC}"
MOCK_GPU_RESOLVE = "soup_cli.utils.gpu.resolve_inference_device_map_and_dtype"


def test_chat_missing_model_path_escapes_markup_and_strips_control(monkeypatch):
    """chat: Model path not found line must not raise MarkupError and must escape
    brackets and strip ESC bytes.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))

    result = runner.invoke(cli_app, ["chat", "-m", EVIL_PATH_STR])
    assert result.exit_code == 1
    out = buffer.getvalue()
    assert "Model path not found: models[/x]/run-[red]7" in out
    assert chr(27) not in out


def test_chat_loading_model_panel_and_progress_escapes_markup(monkeypatch):
    """chat: Model line in 'Loading model' panel and 'Loading model:' /
    'Loading LoRA adapter:' dim messages escape brackets and strip ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = Path(tmpdir) / f"models[/x]-run-[red]7{ESC}"
        fake_model.mkdir(parents=True)

        # Mock device detection and model loading to avoid needing torch/GPU
        monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", {}))
        monkeypatch.setattr(chat, "_load_model", lambda *a, **kw: (MagicMock(), MagicMock()))
        monkeypatch.setattr(chat, "_generate", lambda *a, **kw: "hello")

        # Mock input loop by passing EOF
        result = runner.invoke(
            cli_app,
            ["chat", "-m", str(fake_model), "--device", "cpu"],
            input="/exit\n",
        )
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "models[/x]-run-[red]7" in out
        assert chr(27) not in out


def test_chat_load_model_helper_escapes_model_path(monkeypatch):
    """chat._load_model: 'Loading model: ...' and 'Loading LoRA adapter: ...'
    dim prints escape brackets and strip ESC bytes.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))
    monkeypatch.setattr(MOCK_GPU_RESOLVE, lambda *a, **kw: ("cpu", None))

    # Mock transformers module
    mock_tf = MagicMock()
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    mock_tf.AutoTokenizer.from_pretrained.return_value = fake_tok
    mock_tf.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", mock_tf)

    # Full model branch
    evil_path = Path(f"/tmp/evil[/x]-[red]{ESC}")
    chat._load_model(
        evil_path,
        is_adapter=False,
        base_model=None,
        device="cpu",
        trust_remote_code=False,
    )
    out = buffer.getvalue()
    assert "evil[/x]-[red]" in out
    assert chr(27) not in out


def test_diff_missing_model_paths_escape_markup(monkeypatch):
    """diff: 'Model A not found' and 'Model B not found' must escape brackets
    and strip ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    evil_a = f"nope[/x]-a-[red]{ESC}"
    evil_b = f"nope[/y]-b-[blue]{ESC}"

    result = runner.invoke(cli_app, ["diff", "-a", evil_a, "-b", evil_b])
    assert result.exit_code == 1
    out = buffer.getvalue()
    assert "Model A not found: nope[/x]-a-[red]" in out
    assert chr(27) not in out


def test_diff_results_saved_and_cannot_detect_base_escape_markup(monkeypatch):
    """diff: 'Results saved to {out_path}' and 'Cannot detect base model for {path}'
    escape brackets and strip ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_a = Path(tmpdir) / "model_a"
        fake_b = Path(tmpdir) / "model_b"
        fake_a.mkdir(parents=True)
        fake_b.mkdir(parents=True)

        out_file = Path(tmpdir) / f"out[/x]-[red]{ESC}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(diff, "_load_model", lambda *a, **kw: (None, None))
        monkeypatch.setattr(diff, "_generate", lambda *a, **kw: "reply")

        result = runner.invoke(
            cli_app,
            [
                "diff",
                "-a", str(fake_a),
                "-b", str(fake_b),
                "--device", "cpu",
                "--prompt", "hello",
                "-o", str(out_file),
            ],
        )
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "Results saved to " in out
        assert "out[/x]-[red]" in out
        assert chr(27) not in out


def test_diff_cannot_detect_base_model_escapes_markup(monkeypatch):
    """diff._load_model: 'Cannot detect base model for {path}' escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))
    monkeypatch.setattr(MOCK_GPU_RESOLVE, lambda *a, **kw: ("cpu", None))
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())

    with tempfile.TemporaryDirectory() as tmpdir:
        ad_dir = Path(tmpdir) / f"adapter[/x]-[red]{ESC}"
        ad_dir.mkdir(parents=True)
        (ad_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(typer.Exit):
            diff._load_model(str(ad_dir), base_model=None, device="cpu", trust_remote_code=False)

        out = buffer.getvalue()
        assert "Cannot detect base model for " in out
        assert "adapter[/x]-[red]" in out
        assert chr(27) not in out


def test_eval_benchmark_and_custom_missing_model_escape_markup(monkeypatch):
    """eval: 'Model path not found: {model_path}' in benchmark and custom commands
    escapes brackets and strips ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    evil_model = f"models[/x]-[red]{ESC}"

    res_bench = runner.invoke(cli_app, ["eval", "benchmark", "-m", evil_model, "-b", "arc"])
    assert res_bench.exit_code == 1
    out_bench = buffer.getvalue()
    assert "Model path not found: models[/x]-[red]" in out_bench
    assert chr(27) not in out_bench

    buffer.truncate(0)
    buffer.seek(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_file = Path(tmpdir) / "tasks.jsonl"
        tasks_file.write_text(json.dumps({"prompt": "hi", "reference": "hi"}) + "\n")
        res_custom = runner.invoke(
            cli_app,
            ["eval", "custom", "-m", evil_model, "-t", str(tasks_file)],
        )
        assert res_custom.exit_code == 1
        out_custom = buffer.getvalue()
        assert "Model path not found: models[/x]-[red]" in out_custom
        assert chr(27) not in out_custom


def test_eval_human_missing_model_and_panel_escape_markup(monkeypatch):
    """eval human: 'Model not found: {mpath}', 'A/B Comparison' panel (Model A / B),
    and 'Results saved to {output}' escape brackets and strip ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        p_file = Path(tmpdir) / "prompts.jsonl"
        p_file.write_text(json.dumps({"prompt": "hello"}) + "\n")

        # Missing model
        evil_a = f"models[/x]-a-[red]{ESC}"
        evil_b = f"models[/y]-b-[blue]{ESC}"
        res = runner.invoke(
            cli_app,
            ["eval", "human", "-i", str(p_file), "-a", evil_a, "-b", evil_b],
        )
        assert res.exit_code == 1
        out = buffer.getvalue()
        assert "Model not found: models[/x]-a-[red]" in out
        assert chr(27) not in out

        # Now test panel and results saved with existing directories
        dir_a = Path(tmpdir) / f"a[/x]-[red]{ESC}"
        dir_b = Path(tmpdir) / f"b[/y]-[blue]{ESC}"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        out_eval = Path(tmpdir) / f"saved[/out]-[green]{ESC}.json"
        out_eval.parent.mkdir(parents=True, exist_ok=True)

        buffer.truncate(0)
        buffer.seek(0)

        def mock_gen(p):
            return lambda pr: "resp"

        monkeypatch.setattr("soup_cli.eval.custom._create_default_generator", mock_gen)
        res_ok = runner.invoke(
            cli_app,
            [
                "eval", "human",
                "-i", str(p_file),
                "-a", str(dir_a),
                "-b", str(dir_b),
                "-o", str(out_eval),
            ],
            input="A\n",
        )
        assert res_ok.exit_code == 0
        out_ok = buffer.getvalue()
        assert "Model A: " in out_ok
        assert "a[/x]-[red]" in out_ok
        assert "Model B: " in out_ok
        assert "b[/y]-[blue]" in out_ok
        assert "Results saved to " in out_ok
        assert "saved[/out]-[green]" in out_ok
        assert chr(27) not in out_ok


def test_eval_short_model_name_and_tables_escape_markup(monkeypatch):
    """_short_model_name returns for_terminal(...) so Elo table and Leaderboard table
    rows escape brackets and strip ESC bytes.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, color_system=None, width=400)
    monkeypatch.setattr(eval_cmd, "console", console)

    evil_path = f"org/checkpoint[/x]-[red]7{ESC}"
    short = eval_cmd._short_model_name(evil_path)
    # The escaped markup has \\[red]
    assert "[/x]" in short
    assert "\\[red]" in short
    assert chr(27) not in short

    # Render through a rich table to prove it renders literal [red] and doesn't get swallowed
    from rich.table import Table
    table = Table()
    table.add_column("Model")
    table.add_row(short)
    console.print(table)
    table_out = buffer.getvalue()
    assert "checkpoint[/x]-[red]7" in table_out
    assert chr(27) not in table_out
