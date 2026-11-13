"""Tests for Issue #1864: model and output paths printed as raw Rich markup
in chat, diff, and eval commands. Terminal hygiene and Windows-safe fixtures.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app
from soup_cli.commands import chat, diff
from soup_cli.commands import eval as eval_cmd
from soup_cli.eval.custom import EvalResult, EvalTask
from soup_cli.eval.human import EloRating, HumanEvalResults
from soup_cli.eval.leaderboard import Leaderboard, LeaderboardEntry
from soup_cli.utils.terminal import for_terminal, strip_control

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
    expected_display = strip_control(Path(EVIL_PATH_STR))
    assert f"Model path not found: {expected_display}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_chat_loading_model_panel_escapes_markup(monkeypatch):
    """chat: Model line in 'Loading model' panel escapes brackets and strips ESC.
    Uses mocked filesystem seam so Windows physical filename constraints are avoided.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))

    evil_model_arg = f"models[/x]-run-[red]7{ESC}"
    orig_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: (
            False
            if str(self).endswith("adapter_config.json")
            else ("models[/x]-run-[red]7" in Path(self).as_posix() or orig_exists(self))
        ),
    )

    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", {}))
    monkeypatch.setattr(chat, "_load_model", lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(chat, "_generate", lambda *a, **kw: "hello")

    result = runner.invoke(
        cli_app,
        ["chat", "-m", evil_model_arg, "--device", "cpu"],
        input="/exit\n",
    )
    assert result.exit_code == 0
    out = buffer.getvalue()
    expected_display = strip_control(Path(evil_model_arg))
    assert expected_display in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_chat_load_model_full_model_escapes_markup(monkeypatch):
    """chat._load_model: 'Loading model: ...' dim message escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))
    monkeypatch.setattr(MOCK_GPU_RESOLVE, lambda *a, **kw: ("cpu", None))

    mock_tf = MagicMock()
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    mock_tf.AutoTokenizer.from_pretrained.return_value = fake_tok
    mock_tf.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", mock_tf)

    evil_path = Path(f"evil[/x]-[red]{ESC}")
    chat._load_model(
        evil_path,
        is_adapter=False,
        base_model=None,
        device="cpu",
        trust_remote_code=False,
    )
    out = buffer.getvalue()
    expected_display = strip_control(evil_path)
    assert f"Loading model: {expected_display}..." in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_chat_load_model_lora_adapter_escapes_markup(monkeypatch):
    """chat._load_model: 'Loading LoRA adapter: ...' dim message escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))
    monkeypatch.setattr(MOCK_GPU_RESOLVE, lambda *a, **kw: ("cpu", None))

    mock_tf = MagicMock()
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    mock_tf.AutoTokenizer.from_pretrained.return_value = fake_tok
    mock_tf.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", mock_tf)

    mock_peft = MagicMock()
    mock_peft.PeftModel.from_pretrained.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "peft", mock_peft)

    evil_adapter_path = Path(f"evil_adapter[/lora]-[green]{ESC}")
    chat._load_model(
        evil_adapter_path,
        is_adapter=True,
        base_model="base-model-id",
        device="cpu",
        trust_remote_code=False,
    )
    out = buffer.getvalue()
    expected_display = strip_control(evil_adapter_path)
    assert f"Loading LoRA adapter: {expected_display}..." in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_diff_missing_model_a_escapes_markup(monkeypatch):
    """diff: 'Model A not found: ...' escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    evil_a = f"nope[/x]-a-[red]{ESC}"
    evil_b = f"nope[/y]-b-[blue]{ESC}"

    result = runner.invoke(cli_app, ["diff", "-a", evil_a, "-b", evil_b])
    assert result.exit_code == 1
    out = buffer.getvalue()
    expected_a = strip_control(Path(evil_a))
    assert f"Model A not found: {expected_a}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_diff_missing_model_b_escapes_markup(monkeypatch, tmp_path):
    """diff: 'Model B not found: ...' distinct sink regression test.
    When Model A exists (portable physical fixture), Model B refusal
    escapes brackets and strips ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    valid_a = tmp_path / "valid_model_a"
    valid_a.mkdir()

    evil_b = f"nope[/y]-b-[blue]{ESC}"

    result = runner.invoke(cli_app, ["diff", "-a", str(valid_a), "-b", evil_b])
    assert result.exit_code == 1
    out = buffer.getvalue()
    expected_b = strip_control(Path(evil_b))
    assert f"Model B not found: {expected_b}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_diff_results_saved_escapes_markup(monkeypatch, tmp_path):
    """diff: 'Results saved to {out_path}' escapes brackets and strips ESC.
    Uses portable fixture directories for models and mocks open() for the evil output path.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    valid_a = tmp_path / "model_a"
    valid_b = tmp_path / "model_b"
    valid_a.mkdir()
    valid_b.mkdir()

    evil_out_str = f"out[/x]-[red]{ESC}.jsonl"

    monkeypatch.setattr(diff, "_load_model", lambda *a, **kw: (None, None))
    monkeypatch.setattr(diff, "_generate", lambda *a, **kw: "reply")

    # Mock open to avoid OS errors creating evil filenames on Windows
    m_open = mock_open()
    orig_open = open

    def guarded_open(file, *args, **kwargs):
        if "out[/x]" in Path(file).as_posix():
            return m_open(file, *args, **kwargs)
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)

    result = runner.invoke(
        cli_app,
        [
            "diff",
            "-a", str(valid_a),
            "-b", str(valid_b),
            "--device", "cpu",
            "--prompt", "hello",
            "-o", evil_out_str,
        ],
    )
    assert result.exit_code == 0
    out = buffer.getvalue()
    expected_out = strip_control(Path(evil_out_str))
    assert f"Results saved to {expected_out}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_diff_cannot_detect_base_model_escapes_markup(monkeypatch):
    """diff._load_model: 'Cannot detect base model for {path}' escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))
    monkeypatch.setattr(MOCK_GPU_RESOLVE, lambda *a, **kw: ("cpu", None))
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())

    evil_ad_path = Path(f"adapter[/x]-[red]{ESC}")

    orig_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if "adapter_config.json" in str(self) else orig_exists(self),
    )

    m_open = mock_open(read_data="{}")
    orig_open = open

    def guarded_open(file, *args, **kwargs):
        if "adapter_config.json" in str(file):
            return m_open(file, *args, **kwargs)
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)

    with pytest.raises(typer.Exit):
        diff._load_model(str(evil_ad_path), base_model=None, device="cpu", trust_remote_code=False)

    out = buffer.getvalue()
    expected_path = strip_control(evil_ad_path)
    assert f"Cannot detect base model for {expected_path}. Use --base-a/--base-b." in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_eval_benchmark_missing_model_escapes_markup(monkeypatch):
    """eval benchmark: 'Model path not found: ...' escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    evil_model = f"models[/x]-[red]{ESC}"
    res = runner.invoke(cli_app, ["eval", "benchmark", "-m", evil_model, "-b", "arc"])
    assert res.exit_code == 1
    out = buffer.getvalue()
    expected_model = strip_control(Path(evil_model))
    assert f"Model path not found: {expected_model}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_eval_custom_missing_model_escapes_markup(monkeypatch, tmp_path):
    """eval custom: 'Model path not found: ...' escapes brackets and strips ESC."""
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps({"prompt": "hi", "reference": "hi"}) + "\n")

    evil_model = f"models[/x]-[red]{ESC}"
    res = runner.invoke(
        cli_app,
        ["eval", "custom", "-m", evil_model, "-t", str(tasks_file)],
    )
    assert res.exit_code == 1
    out = buffer.getvalue()
    expected_model = strip_control(Path(evil_model))
    assert f"Model path not found: {expected_model}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_eval_custom_loaded_tasks_path_escapes_markup(monkeypatch, tmp_path):
    """eval custom: 'Loaded N eval tasks from {tasks_path}' distinct sink regression test.
    Escapes brackets and strips ESC when tasks_path contains control characters/markup.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    valid_model = tmp_path / "model_fixture"
    valid_model.mkdir()

    evil_tasks_str = f"tasks[/x]-[yellow]{ESC}.jsonl"

    fake_task = EvalTask(prompt="hi", expected="hi")
    monkeypatch.setattr(
        "soup_cli.eval.custom.load_eval_tasks",
        lambda p: [fake_task],
    )
    monkeypatch.setattr(
        "soup_cli.eval.custom._create_default_generator",
        lambda *a, **kw: (lambda pr: "hi"),
    )
    monkeypatch.setattr(
        "soup_cli.eval.custom.score_task",
        lambda task, resp: EvalResult(task=task, output=resp, score=1.0, matched=True),
    )

    result = runner.invoke(
        cli_app,
        ["eval", "custom", "-m", str(valid_model), "-t", evil_tasks_str],
    )
    assert result.exit_code == 0
    out = buffer.getvalue()
    expected_tasks = strip_control(Path(evil_tasks_str))
    assert f"Loaded 1 eval tasks from {expected_tasks}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_eval_human_missing_model_escapes_markup(monkeypatch, tmp_path):
    """eval human: 'Model not found: ...' escapes brackets and strips ESC for both Model A and B."""
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "hello"}) + "\n")

    # 1. Model A missing
    evil_a = f"models[/x]-a-[red]{ESC}"
    evil_b = f"models[/y]-b-[blue]{ESC}"
    res_a = runner.invoke(
        cli_app,
        ["eval", "human", "-i", str(prompts_file), "-a", evil_a, "-b", evil_b],
    )
    assert res_a.exit_code == 1
    out_a = buffer.getvalue()
    expected_a = strip_control(evil_a)
    assert f"Model not found: {expected_a}" in out_a
    assert chr(27) not in out_a
    assert chr(7) not in out_a

    # 2. Model A exists, Model B missing
    valid_a = tmp_path / "valid_model_a"
    valid_a.mkdir()
    buffer.truncate(0)
    buffer.seek(0)

    res_b = runner.invoke(
        cli_app,
        ["eval", "human", "-i", str(prompts_file), "-a", str(valid_a), "-b", evil_b],
    )
    assert res_b.exit_code == 1
    out_b = buffer.getvalue()
    expected_b = strip_control(evil_b)
    assert f"Model not found: {expected_b}" in out_b
    assert chr(27) not in out_b
    assert chr(7) not in out_b


def test_eval_human_panel_and_results_saved_escapes_markup(monkeypatch, tmp_path):
    """eval human: 'A/B Comparison' panel and 'Results saved to ...'
    escape brackets and strip ESC.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "hello"}) + "\n")

    evil_a = f"a[/x]-[red]{ESC}"
    evil_b = f"b[/y]-[blue]{ESC}"
    evil_out = f"saved[/out]-[green]{ESC}.json"

    orig_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: (
            True
            if "a[/x]" in Path(self).as_posix() or "b[/y]" in Path(self).as_posix()
            else orig_exists(self)
        ),
    )
    monkeypatch.setattr(
        "soup_cli.eval.custom._create_default_generator",
        lambda p: (lambda pr: "mock reply"),
    )
    monkeypatch.setattr("soup_cli.eval.human.save_results", lambda *a, **kw: None)

    result = runner.invoke(
        cli_app,
        [
            "eval", "human",
            "-i", str(prompts_file),
            "-a", evil_a,
            "-b", evil_b,
            "-o", evil_out,
        ],
        input="A\n",
    )
    assert result.exit_code == 0
    out = buffer.getvalue()
    assert f"Model A: {strip_control(evil_a)}" in out
    assert f"Model B: {strip_control(evil_b)}" in out
    assert f"Results saved to {strip_control(evil_out)}" in out
    assert chr(27) not in out
    assert chr(7) not in out


def test_eval_short_model_name_long_path_branch():
    """eval._short_model_name branch 1 (len(parts) > 2) returns
    for_terminal('/'.join(parts[-2:])).
    """
    evil_long = f"org/sub/checkpoint[/x]-[red]7{ESC}"
    short = eval_cmd._short_model_name(evil_long)
    parts = Path(evil_long).parts
    expected = for_terminal("/".join(parts[-2:]))
    assert short == expected
    assert "\\[red]" in short
    assert chr(27) not in short
    assert chr(7) not in short


def test_eval_short_model_name_short_path_branch():
    """eval._short_model_name branch 2 (len(parts) <= 2) returns for_terminal(path)."""
    evil_short = f"model[/y]-[blue]8{ESC}"
    short = eval_cmd._short_model_name(evil_short)
    expected = for_terminal(evil_short)
    assert short == expected
    assert "\\[blue]" in short
    assert chr(27) not in short
    assert chr(7) not in short


def test_eval_elo_ratings_table_consumer_forced_ansi(monkeypatch):
    """eval._display_elo_ratings consumer executes both short and long path branches,
    strips controls, preserves literal markup, and renders table formatting under forced ANSI.
    """
    buffer = io.StringIO()
    ansi_console = Console(file=buffer, color_system="standard", force_terminal=True, width=120)
    monkeypatch.setattr(eval_cmd, "console", ansi_console)

    evil_long = f"org/sub/model_long[/x]-[red]7{ESC}"
    evil_short = f"model_short[/y]-[blue]8{ESC}"

    results = HumanEvalResults(
        ratings={
            evil_long: EloRating(model=evil_long, rating=1250.0, wins=2, losses=1, ties=0),
            evil_short: EloRating(model=evil_short, rating=1150.0, wins=1, losses=2, ties=0),
        }
    )

    eval_cmd._display_elo_ratings(results)
    table_out = buffer.getvalue()

    # 1. Forced-ANSI table styling: bold header/columns inserted by Rich
    assert "\x1b[1m" in table_out

    # 2. Controls supplied in the path are stripped
    assert chr(27) + "]0;t" not in table_out
    assert chr(7) not in table_out

    # 3. Literal markup is preserved as text, not interpreted as colors
    assert "model_long[/x]-[red]7" in table_out
    assert "model_short[/y]-[blue]8" in table_out

    # 4. Escaped markup tags do NOT apply ANSI color codes (\x1b[31m = red, \x1b[34m = blue)
    assert "\x1b[31m" not in table_out
    assert "\x1b[34m" not in table_out


def test_eval_leaderboard_table_consumer_forced_ansi(monkeypatch):
    """eval.leaderboard command table consumer executes both short and long path branches,
    strips controls, preserves literal markup, and renders table formatting under forced ANSI.
    """
    buffer = io.StringIO()
    ansi_console = Console(file=buffer, color_system="standard", force_terminal=True, width=120)
    monkeypatch.setattr(eval_cmd, "console", ansi_console)

    evil_long = f"org/sub/model_long[/x]-[red]7{ESC}"
    evil_short = f"model_short[/y]-[blue]8{ESC}"

    lb = Leaderboard(
        entries=[
            LeaderboardEntry(model_path=evil_long, benchmark="arc", score=0.85),
            LeaderboardEntry(model_path=evil_short, benchmark="arc", score=0.75),
        ]
    )
    lb.compute()

    monkeypatch.setattr(
        "soup_cli.eval.leaderboard.build_leaderboard_from_tracker",
        lambda tracker: lb,
    )

    result = runner.invoke(cli_app, ["eval", "leaderboard", "--format", "table"])
    assert result.exit_code == 0
    table_out = buffer.getvalue()

    # 1. Forced-ANSI table styling: bold header/columns inserted by Rich
    assert "\x1b[1m" in table_out

    # 2. Controls supplied in the path are stripped
    assert chr(27) + "]0;t" not in table_out
    assert chr(7) not in table_out

    # 3. Literal markup is preserved as text
    assert "model_long[/x]-[red]7" in table_out
    assert "model_short[/y]-[blue]8" in table_out

    # 4. Escaped markup tags do NOT apply ANSI color codes
    assert "\x1b[31m" not in table_out
    assert "\x1b[34m" not in table_out
