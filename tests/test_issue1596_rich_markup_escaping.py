"""Tests for Issue #1596: Square-bracket text in model replies and dataset keys as literal text."""

import io
import json
import tempfile
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app
from soup_cli.commands import chat, data, diff
from soup_cli.commands import eval as eval_cmd

runner = CliRunner()


def test_data_inspect_columns_with_square_brackets(monkeypatch):
    """Columns with brackets like tags[i], [x] reviewed, note[/b] should render literally."""
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "keys.jsonl"
        path.write_text(
            '{"text": "row 1", "tags[i]": "a", "[x] reviewed": "yes", "note[/b]": "ok"}\n',
            encoding="utf-8",
        )
        result = runner.invoke(cli_app, ["data", "inspect", str(path), "--rows", "1"])
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "tags[i]" in out
        assert "[x] reviewed" in out
        assert "note[/b]" in out


def test_data_validate_unrecognized_keys_with_markup(monkeypatch):
    """Unknown keys with brackets like notes[/b] should not trigger MarkupError."""
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "unknown.jsonl"
        path.write_text('{"question": "hi", "notes[/b]": "hello"}\n', encoding="utf-8")
        result = runner.invoke(cli_app, ["data", "validate", str(path)])
        # Exit code should be usage error (3), not unhandled MarkupError (1)
        assert result.exit_code == 3
        out = buffer.getvalue()
        assert "notes[/b]" in out


def test_data_validate_kto_issue_with_markup(monkeypatch):
    """KTO issue with label '[/maybe]' should not crash with MarkupError."""
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kto.jsonl"
        path.write_text(
            '{"prompt": "p is long", "completion": "c is long", "label": "[/maybe]"}\n'
            '{"prompt": "q is long", "completion": "d is long", "label": true}\n',
            encoding="utf-8",
        )
        result = runner.invoke(cli_app, ["data", "validate", str(path), "--format", "kto"])
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "[/maybe]" in out


def test_chat_replies_and_system_with_markup(monkeypatch):
    """Assistant replies and system prompt containing brackets should render literally."""
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))

    replies = iter([
        "Index it as arr[i]; tick [x] when done.",
        "Llama-2 wraps a turn as [INST] hi [/INST] hello.",
    ])
    monkeypatch.setattr(chat, "_load_model", lambda **kw: (None, None))
    monkeypatch.setattr(chat, "_generate", lambda *a, **kw: next(replies))

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = Path(tmpdir) / "fake_model"
        fake_model.mkdir()
        result = runner.invoke(
            cli_app,
            ["chat", "-m", str(fake_model), "--device", "cpu"],
            input="/system Be [b]helpful[/b]!\nq1\nq2\n/quit\n",
        )
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "arr[i]" in out
        assert "[x]" in out
        assert "[INST] hi [/INST]" in out
        assert "Be [b]helpful[/b]!" in out


def test_diff_prompts_and_replies_with_markup(monkeypatch):
    """Diff prompt and side-by-side panels containing markup should render literally."""
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    replies = iter([
        "A1 uses arr[i]",
        "B1 closes with [/code]",
    ])
    monkeypatch.setattr(diff, "_load_model", lambda *a, **kw: (None, None))
    monkeypatch.setattr(diff, "_generate", lambda *a, **kw: next(replies))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "diff.jsonl"
        fake_a = Path(tmpdir) / "model_a"
        fake_b = Path(tmpdir) / "model_b"
        fake_a.mkdir()
        fake_b.mkdir()

        result = runner.invoke(
            cli_app,
            [
                "diff",
                "-a", str(fake_a),
                "-b", str(fake_b),
                "--device", "cpu",
                "--prompt", "Check [/tag] here",
                "-o", str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        out = buffer.getvalue()
        assert "Check [/tag] here" in out
        assert "A1 uses arr[i]" in out
        assert "B1 closes with [/code]" in out


def test_diff_model_paths_and_names_with_markup(monkeypatch):
    """Diff plan panel and summary table with brackets in model paths/names render literally."""
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    monkeypatch.setattr(diff, "_load_model", lambda *a, **kw: (None, None))
    monkeypatch.setattr(diff, "_generate", lambda *a, **kw: "reply")

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_a = Path(tmpdir) / "ckpt[" / "]model[b]"
        fake_b = Path(tmpdir) / "model[v1]"
        fake_a.mkdir(parents=True)
        fake_b.mkdir()

        result = runner.invoke(
            cli_app,
            [
                "diff",
                "-a", str(fake_a),
                "-b", str(fake_b),
                "--device", "cpu",
                "--prompt", "test",
            ],
        )
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "ckpt[/]model[b]" in out
        assert "model[v1]" in out
        assert "Model A (]model[b])" in out
        assert "Model B (model[v1])" in out


def test_eval_human_with_markup(monkeypatch):
    """Human eval with markup in prompt and responses should render without crash and save."""
    from soup_cli.eval import custom

    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    replies = {
        "model_a": iter(["A1 with arr[i]"]),
        "model_b": iter(["B1 with [/INST] tags"]),
    }
    monkeypatch.setattr(
        custom,
        "_create_default_generator",
        lambda path: (lambda prompt: next(replies[Path(path).name])),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = Path(tmpdir) / "prompts.jsonl"
        out_file = Path(tmpdir) / "judgments.json"
        prompts_file.write_text(
            json.dumps({"prompt": "Prompt with [/tag]"}) + "\n",
            encoding="utf-8",
        )
        fake_a = Path(tmpdir) / "model_a"
        fake_b = Path(tmpdir) / "model_b"
        fake_a.mkdir()
        fake_b.mkdir()

        monkeypatch.setattr(eval_cmd.console, "input", lambda *a: "a")

        result = runner.invoke(
            cli_app,
            [
                "eval",
                "human",
                "-i", str(prompts_file),
                "-a", str(fake_a),
                "-b", str(fake_b),
                "-o", str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        out = buffer.getvalue()
        assert "Prompt with [/tag]" in out
        assert "A1 with arr[i]" in out
        assert "B1 with [/INST] tags" in out
