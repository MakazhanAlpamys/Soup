"""Tests for Issue #1596: Square-bracket text in model replies and dataset keys as literal text."""

import io
import json
import os
import tempfile
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app as cli_app
from soup_cli.commands import chat, data, diff
from soup_cli.commands import eval as eval_cmd

runner = CliRunner()
ESC = "\x1b]0;t\x07"


def test_data_inspect_columns_with_square_brackets(monkeypatch):
    """Columns with brackets like tags[i], [x] reviewed, note[/b] should render literally
    and strip ESC control bytes.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "keys.jsonl"
        path.write_text(
            json.dumps({
                "text": "row 1",
                f"tags[i]{ESC}": "a",
                f"[x] reviewed{ESC}": "yes",
                f"note[/b]{ESC}": "ok",
            })
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(cli_app, ["data", "inspect", str(path), "--rows", "0"])
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "tags[i]" in out
        assert "[x] reviewed" in out
        assert "note[/b]" in out
        assert chr(27) not in out


def test_data_validate_unrecognized_keys_with_markup(monkeypatch):
    """Unknown keys with brackets like notes[/b] should not trigger MarkupError
    and should strip ESC control bytes in error reporting.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "unknown.jsonl"
        path.write_text(
            json.dumps({"question": "hi", f"notes[/b]{ESC}": "hello"}) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(cli_app, ["data", "validate", str(path)])
        # Exit code should be usage error (3), not unhandled MarkupError (1)
        assert result.exit_code == 3
        out = buffer.getvalue()
        assert "notes[/b]" in out
        assert chr(27) not in out


def test_data_validate_kto_issue_with_markup(monkeypatch):
    """KTO issue with label '[/maybe]' should not crash with MarkupError
    and should strip ESC control bytes in issue reporting.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(data, "console", Console(file=buffer, color_system=None, width=400))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kto.jsonl"
        row1 = json.dumps({
            "prompt": "p is long",
            "completion": "c is long",
            "label": f"[/maybe]{ESC}",
        })
        row2 = json.dumps({"prompt": "q is long", "completion": "d is long", "label": True})
        path.write_text(f"{row1}\n{row2}\n", encoding="utf-8")
        result = runner.invoke(cli_app, ["data", "validate", str(path), "--format", "kto"])
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "[/maybe]" in out
        assert chr(27) not in out


def test_chat_replies_and_system_with_markup(monkeypatch):
    """Assistant replies and system prompt containing brackets should render literally
    and strip ESC bytes.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buffer, color_system=None, width=400))

    replies = iter([
        f"Index it as arr[i]; tick [x] when done.{ESC}",
        f"Llama-2 wraps a turn as [INST] hi [/INST] hello.{ESC}",
        f"Title attack: {ESC}arr[i] with ESC control byte.",
    ])
    monkeypatch.setattr(chat, "_load_model", lambda **kw: (None, None))
    monkeypatch.setattr(chat, "_generate", lambda *a, **kw: next(replies))

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = Path(tmpdir) / "fake_model"
        fake_model.mkdir()
        result = runner.invoke(
            cli_app,
            ["chat", "-m", str(fake_model), "--device", "cpu"],
            input=f"/system Be [b]helpful[/b]!{ESC}\nq1\nq2\nq3\n/quit\n",
        )
        assert result.exit_code == 0
        out = buffer.getvalue()
        assert "arr[i]" in out
        assert "[x]" in out
        assert "[INST] hi [/INST]" in out
        assert "Be [b]helpful[/b]!" in out
        assert chr(27) not in out


def test_diff_prompts_and_replies_with_markup(monkeypatch):
    """Diff prompt and side-by-side panels containing markup should render literally,
    strip ESC control bytes, and preserve raw file data.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(diff, "console", Console(file=buffer, color_system=None, width=400))

    prompt_text = f"Check [/INST] with arr[i] and \\path{ESC}"
    reply_a = f"A1 uses arr[i] and \\backslash [/INST]{ESC}"
    reply_b = f"B1 closes with [/INST], arr[i] and \\foo [/code]{ESC}"
    replies = iter([reply_a, reply_b])
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
                "--prompt", prompt_text,
                "-o", str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        out = buffer.getvalue()
        assert prompt_text.replace(ESC, "") in out
        assert reply_a.replace(ESC, "") in out
        assert reply_b.replace(ESC, "") in out
        assert chr(27) not in out

        # Pin written output: saved file must preserve the exact raw unescaped strings
        content = out_file.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["prompt"] == prompt_text
        assert rows[0]["response_a"] == reply_a
        assert rows[0]["response_b"] == reply_b


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
        assert f"ckpt[{os.sep}]model[b]" in out
        assert f"Model B:  {fake_b}" in out
        assert f"Model A: {fake_a.name}" in out
        assert f"Model B: {fake_b.name}" in out
        assert "Model A (]model[b])" in out
        assert "Model B (model[v1])" in out


def test_eval_human_with_markup(monkeypatch):
    """Human eval with markup in prompt and responses should render without crash,
    strip ESC control bytes, and save raw file data.
    """
    from soup_cli.eval import custom

    buffer = io.StringIO()
    monkeypatch.setattr(eval_cmd, "console", Console(file=buffer, color_system=None, width=400))

    prompt_text = f"Prompt with [/INST], arr[i] and \\backslash [/tag]{ESC}"
    reply_a = f"A1 with arr[i], [/INST] and \\slash{ESC}"
    reply_b = f"B1 with [/INST] tags, arr[i] and \\test{ESC}"
    replies = {
        "model_a": iter([reply_a]),
        "model_b": iter([reply_b]),
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
            json.dumps({"prompt": prompt_text}) + "\n",
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
        assert prompt_text.replace(ESC, "") in out
        assert reply_a.replace(ESC, "") in out
        assert reply_b.replace(ESC, "") in out
        assert chr(27) not in out

        # Pin written output: saved file must preserve the exact raw unescaped strings
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(data["judgments"]) == 1
        j = data["judgments"][0]
        assert j["prompt"] == prompt_text
        assert j["response_a"] == reply_a
        assert j["response_b"] == reply_b
