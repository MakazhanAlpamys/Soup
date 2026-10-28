"""Names from a config or a command line reach the terminal through ``for_terminal``.

``soup profile`` prints the modules LoRA adapts (``lora.target_modules`` from the
config) and ``soup adapters bisect`` prints checkpoint ids from its arguments. Text from
outside the program goes through ``utils/terminal.py:for_terminal``, which drops control
characters and bidi marks as well as escaping Rich markup; ``rich.markup.escape`` alone
does only the second.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import strip_ansi

ESC = "\x1b"
BIDI = "\u202e"
NAME = f"q_proj{ESC}]0;title{BIDI}\x07x"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


def _profile(workdir, targets) -> str:
    document = {
        "base": str(workdir / "tiny"),
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "lora": {"r": 8, "target_modules": targets}},
    }
    (workdir / "soup.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--gpu", "a100"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return strip_ansi(result.output)


class TestProfileTargetModules:
    def test_control_characters_and_bidi_marks_do_not_reach_the_terminal(self, workdir):
        output = _profile(workdir, [NAME, "v_proj"])
        assert ESC not in output
        assert BIDI not in output
        assert "\x07" not in output

    def test_the_names_are_still_shown(self, workdir):
        output = " ".join(_profile(workdir, [NAME, "v_proj"]).split())
        assert "q_proj" in output
        assert "v_proj" in output

    def test_markup_in_a_name_is_shown_not_applied(self, workdir):
        output = " ".join(_profile(workdir, ["[bold red]x[/]", "v_proj"]).split())
        assert "[bold red]x[/]" in output

    def test_ordinary_names_are_unchanged(self, workdir):
        output = " ".join(_profile(workdir, ["q_proj", "v_proj"]).split())
        assert "on q_proj, v_proj)" in output


class TestBisectCheckpointIds:
    def _run(self, *ids: str):
        result = CliRunner().invoke(
            app, ["adapters", "bisect", *ids, "--eval-command", "{ckpt}"]
        )
        result.plain = strip_ansi(result.output)  # Rich's colour codes out, the rest as printed
        return result

    def test_a_failed_probe_names_the_checkpoint_without_control_characters(self):
        result = self._run(f"ckpt{ESC}]0;t{BIDI}-1", "ckpt-2")
        assert "Eval failed for" in result.plain, (result.output, repr(result.exception))
        assert ESC not in result.plain
        assert BIDI not in result.plain

    def test_the_checkpoint_name_is_still_shown(self):
        result = self._run(f"ckpt{ESC}-1", "ckpt-2")
        assert "Eval failed for ckpt-1" in " ".join(result.plain.split())
