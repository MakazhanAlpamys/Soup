"""The distill preview renders model names and copyable YAML literally."""

import json
import os

import pytest
import yaml
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import draft as draft_cmd
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("suffix", "display_suffix"),
    [
        pytest.param("plain", "plain", id="plain-control"),
        pytest.param("[red]x", "[red]x", id="markup-control"),
        pytest.param("[bold]x\x9b\u202e", "[bold]x", id="terminal-controls"),
        pytest.param(r"\[unclosed", r"\[unclosed", id="backslash-before-bracket"),
        pytest.param(
            ":warning:",
            ":warning:",
            id="emoji-alias",
            marks=pytest.mark.skipif(os.name == "nt", reason="Colon is not a Windows filename"),
        ),
    ],
)
def test_distill_plan_keeps_header_safe_and_yaml_literal(
    tmp_path, monkeypatch, suffix, display_suffix
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(draft_cmd, "console", Console(width=200, height=40))
    target, base = f"target-{suffix}", f"student-{suffix}"
    output, data = f"draft-{suffix}", f"data-{suffix}.jsonl"

    # Real local configs take the mismatched-vocab branch without loading tokenizers or weights.
    for name, vocab in ((target, 32), (base, 16)):
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(
            json.dumps({"model_type": "gpt2", "vocab_size": vocab}), encoding="utf-8"
        )
    data_path = tmp_path / data
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text('{"text": "test row"}\n', encoding="utf-8")
    before = set(tmp_path.iterdir())

    result = make_cli_runner().invoke(
        app,
        [
            "draft",
            "distill",
            "--target",
            target,
            "--draft-base",
            base,
            "--data",
            data,
            "-o",
            output,
            "--steps",
            "1",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert set(tmp_path.iterdir()) == before
    stdout = result.stdout
    rendered = strip_ansi(stdout)
    header, preview = rendered.split("\n", 1)
    yaml_text, footer = preview.rsplit("--plan-only:", 1)
    config = yaml.safe_load(yaml_text)
    # The YAML quotes raw characters so the preview still round-trips into the same config.
    assert config["training"]["teacher_model"] == target
    assert config["base"] == base
    assert config["output"] == output
    assert config["data"]["train"] == data
    assert config["training"]["uld_strategy"] == "wasserstein_aligned"
    assert footer.strip() == "nothing written."
    # Inspect the raw stream so stripping ANSI for the header assertion cannot hide a leak.
    assert "\x9b" not in stdout and "\u202e" not in stdout
    assert header == (
        f"Plan — distil target-{display_suffix} into student-{display_suffix} over 1 rows "
        "(cross-tokenizer: target=32, draft=16 -> uld_strategy=wasserstein_aligned)"
    )
