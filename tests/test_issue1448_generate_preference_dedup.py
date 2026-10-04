"""Tests for issue #1448: soup data generate --dedup-with on --template preference.

Acceptance criteria:
- generate --template preference --dedup-with drops a repeated row (both DPO and KTO formats).
- generate --template preference deduplicates repeated rows within a run.
- _row_to_text produces a stable key for preference rows regardless of dictionary key ordering.
"""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.generate import _row_to_text
from tests.conftest import strip_ansi


def test_row_to_text_preference_stable_key():
    """_row_to_text should produce identical keys regardless of dict insertion order."""
    dpo_1 = {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
    dpo_2 = {"chosen": "4", "rejected": "5", "prompt": "What is 2+2?"}
    assert _row_to_text(dpo_1) == _row_to_text(dpo_2)

    kto_1 = {"prompt": "Write a poem", "completion": "Roses are red", "label": True}
    kto_2 = {"label": True, "prompt": "Write a poem", "completion": "Roses are red"}
    assert _row_to_text(kto_1) == _row_to_text(kto_2)


def test_preference_dedup_with_drops_repeated_rows(tmp_path, monkeypatch):
    """--dedup-with must drop repeated DPO and KTO rows when --template preference is used."""
    monkeypatch.chdir(tmp_path)

    existing_file = tmp_path / "existing.jsonl"
    existing_records = [
        {
            "prompt": "Explain gravity",
            "chosen": "Mass curves spacetime",
            "rejected": "Things fall down",
        },
        {"prompt": "Is Python fast?", "completion": "It depends on the workload", "label": True},
    ]
    with open(existing_file, "w", encoding="utf-8") as f:
        for r in existing_records:
            f.write(json.dumps(r) + "\n")

    generated_batch = [
        # Repeated DPO from existing.jsonl
        {
            "prompt": "Explain gravity",
            "chosen": "Mass curves spacetime",
            "rejected": "Things fall down",
        },
        # New DPO
        {"prompt": "Explain entropy", "chosen": "Measure of disorder", "rejected": "Heat"},
        # Repeated KTO from existing.jsonl
        {"prompt": "Is Python fast?", "completion": "It depends on the workload", "label": True},
        # New KTO
        {"prompt": "Is Rust safe?", "completion": "Yes, memory safety without GC", "label": True},
    ]

    runner = CliRunner()
    with patch("soup_cli.commands.generate._generate_batch", return_value=generated_batch):
        result = runner.invoke(app, [
            "data", "generate",
            "--prompt", "Generate preference data",
            "--template", "preference",
            "--dedup-with", str(existing_file.name),
            "--output", "out.jsonl",
            "--count", "4",
            "--provider", "server",
        ])

    assert result.exit_code == 0, result.output
    assert "Duplicates: 2 removed" in " ".join(strip_ansi(result.output).split())

    out_file = tmp_path / "out.jsonl"
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().splitlines()]

    # Exactly 2 new rows kept, 2 duplicates dropped
    assert len(lines) == 2
    assert lines[0]["prompt"] == "Explain entropy"
    assert lines[1]["prompt"] == "Is Rust safe?"


def test_preference_within_run_dedup(tmp_path, monkeypatch):
    """Repeated preference rows generated within the same run must be deduplicated."""
    monkeypatch.chdir(tmp_path)

    generated_batch = [
        {"prompt": "P1", "chosen": "C1", "rejected": "R1"},
        {"prompt": "P1", "chosen": "C1", "rejected": "R1"},  # duplicate
        {"prompt": "P2", "completion": "Comp2", "label": False},
        {"prompt": "P2", "completion": "Comp2", "label": False},  # duplicate
    ]

    runner = CliRunner()
    with patch("soup_cli.commands.generate._generate_batch", return_value=generated_batch):
        result = runner.invoke(app, [
            "data", "generate",
            "--prompt", "Generate preference data",
            "--template", "preference",
            "--output", "out.jsonl",
            "--count", "4",
            "--provider", "server",
        ])

    assert result.exit_code == 0, result.output
    out_file = tmp_path / "out.jsonl"
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().splitlines()]

    assert len(lines) == 2
    assert lines[0] == {"prompt": "P1", "chosen": "C1", "rejected": "R1"}
    assert lines[1] == {"prompt": "P2", "completion": "Comp2", "label": False}


def test_preference_dedup_keeps_rows_that_differ_beyond_the_prompt(tmp_path, monkeypatch):
    """Only exact repeats are dropped: a row that shares the prompt but differs in
    chosen, rejected, completion or label is a different example and is kept."""
    monkeypatch.chdir(tmp_path)
    batch = [
        {"prompt": "P", "chosen": "C1", "rejected": "R1"},
        {"prompt": "P", "chosen": "C2", "rejected": "R1"},
        {"prompt": "P", "chosen": "C1", "rejected": "R2"},
        {"prompt": "P", "completion": "A", "label": True},
        {"prompt": "P", "completion": "B", "label": True},
        {"prompt": "P", "completion": "A", "label": False},
        {"prompt": "P", "chosen": "C1", "rejected": "R1"},  # the only exact repeat
    ]
    with patch("soup_cli.commands.generate._generate_batch", return_value=batch):
        result = CliRunner().invoke(app, [
            "data", "generate", "--prompt", "x", "--template", "preference",
            "--output", "out.jsonl", "--count", "7", "--batch-size", "7",
            "--provider", "server",
        ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    rows = [json.loads(line) for line in
            (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == batch[:6]
    assert "Duplicates: 1 removed" in " ".join(strip_ansi(result.output).split())

