"""Tests for count retries and empty round bounds in soup data generate (#2415)."""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def test_generate_retries_until_target_count(tmp_path, monkeypatch):
    """The generation loop should keep requesting batches until --count is satisfied."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = []
    counter = 0

    def fake_generate_batch(**kwargs):
        nonlocal counter
        calls.append(kwargs["count"])
        res = [
            {"instruction": f"Question {counter + i}", "output": f"Answer {counter + i}"}
            for i in range(2)
        ]
        counter += 2
        return res

    with patch("soup_cli.commands.generate._generate_batch", side_effect=fake_generate_batch):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "5",
                "--batch-size",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # 5 requested, 2 per call -> calls were: [5, 3, 1]
    assert calls == [5, 3, 1]

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 5


def test_generate_exits_1_when_provider_repeatedly_empty(tmp_path, monkeypatch):
    """Three consecutive empty rounds must halt generation and exit with code 1."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = 0

    def fake_generate_batch_empty(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"instruction": "Q1", "output": "A1"}]
        return []

    with patch("soup_cli.commands.generate._generate_batch", side_effect=fake_generate_batch_empty):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "5",
                "--batch-size",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert calls == 4

    output_text = " ".join(strip_ansi(result.output).split())
    expected_msg = (
        "Generation halted: received no valid, unique examples after 3 consecutive attempts "
        "(1/5 generated)"
    )
    assert expected_msg in output_text

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1


def test_generate_resets_consecutive_empty_rounds_on_success(tmp_path, monkeypatch):
    """An empty round followed by a successful round must reset the consecutive empty counter."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = 0

    def fake_generate_intermittent(**kwargs):
        nonlocal calls
        calls += 1
        # Calls 1, 3, 6, 7 return an example; calls 2, 4, 5 return empty list
        if calls in (1, 3, 6, 7):
            return [{"instruction": f"Q{calls}", "output": f"A{calls}"}]
        return []

    patch_fn = "soup_cli.commands.generate._generate_batch"
    with patch(patch_fn, side_effect=fake_generate_intermittent):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "4",
                "--batch-size",
                "2",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == 7

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 4


@pytest.mark.parametrize("existing", [False, True])
def test_generate_halt_without_rows_preserves_output(tmp_path, monkeypatch, existing):
    """No accepted rows must neither create nor truncate the output file."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    sentinel = b"existing dataset\n"
    if existing:
        output_file.write_bytes(sentinel)
    with patch("soup_cli.commands.generate._generate_batch", return_value=[]) as provider:
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test",
                "--count",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert provider.call_count == 3
    assert "after 3 consecutive attempts (0/5 generated)" in " ".join(
        strip_ansi(result.output).split()
    )
    if existing:
        assert output_file.read_bytes() == sentinel
    else:
        assert not output_file.exists()


@pytest.mark.requires_symlink
def test_generate_halt_does_not_save_through_escaped_symlink(tmp_path, monkeypatch):
    """Re-resolve the output at partial save time rather than trust its old path."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    outside = tmp_path / "outside.jsonl"
    sentinel = b"outside dataset\n"
    outside.write_bytes(sentinel)
    output_file = work / "output.jsonl"
    calls = 0

    def generate_then_redirect(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"instruction": "Q", "output": "A"}]
        if calls == 4:
            output_file.symlink_to(outside)
        return []

    with patch("soup_cli.commands.generate._generate_batch", side_effect=generate_then_redirect):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test",
                "--count",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert calls == 4
    assert "after 3 consecutive attempts (1/5 generated)" in " ".join(
        strip_ansi(result.output).split()
    )
    assert output_file.is_symlink()
    assert outside.read_bytes() == sentinel
