"""Dataset preview output must not carry terminal control bytes."""

import json

from typer.testing import CliRunner

from soup_cli.cli import app


def test_inspect_sample_rows_strip_control_bytes(tmp_path):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "hello\x1b]0;title\x07 [bold]world[/]", "output": "ok"}) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "1"])

    assert result.exit_code == 0, result.output
    assert "hello" in result.output
    assert "world" in result.output
    assert "\x1b" not in result.output
    assert "\x07" not in result.output


def test_review_preview_strips_control_bytes(tmp_path):
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": "p\x1b]0;x\x07",
                "chosen": "c\x1b]0;x\x07",
                "rejected": "r\x1b]0;x\x07",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["data", "review", str(dataset), "--sample", "1"])

    assert result.exit_code == 0, result.output
    assert "Prompt:" in result.output
    assert "Chosen:" in result.output
    assert "Rejected:" in result.output
    assert "\x1b" not in result.output
    assert "\x07" not in result.output


def test_inspect_sanitizes_dataset_path(tmp_path):
    dataset = tmp_path / "data\x1b]0;title\x07.jsonl"
    dataset.write_text(json.dumps({"instruction": "hello", "output": "ok"}) + "\n")

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "0"])

    assert result.exit_code == 0, result.output
    assert "Inspecting" in result.output
    assert "\x1b" not in result.output
    assert "\x07" not in result.output


def test_inspect_sanitizes_vision_format(tmp_path):
    image_name = "image.\x1b]0;title\x07png"
    (tmp_path / image_name).write_bytes(b"image")
    dataset = tmp_path / "vision.jsonl"
    dataset.write_text(json.dumps({"image": image_name, "text": "hello"}) + "\n")

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "0"])

    assert result.exit_code == 0, result.output
    assert "Image formats" in result.output
    assert "\x1b" not in result.output
    assert "\x07" not in result.output
