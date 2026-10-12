"""Dataset previews must not expose terminal control bytes from untrusted input."""

import ast
import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def assert_no_data_control_bytes(output: str) -> None:
    # Rich may emit its own SGR styling when FORCE_COLOR=1.
    unstyled = ANSI_SGR.sub("", output)
    assert "\x1b" not in unstyled
    assert "\x07" not in unstyled


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
    assert_no_data_control_bytes(result.output)


def test_inspect_sample_header_strips_control_bytes(tmp_path):
    dataset = tmp_path / "columns.jsonl"
    dataset.write_text(json.dumps({"col\x1b]0;title\x07": "safe"}) + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "1"])

    assert result.exit_code == 0, result.output
    assert "safe" in result.output
    # The separate stats Columns row is handled by #1602; guard the sample header here.
    assert_no_data_control_bytes(ANSI_SGR.sub("", result.output).split("Sample rows (", 1)[1])


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
    assert_no_data_control_bytes(result.output)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows rejects control bytes in filenames")
def test_inspect_sanitizes_dataset_path(tmp_path):
    dataset = tmp_path / "data\x1b]0;title\x07.jsonl"
    dataset.write_text(json.dumps({"instruction": "hello", "output": "ok"}) + "\n")

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "0"])

    assert result.exit_code == 0, result.output
    assert "Inspecting" in result.output
    assert_no_data_control_bytes(result.output)


def test_inspect_sanitizes_missing_dataset_path(tmp_path):
    dataset = tmp_path / "missing\x1b]0;title\x07.jsonl"

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset)])

    assert result.exit_code == 1
    assert "File not found" in result.output
    assert_no_data_control_bytes(result.output)


def test_inspect_sanitizes_vision_format(tmp_path):
    image_name = "image.\x1b]0;title\x07png"
    dataset = tmp_path / "vision.jsonl"
    dataset.write_text(json.dumps({"image": image_name, "text": "hello"}) + "\n")

    result = CliRunner().invoke(app, ["data", "inspect", str(dataset), "--rows", "0"])

    assert result.exit_code == 0, result.output
    assert "Image formats" in result.output
    assert_no_data_control_bytes(result.output)


def test_data_preview_sinks_keep_terminal_sanitizer():
    """Catch new or reverted unsanitized dataset table/panel sinks in this scope."""
    source = Path(__file__).resolve().parents[1] / "src/soup_cli/commands/data.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}

    def calls(function, object_name, method):
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == object_name
            and node.func.attr == method
        ]

    def sanitized(expression):
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "for_terminal"
            for node in ast.walk(expression)
        )

    inspect = functions["inspect"]
    headers = calls(inspect, "sample_table", "add_column")
    assert len(headers) == 1 and sanitized(headers[0].args[0])
    values = [
        node
        for node in ast.walk(inspect)
        if isinstance(node, ast.ListComp)
        and any(isinstance(child, ast.Name) and child.id == "row" for child in ast.walk(node))
    ]
    assert len(values) == 1 and sanitized(values[0].elt)
    panels = [
        node
        for node in ast.walk(functions["review_cmd"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Panel"
    ]
    assert len(panels) == 1
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "for_terminal"
            for node in ast.walk(panels[0].args[0])
        )
        == 3
    )
    formats = [
        call
        for call in calls(functions["_show_vision_stats"], "vision_table", "add_row")
        if isinstance(call.args[0], ast.Constant) and call.args[0].value == "Image formats"
    ]
    assert len(formats) == 1 and sanitized(formats[0].args[1])
