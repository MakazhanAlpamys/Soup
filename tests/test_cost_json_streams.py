"""Errors from cost --json belong to stderr, leaving stdout for records."""

import json

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize("failure", ["missing", "empty", "invalid_yaml", "schema", "gpu"])
def test_cost_refusal_streams(tmp_path, failure, json_mode):
    path = tmp_path / "config.yaml"
    if failure == "empty":
        path.write_text("", encoding="utf-8")
    elif failure == "invalid_yaml":
        path.write_text("base: [", encoding="utf-8")
    elif failure == "schema":
        path.write_text("base: tiny\ntraining:\n  batch_size: 0\n", encoding="utf-8")
    elif failure == "gpu":
        path.write_text("base: tiny\ndata:\n  train: missing.jsonl\n", encoding="utf-8")

    args = ["cost", "--config", str(path)]
    if json_mode:
        args.append("--json")
    if failure == "gpu":
        args.extend(["--gpu", "no-such-gpu"])

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    diagnostic = result.stderr if json_mode else result.stdout
    assert strip_ansi(diagnostic).strip()
    if json_mode:
        assert result.stdout == ""
    else:
        assert result.stderr == ""
    expected = {
        "missing": "Config file not found:",
        "empty": "Config file is empty",
        "invalid_yaml": "config.yaml",
        "schema": "batch_size",
        "gpu": "No matching GPUs found for:",
    }
    assert expected[failure] in strip_ansi(diagnostic)


def test_cost_success_keeps_a_json_document_on_stdout(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("base: tiny\ndata:\n  train: missing.jsonl\n", encoding="utf-8")

    result = runner.invoke(app, ["cost", "--config", str(path), "--json", "--gpu", "H100"])

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 1
    assert rows[0]["gpu"] == "H100"
    assert rows[0]["total_cost"] >= 0
    assert result.stderr == ""
