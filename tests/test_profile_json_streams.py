"""Errors from profile --json belong to stderr, leaving stdout for the document."""

import json
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.utils import mlx as mlx_utils
from tests._tiny_checkpoint import TINY_LLAMA
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

_VALID_CONFIG = (
    "base: meta-llama/Llama-3.1-8B-Instruct\n"
    "task: sft\n"
    "data:\n"
    "  train: ./data/train.jsonl\n"
    "  max_length: 2048\n"
    "training:\n"
    "  batch_size: 4\n"
    "  quantization: 4bit\n"
    "output: ./output\n"
)


@pytest.fixture(autouse=True)
def _not_a_mac(monkeypatch):
    """No test here may read this machine's GPU."""
    monkeypatch.setattr(mlx_utils, "is_apple_silicon", lambda: False)


def _tiny_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(TINY_LLAMA), encoding="utf-8")
    return directory


def _config(tmp_path: Path, failure: str) -> Path:
    path = tmp_path / ("config." + "x" * 80 + ".yaml")
    if failure == "empty":
        path.write_text("", encoding="utf-8")
    elif failure == "invalid_yaml":
        path.write_text("base: [", encoding="utf-8")
    elif failure == "schema":
        path.write_text("base: tiny\ntraining:\n  batch_size: 0\n", encoding="utf-8")
    elif failure == "unknown_gpu":
        path.write_text(_VALID_CONFIG, encoding="utf-8")
    return path


def _assert_streams(result, json_mode: bool) -> str:
    """Exit 1 with the diagnostic on stderr under --json, on stdout without it."""
    assert result.exit_code == 1
    diagnostic = result.stderr if json_mode else result.stdout
    plain = " ".join(strip_ansi(diagnostic).split())
    assert plain
    if json_mode:
        assert result.stdout == ""
    else:
        assert result.stderr == ""
    return plain


@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize("failure", ["missing", "empty", "invalid_yaml", "schema", "unknown_gpu"])
@pytest.mark.parametrize("width", [40, 80])
def test_profile_config_refusal_streams(tmp_path, monkeypatch, failure, json_mode, width):
    monkeypatch.setenv("COLUMNS", str(width))
    path = _config(tmp_path, "missing" if failure == "missing" else failure)

    args = ["profile", "--config", str(path)]
    if failure == "unknown_gpu":
        args.extend(["--gpu", "no-such-gpu"])
    if json_mode:
        args.append("--json")

    plain = _assert_streams(runner.invoke(app, args), json_mode)

    expected = {
        "missing": "Config file not found:",
        "empty": "Config file is empty",
        "invalid_yaml": "is not valid YAML:",
        "schema": "batch_size",
        "unknown_gpu": "Unknown GPU:",
    }
    assert expected[failure] in plain


@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--config", "soup.yaml"], "--model and --config cannot be used together."),
        (["--method", "full", "--lora-r", "8"], "--lora-r only applies with --method"),
        (["--method", "bogus"], "Unknown --method:"),
        (["--optimizer", "bogus"], "Unknown --optimizer:"),
        (["--gpu", "no-such-gpu"], "Unknown GPU:"),
    ],
)
def test_profile_model_refusal_streams(tmp_path, json_mode, extra, expected):
    model = _tiny_model(tmp_path / "tiny")
    args = ["profile", "--model", str(model), *extra]
    if json_mode:
        args.append("--json")

    plain = _assert_streams(runner.invoke(app, args), json_mode)

    assert expected in plain


@pytest.mark.parametrize("json_mode", [True, False])
def test_profile_model_query_flag_without_model_streams(tmp_path, json_mode):
    path = tmp_path / "soup.yaml"
    path.write_text(_VALID_CONFIG, encoding="utf-8")
    args = ["profile", "--config", str(path), "--seq", "128"]
    if json_mode:
        args.append("--json")

    plain = _assert_streams(runner.invoke(app, args), json_mode)

    assert "--seq only applies with --model" in plain


@pytest.mark.parametrize("json_mode", [True, False])
def test_profile_unreadable_model_streams(tmp_path, monkeypatch, json_mode):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    args = ["profile", "--model", str(tmp_path / "no-such-model")]
    if json_mode:
        args.append("--json")

    plain = _assert_streams(runner.invoke(app, args), json_mode)

    assert "Nothing was estimated" in plain


def test_profile_config_success_keeps_a_json_document_on_stdout(tmp_path):
    path = tmp_path / "soup.yaml"
    path.write_text(_VALID_CONFIG, encoding="utf-8")

    result = runner.invoke(app, ["profile", "--config", str(path), "--json", "--gpu", "a100"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["total_memory_gb"] > 0
    assert result.stderr == ""


def test_profile_model_success_keeps_a_json_document_on_stdout(tmp_path):
    model = _tiny_model(tmp_path / "tiny")

    result = runner.invoke(app, ["profile", "--model", str(model), "--json", "--gpu", "a100"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["total_memory_gb"] > 0
    assert result.stderr == ""
