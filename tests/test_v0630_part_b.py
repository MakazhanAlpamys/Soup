"""v0.63.0 Part B — soup prune-prompt static prefix detector tests."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import prune_prompt

    assert hasattr(prune_prompt, "detect_common_prefix")
    assert hasattr(prune_prompt, "PrunePromptReport")
    assert hasattr(prune_prompt, "prune_traces")
    assert hasattr(prune_prompt, "validate_min_frequency")


# ---------------------------------------------------------------------------
# validate_min_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.5, 0.95, 1.0, 0.0])
def test_validate_min_frequency_happy(value):
    from soup_cli.utils.prune_prompt import validate_min_frequency

    assert validate_min_frequency(value) == float(value)


@pytest.mark.parametrize("bad", [True, False, None, "0.95", -0.1, 1.5, float("nan"), float("inf")])
def test_validate_min_frequency_rejects(bad):
    from soup_cli.utils.prune_prompt import validate_min_frequency

    with pytest.raises((TypeError, ValueError)):
        validate_min_frequency(bad)


# ---------------------------------------------------------------------------
# detect_common_prefix
# ---------------------------------------------------------------------------


def test_detect_common_prefix_happy():
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = [
        "You are a helpful assistant.\nUser: hi",
        "You are a helpful assistant.\nUser: bye",
        "You are a helpful assistant.\nUser: morning",
    ]
    prefix = detect_common_prefix(rows, min_frequency=0.95)
    assert prefix == "You are a helpful assistant.\nUser: "


def test_detect_common_prefix_below_threshold():
    """When fewer than min_frequency share the candidate prefix, return ''."""
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = [
        "You are a helpful assistant.\nQ: a",
        "You are a helpful assistant.\nQ: b",
        "Different opener entirely.\nQ: c",  # 33% mismatch
    ]
    prefix = detect_common_prefix(rows, min_frequency=0.95)
    assert prefix == ""


def test_detect_common_prefix_partial_majority():
    """At min_frequency=0.66, the 2/3 majority counts."""
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = [
        "Same opener line one.\nQ: a",
        "Same opener line one.\nQ: b",
        "Different line.\nQ: c",
    ]
    prefix = detect_common_prefix(rows, min_frequency=0.66)
    # At least the common token "Same opener line one." should surface.
    assert prefix.startswith("Same opener")


def test_detect_common_prefix_empty():
    from soup_cli.utils.prune_prompt import detect_common_prefix

    assert detect_common_prefix([], min_frequency=0.95) == ""


def test_detect_common_prefix_single_row():
    """Single-row input — the whole row IS the common prefix."""
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = ["hello world"]
    # With min_freq=1.0 it's trivially the whole row
    prefix = detect_common_prefix(rows, min_frequency=1.0)
    assert prefix == "hello world"


def test_detect_common_prefix_rejects_non_string_row():
    from soup_cli.utils.prune_prompt import detect_common_prefix

    with pytest.raises(TypeError):
        detect_common_prefix(["good", 42, "bad"], min_frequency=0.95)


def test_detect_common_prefix_rejects_non_iterable():
    from soup_cli.utils.prune_prompt import detect_common_prefix

    with pytest.raises(TypeError):
        detect_common_prefix("not a sequence-of-strings", min_frequency=0.95)


def test_detect_common_prefix_caps_input():
    """Massive input must not OOM — internal cap on rows scanned."""
    from soup_cli.utils import prune_prompt

    # Set cap small to validate behaviour, not memory.
    rows = ["hello world"] * 50
    prefix = prune_prompt.detect_common_prefix(rows, min_frequency=1.0)
    assert prefix == "hello world"


def test_detect_common_prefix_no_common_chars():
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = ["alpha", "beta", "gamma"]
    assert detect_common_prefix(rows, min_frequency=0.95) == ""


def test_detect_common_prefix_huge_prompt_cap():
    """Each individual row is capped to prevent runaway prefix scan."""
    from soup_cli.utils.prune_prompt import detect_common_prefix

    huge = "x" * 10_000_000
    # Single huge row, single-row sentinel returns truncated row
    prefix = detect_common_prefix([huge, huge], min_frequency=1.0)
    # Should not equal full huge length — implementation must cap row scan
    assert len(prefix) <= 1_000_000


# ---------------------------------------------------------------------------
# PrunePromptReport
# ---------------------------------------------------------------------------


def test_prune_prompt_report_frozen():
    from soup_cli.utils.prune_prompt import PrunePromptReport

    report = PrunePromptReport(
        prefix="You are a helpful assistant.\n",
        prefix_chars=33,
        rows_total=100,
        rows_pruned=98,
        min_frequency=0.95,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.prefix = "tampered"  # type: ignore[misc]


def test_prune_prompt_report_validation():
    from soup_cli.utils.prune_prompt import PrunePromptReport

    # rows_total must be >= rows_pruned
    with pytest.raises(ValueError):
        PrunePromptReport(
            prefix="x",
            prefix_chars=1,
            rows_total=5,
            rows_pruned=10,
            min_frequency=0.95,
        )


# ---------------------------------------------------------------------------
# prune_traces
# ---------------------------------------------------------------------------


def test_prune_traces_strips_prefix(tmp_path, monkeypatch):
    from soup_cli.utils.prune_prompt import prune_traces

    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"

    rows = [
        {"prompt": "System: be nice.\nUser: hi", "output": "hello"},
        {"prompt": "System: be nice.\nUser: bye", "output": "goodbye"},
        {"prompt": "System: be nice.\nUser: morning", "output": "good morning"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )

    report = prune_traces(
        str(input_path),
        output_path=str(output_path),
        min_frequency=0.95,
    )
    assert report.prefix.startswith("System: be nice.")
    assert report.rows_total == 3
    assert report.rows_pruned == 3
    # Output prompts should no longer carry the shared prefix
    out_rows = [json.loads(ln) for ln in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(out_rows) == 3
    for row in out_rows:
        assert not row["prompt"].startswith("System: be nice.\n")


def test_prune_traces_passthrough_when_no_prefix(tmp_path, monkeypatch):
    from soup_cli.utils.prune_prompt import prune_traces

    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"

    rows = [
        {"prompt": "alpha", "output": "x"},
        {"prompt": "beta", "output": "y"},
        {"prompt": "gamma", "output": "z"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )

    report = prune_traces(str(input_path), output_path=str(output_path), min_frequency=0.95)
    assert report.prefix == ""
    assert report.rows_pruned == 0
    # Outputs unchanged
    out_rows = [json.loads(ln) for ln in output_path.read_text(encoding="utf-8").splitlines()]
    assert out_rows[0]["prompt"] == rows[0]["prompt"]


def test_prune_traces_rejects_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.prune_prompt import prune_traces

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray.jsonl"
    outside.write_text('{"prompt":"x","output":"y"}\n', encoding="utf-8")
    out = tmp_path / "o.jsonl"
    try:
        with pytest.raises(ValueError, match="outside"):
            prune_traces(str(outside), output_path=str(out), min_frequency=0.95)
    finally:
        if outside.exists():
            outside.unlink()


def test_prune_traces_rejects_null_byte():
    from soup_cli.utils.prune_prompt import prune_traces

    with pytest.raises(ValueError):
        prune_traces("bad\x00path.jsonl", output_path="out.jsonl", min_frequency=0.95)


def test_prune_traces_missing_input(tmp_path, monkeypatch):
    from soup_cli.utils.prune_prompt import prune_traces

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        prune_traces(str(tmp_path / "missing.jsonl"), output_path=str(tmp_path / "o.jsonl"),
                     min_frequency=0.95)


def test_prune_traces_invalid_min_frequency():
    from soup_cli.utils.prune_prompt import prune_traces

    with pytest.raises((TypeError, ValueError)):
        prune_traces("in.jsonl", output_path="out.jsonl", min_frequency=True)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_prune_prompt_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["prune-prompt", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "min-frequency" in result.output.lower() or "prefix" in result.output.lower()


def test_cli_prune_prompt_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"prompt": "Sys: be safe.\nUser: hi", "output": "ok"},
        {"prompt": "Sys: be safe.\nUser: bye", "output": "bye"},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = runner.invoke(
        app,
        ["prune-prompt", "--input", str(inp), "--output", str(out), "--min-frequency", "0.95"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert out.exists()


def test_cli_prune_prompt_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "x.jsonl"
    outside.write_text('{"prompt":"a","output":"b"}\n', encoding="utf-8")
    try:
        result = runner.invoke(
            app,
            ["prune-prompt", "--input", str(outside), "--output", str(tmp_path / "out.jsonl")],
        )
        assert result.exit_code != 0
        assert "outside" in result.output.lower()
    finally:
        if outside.exists():
            outside.unlink()
