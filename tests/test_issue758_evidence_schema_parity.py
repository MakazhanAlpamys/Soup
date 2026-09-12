"""Regression contract for issue #758: all evidence readers stay in lockstep."""

from __future__ import annotations

import copy
import json

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.commands import ship as ship_cmd
from soup_cli.commands.ship import _verdict_from_evidence
from soup_cli.mcp_server.registry import McpToolError, tool_ship_evidence
from soup_cli.utils.ship_verdict import (
    DECISION_SHIP,
    EVIDENCE_SCHEMA_FIELDS,
    floor_exceeds_threshold,
    verdict_to_dict,
    verdict_to_evidence,
)

FORGETTING_THRESHOLD = 0.05
ALTERNATE_FORGETTING_THRESHOLD = 0.10
EVIDENCE_FILENAME = "evidence.json"
CLI_OUTPUT_FILENAME = "cli-verdict.json"
PROVENANCE_FIELD = "provenance"


EVIDENCE_CORPUS = (
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.80},
        "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
    },
    {
        "task": {"mode": "judge_score", "base": 0.45, "tuned": 0.65},
        "benchmarks": {"mini_math": {"base": 0.80, "tuned": 0.72}},
        "noise_floor": {
            "runs": 2,
            "floors": {"__task__": 0.02, "mini_math": 0.10},
            "judge_inclusive": True,
        },
        "provenance": {"config_sha": "a" * 64},
    },
    {
        "task": {"mode": "pairwise", "base": 0.50, "tuned": 0.50},
        "benchmarks": {"mini_safety": {"base": 0.95, "tuned": 0.70}},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.80},
        "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
        "numerics": "4bit",
    },
)


def test_corpus_covers_every_registered_evidence_field():
    """Every current optional field has a non-default parity example."""
    observed = {
        "root": set().union(*(evidence.keys() for evidence in EVIDENCE_CORPUS)),
        "task": set().union(
            *(evidence["task"].keys() for evidence in EVIDENCE_CORPUS)
        ),
        "benchmark": set().union(
            *(
                entry.keys()
                for evidence in EVIDENCE_CORPUS
                for entry in evidence["benchmarks"].values()
            )
        ),
        "noise_floor": set().union(
            *(
                evidence["noise_floor"].keys()
                for evidence in EVIDENCE_CORPUS
                if "noise_floor" in evidence
            )
        ),
    }

    assert observed == EVIDENCE_SCHEMA_FIELDS


MALFORMED_CORPUS = (
    [],
    {"benchmarks": {}},
    {"task": {"mode": "unknown", "base": 0.50, "tuned": 0.60}},
    {"task": {"mode": "metric", "base": 0.50}, "benchmarks": {}},
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": [],
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {"mini_mmlu": {"base": 0.70}},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": "invalid"}},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {},
        "noise_floor": {"runs": 1, "floors": {}},
    },
    {
        "task": {"mode": "metric", "base": "invalid", "tuned": 0.60},
        "benchmarks": {},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {},
        "future_optional": {},
    },
    {
        "task": {
            "mode": "metric",
            "base": 0.50,
            "tuned": 0.60,
            "future_optional": True,
        },
        "benchmarks": {},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {
            "mini_mmlu": {"base": 0.70, "tuned": 0.71, "future_optional": True}
        },
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.60},
        "benchmarks": {},
        "noise_floor": {"runs": 2, "floors": {}, "future_optional": True},
    },
    {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.80},
        "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
        "numerics": "nf4",
    },
)


def _write_evidence(tmp_path, payload: object) -> None:
    (tmp_path / EVIDENCE_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _mcp_verdict(tmp_path, monkeypatch, payload: object) -> dict:
    monkeypatch.chdir(tmp_path)
    _write_evidence(tmp_path, payload)
    return tool_ship_evidence(
        {
            "evidence": EVIDENCE_FILENAME,
            "forgetting_threshold": FORGETTING_THRESHOLD,
        }
    )


def _cli_verdict(tmp_path, monkeypatch, payload: object):
    monkeypatch.chdir(tmp_path)
    _write_evidence(tmp_path, payload)
    result = CliRunner().invoke(
        ship_cmd.app,
        [
            "--evidence",
            EVIDENCE_FILENAME,
            "--forgetting-threshold",
            str(FORGETTING_THRESHOLD),
            "--output",
            CLI_OUTPUT_FILENAME,
        ],
    )
    output_path = tmp_path / CLI_OUTPUT_FILENAME
    output = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else None
    )
    return result, output


@pytest.mark.parametrize("evidence", EVIDENCE_CORPUS)
def test_all_readers_report_every_verdict_field_identically(
    tmp_path, monkeypatch, evidence
):
    """Both public surfaces must be projections of one canonical decoder."""
    from soup_cli.utils.ship_verdict import verdict_from_evidence

    canonical = verdict_from_evidence(
        evidence, forgetting_threshold=FORGETTING_THRESHOLD
    )
    cli_result, cli = _cli_verdict(
        tmp_path, monkeypatch, copy.deepcopy(evidence)
    )
    mcp = _mcp_verdict(tmp_path, monkeypatch, copy.deepcopy(evidence))
    mcp_warnings = mcp.pop("warnings")

    expected = verdict_to_dict(canonical)
    expected_exit = 0 if canonical.decision == DECISION_SHIP else 2
    assert cli_result.exit_code == expected_exit, (
        cli_result.output,
        repr(cli_result.exception),
    )
    assert cli == expected
    assert mcp == expected

    widened = floor_exceeds_threshold(
        canonical.noise_floor, canonical.forgetting_threshold
    )
    assert bool(mcp_warnings) == bool(widened)
    assert ("LOOSER" in cli_result.output) == bool(widened)
    for name, value in widened:
        warning_text = "\n".join(mcp_warnings)
        for token in (
            name,
            f"{value:.4f}",
            f"{canonical.forgetting_threshold:.4f}",
            "LOOSER",
        ):
            assert token in cli_result.output
            assert token in warning_text


@pytest.mark.parametrize("evidence", EVIDENCE_CORPUS)
def test_schema_keys_are_derived_from_the_canonical_serializer(evidence):
    """A producer-added schema key cannot be silently ignored by a reader."""
    from soup_cli.utils.ship_verdict import verdict_from_evidence

    canonical = verdict_from_evidence(
        evidence, forgetting_threshold=FORGETTING_THRESHOLD
    )
    replay = verdict_to_evidence(
        canonical, provenance=evidence.get(PROVENANCE_FIELD)
    )

    assert replay == evidence


@pytest.mark.parametrize("evidence", MALFORMED_CORPUS)
def test_all_readers_refuse_the_same_malformed_corpus(
    tmp_path, monkeypatch, evidence
):
    """Malformed evidence is a refusal everywhere, never a partial verdict."""
    from soup_cli.utils.ship_verdict import verdict_from_evidence

    with pytest.raises((TypeError, ValueError)):
        verdict_from_evidence(evidence, forgetting_threshold=FORGETTING_THRESHOLD)

    cli_result, cli_output = _cli_verdict(
        tmp_path, monkeypatch, copy.deepcopy(evidence)
    )
    assert cli_result.exit_code == 1, (
        cli_result.output,
        repr(cli_result.exception),
    )
    assert cli_output is None

    with pytest.raises(McpToolError):
        _mcp_verdict(tmp_path, monkeypatch, copy.deepcopy(evidence))


@pytest.mark.parametrize("threshold", (True, -0.01, 1.01, float("nan")))
def test_all_readers_refuse_the_same_invalid_threshold(
    tmp_path, monkeypatch, threshold
):
    """Threshold validation is part of the shared evidence-reader contract."""
    from soup_cli.utils.ship_verdict import verdict_from_evidence

    evidence = copy.deepcopy(EVIDENCE_CORPUS[0])
    with pytest.raises((TypeError, ValueError)):
        verdict_from_evidence(evidence, forgetting_threshold=threshold)

    with pytest.raises(typer.Exit) as cli_error:
        _verdict_from_evidence(evidence, forgetting_threshold=threshold)
    assert cli_error.value.exit_code == 1

    monkeypatch.chdir(tmp_path)
    _write_evidence(tmp_path, evidence)
    with pytest.raises(McpToolError):
        tool_ship_evidence(
            {"evidence": EVIDENCE_FILENAME, "forgetting_threshold": threshold}
        )


def test_mcp_omitted_threshold_uses_canonical_default(tmp_path, monkeypatch):
    """The MCP default must follow the canonical decoder rather than drift."""
    from soup_cli.utils import ship_verdict

    monkeypatch.setattr(
        ship_verdict,
        "DEFAULT_FORGETTING_THRESHOLD",
        ALTERNATE_FORGETTING_THRESHOLD,
    )
    monkeypatch.chdir(tmp_path)
    _write_evidence(tmp_path, copy.deepcopy(EVIDENCE_CORPUS[0]))

    verdict = tool_ship_evidence({"evidence": EVIDENCE_FILENAME})

    assert verdict["forgetting_threshold"] == ALTERNATE_FORGETTING_THRESHOLD
