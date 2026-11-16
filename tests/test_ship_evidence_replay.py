"""Replay refuses a task side whose every recorded generation failed (#2899).

The live ship path refuses such a side in ``_refuse_a_dead_side`` (commands/ship.py), so
a file written by a live run cannot carry a dead side. A hand-made or edited evidence
file could, and before the fix it replayed to a verdict: a dead base with a positive
tuned score printed SHIP. The shared reader now applies the same rule the live path
applies, reusing the first part of the live message (the pinned shape a program can
match) and dropping the "Check that the model loads" tail, which means nothing for a
file. The refusal is a ValueError, so replay exits 3 (usage), not 1.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.mcp_server.registry import McpToolError, tool_ship_evidence
from soup_cli.utils.ship_verdict import verdict_from_evidence, verdict_to_dict
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()
THRESHOLD = 0.05

#: The part the live refusal shares: the message a program matches stops here.
_DEAD_SIDE_REPLAY = (
    "every task generation failed on the {side} side ({failed} of {n})"
)

V2_EVIDENCE = {
    "schema_version": 2,
    "task": {
        "mode": "metric",
        "base": 0.0,
        "tuned": 0.75,
        "generation_failures": {"base": 8, "tuned": 0, "n": 8},
    },
    "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
}


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _with(base: dict, *args) -> dict:
    """Apply edits of two shapes: ``(path_tuple, value)`` pairs or ``path..., value`` sequences.

    ``...`` in place of a value deletes the key the path names.
    """
    edits: list = []
    i = 0
    while i < len(args):
        arg = args[i]
        if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], tuple):
            edits.append((list(arg[0]), arg[1]))
            i += 1
        else:
            edits.append((list(args[i:-1]), args[-1]))
            break
    out = copy.deepcopy(base)
    for path, value in edits:
        node = out
        for key in path[:-1]:
            node = node[key]
        if value is ...:
            del node[path[-1]]
        else:
            node[path[-1]] = value
    return out


def _cli(payload, *extra):
    """Replay ``payload`` through the CLI; returns (result, written verdict json or None)."""
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(
            ship_cmd.app, ["--evidence", "ev.json", "--output", "out.json", *extra],
        )
        out = Path("out.json")
        verdict = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return result, verdict


def _dead_message(side: str, failed: int, n: int) -> str:
    """The literal message text (escape it before use in ``pytest.raises(match=...)``)."""
    return _DEAD_SIDE_REPLAY.format(side=side, failed=failed, n=n)


# -- the reader refuses a dead side ---------------------------------------------------


@pytest.mark.parametrize("side", ["base", "tuned"])
def test_an_all_failed_side_is_refused(side):
    other = "tuned" if side == "base" else "base"
    payload = _with(
        V2_EVIDENCE,
        (("task", "generation_failures"),
         {"base": 8, "tuned": 8, "n": 8} if side == "base" else {side: 8, other: 0, "n": 8}),
    )
    with pytest.raises(ValueError, match=re.escape(_dead_message(side, 8, 8))):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


def test_both_sides_failed_is_refused_naming_base_first():
    payload = _with(V2_EVIDENCE, (("task", "generation_failures"), {"base": 8, "tuned": 8, "n": 8}))
    with pytest.raises(ValueError, match=re.escape(_dead_message("base", 8, 8))):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


def test_the_smallest_file_1_of_1_is_refused():
    payload = _with(
        V2_EVIDENCE,
        (("task", "generation_failures"), {"base": 1, "tuned": 0, "n": 1}),
        (("task", "base"), 0.0),
        (("task", "tuned"), 1.0),
    )
    with pytest.raises(ValueError, match=re.escape(_dead_message("base", 1, 1))):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


def test_n_minus_1_of_n_still_replays():
    payload = _with(V2_EVIDENCE, (("task", "generation_failures"), {"base": 7, "tuned": 0, "n": 8}))
    verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert verdict_to_dict(verdict)["decision"] == "SHIP"


# -- the controls stay valid ----------------------------------------------------------


def test_a_measured_zero_with_no_failures_still_replays():
    payload = _with(V2_EVIDENCE, (("task", "generation_failures"), {"base": 0, "tuned": 0, "n": 8}))
    verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert verdict_to_dict(verdict)["decision"] == "SHIP"


def test_a_schema_1_file_is_unchanged():
    payload = _with(V2_EVIDENCE, (("schema_version",), ...), (("task", "generation_failures"), ...))
    verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert verdict_to_dict(verdict)["decision"] == "SHIP"


def test_a_partial_block_absent_still_replays():
    payload = _with(V2_EVIDENCE, (("task", "generation_failures"), ...))
    verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert verdict_to_dict(verdict)["decision"] == "SHIP"


# -- judge_score and pairwise: same rule, by design ------------------------------------


def test_judge_score_all_failed_base_is_refused():
    payload = _with(V2_EVIDENCE, (("task", "mode"), "judge_score"))
    with pytest.raises(ValueError, match=re.escape(_dead_message("base", 8, 8))):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


def test_pairwise_all_failed_tuned_is_refused():
    payload = _with(
        V2_EVIDENCE,
        (("task", "mode"), "pairwise"),
        (("task", "generation_failures"), {"base": 0, "tuned": 8, "n": 8}),
    )
    with pytest.raises(ValueError, match=re.escape(_dead_message("tuned", 8, 8))):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


# -- the CLI surface ------------------------------------------------------------------


def test_the_cli_replay_exits_3_with_the_message():
    result, verdict = _cli(V2_EVIDENCE)
    assert result.exit_code == 3
    assert verdict is None
    plain = _plain(result.output)
    assert _dead_message("base", 8, 8) in plain


# -- the MCP surface ------------------------------------------------------------------


def test_the_mcp_ship_evidence_error_carries_the_message():
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(V2_EVIDENCE), encoding="utf-8")
        with pytest.raises(McpToolError) as exc_info:
            tool_ship_evidence({"evidence": "ev.json"})
    assert _dead_message("base", 8, 8) in str(exc_info.value)
