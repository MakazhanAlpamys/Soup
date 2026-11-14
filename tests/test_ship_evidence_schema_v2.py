"""Evidence schema version 2: ``--emit-evidence`` carries the task detail, the reader checks it.

A v2 file is the v1 file plus ``schema_version: 2``, ``task.rows``, ``task.truncated``,
``task.generation_failures`` and ``judged``. The reader reads an absent ``schema_version`` as 1
(a file written by soup-cli 0.77.x), accepts 2, and refuses anything else with the key named. The
truncation warning is never stored: it is computed from the counts on every read, so a stored
verdict cannot disagree with its own counts. A soup-cli 0.77.x reader refuses a v2 file with
``unsupported field(s)``, which is the documented migration.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.mcp_server.registry import McpToolError, tool_ship_evidence
from soup_cli.utils.ship_verdict import (
    DECISION_DONT_SHIP,
    DECISION_SHIP,
    EVIDENCE_SCHEMA_FIELDS,
    SCHEMA_VERSION,
    WARNING_TRUNCATION_GROWTH,
    GenerationFailures,
    Judged,
    TaskRows,
    Truncation,
    verdict_from_evidence,
    verdict_to_dict,
    verdict_to_evidence,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi
from tests.test_ship_task_detail import _answers, _scripted, _tasks

runner = CliRunner()
THRESHOLD = 0.05

V1_EVIDENCE = {
    "task": {"mode": "metric", "base": 0.50, "tuned": 0.80},
    "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
}

V2_EVIDENCE = {
    "schema_version": 2,
    "task": {
        "mode": "metric",
        "base": 0.75,
        "tuned": 0.9,
        "rows": {"better": 25, "worse": 17, "same": 22, "n": 64},
        "truncated": {"base": 5, "tuned": 14, "n": 64, "max_new_tokens": 256},
        "generation_failures": {"base": 0, "tuned": 0, "n": 64},
    },
    "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.72}},
    "judged": {
        "base": "base-model",
        "adapter": "out/checkpoint-12",
        "tuned": None,
        "base_adapter": None,
        "checkpoint_step": 12,
    },
}


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _with(base: dict, *path_and_value) -> dict:
    """A deep copy of ``base`` with ``path -> value`` replaced; ``...`` as value deletes."""
    out = copy.deepcopy(base)
    *path, value = path_and_value
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


# -- the reader ------------------------------------------------------------------------------


def test_the_version_constant_is_2():
    assert SCHEMA_VERSION == 2


def test_a_v2_file_round_trips_to_the_same_file_and_the_same_verdict():
    verdict = verdict_from_evidence(V2_EVIDENCE, forgetting_threshold=THRESHOLD)
    assert verdict_to_evidence(verdict) == V2_EVIDENCE
    again = verdict_from_evidence(verdict_to_evidence(verdict), forgetting_threshold=THRESHOLD)
    assert again == verdict
    detail = verdict.task_win.detail
    assert detail.rows == TaskRows(better=25, worse=17, same=22)
    assert detail.truncated == Truncation(base=5, tuned=14, n=64, max_new_tokens=256)
    assert detail.generation_failures == GenerationFailures(base=0, tuned=0, n=64)
    assert verdict.judged == Judged(
        base="base-model", adapter="out/checkpoint-12", tuned=None, base_adapter=None,
        checkpoint_step=12,
    )


def test_the_warning_is_recomputed_on_read_and_never_stored():
    verdict = verdict_from_evidence(V2_EVIDENCE, forgetting_threshold=THRESHOLD)
    assert verdict.task_win.detail.truncated.warning == WARNING_TRUNCATION_GROWTH  # 5 -> 14 of 64
    assert "warning" not in verdict_to_evidence(verdict)["task"]["truncated"]
    assert (
        verdict_to_dict(verdict)["task_win"]["truncated"]["warning"] == WARNING_TRUNCATION_GROWTH
    )
    quiet = _with(V2_EVIDENCE, "task", "truncated", {"base": 5, "tuned": 11, "n": 64,
                                                      "max_new_tokens": 256})
    assert verdict_from_evidence(quiet).task_win.detail.truncated.warning is None


def test_a_stored_warning_is_refused_because_it_could_contradict_the_counts():
    payload = _with(V2_EVIDENCE, "task", "truncated", {
        "base": 5, "tuned": 6, "n": 64, "max_new_tokens": 256, "warning": "truncation_growth",
    })
    with pytest.raises(ValueError, match="unsupported field"):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


@pytest.mark.parametrize("version_key", [False, True])
def test_a_v1_file_reads_as_it_always_did(version_key):
    payload = copy.deepcopy(V1_EVIDENCE)
    if version_key:
        payload["schema_version"] = 1  # explicit 1 is the same as absent
    verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert verdict.task_win.detail is None
    assert verdict.judged is None
    assert verdict.decision == DECISION_SHIP
    data = verdict_to_dict(verdict)
    assert data["task_win"]["rows"] is None
    assert data["task_win"]["truncated"] is None
    assert data["judged"] is None
    assert data["schema_version"] == 2  # the --output format is v2; null says "not recorded"


def test_a_v1_file_re_emits_as_a_v1_file_because_there_is_nothing_new_to_write():
    verdict = verdict_from_evidence(V1_EVIDENCE, forgetting_threshold=THRESHOLD)
    assert verdict_to_evidence(verdict) == V1_EVIDENCE  # no schema_version: 0.77.x can read it


@pytest.mark.parametrize(
    "value",
    [3, 0, -1, 1.0, 2.0, True, False, "2", "1", None, [2], {"v": 2}, 10**30],
    ids=repr,
)
def test_any_schema_version_but_1_or_2_is_refused_naming_the_key(value):
    payload = _with(V2_EVIDENCE, "schema_version", value)
    with pytest.raises(ValueError, match="schema_version"):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)


@pytest.mark.parametrize(
    ("where", "key", "value"),
    [
        (None, "judged", V2_EVIDENCE["judged"]),
        ("task", "rows", V2_EVIDENCE["task"]["rows"]),
        ("task", "truncated", V2_EVIDENCE["task"]["truncated"]),
        ("task", "generation_failures", V2_EVIDENCE["task"]["generation_failures"]),
    ],
)
@pytest.mark.parametrize("declared", [None, 1])
def test_v2_keys_under_schema_version_1_are_refused_naming_the_key(where, key, value, declared):
    payload = copy.deepcopy(V1_EVIDENCE)
    if declared is not None:
        payload["schema_version"] = declared
    (payload if where is None else payload[where])[key] = copy.deepcopy(value)
    with pytest.raises(ValueError) as error:
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    message = str(error.value)
    assert key in message
    assert "schema_version 2" in message


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (("task", "rows", "better"), -1, "task.rows.better"),
        (("task", "rows", "better"), True, "task.rows.better"),
        (("task", "rows", "better"), 1.5, "task.rows.better"),
        (("task", "rows", "n"), 65, "task.rows"),  # better + worse + same must equal n
        (("task", "rows", "n"), 10**6, "task.rows.n"),
        (("task", "rows", "extra"), 1, "unsupported field"),
        (("task", "rows"), [], "task.rows must be an object"),
        (("task", "rows", "same"), ..., "task.rows"),
        (("task", "truncated", "base"), 65, "task.truncated.base"),  # more than n
        (("task", "truncated", "tuned"), -1, "task.truncated.tuned"),
        (("task", "truncated", "tuned"), False, "task.truncated.tuned"),
        (("task", "truncated", "n"), 0, "task.truncated.n"),
        (("task", "truncated", "max_new_tokens"), 0, "task.truncated.max_new_tokens"),
        (("task", "truncated", "max_new_tokens"), True, "task.truncated.max_new_tokens"),
        (("task", "truncated", "max_new_tokens"), ..., "task.truncated"),
        (("task", "truncated", "n"), 63, "task.truncated.n"),  # rows.n is 64
        (("task", "generation_failures", "tuned"), 65, "task.generation_failures.tuned"),
        (("task", "generation_failures", "base"), -1, "task.generation_failures.base"),
        (("task", "generation_failures", "n"), 63, "task.generation_failures.n"),
        (("judged", "base"), None, "judged.base"),
        (("judged", "base"), 3, "judged.base"),
        (("judged", "adapter"), 3, "judged.adapter"),
        (("judged", "adapter"), "x" * 5000, "judged.adapter"),
        (("judged", "adapter"), "out/checkpoint-12\x00", "judged.adapter"),
        (("judged", "checkpoint_step"), True, "judged.checkpoint_step"),
        (("judged", "checkpoint_step"), -3, "judged.checkpoint_step"),
        (("judged", "checkpoint_step"), 99, "judged.checkpoint_step"),  # not what the path says
        (("judged", "checkpoint_step"), None, "judged.checkpoint_step"),  # path says 12
        (("judged", "extra"), 1, "unsupported field"),
        (("judged",), "out", "judged must be an object"),
    ],
    ids=lambda item: "-".join(map(str, item)) if isinstance(item, tuple) else None,
)
def test_malformed_v2_blocks_are_refused_naming_the_field(path, value, needle):
    payload = _with(V2_EVIDENCE, *path, value)
    with pytest.raises(ValueError) as error:
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    assert needle in str(error.value)


def test_checkpoint_step_must_be_null_when_the_path_has_no_step():
    payload = _with(V2_EVIDENCE, "judged", "adapter", "out/final")
    with pytest.raises(ValueError, match="judged.checkpoint_step"):
        verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
    ok = _with(payload, "judged", "checkpoint_step", None)
    assert verdict_from_evidence(ok).judged.checkpoint_step is None


def test_partial_v2_blocks_are_fine_because_each_block_is_optional():
    for key in ("rows", "truncated", "generation_failures"):
        payload = _with(V2_EVIDENCE, "task", key, ...)
        verdict = verdict_from_evidence(payload, forgetting_threshold=THRESHOLD)
        assert verdict_to_evidence(verdict) == payload
    no_judged = _with(V2_EVIDENCE, "judged", ...)
    assert verdict_from_evidence(no_judged).judged is None
    assert verdict_to_evidence(verdict_from_evidence(no_judged)) == no_judged


def test_unmeasured_counts_are_null_in_the_file_and_survive_a_round_trip():
    payload = _with(V2_EVIDENCE, "task", "truncated",
                    {"base": None, "tuned": None, "n": 64, "max_new_tokens": 256})
    verdict = verdict_from_evidence(payload)
    assert verdict.task_win.detail.truncated == Truncation(
        base=None, tuned=None, n=64, max_new_tokens=256,
    )
    assert verdict.task_win.detail.truncated.warning is None
    assert verdict_to_evidence(verdict) == payload


# -- the old reader (soup-cli 0.77.x) refuses a v2 file --------------------------------------

# A literal copy of the field sets of 0.77.2, and of the check it ran on them.
_OLD_FIELDS = {
    "root": frozenset({"task", "benchmarks", "noise_floor", "provenance", "numerics"}),
    "task": frozenset({"mode", "base", "tuned"}),
    "benchmark": frozenset({"base", "tuned"}),
    "noise_floor": frozenset({"runs", "floors", "judge_inclusive"}),
}


def _old_reader_check(payload: dict) -> None:
    def check(block, section, path):
        unknown = sorted(str(key) for key in block if key not in _OLD_FIELDS[section])
        if unknown:
            raise ValueError(
                f"{path} has unsupported field(s): " + ", ".join(repr(key) for key in unknown)
            )

    check(payload, "root", "evidence")
    check(payload["task"], "task", "evidence.task")


def test_the_0_77_reader_refuses_a_v2_file_and_names_schema_version():
    verdict = verdict_from_evidence(V2_EVIDENCE, forgetting_threshold=THRESHOLD)
    emitted = verdict_to_evidence(verdict)
    with pytest.raises(ValueError, match="unsupported field") as error:
        _old_reader_check(emitted)
    assert "'schema_version'" in str(error.value)
    assert "'judged'" in str(error.value)
    _old_reader_check(V1_EVIDENCE)  # the old reader still takes a v1 file


def test_the_registry_adds_exactly_the_v2_fields_to_the_0_77_sets():
    assert EVIDENCE_SCHEMA_FIELDS["root"] == _OLD_FIELDS["root"] | {"schema_version", "judged"}
    assert EVIDENCE_SCHEMA_FIELDS["task"] == _OLD_FIELDS["task"] | {
        "rows", "truncated", "generation_failures",
    }
    assert EVIDENCE_SCHEMA_FIELDS["benchmark"] == _OLD_FIELDS["benchmark"]
    assert EVIDENCE_SCHEMA_FIELDS["noise_floor"] == _OLD_FIELDS["noise_floor"]
    assert EVIDENCE_SCHEMA_FIELDS["rows"] == {"better", "worse", "same", "n"}
    assert EVIDENCE_SCHEMA_FIELDS["truncated"] == {"base", "tuned", "n", "max_new_tokens"}
    assert EVIDENCE_SCHEMA_FIELDS["generation_failures"] == {"base", "tuned", "n"}
    assert EVIDENCE_SCHEMA_FIELDS["judged"] == {
        "base", "adapter", "tuned", "base_adapter", "checkpoint_step",
    }


# -- the CLI and MCP readers -------------------------------------------------------------------


def test_cli_replay_prints_the_same_counts_and_the_same_warning():
    result, verdict = _cli(V2_EVIDENCE)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    plain = _plain(result.output)
    assert "Rows (tuned vs base): better 25, worse 17, same 22 of 64" in plain
    assert "Truncated answers: base 5, tuned 14 of 64 (cap 256 new tokens)" in plain
    assert "Warning truncation_growth: truncated 5 -> 14 of 64 (rise 9, threshold 7)" in plain
    assert "Judged: base base-model, adapter out/checkpoint-12 (checkpoint step 12)" in plain
    assert verdict["task_win"]["truncated"]["warning"] == "truncation_growth"
    assert verdict["decision"] == DECISION_SHIP  # the warning changed nothing
    assert verdict["schema_version"] == 2


def test_cli_replay_exit_codes_do_not_depend_on_the_warning():
    ship_with_warning, _ = _cli(V2_EVIDENCE)
    quiet = _with(V2_EVIDENCE, "task", "truncated", {"base": 5, "tuned": 6, "n": 64,
                                                      "max_new_tokens": 256})
    ship_without_warning, _ = _cli(quiet)
    assert ship_with_warning.exit_code == ship_without_warning.exit_code == 0
    losing = _with(V2_EVIDENCE, "task", "tuned", 0.1)
    dont_with, verdict = _cli(losing)
    assert dont_with.exit_code == 2, (dont_with.output, repr(dont_with.exception))
    assert verdict["decision"] == DECISION_DONT_SHIP
    assert verdict["task_win"]["truncated"]["warning"] == "truncation_growth"
    losing_quiet = _with(losing, "task", "truncated", {"base": 5, "tuned": 6, "n": 64,
                                                       "max_new_tokens": 256})
    dont_without, _ = _cli(losing_quiet)
    assert dont_without.exit_code == 2


@pytest.mark.parametrize(
    "payload",
    [
        _with(V2_EVIDENCE, "schema_version", 3),
        _with(V2_EVIDENCE, "schema_version", True),
        _with(V2_EVIDENCE, "schema_version", "2"),
        {**V1_EVIDENCE, "judged": V2_EVIDENCE["judged"]},
        {**V1_EVIDENCE, "schema_version": 1, "judged": V2_EVIDENCE["judged"]},
        _with(V1_EVIDENCE, "task", "rows", {"better": 0, "worse": 0, "same": 1, "n": 1}),
        _with(V2_EVIDENCE, "task", "rows", "n", 65),
    ],
    ids=["v3", "bool", "string", "judged-without-version", "judged-under-1", "rows-under-1",
         "rows-do-not-add-up"],
)
def test_cli_and_mcp_refuse_the_same_files(tmp_path, monkeypatch, payload):
    result, verdict = _cli(payload)
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert verdict is None
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ev.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(McpToolError) as error:
        tool_ship_evidence({"evidence": "ev.json"})
    message = str(error.value)
    cli_text = _plain(result.output)
    # both name the field they refused, in words that survive the MCP redaction of quoted text
    for token in ("schema_version", "judged", "task.rows"):
        if token in cli_text:
            assert token in message


def test_the_cli_refusal_for_a_future_version_says_what_to_do():
    result, _ = _cli(_with(V2_EVIDENCE, "schema_version", 3))
    plain = _plain(result.output)
    assert result.exit_code == 3
    assert "evidence.schema_version must be 1 or 2, got 3" in plain


def test_mcp_returns_the_same_dict_as_the_cli_for_a_v2_file(tmp_path, monkeypatch):
    _, cli_verdict = _cli(V2_EVIDENCE)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ev.json").write_text(json.dumps(V2_EVIDENCE), encoding="utf-8")
    mcp = tool_ship_evidence({"evidence": "ev.json", "forgetting_threshold": THRESHOLD})
    mcp.pop("warnings")
    assert mcp == cli_verdict


# -- a live run writes v2, and replaying it gives the same verdict ------------------------------


def _live_then_replay(monkeypatch, base_gen, tuned_gen, *, adapter="out/checkpoint-12"):
    def fake_resolve(base, tuned, adapter, device, quantization=None, **kwargs):
        return base_gen, tuned_gen

    monkeypatch.setattr(ship_cmd, "_resolve_generators", fake_resolve)
    with runner.isolated_filesystem():
        Path("task.jsonl").write_text(
            "\n".join(json.dumps(task) for task in _tasks()) + "\n", encoding="utf-8"
        )
        live = runner.invoke(
            ship_cmd.app,
            ["--base", "base-model", "--adapter", adapter, "--task-eval", "task.jsonl",
             "--general-suite", "mini_arithmetic", "--device", "cpu",
             "--output", "live.json", "--emit-evidence", "ev.json"],
        )
        replay = runner.invoke(
            ship_cmd.app, ["--evidence", "ev.json", "--output", "replay.json"],
        )
        files = {
            name: json.loads(Path(name).read_text(encoding="utf-8"))
            for name in ("live.json", "ev.json", "replay.json") if Path(name).exists()
        }
    return live, replay, files


def test_a_live_run_emits_v2_and_the_replay_is_identical(monkeypatch):
    base = _scripted(_answers(correct={0, 1, 2, 3}))
    tuned = _scripted(_answers(correct={0, 1, 2, 3, 4, 5, 6}, truncated={5, 6, 7}))
    live, replay, files = _live_then_replay(monkeypatch, base, tuned)
    assert live.exit_code == 0, (live.output, repr(live.exception))
    assert replay.exit_code == 0, (replay.output, repr(replay.exception))
    evidence = files["ev.json"]
    assert evidence["schema_version"] == 2
    assert evidence["task"]["rows"] == {"better": 3, "worse": 0, "same": 5, "n": 8}
    assert evidence["task"]["truncated"] == {"base": 0, "tuned": 3, "n": 8, "max_new_tokens": 256}
    assert evidence["task"]["generation_failures"] == {"base": 0, "tuned": 0, "n": 8}
    assert evidence["judged"]["checkpoint_step"] == 12
    assert "warning" not in evidence["task"]["truncated"]
    assert files["replay.json"] == files["live.json"]  # one verdict, written twice
    assert files["live.json"]["task_win"]["truncated"]["warning"] == "truncation_growth"
    for line in ("Rows (tuned vs base): better 3, worse 0, same 5 of 8",
                 "Truncated answers: base 0, tuned 3 of 8 (cap 256 new tokens)",
                 "Warning truncation_growth: truncated 0 -> 3 of 8 (rise 3, threshold 3)"):
        assert line in _plain(live.output)
        assert line in _plain(replay.output)


def test_a_replayed_v2_file_re_emits_the_same_v2_file():
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(V2_EVIDENCE), encoding="utf-8")
        result = runner.invoke(
            ship_cmd.app, ["--evidence", "ev.json", "--emit-evidence", "again.json"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        again = json.loads(Path("again.json").read_text(encoding="utf-8"))
    again.pop("provenance")
    assert again == V2_EVIDENCE


def test_replaying_a_v1_file_does_not_stamp_a_version_it_has_no_content_for():
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(V1_EVIDENCE), encoding="utf-8")
        result = runner.invoke(
            ship_cmd.app, ["--evidence", "ev.json", "--emit-evidence", "again.json"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        again = json.loads(Path("again.json").read_text(encoding="utf-8"))
    again.pop("provenance")
    assert again == V1_EVIDENCE
