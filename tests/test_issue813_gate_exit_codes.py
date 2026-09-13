"""Tests for Issue #813 — Gate commands exit-code unification.

Taxonomy contract:
  EXIT_OK = 0            (PASS / OK / SHIP / valid data)
  EXIT_GATE_FAILED = 2    (MAJOR / REGRESSION / DRIFT / DON'T SHIP / unusable data)
  EXIT_USAGE_ERROR = 3    (bad flag, missing or unparseable input file, empty series)
  EXIT_RUNTIME_ERROR = 1  (unexpected runtime failure)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_USAGE_ERROR,
)
from soup_cli.utils.soup_lock import SoupLock, compute_lock_closure, write_lock


def _setup_eval_gate(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_tasks_file = tmp_path / "gate_tasks_pass.jsonl"
    pass_tasks_file.write_text(
        json.dumps({"prompt": "ping", "expected": ""}) + "\n",
        encoding="utf-8",
    )

    fail_tasks_file = tmp_path / "gate_tasks_fail.jsonl"
    fail_tasks_file.write_text(
        json.dumps({"prompt": "ping", "expected": "definite_mismatch"}) + "\n",
        encoding="utf-8",
    )

    pass_suite = tmp_path / "gate_pass.yaml"
    pass_suite.write_text(
        yaml.safe_dump({
            "suite": "pass-gate",
            "tasks": [
                {
                    "type": "custom",
                    "name": "t_pass",
                    "scorer": "exact",
                    "threshold": 0.0,
                    "tasks": str(pass_tasks_file),
                }
            ],
        }),
        encoding="utf-8",
    )

    fail_suite = tmp_path / "gate_fail.yaml"
    fail_suite.write_text(
        yaml.safe_dump({
            "suite": "fail-gate",
            "tasks": [
                {
                    "type": "custom",
                    "name": "t_fail",
                    "scorer": "exact",
                    "threshold": 1.0,
                    "tasks": str(fail_tasks_file),
                }
            ],
        }),
        encoding="utf-8",
    )

    pass_args = ["eval", "gate", "--suite", str(pass_suite)]
    fail_args = ["eval", "gate", "--suite", str(fail_suite)]
    missing_args = ["eval", "gate", "--suite", str(tmp_path / "missing_suite.yaml")]
    return pass_args, fail_args, missing_args


def _setup_eval_against(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_args = [
        "eval", "against", "run-pass-base", "--candidate", "run-pass-cand",
    ]
    fail_args = [
        "eval", "against", "run-fail-base", "--candidate", "run-fail-cand",
    ]
    missing_args = [
        "eval", "against", "run-base", "--candidate", "run-cand",
        "--suite", str(tmp_path / "nonexistent_locked.json"),
    ]
    return pass_args, fail_args, missing_args


def _setup_eval_checklist(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    spec_file = tmp_path / "checklist_spec.yaml"
    spec_file.write_text(
        yaml.safe_dump({
            "tests": [
                {"name": "t1", "kind": "mft", "prompts": ["hi"], "expected": ["hello"]}
            ]
        }),
        encoding="utf-8",
    )

    pass_ev = tmp_path / "checklist_pass.json"
    pass_ev.write_text(json.dumps({"t1": ["hello"]}), encoding="utf-8")

    fail_ev = tmp_path / "checklist_fail.json"
    fail_ev.write_text(json.dumps({"t1": ["bad_response"]}), encoding="utf-8")

    pass_args = ["eval", "checklist", str(spec_file), "--evidence", str(pass_ev)]
    fail_args = ["eval", "checklist", str(spec_file), "--evidence", str(fail_ev)]
    missing_args = ["eval", "checklist", str(tmp_path / "nonexistent_spec.yaml")]
    return pass_args, fail_args, missing_args


def _setup_eval_behavior(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_ev = tmp_path / "behavior_pass.json"
    pass_ev.write_text(
        json.dumps({
            "pre_responses": ["safe"] * 5,
            "post_responses": ["safe"] * 5,
            "oracle": ["safe"] * 5,
        }),
        encoding="utf-8",
    )

    fail_ev = tmp_path / "behavior_fail.json"
    fail_ev.write_text(
        json.dumps({
            "pre_responses": ["safe"] * 5,
            "post_responses": ["unsafe"] * 5,
            "oracle": ["safe"] * 5,
        }),
        encoding="utf-8",
    )

    pass_args = [
        "eval", "behavior", "run1", "--battery", "xstest", "--evidence", str(pass_ev),
    ]
    fail_args = [
        "eval", "behavior", "run1", "--battery", "xstest", "--evidence", str(fail_ev),
    ]
    missing_args = [
        "eval", "behavior", "run1", "--battery", "xstest",
        "--evidence", str(tmp_path / "nonexistent_behavior.json"),
    ]
    return pass_args, fail_args, missing_args


def _setup_eval_quant_check(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    before_model = tmp_path / "before.safetensors"
    before_model.write_bytes(b"model_before")
    after_model = tmp_path / "after.safetensors"
    after_model.write_bytes(b"model_after")

    tasks_file = tmp_path / "quant_tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"prompt": "p", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )

    pass_args = [
        "eval", "quant-check",
        "--before", str(before_model),
        "--after", str(after_model),
        "--tasks", str(tasks_file),
    ]
    fail_args = [
        "eval", "quant-check",
        "--before", str(before_model),
        "--after", str(after_model),
        "--tasks", str(tasks_file),
    ]
    missing_args = [
        "eval", "quant-check",
        "--before", str(before_model),
        "--after", str(after_model),
        "--tasks", str(tmp_path / "nonexistent_tasks.jsonl"),
    ]
    return pass_args, fail_args, missing_args


def _setup_lock_check(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    base_sha = "a" * 64
    dataset_sha = "b" * 64
    env_hash = "c" * 64
    closure = compute_lock_closure(
        base_model_sha=base_sha,
        dataset_sha=dataset_sha,
        env_hash=env_hash,
    )
    lock_file = tmp_path / "soup.lock"
    lock = SoupLock(
        soup_version="0.75.0",
        base_model="test-model",
        base_model_sha=base_sha,
        dataset_sha=dataset_sha,
        env_hash=env_hash,
        closure_sha=closure,
        created_at="2026-09-13T00:00:00Z",
    )
    write_lock(lock, str(lock_file))

    pass_args = [
        "lock", "check", str(lock_file),
        "--base-model", "test-model",
        "--base-sha", base_sha,
        "--dataset-sha", dataset_sha,
        "--env-hash", env_hash,
    ]
    fail_args = [
        "lock", "check", str(lock_file),
        "--base-model", "test-model",
        "--base-sha", "d" * 64,  # causes drift
        "--dataset-sha", dataset_sha,
        "--env-hash", env_hash,
    ]
    missing_args = [
        "lock", "check", str(tmp_path / "nonexistent.lock"),
        "--base-model", "test-model",
        "--base-sha", base_sha,
        "--dataset-sha", dataset_sha,
        "--env-hash", env_hash,
    ]
    return pass_args, fail_args, missing_args


def _setup_data_validate(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_file = tmp_path / "valid.jsonl"
    pass_file.write_text(
        json.dumps({"instruction": "hi", "output": "there"}) + "\n",
        encoding="utf-8",
    )

    fail_file = tmp_path / "invalid.jsonl"
    fail_file.write_text(
        "".join(json.dumps({"instruction": None, "output": None}) + "\n" for _ in range(5)),
        encoding="utf-8",
    )

    pass_args = ["data", "validate", str(pass_file), "--format", "alpaca"]
    fail_args = ["data", "validate", str(fail_file), "--format", "alpaca"]
    missing_args = ["data", "validate", str(tmp_path / "nonexistent.jsonl"), "--format", "alpaca"]
    return pass_args, fail_args, missing_args


def _setup_data_lint(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_file = tmp_path / "dpo_pass.jsonl"
    pass_file.write_text(
        json.dumps({
            "prompt": "hello",
            "chosen": "good morning how are you",
            "rejected": "good night see you tomorrow",
        }) + "\n",
        encoding="utf-8",
    )

    fail_file = tmp_path / "dpo_fail.jsonl"
    fail_file.write_text(
        json.dumps({
            "prompt": "hello",
            "chosen": "good morning how are you",
            "rejected": "good morning how are you",  # chosen == rejected triggers MAJOR
        }) + "\n",
        encoding="utf-8",
    )

    pass_args = ["data", "lint", str(pass_file), "--format", "dpo"]
    fail_args = ["data", "lint", str(fail_file), "--format", "dpo"]
    missing_args = ["data", "lint", str(tmp_path / "nonexistent.jsonl"), "--format", "dpo"]
    return pass_args, fail_args, missing_args


def _setup_ship(tmp_path: Path) -> Tuple[list[str], list[str], list[str]]:
    pass_file = tmp_path / "ship_pass.json"
    pass_file.write_text(
        json.dumps({
            "task": {"mode": "metric", "base": 0.4, "tuned": 0.8},
            "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
        }),
        encoding="utf-8",
    )

    fail_file = tmp_path / "ship_fail.json"
    fail_file.write_text(
        json.dumps({
            "task": {"mode": "metric", "base": 0.8, "tuned": 0.2},
            "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
        }),
        encoding="utf-8",
    )

    pass_args = ["ship", "--evidence", str(pass_file)]
    fail_args = ["ship", "--evidence", str(fail_file)]
    missing_args = ["ship", "--evidence", str(tmp_path / "nonexistent.json")]
    return pass_args, fail_args, missing_args


SETUP_MAP = {
    "eval gate": _setup_eval_gate,
    "eval against": _setup_eval_against,
    "eval checklist": _setup_eval_checklist,
    "eval behavior": _setup_eval_behavior,
    "eval quant-check": _setup_eval_quant_check,
    "lock check": _setup_lock_check,
    "data validate": _setup_data_validate,
    "data lint": _setup_data_lint,
    "ship": _setup_ship,
}


@pytest.mark.parametrize(
    "command_name",
    [
        "eval gate",
        "eval against",
        "eval checklist",
        "eval behavior",
        "eval quant-check",
        "lock check",
        "data validate",
        "data lint",
        "ship",
    ],
)
def test_gate_commands_follow_unified_exit_code_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
) -> None:
    """Every gate command must exit:

      0 = EXIT_OK on passing input
      2 = EXIT_GATE_FAILED on failing input / regression / drift
      3 = EXIT_USAGE_ERROR on missing input file
    """
    monkeypatch.chdir(tmp_path)

    # Command-specific mocks for isolated execution without GPU or network
    if command_name == "eval against":
        def fake_get_metric_series(self, run_id: str, metric: str):
            if "pass" in run_id:
                return [0.8, 0.8, 0.8] if "cand" in run_id else [0.4, 0.4, 0.4]
            return [0.2, 0.2, 0.2] if "cand" in run_id else [0.8, 0.8, 0.8]

        monkeypatch.setattr(
            "soup_cli.experiment.tracker.ExperimentTracker.get_metric_series",
            fake_get_metric_series,
        )

    if command_name == "eval quant-check":
        class _MockState:
            is_fail = False

        state = _MockState()

        def fake_make_model_gen(ref: str):
            if state.is_fail:
                return (lambda p: "bad") if "after" in str(ref) else (lambda p: "hi")
            return lambda p: "hi"

        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            fake_make_model_gen,
        )

    setup_fn = SETUP_MAP[command_name]
    pass_args, fail_args, missing_args = setup_fn(tmp_path)
    runner = CliRunner()

    # (a) Passing input -> 0 (EXIT_OK)
    res_pass = runner.invoke(app, pass_args)
    assert res_pass.exit_code == EXIT_OK, (
        f"{command_name} pass: expected {EXIT_OK}, got {res_pass.exit_code}\n"
        f"output: {res_pass.output}\nexc: {repr(res_pass.exception)}"
    )

    # (b) Failing input -> 2 (EXIT_GATE_FAILED)
    if command_name == "eval quant-check":
        state.is_fail = True

    res_fail = runner.invoke(app, fail_args)
    assert res_fail.exit_code == EXIT_GATE_FAILED, (
        f"{command_name} fail: expected {EXIT_GATE_FAILED}, got {res_fail.exit_code}\n"
        f"output: {res_fail.output}\nexc: {repr(res_fail.exception)}"
    )

    # (c) Missing input file -> 3 (EXIT_USAGE_ERROR)
    res_missing = runner.invoke(app, missing_args)
    assert res_missing.exit_code == EXIT_USAGE_ERROR, (
        f"{command_name} missing file: expected {EXIT_USAGE_ERROR}, got {res_missing.exit_code}\n"
        f"output: {res_missing.output}\nexc: {repr(res_missing.exception)}"
    )
