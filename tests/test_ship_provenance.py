"""Opt-in drift detection binds evidence to inputs and effective verdict policy."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

ship = importlib.import_module("soup_cli.commands.ship")
runner = CliRunner()


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"local fixture weights")
    (tmp_path / "train.jsonl").write_text('{"text":"train"}\n', encoding="utf-8")
    (tmp_path / "task.jsonl").write_text(
        '{"prompt":"task","expected":"answer"}\n', encoding="utf-8",
    )
    (tmp_path / "soup.yaml").write_text(
        "base: owner/base\ndata: {train: train.jsonl}\n"
        "training: {quantization: none}\neval:\n  ship:\n"
        "    task_eval: task.jsonl\n    general_suite: mini_mmlu\n"
        "    forgetting_threshold: 0.05\n",
        encoding="utf-8",
    )
    (tmp_path / "raw.json").write_text(json.dumps({
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.7},
        "benchmarks": {"mini_mmlu": {"base": 0.6, "tuned": 0.6}},
        "numerics": "float32",
    }), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "selection", ["mini_tool_call,mini_mmlu", None, "mini_mmlu,mini_tool_call"],
)
def test_restamped_suite_order_and_default_roundtrip(work, selection):
    from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE

    cfg = work / "soup.yaml"
    text = cfg.read_text(encoding="utf-8")
    suites = list(DEFAULT_GENERAL_SUITE) if selection is None else selection.split(",")
    text = text.replace(
        "    general_suite: mini_mmlu\n",
        "" if selection is None else f"    general_suite: {selection}\n",
    )
    cfg.write_text(text, encoding="utf-8")
    raw = work / "raw.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["benchmarks"] = {name: {"base": 0.6, "tuned": 0.6} for name in suites}
    raw.write_text(json.dumps(payload), encoding="utf-8")
    bound = _produce()
    binding = bound["provenance"]["binding"]
    assert binding["suites"]["names"] == sorted(suites)
    assert binding["policy"]["general_suite"] == sorted(suites)
    result = _gate()
    assert result.exit_code == 0, (result.output, result.exception)
    assert "SHIP" in strip_ansi(result.output)
    # A different selection must still be refused after canonicalization.
    changed = _gate("--general-suite", "mini_arithmetic")
    assert changed.exit_code == 3
    assert "provenance" in strip_ansi(changed.output).lower()


def test_binding_canonicalizes_both_suite_fields_without_mutating_inputs():
    from soup_cli.utils.ship_provenance import make_binding

    names = ["mini_tool_call", "mini_mmlu"]
    policy = {"general_suite": names.copy(), "forgetting_threshold": 0.05}
    binding = make_binding(
        adapter=None, task_eval_sha256=None, source_data_sha256=None,
        data_sha256=None, suites=names, policy=policy,
    )
    assert binding["suites"]["names"] == sorted(names)
    assert binding["policy"]["general_suite"] == sorted(names)
    assert names == policy["general_suite"] == ["mini_tool_call", "mini_mmlu"]
    assert binding["policy_sha256"] == hashlib.sha256(json.dumps(
        binding["policy"], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _produce(*extra):
    result = runner.invoke(app, [
        "ship", "--evidence", "raw.json", "--config", "soup.yaml", "--adapter", "adapter",
        "--emit-evidence", "bound.json", *extra,
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    return json.loads(Path("bound.json").read_text(encoding="utf-8"))


def _gate(*extra):
    return runner.invoke(app, [
        "ship", "--evidence", "bound.json", "--config", "soup.yaml",
        "--adapter", "adapter", "--require-provenance", *extra,
    ])


@pytest.mark.parametrize("runs", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("restamp", [False, True], ids=["read", "restamp"])
@pytest.mark.parametrize("color", [False, True], ids=["plain", "ansi"])
def test_strict_nonfinite_floor_runs_keep_usage_diagnostic(work, runs, restamp, color):
    payload = _produce()
    payload["noise_floor"] = {"runs": runs, "floors": {"mini_mmlu": 0.0}}
    (work / "bound.json").write_text(json.dumps(payload), encoding="utf-8")
    arguments = [
        "ship", "--evidence", "bound.json", "--config", "soup.yaml",
        "--adapter", "adapter", "--require-provenance", "--output", "verdict.json",
    ]
    if restamp:
        arguments.extend(["--emit-evidence", "restamped.json"])
    result = runner.invoke(app, arguments, color=color)
    assert result.exit_code == 3, (result.output, result.exception)
    diagnostic = " ".join(strip_ansi(result.output).split())
    assert "evidence.noise_floor.runs must be an integer" in diagnostic
    assert not (work / "verdict.json").exists()
    assert not (work / "restamped.json").exists()


@pytest.mark.parametrize("runs", [None, 2, 10], ids=["no-floor", "min-floor", "max-floor"])
@pytest.mark.parametrize("restamp", [False, True], ids=["read", "restamp"])
def test_strict_valid_floor_and_no_floor_controls(work, runs, restamp):
    if runs is not None:
        raw_path = work / "raw.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["noise_floor"] = {"runs": runs, "floors": {"mini_mmlu": 0.0}}
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
    payload = _produce()
    extra = ["--emit-evidence", "restamped.json"] if restamp else []
    result = _gate("--output", "verdict.json", *extra)
    assert result.exit_code == 0, (result.output, result.exception)
    assert (work / "verdict.json").is_file()
    if restamp:
        restored = json.loads((work / "restamped.json").read_text(encoding="utf-8"))
        assert restored.get("noise_floor") == payload.get("noise_floor")
        assert restored["provenance"]["binding"] == payload["provenance"]["binding"]


def test_emit_binds_actual_bytes_and_effective_policy(work):
    from soup_cli.utils.adapter_sign import compute_adapter_manifest

    payload = _produce("--forgetting-threshold", "0.2")
    binding = payload["provenance"]["binding"]
    assert binding["schema_version"] == 1
    assert binding["adapter_merkle_root"] == compute_adapter_manifest("adapter").merkle_root
    expected_task = hashlib.sha256((work / "task.jsonl").read_bytes()).hexdigest()
    assert binding["task_eval_sha256"] == expected_task
    assert binding["source_data_at_verdict_sha256"] == hashlib.sha256(
        (work / "train.jsonl").read_bytes(),
    ).hexdigest()
    assert binding["policy"]["forgetting_threshold"] == 0.2
    assert binding["suites"]["names"] == ["mini_mmlu"]
    assert isinstance(binding["suites"]["soup_version"], str)
    assert len(binding["policy_sha256"]) == 64
    canonical = json.dumps(binding["policy"], sort_keys=True, separators=(",", ":"))
    assert binding["policy_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert _gate("--forgetting-threshold", "0.2").exit_code == 0


@pytest.mark.parametrize("changed", ["adapter", "task", "training", "threshold", "suite", "judge"])
def test_strict_gate_refuses_input_and_policy_drift(work, changed):
    _produce()
    extra = []
    if changed == "adapter":
        (work / "adapter" / "adapter_model.safetensors").write_bytes(b"other weights")
    elif changed == "task":
        (work / "task.jsonl").write_text('{"prompt":"other"}\n', encoding="utf-8")
    elif changed == "training":
        (work / "train.jsonl").write_text('{"text":"reimported"}\n', encoding="utf-8")
    else:
        extra = {
            "threshold": ["--forgetting-threshold", "0.2"],
            "suite": ["--general-suite", "mini_arithmetic"],
            "judge": ["--task-mode", "judge_score", "--judge-model", "http://localhost:8000"],
        }[changed]
    result = _gate(*extra)
    assert result.exit_code == 3, (result.output, result.exception)
    assert "provenance" in strip_ansi(result.output).lower()


def test_default_gate_preserves_recipe_only_semantics(work):
    _produce()
    (work / "train.jsonl").write_text("changed source", encoding="utf-8")
    result = runner.invoke(app, [
        "ship", "--evidence", "bound.json", "--config", "soup.yaml",
        "--forgetting-threshold", "0.2",
    ])
    assert result.exit_code == 0, (result.output, result.exception)


def test_strict_restamp_does_not_bypass_verification(work):
    _produce()
    (work / "adapter" / "adapter_model.safetensors").write_bytes(b"other weights")
    result = _gate("--emit-evidence", "laundered.json")
    assert result.exit_code == 3
    assert not (work / "laundered.json").exists()


def test_strict_restamp_keeps_config_staleness_gate(work):
    _produce()
    cfg = work / "soup.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "output: different-output\n", encoding="utf-8")
    result = _gate("--emit-evidence", "laundered.json")
    assert result.exit_code == 3
    assert "config_sha" in strip_ansi(result.output)
    assert not (work / "laundered.json").exists()


@pytest.mark.parametrize(
    "field", ["binding", "adapter_merkle_root", "task_eval_sha256", "policy_sha256"],
)
def test_missing_or_malformed_bindings_fail_closed(work, field):
    payload = _produce()
    if field == "binding":
        del payload["provenance"][field]
    else:
        payload["provenance"]["binding"][field] = "\x1b[31mPRIVATE"
    (work / "bound.json").write_text(json.dumps(payload), encoding="utf-8")
    result = _gate()
    assert result.exit_code == 3
    assert "PRIVATE" not in result.output


def test_caller_training_identity_is_explicit_and_checked(work):
    digest = "a" * 64
    payload = _produce("--data-sha256", digest)
    assert payload["provenance"]["binding"]["training_data"] == {
        "sha256": digest, "source": "caller",
    }
    assert _gate("--data-sha256", digest).exit_code == 0
    assert _gate("--data-sha256", "b" * 64).exit_code == 3
    assert _gate().exit_code == 3


def test_unknown_composite_source_is_not_a_partial_identity(work):
    cfg = work / "soup.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace(
        "data: {train: train.jsonl}",
        "data: {train: [train.jsonl, missing.jsonl], interleave: concat}",
    ), encoding="utf-8")
    payload = _produce()
    assert payload["provenance"]["binding"]["source_data_at_verdict_sha256"] is None
    assert _gate().exit_code == 3
    digest = "a" * 64
    _produce("--data-sha256", digest)
    result = _gate("--data-sha256", digest)
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.mark.parametrize("value", ["partial", "A" * 64, "a" * 64 + "\n"])
def test_invalid_caller_digest_refused_before_outputs(work, value):
    result = runner.invoke(app, [
        "ship", "--evidence", "raw.json", "--emit-evidence", "bad.json", "--data-sha256", value,
    ])
    assert result.exit_code == 3
    assert not (work / "bad.json").exists()


def test_strict_requires_comparison_inputs(work):
    _produce()
    for args in (
        ["ship", "--evidence", "bound.json", "--require-provenance"],
        ["ship", "--evidence", "bound.json", "--config", "soup.yaml", "--require-provenance"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 3


def test_strict_rejects_scores_for_a_different_suite_or_task_mode(work):
    payload = _produce()
    payload["benchmarks"] = {"mini_arithmetic": {"base": 0.6, "tuned": 0.6}}
    (work / "bound.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _gate().exit_code == 3


def test_judge_endpoint_is_hashed_without_credentials(work):
    raw = json.loads((work / "raw.json").read_text(encoding="utf-8"))
    raw["task"]["mode"] = "judge_score"
    (work / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    payload = _produce(
        "--task-mode", "judge_score", "--judge-model", "http://user:secret@localhost:8000/private",
    )
    text = json.dumps(payload["provenance"]["binding"])
    assert "secret" not in text and "/private" not in text
    assert payload["provenance"]["binding"]["policy"]["judge_model_sha256"] == hashlib.sha256(
        b"http://user:secret@localhost:8000/private",
    ).hexdigest()


@pytest.mark.parametrize("floor_runs", [None, 3])
def test_real_scorers_emit_replayable_binding(work, monkeypatch, floor_runs):
    """Only generation is a stand-in; CLI, task/suite scorers and verdict are real."""
    calls = []

    def generators(*args, **kwargs):
        def base(prompt):
            calls.append(("base", prompt))
            return "wrong" if prompt == "task" else "B"

        def tuned(prompt):
            calls.append(("tuned", prompt))
            return "answer" if prompt == "task" else "B"

        return base, tuned

    monkeypatch.setattr(ship, "_resolve_generators", generators)
    extra = ["--noise-floor", str(floor_runs)] if floor_runs else []
    result = runner.invoke(app, [
        "ship", "--base", "owner/base", "--adapter", "adapter", "--config", "soup.yaml",
        "--device", "cpu", "--emit-evidence", "bound.json", *extra,
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    payload = json.loads((work / "bound.json").read_text(encoding="utf-8"))
    assert payload["task"] == {"mode": "metric", "base": 0.0, "tuned": 1.0}
    assert payload["benchmarks"]["mini_mmlu"]["base"] == payload["benchmarks"]["mini_mmlu"]["tuned"]
    assert payload["provenance"]["binding"]["policy"]["noise_floor_runs"] == floor_runs
    assert {side for side, _ in calls} == {"base", "tuned"}
    replay = _gate()
    assert replay.exit_code == 0, (replay.output, replay.exception)


@pytest.mark.parametrize("changed", ["task.jsonl", "adapter/adapter_model.safetensors"])
def test_live_input_changes_refuse_emission(work, monkeypatch, changed):
    def generators(*args, **kwargs):
        (work / changed).write_bytes(b'{"prompt":"changed","expected":"answer"}\n')
        return lambda _: "wrong", lambda _: "answer"

    monkeypatch.setattr(ship, "_resolve_generators", generators)
    result = runner.invoke(app, [
        "ship", "--base", "owner/base", "--adapter", "adapter", "--config", "soup.yaml",
        "--device", "cpu", "--emit-evidence", "bound.json",
    ])
    assert result.exit_code == 3, (result.output, result.exception)
    assert "changed during live evaluation" in strip_ansi(result.output)
    assert not (work / "bound.json").exists()


@pytest.mark.parametrize("mode", ["judge_score", "pairwise"])
def test_external_judge_modes_bind_declared_endpoint(work, mode):
    raw = json.loads((work / "raw.json").read_text(encoding="utf-8"))
    raw["task"]["mode"] = mode
    (work / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    options = ["--task-mode", mode, "--judge-model", "http://localhost:8000"]
    _produce(*options)
    result = _gate(*options)
    assert result.exit_code == 0, (result.output, result.exception)
    assert _gate(*options[:-1], "http://localhost:9000").exit_code == 3


@pytest.mark.parametrize("missing", ["adapter/adapter_model.safetensors", "task.jsonl"])
def test_empty_adapter_and_unknown_task_are_not_verifiable(work, missing):
    _produce()
    (work / missing).unlink()
    assert _gate().exit_code == 3


@pytest.mark.parametrize("tamper", ["schema", "version", "policy", "mode"])
def test_tampered_descriptor_and_score_mode_are_refused(work, tamper):
    payload = _produce()
    binding = payload["provenance"]["binding"]
    if tamper == "schema":
        binding["schema_version"] = True
    elif tamper == "version":
        binding["suites"]["soup_version"] = "older-version"
    elif tamper == "policy":
        binding["policy"]["forgetting_threshold"] = 0.3
    else:
        payload["task"]["mode"] = "pairwise"
    (work / "bound.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _gate().exit_code == 3
