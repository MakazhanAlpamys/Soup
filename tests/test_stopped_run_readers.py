"""Stopped training is visible to ship/eval callers without changing their results."""

import importlib
import json
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import Result

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base-model").mkdir()
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt":"task","expected":"answer"}\n', encoding="utf-8"
    )
    return tmp_path


def _write_run(artifact: Path, *, stopped: bool) -> None:
    artifact.mkdir()
    metadata = {
        "schema_version": 1,
        "run_id": "run_fixture",
        "base_model": "base-model",
        "base_revision": None,
        "base_revision_source": "local",
        "training_data_sha256": None,
    }
    if stopped:
        metadata["status"] = "stopped"
    (artifact / "soup_run.json").write_text(json.dumps(metadata), encoding="utf-8")


def _assert_stopped_warning(result: Result, *, stopped: bool, progress: bool = False) -> None:
    stdout = " ".join(strip_ansi(result.stdout).lower().split())
    stderr = " ".join(strip_ansi(result.stderr).lower().split())
    diagnostic, other = (stdout, stderr) if progress else (stderr, stdout)
    assert not all(word in other for word in ("stopped", "training run", "user"))
    if stopped:
        assert "stopped" in diagnostic, result.output
        assert "training run" in diagnostic and "user" in diagnostic, diagnostic
    else:
        assert "stopped" not in diagnostic


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "legacy"])
def test_ship_live_warns_and_emits_training_status(
    work: Path, monkeypatch: pytest.MonkeyPatch, stopped: bool
) -> None:
    adapter = work / "candidate"
    _write_run(adapter, stopped=stopped)
    # A real, small safetensors file with one nonzero standalone tensor.
    header = b'{"head.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    header += b" " * (-len(header) % 8)
    (adapter / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    )
    (work / "train.jsonl").write_text('{"text":"train"}\n', encoding="utf-8")
    (work / "soup.yaml").write_text(
        "base: base-model\ndata: {train: train.jsonl}\n"
        "training: {quantization: none}\neval:\n  ship:\n"
        "    task_eval: tasks.jsonl\n    general_suite: mini_mmlu\n",
        encoding="utf-8",
    )
    ship = importlib.import_module("soup_cli.commands.ship")

    def generators(*args: object, **kwargs: object) -> tuple[Callable[[str], str], ...]:
        return (
            lambda prompt: "wrong" if prompt == "task" else "B",
            lambda prompt: "answer" if prompt == "task" else "B",
        )
    monkeypatch.setattr(ship, "_resolve_generators", generators)
    ship_args = ["ship", "--config", "soup.yaml", "--adapter", "candidate"]
    live = runner.invoke(
        app,
        [
            *ship_args, "--base", "base-model", "--device", "cpu",
            "--emit-evidence", "evidence.json", "--output", "live.json",
        ],
    )
    assert live.exit_code == 0, (live.output, live.exception)
    assert json.loads((work / "live.json").read_text(encoding="utf-8"))["decision"] == "SHIP"
    _assert_stopped_warning(live, stopped=stopped)
    provenance = json.loads((work / "evidence.json").read_text(encoding="utf-8"))["provenance"]
    if stopped:
        assert provenance["training_status"] == "stopped"
    else:
        assert "training_status" not in provenance

    replay = runner.invoke(
        app,
        [
            *ship_args, "--evidence", "evidence.json", "--require-provenance",
            "--output", "replay.json",
        ],
    )
    assert replay.exit_code == 0, (replay.output, replay.exception)
    assert json.loads((work / "replay.json").read_text(encoding="utf-8"))["decision"] == "SHIP"
    _assert_stopped_warning(replay, stopped=stopped)


@pytest.mark.parametrize("progress", [False, True], ids=["ordinary", "jsonl"])
def test_ship_replays_stopped_evidence_without_changing_verdict(
    work: Path, progress: bool
) -> None:
    payload = {
        "schema_version": 2,
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.5},
        "benchmarks": {"mini_mmlu": {"base": 0.7, "tuned": 0.7}},
    }
    evidence = work / "evidence.json"
    output = work / "verdict.json"
    args = ["ship", "--evidence", evidence.name, "--output", output.name]
    if progress:
        args += ["--progress", "jsonl"]
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    legacy = runner.invoke(app, args)
    assert legacy.exit_code == 2, (legacy.output, legacy.exception)
    legacy_verdict = json.loads(output.read_text(encoding="utf-8"))
    assert legacy_verdict["decision"] == "DON'T SHIP"
    _assert_stopped_warning(legacy, stopped=False, progress=progress)
    output.unlink()

    payload["provenance"] = {"training_status": "stopped"}
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    replay = runner.invoke(app, args)
    assert replay.exit_code == legacy.exit_code, (replay.output, replay.exception)
    assert json.loads(output.read_text(encoding="utf-8")) == legacy_verdict
    if progress:
        events = [json.loads(line) for line in replay.stderr.splitlines()]
        assert [event["event"] for event in events] == ["start", "end"]
    _assert_stopped_warning(replay, stopped=True, progress=progress)


@pytest.mark.parametrize(
    "drive_relative",
    [
        pytest.param(False, id="relative"),
        pytest.param(
            True,
            id="drive-relative",
            marks=pytest.mark.skipif(sys.platform != "win32", reason="Windows drive-relative path"),
        ),
    ],
)
def test_eval_custom_warns_but_scores_stopped_local_model(
    work: Path, monkeypatch: pytest.MonkeyPatch, drive_relative: bool
) -> None:
    _write_run(work / "candidate", stopped=True)
    model = f"{work.drive}candidate" if drive_relative else "candidate"

    def generator(model_path: str, trust_remote_code: bool = False) -> Callable[[str], str]:
        return lambda prompt: "answer" if prompt == "task" else "wrong"

    monkeypatch.setattr("soup_cli.eval.custom._create_default_generator", generator)
    result = runner.invoke(
        app,
        [
            "eval", "custom", "--model", model, "--tasks", "tasks.jsonl",
            "--output", "eval.json",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    scores = json.loads((work / "eval.json").read_text(encoding="utf-8"))
    assert (scores["total"], scores["correct"], scores["accuracy"]) == (1, 1, 1.0)
    _assert_stopped_warning(result, stopped=True)


@pytest.mark.parametrize("mode", ["retention", "behavior", "capability"])
def test_live_eval_modes_warn_about_stopped_inputs(
    work: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _write_run(work / "candidate", stopped=True)
    if mode == "capability":
        monkeypatch.setattr(
            "soup_cli.utils.capability_suite.run_capability_suite",
            lambda **_: {"results": [{"benchmark": "mini", "metric": "acc", "score": 1.0}]},
        )
        args = ["run-fixture", "--live", "--model", "candidate"]
    else:
        monkeypatch.setattr(
            "soup_cli.utils.live_eval.make_generator", lambda *_a, **_kw: lambda _: "answer",
        )
        if mode == "behavior":
            monkeypatch.setattr(
                "soup_cli.utils.live_eval.load_model_and_tokenizer",
                lambda *_a, **_kw: (object(), object()),
            )
            args = ["run-fixture", "--base-model", "base-model", "--adapter", "candidate"]
        else:
            args = [
                "--model", "base-model", "--adapter", "candidate", "--label", "test",
                "--device", "cpu", "--add-task", "control=tasks.jsonl",
            ]
    result = runner.invoke(app, ["eval", mode, *args, "--output", "report.json"])
    assert result.exit_code == 0, (result.output, result.exception)
    _assert_stopped_warning(result, stopped=True)
