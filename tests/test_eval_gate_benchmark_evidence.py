"""Gate benchmark failures are distinct from completed zero scores."""

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval.forgetting import MINI_BENCHMARKS, ForgettingDetector
from tests.conftest import strip_ansi


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        MINI_BENCHMARKS, "mini_mmlu",
        [{"question": "First?", "answer": "Paris"}, {"question": "Second?", "answer": "Tokyo"}],
    )
    path = tmp_path / "suite.yaml"
    path.write_text(
        "suite: benchmark-evidence\ntasks:\n"
        "  - {name: bench, type: benchmark, benchmark: mini_mmlu, threshold: 0}\n",
        encoding="utf-8",
    )
    return path


def failed_generation(_prompt: str) -> str:
    raise RuntimeError("synthetic generation failure")


@pytest.mark.parametrize(
    ("write", "existing"), [(False, False), (True, False), (True, True)],
    ids=["gate", "new-baseline", "existing-baseline"],
)
def test_all_failed_generations_have_no_score(
    suite: Path, monkeypatch: pytest.MonkeyPatch, write: bool, existing: bool,
) -> None:
    baseline = suite.with_name("baseline.json")
    original = b'{"bench":0.75}\n'
    if existing:
        baseline.write_bytes(original)
    monkeypatch.setattr(eval_command, "console", Console(width=140))
    monkeypatch.setattr(
        "soup_cli.eval.quant_check.make_model_generator", lambda _model: failed_generation,
    )
    args = ["eval", "gate", "--suite", str(suite), "--model", "synthetic"]
    if write:
        args.extend(["--write-baseline", str(baseline)])

    result = CliRunner().invoke(app, args)
    output = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == (1 if write else 2), (output, result.exception)
    error = "Task 'bench' has no successful generations for benchmark 'mini_mmlu' (2 failed)"
    assert error in output
    assert "Wrote stamped baseline" not in output
    if existing:
        assert baseline.read_bytes() == original
    else:
        assert not baseline.exists()


def test_baseline_omits_unscored_benchmark_when_another_task_succeeds(
    suite: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        MINI_BENCHMARKS, "mini_arithmetic", [{"question": "Control?", "answer": "Paris"}],
    )
    with suite.open("a", encoding="utf-8") as file:
        file.write(
            "  - {name: control, type: benchmark, benchmark: mini_arithmetic, threshold: 1}\n"
        )

    def generate(prompt: str) -> str:
        return "Paris" if prompt == "Control?" else failed_generation(prompt)

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", lambda _model: generate)
    baseline = suite.with_name("baseline.json")
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic",
        "--write-baseline", str(baseline),
    ])

    assert result.exit_code == 2, (result.output, result.exception)
    assert json.loads(baseline.read_text(encoding="utf-8"))["scores"] == {"control": 1.0}


@pytest.mark.parametrize(
    ("responses", "score"), [(("", ""), 0.0), (("wrong", "wrong"), 0.0), (("Paris", None), 0.5)],
)
def test_measured_benchmarks_keep_their_score(
    suite: Path, monkeypatch: pytest.MonkeyPatch, responses: tuple[str | None, str | None],
    score: float,
) -> None:
    baseline = suite.with_name("baseline.json")
    values = iter(responses)

    def generate(prompt: str) -> str:
        value = next(values)
        return failed_generation(prompt) if value is None else value

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", lambda _model: generate)
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic",
        "--write-baseline", str(baseline),
    ])

    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(baseline.read_text())["scores"] == {"bench": score}


def test_training_detector_retains_its_default_policy(suite: Path) -> None:
    detector = ForgettingDetector(generate_fn=failed_generation, benchmark="mini_mmlu")
    assert detector.run_baseline() == 0.0
