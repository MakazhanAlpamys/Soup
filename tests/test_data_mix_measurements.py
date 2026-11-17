"""Mix recipes require usable observations, and live configs fail before training."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.data_mix import MixCandidate, MixOptimizationReport
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.data_mix.console", Console(width=200))
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text('{"text": "sample"}\n', encoding="utf-8")
    (tmp_path / "soup.yaml").write_text("base: example/model\n", encoding="utf-8")


def _mix(*extra: str):
    return runner.invoke(
        app,
        [
            "data", "mix", "--optimize", "--datasets", "a.jsonl,b.jsonl",
            "--num-probes", "3", "--output", "recipe.yaml", *extra,
        ],
    )


def _mock_proxy(monkeypatch: pytest.MonkeyPatch, live: bool, proxy: Mock) -> list[str]:
    if live:
        monkeypatch.setattr(
            "soup_cli.utils.mix_proxy.validate_proxy_config", Mock(), raising=False
        )
        monkeypatch.setattr("soup_cli.utils.mix_proxy.proxy_run_for_weights", proxy)
        return ["--live", "--base-yaml", "soup.yaml"]
    monkeypatch.setattr("soup_cli.commands.data_mix._offline_proxy", proxy)
    return []


@pytest.mark.parametrize("live", [False, True], ids=["offline", "live"])
@pytest.mark.parametrize("failure", [RuntimeError("training failed"), float("nan")])
def test_all_failed_candidates_do_not_write_recipe(
    monkeypatch: pytest.MonkeyPatch, live: bool, failure: object
) -> None:
    proxy = Mock(side_effect=[failure] * 3)
    extra = _mock_proxy(monkeypatch, live, proxy)

    result = _mix(*extra)

    assert result.exit_code == 2, (result.output, result.exception)
    assert proxy.call_count == 3
    assert not Path("recipe.yaml").exists()
    assert "No valid proxy measurements" in strip_ansi(result.output)
    assert "No recipe written" in strip_ansi(result.output)
    assert "wrote recipe:" not in result.output


@pytest.mark.parametrize("live", [False, True], ids=["offline", "live"])
def test_one_valid_candidate_survives_other_failures(
    monkeypatch: pytest.MonkeyPatch, live: bool
) -> None:
    proxy = Mock(side_effect=[RuntimeError("OOM"), 0.75, float("nan")])
    extra = _mock_proxy(monkeypatch, live, proxy)

    result = _mix(*extra)

    assert result.exit_code == 0, (result.output, result.exception)
    assert proxy.call_count == 3
    recipe = yaml.safe_load(Path("recipe.yaml").read_text(encoding="utf-8"))
    assert len(recipe["data"]["train"]) == 2
    assert sum(recipe["data"]["interleave"]["probs"]) == pytest.approx(1.0, abs=1e-6)
    assert "best_loss=0.750000" in strip_ansi(result.output)


@pytest.mark.parametrize("exc_type", [ValueError, TypeError, OSError])
def test_live_preflight_rejects_before_optimizer_or_subprocess(
    monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
) -> None:
    validation = Mock(side_effect=exc_type("bad [bold]config[/] :warning:\x9b\u202e"))
    optimizer = Mock()
    subprocess = Mock()
    monkeypatch.setattr(
        "soup_cli.utils.mix_proxy.validate_proxy_config", validation, raising=False
    )
    monkeypatch.setattr("soup_cli.utils.data_mix.run_mix_optimizer", optimizer)
    monkeypatch.setattr("soup_cli.utils.mix_proxy.subprocess.run", subprocess)

    result = _mix("--live", "--base-yaml", "soup.yaml")

    assert result.exit_code == 2, (result.output, result.exception)
    validation.assert_called_once_with("soup.yaml")
    optimizer.assert_not_called()
    subprocess.assert_not_called()
    assert not Path("recipe.yaml").exists()
    output = strip_ansi(result.output)
    assert "live proxy validation failed: bad [bold]config[/] :warning:" in output
    assert "\x9b" not in output and "\u202e" not in output


def test_live_preflight_runs_before_valid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "soup_cli.utils.mix_proxy.validate_proxy_config",
        Mock(side_effect=lambda _: events.append("validate")),
        raising=False,
    )

    def proxy(*args, **kwargs) -> float:
        events.append("proxy")
        return 0.75

    monkeypatch.setattr("soup_cli.utils.mix_proxy.proxy_run_for_weights", proxy)

    result = _mix("--live", "--base-yaml", "soup.yaml")

    assert result.exit_code == 0, (result.output, result.exception)
    assert events == ["validate", "proxy", "proxy", "proxy"]


@pytest.mark.parametrize("invalid", ["empty_candidates", "nonfinite_best"])
def test_recipe_requires_candidates_and_finite_best(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    report = MixOptimizationReport(
        datasets=("a.jsonl", "b.jsonl"),
        candidates=(MixCandidate((0.5, 0.5), 0.75, 0.1),),
        best_weights=(0.5, 0.5),
        best_eval_loss=0.75,
        partial=False,
        elapsed_seconds=0.1,
    )
    if invalid == "empty_candidates":
        report = replace(report, candidates=())
    else:
        report = replace(report, best_eval_loss=float("inf"))
    monkeypatch.setattr("soup_cli.utils.data_mix.run_mix_optimizer", Mock(return_value=report))

    result = _mix()

    assert result.exit_code == 2, (result.output, result.exception)
    assert not Path("recipe.yaml").exists()


def test_invalid_offline_proxy_value_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("soup_cli.commands.data_mix._offline_proxy", Mock(return_value="invalid"))

    result = _mix()

    assert result.exit_code == 2, (result.output, result.exception)
    assert "optimization failed: proxy_run must return float" in strip_ansi(result.output)
    assert not Path("recipe.yaml").exists()


def test_offline_panel_identifies_synthetic_results() -> None:
    result = _mix()

    assert result.exit_code == 0, (result.output, result.exception)
    output = strip_ansi(result.output)
    assert "offline synthetic" in output
    assert "no training or evaluation" in output
