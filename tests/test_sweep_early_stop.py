"""#3006: ``soup sweep --early-stop`` takes a factor of 1.0 or more, and not ``nan``.

The check is ``loss > best_loss * factor``, and ``best_loss`` already holds the
arm that just finished. A factor below 1 therefore fired on the best arm itself,
so the sweep stopped after its second arm even when every arm improved. A
negative factor did the same. ``0`` was falsy and turned the check off, and
``nan`` was truthy while every comparison with it was false, so it also turned
the check off without a word. The option now carries ``min=1.0`` and the shared
``refuse_nan`` callback, and both refuse while the options are parsed, before
the config is read and before any arm runs, with or without ``--dry-run``.
"""

from __future__ import annotations

import pytest

import soup_cli.commands.sweep as sweep_mod
from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

pytestmark = pytest.mark.unit

_IMPROVING = [1.0, 0.8, 0.7, 0.6]
_ARM_TWO_TEN_TIMES_WORSE = [0.5, 5.0, 0.4, 0.3]
_NAN_SPELLINGS = ["nan", "NaN", "NAN", "+nan", "-nan"]


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """Run ``soup sweep`` over four arms, each arm's training replaced by a fixed loss.

    Returns ``(result, arms that ran)``.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    # Keep Click's error on one line of Rich's error panel.
    monkeypatch.setenv("COLUMNS", "400")
    (tmp_path / "soup.yaml").write_text(
        "base: some-model\ndata:\n  train: ./d.jsonl\n", encoding="utf-8"
    )

    def run(losses, *extra):
        ran = []

        def fake_run(cfg, combo, name, path):
            ran.append(name)
            return {"final_loss": losses[len(ran) - 1], "run_id": name, "duration": "1s"}

        monkeypatch.setattr(sweep_mod, "_run_single", fake_run)
        argv = ["sweep", "--config", "soup.yaml", "--yes",
                "--param", "lr=1e-5,2e-5,3e-5,4e-5", *extra]
        return make_cli_runner().invoke(app, argv), ran

    return run


def _plain(result) -> str:
    """The output without colour, panel borders or line breaks."""
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


def _refusal(shown: str) -> str:
    return f"Invalid value for '--early-stop': {shown} is not in the range x>=1.0."


# --- refused ------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
@pytest.mark.parametrize(
    "value", ["0.9", "0.5", "0", "-1", "-inf", "-Infinity", "-1e999"]
)
def test_a_factor_below_one_is_refused_before_any_arm_runs(sweep, value, dry_run):
    result, ran = sweep(_IMPROVING, "--early-stop", value, *(["--dry-run"] if dry_run else []))

    assert result.exit_code == 2, result.output
    assert _refusal(str(float(value))) in _plain(result), result.output
    assert ran == []
    assert "Sweep Plan" not in _plain(result)


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
@pytest.mark.parametrize("spelling", _NAN_SPELLINGS)
def test_nan_is_refused_in_every_spelling_before_any_arm_runs(sweep, spelling, dry_run):
    result, ran = sweep(
        _IMPROVING, "--early-stop", spelling, *(["--dry-run"] if dry_run else [])
    )

    assert result.exit_code == 2, result.output
    assert _refusal("nan") in _plain(result), result.output
    assert ran == []
    assert "Sweep Plan" not in _plain(result)


# --- controls that the refusal must not touch -----------------------------------


@pytest.mark.parametrize(
    "extra",
    [[], ["--early-stop", "1.0"], ["--early-stop", "1.5"], ["--early-stop", "inf"],
     ["--early-stop", "Infinity"], ["--early-stop", "1e999"]],
    ids=["unset", "1.0", "1.5", "inf", "Infinity", "1e999"],
)
def test_improving_arms_all_run_at_an_accepted_factor(sweep, extra):
    result, ran = sweep(_IMPROVING, *extra)

    assert result.exit_code == 0, result.output
    assert len(ran) == 4
    assert "exceeds threshold" not in _plain(result)
    assert "Early stopping" not in _plain(result)


def test_a_ten_times_worse_arm_still_stops_the_sweep_at_1_5(sweep):
    result, ran = sweep(_ARM_TWO_TEN_TIMES_WORSE, "--early-stop", "1.5")

    assert result.exit_code == 0, result.output
    assert len(ran) == 2
    text = _plain(result)
    assert "Loss 5.0000 exceeds threshold (0.5000 x 1.5 = 0.7500)" in text
    assert "Early stopping: skipping 2 remaining run(s). Last loss 5.0000 exceeded" in text


@pytest.mark.parametrize("factor", ["1.0", "1.5", "inf"])
def test_a_diverged_arm_still_stops_the_sweep_at_any_accepted_factor(sweep, factor):
    result, ran = sweep([0.5, float("nan"), 0.4, 0.3], "--early-stop", factor)

    assert result.exit_code == 0, result.output
    assert len(ran) == 2
    assert "Early stopping: skipping 2 remaining run(s). Last run diverged." in _plain(result)


def test_an_accepted_factor_still_shows_the_dry_run_plan(sweep):
    result, ran = sweep(_IMPROVING, "--early-stop", "1.5", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "Sweep Plan" in _plain(result)
    assert ran == []
