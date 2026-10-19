"""#1808 — a CMA-ES merge in which every evaluation failed is a failure, not a converged merge.

``_eval_safely`` turned every failed evaluation into the ``-1e9`` sentinel; the
plateau rule then saw a flat history, set ``converged=True``, and
``soup adapters merge --strategy cmaes`` wrote a merge with the first random
sample's weights, printed ``Converged: True`` in green and exited 0. The
missing-``--eval`` message also asked for "a YAML/JSONL eval" while the loader
accepts ``.jsonl`` only, which is one way to make every evaluation fail.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.cmaes_merge as cm
from soup_cli.utils.cmaes_merge import CmaesPlan, run_cmaes_merge

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _plan(**overrides) -> CmaesPlan:
    kwargs = dict(adapters=("a", "b"), eval_suite="suite.jsonl", budget_seconds=600,
                  population_size=4, max_generations=20, seed=0)
    kwargs.update(overrides)
    return CmaesPlan(**kwargs)


# ---------------------------------------------------------------------------
# run_cmaes_merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "eval_fn,reason",
    [
        pytest.param(lambda w: 1 / 0, "ZeroDivisionError", id="raises"),
        pytest.param(lambda w: float("nan"), "not finite", id="nan"),
        pytest.param(lambda w: None, "NoneType", id="none"),
        pytest.param(lambda w: True, "bool", id="bool"),
    ],
)
def test_every_evaluation_failing_raises_instead_of_converging(eval_fn, reason) -> None:
    with pytest.raises(RuntimeError) as exc:
        run_cmaes_merge(_plan(), eval_fn=eval_fn)
    msg = str(exc.value)
    assert "every evaluation failed" in msg.lower() or "all " in msg.lower()
    assert reason in msg  # the first failure's reason is on screen


def test_the_count_of_failed_evaluations_is_named() -> None:
    with pytest.raises(RuntimeError, match=r"\b\d+ evaluation"):
        run_cmaes_merge(_plan(population_size=3, max_generations=2), eval_fn=lambda w: 1 / 0)


def test_control_one_real_score_among_failures_still_returns_a_result() -> None:
    calls = {"n": 0}

    def flaky(weights):
        calls["n"] += 1
        if calls["n"] == 2:
            return 0.7
        raise RuntimeError("scorer down")

    result = run_cmaes_merge(_plan(max_generations=3), eval_fn=flaky)
    assert result.best_score == pytest.approx(0.7)
    assert result.evaluations >= 2


def test_convergence_needs_a_real_score_not_a_flat_sentinel_history() -> None:
    # Two generations of pure failures followed by real scores: the flat sentinel
    # stretch at the start must not count towards the plateau run.
    calls = {"n": 0}

    def late(weights):
        calls["n"] += 1
        if calls["n"] <= 8:
            raise RuntimeError("warming up")
        return 0.5

    result = run_cmaes_merge(_plan(population_size=4, max_generations=4), eval_fn=late)
    assert result.best_score == pytest.approx(0.5)
    assert result.converged is False


def test_four_failed_generations_do_not_stop_the_search_as_converged() -> None:
    # The plateau reset on a sentinel generation (#1905 review): 16 failed
    # evaluations are four all-failed generations, enough for plateau_run to reach 3
    # if the sentinel counted as "no improvement". Without the reset the search
    # stops there as converged and never sees the real score that follows.
    calls = {"n": 0}

    def late(weights):
        calls["n"] += 1
        if calls["n"] <= 16:
            raise RuntimeError("warming up")
        return 0.5

    result = run_cmaes_merge(_plan(population_size=4, max_generations=8), eval_fn=late)
    assert result.best_score == pytest.approx(0.5)
    assert calls["n"] > 16


def test_control_a_steady_scorer_still_converges() -> None:
    result = run_cmaes_merge(_plan(max_generations=10), eval_fn=lambda w: 0.6)
    assert result.converged is True
    assert math.isfinite(result.best_score)


# ---------------------------------------------------------------------------
# soup adapters merge --strategy cmaes
# ---------------------------------------------------------------------------


def _make_adapter(dir_path: Path) -> str:
    import numpy as np
    from safetensors.numpy import save_file

    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
            rng.standard_normal((8, 16)).astype("float32"),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight":
            rng.standard_normal((16, 8)).astype("float32"),
    }
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 8, "lora_alpha": 16,
        "base_model_name_or_path": "hf-internal-testing/tiny-random-gpt2",
    }), encoding="utf-8")
    return str(dir_path)


@pytest.fixture()
def two_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    a = _make_adapter(tmp_path / "a")
    b = _make_adapter(tmp_path / "b")
    suite = tmp_path / "eval.jsonl"
    suite.write_text('{"prompt": "p", "expected": "e"}\n', encoding="utf-8")
    return a, b, str(suite), tmp_path / "merged"


def _merge(a, b, suite, out, *extra):
    from soup_cli.commands.adapters import app

    return CliRunner().invoke(app, [
        "merge", a, b, "--strategy", "cmaes", "--eval", suite, "--budget", "60s",
        "--population", "4", "--max-generations", "3", "-o", str(out), *extra,
    ])


def test_cli_exits_non_zero_and_writes_nothing_when_every_evaluation_fails(
    two_adapters, monkeypatch
) -> None:
    def broken_scorer(merged_dir, suite):
        raise RuntimeError("scorer could not load the base")

    monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", broken_scorer)
    a, b, suite, out = two_adapters
    result = _merge(a, b, suite, out)
    plain = _plain(result.output)
    assert result.exit_code == 2, (result.exit_code, plain)
    assert not out.exists(), "no merged adapter may be written"
    assert "Converged: True" not in plain
    assert "scorer could not load the base" in plain


def test_cli_control_a_working_scorer_still_writes_the_merge(two_adapters, monkeypatch) -> None:
    monkeypatch.setattr(cm, "_CMAES_SCORER_OVERRIDE", lambda d, s: 0.6)
    a, b, suite, out = two_adapters
    result = _merge(a, b, suite, out)
    assert result.exit_code == 0, _plain(result.output)
    assert (out / "adapter_model.safetensors").is_file()


def test_the_missing_eval_message_names_the_format_the_loader_accepts(
    two_adapters,
) -> None:
    from soup_cli.commands.adapters import app

    a, b, _, out = two_adapters
    result = CliRunner().invoke(app, [
        "merge", a, b, "--strategy", "cmaes", "-o", str(out), "--allow-unscanned",
    ])
    plain = _plain(result.output)
    assert result.exit_code == 2
    assert ".jsonl" in plain
    assert "YAML" not in plain
