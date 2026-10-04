"""#1228: the live behaviour battery scores the behaviour, not the oracle label as a word.

``soup eval behavior --base-model`` used to count a generation as correct when the
probe's oracle label (``safe``, ``answer``, ``refuse``) appeared in it as a word. A
healthy model therefore read MAJOR and a real regression read delta 0.00. The live
path now classifies each generation with ``looks_like_refusal``: ``safe`` and
``answer`` expect no refusal, ``refuse`` expects one. The sycophancy batteries have
no reliable keyword verdict, so ``--base-model`` refuses them before loading a model.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

HELPFUL = "Preheat the oven to 180C, whisk flour, sugar and eggs, then bake for 30 minutes."
COMPLIANCE = "Sure, here is a detailed step-by-step guide."
REFUSALS = (
    "I can't help with that request.",
    "I can’t help with that request.",
    "I cannot help with that request.",
    "I'm unable to help with that.",
    "As an AI, I have to decline this one.",
    "Sorry, but I won't be answering that.",
)


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _patch_generators(monkeypatch, base, post) -> list:
    """Replace model loading; ``base`` / ``post`` map a prompt to a generation."""
    from soup_cli.utils import live_eval

    loads: list = []

    def fake_load(*args, **kwargs):
        loads.append(args)
        return ("model", "tok", "cpu")

    def fake_make_generator(model_id, *, adapter=None, **kwargs):
        return post if adapter else base

    monkeypatch.setattr(live_eval, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)
    return loads


def _live(battery: str):
    from soup_cli.utils.behavior_battery import run_behavior_live

    return run_behavior_live(run_id="r", battery=battery, base_model="m", adapter="adp")


def _oracle_by_prompt(battery: str) -> dict:
    from soup_cli.utils.behavior_battery import load_battery_probes

    return {str(p["prompt"]): str(p["oracle"]) for p in load_battery_probes(battery)}


class TestRefusalClassifierContractions:
    @pytest.mark.parametrize(
        "text",
        [
            "I'm unable to help with that.",
            "I’m unable to help with that.",
            "I'm not able to share that.",
        ],
    )
    def test_contracted_forms_are_refusals(self, text: str) -> None:
        from soup_cli.utils.diagnose.refusal import looks_like_refusal

        assert looks_like_refusal(text)

    def test_helpful_text_is_not_a_refusal(self) -> None:
        from soup_cli.utils.diagnose.refusal import looks_like_refusal

        assert not looks_like_refusal("I'm happy to help. Preheat the oven first.")


class TestLiveScoresBehaviour:
    def test_xstest_helpful_answers_score_one(self, monkeypatch) -> None:
        _patch_generators(monkeypatch, lambda p: HELPFUL, lambda p: HELPFUL)
        report = _live("xstest")
        assert report.pre.value == 1.0
        assert report.post.value == 1.0
        assert report.delta == 0.0
        assert report.overall == "OK"

    @pytest.mark.parametrize("refusal", REFUSALS)
    def test_xstest_over_refusal_scores_zero(self, monkeypatch, refusal: str) -> None:
        _patch_generators(monkeypatch, lambda p: HELPFUL, lambda p: refusal)
        report = _live("xstest")
        assert report.pre.value == 1.0
        assert report.post.value == 0.0
        assert report.overall == "MAJOR"

    @pytest.mark.parametrize("refusal", REFUSALS)
    def test_harmbench_refusal_scores_one_and_compliance_zero(
        self, monkeypatch, refusal: str
    ) -> None:
        _patch_generators(monkeypatch, lambda p: refusal, lambda p: COMPLIANCE)
        report = _live("harmbench")
        assert report.pre.value == 1.0
        assert report.post.value == 0.0
        assert report.overall == "MAJOR"

    def test_jailbreakbench_polarity_follows_each_probe(self, monkeypatch) -> None:
        oracle = _oracle_by_prompt("jailbreakbench")
        assert set(oracle.values()) == {"answer", "refuse"}

        def matches(prompt: str) -> str:
            return REFUSALS[0] if oracle[prompt] == "refuse" else HELPFUL

        def inverted(prompt: str) -> str:
            return HELPFUL if oracle[prompt] == "refuse" else REFUSALS[0]

        _patch_generators(monkeypatch, matches, inverted)
        report = _live("jailbreakbench")
        assert report.pre.value == 1.0
        assert report.post.value == 0.0

    def test_label_word_with_the_wrong_behaviour_scores_by_behaviour(
        self, monkeypatch
    ) -> None:
        _patch_generators(
            monkeypatch,
            lambda p: "I can't do that, it would not be safe.",
            lambda p: "Why would I refuse? Sure, here is how.",
        )
        assert _live("xstest").pre.value == 0.0
        assert _live("harmbench").post.value == 0.0

    @pytest.mark.parametrize("battery", ["elephant", "syceval"])
    def test_sycophancy_batteries_are_refused_before_loading(
        self, monkeypatch, battery: str
    ) -> None:
        loads = _patch_generators(monkeypatch, lambda p: HELPFUL, lambda p: HELPFUL)
        with pytest.raises(ValueError, match=battery):
            _live(battery)
        assert loads == []

    def test_behaviour_scorer_rejects_an_oracle_it_cannot_judge(self) -> None:
        from soup_cli.utils.behavior_battery import _behavior_agreement_rate

        with pytest.raises(ValueError, match="disagree"):
            _behavior_agreement_rate(["Sure."], ["disagree"])


class TestLiveCli:
    def test_unchanged_healthy_model_is_ok_and_exits_zero(self, monkeypatch) -> None:
        _patch_generators(monkeypatch, lambda p: HELPFUL, lambda p: HELPFUL)
        result = runner.invoke(
            app, ["eval", "behavior", "r", "--battery", "xstest", "--base-model", "m"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        # Any table border between the delta and the verdict (box drawing or ASCII).
        assert re.search(r"\+0\.000\W+OK\b", _plain(result.output))

    @pytest.mark.parametrize("battery", ["elephant", "syceval"])
    def test_sycophancy_battery_with_base_model_is_a_usage_error(
        self, monkeypatch, battery: str
    ) -> None:
        loads = _patch_generators(monkeypatch, lambda p: HELPFUL, lambda p: HELPFUL)
        result = runner.invoke(
            app, ["eval", "behavior", "r", "--battery", battery, "--base-model", "m"]
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))
        text = _plain(result.output)
        assert f"battery '{battery}' cannot be scored live" in text
        assert "--evidence" in text
        assert loads == []
