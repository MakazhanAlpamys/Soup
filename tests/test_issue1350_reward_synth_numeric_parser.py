"""#1350: the numeric verifier ``soup reward synth`` emits reads golds like the ``math`` reward.

The emitted body parses both sides with ``soup_cli.utils.final_answer``, so ``#### 1,000``,
``\\boxed {42}`` and a ``6*7=42\\n#### 42`` gold score the way ``math_verify_reward`` does. The
calibration report names every reference the verifier rejects, and ``1,000`` is numeric.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.trainer.rewards import math_verify_reward
from soup_cli.utils import reward_synth as rs

# (gold, correct completion) -- the table in #1350.
_PAIRS = [
    ("1000", "#### 1,000"),
    ("42", r"\boxed {42}. The other option, 41, is wrong."),
    ("42", r"First guess \boxed{41}; rechecking, 6*7 = \boxed{42}"),
    ("42", "Six times seven.\n#### 42\nThat was step 3 of 5."),
    ("-42", "#### \u221242"),
    ("6*7=42\n#### 42", "#### 42"),
    ("6*7=42\n#### 42", r"6*7 = \boxed{42}"),
    ("1,000", "1000"),
    ("1,000", "#### 1,000"),
]
_WRONG = [("42", "#### 420"), ("42", r"\boxed{41}")]
_NINE = ["3", "8", "15", "16", "23", "42", "108", "7", "12"]


def _msgs(*contents: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": c}] for c in contents]


def _write_jsonl(path: Path, golds: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"prompt": "q", "answer": g}) + "\n" for g in golds), encoding="utf-8"
    )


@pytest.fixture
def verifier(tmp_path):
    result = rs.synthesize([{"answer": g} for g in ["1000", "42", "-42", "1,000", "2,500"]])
    path = tmp_path / "gen_reward.py"
    path.write_text(result.source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gen_reward_1350", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_fn


@pytest.mark.parametrize(("gold", "completion"), _PAIRS)
def test_a_synthesized_verifier_pays_every_correct_completion(verifier, gold, completion):
    assert verifier(_msgs(completion), answer=[gold]) == [1.0]
    assert math_verify_reward(_msgs(completion), answer=[gold]) == [1.0]


@pytest.mark.parametrize(("gold", "completion"), _WRONG)
def test_a_synthesized_verifier_still_refuses_a_wrong_answer(verifier, gold, completion):
    assert verifier(_msgs(completion), answer=[gold]) == [0.0]


def test_grouped_references_are_detected_as_numeric():
    golds = _NINE[:8] + ["1,000", "2,500"]
    assert rs.detect_kind(golds) == "numeric"
    assert rs.induce_numeric(golds).is_float is False


def _synth(tmp_path, monkeypatch, golds: list[str]):
    from soup_cli.commands.reward import app

    monkeypatch.chdir(tmp_path)
    _write_jsonl(Path("refs.jsonl"), golds)
    return CliRunner().invoke(
        app, ["synth", "refs.jsonl", "-o", "reward.py", "--output-report", "report.json"]
    )


def test_the_issue_reference_set_is_accepted_whole(tmp_path, monkeypatch):
    res = _synth(tmp_path, monkeypatch, _NINE + ["6*7=42\n#### 42"])
    assert res.exit_code == 0, (res.output, repr(res.exception))
    report = json.loads(Path("report.json").read_text(encoding="utf-8"))
    assert report["pos_accept"] == 1.0
    assert report["rejected_references"] == []


def test_a_refusing_calibration_names_every_rejected_reference():
    def digits_only(completions, answer):
        return [1.0 if c[-1]["content"].isdigit() else 0.0 for c in completions]

    refs = ["1", "2", "3", "forty-two", "a dozen"]
    report = rs.calibrate(digits_only, refs, ["x", "y"], kind="numeric")
    assert report.refused
    assert report.rejected_references == ("forty-two", "a dozen")


def test_the_report_names_a_rejected_reference(tmp_path, monkeypatch):
    res = _synth(tmp_path, monkeypatch, _NINE + ["forty-two"])
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert "forty-two" in res.output
    report = json.loads(Path("report.json").read_text(encoding="utf-8"))
    assert report["rejected_references"] == ["forty-two"]


@pytest.mark.parametrize("gold", ["10,00", "1,2", "12,34,567", "1,0000"])
def test_malformed_comma_grouping_is_not_numeric(gold):
    # comma groups must be exactly three digits: "10,00" is not a number
    assert rs.detect_kind([gold]) != "numeric"


def test_perturb_negatives_reads_the_grouped_value():
    # the negative for "1,000" perturbs 1000, not the ",000" tail
    assert "10999.0" in rs.perturb_negatives(["1,000"], "numeric")


def test_a_synthesized_verifier_keeps_large_ints_exact(tmp_path):
    # 2**53 and 2**53 + 1 are the same float; the emitted Decimal compare keeps them apart
    gold = "9007199254740993"
    result = rs.synthesize([{"answer": gold}, {"answer": "42"}, {"answer": "7"}])
    path = tmp_path / "gen_reward_bigint.py"
    path.write_text(result.source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gen_reward_1350_bigint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.reward_fn(_msgs("#### 9007199254740992"), answer=[gold]) == [0.0]
    assert module.reward_fn(_msgs(f"#### {gold}"), answer=[gold]) == [1.0]


def test_a_synthesized_float_verifier_refuses_an_unparseable_gold(tmp_path):
    # float golds bake tol=1e-6, so a gold the parser cannot read must score 0,
    # not reach float(None) mid-eval
    result = rs.synthesize([{"answer": "0.5"}, {"answer": "1.5"}, {"answer": "2.5"}])
    path = tmp_path / "gen_reward_float.py"
    path.write_text(result.source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gen_reward_1350_float", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.reward_fn(_msgs("#### 0.5"), answer=["forty-two"]) == [0.0]
    assert module.reward_fn(_msgs("#### 0.5"), answer=["0.5"]) == [1.0]
