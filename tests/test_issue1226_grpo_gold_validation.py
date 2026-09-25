"""GRPO data-side regression tests for #1226: loader -> validation -> reward.

``_validate_grpo_reward_metadata`` reads every gold with the same parser the rewards use
(``soup_cli.utils.final_answer``). It refuses a gold only when no final answer can be extracted
from it at all, and it reports how many golds are non-numeric (compared as normalised text) so a
dataset that silently scores 0.0 is visible. The parsing matrix itself is in
``test_issue1226_reward_gold_parsing.py``.
"""

from __future__ import annotations

import importlib
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig, TrainingConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.grpo import _prepare_grpo_dataset, _validate_grpo_reward_metadata
from soup_cli.trainer.rewards import (
    load_reward_fn,
    load_reward_fns,
    math_verify_reward,
    validate_reward_funcs,
)

_REPO = Path(__file__).resolve().parents[1]
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY = "golds are non-numeric and are compared as normalised text"


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed console output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _load_prepared(tmp_path: Path, rows: list[dict], fmt: str) -> list[dict]:
    path = _write_jsonl(tmp_path / "train.jsonl", rows)
    loaded = load_dataset(
        DataConfig(train=str(path), format=fmt, val_split=0), preserve_source_columns=True
    )
    return _prepare_grpo_dataset(loaded["train"])


# ===========================================================================
# the real data path: loader -> _prepare_grpo_dataset -> validation -> reward
# ===========================================================================

_GROUP = [
    "Six times seven is 42.\n#### 42",
    r"6*7 = \boxed{42}",
    "The product is 42.\n#### 42",
    "I think it is 41.\n#### 41",
]


def _alpaca_rows() -> list[dict]:
    return [
        {"instruction": f"What is 6*7? (v{i})", "input": "", "output": "6*7=42\n#### 42"}
        for i in range(4)
    ]


def _sharegpt_rows() -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": f"What is 6*7? (v{i})"},
                {"from": "gpt", "value": r"6 times 7 is \boxed{42}"},
            ]
        }
        for i in range(4)
    ]


def _chatml_rows() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"What is 6*7? (v{i})"},
                {"role": "assistant", "content": "The answer is 42."},
            ]
        }
        for i in range(4)
    ]


class TestGrpoDataPath:
    @pytest.mark.parametrize(
        ("fmt", "make_rows"),
        [("alpaca", _alpaca_rows), ("sharegpt", _sharegpt_rows), ("chatml", _chatml_rows)],
    )
    @pytest.mark.parametrize(("reward_fn", "domain"), [("accuracy", None), ("verifiable", "math")])
    def test_three_correct_one_wrong_group_has_reward_variance(
        self, tmp_path, fmt, make_rows, reward_fn, domain
    ):
        prepared = _load_prepared(tmp_path, make_rows(), fmt)
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="train")

        reward = validate_reward_funcs(
            load_reward_fns(tcfg.reward_fn, verifiable_domain=tcfg.verifiable_domain)
        )[0]
        gold = prepared[0]["answer"]
        rewards = reward(
            completions=[_msg(text)[0] for text in _GROUP], answer=[gold] * len(_GROUP)
        )
        assert rewards == [1.0, 1.0, 1.0, 0.0]
        assert len(set(rewards)) > 1  # non-zero group variance: GRPO gets an advantage signal


# ===========================================================================
# a MATH-500-like dataset mixing numeric and LaTeX golds (maintainer ruling)
# ===========================================================================

# (problem, gold, a correct completion, a wrong completion)
MIXED_MATH = [
    ("What is 6*7?", "42", r"\boxed{42}", r"\boxed{41}"),
    (
        "Simplify 28/6.",
        r"\frac{14}{3}",
        r"So it is \boxed{\dfrac{14}{3}}.",
        r"\boxed{\frac{3}{14}}",
    ),
    (
        "Convert (0,3) to polar coordinates.",
        r"\left( 3, \frac{\pi}{2} \right)",
        r"The answer is \left(3,\frac{\pi}{2}\right).",
        r"\boxed{(3, \pi)}",
    ),
    ("Simplify -(q - p).", "p - q", r"\boxed{p-q}", r"\boxed{q - p}"),
    ("How many clips?", "Natalia sold 72 clips.\n#### 72", "#### 72", "#### 27"),
    ("The angle?", r"90^\circ", r"\boxed{90^\circ}", r"\boxed{45^\circ}"),
    ("Half of one?", r"\dfrac{1}{2}", r"\boxed{\tfrac{1}{2}}", r"\boxed{\frac{1}{3}}"),
    ("How many?", "10", "There are 10 of them.", "There are 11 of them."),
]


def _mixed_rows() -> list[dict]:
    return [
        {"instruction": problem, "input": "", "output": "A worked solution.", "answer": gold}
        for problem, gold, _right, _wrong in MIXED_MATH
    ]


class TestMixedMathDataset:
    @pytest.mark.parametrize(("reward_fn", "domain"), [("verifiable", "math"), ("accuracy", None)])
    def test_a_mixed_dataset_loads_and_reports_its_text_golds(
        self, tmp_path, capsys, reward_fn, domain
    ):
        prepared = _load_prepared(tmp_path, _mixed_rows(), "alpaca")
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="train")  # no refusal
        assert f"GRPO train: 5 of 8 {_SUMMARY}" in _plain(capsys.readouterr().out)

    def test_every_row_pays_its_right_answer_and_not_its_wrong_one(self, tmp_path):
        prepared = _load_prepared(tmp_path, _mixed_rows(), "alpaca")
        golds = [row["answer"] for row in prepared]
        right = [_msg(right)[0] for _p, _g, right, _w in MIXED_MATH]
        wrong = [_msg(wrong)[0] for _p, _g, _r, wrong in MIXED_MATH]
        assert math_verify_reward(right, answer=golds) == [1.0] * len(golds)
        assert math_verify_reward(wrong, answer=golds) == [0.0] * len(golds)

    def test_no_summary_when_every_gold_is_a_number(self, tmp_path, capsys):
        prepared = _load_prepared(tmp_path, _alpaca_rows(), "alpaca")
        tcfg = TrainingConfig(reward_fn="verifiable", verifiable_domain="math")
        _validate_grpo_reward_metadata(prepared, tcfg, split="train")
        assert _SUMMARY not in _plain(capsys.readouterr().out)


# ===========================================================================
# a gold with no extractable final answer is refused before generation
# ===========================================================================


class TestUnparseableGoldRefused:
    @pytest.mark.parametrize(
        ("reward_fn", "domain", "reward_name", "gold"),
        [
            ("verifiable", "math", "verifiable/math", "6*7 is\nforty-two"),
            ("verifiable", "math", "verifiable/math", "6*7=42\n####"),
            ("verifiable", "math", "verifiable/math", r"\boxed{}"),
            ("accuracy", None, "accuracy", "Step 1: multiply six by seven.\nStep 2: report it."),
            ("accuracy", None, "accuracy", "6*7=42\n####"),
            ("accuracy", None, "accuracy", r"\boxed{}"),
            ("accuracy", None, "accuracy", "Answer:"),
            ("accuracy,format", None, "accuracy", "Line one\nLine two"),
        ],
    )
    def test_refusal_names_the_row_the_field_and_the_remedy(
        self, reward_fn, domain, reward_name, gold
    ):
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        rows = [
            {"prompt": "q0", "answer": "#### 42"},
            {"prompt": "q1", "answer": r"\frac{14}{3}"},
            {"prompt": "q2", "answer": gold},
        ]
        with pytest.raises(ValueError) as excinfo:
            _validate_grpo_reward_metadata(rows, tcfg, split="train")
        message = str(excinfo.value)
        assert "GRPO train row 2 'answer'" in message
        assert f"reward {reward_name!r}" in message
        assert "Alpaca 'output'" in message  # where the field comes from
        assert "####" in message and r"\boxed{}" in message and "drop the row" in message
        assert "more of the" not in message  # only one row is unreadable

    def test_refusal_counts_every_unreadable_row(self):
        tcfg = TrainingConfig(reward_fn="verifiable", verifiable_domain="math")
        rows = [
            {"prompt": "q0", "answer": "42"},
            {"prompt": "q1", "answer": "Step 1.\nStep 2."},
            {"prompt": "q2", "answer": r"\frac{1}{2}"},
            {"prompt": "q3", "answer": r"\boxed{}"},
            {"prompt": "q4", "answer": "Line one\nLine two"},
        ]
        with pytest.raises(ValueError) as excinfo:
            _validate_grpo_reward_metadata(rows, tcfg, split="train")
        message = str(excinfo.value)
        assert "GRPO train row 1 'answer'" in message
        assert "2 more of the 5 rows have the same problem" in message

    def test_refusal_names_the_split(self):
        with pytest.raises(ValueError, match=r"GRPO validation row 0 'answer'"):
            _validate_grpo_reward_metadata(
                [{"prompt": "q", "answer": "Line one\nLine two"}],
                TrainingConfig(reward_fn="verifiable", verifiable_domain="math"),
                split="validation",
            )

    @pytest.mark.parametrize(
        ("reward_fn", "domain", "gold"),
        [
            (reward_fn, domain, gold)
            for reward_fn, domain in (("accuracy", None), ("verifiable", "math"))
            for gold in (
                "42", 42, 3.5, "1,000", "#### 42", "6*7=42\n#### 42", r"\boxed{42}",
                "The answer is 42.", "**Answer**: 42", r"\(42\)", "Answer:\n42",
                "#### Final Answer\n42", "Paris", "The answer is forty-two.", r"\frac{1}{2}",
                r"\left( 3, \frac{\pi}{2} \right)", "p - q", r"90^\circ",
            )
        ]
        + [("verifiable", "code", "multi\nline stdout")],
    )
    def test_extractable_golds_pass_for_both_rewards(self, reward_fn, domain, gold):
        _validate_grpo_reward_metadata(
            [{"prompt": "q", "answer": gold}],
            TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain),
            split="train",
        )


# ===========================================================================
# seed rows that a rollout backend replaces are never scored, so they are not parsed
# ===========================================================================


class TestSeedRowsBeforeARollout:
    def _tcfg(self, rollout: bool) -> TrainingConfig:
        extra = (
            {"rollout_backend": "openenv", "rollout_func": "soup_cli.envs.calculator:rollout"}
            if rollout
            else {}
        )
        return TrainingConfig(reward_fn="verifiable", verifiable_domain="math", **extra)

    def test_seed_rows_keep_the_presence_check_but_are_not_parsed(self):
        unreadable = [{"prompt": "q", "answer": "Step 1.\nStep 2."}]
        _validate_grpo_reward_metadata(unreadable, self._tcfg(rollout=True), split="train")
        with pytest.raises(ValueError, match="missing or empty 'answer'"):
            _validate_grpo_reward_metadata(
                [{"prompt": "q"}], self._tcfg(rollout=True), split="train"
            )

    def test_the_rows_that_are_scored_are_still_parsed(self):
        unreadable = [{"prompt": "q", "answer": "Step 1.\nStep 2."}]
        for split, rollout in (("rollout", True), ("validation", True), ("train", False)):
            with pytest.raises(ValueError, match=f"GRPO {split} row 0 'answer'"):
                _validate_grpo_reward_metadata(unreadable, self._tcfg(rollout), split=split)


# ===========================================================================
# the repo's own GRPO data still validates, and now gets a signal
# ===========================================================================


class TestShippedDataStillValidates:
    @pytest.mark.parametrize(
        ("reward_fn", "domain", "summary"),
        [("accuracy", None, "2 of 5"), ("verifiable", "math", "2 of 5")],
    )
    def test_the_grpo_reasoning_example_data_validates(
        self, capsys, reward_fn, domain, summary
    ):
        from soup_cli.utils.final_answer import parse_reference

        example = _REPO / "examples" / "configs" / "grpo_reasoning.yaml"
        cfg = load_config_from_string(example.read_text(encoding="utf-8"))
        assert cfg.training.reward_fn == "accuracy"
        data = DataConfig(
            train=str(_REPO / "examples" / "data" / "reasoning_math.jsonl"),
            format="alpaca",
            val_split=0,
        )
        prepared = _prepare_grpo_dataset(load_dataset(data, preserve_source_columns=True)["train"])
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="train")
        assert f"GRPO train: {summary} {_SUMMARY}" in _plain(capsys.readouterr().out)

        parsed = [parse_reference(row["answer"]) for row in prepared]
        assert [p.text for p in parsed] == ["17", "c = 5", "6", "x = 7", "30"]
        assert [p.number for p in parsed] == [Decimal(17), None, Decimal(6), None, Decimal(30)]

    @pytest.mark.parametrize(
        ("module", "reward_fn", "domain"),
        [
            ("calculator", "verifiable", "math"),
            ("guess_number", "verifiable", "math"),
            ("retrieval_qa", "accuracy", None),
        ],
    )
    def test_bundled_rollout_envs_validate_and_pay_a_just_the_value_reply(
        self, module, reward_fn, domain
    ):
        rows = importlib.import_module(f"soup_cli.envs.{module}").rollout([])
        prepared = _prepare_grpo_dataset([dict(row) for row in rows])
        tcfg = TrainingConfig(reward_fn=reward_fn, verifiable_domain=domain)
        _validate_grpo_reward_metadata(prepared, tcfg, split="rollout")

        # Every env prompt says "Reply with just the value/number".
        reward = load_reward_fns(reward_fn, verifiable_domain=domain)[0]
        golds = [row["answer"] for row in prepared]
        assert reward([_msg(gold)[0] for gold in golds], answer=golds) == [1.0] * len(golds)


# ===========================================================================
# the reward-hacking surface: answer spray no longer pays
# ===========================================================================


class TestAnswerSprayNoLongerPays:
    def test_builtin_accuracy_rejects_every_answer_spray_variant(self):
        import soup_cli.utils.reward_stress as rst

        report = rst.run_stress(load_reward_fn("accuracy"), ["42", "17", "1000", "-3"])
        per_attack = {attack.kind: attack for attack in report.attacks}
        assert per_attack["answer_spray"].accepted == 0
        assert report.gameable is False
        assert report.reference_accept == 1.0

    def test_reward_stress_cli_reports_builtin_accuracy_robust(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app as soup_app

        monkeypatch.chdir(tmp_path)
        refs = _write_jsonl(tmp_path / "refs.jsonl", [{"answer": "42"}, {"answer": "17"}])
        report_path = tmp_path / "report.json"
        result = CliRunner().invoke(
            soup_app,
            [
                "reward", "stress", "accuracy",
                "--references", str(refs),
                "--output-report", str(report_path),
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "robust (not gameable)" in _plain(result.output).lower()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_attack = {attack["kind"]: attack for attack in report["attacks"]}
        assert per_attack["answer_spray"]["accepted"] == 0
