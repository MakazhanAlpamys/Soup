"""v0.71.31 — Judge-in-the-loop suite.

Covers the shared pairwise-judge layer (``eval/judge.pairwise_compare`` /
``pairwise_winrate`` / ``make_soup_pairwise_judge``), ``soup ship --task-mode
pairwise`` (#284), ``task='online_dpo'`` (schema + trainer + routing),
``soup data best-of-n``, and ``soup data evolve``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Deterministic pairwise judge: prefers the LONGER response (A vs B)."""

    def __init__(self, rubric=None):
        self.rubric = rubric or {"scale": {"min": 1, "max": 5}, "criteria": []}

    def compare_pair(self, prompt, resp_a, resp_b):
        if len(resp_a) == len(resp_b):
            return -1
        return 0 if len(resp_a) > len(resp_b) else 1


class _PosBias:
    """A biased judge that ALWAYS says the first response is best."""

    def compare_pair(self, prompt, resp_a, resp_b):
        return 0


# ---------------------------------------------------------------------------
# Task 1 — pairwise_compare / pairwise_winrate
# ---------------------------------------------------------------------------


class TestPairwiseCompare:
    def test_a_preferred(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "longer response", "short", _FakeJudge(), swap=True) == 0

    def test_b_preferred(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "short", "longer response", _FakeJudge(), swap=True) == 1

    def test_tie_on_equal(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "aaaa", "bbbb", _FakeJudge(), swap=True) == -1

    def test_swap_debias_disagreement_is_tie(self):
        from soup_cli.eval.judge import pairwise_compare

        # A judge that ALWAYS says "first is best" disagrees under swap -> tie.
        assert pairwise_compare("p", "x", "y", _PosBias(), swap=True) == -1

    def test_no_swap_uses_single_call(self):
        from soup_cli.eval.judge import pairwise_compare

        assert pairwise_compare("p", "x", "y", _PosBias(), swap=False) == 0

    def test_one_side_failure_is_tie(self):
        from soup_cli.eval.judge import pairwise_compare

        # Definite verdict on the first order, failure (-1) on the swap -> the
        # single unconfirmed verdict must NOT be trusted; result is a tie.
        class _Flaky:
            def __init__(self):
                self.calls = 0

            def compare_pair(self, prompt, resp_a, resp_b):
                self.calls += 1
                return 0 if self.calls == 1 else -1

        assert pairwise_compare("p", "x", "y", _Flaky(), swap=True) == -1


class TestPairwiseWinrate:
    def test_tuned_always_wins(self):
        from soup_cli.eval.judge import pairwise_winrate

        # base short, tuned long -> _FakeJudge prefers tuned every time -> 1.0
        pairs = [("p", "s", "longer"), ("q", "s", "longer")]
        assert pairwise_winrate(pairs, _FakeJudge()) == 1.0

    def test_all_ties_is_half(self):
        from soup_cli.eval.judge import pairwise_winrate

        pairs = [("p", "aaa", "bbb")]  # equal length -> tie -> 0.5
        assert pairwise_winrate(pairs, _FakeJudge()) == 0.5

    def test_empty_pairs_is_half(self):
        from soup_cli.eval.judge import pairwise_winrate

        assert pairwise_winrate([], _FakeJudge()) == 0.5

    def test_mixed_winrate(self):
        from soup_cli.eval.judge import pairwise_winrate

        # tuned wins (long), tuned loses (short), tie (equal) -> (1 + 0 + 0.5)/3
        pairs = [("a", "s", "longer"), ("b", "longer", "s"), ("c", "xx", "yy")]
        assert pairwise_winrate(pairs, _FakeJudge()) == (1.0 + 0.0 + 0.5) / 3


class TestParsePairwise:
    def test_json_winner_a(self):
        from soup_cli.eval.judge import _parse_pairwise

        assert _parse_pairwise('{"winner": "A"}') == 0

    def test_json_winner_b_lowercase(self):
        from soup_cli.eval.judge import _parse_pairwise

        assert _parse_pairwise('the answer is {"winner": "b"}') == 1

    def test_bare_token_fallback(self):
        from soup_cli.eval.judge import _parse_pairwise

        assert _parse_pairwise("A is clearly better") == 0
        assert _parse_pairwise("B wins here") == 1

    def test_unparseable_is_tie(self):
        from soup_cli.eval.judge import _parse_pairwise

        assert _parse_pairwise("I cannot decide") == -1
        assert _parse_pairwise("") == -1


class TestCompareePairMethod:
    def test_compare_pair_parses_llm_reply(self, monkeypatch):
        from soup_cli.eval.judge import JudgeEvaluator

        ev = JudgeEvaluator(provider="ollama", model="m")
        monkeypatch.setattr(ev, "_call_llm", lambda prompt: '{"winner": "B"}')
        assert ev.compare_pair("p", "x", "y") == 1

    def test_compare_pair_llm_failure_is_tie(self, monkeypatch):
        from soup_cli.eval.judge import JudgeEvaluator

        ev = JudgeEvaluator(provider="ollama", model="m")

        def _boom(prompt):
            raise RuntimeError("network down")

        monkeypatch.setattr(ev, "_call_llm", _boom)
        assert ev.compare_pair("p", "x", "y") == -1


# ---------------------------------------------------------------------------
# Task 2 — make_soup_pairwise_judge (TRL BasePairwiseJudge adapter)
# ---------------------------------------------------------------------------


class TestSoupPairwiseJudge:
    def test_judge_returns_best_index(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())  # prefers longer
        # prompt0: [short, long] -> B(1); prompt1: [long, short] -> A(0)
        out = j.judge(["p0", "p1"], [["s", "longer"], ["longer", "s"]])
        assert out == [1, 0]

    def test_judge_tie_returns_minus_one(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())
        assert j.judge(["p"], [["aaaa", "bbbb"]]) == [-1]

    def test_shuffle_order_false_no_swap(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_PosBias())
        assert j.judge(["p"], [["x", "y"]], shuffle_order=False) == [0]

    def test_malformed_pair_returns_minus_one(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        j = make_soup_pairwise_judge(_FakeJudge())
        assert j.judge(["p"], [["only-one"]]) == [-1]

    def test_is_trl_base_pairwise_judge(self):
        from trl import BasePairwiseJudge

        from soup_cli.eval.judge import make_soup_pairwise_judge

        assert isinstance(make_soup_pairwise_judge(_FakeJudge()), BasePairwiseJudge)


# ---------------------------------------------------------------------------
# Task 3 — soup ship --task-mode pairwise (#284)
# ---------------------------------------------------------------------------


class _Task:
    def __init__(self, prompt):
        self.prompt = prompt
        self.category = "default"


class TestShipPairwise:
    def test_pairwise_in_supported(self):
        from soup_cli.utils.ship_verdict import SUPPORTED_TASK_MODES

        assert "pairwise" in SUPPORTED_TASK_MODES

    def test_leg1_pairwise_builds_taskwin(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd

        monkeypatch.setattr(
            "soup_cli.eval.custom.load_eval_tasks",
            lambda path: [_Task("q1"), _Task("q2")],
        )
        monkeypatch.setattr(
            "soup_cli.eval.gate._parse_judge_url",
            lambda url: ("ollama", "m", None),
        )
        monkeypatch.setattr(
            "soup_cli.eval.judge.JudgeEvaluator", lambda **kw: _FakeJudge()
        )
        # base_gen short, tuned_gen long; _FakeJudge prefers long -> winrate 1.0
        tw = ship_cmd._leg1_pairwise(
            lambda p: "s", lambda p: "longer", "x.jsonl", "ollama://m"
        )
        assert tw.mode == "pairwise"
        assert tw.base == 0.5
        assert tw.tuned == 1.0
        assert tw.won is True

    def test_evidence_pairwise_accepted(self):
        import json
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.ship import app

        path = os.path.join(os.getcwd(), "_ev_pairwise_v07131.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.7},
                    "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
                },
                fh,
            )
        try:
            result = CliRunner().invoke(app, ["--evidence", path])
            assert result.exit_code == 0, (result.output, repr(result.exception))
            assert "SHIP" in result.output
        finally:
            os.remove(path)

    def test_evidence_pairwise_tie_is_dont_ship(self):
        import json
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.ship import app

        path = os.path.join(os.getcwd(), "_ev_pairwise_tie_v07131.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.5},
                    "benchmarks": {"mini_mmlu": {"base": 0.8, "tuned": 0.8}},
                },
                fh,
            )
        try:
            result = CliRunner().invoke(app, ["--evidence", path])
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "DON'T SHIP" in result.output
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# Task 4 — online_dpo schema (task literal + fields + cross-validator)
# ---------------------------------------------------------------------------

_ODPO = """
base: sshleifer/tiny-gpt2
task: online_dpo
data:
  train: x.jsonl
training:
  online_dpo_judge: "ollama://llama3"
"""


class TestOnlineDpoSchema:
    def test_happy_parse(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_ODPO)
        assert cfg.task == "online_dpo"
        assert cfg.training.online_dpo_judge == "ollama://llama3"
        assert cfg.training.online_dpo_loss_type == "sigmoid"
        assert cfg.training.online_dpo_max_new_tokens == 64

    def test_reward_model_leg_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: sshleifer/tiny-gpt2\ntask: online_dpo\n"
            "data:\n  train: x.jsonl\n"
            "training:\n  reward_model: some/rm\n"
        )
        assert cfg.training.reward_model == "some/rm"
        assert cfg.training.online_dpo_judge is None

    def test_reject_both_judge_and_reward(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="exactly one"):
            load_config_from_string(_ODPO + "  reward_model: some/rm\n")

    def test_reject_neither(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="judge|reward_model"):
            load_config_from_string(
                "base: sshleifer/tiny-gpt2\ntask: online_dpo\n"
                "data:\n  train: x.jsonl\n"
            )

    def test_reject_mlx_backend(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="transformers"):
            load_config_from_string(
                "base: sshleifer/tiny-gpt2\ntask: online_dpo\nbackend: mlx\n"
                "data:\n  train: x.jsonl\n"
                "training:\n  online_dpo_judge: \"ollama://m\"\n"
            )

    def test_footgun_field_without_task(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="online_dpo"):
            load_config_from_string(
                "base: sshleifer/tiny-gpt2\ntask: sft\n"
                "data:\n  train: x.jsonl\n"
                "training:\n  online_dpo_judge: \"ollama://m\"\n"
            )

    def test_reject_empty_judge(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="non-empty|empty"):
            load_config_from_string(
                "base: sshleifer/tiny-gpt2\ntask: online_dpo\n"
                "data:\n  train: x.jsonl\n"
                "training:\n  online_dpo_judge: \"\"\n"
            )

    def test_loss_type_and_max_new_tokens(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            _ODPO + "  online_dpo_loss_type: ipo\n  online_dpo_max_new_tokens: 128\n"
        )
        assert cfg.training.online_dpo_loss_type == "ipo"
        assert cfg.training.online_dpo_max_new_tokens == 128

    def test_reject_vision_modality(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="modality"):
            load_config_from_string(
                "base: hf-internal-testing/tiny-random-gpt2\ntask: online_dpo\n"
                "modality: vision\ndata:\n  train: x.jsonl\n"
                "training:\n  online_dpo_judge: \"ollama://m\"\n"
            )

    def test_footgun_loss_type_without_task(self):
        import pytest

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception, match="online_dpo"):
            load_config_from_string(
                "base: m\ntask: sft\ndata:\n  train: x.jsonl\n"
                "training:\n  online_dpo_loss_type: ipo\n"
            )

    def test_field_validator_direct(self):
        import pytest

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(Exception, match="null"):
            TrainingConfig(online_dpo_judge="ollama://\x00m")
        with pytest.raises(Exception, match="512"):
            TrainingConfig(online_dpo_judge="x" * 513)
        with pytest.raises(Exception, match="string"):
            TrainingConfig(online_dpo_judge=123)

    def test_max_new_tokens_bounds(self):
        import pytest

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(Exception):
            TrainingConfig(online_dpo_max_new_tokens=0)
        with pytest.raises(Exception):
            TrainingConfig(online_dpo_max_new_tokens=4097)


# ---------------------------------------------------------------------------
# Task 5 — OnlineDPOTrainerWrapper
# ---------------------------------------------------------------------------


def _make_len_judge():
    """A synthetic length-preferring BasePairwiseJudge (no network)."""
    from trl import BasePairwiseJudge

    class _LenJudge(BasePairwiseJudge):
        def judge(self, prompts, completions, shuffle_order=True):
            return [0 if len(c[0]) >= len(c[1]) else 1 for c in completions]

    return _LenJudge()


class TestOnlineDpoWrapper:
    def test_prompt_rows_from_messages(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        ]
        out = OnlineDPOTrainerWrapper._to_prompt_rows(rows)
        assert out == [{"prompt": [{"role": "user", "content": "hi"}]}]

    def test_prompt_rows_from_prompt_field(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        out = OnlineDPOTrainerWrapper._to_prompt_rows([{"prompt": "hello"}])
        assert out == [{"prompt": [{"role": "user", "content": "hello"}]}]

    def test_prompt_rows_keeps_system(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        rows = [
            {
                "messages": [
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        ]
        out = OnlineDPOTrainerWrapper._to_prompt_rows(rows)
        assert out == [
            {"prompt": [{"role": "system", "content": "be nice"},
                        {"role": "user", "content": "hi"}]}
        ]

    def test_prompt_rows_multiturn_keeps_alternation(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ]
            }
        ]
        out = OnlineDPOTrainerWrapper._to_prompt_rows(rows)
        # Interleaved assistant turn kept; conversation ends on the last user
        # turn (trailing assistant dropped -> model generates it on-policy).
        assert out == [
            {
                "prompt": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                ]
            }
        ]

    def test_prompt_rows_no_user_skipped(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        out = OnlineDPOTrainerWrapper._to_prompt_rows(
            [{"messages": [{"role": "assistant", "content": "x"}]}]
        )
        assert out == []

    def test_synthetic_judge_prefers_longer(self):
        j = _make_len_judge()
        # completion 0 longer -> index 0; completion 1 longer -> index 1
        assert j.judge(["p"], [["longer", "s"]]) == [0]
        assert j.judge(["p"], [["s", "longer"]]) == [1]

    def test_setup_builds_trainer_with_synthetic_judge(self):
        import soup_cli.trainer.online_dpo as od
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        cfg = load_config_from_string(
            "base: hf-internal-testing/tiny-random-gpt2\ntask: online_dpo\n"
            "data:\n  train: x.jsonl\n  max_length: 64\n"
            "training:\n  online_dpo_judge: \"ollama://m\"\n"
            "  epochs: 1\n  batch_size: 2\n  online_dpo_max_new_tokens: 8\n"
        )
        od._ONLINE_DPO_JUDGE_OVERRIDE = _make_len_judge()
        try:
            wrapper = OnlineDPOTrainerWrapper(cfg, device="cpu")
            wrapper.setup(
                {"train": [{"messages": [{"role": "user", "content": "hi there"}]},
                           {"messages": [{"role": "user", "content": "hello"}]}]}
            )
            assert wrapper.trainer is not None
        finally:
            od._ONLINE_DPO_JUDGE_OVERRIDE = None


class _Tcfg:
    """Minimal tcfg stub for _build_judge_or_reward (reads two attrs)."""

    def __init__(self, judge=None, reward=None):
        self.online_dpo_judge = judge
        self.reward_model = reward


def _online_dpo_wrapper():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    cfg = load_config_from_string(
        "base: hf-internal-testing/tiny-random-gpt2\ntask: online_dpo\n"
        "data:\n  train: x.jsonl\n"
        "training:\n  online_dpo_judge: \"ollama://m\"\n"
    )
    return OnlineDPOTrainerWrapper(cfg, device="cpu")


class TestBuildJudgeOrReward:
    def test_judge_url_branch(self):
        import soup_cli.trainer.online_dpo as od
        from trl import BasePairwiseJudge

        od._ONLINE_DPO_JUDGE_OVERRIDE = None
        result = _online_dpo_wrapper()._build_judge_or_reward(_Tcfg(judge="ollama://m"))
        assert isinstance(result["judge"], BasePairwiseJudge)
        assert "reward_model" not in result

    def test_reward_model_branch(self, monkeypatch):
        import soup_cli.trainer.online_dpo as od

        od._ONLINE_DPO_JUDGE_OVERRIDE = None
        monkeypatch.setattr(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
        )
        result = _online_dpo_wrapper()._build_judge_or_reward(_Tcfg(reward="some/rm"))
        assert set(result.keys()) == {"reward_model", "reward_processing_class"}

    def test_precedence_override_wins(self):
        import soup_cli.trainer.online_dpo as od

        sentinel = object()
        od._ONLINE_DPO_JUDGE_OVERRIDE = sentinel
        try:
            result = _online_dpo_wrapper()._build_judge_or_reward(_Tcfg(judge="ollama://m"))
            assert result["judge"] is sentinel  # seam beats the URL branch
        finally:
            od._ONLINE_DPO_JUDGE_OVERRIDE = None

    def test_neither_raises(self):
        import pytest

        import soup_cli.trainer.online_dpo as od

        od._ONLINE_DPO_JUDGE_OVERRIDE = None
        with pytest.raises(ValueError, match="judge|reward_model"):
            _online_dpo_wrapper()._build_judge_or_reward(_Tcfg())


# ---------------------------------------------------------------------------
# Task 6 — train.py routing
# ---------------------------------------------------------------------------


class TestOnlineDpoRouting:
    def test_train_routes_online_dpo(self):
        # A distinct-string `elif cfg.task == ...` branch (cannot be shadowed by
        # another equality branch) that instantiates the wrapper.
        import inspect

        from soup_cli.commands import train as train_cmd

        src = inspect.getsource(train_cmd)
        assert 'elif cfg.task == "online_dpo":' in src
        assert "OnlineDPOTrainerWrapper(cfg, **trainer_kwargs)" in src


# ---------------------------------------------------------------------------
# Task 7 — utils/best_of_n.py
# ---------------------------------------------------------------------------


class _ScoreJudge:
    """evaluate(prompt, response) -> JudgeScore(weighted_score=len(response))."""

    def evaluate(self, prompt, response, category="default"):
        from soup_cli.eval.judge import JudgeScore

        return JudgeScore(
            prompt=prompt, response=response, weighted_score=float(len(response))
        )


class TestBestOfN:
    def test_pick_best_argmax(self):
        from soup_cli.utils.best_of_n import judge_pick_best

        pick = judge_pick_best("p", ["a", "abcd", "ab"], _ScoreJudge())
        assert pick.winner_idx == 1
        assert pick.winner == "abcd"
        assert pick.scores == (1.0, 4.0, 2.0)

    def test_pick_best_ties_first(self):
        from soup_cli.utils.best_of_n import judge_pick_best

        pick = judge_pick_best("p", ["ab", "cd"], _ScoreJudge())
        assert pick.winner_idx == 0

    def test_pick_best_empty_raises(self):
        import pytest

        from soup_cli.utils.best_of_n import judge_pick_best

        with pytest.raises(ValueError, match="candidate"):
            judge_pick_best("p", [], _ScoreJudge())

    def test_build_sft_row(self):
        from soup_cli.utils.best_of_n import BestOfNPick, build_sft_row

        row = build_sft_row(
            "p", BestOfNPick(1, "win", (1.0, 3.0)), judge_model="ollama://m"
        )
        assert row["messages"] == [
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "win"},
        ]
        assert row["_best_of_n"]["winner_idx"] == 1
        assert row["_best_of_n"]["judge_model"] == "ollama://m"
        assert row["_best_of_n"]["n"] == 2
        assert row["_best_of_n"]["scores"] == [1.0, 3.0]

    def test_build_dpo_pair(self):
        from soup_cli.utils.best_of_n import BestOfNPick, build_dpo_pair

        pair = build_dpo_pair("p", BestOfNPick(1, "win", (1.0, 3.0)), ["lose", "win"])
        assert pair == {"prompt": "p", "chosen": "win", "rejected": "lose"}

    def test_build_dpo_pair_all_equal_none(self):
        from soup_cli.utils.best_of_n import BestOfNPick, build_dpo_pair

        assert build_dpo_pair("p", BestOfNPick(0, "x", (2.0, 2.0)), ["x", "x"]) is None

    def test_three_candidates_non_boundary_winner_and_loser(self):
        from soup_cli.utils.best_of_n import build_dpo_pair, judge_pick_best

        # scores by length: ["xx"(2), "xxxx"(4), "x"(1)] -> winner idx1, loser idx2
        cands = ["xx", "xxxx", "x"]
        pick = judge_pick_best("p", cands, _ScoreJudge())
        assert pick.winner_idx == 1 and pick.winner == "xxxx"
        pair = build_dpo_pair("p", pick, cands)
        assert pair == {"prompt": "p", "chosen": "xxxx", "rejected": "x"}

    def test_no_top_level_torch(self):
        import ast
        import pathlib

        import soup_cli

        src = (
            pathlib.Path(soup_cli.__file__).parent / "utils" / "best_of_n.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        top = {
            n.names[0].name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import) and n.col_offset == 0
        }
        top |= {
            n.module.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.col_offset == 0 and n.module
        }
        assert "torch" not in top
        assert "transformers" not in top


# ---------------------------------------------------------------------------
# Task 8 — soup data best-of-n command
# ---------------------------------------------------------------------------


class TestBestOfNCli:
    def test_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        result = CliRunner().invoke(app, ["best-of-n", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "best-of-n" in result.output.lower() or "best of n" in result.output.lower()

    def test_reject_bad_n(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = os.path.join(os.getcwd(), "_bon_prompts_bad_n.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "hi"}\n')
        try:
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--base", "m", "--prompts", path, "--n", "1",
                 "--judge", "ollama://m", "-o", "o.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "n must be" in result.output or "between 2" in result.output
        finally:
            os.remove(path)

    def test_reject_ssrf_judge(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = os.path.join(os.getcwd(), "_bon_prompts_ssrf.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "hi"}\n')
        try:
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--base", "m", "--prompts", path, "--n", "4",
                 "--judge", "http://evil.example.com/m", "-o", "o.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
        finally:
            os.remove(path)

    def test_reject_output_outside_cwd(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = os.path.join(os.getcwd(), "_bon_prompts_outcwd.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "hi"}\n')
        try:
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--base", "m", "--prompts", path, "--n", "4",
                 "--judge", "ollama://m", "-o", "../escape.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
        finally:
            os.remove(path)

    def test_happy_path(self, monkeypatch):
        import json
        import os

        from typer.testing import CliRunner

        import soup_cli.commands.data as data_cmd
        import soup_cli.utils.best_of_n as bon
        from soup_cli.commands.data import app

        monkeypatch.setattr(data_cmd, "_load_bon_model", lambda base, device, trust: (None, None))
        monkeypatch.setattr(
            bon, "sample_candidates",
            lambda model, tok, prompt, **kw: ["a", "abcd", "xy"],
        )
        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", lambda **kw: _ScoreJudge())

        ppath = os.path.join(os.getcwd(), "_bon_prompts_ok.jsonl")
        opath = os.path.join(os.getcwd(), "_bon_out.jsonl")
        dpath = os.path.join(os.getcwd(), "_bon_pairs.jsonl")
        with open(ppath, "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "q1"}\n{"prompt": "q2"}\n')
        try:
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--base", "m", "--prompts", ppath, "--n", "3",
                 "--judge", "ollama://m", "-o", opath, "--emit-pairs", dpath],
            )
            assert result.exit_code == 0, (result.output, repr(result.exception))
            rows = [json.loads(x) for x in open(opath, encoding="utf-8") if x.strip()]
            assert len(rows) == 2
            assert rows[0]["messages"][1]["content"] == "abcd"  # longest wins
            assert rows[0]["_best_of_n"]["n"] == 3
            pairs = [json.loads(x) for x in open(dpath, encoding="utf-8") if x.strip()]
            assert pairs[0] == {"prompt": "q1", "chosen": "abcd", "rejected": "a"}
        finally:
            for p in (ppath, opath, dpath):
                if os.path.exists(p):
                    os.remove(p)

    def test_plan_only(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = os.path.join(os.getcwd(), "_bon_prompts_plan.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "hi"}\n')
        try:
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--base", "m", "--prompts", path, "--n", "4",
                 "--judge", "ollama://m", "--plan-only"],
            )
            assert result.exit_code == 0, (result.output, repr(result.exception))
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# Task 9 — utils/evolve.py
# ---------------------------------------------------------------------------


class TestEvolve:
    def test_depth_round_produces_lineage(self):
        from soup_cli.utils.evolve import run_evolve

        # A generate_fn that returns a fresh, distinct instruction each call
        # (so no evolution is eliminated as "unchanged").
        counter = {"i": 0}

        def _gen(prompt):
            counter["i"] += 1
            return f"evolved instruction number {counter['i']}"

        rows = run_evolve(["write a poem"], "depth", 2, _gen)
        assert len(rows) == 2  # one per round
        assert rows[0].round == 1 and rows[1].round == 2
        assert rows[0].seed == "write a poem"
        assert rows[1].seed == "evolved instruction number 1"  # lineage chains
        assert all(r.strategy == "depth" for r in rows)

    def test_breadth_strategy(self):
        from soup_cli.utils.evolve import run_evolve

        rows = run_evolve(["x"], "breadth", 1, lambda p: "brand new instruction")
        assert rows and rows[0].strategy == "breadth"
        assert rows[0].instruction == "brand new instruction"

    def test_unchanged_is_eliminated(self):
        from soup_cli.utils.evolve import run_evolve

        rows = run_evolve(["same"], "depth", 1, lambda p: "same")  # echoes seed
        assert rows == []

    def test_empty_is_eliminated(self):
        from soup_cli.utils.evolve import run_evolve

        assert run_evolve(["x"], "depth", 1, lambda p: "   ") == []

    def test_meta_prompt_echo_eliminated(self):
        from soup_cli.utils.evolve import run_evolve

        rows = run_evolve(["x"], "depth", 1, lambda p: "#Given Prompt#: x")
        assert rows == []

    def test_bad_strategy(self):
        import pytest

        from soup_cli.utils.evolve import run_evolve

        with pytest.raises(ValueError, match="strategy"):
            run_evolve(["x"], "sideways", 1, lambda p: "y")

    def test_bad_rounds(self):
        import pytest

        from soup_cli.utils.evolve import run_evolve

        with pytest.raises(ValueError, match="rounds"):
            run_evolve(["x"], "depth", 0, lambda p: "y")
        with pytest.raises(ValueError, match="rounds"):
            run_evolve(["x"], "depth", 6, lambda p: "y")

    def test_full_elimination_carries_forward_seed(self):
        from soup_cli.utils.evolve import run_evolve

        calls = {"i": 0}

        def _gen(prompt):
            calls["i"] += 1
            # Round 1 echoes the seed (eliminated for all); round 2+ is valid.
            return "seedX" if calls["i"] == 1 else "evolved-real"

        rows = run_evolve(["seedX"], "depth", 2, _gen)
        assert len(rows) == 1
        assert rows[0].round == 2  # round 1 fully eliminated
        assert rows[0].seed == "seedX"  # original seed carried forward

    def test_evolve_instruction_renders_seed(self):
        from soup_cli.utils.evolve import evolve_instruction

        captured = {}

        def _gen(prompt):
            captured["prompt"] = prompt
            return "evolved"

        out = evolve_instruction("my seed", "depth", _gen)
        assert out == "evolved"
        assert "my seed" in captured["prompt"]

    def test_no_top_level_torch(self):
        import ast
        import pathlib

        import soup_cli

        tree = ast.parse(
            (pathlib.Path(soup_cli.__file__).parent / "utils" / "evolve.py").read_text(
                encoding="utf-8"
            )
        )
        top = {
            n.module.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.col_offset == 0 and n.module
        }
        assert "torch" not in top


# ---------------------------------------------------------------------------
# Task 10 — soup data evolve command
# ---------------------------------------------------------------------------


def _write_seeds(name):
    import os

    path = os.path.join(os.getcwd(), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"prompt": "write a poem"}\n{"instruction": "sort a list"}\n')
    return path


class TestEvolveCli:
    def test_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        result = CliRunner().invoke(app, ["evolve", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "evolve" in result.output.lower()

    def test_reject_bad_strategy(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = _write_seeds("_evolve_badstrat.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", path, "--provider", "ollama", "--model", "m",
                 "--strategy", "sideways", "-o", "o.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "strategy" in result.output.lower()
        finally:
            os.remove(path)

    def test_reject_bad_rounds(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = _write_seeds("_evolve_badrounds.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", path, "--provider", "ollama", "--model", "m",
                 "--rounds", "9", "-o", "o.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "rounds" in result.output.lower()
        finally:
            os.remove(path)

    def test_reject_anthropic(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = _write_seeds("_evolve_anthropic.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", path, "--provider", "anthropic", "--model", "m",
                 "-o", "o.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
            assert "anthropic" in result.output.lower()
        finally:
            os.remove(path)

    def test_plan_only(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = _write_seeds("_evolve_plan.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", path, "--provider", "ollama", "--model", "m",
                 "--plan-only"],
            )
            assert result.exit_code == 0, (result.output, repr(result.exception))
        finally:
            os.remove(path)

    def test_reject_output_outside_cwd(self):
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        path = _write_seeds("_evolve_outcwd.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", path, "--provider", "ollama", "--model", "m",
                 "-o", "../escape.jsonl"],
            )
            assert result.exit_code == 2, (result.output, repr(result.exception))
        finally:
            os.remove(path)

    def test_happy_path(self, monkeypatch):
        import json
        import os

        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        counter = {"i": 0}

        def _fake_make(*args, **kwargs):
            def _gen(prompt):
                counter["i"] += 1
                return f"evolved {counter['i']}"

            return _gen

        monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", _fake_make)

        ipath = _write_seeds("_evolve_ok_in.jsonl")
        opath = os.path.join(os.getcwd(), "_evolve_ok_out.jsonl")
        try:
            result = CliRunner().invoke(
                app,
                ["evolve", "--input", ipath, "--provider", "ollama", "--model", "m",
                 "--strategy", "depth", "--rounds", "1", "-o", opath],
            )
            assert result.exit_code == 0, (result.output, repr(result.exception))
            rows = [json.loads(x) for x in open(opath, encoding="utf-8") if x.strip()]
            assert len(rows) == 2  # 2 seeds, 1 round
            assert rows[0]["messages"][0]["role"] == "user"
            assert rows[0]["messages"][0]["content"].startswith("evolved")
            assert rows[0]["_evolve"]["strategy"] == "depth"
            assert rows[0]["_evolve"]["round"] == 1
        finally:
            for p in (ipath, opath):
                if os.path.exists(p):
                    os.remove(p)
