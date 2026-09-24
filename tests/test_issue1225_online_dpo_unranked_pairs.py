"""#1225 — Online DPO must not train a pair the judge could not rank.

Soup's pairwise judge answers ``-1`` when it cannot name a winner: the call
failed, the reply did not parse, the judge said "tie", or its verdict changed
when the two completions were swapped. trl 0.29's ``OnlineDPOTrainer`` does not
drop a ``-1``: it builds ``mask = rank == 0``, so every such pair trained as
"the second completion wins", and the loss was averaged over every pair. With
the judge server down, a whole run learned arbitrary labels and finished
normally.

These tests run the REAL wrapper and the REAL trl trainer on
``hf-internal-testing/tiny-random-gpt2`` on CPU. The generated tokens are
captured once and replayed, so two steps that differ only in the judge's ranks
see identical completions, and the LoRA ``B`` matrices are moved off zero so
the policy differs from the reference and every pair has its own loss.
"""

from __future__ import annotations

import logging
import math
import re

import pytest

torch = pytest.importorskip("torch", reason="online_dpo needs the [train] extra")
pytest.importorskip("trl", reason="online_dpo needs the [train] extra")

from soup_cli.trainer.online_dpo import _trl_has_judges  # noqa: E402

needs_pairwise_judge = pytest.mark.skipif(
    not _trl_has_judges(), reason="the pairwise judge= path exists only on trl < 1"
)

_PROMPTS = (
    "hi there friend",
    "hello",
    "what is up",
    "tell me a story",
    "name a colour",
    "count to three",
    "say something kind",
    "what is a cat",
    "pick a number",
    "describe the sea",
    "who are you",
    "what time is it",
)

# Forward passes over a 4-pair batch and over a 2-pair batch are not
# bit-identical on CPU (measured: up to 4.8e-7 in the per-token log-probs), so a
# comparison between batches of different sizes needs a tolerance. Measured on
# the mixed-rank step: 7.5e-9 max gradient difference from the ranked-only
# batch, against 2.1e-2 for what trl computes without the fix.
_ATOL = 1e-6
_RTOL = 1e-5


_RANKING_LOGGER = "soup_cli.trainer.online_dpo_ranking"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed console output (Rich colours and wraps)."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _stop_after(monkeypatch, pairs):
    """Lower the unranked-pair stop so a tiny run reaches it.

    Without the fix there is no stop to lower; the test then runs anyway and
    fails on what the run does, not on a missing import.
    """
    import importlib

    try:
        ranking = importlib.import_module("soup_cli.trainer.online_dpo_ranking")
    except ImportError:
        return
    monkeypatch.setattr(ranking, "MAX_CONSECUTIVE_UNRANKED_PAIRS", pairs)


def _base_trainer_cls():
    """trl's own ``OnlineDPOTrainer``, resolved the way production resolves it."""
    from soup_cli.trainer._trl_compat import resolve_trl_symbol

    return resolve_trl_symbol("OnlineDPOTrainer", "trl.experimental.online_dpo")


def _fixed_judge(ranks):
    """A trl pairwise judge that answers ``ranks`` for every batch."""
    from soup_cli.eval.judge import _base_pairwise_judge_cls

    base = _base_pairwise_judge_cls()

    class _Fixed(base):  # type: ignore[misc, valid-type]
        def __init__(self, answer):
            self.answer = list(answer)
            self.calls = 0

        def judge(self, prompts, completions, shuffle_order=True):
            self.calls += 1
            return list(self.answer)

    return _Fixed(ranks)


def _soup_judge(reply=None, *, down=False):
    """Soup's real ``JudgeEvaluator`` with the network call replaced."""
    import httpx

    from soup_cli.eval.judge import JudgeEvaluator

    evaluator = JudgeEvaluator(provider="ollama", model="llama3.1")

    def _call_llm(prompt):
        if down:
            raise httpx.ConnectError("connection refused")
        return reply

    evaluator._call_llm = _call_llm
    return evaluator


class _Ranked:
    """A reachable Soup evaluator that always prefers the longer response."""

    def compare_pair(self, prompt, resp_a, resp_b):
        return 0 if len(resp_a) >= len(resp_b) else 1


def _build(path, monkeypatch, evaluator, *, batch_size=4, n_prompts=4, extra=""):
    """The real wrapper, set up on the tiny model, writing under ``path / "out"``."""
    import soup_cli.trainer.online_dpo as od
    from soup_cli.config.loader import load_config_from_string

    monkeypatch.setattr(od, "_ONLINE_DPO_JUDGE_OVERRIDE", evaluator)
    cfg = load_config_from_string(
        "base: hf-internal-testing/tiny-random-gpt2\n"
        "task: online_dpo\n"
        "data:\n  train: x.jsonl\n  max_length: 64\n"
        "training:\n"
        '  online_dpo_judge: "ollama://llama3.1"\n'
        f"  epochs: 1\n  batch_size: {batch_size}\n"
        "  online_dpo_max_new_tokens: 6\n  lr: 1e-4\n  quantization: none\n"
        "  seed: 7\n" + extra + f"output: {(path / 'out').as_posix()}\n"
    )
    rows = [
        {"messages": [{"role": "user", "content": text}]} for text in _PROMPTS[:n_prompts]
    ]
    wrapper = od.OnlineDPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows})
    return wrapper


def _move_off_the_reference(model):
    """Give LoRA ``B`` non-zero values so the policy and the reference differ."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)


class _Replay:
    """Generate once, then serve the same tokens (or a subset of pairs) again."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.batch = next(iter(trainer.get_train_dataloader()))
        self.n = len(self.batch["prompt"])
        torch.manual_seed(1234)
        self.generated = trainer._generate(trainer.model, self.batch["prompt"], None)
        self.trainable = [p for p in trainer.model.parameters() if p.requires_grad]

    def step(self, judge, pairs=None, *, step_fn=None, zero_grad=True):
        """One ``training_step`` on ``pairs`` (default: all) with ``judge``.

        ``step_fn`` defaults to the trainer's own ``training_step``; pass trl's
        unbound method to measure what trl does without Soup's subclass.
        """
        pairs = list(range(self.n)) if pairs is None else list(pairs)
        rows = torch.tensor(pairs + [index + self.n for index in pairs])
        generated = tuple(tensor[rows] for tensor in self.generated)
        trainer = self.trainer
        trainer._generate = lambda model, prompts, images=None: generated
        trainer.judge = judge
        model = trainer.model
        if zero_grad:
            model.zero_grad(set_to_none=True)
        inputs = {"prompt": [self.batch["prompt"][index] for index in pairs]}
        try:
            if step_fn is None:
                loss = trainer.training_step(model, inputs)
            else:
                loss = step_fn(trainer, model, inputs)
        finally:
            del trainer._generate
        return loss.detach().clone(), self.grads()

    def grads(self):
        return torch.cat(
            [
                p.grad.detach().flatten().clone()
                if p.grad is not None
                else torch.zeros(p.numel())
                for p in self.trainable
            ]
        )


@pytest.fixture(scope="module")
def _shared_replay(tmp_path_factory):
    """One trainer for the step-level tests: ``training_step`` computes
    gradients and never changes the weights, and every step replays the same
    generated tokens, so the tests cannot affect each other through it."""
    if not _trl_has_judges():
        pytest.skip("the pairwise judge= path exists only on trl < 1")
    with pytest.MonkeyPatch.context() as patch:
        wrapper = _build(tmp_path_factory.mktemp("replay"), patch, _Ranked())
    _move_off_the_reference(wrapper.trainer.model)
    return _Replay(wrapper.trainer)


@pytest.fixture
def replay(_shared_replay):
    trainer = _shared_replay.trainer
    # the judge watch counts across steps; start every test from a clean count
    trainer._soup_unranked_streak = 0
    trainer._soup_warned_partial = False
    return _shared_replay


# ---------------------------------------------------------------------------
# A pair the judge could not rank contributes nothing
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestAnUnrankedStepTrainsNothing:
    @pytest.mark.parametrize(
        "judge_kwargs",
        [
            {"down": True},
            {"reply": '{"winner": "tie"}'},
            {"reply": '{"winner": "A"}'},  # position-biased: A in both orders
            {"reply": "I cannot decide between them."},
        ],
        ids=["server-down", "tie", "position-biased", "unparseable"],
    )
    def test_soup_judge_failure_gives_exactly_zero_gradient(
        self, replay, judge_kwargs
    ):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        judge = make_soup_pairwise_judge(_soup_judge(**judge_kwargs))
        assert judge.judge(["p"], [["first", "second"]]) == [-1]

        loss, grads = replay.step(judge)
        assert torch.count_nonzero(grads).item() == 0, (
            "a step in which the judge ranked no pair still moved "
            f"{torch.count_nonzero(grads).item()} gradient entries"
        )
        assert loss.item() == 0.0

        # the same step with a judge that names the second completion every
        # time does train, which is what the unranked step used to be equal to
        _, grads_second_wins = replay.step(_fixed_judge([1] * replay.n))
        assert torch.count_nonzero(grads_second_wins).item() > 0
        assert not torch.equal(grads, grads_second_wins)

    @pytest.mark.parametrize(
        "rank", [-1, None, 2, "1", float("nan")], ids=["-1", "None", "2", "str", "nan"]
    )
    def test_every_rank_that_is_not_0_or_1_is_unranked(self, replay, rank):
        loss, grads = replay.step(_fixed_judge([rank] * replay.n))
        assert torch.count_nonzero(grads).item() == 0
        assert loss.item() == 0.0

    @pytest.mark.parametrize("rank", [1, 1.0, True], ids=["int", "float", "bool"])
    def test_other_spellings_of_a_real_rank_still_train(self, replay, rank):
        _, grads_spelled = replay.step(_fixed_judge([rank] * replay.n))
        # counted as ranked, not dropped: the fix must not over-reach
        assert replay.trainer.stats["judge/invalid_rate"][-1] == 0.0
        _, grads_int = replay.step(_fixed_judge([1] * replay.n))
        assert torch.equal(grads_spelled, grads_int)


# ---------------------------------------------------------------------------
# Mixed ranks: the step equals a batch of only the ranked pairs
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestMixedRanks:
    def test_gradient_and_loss_equal_a_batch_of_only_the_ranked_pairs(self, replay):
        loss_mixed, grads_mixed = replay.step(_fixed_judge([0, -1, 1, -1]))
        loss_ref, grads_ref = replay.step(_fixed_judge([0, 1]), pairs=[0, 2])

        assert torch.allclose(grads_mixed, grads_ref, atol=_ATOL, rtol=_RTOL), (
            float((grads_mixed - grads_ref).abs().max())
        )
        # the loss is a mean over the KEPT pairs, divided by the accumulation
        # steps exactly as trl divides the mean over all pairs
        assert torch.allclose(loss_mixed, loss_ref, atol=0, rtol=1e-6), (
            loss_mixed.item(),
            loss_ref.item(),
        )

        # what trl itself computes for the same ranks: the two -1 pairs train
        # as "second wins" - far outside the tolerance used above
        _, grads_trl = replay.step(
            _fixed_judge([0, -1, 1, -1]), step_fn=_base_trainer_cls().training_step
        )
        assert float((grads_trl - grads_ref).abs().max()) > 100 * _ATOL

    def test_a_single_ranked_pair_equals_a_batch_of_that_pair(self, replay):
        loss_mixed, grads_mixed = replay.step(_fixed_judge([-1, -1, 1, -1]))
        loss_ref, grads_ref = replay.step(_fixed_judge([1]), pairs=[2])
        assert torch.allclose(grads_mixed, grads_ref, atol=_ATOL, rtol=_RTOL)
        assert torch.allclose(loss_mixed, loss_ref, atol=0, rtol=1e-6)

    def test_chosen_and_rejected_statistics_describe_only_the_ranked_pairs(
        self, replay
    ):
        stats = replay.trainer.stats
        replay.step(_fixed_judge([0, 1]), pairs=[0, 2])
        reference = {key: stats[key][-1] for key in _LABELLED}
        replay.step(_fixed_judge([0, -1, 1, -1]))
        for key in _LABELLED:
            assert math.isclose(stats[key][-1], reference[key], rel_tol=1e-5, abs_tol=1e-6), (
                key,
                stats[key][-1],
                reference[key],
            )
        assert stats["judge/invalid_rate"][-1] == 0.5


_LABELLED = (
    "logps/chosen",
    "logps/rejected",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/margins",
    "rewards/accuracies",
)


# ---------------------------------------------------------------------------
# Control: clean 0/1 ranks train exactly as before
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestCleanRanksAreUnchanged:
    def test_a_clean_step_is_bit_identical_to_trl(self, replay):
        judge_ranks = [0, 1, 1, 0]
        loss_trl, grads_trl = replay.step(
            _fixed_judge(judge_ranks), step_fn=_base_trainer_cls().training_step
        )
        stats_trl = {key: replay.trainer.stats[key][-1] for key in _LABELLED}
        loss_soup, grads_soup = replay.step(_fixed_judge(judge_ranks))
        stats_soup = {key: replay.trainer.stats[key][-1] for key in _LABELLED}
        assert torch.equal(grads_soup, grads_trl)
        assert torch.equal(loss_soup, loss_trl)
        assert stats_soup == stats_trl
        assert replay.trainer.stats["judge/invalid_rate"][-1] == 0.0

    @staticmethod
    def _train_clean(path, monkeypatch):
        path.mkdir()
        wrapper = _build(
            path, monkeypatch, _Ranked(), batch_size=2, n_prompts=8,
            extra="  gradient_accumulation_steps: 2\n  logging_steps: 1\n",
        )
        trainer = wrapper.trainer
        trainer.judge = _fixed_judge([0, 1])
        torch.manual_seed(99)
        trainer.train()
        weights = {
            name: param.detach().clone()
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        }
        return type(trainer), weights

    def test_a_clean_run_trains_bit_identically_to_trl(self, tmp_path, monkeypatch):
        import soup_cli.trainer.online_dpo as od

        soup_cls, soup = self._train_clean(tmp_path / "soup", monkeypatch)
        assert soup_cls is not _base_trainer_cls(), "the fix's trainer was not used"

        # the same run on trl's own trainer, i.e. the code before the fix
        monkeypatch.setattr(od, "make_ranked_pairs_trainer", lambda base: base, raising=False)
        plain_cls, plain = self._train_clean(tmp_path / "trl", monkeypatch)
        assert plain_cls is _base_trainer_cls()

        assert soup.keys() == plain.keys() and soup
        moved = [name for name in soup if "lora_B" in name and soup[name].abs().sum() > 0]
        assert moved, "the control run did not train at all"
        for name in soup:
            assert torch.equal(soup[name], plain[name]), name


# ---------------------------------------------------------------------------
# Gradient accumulation: an unranked micro-batch adds nothing
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestGradientAccumulation:
    def test_an_unranked_micro_batch_adds_nothing_to_the_accumulated_gradient(
        self, replay
    ):
        _, grads_alone = replay.step(_fixed_judge([0, 1, 1, 0]))
        loss_first, _ = replay.step(_fixed_judge([-1] * replay.n))
        loss_second, grads_accumulated = replay.step(
            _fixed_judge([0, 1, 1, 0]), zero_grad=False
        )
        assert loss_first.item() == 0.0
        assert torch.equal(grads_accumulated, grads_alone)

    def test_the_mean_uses_the_ranked_count_not_the_batch_size(self, replay):
        gas = replay.trainer.args.gradient_accumulation_steps
        loss_mixed, _ = replay.step(_fixed_judge([0, -1, 1, -1]))
        loss_ref, _ = replay.step(_fixed_judge([0, 1]), pairs=[0, 2])
        # the reference is trl's own mean over two pairs, divided by `gas`;
        # a denominator of 4 (the batch) would halve it
        assert gas > 1
        assert torch.allclose(loss_mixed, loss_ref, atol=0, rtol=1e-6)
        assert not torch.allclose(loss_mixed, loss_ref / 2, atol=0, rtol=1e-3)


# ---------------------------------------------------------------------------
# Failures are visible: metric, warning, and a stop naming the judge
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestAJudgeThatRanksNothingIsVisible:
    def test_a_server_down_run_stops_with_an_error_naming_the_judge(
        self, tmp_path, monkeypatch, caplog
    ):
        import httpx

        from soup_cli.eval.judge import JudgeEvaluator

        def refused(self, prompt):
            raise httpx.ConnectError("connection refused")

        # the real URL path: no test seam, the judge is built from the config
        monkeypatch.setattr(JudgeEvaluator, "_call_llm", refused)
        _stop_after(monkeypatch, 8)
        wrapper = _build(
            tmp_path, monkeypatch, None, batch_size=4, n_prompts=12,
            extra="  gradient_accumulation_steps: 1\n  logging_steps: 1\n",
        )
        caplog.set_level(logging.WARNING, logger=_RANKING_LOGGER)

        with pytest.raises(RuntimeError) as excinfo:
            wrapper.train()

        message = str(excinfo.value)
        assert type(excinfo.value).__name__ == "JudgeUnusableError", repr(excinfo.value)
        assert "ollama://llama3.1" in message
        assert "training.online_dpo_judge" in message
        assert "8" in message
        # it stopped after the second step (4 + 4 unranked pairs), not later
        assert wrapper.trainer._soup_judged_pairs == 8
        # the first step warned, with the rate, before the stop
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("could not rank any of the 4 pairs" in text for text in warnings), warnings
        # nothing was saved as if the run had succeeded
        assert not (tmp_path / "out" / "adapter_model.safetensors").exists()

    def test_a_position_biased_judge_logs_the_rate_and_the_run_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        """A tiny run (4 pairs, under the 32-pair stop) whose judge always says
        "A": it logs the rate on every step, then ends with an error rather than
        finishing as a success with an untrained adapter."""
        from soup_cli.eval.judge import make_soup_pairwise_judge

        class AlwaysA:
            def compare_pair(self, prompt, resp_a, resp_b):
                return 0

        assert make_soup_pairwise_judge(AlwaysA()).judge(["p"], [["a", "b"]]) == [-1]
        wrapper = _build(
            tmp_path, monkeypatch, AlwaysA(), batch_size=2, n_prompts=4,
            extra="  gradient_accumulation_steps: 1\n  logging_steps: 1\n",
        )
        caplog.set_level(logging.WARNING, logger=_RANKING_LOGGER)
        with pytest.raises(RuntimeError) as excinfo:
            wrapper.train()

        message = str(excinfo.value)
        assert type(excinfo.value).__name__ == "JudgeUnusableError", repr(excinfo.value)
        assert "ollama://llama3.1" in message
        assert "ranked none of the 4 completion pairs of the run" in message
        trainer = wrapper.trainer
        logged = [entry for entry in trainer.state.log_history if "judge/invalid_rate" in entry]
        assert [entry["judge/invalid_rate"] for entry in logged] == [1.0, 1.0]
        # no pair was ranked, so no chosen/rejected statistic is invented
        assert not any("rewards/chosen" in entry for entry in logged)
        assert trainer._soup_judged_pairs == 4
        assert trainer._soup_unranked_pairs == 4
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("could not rank any" in text for text in warnings), warnings
        assert not (tmp_path / "out" / "adapter_model.safetensors").exists()

    def test_partial_ties_warn_once_and_log_their_rate(
        self, tmp_path, monkeypatch, caplog, capsys
    ):
        wrapper = _build(
            tmp_path, monkeypatch, _Ranked(), batch_size=4, n_prompts=12,
            extra="  gradient_accumulation_steps: 1\n  logging_steps: 1\n",
        )
        wrapper.trainer.judge = _fixed_judge([0, -1, 1, 1])
        caplog.set_level(logging.WARNING, logger=_RANKING_LOGGER)
        capsys.readouterr()
        wrapper.train()

        assert "could not rank 3 of 12 pairs (25.0%)" in _plain(capsys.readouterr().out)
        trainer = wrapper.trainer
        rates = [
            entry["judge/invalid_rate"]
            for entry in trainer.state.log_history
            if "judge/invalid_rate" in entry
        ]
        assert rates == [0.25, 0.25, 0.25]
        assert trainer._soup_unranked_pairs == 3
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "could not rank" in r.getMessage()
        ]
        assert len(warnings) == 1, warnings
        assert "1 of 4 pairs" in warnings[0]

    def test_a_ranked_step_resets_the_count_toward_the_stop(self, replay, monkeypatch):
        _stop_after(monkeypatch, 8)
        replay.step(_fixed_judge([-1] * replay.n))
        replay.step(_fixed_judge([0, -1, -1, -1]))  # one ranked pair: not a failure run
        replay.step(_fixed_judge([-1] * replay.n))
        with pytest.raises(RuntimeError, match="ranked none of the last 8"):
            replay.step(_fixed_judge([-1] * replay.n))


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------


@needs_pairwise_judge
class TestContract:
    def test_a_judge_returning_the_wrong_number_of_ranks_is_refused(self, replay):
        with pytest.raises(ValueError, match="returned 3 ranks for 4 completion pairs"):
            replay.step(_fixed_judge([0, 1, 0]))

    @pytest.mark.parametrize(
        ("ranks", "rate"),
        [([0, 1, 1, 0], 0.0), ([0, -1, 1, -1], 0.5), ([-1, -1, -1, -1], 1.0)],
        ids=["clean", "mixed", "none"],
    )
    def test_every_process_gathers_the_same_way_whatever_the_judge_says(
        self, replay, ranks, rate
    ):
        """A distributed run deadlocks if one process issues a collective the
        others do not. What the judge returned must not change which gathers a
        step performs, or the shape of what it gathers."""
        accelerator = replay.trainer.accelerator
        calls = []
        for name in ("gather", "gather_for_metrics"):
            original = getattr(accelerator, name)

            def spy(tensor, *args, _name=name, _original=original, **kwargs):
                calls.append((_name, tuple(tensor.shape)))
                return _original(tensor, *args, **kwargs)

            setattr(accelerator, name, spy)
        try:
            replay.step(_fixed_judge([0, 1, 1, 0]))
            baseline = list(calls)
            calls.clear()
            replay.step(_fixed_judge(ranks))
        finally:
            for name in ("gather", "gather_for_metrics"):
                delattr(accelerator, name)
        assert calls == baseline
        assert replay.trainer.stats["judge/invalid_rate"][-1] == rate


class _Counted:
    """A stand-in trainer carrying only the pair counters."""

    def __init__(self, judged, unranked):
        self._soup_judged_pairs = judged
        self._soup_unranked_pairs = unranked
        self._soup_judge_label = "ollama://llama3.1"


class TestTheEndOfRunAccount:
    def test_a_run_with_no_ranked_pair_raises_naming_the_judge(self):
        from soup_cli.trainer.online_dpo_ranking import (
            JudgeUnusableError,
            report_unranked_pairs,
        )

        with pytest.raises(JudgeUnusableError, match="ollama://llama3.1 ranked none of the 6"):
            report_unranked_pairs(_Counted(6, 6))

    def test_a_run_with_some_unranked_pairs_reports_them(self):
        from soup_cli.trainer.online_dpo_ranking import report_unranked_pairs

        assert report_unranked_pairs(_Counted(8, 2)) == (
            "the judge could not rank 2 of 8 pairs (25.0%); they were left out of the loss."
        )

    @pytest.mark.parametrize(
        "trainer",
        [_Counted(8, 0), _Counted(0, 0), object()],
        ids=["all-ranked", "nothing-judged", "not-counting"],
    )
    def test_nothing_to_report(self, trainer):
        from soup_cli.trainer.online_dpo_ranking import report_unranked_pairs

        assert report_unranked_pairs(trainer) is None


class TestJudgeLabel:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("ollama://llama3.1", "ollama://llama3.1"),
            ("http://localhost:8000/Qwen2.5", "http://localhost:8000/Qwen2.5"),
            ("https://user:s3cret@judge.example.com/m", "https://***@judge.example.com/m"),
            (None, "the configured judge"),
        ],
    )
    def test_the_label_names_the_judge_without_its_credentials(self, url, expected):
        from soup_cli.trainer.online_dpo_ranking import judge_label

        assert judge_label(url) == expected


class TestTheDocstringSaysWhatHappens:
    def test_make_soup_pairwise_judge_no_longer_claims_trl_drops_minus_one(self):
        from soup_cli.eval.judge import make_soup_pairwise_judge

        doc = " ".join((make_soup_pairwise_judge.__doc__ or "").split())
        assert "TRL treats -1 as a dropped sample" not in doc
        assert "rank == 0" in doc
        assert "second completion" in doc
