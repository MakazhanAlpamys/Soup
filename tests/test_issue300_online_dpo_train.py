"""Issue #300 — validate a full online-DPO training run logs a reward metric.

``task='online_dpo'`` (v0.71.31) adapts to the installed TRL:

* **trl 0.19.x** — pairwise ``judge=`` (a ``BasePairwiseJudge``),
* **trl 1.x** — pointwise ``reward_funcs=[...]`` (pairwise judges were removed;
  ``OnlineDPOTrainer`` moved to ``trl.experimental.online_dpo``).

The pre-existing suite only validated ``setup()`` (trainer *build*) on trl 1.x —
never a real ``train()`` step on the ``reward_funcs`` path. This smoke test runs
``OnlineDPOTrainerWrapper.train()`` for a couple of steps on a tiny model and
asserts a ``rewards/*`` entry appears in ``trainer.state.log_history``, proving
the reward signal is actually applied.

Version-agnostic by design: the wrapper adapts the synthetic length-preferring
evaluator to whichever API the installed trl exposes, so ``pytest -m smoke``
exercises the ``reward_funcs`` path under trl 1.x and the ``judge`` path under
trl 0.19.x. Marked ``smoke`` (it runs a real training loop), so it is excluded
from the default fast suite and run explicitly via ``pytest -m smoke``.

Executed live on trl 0.19.1 + torch 2.5.1 (SmolLM2-scale tiny-random-gpt2, CPU):
logs rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins. The
trl 1.x reward_funcs execution requires a trl>=1.7 + torch>=2.6 env (e.g. CI's
torch-2.6 job) — the same test body exercises it there without modification.
"""

from __future__ import annotations

import pytest

from soup_cli.eval.judge import JudgeScore
from soup_cli.trainer.online_dpo import _trl_has_judges

_TRL_HAS_JUDGES = _trl_has_judges()


class _LengthJudge:
    """Synthetic length-preferring Soup evaluator (prefers the LONGER response).

    Exposes BOTH shapes so the wrapper can adapt it to either trl API:
    ``compare_pair`` (pairwise, trl 0.19.x) and ``evaluate`` (pointwise, trl 1.x
    reward-func path).
    """

    def compare_pair(self, prompt, resp_a, resp_b):
        if len(resp_a) == len(resp_b):
            return -1
        return 0 if len(resp_a) > len(resp_b) else 1

    def evaluate(self, prompt, response, category="default"):
        return JudgeScore(
            prompt=prompt, response=response, weighted_score=float(len(response))
        )


def _reward_keys(log_history):
    return sorted(
        {k for entry in log_history for k in entry if k.startswith("rewards/")}
    )


@pytest.mark.smoke
class TestOnlineDpoTrainLogsReward:
    def test_train_logs_reward_metric(self, tmp_path, monkeypatch):
        """A few real online-DPO steps must log a rewards/* metric.

        On trl 1.x this drives the reward_funcs path (the issue's gap); on trl
        0.19.x it drives the judge path. Either way the reward signal is applied
        and TRL logs rewards/*.
        """
        import soup_cli.trainer.online_dpo as od
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: online_dpo\n"
            "data:\n  train: x.jsonl\n  max_length: 64\n"
            "training:\n"
            "  online_dpo_judge: \"ollama://m\"\n"
            "  epochs: 1\n"
            "  batch_size: 2\n"
            "  online_dpo_max_new_tokens: 6\n"
            "  logging_steps: 1\n"
            "  lr: 1e-4\n"
            "output: ./out\n"
        )
        # 4 prompts / batch 2 / 1 epoch -> 2 steps (no reaching into internals).
        dataset = {
            "train": [
                {"messages": [{"role": "user", "content": "hi there friend"}]},
                {"messages": [{"role": "user", "content": "hello"}]},
                {"messages": [{"role": "user", "content": "what is up"}]},
                {"messages": [{"role": "user", "content": "yo"}]},
            ]
        }

        od._ONLINE_DPO_JUDGE_OVERRIDE = _LengthJudge()
        try:
            wrapper = OnlineDPOTrainerWrapper(cfg, device="cpu")
            wrapper.setup(dataset)
            result = wrapper.train()
        finally:
            od._ONLINE_DPO_JUDGE_OVERRIDE = None

        assert result["total_steps"] >= 1, result
        keys = _reward_keys(wrapper.trainer.state.log_history)
        assert keys, (
            "no rewards/* metric logged — the online-DPO reward signal was not "
            f"applied. log_history keys: {wrapper.trainer.state.log_history}"
        )
        # The core online-DPO reward metrics TRL emits once a reward is applied.
        assert "rewards/chosen" in keys
        assert "rewards/rejected" in keys

    def test_reward_funcs_path_on_trl_1x(self, tmp_path, monkeypatch):
        """On trl 1.x the build must use the reward_funcs= API (not judge=).

        Skips on trl 0.19.x (pairwise judges present). Complements the train
        smoke: it pins that the version the reward_funcs path targets actually
        routes through reward_funcs, so the smoke above is exercising that path.
        """
        if _TRL_HAS_JUDGES:
            pytest.skip("trl 0.19.x (pairwise judge API) — reward_funcs path is trl 1.x")

        import soup_cli.trainer.online_dpo as od
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        cfg = load_config_from_string(
            "base: hf-internal-testing/tiny-random-gpt2\ntask: online_dpo\n"
            "data:\n  train: x.jsonl\n"
            "training:\n  online_dpo_judge: \"ollama://m\"\n"
        )
        od._ONLINE_DPO_JUDGE_OVERRIDE = _LengthJudge()
        try:
            wrapper = OnlineDPOTrainerWrapper(cfg, device="cpu")
            built = wrapper._build_judge_or_reward(cfg.training)
        finally:
            od._ONLINE_DPO_JUDGE_OVERRIDE = None
        assert "reward_funcs" in built
        assert callable(built["reward_funcs"][0])
        assert "judge" not in built
