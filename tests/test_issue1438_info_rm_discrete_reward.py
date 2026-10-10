"""#1438 — info_rm must not vote on a reward too discrete for a median split.

The default ``reward_fn: accuracy`` (and the ``verifiable`` code reward) return
only 0.0/1.0. A median split of such a step's rewards always leaves at least
one constant half, so the cluster "separation" is a function of the success
rate alone — about 31623 at exactly 50% — not a reward-model health signal.
The run then records that phantom value as the baseline and every later step
reads as a collapse: a policy whose accuracy rises past 50% is halted as
HACK, and the kl_control controller raises beta. The detector now returns no
signal for those steps and warns once, instead of voting.
"""

from __future__ import annotations

import logging
import types

import pytest

from soup_cli.utils.reward_hacking import build_reward_hack_callback


def _binary_rewards(correct: int, n: int = 8) -> list[float]:
    """A 0/1 reward such as reward_fn: accuracy returns."""
    return [0.0] * (n - correct) + [1.0] * correct


class _SeqBuffer:
    """Fake RLSignalBuffer returning a scripted sequence of snapshots."""

    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._index = 0

    def snapshot(self):
        snap = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snap


def _snapshot(rewards, completions=()):
    return {
        "completions": list(completions) or ["a", "b", "c", "d"] * (len(rewards) // 4),
        "per_func": {"accuracy": list(rewards)},
        "rewards": list(rewards),
    }


def _run_steps(cb, buffer, n_steps):
    control = types.SimpleNamespace(should_training_stop=False)
    state = types.SimpleNamespace(global_step=0, log_history=[])
    for step in range(1, n_steps + 1):
        state.global_step = step
        cb.on_step_end(None, state, control)
    return state, control


class TestDiscreteRewardSilence:
    """A 0/1 reward must produce no signal, no baseline, no verdict."""

    def test_rising_accuracy_never_halts(self):
        # Issue's headline case: accuracy 3/8 -> 7/8 is a healthy run.
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        buf = _SeqBuffer([_snapshot(_binary_rewards(k)) for k in (3, 4, 5, 6, 7)])
        cb.buffer = buf
        state, control = _run_steps(cb, buf, 5)
        assert control.should_training_stop is False
        verdicts = [e["reward_hack_verdict"] for e in state.log_history
                    if "reward_hack_verdict" in e]
        assert "HACK" not in verdicts

    def test_first_step_at_fifty_percent_is_not_a_baseline(self):
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        # 4/8 = the phantom ~31623 baseline of the bug report.
        assert cb.compute_signal(_snapshot(_binary_rewards(4))) is None
        assert cb._baseline_health is None
        assert cb._baseline_raw is None

    def test_every_two_valued_split_is_silent(self):
        cb = build_reward_hack_callback(detector="info_rm")
        for k in range(9):
            assert cb.compute_signal(_snapshot(_binary_rewards(k))) is None

    def test_constant_rewards_are_silent(self):
        cb = build_reward_hack_callback(detector="info_rm")
        assert cb.compute_signal(_snapshot([0.5] * 8)) is None

    def test_binary_run_reaching_full_accuracy_stays_silent(self):
        # 0/1 rewards never record a baseline, so a run converging to all-1.0
        # keeps producing no signal rather than a late "collapse" verdict.
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        cb.buffer = _SeqBuffer([_snapshot(_binary_rewards(k)) for k in (3, 5, 8)])
        state, control = _run_steps(cb, cb.buffer, 3)
        assert control.should_training_stop is False
        assert not any("reward_hack_verdict" in e for e in state.log_history)

    def test_warns_once_per_run(self, caplog):
        cb = build_reward_hack_callback(detector="info_rm")
        buf = _SeqBuffer([_snapshot(_binary_rewards(k)) for k in (3, 5, 6)])
        cb.buffer = buf
        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.reward_hacking"):
            _run_steps(cb, buf, 3)
        warnings = [r for r in caplog.records if "info_rm" in r.getMessage()]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "rm_ensemble" in msg  # names the suggested alternatives

    def test_reward_hack_signals_mentioned_alongside_rm_ensemble(self, caplog):
        cb = build_reward_hack_callback(detector="info_rm")
        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.reward_hacking"):
            cb.compute_signal(_snapshot(_binary_rewards(4)))
        assert any("reward_hack_signals" in r.getMessage() for r in caplog.records)


class TestContinuousRewardControl:
    """Control: a genuine separation collapse on continuous rewards still
    reads HACK — the guard must not swallow real signals."""

    _WIDE = _snapshot([0.0, 0.05, 0.95, 1.0])
    _BUNCHED = _snapshot([0.45, 0.49, 0.51, 0.55])

    def test_collapse_still_halts(self):
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        cb.buffer = _SeqBuffer([self._WIDE, self._BUNCHED])
        state, control = _run_steps(cb, cb.buffer, 2)
        assert control.should_training_stop is True
        assert any(
            e.get("reward_hack_verdict") == "HACK" for e in state.log_history
        )

    def test_total_collapse_after_a_varying_baseline_still_halts(self):
        # A fully constant step votes 0.0 once a baseline exists: a 0/1 run
        # never records one, so reaching it means a continuous reward
        # collapsed — a real HACK signal, as on main.
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        cb.buffer = _SeqBuffer([self._WIDE, _snapshot([1.0] * 8)])
        state, control = _run_steps(cb, cb.buffer, 2)
        assert control.should_training_stop is True
        assert state.log_history[-1]["reward_hack_verdict"] == "HACK"

    def test_a_constant_half_with_a_baseline_stays_silent(self):
        # The 0.0 vote is only for a step where EVERY reward is identical.
        # A continuous reward clipped at 0 leaves one constant half but still
        # varies across the split — it carries no separation signal either.
        cb = build_reward_hack_callback(detector="info_rm", halt_on_hack=True)
        cb.buffer = _SeqBuffer(
            [self._WIDE, _snapshot([0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0])]
        )
        state, control = _run_steps(cb, cb.buffer, 2)
        assert control.should_training_stop is False
        assert state.log_history[-1].get("reward_hack_verdict") != "HACK"

    def test_three_distinct_values_still_signal(self):
        cb = build_reward_hack_callback(detector="info_rm")
        assert cb.compute_signal(_snapshot([0.0, 0.1, 0.5, 0.9])) is not None


class TestKlControlController:
    """The same sequence through the mitigation controller must leave beta
    at its starting value (acceptance: built by _attach_reward_hack)."""

    def test_beta_unchanged_on_binary_rewards(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import _attach_reward_hack

        tcfg = TrainingConfig(
            reward_hack_detector="info_rm",
            reward_hack_mitigation="kl_control",
        )
        added = []
        trainer = types.SimpleNamespace(
            beta=0.04,
            args=types.SimpleNamespace(beta=0.04),
            add_callback=lambda cb: added.append(cb),
        )
        buf = _SeqBuffer([_snapshot(_binary_rewards(k)) for k in (3, 5, 6, 7, 7, 7)])
        n = _attach_reward_hack(
            trainer, tcfg, buffer=buf, tokenizer=None,
            output_dir=str(tmp_path), task="grpo",
        )
        assert n == 1 and added
        cb = added[0]
        cb.attach(trainer)
        for step in range(1, 7):
            cb.on_step_end(None, types.SimpleNamespace(global_step=step), None)
        assert trainer.beta == pytest.approx(0.04)
        assert trainer.args.beta == pytest.approx(0.04)


class TestSeparationFloorIsRangeTied:
    """The pooled-variance floor scales with the reward range, so a
    degenerate split can no longer produce a ~31623 baseline."""

    def test_degenerate_split_is_bounded(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        value = compute_cluster_separation([1.0, 1.0], [0.0, 0.0])
        assert 0.0 < value <= 200.0  # was ~31623 under the constant 1e-9 floor

    def test_floor_scales_with_range(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        small = compute_cluster_separation([1.0, 1.0], [0.0, 0.0])
        big = compute_cluster_separation([100.0, 100.0], [0.0, 0.0])
        assert big == pytest.approx(small)  # delta and floor scale together
