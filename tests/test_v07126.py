"""v0.71.26 — Closed-loop reward-hacking auto-mitigation.

The trainer DETECTS reward hacking mid-run and SELF-CORRECTS (raise β/kl_coef,
roll back, early-stop, shape reward) instead of only halting. Staged internally:

- Part A / Stage 0: instrumentation + ``log_only`` telemetry (no control action).
- Part B / Stage 1: reversible bang-bang + hysteresis KL controller (first loop).
- Part C / Stage 2: PID-Lagrangian controller + rollback escalation ladder.
- Part D / Stage 3: anti-gaming hardening (smoothing, drift guard, shaping).

Proof-of-mechanism only — validated on SmolLM2-135M + a synthetic reward-hacking
task on one RTX 3050. PPO ships BETA (unit-tested; GPU proof is GRPO-only).
"""

from __future__ import annotations

import json
import os
import types

import pytest
from pydantic import ValidationError


def _fake_buffer(snapshot):
    """A minimal RLSignalBuffer stand-in exposing ``.snapshot()``."""

    class _Buf:
        def snapshot(self):
            return snapshot

    return _Buf()


def _grpo_snapshot(rewards, completions):
    return {
        "completions": completions,
        "per_func": {"length_hack": rewards},
        "rewards": rewards,
    }


def _fake_grpo_trainer(beta=0.02):
    args = types.SimpleNamespace(beta=beta)
    return types.SimpleNamespace(beta=beta, args=args)


def _fake_ppo_trainer(kl_coef=0.2):
    args = types.SimpleNamespace(kl_coef=kl_coef)
    return types.SimpleNamespace(args=args)

# =====================================================================
# Part A / Stage 0 — schema: reward_hack_mitigation field + gate (Task A1)
# =====================================================================


def _yaml(
    task: str = "grpo",
    *,
    backend: str = "transformers",
    detector: str | None = "info_rm",
    mitigation: str = "log_only",
    extra: str = "",
) -> str:
    """Minimal SoupConfig YAML that exercises the mitigation gate."""
    lines = [
        "base: HuggingFaceTB/SmolLM2-135M",
        f"task: {task}",
        f"backend: {backend}",
        "data:",
        "  train: ./data/train.jsonl",
        "  format: chatml",
        "training:",
        f"  reward_hack_mitigation: {mitigation}",
    ]
    if detector is not None:
        lines.append(f"  reward_hack_detector: {detector}")
    if extra:
        lines.extend("  " + ln for ln in extra.strip().splitlines())
    return "\n".join(lines) + "\n"


class TestMitigationSchemaField:
    """``reward_hack_mitigation`` Literal field + task/backend/detector gate."""

    def test_default_is_off(self):
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().reward_hack_mitigation == "off"

    def test_log_only_grpo_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_yaml("grpo"))
        assert cfg.training.reward_hack_mitigation == "log_only"

    def test_ppo_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_yaml("ppo"))
        assert cfg.training.reward_hack_mitigation == "log_only"

    def test_bogus_mode_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises((ValidationError, ValueError)):
            load_config_from_string(_yaml("grpo", mitigation="bogus"))

    def test_sft_rejected_names_field(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_mitigation"):
            load_config_from_string(_yaml("sft"))

    def test_mlx_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(_yaml("grpo", backend="mlx"))

    def test_requires_detector(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_detector"):
            load_config_from_string(_yaml("grpo", detector=None))

    def test_off_without_detector_ok(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_yaml("grpo", detector=None, mitigation="off"))
        assert cfg.training.reward_hack_mitigation == "off"

    def test_yaml_bool_false_coerces_to_off(self):
        # YAML 1.1 coerces unquoted `off` / `no` to bool False. DWIM: map
        # a False (the user wrote `off`) back to the "off" mode.
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig(reward_hack_mitigation=False).reward_hack_mitigation == "off"

    def test_yaml_bool_true_rejected_with_hint(self):
        # `on` / `yes` / `true` coerce to True, which is ambiguous — reject
        # with a message telling the user to quote the mode.
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises((ValidationError, ValueError), match="quote"):
            TrainingConfig(reward_hack_mitigation=True)


# =====================================================================
# Part A / Stage 0 — MitigationLogWriter (Task A2)
# =====================================================================


class TestMitigationLogWriter:
    """Append-only JSONL mitigation log (mirrors TraceLogWriter)."""

    def test_records_one_line_per_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        w = MitigationLogWriter("logs/mit.jsonl")
        w.record(step=1, snapshot={"drop_pct": 0.2, "verdict": "WARN"})
        w.record(step=2, snapshot={"drop_pct": 0.4, "verdict": "HACK"})
        lines = (tmp_path / "logs" / "mit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        e0 = json.loads(lines[0])
        assert e0["step"] == 1 and e0["drop_pct"] == 0.2 and e0["verdict"] == "WARN"
        assert "ts" in e0

    def test_rejects_path_outside_cwd(self, tmp_path, monkeypatch):
        sub = tmp_path / "run"
        sub.mkdir()
        monkeypatch.chdir(sub)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        with pytest.raises(ValueError, match="cwd"):
            MitigationLogWriter(str(tmp_path / "evil.jsonl"))

    def test_cap_mb_bool_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        with pytest.raises((ValueError, TypeError)):
            MitigationLogWriter("m.jsonl", cap_mb=True)

    def test_cap_mb_zero_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        with pytest.raises(ValueError):
            MitigationLogWriter("m.jsonl", cap_mb=0)

    def test_rotation_creates_backup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        w = MitigationLogWriter("m.jsonl")
        # Shrink the cap directly so rotation triggers without writing 1 MB.
        object.__setattr__(w, "_cap_bytes", 120)
        for i in range(6):
            w.record(step=i, snapshot={"x": i})
        assert (tmp_path / "m.jsonl.1").exists()

    def test_redacts_secrets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        w = MitigationLogWriter("m.jsonl")
        w.record(step=1, snapshot={"note": "leaked hf_abcdefgh12345 here"})
        text = (tmp_path / "m.jsonl").read_text()
        assert "hf_abcdefgh12345" not in text
        assert "<redacted>" in text

    @pytest.mark.skipif(os.name == "nt", reason="symlink needs privilege on Windows")
    def test_rotation_refuses_symlink_backup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        target = tmp_path / "secret.txt"
        target.write_text("keep me")
        os.symlink(target, tmp_path / "m.jsonl.1")
        w = MitigationLogWriter("m.jsonl")
        object.__setattr__(w, "_cap_bytes", 120)
        for i in range(6):
            w.record(step=i, snapshot={"x": i})
        assert target.read_text() == "keep me"  # symlink target untouched


# =====================================================================
# Part A / Stage 0 — ControllerState + combine_signals + smooth_signal (Task A3)
# =====================================================================


class TestControllerState:
    """Frozen controller state with bool-before-int + finite validation."""

    def test_defaults(self):
        from soup_cli.utils.reward_hack_control import ControllerState

        s = ControllerState()
        assert s.step == 0 and s.beta == 0.0 and s.kl_coef == 0.0
        assert not s.tripped and s.dwell_count == 0 and s.recovery_attempts == 0

    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        from soup_cli.utils.reward_hack_control import ControllerState

        s = ControllerState()
        with pytest.raises(FrozenInstanceError):
            s.step = 5  # type: ignore[misc]

    def test_bool_step_rejected(self):
        from soup_cli.utils.reward_hack_control import ControllerState

        with pytest.raises((ValueError, TypeError), match="step"):
            ControllerState(step=True)

    def test_negative_counter_rejected(self):
        from soup_cli.utils.reward_hack_control import ControllerState

        with pytest.raises(ValueError, match="dwell_count"):
            ControllerState(dwell_count=-1)

    def test_nonfinite_beta_rejected(self):
        from soup_cli.utils.reward_hack_control import ControllerState

        with pytest.raises(ValueError, match="finite"):
            ControllerState(beta=float("inf"))

    def test_negative_beta_rejected(self):
        from soup_cli.utils.reward_hack_control import ControllerState

        with pytest.raises(ValueError, match="beta"):
            ControllerState(beta=-0.1)


class TestCombineSignals:
    """Multi-signal vote: clamp each finite value to [0,1], then mean."""

    def test_mean(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        got = combine_signals(
            {"info_rm": 0.4, "length_trend": 0.2}, ["info_rm", "length_trend"]
        )
        assert got == pytest.approx(0.3)

    def test_missing_signal_skipped(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        assert combine_signals({"info_rm": 0.4}, ["info_rm", "length_trend"]) == pytest.approx(0.4)

    def test_empty_is_zero(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        assert combine_signals({}, []) == 0.0

    def test_nonfinite_dropped(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        assert combine_signals({"info_rm": float("nan")}, ["info_rm"]) == 0.0

    def test_clamped_high(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        assert combine_signals({"info_rm": 5.0}, ["info_rm"]) == 1.0

    def test_clamped_low(self):
        from soup_cli.utils.reward_hack_control import combine_signals

        assert combine_signals({"info_rm": -2.0}, ["info_rm"]) == 0.0


class TestSmoothSignal:
    """EMA / median / none smoothing of a scalar signal over a window."""

    def test_none_returns_new(self):
        from soup_cli.utils.reward_hack_control import smooth_signal

        assert smooth_signal(0.7, [0.1, 0.2], method="none") == 0.7

    def test_ema(self):
        from soup_cli.utils.reward_hack_control import smooth_signal

        # 0.5*prev + 0.5*new; prev = window[-1] = 0.2
        assert smooth_signal(0.4, [0.1, 0.2], method="ema") == pytest.approx(0.3)

    def test_ema_empty_window_returns_new(self):
        from soup_cli.utils.reward_hack_control import smooth_signal

        assert smooth_signal(0.4, [], method="ema") == 0.4

    def test_median(self):
        from soup_cli.utils.reward_hack_control import smooth_signal

        # median of [0.1, 0.2, 0.9] = 0.2
        assert smooth_signal(0.9, [0.1, 0.2], method="median") == pytest.approx(0.2)

    def test_bad_method_rejected(self):
        from soup_cli.utils.reward_hack_control import smooth_signal

        with pytest.raises(ValueError, match="method"):
            smooth_signal(0.4, [], method="bogus")


# =====================================================================
# Part A / Stage 0 — telemetry helpers (Task A4a)
# =====================================================================


class TestTelemetryHelpers:
    """Pure completion/reward statistics for the log_only telemetry line."""

    def test_mean_token_len(self):
        from soup_cli.utils.reward_hack_control import mean_token_len

        assert mean_token_len(["a b c", "d e"]) == pytest.approx(2.5)

    def test_mean_token_len_empty(self):
        from soup_cli.utils.reward_hack_control import mean_token_len

        assert mean_token_len([]) == 0.0
        assert mean_token_len(["", "  "]) == 0.0

    def test_reward_mean_std(self):
        from soup_cli.utils.reward_hack_control import reward_mean_std

        mean, std = reward_mean_std([1.0, 3.0])
        assert mean == pytest.approx(2.0) and std == pytest.approx(1.0)

    def test_reward_mean_std_empty(self):
        from soup_cli.utils.reward_hack_control import reward_mean_std

        assert reward_mean_std([]) == (0.0, 0.0)

    def test_reward_mean_std_drops_nonfinite(self):
        from soup_cli.utils.reward_hack_control import reward_mean_std

        mean, std = reward_mean_std([1.0, float("nan"), 3.0])
        assert mean == pytest.approx(2.0) and std == pytest.approx(1.0)

    def test_mean_repetition_detects_repeats(self):
        from soup_cli.utils.reward_hack_control import mean_repetition

        repetitive = mean_repetition(["a a a a a a"])
        diverse = mean_repetition(["a b c d e f"])
        assert 0.0 <= diverse < repetitive <= 1.0

    def test_mean_repetition_empty(self):
        from soup_cli.utils.reward_hack_control import mean_repetition

        assert mean_repetition([]) == 0.0


class TestRewardHackLastDropPct:
    """RewardHackCallback exposes the numeric drop_pct for the controller."""

    def test_last_drop_pct_baseline_zero(self):
        from soup_cli.utils.reward_hacking import RewardHackCallback

        cb = RewardHackCallback(detector="info_rm", halt_on_hack=False)
        cb.observe_signal(2.0, step=1)  # establishes baseline health
        assert cb.last_drop_pct() == pytest.approx(0.0)

    def test_last_drop_pct_tracks_drop(self):
        from soup_cli.utils.reward_hacking import RewardHackCallback

        cb = RewardHackCallback(detector="info_rm", halt_on_hack=False)
        cb.observe_signal(2.0, step=1)
        cb.observe_signal(1.0, step=2)  # separation halved → 50% drop
        assert cb.last_drop_pct() == pytest.approx(0.5)


# =====================================================================
# Part A / Stage 0 — synthetic reward-hacking fixtures (Task A4c)
# =====================================================================


class TestSyntheticRewards:
    """The examples/reward_hacking/rewards.py GPU-experiment fixture."""

    def _load(self):
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "reward_hacking"
            / "rewards.py"
        )
        spec = importlib.util.spec_from_file_location("soup_synth_rewards", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_length_hack_scales_with_length(self):
        mod = self._load()
        short = mod.length_hack_reward(["p"], ["a b"])[0]
        longer = mod.length_hack_reward(["p"], ["a b c d e f g h i j"])[0]
        assert 0.0 <= short < longer <= 1.0

    def test_length_hack_capped_at_one(self):
        mod = self._load()
        assert mod.length_hack_reward(["p"], ["w " * 100])[0] == 1.0

    def test_sentinel_reward(self):
        mod = self._load()
        assert mod.sentinel_reward(["p", "p"], ["say GOLD now", "nope"]) == [1.0, 0.0]

    def test_true_score_decoupled_from_length(self):
        mod = self._load()
        padded = "pad " * 40  # games length, no correct answer
        assert mod.length_hack_reward(["p"], [padded])[0] == 1.0
        assert mod.true_score(["p"], [padded])[0] == 0.0

    def test_true_score_decoupled_from_sentinel(self):
        mod = self._load()
        spam = "GOLD GOLD GOLD"
        assert mod.sentinel_reward(["p"], [spam])[0] == 1.0
        assert mod.true_score(["p"], [spam])[0] == 0.0

    def test_true_score_rewards_concise_answer(self):
        mod = self._load()
        assert mod.true_score(["p"], ["the answer is 42"])[0] == 1.0


# =====================================================================
# Part A / Stage 0 — RewardHackMitigationCallback log_only path (Task A4d)
# =====================================================================


class TestMitigationCallbackLogOnly:
    """log_only mode records telemetry and PROVABLY never mutates β."""

    def test_constructs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        cb = RewardHackMitigationCallback(
            mode="log_only",
            detector="info_rm",
            log_writer=MitigationLogWriter("m.jsonl"),
            task="grpo",
        )
        assert cb.mode == "log_only" and cb.detector == "info_rm"

    def test_bad_mode_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        with pytest.raises(ValueError, match="mode"):
            RewardHackMitigationCallback(
                mode="bogus",
                detector="info_rm",
                log_writer=MitigationLogWriter("m.jsonl"),
            )

    def test_bad_signal_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        with pytest.raises(ValueError, match="signal"):
            RewardHackMitigationCallback(
                mode="log_only",
                detector="info_rm",
                log_writer=MitigationLogWriter("m.jsonl"),
                signals=("info_rm", "not_a_signal"),
            )

    def test_log_only_records_and_never_mutates_beta(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        writer = MitigationLogWriter("mit.jsonl")
        buf = _fake_buffer(
            _grpo_snapshot(
                [0.1, 0.2, 0.9, 0.95],
                ["a b", "c GOLD d", "e f g " * 5, "h i"],
            )
        )
        cb = RewardHackMitigationCallback(
            mode="log_only", detector="info_rm", log_writer=writer, buffer=buf, task="grpo"
        )
        trainer = _fake_grpo_trainer(beta=0.02)
        cb.attach(trainer)
        cb.on_step_end(args=None, state=types.SimpleNamespace(global_step=1), control=None)

        # β provably untouched in log_only mode.
        assert trainer.beta == 0.02
        assert trainer.args.beta == 0.02

        lines = (tmp_path / "mit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["mode"] == "log_only" and entry["step"] == 1
        for key in ("drop_pct", "verdict", "beta", "reward_mean", "completion_length_mean", "repetition"):
            assert key in entry

    def test_no_buffer_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        writer = MitigationLogWriter("mit.jsonl")
        cb = RewardHackMitigationCallback(
            mode="log_only", detector="info_rm", log_writer=writer, buffer=None, task="grpo"
        )
        cb.on_step_end(args=None, state=types.SimpleNamespace(global_step=1), control="ctrl")
        assert not (tmp_path / "mit.jsonl").exists()


# =====================================================================
# Part A / Stage 0 — trainer wiring (Task A5)
# =====================================================================


def _fake_trainer_recording(added):
    trainer = types.SimpleNamespace(
        beta=0.02,
        args=types.SimpleNamespace(beta=0.02),
        add_callback=lambda cb: added.append(cb),
    )
    return trainer


class TestAttachMitigationWiring:
    """rl_callbacks_need_buffer + attach_rl_callbacks build the mitigation cb."""

    def test_need_buffer_true_for_mitigation(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import rl_callbacks_need_buffer

        tcfg = TrainingConfig(
            reward_hack_mitigation="log_only", reward_hack_detector="info_rm"
        )
        assert rl_callbacks_need_buffer(tcfg) is True

    def test_need_buffer_false_when_off(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import rl_callbacks_need_buffer

        assert rl_callbacks_need_buffer(TrainingConfig()) is False

    def test_attach_builds_mitigation_callback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback

        tcfg = TrainingConfig(
            reward_hack_mitigation="log_only", reward_hack_detector="info_rm"
        )
        added: list = []
        trainer = _fake_trainer_recording(added)
        attach_rl_callbacks(
            trainer, tcfg, buffer=object(), output_dir=str(tmp_path), task="grpo"
        )
        mit = [c for c in added if isinstance(c, RewardHackMitigationCallback)]
        assert len(mit) == 1
        assert mit[0]._trainer is trainer  # .attach(trainer) was called
        assert mit[0].log_writer.path.name == "mitigation_log.jsonl"

    def test_mitigation_replaces_plain_detector(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback
        from soup_cli.utils.reward_hacking import RewardHackCallback

        tcfg = TrainingConfig(
            reward_hack_mitigation="log_only", reward_hack_detector="info_rm"
        )
        added: list = []
        attach_rl_callbacks(
            _fake_trainer_recording(added),
            tcfg,
            buffer=object(),
            output_dir=str(tmp_path),
            task="grpo",
        )
        assert not any(
            isinstance(c, RewardHackCallback)
            and not isinstance(c, RewardHackMitigationCallback)
            for c in added
        )

    def test_off_keeps_plain_detector(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback
        from soup_cli.utils.reward_hacking import RewardHackCallback

        tcfg = TrainingConfig(reward_hack_detector="info_rm", reward_hack_mitigation="off")
        added: list = []
        attach_rl_callbacks(
            _fake_trainer_recording(added),
            tcfg,
            buffer=object(),
            output_dir=str(tmp_path),
            task="grpo",
        )
        assert any(isinstance(c, RewardHackCallback) for c in added)
        assert not any(isinstance(c, RewardHackMitigationCallback) for c in added)
