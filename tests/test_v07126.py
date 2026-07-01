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
import math
import os
import random
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


class _SeqBuffer:
    """Fake RLSignalBuffer returning a scripted sequence of snapshots."""

    def __init__(self, snapshots):
        self._snapshots = snapshots
        self._index = 0

    def snapshot(self):
        snap = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snap


# Well-separated rewards (healthy) vs bunched rewards (reward model losing grip).
_HEALTHY = _grpo_snapshot([0.0, 0.0, 1.0, 1.0], ["a", "b", "c", "d"])
_HACK = _grpo_snapshot([0.5, 0.5, 0.5, 0.5], ["a", "b", "c", "d"])

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
        for key in (
            "drop_pct",
            "verdict",
            "beta",
            "reward_mean",
            "completion_length_mean",
            "repetition",
        ):
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


# =====================================================================
# Part B / Stage 1 — schema fields + validators (Task B1)
# =====================================================================


class TestStage1Schema:
    """Bang-bang controller tunables + bounds + mutual-exclusion."""

    def _cfg(self, extra: str, *, mitigation: str = "kl_control"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(_yaml("grpo", mitigation=mitigation, extra=extra))

    def test_kl_control_parses(self):
        cfg = self._cfg("reward_hack_beta_floor: 0.05\nreward_hack_beta_ceil: 2.0")
        assert cfg.training.reward_hack_mitigation == "kl_control"
        assert cfg.training.reward_hack_beta_floor == 0.05
        assert cfg.training.reward_hack_beta_ceil == 2.0

    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        t = TrainingConfig()
        assert t.reward_hack_beta_floor == 0.02 and t.reward_hack_beta_ceil == 1.0
        assert t.reward_hack_trip_band == 0.30 and t.reward_hack_release_band == 0.10
        assert t.reward_hack_dwell_steps == 2 and t.reward_hack_release_patience == 3
        assert t.reward_hack_kl_gain == 1.5 and t.reward_hack_signals == ["info_rm"]

    def test_floor_ge_ceil_rejected(self):
        with pytest.raises(ValueError, match="beta_floor"):
            self._cfg("reward_hack_beta_floor: 1.0\nreward_hack_beta_ceil: 0.5")

    def test_release_ge_trip_rejected(self):
        with pytest.raises(ValueError, match="release_band"):
            self._cfg("reward_hack_trip_band: 0.2\nreward_hack_release_band: 0.3")

    def test_unknown_signal_rejected(self):
        with pytest.raises(ValueError, match="signal"):
            self._cfg("reward_hack_signals: [info_rm, not_a_signal]")

    def test_known_signals_accepted(self):
        cfg = self._cfg("reward_hack_signals: [info_rm, length_trend, repetition]")
        assert cfg.training.reward_hack_signals == ["info_rm", "length_trend", "repetition"]

    def test_kl_control_excludes_ref_ema(self):
        with pytest.raises(ValueError, match="ref_model_ema_alpha"):
            self._cfg("ref_model_ema_alpha: 0.9")

    def test_beta_floor_must_be_positive(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_beta_floor: 0.0")

    def test_kl_gain_must_exceed_one(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_kl_gain: 1.0")

    def test_dwell_must_be_positive(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_dwell_steps: 0")

    def test_tunable_without_mode_rejected(self):
        # footgun: setting a control tunable while mitigation is off.
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_mitigation"):
            load_config_from_string(
                _yaml(
                    "grpo", detector=None, mitigation="off", extra="reward_hack_kl_gain: 2.0"
                )
            )


# =====================================================================
# Part B / Stage 1 — BangBangPolicy + bang_bang_step (Task B2)
# =====================================================================


def _bang_policy(**kw):
    from soup_cli.utils.reward_hack_control import BangBangPolicy

    defaults = dict(
        beta_floor=0.02,
        beta_ceil=1.0,
        trip_band=0.3,
        release_band=0.1,
        dwell_steps=2,
        release_patience=2,
        kl_gain=1.5,
    )
    defaults.update(kw)
    return BangBangPolicy(**defaults)


def _run_bang(policy, votes, beta0=None):
    from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

    state = ControllerState(beta=beta0 if beta0 is not None else policy.beta_floor)
    actions = []
    for vote in votes:
        state, action = bang_bang_step(policy, state, vote=vote)
        actions.append(action)
    return state, actions


class TestBangBang:
    """Reversible bang-bang controller with dwell + release hysteresis."""

    def test_policy_floor_lt_ceil(self):
        from soup_cli.utils.reward_hack_control import BangBangPolicy

        with pytest.raises(ValueError, match="floor"):
            BangBangPolicy(
                beta_floor=1.0,
                beta_ceil=0.5,
                trip_band=0.3,
                release_band=0.1,
                dwell_steps=2,
                release_patience=2,
                kl_gain=1.5,
            )

    def test_policy_release_lt_trip(self):
        with pytest.raises(ValueError, match="release"):
            _bang_policy(trip_band=0.2, release_band=0.3)

    def test_policy_kl_gain_gt_one(self):
        with pytest.raises(ValueError, match="kl_gain"):
            _bang_policy(kl_gain=1.0)

    def test_policy_beta_floor_positive(self):
        with pytest.raises(ValueError, match="beta_floor"):
            _bang_policy(beta_floor=0.0)

    def test_below_band_no_trip(self):
        state, _ = _run_bang(_bang_policy(), [0.1, 0.1, 0.1, 0.1])
        assert not state.tripped and state.beta == pytest.approx(0.02)

    def test_dwell_then_trip(self):
        state, _ = _run_bang(_bang_policy(), [0.5, 0.5])
        assert state.tripped and state.beta == pytest.approx(0.03)  # 0.02 * 1.5

    def test_keeps_raising_while_hacking(self):
        state, _ = _run_bang(_bang_policy(), [0.5, 0.5, 0.5, 0.5])
        assert state.tripped and state.beta == pytest.approx(0.02 * 1.5**3)

    def test_no_flap_on_alternating(self):
        # A naive controller would flap; dwell + release hysteresis prevents it.
        state, _ = _run_bang(_bang_policy(), [0.5, 0.05, 0.5, 0.05, 0.5, 0.05])
        assert not state.tripped and state.beta == pytest.approx(0.02)

    def test_release_reverses(self):
        # trip (2 steps) then release_patience (2 steps low) → β back to floor.
        state, _ = _run_bang(_bang_policy(), [0.5, 0.5, 0.05, 0.05])
        assert state.beta == pytest.approx(0.02) and not state.tripped

    def test_beta_clamped_to_ceil(self):
        state, _ = _run_bang(_bang_policy(beta_ceil=0.05), [0.5] * 10)
        assert state.beta == pytest.approx(0.05)

    def test_action_is_frozen(self):
        from dataclasses import FrozenInstanceError

        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        _, action = bang_bang_step(_bang_policy(), ControllerState(beta=0.02), vote=0.5)
        with pytest.raises(FrozenInstanceError):
            action.new_beta = 1.0  # type: ignore[misc]

    def test_action_verdict_and_reason(self):
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        _, action = bang_bang_step(
            _bang_policy(dwell_steps=1), ControllerState(beta=0.02), vote=0.9
        )
        assert action.verdict == "HACK" and "raise" in action.reason.lower()


# =====================================================================
# Part B / Stage 1 — kl_control callback: β dual-write + kl_coef (Task B3)
# =====================================================================


def _kl_policy(**kw):
    from soup_cli.utils.reward_hack_control import BangBangPolicy

    defaults = dict(
        beta_floor=0.02,
        beta_ceil=1.0,
        trip_band=0.3,
        release_band=0.1,
        dwell_steps=1,
        release_patience=1,
        kl_gain=2.0,
    )
    defaults.update(kw)
    return BangBangPolicy(**defaults)


def _kl_callback(tmp_path, buffer, *, task="grpo", signals=("info_rm",), policy=None):
    from soup_cli.utils.reward_hack_control import (
        MitigationLogWriter,
        RewardHackMitigationCallback,
    )

    return RewardHackMitigationCallback(
        mode="kl_control",
        detector="info_rm",
        log_writer=MitigationLogWriter(str(tmp_path / "m.jsonl")),
        signals=signals,
        buffer=buffer,
        task=task,
        bang_bang=policy or _kl_policy(),
    )


class TestKlControlCallback:
    """kl_control mutates β (GRPO, dual-write) / kl_coef (PPO) via the controller."""

    def test_kl_control_requires_policy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        with pytest.raises(ValueError, match="kl_control"):
            RewardHackMitigationCallback(
                mode="kl_control",
                detector="info_rm",
                log_writer=MitigationLogWriter("m.jsonl"),
            )

    def test_grpo_dual_write_on_hack(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _kl_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK]))
        trainer = _fake_grpo_trainer(beta=0.02)
        cb.attach(trainer)
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)  # baseline
        cb.on_step_end(None, types.SimpleNamespace(global_step=2), None)  # hack → raise
        assert trainer.beta == pytest.approx(0.04)  # 0.02 * 2.0
        assert trainer.args.beta == pytest.approx(0.04)  # DUAL write (variant path)

    def test_ppo_kl_coef_mutated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _kl_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK]), task="ppo")
        trainer = _fake_ppo_trainer(kl_coef=0.2)
        cb.attach(trainer)
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        cb.on_step_end(None, types.SimpleNamespace(global_step=2), None)
        assert trainer.args.kl_coef == pytest.approx(0.4)  # 0.2 * 2.0

    def test_recovery_relaxes_beta(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _kl_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK, _HEALTHY]))
        trainer = _fake_grpo_trainer(beta=0.02)
        cb.attach(trainer)
        for step in (1, 2, 3):  # baseline, hack (raise), recovery (relax)
            cb.on_step_end(None, types.SimpleNamespace(global_step=step), None)
        assert trainer.beta == pytest.approx(0.02)  # relaxed back to floor
        assert not cb._state.tripped

    def test_kl_control_logs_vote_and_action(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _kl_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK]))
        cb.attach(_fake_grpo_trainer(beta=0.02))
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        cb.on_step_end(None, types.SimpleNamespace(global_step=2), None)
        lines = (tmp_path / "m.jsonl").read_text().strip().splitlines()
        last = json.loads(lines[-1])
        for key in ("vote", "new_beta", "tripped", "action"):
            assert key in last


class TestAttachKlControl:
    """attach_rl_callbacks builds the bang-bang policy from tcfg for kl_control."""

    def test_attach_kl_control_builds_policy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback

        tcfg = TrainingConfig(
            reward_hack_mitigation="kl_control",
            reward_hack_detector="info_rm",
            reward_hack_kl_gain=3.0,
            reward_hack_signals=["info_rm", "length_trend"],
        )
        added: list = []
        attach_rl_callbacks(
            _fake_trainer_recording(added),
            tcfg,
            buffer=object(),
            output_dir=str(tmp_path),
            task="grpo",
        )
        mit = [c for c in added if isinstance(c, RewardHackMitigationCallback)]
        assert len(mit) == 1
        assert mit[0].mode == "kl_control"
        assert mit[0].bang_bang is not None and mit[0].bang_bang.kl_gain == 3.0
        assert mit[0].signals == ("info_rm", "length_trend")


class TestPpoBufferParity:
    """PPO wires the RLSignalBuffer + mitigation callback (BETA — GRPO gets the
    on-GPU proof; PPO's kl_coef mutation is unit-tested in TestKlControlCallback)."""

    def test_ppo_setup_wires_buffer_and_mitigation(self):
        import inspect

        from soup_cli.trainer import ppo

        src = inspect.getsource(ppo)
        assert "RLSignalBuffer" in src
        assert "rl_callbacks_need_buffer" in src
        assert "attach_rl_callbacks" in src
        assert 'task="ppo"' in src or "task='ppo'" in src


class TestRewardHackMitigationCli:
    """`soup train --reward-hack-mitigation <mode>` flag + re-exec passthrough."""

    def _runner(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        return CliRunner(), app

    def test_help_shows_flag(self):
        runner, app = self._runner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, result.output
        assert "--reward-hack-mitigation" in result.output

    def test_override_without_detector_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfg.yaml").write_text(
            "base: HuggingFaceTB/SmolLM2-135M\n"
            "task: grpo\n"
            "data:\n  train: ./train.jsonl\n  format: chatml\n"
            "training:\n  reward_fn: accuracy\n"
        )
        runner, app = self._runner()
        result = runner.invoke(
            app,
            ["train", "--config", "cfg.yaml", "--reward-hack-mitigation", "log_only", "--yes"],
        )
        assert result.exit_code == 1
        assert "reward_hack_detector" in result.output

    def test_reexec_passthrough_present(self):
        import inspect

        from soup_cli.commands import train as train_mod

        src = inspect.getsource(train_mod)
        assert "--reward-hack-mitigation" in src
        assert "cfg.training.reward_hack_mitigation" in src


# =====================================================================
# Part C / Stage 2 — schema fields (PID + rollback) (Task C1)
# =====================================================================


class TestStage2Schema:
    """PID-Lagrangian + rollback tunables + bounds + cross-validators."""

    def _cfg(self, extra: str = "", *, mitigation: str = "pid_lagrangian"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(_yaml("grpo", mitigation=mitigation, extra=extra))

    def test_pid_parses(self):
        cfg = self._cfg("reward_hack_pid_kp: 0.8\nreward_hack_signal_target: 0.2")
        assert cfg.training.reward_hack_mitigation == "pid_lagrangian"
        assert cfg.training.reward_hack_pid_kp == 0.8
        assert cfg.training.reward_hack_signal_target == 0.2

    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        t = TrainingConfig()
        assert t.reward_hack_pid_kp == 0.5 and t.reward_hack_pid_ki == 0.1
        assert t.reward_hack_pid_kd == 0.05 and t.reward_hack_signal_target == 0.15
        assert t.reward_hack_rollback is False
        assert t.reward_hack_rollback_patience == 3
        assert t.reward_hack_max_recovery_attempts == 2

    def test_rollback_requires_checkpoint_cadence(self):
        with pytest.raises(ValueError, match="rl_checkpoint_save_every_steps"):
            self._cfg("reward_hack_rollback: true")

    def test_rollback_with_cadence_ok(self):
        cfg = self._cfg("reward_hack_rollback: true\nrl_checkpoint_save_every_steps: 10")
        assert cfg.training.reward_hack_rollback is True

    def test_pid_param_requires_pid_mode(self):
        # setting a PID param under kl_control is a no-op footgun.
        with pytest.raises(ValueError, match="pid_lagrangian"):
            self._cfg("reward_hack_pid_kp: 0.9", mitigation="kl_control")

    def test_pid_param_under_off_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_mitigation"):
            load_config_from_string(
                _yaml(
                    "grpo",
                    detector=None,
                    mitigation="off",
                    extra="reward_hack_pid_kp: 0.9",
                )
            )

    def test_pid_kp_non_negative(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_pid_kp: -0.1")

    def test_signal_target_below_one(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_signal_target: 1.0")


# =====================================================================
# Part C / Stage 2 — PIDLagrangianPolicy + pid_step (Task C2)
# =====================================================================


def _pid_policy(**kw):
    from soup_cli.utils.reward_hack_control import PIDLagrangianPolicy

    defaults = dict(
        kp=1.0,
        ki=0.0,
        kd=0.0,
        signal_target=0.15,
        beta_floor=0.02,
        beta_ceil=10.0,
        integral_clamp=10.0,
    )
    defaults.update(kw)
    return PIDLagrangianPolicy(**defaults)


def _run_pid(policy, signals, beta0=None):
    from soup_cli.utils.reward_hack_control import ControllerState, pid_step

    state = ControllerState(beta=beta0 if beta0 is not None else policy.beta_floor)
    states = []
    for signal in signals:
        state, _ = pid_step(policy, state, signal=signal)
        states.append(state)
    return states


class TestPidLagrangian:
    """PID-Lagrangian controller: P/I/D isolated, anti-windup, output clamp."""

    def test_policy_gains_non_negative(self):
        with pytest.raises(ValueError, match="kp"):
            _pid_policy(kp=-1.0)

    def test_policy_floor_lt_ceil(self):
        with pytest.raises(ValueError, match="floor"):
            _pid_policy(beta_floor=5.0, beta_ceil=1.0)

    def test_policy_integral_clamp_positive(self):
        with pytest.raises(ValueError, match="integral_clamp"):
            _pid_policy(integral_clamp=0.0)

    def test_proportional_only(self):
        # kp=1, error = 0.65 - 0.15 = 0.5 → β = floor + 0.5
        states = _run_pid(_pid_policy(kp=1.0, ki=0.0, kd=0.0), [0.65])
        assert states[0].beta == pytest.approx(0.02 + 0.5)

    def test_integral_accumulates(self):
        states = _run_pid(_pid_policy(kp=0.0, ki=1.0, kd=0.0), [0.35, 0.35])
        assert states[1].beta > states[0].beta  # integral grows β

    def test_integral_anti_windup(self):
        # constant error 0.5 with clamp 0.3 → integral saturates → β flat.
        states = _run_pid(
            _pid_policy(kp=0.0, ki=1.0, kd=0.0, integral_clamp=0.3, beta_ceil=100.0),
            [0.65] * 5,
        )
        assert states[0].beta == pytest.approx(0.02 + 0.3)
        assert states[4].beta == pytest.approx(states[0].beta)

    def test_derivative_spikes_then_settles(self):
        # kd=1: β spikes on the error jump then returns to floor on constant error.
        states = _run_pid(
            _pid_policy(kp=0.0, ki=0.0, kd=1.0), [0.15, 0.65, 0.65]
        )
        assert states[1].beta > 0.02  # jump
        assert states[2].beta == pytest.approx(0.02)  # deriv back to 0

    def test_output_clamped_to_ceil(self):
        states = _run_pid(_pid_policy(kp=1000.0, beta_ceil=0.5), [0.9])
        assert states[0].beta == pytest.approx(0.5)

    def test_output_clamped_to_floor(self):
        # negative error → control negative → β clamped to floor (never < 0).
        states = _run_pid(_pid_policy(kp=1.0), [0.0])
        assert states[0].beta == pytest.approx(0.02)

    def test_relaxes_after_raise(self):
        # raise on positive error, then relax when the signal drops below target.
        pol = _pid_policy(kp=0.0, ki=1.0, kd=0.0, integral_clamp=10.0, beta_ceil=100.0)
        states = _run_pid(pol, [0.65, 0.65, 0.0, 0.0, 0.0])
        assert states[-1].beta < states[1].beta


# =====================================================================
# Part C / Stage 2 — RLCheckpointCallback.restore_checkpoint (Task C3)
# =====================================================================


class _FakeSavableModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        import os

        os.makedirs(path, exist_ok=True)
        self.saved_to = path


class TestRestoreCheckpoint:
    """restore_checkpoint reloads adapter + optimizer state (RL rollback)."""

    def _cb(self, tmp_path):
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        return build_rl_checkpoint_callback(
            RLCheckpointConfig(save_every_steps=1), output_dir="run", task="grpo"
        )

    def test_restore_optimizer_roundtrip(self, tmp_path, monkeypatch):
        import copy

        import torch

        monkeypatch.chdir(tmp_path)
        cb = self._cb(tmp_path)
        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.1, momentum=0.9)
        param.grad = torch.ones(2)
        opt.step()  # creates a momentum buffer
        saved = copy.deepcopy(opt.state_dict()["state"][0]["momentum_buffer"])
        cb.save_checkpoint(step=1, model=_FakeSavableModel(), optimizer=opt)
        # mutate optimizer state
        param.grad = torch.ones(2) * 7
        opt.step()
        assert not torch.allclose(
            opt.state_dict()["state"][0]["momentum_buffer"], saved
        )
        ok = cb.restore_checkpoint(step=1, model=_FakeSavableModel(), optimizer=opt)
        assert ok is True
        assert torch.allclose(
            opt.state_dict()["state"][0]["momentum_buffer"], saved
        )

    def test_restore_missing_step_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._cb(tmp_path)
        assert cb.restore_checkpoint(step=99, model=None, optimizer=None) is False

    def test_restore_adapter_roundtrip(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        peft = pytest.importorskip("peft")
        import torch.nn as nn

        monkeypatch.chdir(tmp_path)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.lin(x)

        model = peft.get_peft_model(
            Tiny(), peft.LoraConfig(target_modules=["lin"], r=2)
        )
        key = "base_model.model.lin.lora_A.default.weight"
        original = model.state_dict()[key].clone()
        cb = self._cb(tmp_path)
        cb.save_checkpoint(step=1, model=model, optimizer=None)
        # mutate the LoRA weight in place
        with torch.no_grad():
            dict(model.named_parameters())[
                "base_model.model.lin.lora_A.default.weight"
            ].add_(5.0)
        assert not torch.allclose(model.state_dict()[key], original)
        ok = cb.restore_checkpoint(step=1, model=model, optimizer=None)
        assert ok is True
        assert torch.allclose(model.state_dict()[key], original)


# =====================================================================
# Part C / Stage 2 — pid_lagrangian callback + escalation ladder (Task C4)
# =====================================================================


class _FakeCkptCb:
    def __init__(self, saved):
        self._saved = list(saved)
        self.restore_calls = []

    def restore_checkpoint(self, *, step, model, optimizer):
        self.restore_calls.append(step)
        return True


def _pid_callback(tmp_path, buffer, *, rollback=False, rollback_patience=2,
                  max_recovery_attempts=1, ckpt_cb=None):
    from soup_cli.utils.reward_hack_control import (
        MitigationLogWriter,
        PIDLagrangianPolicy,
        RewardHackMitigationCallback,
    )

    pid = PIDLagrangianPolicy(
        kp=1.0, ki=0.0, kd=0.0, signal_target=0.15,
        beta_floor=0.02, beta_ceil=10.0, integral_clamp=10.0,
    )
    return RewardHackMitigationCallback(
        mode="pid_lagrangian",
        detector="info_rm",
        log_writer=MitigationLogWriter(str(tmp_path / "m.jsonl")),
        buffer=buffer,
        task="grpo",
        pid=pid,
        rollback=rollback,
        rollback_patience=rollback_patience,
        max_recovery_attempts=max_recovery_attempts,
        rl_checkpoint_cb=ckpt_cb,
    )


class TestPidCallback:
    """pid_lagrangian drives β via PID and runs the rollback escalation ladder."""

    def test_pid_requires_policy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        with pytest.raises(ValueError, match="pid_lagrangian"):
            RewardHackMitigationCallback(
                mode="pid_lagrangian",
                detector="info_rm",
                log_writer=MitigationLogWriter("m.jsonl"),
            )

    def test_pid_mutates_beta_on_hack(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = _pid_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK]))
        trainer = _fake_grpo_trainer(beta=0.02)
        cb.attach(trainer)
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        cb.on_step_end(None, types.SimpleNamespace(global_step=2), None)
        assert trainer.beta > 0.5  # PID raised β on the hacking step
        assert trainer.args.beta == trainer.beta  # dual write

    def test_escalation_rollback_then_stop(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt = _FakeCkptCb(saved=[10])
        cb = _pid_callback(
            tmp_path,
            _SeqBuffer([_HEALTHY, _HACK, _HACK, _HACK, _HACK]),
            rollback=True,
            rollback_patience=2,
            max_recovery_attempts=1,
            ckpt_cb=ckpt,
        )
        cb.attach(_fake_grpo_trainer(beta=0.02))
        control = types.SimpleNamespace(should_training_stop=False)
        model, opt = object(), object()
        for step in range(1, 6):
            control = cb.on_step_end(
                None,
                types.SimpleNamespace(global_step=step),
                control,
                model=model,
                optimizer=opt,
            )
        # rolled back to the last-good saved checkpoint (step 10)…
        assert ckpt.restore_calls == [10]
        # …then early-stopped after recovery attempts were exhausted.
        assert control.should_training_stop is True

    def test_no_rollback_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt = _FakeCkptCb(saved=[10])
        cb = _pid_callback(
            tmp_path,
            _SeqBuffer([_HEALTHY, _HACK, _HACK, _HACK]),
            rollback=False,
            ckpt_cb=ckpt,
        )
        cb.attach(_fake_grpo_trainer(beta=0.02))
        control = types.SimpleNamespace(should_training_stop=False)
        for step in range(1, 5):
            control = cb.on_step_end(
                None, types.SimpleNamespace(global_step=step), control,
                model=object(), optimizer=object(),
            )
        assert ckpt.restore_calls == []  # rollback disabled
        assert control.should_training_stop is False


class TestAttachPidControl:
    """attach_rl_callbacks builds the PID policy + wires the checkpoint ref."""

    def test_attach_pid_builds_policy_and_ckpt_ref(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback
        from soup_cli.utils.rl_checkpoint import RLCheckpointCallback

        tcfg = TrainingConfig(
            reward_hack_mitigation="pid_lagrangian",
            reward_hack_detector="info_rm",
            reward_hack_rollback=True,
            rl_checkpoint_save_every_steps=5,
            reward_hack_pid_kp=0.7,
        )
        added: list = []
        attach_rl_callbacks(
            _fake_trainer_recording(added),
            tcfg,
            buffer=object(),
            output_dir=str(tmp_path),
            task="grpo",
        )
        mit = [c for c in added if isinstance(c, RewardHackMitigationCallback)]
        assert len(mit) == 1 and mit[0].mode == "pid_lagrangian"
        assert mit[0].pid is not None and mit[0].pid.kp == 0.7
        assert mit[0].rollback is True
        ckpts = [c for c in added if isinstance(c, RLCheckpointCallback)]
        assert len(ckpts) == 1
        assert mit[0].rl_checkpoint_cb is ckpts[0]


# =====================================================================
# Part D / Stage 3 — schema fields (smoothing / conservative / shaping) (Task D1)
# =====================================================================


class TestStage3Schema:
    """Anti-gaming tunables: smoothing, conservative-on-disagreement, shaping."""

    def _cfg(self, extra: str = "", *, mitigation: str = "kl_control"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(_yaml("grpo", mitigation=mitigation, extra=extra))

    def test_smoothing_parses(self):
        cfg = self._cfg("reward_hack_signal_smoothing: ema\nreward_hack_smoothing_window: 16")
        assert cfg.training.reward_hack_signal_smoothing == "ema"
        assert cfg.training.reward_hack_smoothing_window == 16

    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        t = TrainingConfig()
        assert t.reward_hack_signal_smoothing == "none"
        assert t.reward_hack_smoothing_window == 8
        assert t.reward_hack_conservative_on_disagreement is False
        assert t.reward_hack_reward_shaping is False
        assert t.reward_hack_shaping_kind == "length"
        assert t.reward_hack_shaping_strength == 0.0

    def test_shaping_parses(self):
        cfg = self._cfg(
            "reward_hack_reward_shaping: true\n"
            "reward_hack_shaping_kind: repetition\n"
            "reward_hack_shaping_strength: 0.2"
        )
        assert cfg.training.reward_hack_reward_shaping is True
        assert cfg.training.reward_hack_shaping_kind == "repetition"

    def test_shaping_requires_strength(self):
        with pytest.raises(ValueError, match="shaping_strength"):
            self._cfg("reward_hack_reward_shaping: true")

    def test_shaping_requires_control_mode(self):
        # log_only is observe-only — reward shaping mutates rewards, reject.
        with pytest.raises(ValueError, match="log_only|control|kl_control"):
            self._cfg(
                "reward_hack_reward_shaping: true\nreward_hack_shaping_strength: 0.2",
                mitigation="log_only",
            )

    def test_smoothing_window_bounds(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_smoothing_window: 1")

    def test_shaping_strength_bounds(self):
        with pytest.raises(ValueError):
            self._cfg("reward_hack_shaping_strength: 2.0\nreward_hack_reward_shaping: true")

    def test_stage3_tunable_under_off_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_mitigation"):
            load_config_from_string(
                _yaml(
                    "grpo",
                    detector=None,
                    mitigation="off",
                    extra="reward_hack_signal_smoothing: ema",
                )
            )


# =====================================================================
# Part D / Stage 3 — conservative vote + distribution-drift guard (Task D2)
# =====================================================================


class TestConservativeAndDrift:
    """conservative-on-disagreement + reward-distribution-drift guard (pure)."""

    def test_conservative_agreement_uses_mean(self):
        from soup_cli.utils.reward_hack_control import combine_conservative

        assert combine_conservative([0.4, 0.42], disagree_tol=0.2) == pytest.approx(0.41)

    def test_conservative_disagreement_uses_max(self):
        from soup_cli.utils.reward_hack_control import combine_conservative

        # detectors disagree beyond tol → stay cautious (keep KL high).
        assert combine_conservative([0.1, 0.9], disagree_tol=0.2) == 0.9

    def test_conservative_empty_zero(self):
        from soup_cli.utils.reward_hack_control import combine_conservative

        assert combine_conservative([], disagree_tol=0.2) == 0.0

    def test_drift_flags_bimodal_collapse(self):
        from soup_cli.utils.reward_hack_control import (
            detect_reward_distribution_drift,
        )

        assert detect_reward_distribution_drift([0, 0, 0, 0, 1, 1, 1, 1]) is True

    def test_drift_ignores_spread(self):
        from soup_cli.utils.reward_hack_control import (
            detect_reward_distribution_drift,
        )

        assert (
            detect_reward_distribution_drift([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
            is False
        )

    def test_drift_ignores_constant(self):
        from soup_cli.utils.reward_hack_control import (
            detect_reward_distribution_drift,
        )

        assert detect_reward_distribution_drift([0.5] * 8) is False

    def test_drift_too_few(self):
        from soup_cli.utils.reward_hack_control import (
            detect_reward_distribution_drift,
        )

        assert detect_reward_distribution_drift([0.0, 1.0]) is False


# =====================================================================
# Part D / Stage 3 — reward-shaping shim (Task D3)
# =====================================================================


def _inner_reward(prompts, completions, **kwargs):
    return [1.0 for _ in completions]


class TestRewardShaping:
    """Bounded reward-shaping shim over the wrap_reward_funcs seam."""

    def test_length_penalises_long(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        shaped = shape_reward_fn(_inner_reward, kind="length", strength=0.5)
        out = shaped(["p", "p"], ["short", "w " * 40])
        assert out[1] < out[0] <= 1.0
        assert all(o >= 1.0 - 0.5 - 1e-9 for o in out)  # penalty ≤ strength

    def test_strength_zero_is_verbatim(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        shaped = shape_reward_fn(_inner_reward, kind="length", strength=0.0)
        assert shaped(["p"], ["w " * 40]) == [1.0]

    def test_inner_called_once(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        calls = []

        def counted(prompts, completions, **kwargs):
            calls.append(1)
            return [1.0 for _ in completions]

        shape_reward_fn(counted, kind="length", strength=0.5)(["p"], ["a b c"])
        assert len(calls) == 1

    def test_sentinel_penalty(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        shaped = shape_reward_fn(_inner_reward, kind="sentinel", strength=0.3)
        assert shaped(["p"], ["say GOLD"])[0] == pytest.approx(0.7)
        assert shaped(["p"], ["nope"])[0] == pytest.approx(1.0)

    def test_preserves_name(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        shaped = shape_reward_fn(_inner_reward, kind="length", strength=0.5)
        assert shaped.__name__ == "_inner_reward"

    def test_bad_kind_rejected(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        with pytest.raises(ValueError, match="kind"):
            shape_reward_fn(_inner_reward, kind="bogus", strength=0.5)

    def test_bad_strength_rejected(self):
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        with pytest.raises(ValueError, match="strength"):
            shape_reward_fn(_inner_reward, kind="length", strength=2.0)

    def test_apply_reward_shaping_from_tcfg(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.reward_hack_control import apply_reward_shaping

        tcfg = TrainingConfig(
            reward_hack_reward_shaping=True,
            reward_hack_shaping_kind="length",
            reward_hack_shaping_strength=0.5,
        )
        wrapped = apply_reward_shaping(_inner_reward, tcfg)
        assert wrapped(["p"], ["w " * 40])[0] < 1.0

    def test_apply_reward_shaping_noop_when_disabled(self):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.reward_hack_control import apply_reward_shaping

        tcfg = TrainingConfig(reward_hack_reward_shaping=False)
        assert apply_reward_shaping(_inner_reward, tcfg) is _inner_reward


# =====================================================================
# Part D / Stage 3 — give-up explainer (Task D4a)
# =====================================================================


class TestExplainGiveup:
    """Plain-English give-up explanation (mirrors why.py)."""

    def test_names_signal_and_attempts(self):
        from soup_cli.utils.reward_hack_control import ControllerState, explain_giveup

        state = ControllerState(recovery_attempts=2, last_signal=0.8)
        text = explain_giveup(
            state, signal_name="info_rm", action_history=["raise", "rollback"]
        )
        assert "info_rm" in text
        assert "2" in text
        assert "gave up" in text.lower()

    def test_handles_empty_history(self):
        from soup_cli.utils.reward_hack_control import ControllerState, explain_giveup

        text = explain_giveup(
            ControllerState(), signal_name="rm_ensemble", action_history=[]
        )
        assert isinstance(text, str) and "rm_ensemble" in text


# =====================================================================
# Part D / Stage 3 — smoothing/conservative/drift + give-up wiring (Task D4b)
# =====================================================================


class TestStage3CallbackWiring:
    """Callback honours smoothing / conservative / drift-guard + logs give-up."""

    def _cb(self, tmp_path, buffer, *, conservative=False, smoothing="none"):
        from soup_cli.utils.reward_hack_control import (
            MitigationLogWriter,
            RewardHackMitigationCallback,
        )

        return RewardHackMitigationCallback(
            mode="kl_control",
            detector="info_rm",
            log_writer=MitigationLogWriter(str(tmp_path / "m.jsonl")),
            buffer=buffer,
            task="grpo",
            bang_bang=_kl_policy(),
            conservative_on_disagreement=conservative,
            smoothing=smoothing,
            smoothing_window=4,
        )

    def test_constructs_with_stage3_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._cb(tmp_path, None, conservative=True, smoothing="ema")
        assert cb.smoothing == "ema" and cb.conservative_on_disagreement is True

    def test_drift_logged_when_conservative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bimodal = _grpo_snapshot(
            [0, 0, 0, 0, 1, 1, 1, 1], ["a", "b", "c", "d", "e", "f", "g", "h"]
        )
        cb = self._cb(tmp_path, _SeqBuffer([bimodal]), conservative=True)
        cb.attach(_fake_grpo_trainer(beta=0.02))
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        entry = json.loads((tmp_path / "m.jsonl").read_text().strip().splitlines()[-1])
        assert entry.get("drift") is True

    def test_no_drift_key_when_not_conservative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bimodal = _grpo_snapshot(
            [0, 0, 0, 0, 1, 1, 1, 1], ["a", "b", "c", "d", "e", "f", "g", "h"]
        )
        cb = self._cb(tmp_path, _SeqBuffer([bimodal]), conservative=False)
        cb.attach(_fake_grpo_trainer(beta=0.02))
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        entry = json.loads((tmp_path / "m.jsonl").read_text().strip().splitlines()[-1])
        assert "drift" not in entry  # drift guard is opt-in

    def test_giveup_explanation_logged_on_early_stop(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt = _FakeCkptCb(saved=[10])
        cb = _pid_callback(
            tmp_path,
            _SeqBuffer([_HEALTHY, _HACK, _HACK, _HACK, _HACK]),
            rollback=True,
            rollback_patience=2,
            max_recovery_attempts=1,
            ckpt_cb=ckpt,
        )
        cb.attach(_fake_grpo_trainer(beta=0.02))
        control = types.SimpleNamespace(should_training_stop=False)
        for step in range(1, 6):
            control = cb.on_step_end(
                None, types.SimpleNamespace(global_step=step), control,
                model=object(), optimizer=object(),
            )
        entries = [
            json.loads(line)
            for line in (tmp_path / "m.jsonl").read_text().strip().splitlines()
        ]
        explained = [e for e in entries if "explanation" in e]
        assert explained and "gave up" in explained[-1]["explanation"].lower()


class TestRewardShapingWiring:
    """Reward-shaping shim is applied over the reward fn in grpo/ppo."""

    def test_grpo_applies_shaping(self):
        import inspect

        from soup_cli.trainer import grpo

        assert "apply_reward_shaping" in inspect.getsource(grpo)

    def test_ppo_applies_shaping(self):
        import inspect

        from soup_cli.trainer import ppo

        assert "apply_reward_shaping" in inspect.getsource(ppo)


# =====================================================================
# Part D / Stage 3 — adversarial controller fuzz suite (Task D4d)
# =====================================================================


def _fuzz_traces():
    """Sawtooth / step / noisy / adversarial-flip / out-of-range signal traces."""
    rng = random.Random(1234)
    return [
        [(i / 10.0) % 1.0 for i in range(50)],  # sawtooth
        [0.0] * 10 + [0.9] * 10 + [0.0] * 10,  # step
        [rng.random() for _ in range(60)],  # noisy
        [0.9 if i % 2 == 0 else 0.0 for i in range(60)],  # adversarial flip
        [-1.0, 2.0, 0.5, 5.0, -3.0, 0.7],  # out-of-range (must not escape bounds)
    ]


class TestControllerFuzz:
    """No adversarial signal trace can drive the controller out of bounds,
    make it flap, or defeat anti-windup."""

    def test_bang_bang_bounded_and_no_unbounded_jump(self):
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        policy = _bang_policy()
        for trace in _fuzz_traces():
            state = ControllerState(beta=policy.beta_floor)
            prev = state.beta
            for vote in trace:
                state, action = bang_bang_step(policy, state, vote=vote)
                assert policy.beta_floor - 1e-9 <= state.beta <= policy.beta_ceil + 1e-9
                assert state.beta > 0.0 and math.isfinite(state.beta)
                assert state.beta / prev <= policy.kl_gain + 1e-9  # geometric only
                # field-validity invariants (tdd review LOW #8)
                assert 0.0 <= state.last_signal <= 1.0
                assert state.dwell_count >= 0 and state.release_count >= 0
                assert isinstance(state.tripped, bool)
                assert action.new_beta == state.beta
                prev = state.beta

    def test_bang_bang_converges_on_sustained_signal(self):
        # convergence property: sustained above-band input eventually trips.
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        policy = _bang_policy(dwell_steps=3)
        state = ControllerState(beta=0.02)
        for _ in range(5):
            state, _ = bang_bang_step(policy, state, vote=0.9)
        assert state.tripped and state.beta > 0.02

    def test_bang_bang_no_flap_on_alternation(self):
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        policy = _bang_policy(dwell_steps=2, release_patience=2)
        state = ControllerState(beta=policy.beta_floor)
        for i in range(40):
            state, _ = bang_bang_step(policy, state, vote=0.9 if i % 2 == 0 else 0.0)
        assert state.beta == pytest.approx(policy.beta_floor) and not state.tripped

    def test_pid_bounded_and_anti_windup_holds(self):
        from soup_cli.utils.reward_hack_control import ControllerState, pid_step

        policy = _pid_policy(ki=1.0, integral_clamp=0.5, beta_ceil=5.0)
        for trace in _fuzz_traces():
            state = ControllerState(beta=policy.beta_floor)
            for signal in trace:
                state, _ = pid_step(policy, state, signal=signal)
                assert policy.beta_floor - 1e-9 <= state.beta <= policy.beta_ceil + 1e-9
                assert state.beta > 0.0 and math.isfinite(state.beta)
                assert abs(state.integral) <= policy.integral_clamp + 1e-9
                assert math.isfinite(state.integral) and math.isfinite(state.prev_error)
                assert 0.0 <= state.last_signal <= 1.0


# =====================================================================
# python-review fixes (v0.71.26)
# =====================================================================


class TestReviewFixesPython:
    """Regression tests for the python-review findings."""

    def _cfg(self, extra, *, mitigation="kl_control", detector="info_rm", task="grpo"):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(
            _yaml(task, mitigation=mitigation, detector=detector, extra=extra)
        )

    def test_signals_must_include_active_detector(self):
        # CRITICAL #1 — a signal set that omits the active detector silently
        # drops the primary signal from the vote; reject it.
        with pytest.raises(ValueError, match="detector"):
            self._cfg("reward_hack_signals: [length_trend]")

    def test_signals_reject_inactive_detector_name(self):
        # detector=info_rm but signals lists rm_ensemble (never produced) → reject.
        with pytest.raises(ValueError, match="rm_ensemble|active detector"):
            self._cfg("reward_hack_signals: [info_rm, rm_ensemble]")

    def test_integral_clamp_is_a_field(self):
        # CRITICAL #2 — integral_clamp must be its own tunable, not beta_ceil.
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().reward_hack_integral_clamp == 1.0
        cfg = self._cfg(
            "reward_hack_integral_clamp: 5.0", mitigation="pid_lagrangian"
        )
        assert cfg.training.reward_hack_integral_clamp == 5.0

    def test_integral_clamp_wired_into_pid_policy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.peft_wiring import attach_rl_callbacks
        from soup_cli.utils.reward_hack_control import RewardHackMitigationCallback

        tcfg = TrainingConfig(
            reward_hack_mitigation="pid_lagrangian",
            reward_hack_detector="info_rm",
            reward_hack_integral_clamp=7.0,
            reward_hack_beta_ceil=3.0,
        )
        added: list = []
        attach_rl_callbacks(
            _fake_trainer_recording(added),
            tcfg,
            buffer=object(),
            output_dir=str(tmp_path),
            task="grpo",
        )
        mit = [c for c in added if isinstance(c, RewardHackMitigationCallback)][0]
        assert mit.pid.integral_clamp == 7.0  # not beta_ceil (3.0)

    def test_task_gate_before_controller_checks(self):
        # HIGH #3 — a bad beta bound on an sft task should surface the task
        # error, not the numeric one.
        with pytest.raises(ValueError, match="grpo|ppo|task"):
            self._cfg(
                "reward_hack_beta_floor: 1.0\nreward_hack_beta_ceil: 0.5",
                task="sft",
            )

    def test_shape_reward_fn_rejects_non_callable(self):
        # MEDIUM #10 — non-callable inner must fail fast, not at train time.
        from soup_cli.utils.reward_hack_control import shape_reward_fn

        with pytest.raises((TypeError, ValueError), match="callable"):
            shape_reward_fn("not a fn", kind="length", strength=0.5)


class TestReviewFixesCode:
    """Regression tests for the code-review findings."""

    def test_prune_trims_saved_list(self, tmp_path, monkeypatch):
        # HIGH #1 — _prune must keep _saved in sync with surviving dirs so the
        # rollback target can never point at a deleted checkpoint.
        import os

        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        cb = build_rl_checkpoint_callback(
            RLCheckpointConfig(save_every_steps=1, keep_last=2),
            output_dir="run",
            task="grpo",
        )
        for step in (1, 2, 3, 4):
            cb.save_checkpoint(step=step, model=_FakeSavableModel(), optimizer=None)
        root = os.path.join("run", "rl-checkpoints")
        on_disk = sorted(
            int(d.split("-")[1]) for d in os.listdir(root) if d.startswith("step-")
        )
        assert on_disk == [3, 4]
        assert sorted(cb._saved) == on_disk  # in-memory list matches disk

    def test_bang_release_requires_patience_per_relaxation(self):
        # HIGH #4 — each geometric relaxation must re-accumulate release_patience.
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        policy = _bang_policy(dwell_steps=1, release_patience=2, kl_gain=1.5, beta_ceil=1.0)
        state = ControllerState(beta=0.02)
        for _ in range(3):  # raise β three times → ~0.0675
            state, _ = bang_bang_step(policy, state, vote=0.9)
        high = state.beta
        state, _ = bang_bang_step(policy, state, vote=0.0)  # release=1, no relax
        assert state.beta == pytest.approx(high)
        state, _ = bang_bang_step(policy, state, vote=0.0)  # release=2 → relax
        after_first = state.beta
        assert after_first < high
        state, _ = bang_bang_step(policy, state, vote=0.0)  # release reset → 1, no relax
        assert state.beta == pytest.approx(after_first)
        state, _ = bang_bang_step(policy, state, vote=0.0)  # release=2 → relax again
        assert state.beta < after_first

    def test_rollback_requires_nonzero_recovery_attempts(self):
        # MEDIUM #5 — rollback=True with max_recovery_attempts=0 is a footgun.
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="max_recovery_attempts"):
            load_config_from_string(
                _yaml(
                    "grpo",
                    mitigation="pid_lagrangian",
                    extra=(
                        "reward_hack_rollback: true\n"
                        "rl_checkpoint_save_every_steps: 2\n"
                        "reward_hack_max_recovery_attempts: 0"
                    ),
                )
            )

    def test_escalate_no_target_does_not_waste_attempt(self, tmp_path, monkeypatch):
        # MEDIUM #6 — a rollback with no last-good checkpoint must not burn a
        # recovery attempt nor early-stop.
        monkeypatch.chdir(tmp_path)
        ckpt = _FakeCkptCb(saved=[])  # no checkpoints saved yet
        cb = _pid_callback(
            tmp_path,
            _SeqBuffer([_HEALTHY, _HACK, _HACK, _HACK, _HACK]),
            rollback=True,
            rollback_patience=2,
            max_recovery_attempts=1,
            ckpt_cb=ckpt,
        )
        cb.attach(_fake_grpo_trainer(beta=0.02))
        control = types.SimpleNamespace(should_training_stop=False)
        for step in range(1, 6):
            control = cb.on_step_end(
                None, types.SimpleNamespace(global_step=step), control,
                model=object(), optimizer=object(),
            )
        assert ckpt.restore_calls == []
        assert cb._state.recovery_attempts == 0  # not wasted on a None target

    def test_action_history_is_bounded(self, tmp_path, monkeypatch):
        # LOW #9 — _action_history must not grow unbounded.
        monkeypatch.chdir(tmp_path)
        from itertools import repeat

        cb = _kl_callback(tmp_path, _SeqBuffer(list(repeat(_HACK, 1))))
        cb.attach(_fake_grpo_trainer(beta=0.02))
        for step in range(1, 60):
            cb.on_step_end(None, types.SimpleNamespace(global_step=step), None)
        assert len(cb._action_history) <= 1000


class TestReviewFixesSecurity:
    """Regression tests for the security-review findings."""

    @pytest.mark.skipif(os.name == "nt", reason="symlink needs privilege on Windows")
    def test_restore_refuses_symlink_optimizer(self, tmp_path, monkeypatch):
        # HIGH #1 — torch.load(weights_only=False) on an attacker-symlinked
        # optimizer.pt is RCE; restore must refuse a symlinked file.
        import torch

        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        cb = build_rl_checkpoint_callback(
            RLCheckpointConfig(save_every_steps=1), output_dir="run", task="grpo"
        )
        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=0.1)
        cb.save_checkpoint(step=1, model=_FakeSavableModel(), optimizer=opt)
        opt_path = os.path.join("run", "rl-checkpoints", "step-1", "optimizer.pt")
        evil = tmp_path / "evil.pt"
        evil.write_bytes(b"junk")
        os.remove(opt_path)
        os.symlink(str(evil), opt_path)
        # must refuse the symlinked optimizer (return False, never torch.load it)
        assert cb.restore_checkpoint(step=1, model=None, optimizer=opt) is False

    def test_bool_rejected_on_int_fields(self):
        # MEDIUM #2 — bool-before-int policy on the new integer fields.
        from soup_cli.config.schema import TrainingConfig

        for field in (
            "reward_hack_dwell_steps",
            "reward_hack_release_patience",
            "reward_hack_rollback_patience",
            "reward_hack_max_recovery_attempts",
            "reward_hack_smoothing_window",
        ):
            with pytest.raises((ValueError, TypeError), match="bool"):
                TrainingConfig(**{field: True})

    def test_bool_rejected_on_float_fields(self):
        from soup_cli.config.schema import TrainingConfig

        # pid_kp has ge=0 so True→1.0 would pass the bound without a bool guard.
        with pytest.raises((ValueError, TypeError), match="bool"):
            TrainingConfig(reward_hack_pid_kp=True)

    def test_signals_length_capped(self):
        # MEDIUM #3 — unbounded signals list is a per-step DoS.
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises((ValueError, TypeError)):
            TrainingConfig(reward_hack_signals=["info_rm"] * 100)

    def test_callback_rejects_empty_signals(self, tmp_path, monkeypatch):
        # LOW #4 — an empty signals tuple silently disables the controller.
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
                signals=(),
            )


class TestReviewFixesTdd:
    """Coverage gaps identified by the tdd review."""

    def test_bang_bang_deadband_hold_while_tripped(self):
        # GAP 1 (HIGH) — the dead-band 'hold' branch while tripped must keep β
        # and reset both counters, without relaxing.
        from soup_cli.utils.reward_hack_control import ControllerState, bang_bang_step

        policy = _bang_policy(dwell_steps=2, release_patience=2, trip_band=0.3, release_band=0.1)
        state = ControllerState(beta=0.02)
        state, _ = bang_bang_step(policy, state, vote=0.5)
        state, _ = bang_bang_step(policy, state, vote=0.5)  # trip → β=0.03
        assert state.tripped and state.beta == pytest.approx(0.03)
        state, action = bang_bang_step(policy, state, vote=0.2)  # dead-band
        assert state.beta == pytest.approx(0.03) and state.tripped
        assert state.release_count == 0 and state.dwell_count == 0
        assert action.reason == "hold"

    def test_shape_reward_verbatim_on_shim_error(self, monkeypatch):
        # GAP 2 (HIGH) — a shim error must return the verbatim inner reward.
        import soup_cli.utils.reward_hack_control as rhc

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(rhc, "_shaping_penalty", boom)
        shaped = rhc.shape_reward_fn(_inner_reward, kind="length", strength=0.5)
        assert shaped(["p"], ["w " * 40]) == [1.0]  # verbatim despite shim error

    def test_conservative_disagreement_boundary(self):
        # GAP 3 (MEDIUM) — max-min == tol is NOT > tol → mean (not max).
        from soup_cli.utils.reward_hack_control import combine_conservative

        assert combine_conservative([0.1, 0.3], disagree_tol=0.2) == pytest.approx(0.2)
        assert combine_conservative([0.1, 0.301], disagree_tol=0.2) == pytest.approx(0.301)

    def test_dual_write_survives_readonly_beta(self, tmp_path, monkeypatch):
        # GAP 4 (MEDIUM) — a read-only trainer.beta must not block args.beta.
        monkeypatch.chdir(tmp_path)

        class _ROTrainer:
            def __init__(self):
                self.args = types.SimpleNamespace(beta=0.02)

            @property
            def beta(self):
                return 0.02  # read-only property

        cb = _kl_callback(tmp_path, _SeqBuffer([_HEALTHY, _HACK]))
        trainer = _ROTrainer()
        cb.attach(trainer)
        cb.on_step_end(None, types.SimpleNamespace(global_step=1), None)
        cb.on_step_end(None, types.SimpleNamespace(global_step=2), None)
        assert trainer.args.beta == pytest.approx(0.04)  # args.beta still updated

    def test_escalation_postconditions(self, tmp_path, monkeypatch):
        # GAP 5 (MEDIUM) — recovery_attempts increments to 1, hack_streak resets.
        monkeypatch.chdir(tmp_path)
        ckpt = _FakeCkptCb(saved=[10])
        cb = _pid_callback(
            tmp_path,
            _SeqBuffer([_HEALTHY, _HACK, _HACK]),
            rollback=True,
            rollback_patience=2,
            max_recovery_attempts=2,
            ckpt_cb=ckpt,
        )
        cb.attach(_fake_grpo_trainer(beta=0.02))
        control = types.SimpleNamespace(should_training_stop=False)
        for step in (1, 2, 3):
            control = cb.on_step_end(
                None, types.SimpleNamespace(global_step=step), control,
                model=object(), optimizer=object(),
            )
        assert ckpt.restore_calls == [10]
        assert cb._state.recovery_attempts == 1
        assert cb._hack_streak == 0
        assert control.should_training_stop is False  # max=2 → not stopped yet

    def test_pid_derivative_exact_spike(self):
        # GAP 2b — pin the exact D-term magnitude, not just > floor.
        states = _run_pid(
            _pid_policy(kp=0.0, ki=0.0, kd=1.0), [0.65]
        )
        # error=0.65-0.15=0.5, prev_error=0 → deriv=0.5 → β=floor+0.5=0.52
        assert states[0].beta == pytest.approx(0.52)
        assert states[0].prev_error == pytest.approx(0.5)

    def test_drift_negative_gap(self):
        from soup_cli.utils.reward_hack_control import detect_reward_distribution_drift

        # gap <= 0 (sorted halves can't reverse, but constant-ish → gap 0) → False
        assert detect_reward_distribution_drift([1, 1, 1, 1, 1, 1]) is False

    def test_smoothing_window_lower_boundary_ok(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            _yaml("grpo", mitigation="kl_control", extra="reward_hack_smoothing_window: 2")
        )
        assert cfg.training.reward_hack_smoothing_window == 2

    def test_log_cap_boundaries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        assert MitigationLogWriter("a.jsonl", cap_mb=1).cap_bytes == 1024 * 1024
        assert MitigationLogWriter("b.jsonl", cap_mb=10_000).cap_bytes == 10_000 * 1024 * 1024
        with pytest.raises(ValueError):
            MitigationLogWriter("c.jsonl", cap_mb=10_001)

    def test_log_concurrent_writes(self, tmp_path, monkeypatch):
        import threading

        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import MitigationLogWriter

        writer = MitigationLogWriter("cc.jsonl")

        def worker(base):
            for i in range(50):
                writer.record(step=base + i, snapshot={"x": i})

        threads = [threading.Thread(target=worker, args=(b,)) for b in (0, 1000, 2000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = (tmp_path / "cc.jsonl").read_text().strip().splitlines()
        assert len(lines) == 150  # no interleaved/corrupt lines
        for line in lines:
            json.loads(line)  # every line is a complete JSON object

    def test_record_action_caps_at_max(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.reward_hack_control import _MAX_ACTION_HISTORY

        cb = _kl_callback(tmp_path, None)
        for i in range(_MAX_ACTION_HISTORY + 100):
            cb._record_action(f"a{i}")
        assert len(cb._action_history) == _MAX_ACTION_HISTORY
        assert cb._action_history[-1] == f"a{_MAX_ACTION_HISTORY + 99}"  # keeps the tail

    def test_no_top_level_heavy_import_in_source(self):
        import inspect

        from soup_cli.utils import reward_hack_control

        src = inspect.getsource(reward_hack_control)
        for line in src.splitlines():
            stripped = line.strip()
            # module-scope imports have no indentation
            if line and not line[0].isspace():
                assert not stripped.startswith(("import torch", "from torch")), line
                assert not stripped.startswith(
                    ("import transformers", "from transformers")
                ), line
