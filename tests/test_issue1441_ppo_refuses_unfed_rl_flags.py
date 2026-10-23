"""#1441 — `task: ppo` must refuse the RL-signal flags it cannot feed.

The trl PPO trainer this build supports (`trl.experimental.ppo.PPOTrainer`)
takes no `reward_funcs`: its reward comes from the reward model it calls
internally. So the callable wrappers built for the shared signal buffer were
dropped, the buffer was never written, and — because the buffer object existed —
the detector's `on_log` fallback returned early too. A run carrying
`reward_hack_detector`, `reward_hack_mitigation` or `echo_trap_enabled` was
loaded, announced, and inert: never halted, never mitigated, never trapped.

These drive the refusal the trainer calls before loading anything, so they need
no model, no trl import and no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.trainer.ppo import refuse_unfed_rl_flags


def _tcfg(**overrides):
    """A `training` shape with the three flags off, as a real config has them."""
    base = {
        "reward_hack_detector": None,
        "reward_hack_halt": False,
        "reward_hack_mitigation": "off",
        "echo_trap_enabled": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTheRefusal:
    def test_no_flags_means_no_refusal(self):
        # The control: a plain PPO run must not be refused.
        refuse_unfed_rl_flags(_tcfg(), is_experimental=True)

    def test_the_detector_is_refused_and_named(self):
        with pytest.raises(ValueError) as excinfo:
            refuse_unfed_rl_flags(
                _tcfg(reward_hack_detector="info_rm"), is_experimental=True,
            )
        message = str(excinfo.value)
        assert "reward_hack_detector" in message, message
        assert "task: ppo" in message, message

    def test_the_mitigation_mode_is_refused_and_named(self):
        with pytest.raises(ValueError) as excinfo:
            refuse_unfed_rl_flags(
                _tcfg(reward_hack_mitigation="kl_control"), is_experimental=True,
            )
        assert "reward_hack_mitigation" in str(excinfo.value)

    def test_the_echo_trap_is_refused_and_named(self):
        with pytest.raises(ValueError) as excinfo:
            refuse_unfed_rl_flags(
                _tcfg(echo_trap_enabled=True), is_experimental=True,
            )
        assert "echo_trap_enabled" in str(excinfo.value)

    def test_every_set_flag_is_named_not_just_the_first(self):
        with pytest.raises(ValueError) as excinfo:
            refuse_unfed_rl_flags(
                _tcfg(
                    reward_hack_detector="info_rm",
                    reward_hack_mitigation="kl_control",
                    echo_trap_enabled=True,
                ),
                is_experimental=True,
            )
        message = str(excinfo.value)
        for flag in ("reward_hack_detector", "reward_hack_mitigation", "echo_trap_enabled"):
            assert flag in message, message

    def test_the_halt_switch_alone_is_not_a_refusal(self):
        """`reward_hack_halt` only modifies the detector; with no detector
        there is nothing to feed, so it must not refuse on its own."""
        refuse_unfed_rl_flags(_tcfg(reward_hack_halt=True), is_experimental=True)

    def test_the_message_says_the_flags_are_grpo_only(self):
        """A user who wants this feature needs to be told where it works."""
        with pytest.raises(ValueError) as excinfo:
            refuse_unfed_rl_flags(
                _tcfg(reward_hack_detector="info_rm"), is_experimental=True,
            )
        assert "grpo" in str(excinfo.value).lower()

    def test_a_non_experimental_trl_is_not_refused(self):
        """The transitional API does accept `reward_funcs`, so the refusal is
        scoped to the experimental class rather than to `task: ppo` itself."""
        refuse_unfed_rl_flags(
            _tcfg(reward_hack_detector="info_rm"), is_experimental=False,
        )


class TestTheFlagScan:
    def test_a_default_config_scans_empty(self):
        from soup_cli.trainer.ppo import _unsupported_rl_flags

        assert _unsupported_rl_flags(_tcfg()) == []

    def test_a_missing_attribute_is_treated_as_off(self):
        """A config object without the fields at all must not raise."""
        from soup_cli.trainer.ppo import _unsupported_rl_flags

        assert _unsupported_rl_flags(SimpleNamespace()) == []

    def test_a_real_config_with_the_flags_off_scans_empty(self):
        """Against the real schema, not a stand-in."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.ppo import _unsupported_rl_flags

        cfg = load_config_from_string(
            "base: ./tiny\ntask: ppo\ndata: {train: ./d.jsonl, format: plaintext}\n"
            "training: {epochs: 1, batch_size: 1, gradient_accumulation_steps: 1}\n"
            "output: ./out\n"
        )
        assert _unsupported_rl_flags(cfg.training) == []
        refuse_unfed_rl_flags(cfg.training, is_experimental=True)

    def test_a_real_config_that_sets_the_detector_is_refused(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: ./tiny\ntask: ppo\ndata: {train: ./d.jsonl, format: plaintext}\n"
            "training:\n  epochs: 1\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
            "  reward_hack_detector: info_rm\n"
            "output: ./out\n"
        )
        with pytest.raises(ValueError, match="reward_hack_detector"):
            refuse_unfed_rl_flags(cfg.training, is_experimental=True)


class TestSetupRefusesBeforeLoadingAnything:
    """The call site, not just the helper (#1441 review).

    Every other test here calls `refuse_unfed_rl_flags` directly, so replacing
    the one line in `setup()` with `pass` left the file green. This drives
    `PPOTrainerWrapper.setup` with the loaders stubbed, and asserts nothing was
    loaded: the refusal has to land before the model, the reward model or the
    tokenizer.
    """

    _FLAGS = {
        "reward_hack_detector": "  reward_hack_detector: info_rm\n",
        "reward_hack_mitigation": (
            "  reward_hack_detector: info_rm\n  reward_hack_mitigation: kl_control\n"
        ),
        "echo_trap_enabled": "  echo_trap_enabled: true\n",
    }

    @pytest.mark.parametrize("flag", sorted(_FLAGS))
    def test_setup_refuses_before_loading_anything(self, flag, tmp_path, monkeypatch):
        import os

        # `peft` and `trl` are in this list on purpose: this environment's
        # transformers/torchvision mismatch breaks `trl.PPOConfig` and
        # `peft`, so without them the test ERRORS instead of skipping. On a
        # working stack all five import and the test runs (~23 s).
        for module in ("torch", "transformers", "peft", "trl", "datasets"):
            pytest.importorskip(module)
        os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer import ppo

        loaded: list[str] = []
        monkeypatch.setattr(
            ppo.PPOTrainerWrapper, "_setup_reward",
            lambda self, *a: loaded.append("reward"),
        )
        monkeypatch.setattr(
            ppo.PPOTrainerWrapper, "_setup_transformers",
            lambda self, *a: loaded.append("policy"),
        )
        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            "base: ./tiny\ntask: ppo\nbackend: transformers\n"
            "data:\n  train: train.jsonl\n  max_length: 64\n"
            "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
            "  quantization: none\n  reward_model: ./rm\n"
            "  lora:\n    r: 4\n    alpha: 8\n"
            + self._FLAGS[flag]
            + "output: ./out\n"
        )
        wrapper = ppo.PPOTrainerWrapper(cfg, device="cpu")
        rows = [{"messages": [{"role": "user", "content": "hi"}]}] * 4
        with pytest.raises(ValueError, match=flag):
            wrapper.setup({"train": rows})
        assert loaded == [], loaded

    def test_a_plain_ppo_setup_is_not_refused_by_the_call_site(
        self, tmp_path, monkeypatch,
    ):
        """The control: the same path with no flags must get past the refusal.

        It still fails later in this environment (no model on disk), so this
        asserts only that the refusal is not what stops it.
        """
        for module in ("torch", "transformers", "peft", "trl", "datasets"):
            pytest.importorskip(module)

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer import ppo

        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            "base: ./tiny\ntask: ppo\nbackend: transformers\n"
            "data:\n  train: train.jsonl\n  max_length: 64\n"
            "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
            "  quantization: none\n  reward_model: ./rm\n"
            "output: ./out\n"
        )
        wrapper = ppo.PPOTrainerWrapper(cfg, device="cpu")
        rows = [{"messages": [{"role": "user", "content": "hi"}]}] * 4
        try:
            wrapper.setup({"train": rows})
        except Exception as exc:  # noqa: BLE001 — anything but the refusal
            assert "cannot use" not in str(exc), exc


class TestTheRefusalCoversExactlyTheFlagsThatMakeABuffer:
    """`_unsupported_rl_flags` re-states `rl_callbacks_need_buffer`; if a fourth
    flag ever starts a buffer, the refusal must not silently miss it (#1441)."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"reward_hack_detector": "info_rm"},
            {"reward_hack_mitigation": "log_only"},
            {"echo_trap_enabled": True},
            {"reward_hack_halt": True},
        ],
    )
    def test_the_refusal_and_the_buffer_predicate_agree(self, overrides):
        from soup_cli.trainer.ppo import _unsupported_rl_flags
        from soup_cli.utils.peft_wiring import rl_callbacks_need_buffer

        tcfg = _tcfg(**overrides)
        assert bool(_unsupported_rl_flags(tcfg)) == rl_callbacks_need_buffer(tcfg)
