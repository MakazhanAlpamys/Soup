"""v0.71.11 — GRPO / RL callbacks live wiring.

Closes #235 (reward-hack), #236 (ULD), #237 (MiniLLM), #238 (RL checkpoint),
#239 (iterative-DPO), #240 (echo-trap), #159 (variant fallback warning),
#160 (in-place GRPO EMA).

These tests lift the v0.70.0 deferred-stub family to live behaviour and
exercise the math + wiring on CPU / tiny fakes (no GPU). The Step-6 smoke
runs a real SmolLM2-135M GRPO loop separately.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import soup_cli

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


class _FakeState:
    def __init__(self, global_step: int = 1):
        self.global_step = global_step
        self.log_history: list[dict] = []


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False


# --------------------------------------------------------------------------
# Shared RL signal buffer
# --------------------------------------------------------------------------


class TestRLSignalBuffer:
    def test_record_and_snapshot(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        buf.record(
            func_name="reward", completions=["a", "b", "c", "d"], rewards=[1, 2, 3, 4]
        )
        snap = buf.snapshot()
        assert snap["completions"] == ["a", "b", "c", "d"]
        assert snap["rewards"] == [1.0, 2.0, 3.0, 4.0]
        assert "reward" in snap["per_func"]

    def test_aggregate_sums_across_funcs(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        buf.record(func_name="a", completions=["x", "y"], rewards=[1, 2])
        buf.record(func_name="b", completions=["x", "y"], rewards=[10, 20])
        snap = buf.snapshot()
        assert snap["rewards"] == [11.0, 22.0]
        assert set(snap["per_func"]) == {"a", "b"}

    def test_non_finite_reward_dropped(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        buf.record(func_name="r", completions=["a"], rewards=[float("nan")])
        snap = buf.snapshot()
        # NaN coerced to None → aggregate position is 0.0 (no finite value).
        assert snap["per_func"]["r"] == [None]

    def test_conversational_completion_extracted(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        buf.record(
            func_name="r",
            completions=[[{"role": "assistant", "content": "hello world"}]],
            rewards=[1.0],
        )
        snap = buf.snapshot()
        assert snap["completions"] == ["hello world"]

    def test_wrap_preserves_name_and_captures(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, wrap_reward_funcs

        buf = RLSignalBuffer()

        def my_reward(prompts=None, completions=None, **kwargs):
            return [float(len(c)) for c in completions]

        wrapped = wrap_reward_funcs(my_reward, buf)
        assert wrapped.__name__ == "my_reward"
        result = wrapped(prompts=["p"], completions=["aa", "bbb"])
        assert result == [2.0, 3.0]
        snap = buf.snapshot()
        assert snap["completions"] == ["aa", "bbb"]
        assert snap["rewards"] == [2.0, 3.0]

    def test_wrap_list_shape_preserved(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, wrap_reward_funcs

        buf = RLSignalBuffer()
        fns = [lambda completions=None, **k: [1.0]]
        wrapped = wrap_reward_funcs(fns, buf)
        assert isinstance(wrapped, list)
        assert len(wrapped) == 1

    def test_capture_never_breaks_reward(self):
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, wrap_reward_funcs

        buf = RLSignalBuffer()

        def reward(prompts=None, completions=None, **kwargs):
            return [1.0, 2.0]

        wrapped = wrap_reward_funcs(reward, buf)
        # Bizarre completions that the normaliser can't handle must not raise.
        out = wrapped(prompts=None, completions=object())
        assert out == [1.0, 2.0]


# --------------------------------------------------------------------------
# #235 — reward-hack callback
# --------------------------------------------------------------------------


class TestRewardHackCallback:
    def test_build_returns_callback_not_notimplemented(self):
        from soup_cli.utils.reward_hacking import (
            RewardHackCallback,
            build_reward_hack_callback,
        )

        cb = build_reward_hack_callback(detector="info_rm")
        assert isinstance(cb, RewardHackCallback)

    def test_build_rejects_unknown_detector(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        with pytest.raises(ValueError, match="not supported"):
            build_reward_hack_callback(detector="evil")

    def test_build_rejects_non_bool_halt(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        with pytest.raises(TypeError, match="halt_on_hack"):
            build_reward_hack_callback(detector="info_rm", halt_on_hack="yes")

    def test_info_rm_compute_signal(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        cb = build_reward_hack_callback(detector="info_rm")
        snap = {"rewards": [0.0, 0.0, 5.0, 5.0], "per_func": {}}
        sig = cb.compute_signal(snap)
        assert sig is not None and sig > 0.0

    def test_info_rm_insufficient_data_returns_none(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        cb = build_reward_hack_callback(detector="info_rm")
        assert cb.compute_signal({"rewards": [1.0, 2.0], "per_func": {}}) is None

    def test_observe_baseline_then_drop_to_hack(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        cb = build_reward_hack_callback(detector="info_rm")
        r0 = cb.observe_signal(10.0, step=1)  # baseline separation 10
        assert r0.verdict == "OK"
        r1 = cb.observe_signal(2.0, step=2)  # dropped 80% → HACK
        assert r1.verdict == "HACK"
        assert r1.signal == 2.0

    def test_on_step_end_halts_on_hack(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        cb = build_reward_hack_callback(
            detector="info_rm", halt_on_hack=True, buffer=buf
        )
        state, control = _FakeState(1), _FakeControl()
        # Step 1 — high separation = baseline.
        buf.record(func_name="r", completions=["a"] * 4, rewards=[0, 0, 9, 9])
        cb.on_step_end(None, state, control)
        # Step 2 — bunched rewards = HACK.
        buf.record(func_name="r", completions=["a"] * 4, rewards=[5, 5, 5, 5])
        state.global_step = 2
        cb.on_step_end(None, state, control)
        assert control.should_training_stop is True
        assert any("reward_hack_verdict" in e for e in state.log_history)

    def test_rm_ensemble_needs_two_funcs(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        cb = build_reward_hack_callback(detector="rm_ensemble")
        # One func → None.
        assert cb.compute_signal({"rewards": [], "per_func": {"a": [1.0, 2.0]}}) is None
        # Two funcs → divergence.
        sig = cb.compute_signal(
            {"rewards": [], "per_func": {"a": [1.0, 2.0], "b": [3.0, 0.0]}}
        )
        assert sig is not None and sig >= 0.0

    def test_on_log_fallback_without_buffer(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        cb = build_reward_hack_callback(detector="info_rm", buffer=None)
        state, control = _FakeState(1), _FakeControl()
        cb.on_log(None, state, control, logs={"reward": 5.0, "reward_std": 0.1})
        assert cb.last_report() is not None

    def test_compute_separation_from_stats(self):
        from soup_cli.utils.reward_hacking import compute_separation_from_stats

        high = compute_separation_from_stats(5.0, 0.1)
        low = compute_separation_from_stats(5.0, 10.0)
        assert high > low

    def test_separation_stats_rejects_bool(self):
        from soup_cli.utils.reward_hacking import compute_separation_from_stats

        with pytest.raises(ValueError, match="bool"):
            compute_separation_from_stats(True, 1.0)


# --------------------------------------------------------------------------
# #240 — echo-trap callback
# --------------------------------------------------------------------------


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # deterministic id per whitespace token
        return [abs(hash(t)) % 1000 for t in text.split()]


class TestEchoTrapCallback:
    def test_build_returns_callback_not_notimplemented(self):
        from soup_cli.utils.echo_trap import EchoTrapCallback, build_echo_trap_callback

        cb = build_echo_trap_callback(threshold=0.5)
        assert isinstance(cb, EchoTrapCallback)

    def test_build_rejects_bad_threshold(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(ValueError):
            build_echo_trap_callback(threshold=2.0)

    def test_build_rejects_bool_threshold(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(ValueError):
            build_echo_trap_callback(threshold=True)

    def test_build_rejects_non_bool_halt(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(TypeError):
            build_echo_trap_callback(threshold=0.5, halt_on_trap="yes")

    def test_compute_signal_repetitive_high(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        cb = build_echo_trap_callback(threshold=0.5)
        # "a a a a a" — every 2-gram repeats.
        snap = {"completions": ["a a a a a", "b b b b b"]}
        sig = cb.compute_signal(snap)
        assert sig is not None and sig > 0.5

    def test_compute_signal_no_completions(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        cb = build_echo_trap_callback(threshold=0.5)
        assert cb.compute_signal({"completions": []}) is None

    def test_tokenizer_aware_path(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        cb = build_echo_trap_callback(
            threshold=0.5, tokenizer_aware=True, tokenizer=_FakeTokenizer()
        )
        sig = cb.compute_signal({"completions": ["x x x x", "y y y y"]})
        assert sig is not None and sig >= 0.0

    def test_on_step_end_halts_on_trap(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback
        from soup_cli.utils.rl_signal_buffer import RLSignalBuffer

        buf = RLSignalBuffer()
        cb = build_echo_trap_callback(threshold=0.3, halt_on_trap=True, buffer=buf)
        buf.record(
            func_name="r",
            completions=["a a a a a a", "b b b b b b"],
            rewards=[1.0, 1.0],
        )
        state, control = _FakeState(1), _FakeControl()
        cb.on_step_end(None, state, control)
        assert control.should_training_stop is True
        assert any("echo_trap_verdict" in e for e in state.log_history)

    def test_observe_classifies_ok(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        cb = build_echo_trap_callback(threshold=0.5)
        report = cb.observe_signal(0.0, step=1, n_trajectories=3)
        assert report.verdict == "OK"
        assert report.trajectories_seen == 3


# --------------------------------------------------------------------------
# #238 — RL checkpoint callback
# --------------------------------------------------------------------------


class _FakeSavableModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        import os

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"\x00")
        self.saved_to = path


class TestRLCheckpointCallback:
    def test_build_requires_output_dir(self):
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        cfg = RLCheckpointConfig(save_every_steps=1)
        with pytest.raises(ValueError, match="output_dir"):
            build_rl_checkpoint_callback(cfg)

    def test_build_rejects_non_config(self):
        from soup_cli.utils.rl_checkpoint import build_rl_checkpoint_callback

        with pytest.raises(TypeError):
            build_rl_checkpoint_callback({"save_every_steps": 1}, output_dir="x")

    def test_save_checkpoint_writes_manifest(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        monkeypatch.chdir(tmp_path)
        cfg = RLCheckpointConfig(save_every_steps=1)
        cb = build_rl_checkpoint_callback(cfg, output_dir="run", task="grpo")
        model = _FakeSavableModel()
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
        ckpt = cb.save_checkpoint(step=2, model=model, optimizer=opt)
        manifest = Path(ckpt) / "manifest.json"
        assert manifest.is_file()
        data = json.loads(manifest.read_text())
        assert data["step"] == 2 and data["task"] == "grpo"
        assert data["has_optimizer"] is True
        assert (Path(ckpt) / "optimizer.pt").is_file()

    def test_prune_keeps_keep_last(self, tmp_path, monkeypatch):
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        monkeypatch.chdir(tmp_path)
        cfg = RLCheckpointConfig(save_every_steps=1, keep_last=2)
        cb = build_rl_checkpoint_callback(cfg, output_dir="run", task="grpo")
        for step in (1, 2, 3):
            cb.save_checkpoint(step=step, model=_FakeSavableModel(), optimizer=None)
        root = tmp_path / "run" / "rl-checkpoints"
        dirs = sorted(p.name for p in root.iterdir())
        assert dirs == ["step-2", "step-3"]  # step-1 pruned

    def test_on_step_end_cadence(self, tmp_path, monkeypatch):
        from soup_cli.utils.rl_checkpoint import (
            RLCheckpointConfig,
            build_rl_checkpoint_callback,
        )

        monkeypatch.chdir(tmp_path)
        cfg = RLCheckpointConfig(save_every_steps=2)
        cb = build_rl_checkpoint_callback(cfg, output_dir="run", task="grpo")
        # step 1 → no save; step 2 → save.
        cb.on_step_end(None, _FakeState(1), _FakeControl(), model=_FakeSavableModel())
        assert not (tmp_path / "run" / "rl-checkpoints").exists()
        cb.on_step_end(None, _FakeState(2), _FakeControl(), model=_FakeSavableModel())
        assert (tmp_path / "run" / "rl-checkpoints" / "step-2").is_dir()


# --------------------------------------------------------------------------
# #236 — ULD
# --------------------------------------------------------------------------


class TestULD:
    def test_build_returns_projection(self):
        from soup_cli.utils.uld import ULDConfig, ULDProjection, build_uld_projection

        proj = build_uld_projection(
            ULDConfig(strategy="wasserstein", student_vocab_size=10, teacher_vocab_size=12)
        )
        assert isinstance(proj, ULDProjection)

    def test_build_rejects_non_config(self):
        from soup_cli.utils.uld import build_uld_projection

        with pytest.raises(TypeError):
            build_uld_projection({"strategy": "wasserstein"})

    def test_wasserstein_loss_different_vocab(self):
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=8, teacher_vocab_size=12)
        s = torch.randn(2, 3, 8, requires_grad=True)
        t = torch.randn(2, 3, 12)
        loss = uld_distill_loss(s, t, config=cfg)
        assert torch.isfinite(loss)
        loss.backward()
        assert s.grad is not None

    def test_topk_align_loss(self):
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(
            strategy="topk_align", student_vocab_size=8, teacher_vocab_size=12, top_k=4
        )
        s = torch.randn(2, 3, 8, requires_grad=True)
        t = torch.randn(2, 3, 12)
        loss = uld_distill_loss(s, t, config=cfg)
        assert torch.isfinite(loss)
        loss.backward()

    def test_identical_distributions_low_loss(self):
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=8, teacher_vocab_size=8)
        logits = torch.randn(2, 3, 8)
        loss = uld_distill_loss(logits, logits.clone(), config=cfg)
        assert float(loss) < 1e-5

    def test_attention_mask_applied(self):
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=8, teacher_vocab_size=8)
        s = torch.randn(2, 3, 8)
        t = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        loss = uld_distill_loss(s, t, config=cfg, attention_mask=mask)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# #237 — MiniLLM
# --------------------------------------------------------------------------


class TestMiniLLM:
    def test_build_returns_callback(self):
        from soup_cli.utils.minillm import (
            MiniLLMCallback,
            MiniLLMConfig,
            build_minillm_callback,
        )

        cb = build_minillm_callback(MiniLLMConfig(teacher_mix_ratio=0.5))
        assert isinstance(cb, MiniLLMCallback)

    def test_build_rejects_non_config(self):
        from soup_cli.utils.minillm import build_minillm_callback

        with pytest.raises(TypeError):
            build_minillm_callback({})

    def test_distill_term_finite_and_differentiable(self):
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_distill_term

        cfg = MiniLLMConfig(teacher_mix_ratio=0.5, length_normalize=True)
        s = torch.randn(2, 4, 16, requires_grad=True)
        t = torch.randn(2, 4, 16)
        labels = torch.randint(0, 16, (2, 4))
        labels[0, 0] = -100  # masked
        loss = minillm_distill_term(s, t, labels, config=cfg)
        assert torch.isfinite(loss)
        loss.backward()
        assert s.grad is not None

    def test_teacher_mix_ratio_zero_gives_near_zero(self):
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_distill_term

        cfg = MiniLLMConfig(teacher_mix_ratio=0.0)
        s = torch.randn(2, 4, 16)
        t = torch.randn(2, 4, 16)
        labels = torch.randint(0, 16, (2, 4))
        loss = minillm_distill_term(s, t, labels, config=cfg)
        # ratio=0 → target = student_detached → reverse-KL ≈ 0.
        assert abs(float(loss)) < 1e-4

    def test_anchor_term_with_file(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, build_minillm_callback

        # Loading a tokenizer/model from the Hub flakes on CI runners that get
        # HF-rate-limited (the cache-warm step is best-effort). Skip on network
        # failure — the anchor math is covered network-free by
        # ``test_anchor_term_with_fake_model`` below.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
            model = AutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2"
            )
        except OSError as exc:  # pragma: no cover — network-dependent
            pytest.skip(f"HF model unavailable (offline / rate-limited): {exc}")

        monkeypatch.chdir(tmp_path)
        anchor = tmp_path / "anchor.jsonl"
        anchor.write_text(
            "\n".join(json.dumps({"text": f"sentence number {i}"}) for i in range(4))
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        cb = build_minillm_callback(
            MiniLLMConfig(pretrain_anchor_weight=0.1, pretrain_anchor_path="anchor.jsonl"),
            tokenizer=tok,
        )
        term = cb.anchor_term(model)
        assert term is not None
        assert torch.isfinite(term)

    def test_anchor_term_with_fake_model(self, tmp_path, monkeypatch):
        """Network-free coverage of ``_load_anchor`` + ``anchor_term`` — a fake
        tokenizer + tiny ``nn.Module`` exercise the same lines as the Hub-backed
        test above, so the coverage gate does not depend on HF availability."""
        import torch
        import torch.nn as nn

        from soup_cli.utils.minillm import MiniLLMConfig, build_minillm_callback

        class _FakeOut:
            def __init__(self, logits):
                self.logits = logits

        class _TinyLM(nn.Module):
            def __init__(self, vocab=16):
                super().__init__()
                self.emb = nn.Embedding(vocab, 4)
                self.head = nn.Linear(4, vocab)

            def forward(self, input_ids, attention_mask=None):  # noqa: ARG002
                return _FakeOut(self.head(self.emb(input_ids)))

        class _FakeTok:
            def __call__(
                self, texts, return_tensors=None, padding=None,
                truncation=None, max_length=None,
            ):  # noqa: ARG002
                ids = torch.tensor([[1, 2, 3, 4] for _ in texts])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        monkeypatch.chdir(tmp_path)
        anchor = tmp_path / "anchor.jsonl"
        anchor.write_text(
            "\n".join(json.dumps({"text": f"sentence {i}"}) for i in range(4))
        )
        cb = build_minillm_callback(
            MiniLLMConfig(pretrain_anchor_weight=0.25, pretrain_anchor_path="anchor.jsonl"),
            tokenizer=_FakeTok(),
        )
        term = cb.anchor_term(_TinyLM())
        assert term is not None
        assert torch.isfinite(term)

    def test_anchor_term_disabled_returns_none(self):
        import torch.nn as nn

        from soup_cli.utils.minillm import MiniLLMConfig, build_minillm_callback

        cb = build_minillm_callback(MiniLLMConfig(teacher_mix_ratio=0.3))
        assert cb.anchor_term(nn.Linear(2, 2)) is None


# --------------------------------------------------------------------------
# #239 — iterative DPO
# --------------------------------------------------------------------------


class TestIterativeDPO:
    def test_build_pairs_from_scored(self):
        from soup_cli.utils.iterative_dpo import build_pairs_from_scored

        assert build_pairs_from_scored([("a", 1.0), ("b", 3.0)]) == ("b", "a")
        assert build_pairs_from_scored([("a", 1.0)]) is None
        assert build_pairs_from_scored([("a", 2.0), ("b", 2.0)]) is None

    def test_run_iterative_dpo_with_fakes(self, tmp_path, monkeypatch):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOResult,
            build_iterative_dpo_plan,
            run_iterative_dpo,
        )

        monkeypatch.chdir(tmp_path)
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(
            "\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(3))
        )

        plan = build_iterative_dpo_plan(
            base_model="tiny",
            reward_model="rm",
            prompts_path="prompts.jsonl",
            output_dir="out",
            rounds=2,
            pairs_per_round=10,
        )

        calls = {"sample_adapters": [], "score": 0, "train": []}

        def fake_sample(*, base_model, adapter_path, prompts, num_samples,
                        max_new_tokens, device):
            calls["sample_adapters"].append(adapter_path)
            return [[f"{p}-a", f"{p}-b"] for p in prompts]

        def fake_score(*, reward_model, prompt, completions, device):
            calls["score"] += 1
            return [float(len(c)) for c in completions]

        def fake_train(*, base_model, pairs_path, adapter_path):
            calls["train"].append((base_model, adapter_path))
            import os

            os.makedirs(adapter_path, exist_ok=True)

        result = run_iterative_dpo(
            plan, sample_fn=fake_sample, score_fn=fake_score, train_fn=fake_train
        )
        assert isinstance(result, IterativeDPOResult)
        assert result.rounds_completed == 2
        # Training ALWAYS starts from the plan's base (never an adapter dir).
        assert calls["train"][0][0] == "tiny"
        assert calls["train"][1][0] == "tiny"
        # Round 0 samples from base (None adapter); round 1 from round-0 adapter.
        assert calls["sample_adapters"][0] is None
        assert calls["sample_adapters"][1].endswith("round-00/adapter")
        # pairs written
        assert (tmp_path / "out" / "round-00" / "pairs.jsonl").is_file()

    def test_run_rejects_non_plan(self):
        from soup_cli.utils.iterative_dpo import run_iterative_dpo

        with pytest.raises(TypeError):
            run_iterative_dpo({"rounds": 1})

    def test_cli_plan_only_still_exits_zero(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.iterative_dpo import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "p.jsonl").write_text(json.dumps({"prompt": "q"}))
        runner = CliRunner()
        res = runner.invoke(
            app,
            [
                "--base-model", "b", "--reward-model", "rm",
                "--prompts", "p.jsonl", "--output-dir", "o",
                "--rounds", "1", "--pairs-per-round", "10", "--plan-only",
            ],
        )
        assert res.exit_code == 0, res.output

    def test_cli_runs_with_monkeypatched_runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import soup_cli.utils.iterative_dpo as idpo
        from soup_cli.commands.iterative_dpo import app
        from soup_cli.utils.iterative_dpo import IterativeDPOResult

        monkeypatch.chdir(tmp_path)
        (tmp_path / "p.jsonl").write_text(json.dumps({"prompt": "q"}))

        def fake_run(plan, **kwargs):
            return IterativeDPOResult(
                rounds_completed=1, final_adapter="o/round-00/adapter",
                per_round_pairs=(1,),
            )

        monkeypatch.setattr(idpo, "run_iterative_dpo", fake_run)
        runner = CliRunner()
        res = runner.invoke(
            app,
            [
                "--base-model", "b", "--reward-model", "rm",
                "--prompts", "p.jsonl", "--output-dir", "o",
                "--rounds", "1", "--pairs-per-round", "10",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "Done" in res.output


# --------------------------------------------------------------------------
# #159 — GRPO variant fallback warning
# --------------------------------------------------------------------------


class _FakeGRPOBase:
    """Minimal stand-in for trl.GRPOTrainer for the variant subclass."""

    class _Args:
        beta = 0.1

    def __init__(self):
        self.args = self._Args()
        self.super_called = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.super_called += 1
        return 0.0


class TestGrpoVariantFallbackWarning:
    def test_fallback_warns_once(self, caplog):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        cls = make_grpo_trainer_variant(_FakeGRPOBase, "gspo")
        inst = cls()
        # inputs missing per-token logps → fallback path.
        with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.grpo"):
            inst.compute_loss(None, {})
            inst.compute_loss(None, {})
        warnings = [r for r in caplog.records if "fell back" in r.message]
        assert len(warnings) == 1  # one-shot
        assert inst.super_called == 2  # but fallback happened both times


# --------------------------------------------------------------------------
# #160 — in-place GRPO EMA
# --------------------------------------------------------------------------


class TestGrpoEmaInPlace:
    def test_in_place_blends_toward_policy(self):
        import torch
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        ref = nn.Linear(3, 3)
        pol = nn.Linear(3, 3)
        with torch.no_grad():
            ref.weight.fill_(0.0)
            pol.weight.fill_(1.0)
        update_ema_in_place(ref, pol, 0.25)
        # ref = 0.75*0 + 0.25*1 = 0.25
        assert torch.allclose(ref.weight, torch.full_like(ref.weight, 0.25))

    def test_returns_updated_count(self):
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        # nn.Linear(2, 2) has weight + bias → 2 shared params updated.
        n = update_ema_in_place(nn.Linear(2, 2), nn.Linear(2, 2), 0.5)
        assert n == 2

    def test_zero_overlap_returns_zero(self):
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        # Disjoint parameter names → no overlap → count 0.
        class _A(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Linear(2, 2)

        class _B(nn.Module):
            def __init__(self):
                super().__init__()
                self.beta = nn.Linear(2, 2)

        assert update_ema_in_place(_A(), _B(), 0.5) == 0

    def test_rejects_bool_alpha(self):
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        with pytest.raises(TypeError):
            update_ema_in_place(nn.Linear(2, 2), nn.Linear(2, 2), True)

    def test_rejects_out_of_range_alpha(self):
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        with pytest.raises(ValueError):
            update_ema_in_place(nn.Linear(2, 2), nn.Linear(2, 2), 1.5)

    def test_shape_mismatch_skipped(self):
        import torch
        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import update_ema_in_place

        ref = nn.Linear(3, 3)
        pol = nn.Linear(2, 2)  # different shapes for the same param name
        with torch.no_grad():
            ref.weight.fill_(7.0)
        update_ema_in_place(ref, pol, 0.5)
        # shape mismatch → ref untouched.
        assert torch.allclose(ref.weight, torch.full_like(ref.weight, 7.0))

    def test_callback_warns_once_on_zero_overlap(self, caplog):
        import logging

        import torch.nn as nn

        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )

        class _A(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Linear(2, 2)

        class _B(nn.Module):
            def __init__(self):
                super().__init__()
                self.beta = nn.Linear(2, 2)

        cb = GRPOStabilityCallback(ref_model_ema_alpha=0.5)
        cb._policy_model = _A()
        cb._ref_model = _B()
        with caplog.at_level(logging.WARNING):
            cb.on_step_end(args=None, state=None, control=None, model=cb._policy_model)
            cb.on_step_end(args=None, state=None, control=None, model=cb._policy_model)
        warnings = [r for r in caplog.records if "0 shared parameters" in r.message]
        assert len(warnings) == 1  # one-shot


# --------------------------------------------------------------------------
# Source wiring + patch invariants
# --------------------------------------------------------------------------


class TestSourceWiring:
    def _read(self, rel: str) -> str:
        root = Path(__file__).resolve().parent.parent
        return (root / "src" / "soup_cli" / rel).read_text(encoding="utf-8")

    def test_grpo_wires_rl_callbacks(self):
        src = self._read("trainer/grpo.py")
        assert "attach_rl_callbacks" in src
        assert "wrap_reward_funcs" in src

    def test_distill_wires_uld_and_minillm(self):
        src = self._read("trainer/distill.py")
        assert "build_uld_projection" in src
        assert "build_minillm_callback" in src

    def test_stability_callback_uses_in_place_ema(self):
        src = self._read("monitoring/grpo_stability_callback.py")
        assert "update_ema_in_place" in src
        # the old full-state_dict round-trip should be gone from on_step_end.
        assert "self._ref_model.load_state_dict(ref_sd" not in src

    def test_no_top_level_torch_in_new_utils(self):
        for rel in (
            "utils/rl_signal_buffer.py",
            "utils/reward_hacking.py",
            "utils/echo_trap.py",
            "utils/uld.py",
            "utils/minillm.py",
            "utils/iterative_dpo.py",
            "utils/rl_checkpoint.py",
        ):
            src = self._read(rel)
            assert "\nimport torch" not in src, rel
            assert "\nfrom torch" not in src, rel


class TestPatchInvariants:
    def test_version_bumped(self):
        parts = soup_cli.__version__.split(".")
        assert (int(parts[0]), int(parts[1]), int(parts[2])) >= (0, 71, 11)
