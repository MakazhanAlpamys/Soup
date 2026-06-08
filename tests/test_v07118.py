"""v0.71.18 — Distill + agent depth.

Closes #257 (MiniLLM on-policy rollout), #258 (ULD token-sequence
alignment), #110 (agent eval RLVR sandbox), #16 (train --cloud modal).

Tests written first (TDD). Heavy-import (torch/transformers) tests guard
with ``pytest.importorskip`` so the no-torch CI variant skips cleanly.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fake torch LM for the distillation kernels (#257 / #258).
# ---------------------------------------------------------------------------
def _make_fake_lm(vocab: int = 10, hidden: int = 6, seed: int = 0):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class _FakeLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)
            self.config = types.SimpleNamespace(vocab_size=vocab)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            h = self.emb(input_ids)
            logits = self.head(h)
            return types.SimpleNamespace(logits=logits)

    return _FakeLM()


# ===========================================================================
# #257 — MiniLLM on-policy rollout
# ===========================================================================
class TestMiniLLMOnPolicySchema:
    def _yaml(self, extra: str = "") -> str:
        return (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  teacher_model: hf-internal-testing/tiny-random-gpt2\n"
            "  minillm_enabled: true\n"
            f"{extra}"
            "output: ./out\n"
        )

    def test_default_off(self):
        cfg = load_config_from_string(self._yaml())
        assert cfg.training.minillm_on_policy is False

    def test_opt_in(self):
        cfg = load_config_from_string(self._yaml("  minillm_on_policy: true\n"))
        assert cfg.training.minillm_on_policy is True

    def test_on_policy_requires_minillm_enabled(self):
        bad = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  teacher_model: hf-internal-testing/tiny-random-gpt2\n"
            "  minillm_on_policy: true\n"
            "output: ./out\n"
        )
        with pytest.raises(ValueError, match="minillm_on_policy"):
            load_config_from_string(bad)

    def test_on_policy_non_bool_rejected(self):
        # Shared _validate_minillm_bool_fields raises a TypeError naming the
        # bool requirement (consistent with the other minillm bool toggles).
        with pytest.raises((TypeError, ValueError), match="bool"):
            load_config_from_string(self._yaml("  minillm_on_policy: 1\n"))


class TestMiniLLMConfigOnPolicy:
    def test_on_policy_field_default(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        cfg = MiniLLMConfig()
        assert cfg.on_policy is False
        assert cfg.rollout_length == 16

    def test_on_policy_accepts(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        cfg = MiniLLMConfig(on_policy=True, rollout_length=8)
        assert cfg.on_policy is True
        assert cfg.rollout_length == 8

    def test_on_policy_must_be_bool(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises((TypeError, ValueError), match="on_policy"):
            MiniLLMConfig(on_policy=1)

    @pytest.mark.parametrize("bad", [0, -1, 513, True])
    def test_rollout_length_bounds(self, bad):
        from soup_cli.utils.minillm import MiniLLMConfig

        with pytest.raises((TypeError, ValueError), match="rollout_length"):
            MiniLLMConfig(rollout_length=bad)

    def test_rollout_length_upper_boundary_accepted(self):
        from soup_cli.utils.minillm import MiniLLMConfig

        # 512 is the documented max — must be accepted (513 is rejected above).
        cfg = MiniLLMConfig(rollout_length=512)
        assert cfg.rollout_length == 512


class TestMiniLLMRolloutLengthSchema:
    """v0.71.18 #257 — training.minillm_rollout_length is a real knob."""

    def _yaml(self, extra: str) -> str:
        return (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  teacher_model: hf-internal-testing/tiny-random-gpt2\n"
            "  minillm_enabled: true\n"
            f"{extra}"
            "output: ./out\n"
        )

    def test_default_none(self):
        cfg = load_config_from_string(self._yaml(""))
        assert cfg.training.minillm_rollout_length is None

    def test_accepts_with_on_policy(self):
        cfg = load_config_from_string(
            self._yaml("  minillm_on_policy: true\n  minillm_rollout_length: 24\n")
        )
        assert cfg.training.minillm_rollout_length == 24

    def test_requires_on_policy(self):
        # Set without on_policy — a no-op for the offline blend → rejected.
        with pytest.raises(ValueError, match="minillm_on_policy"):
            load_config_from_string(self._yaml("  minillm_rollout_length: 24\n"))

    def test_requires_minillm_enabled(self):
        bad = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  teacher_model: hf-internal-testing/tiny-random-gpt2\n"
            "  minillm_rollout_length: 24\n"
            "output: ./out\n"
        )
        with pytest.raises(ValueError, match="minillm_rollout_length"):
            load_config_from_string(bad)

    def test_bool_rejected(self):
        with pytest.raises((TypeError, ValueError), match="bool"):
            load_config_from_string(
                self._yaml("  minillm_on_policy: true\n  minillm_rollout_length: true\n")
            )

    @pytest.mark.parametrize("bad", [0, 513])
    def test_bounds(self, bad):
        with pytest.raises(ValueError):
            load_config_from_string(
                self._yaml(
                    "  minillm_on_policy: true\n"
                    f"  minillm_rollout_length: {bad}\n"
                )
            )

    def test_distill_trainer_threads_explicit_length(self):
        # Source-grep regression guard: the distill trainer reads
        # tcfg.minillm_rollout_length (not just the auto-derived 32).
        src = (REPO_ROOT / "src" / "soup_cli" / "trainer" / "distill.py").read_text(
            encoding="utf-8"
        )
        assert "tcfg.minillm_rollout_length is not None" in src
        assert "rollout_length=rollout_len" in src


class TestMiniLLMOnPolicyRollout:
    def test_rollout_returns_loss_and_steps(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm(seed=1)
        teacher = _make_fake_lm(seed=2)
        for p in teacher.parameters():
            p.requires_grad_(False)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)

        def _greedy(probs):
            return probs.argmax(dim=-1, keepdim=True)

        loss, steps = minillm_on_policy_rollout(
            student,
            teacher,
            ids,
            mask,
            config=MiniLLMConfig(teacher_mix_ratio=0.3, on_policy=True),
            max_new_tokens=4,
            temperature=1.0,
            sample_fn=_greedy,
        )
        assert steps == 4
        assert loss.requires_grad
        assert torch.isfinite(loss).item()

    def test_rollout_grad_flows_to_student_only(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm(seed=1)
        teacher = _make_fake_lm(seed=2)
        for p in teacher.parameters():
            p.requires_grad_(False)
        ids = torch.tensor([[1, 2]])

        loss, _ = minillm_on_policy_rollout(
            student,
            teacher,
            ids,
            None,
            config=MiniLLMConfig(teacher_mix_ratio=0.5, on_policy=True),
            max_new_tokens=3,
            temperature=1.0,
            sample_fn=lambda p: p.argmax(dim=-1, keepdim=True),
        )
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        assert all(p.grad is None for p in teacher.parameters())

    def test_length_normalize_divides_by_steps(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm(seed=3)
        teacher = _make_fake_lm(seed=4)
        ids = torch.tensor([[1, 2]])
        kw = dict(
            max_new_tokens=4,
            temperature=1.0,
            sample_fn=lambda p: p.argmax(dim=-1, keepdim=True),
        )
        norm, _ = minillm_on_policy_rollout(
            student, teacher, ids, None,
            config=MiniLLMConfig(teacher_mix_ratio=0.0, length_normalize=True, on_policy=True),
            **kw,
        )
        total, _ = minillm_on_policy_rollout(
            student, teacher, ids, None,
            config=MiniLLMConfig(teacher_mix_ratio=0.0, length_normalize=False, on_policy=True),
            **kw,
        )
        # Non-normalised is the sum over 4 steps; normalised is the mean.
        assert total.item() == pytest.approx(norm.item() * 4, rel=1e-4)

    def test_rollout_rejects_bad_config(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import minillm_on_policy_rollout

        student = _make_fake_lm()
        with pytest.raises(TypeError, match="MiniLLMConfig"):
            minillm_on_policy_rollout(
                student, student, torch.tensor([[1]]), None,
                config={"teacher_mix_ratio": 0.5}, max_new_tokens=2, temperature=1.0,
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rollout_rejects_bad_max_new_tokens(self, bad):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm()
        with pytest.raises(ValueError, match="max_new_tokens"):
            minillm_on_policy_rollout(
                student, student, torch.tensor([[1]]), None,
                config=MiniLLMConfig(on_policy=True), max_new_tokens=bad, temperature=1.0,
            )

    def test_rollout_rejects_nonpositive_temperature(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm()
        with pytest.raises(ValueError, match="temperature"):
            minillm_on_policy_rollout(
                student, student, torch.tensor([[1]]), None,
                config=MiniLLMConfig(on_policy=True), max_new_tokens=2, temperature=0.0,
            )

    def test_rollout_rejects_bool_temperature(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm()
        with pytest.raises(TypeError, match="temperature"):
            minillm_on_policy_rollout(
                student, student, torch.tensor([[1]]), None,
                config=MiniLLMConfig(on_policy=True), max_new_tokens=2, temperature=True,
            )


class TestMiniLLMCallbackOnPolicy:
    def test_on_policy_term(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, build_minillm_callback

        cb = build_minillm_callback(
            MiniLLMConfig(teacher_mix_ratio=0.4, on_policy=True, rollout_length=3),
        )
        student = _make_fake_lm(seed=5)
        teacher = _make_fake_lm(seed=6)
        for p in teacher.parameters():
            p.requires_grad_(False)
        ids = torch.tensor([[1, 2, 3]])
        loss = cb.on_policy_term(
            student, teacher, ids, torch.ones_like(ids),
            sample_fn=lambda p: p.argmax(dim=-1, keepdim=True),
        )
        assert torch.isfinite(loss).item()
        assert loss.requires_grad


class TestDistillOnPolicyWiring:
    def test_distill_routes_on_policy(self):
        src = (REPO_ROOT / "src/soup_cli/trainer/distill.py").read_text(encoding="utf-8")
        assert "on_policy" in src
        assert "on_policy_term" in src

    def test_minillm_no_top_level_torch(self):
        src = (REPO_ROOT / "src/soup_cli/utils/minillm.py").read_text(encoding="utf-8")
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


class TestTrainCliMinillmOnPolicy:
    def test_flag_in_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["train", "--help"])
        out = result.stdout.replace("\n", " ")
        # ANSI/Rich may split on dashes — collapse whitespace and check tokens.
        assert "minillm-on-policy" in out.replace(" ", "")


# ===========================================================================
# #258 — ULD token-sequence alignment for disjoint tokenizers
# ===========================================================================
class TestULDAlignedStrategy:
    def test_strategy_registered(self):
        from soup_cli.utils.uld import SUPPORTED_ULD_STRATEGIES, validate_uld_strategy

        assert "wasserstein_aligned" in SUPPORTED_ULD_STRATEGIES
        assert validate_uld_strategy("Wasserstein_Aligned") == "wasserstein_aligned"

    def test_aligned_does_not_require_top_k(self):
        from soup_cli.utils.uld import ULDConfig

        cfg = ULDConfig(
            strategy="wasserstein_aligned",
            student_vocab_size=10,
            teacher_vocab_size=12,
        )
        assert cfg.top_k is None

    def test_aligned_rejects_top_k(self):
        from soup_cli.utils.uld import ULDConfig

        with pytest.raises(ValueError, match="top_k"):
            ULDConfig(
                strategy="wasserstein_aligned",
                student_vocab_size=10,
                teacher_vocab_size=12,
                top_k=5,
            )

    def test_schema_accepts_aligned(self):
        yaml = (
            "base: hf-internal-testing/tiny-random-gpt2\n"
            "task: distill\n"
            "data:\n  train: data.jsonl\n  format: chatml\n"
            "training:\n"
            "  teacher_model: sshleifer/tiny-gpt2\n"
            "  uld_strategy: wasserstein_aligned\n"
            "output: ./out\n"
        )
        cfg = load_config_from_string(yaml)
        assert cfg.training.uld_strategy == "wasserstein_aligned"


class TestAlignTokenSequences:
    def test_identical_tokenization(self):
        from soup_cli.utils.uld import align_token_sequences

        assert align_token_sequences(["a", "b", "c"], ["a", "b", "c"]) == [
            [0], [1], [2]
        ]

    def test_disjoint_tokenization_same_text(self):
        from soup_cli.utils.uld import align_token_sequences

        # "hello" tokenised differently → char-overlap alignment.
        assert align_token_sequences(["he", "llo"], ["hel", "lo"]) == [[0], [0, 1]]

    def test_empty_student(self):
        from soup_cli.utils.uld import align_token_sequences

        assert align_token_sequences([], ["a"]) == []

    def test_empty_teacher(self):
        from soup_cli.utils.uld import align_token_sequences

        assert align_token_sequences(["a", "b"], []) == [[], []]

    def test_different_text_falls_back(self):
        from soup_cli.utils.uld import align_token_sequences

        # Decode artifacts: a typo / extra char — alignment still produced.
        out = align_token_sequences(["hel", "lo"], ["he", "llo", "!"])
        assert isinstance(out, list)
        assert len(out) == 2
        # Each entry is a (possibly empty) list of teacher indices.
        assert all(isinstance(e, list) for e in out)
        # The shared "hello" prefix should align somewhere.
        assert any(e for e in out)

    def test_difflib_fallback_specific_alignment(self):
        from soup_cli.utils.uld import align_token_sequences

        # Student text "abXc" (X is a decode artefact) vs teacher "ab","c".
        # The difflib path (texts differ) must still map the shared chars:
        # student "ab" -> teacher "ab" (idx 0); student "Xc" -> teacher "c" (1).
        out = align_token_sequences(["ab", "Xc"], ["ab", "c"])
        assert len(out) == 2
        assert 0 in out[0]   # "ab" student token aligns to teacher "ab"
        assert 1 in out[1]   # the "c" in "Xc" aligns to teacher "c"

    def test_char_cap_truncates_difflib_branch(self):
        from soup_cli.utils.uld import _MAX_ALIGN_TOKENS, align_token_sequences

        # Disjoint texts (forces the difflib branch) AND oversize → token cap.
        student = ["s"] * (_MAX_ALIGN_TOKENS + 50)
        teacher = ["t"] * (_MAX_ALIGN_TOKENS + 50)
        out = align_token_sequences(student, teacher)
        assert len(out) <= _MAX_ALIGN_TOKENS

    def test_char_cap_truncates(self):
        from soup_cli.utils.uld import _MAX_ALIGN_TOKENS, align_token_sequences

        big = ["x"] * (_MAX_ALIGN_TOKENS + 50)
        out = align_token_sequences(big, big)
        assert len(out) <= _MAX_ALIGN_TOKENS

    def test_non_string_tokens_coerced(self):
        from soup_cli.utils.uld import align_token_sequences

        # Robust to non-str entries (defensive str() coercion).
        out = align_token_sequences(["a", 1], ["a", "1"])
        assert len(out) == 2


class TestAggregateAlignedLogits:
    def test_mean_pool(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import aggregate_aligned_logits

        teacher = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
        )  # [Tt=3, Vt=2]
        # student pos 0 → teacher {0,1} (mean=2.0), pos 1 → teacher {2}.
        out = aggregate_aligned_logits(teacher, [[0, 1], [2]])
        assert out.shape == (2, 2)
        assert out[0, 0].item() == pytest.approx(2.0)
        assert out[1, 0].item() == pytest.approx(5.0)

    def test_empty_alignment_is_zeros(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import aggregate_aligned_logits

        teacher = torch.ones((2, 3))
        out = aggregate_aligned_logits(teacher, [[], [0]])
        assert out.shape == (2, 3)
        assert out[0].abs().sum().item() == 0.0

    def test_out_of_range_indices_ignored(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import aggregate_aligned_logits

        teacher = torch.ones((2, 3))
        out = aggregate_aligned_logits(teacher, [[0, 99]])
        # index 99 ignored — only teacher[0] contributes.
        assert out.shape == (1, 3)
        assert out[0, 0].item() == pytest.approx(1.0)


class TestUldAlignedLoss:
    def test_finite_and_differentiable(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned",
            student_vocab_size=4,
            teacher_vocab_size=5,
        )
        # [B=1, Ts=2, Vs=4]; [B=1, Tt=2, Vt=5]
        s = torch.randn(1, 2, 4, requires_grad=True)
        t = torch.randn(1, 2, 5)
        loss = uld_aligned_loss(
            s, t,
            [["he", "llo"]],
            [["hel", "lo"]],
            config=cfg,
        )
        assert torch.isfinite(loss).item()
        loss.backward()
        assert s.grad is not None and s.grad.abs().sum() > 0

    def test_rejects_non_config(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import uld_aligned_loss

        with pytest.raises(TypeError, match="ULDConfig"):
            uld_aligned_loss(
                torch.randn(1, 1, 2), torch.randn(1, 1, 2),
                [["a"]], [["a"]], config={},
            )

    def test_rejects_short_token_lists(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned",
            student_vocab_size=4,
            teacher_vocab_size=5,
        )
        # Logit batch dim is 2 but only one per-batch token list supplied.
        with pytest.raises(ValueError, match="at least"):
            uld_aligned_loss(
                torch.randn(2, 1, 4), torch.randn(2, 1, 5),
                [["a"]], [["a"]], config=cfg,
            )

    def test_empty_student_tokens_zero_loss(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned",
            student_vocab_size=4,
            teacher_vocab_size=5,
        )
        # Empty per-batch student token list → no positions to align → the
        # ts==0 branch returns a finite zero contribution (no grad, no crash).
        loss = uld_aligned_loss(
            torch.randn(1, 2, 4), torch.randn(1, 2, 5),
            [[]], [["he", "llo"]], config=cfg,
        )
        assert torch.isfinite(loss).item()
        assert loss.item() == pytest.approx(0.0)


class TestDistillUldAlignedWiring:
    def test_distill_routes_aligned(self):
        src = (REPO_ROOT / "src/soup_cli/trainer/distill.py").read_text(encoding="utf-8")
        assert "wasserstein_aligned" in src
        assert "uld_aligned_loss" in src

    def test_uld_no_top_level_torch(self):
        src = (REPO_ROOT / "src/soup_cli/utils/uld.py").read_text(encoding="utf-8")
        assert "\nimport torch" not in src
        assert "\nfrom torch" not in src


# ===========================================================================
# #110 — soup agent eval --sandbox
# ===========================================================================
def _strip_ansi(text: str) -> str:
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", " ", text)


class TestBuildEvalStub:
    def test_stub_is_python_and_b64_only(self):
        from soup_cli.utils.agent_sandbox import build_eval_stub

        stub = build_eval_stub(
            tool="get_user",
            parameters=["user_id"],
            path="/users/{user_id}",
            arguments={"user_id": 1},
        )
        assert "import base64" in stub
        assert "base64.b64decode" in stub
        # The raw argument values are NOT interpolated as code.
        assert "user_id" not in stub.replace("import base64, json", "")

    def test_required_params_extracted(self):
        from soup_cli.utils.agent_sandbox import _required_path_params

        assert _required_path_params("/users/{user_id}/posts/{post_id}") == [
            "user_id",
            "post_id",
        ]
        assert _required_path_params("/posts") == []
        assert _required_path_params(None) == []

    def test_non_mapping_arguments_rejected(self):
        from soup_cli.utils.agent_sandbox import build_eval_stub

        with pytest.raises(ValueError, match="mapping"):
            build_eval_stub(tool="t", parameters=[], path="/x", arguments=[1, 2])


class TestClassifySandboxOutcome:
    def test_timeout(self):
        from soup_cli.utils.agent_sandbox import classify_sandbox_outcome

        assert classify_sandbox_outcome(None, "", True) == "timeout"

    def test_tool_error_nonzero(self):
        from soup_cli.utils.agent_sandbox import classify_sandbox_outcome

        assert classify_sandbox_outcome(1, "", False) == "tool_error"

    def test_tool_error_empty_output(self):
        from soup_cli.utils.agent_sandbox import classify_sandbox_outcome

        assert classify_sandbox_outcome(0, "", False) == "tool_error"

    def test_tool_error_unparseable_output(self):
        from soup_cli.utils.agent_sandbox import classify_sandbox_outcome

        assert classify_sandbox_outcome(0, "not json", False) == "tool_error"

    def test_ok(self):
        from soup_cli.utils.agent_sandbox import classify_sandbox_outcome

        ok_json = json.dumps({"url": "/x"})
        assert classify_sandbox_outcome(0, ok_json, False) == "ok"


class TestRunEvalInSandbox:
    """Real subprocess execution (functional on every platform; strong
    isolation primitives are POSIX-only and skipped on Windows)."""

    def test_run_ok_stub(self):
        from soup_cli.utils.agent_sandbox import build_eval_stub, run_eval_in_sandbox

        stub = build_eval_stub(
            tool="get_user",
            parameters=["user_id"],
            path="/users/{user_id}",
            arguments={"user_id": 42},
        )
        rc, out, timed = run_eval_in_sandbox(stub)
        assert timed is False
        assert rc == 0, out
        assert json.loads(out)["url"] == "/users/42"

    def test_run_missing_required_param(self):
        from soup_cli.utils.agent_sandbox import build_eval_stub, run_eval_in_sandbox

        stub = build_eval_stub(
            tool="get_user",
            parameters=["user_id"],
            path="/users/{user_id}",
            arguments={},
        )
        rc, out, timed = run_eval_in_sandbox(stub)
        assert timed is False
        assert rc != 0


class TestScoreSandbox:
    def test_score_ok(self):
        from soup_cli.utils.agent_sandbox import score_sandbox

        cls = score_sandbox(
            tool="get_user",
            parameters=["user_id"],
            path="/users/{user_id}",
            arguments={"user_id": 1},
        )
        assert cls == "ok"

    def test_score_tool_error(self):
        from soup_cli.utils.agent_sandbox import score_sandbox

        cls = score_sandbox(
            tool="get_user",
            parameters=["user_id"],
            path="/users/{user_id}",
            arguments={},
        )
        assert cls == "tool_error"

    def test_score_uses_override(self, monkeypatch):
        import soup_cli.utils.agent_sandbox as m

        monkeypatch.setattr(
            m, "_AGENT_SANDBOX_RUN_OVERRIDE", lambda stub: (None, "", True)
        )
        assert (
            m.score_sandbox(tool="t", parameters=[], path="/x", arguments={})
            == "timeout"
        )


def _write_sandbox_fixture(tmp_path: Path) -> None:
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/users/{user_id}": {
                "get": {
                    "operationId": "get_user",
                    "parameters": [{"name": "user_id", "in": "path"}],
                }
            },
            "/posts": {
                "get": {
                    "operationId": "list_posts",
                    "parameters": [
                        {"name": "limit", "in": "query"},
                        {"name": "offset", "in": "query"},
                    ],
                }
            },
        },
    }
    (tmp_path / "api.json").write_text(json.dumps(spec), encoding="utf-8")
    preds = [
        {"tool": "get_user", "arguments": {"user_id": 1}},   # ok
        {"tool": "get_user", "arguments": {}},               # tool_error
        {"tool": "list_posts", "arguments": {"bogus": 1}},   # arg_error (param)
        {"tool": "nope", "arguments": {}},                   # arg_error (tool)
    ]
    (tmp_path / "p.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds), encoding="utf-8"
    )


class TestAgentEvalSandboxCli:
    def test_flag_in_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["agent", "eval", "--help"])
        assert "sandbox" in _strip_ansi(result.stdout).replace(" ", "")

    def test_sandbox_scorecard_real(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_sandbox_fixture(tmp_path)
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app,
            [
                "agent", "eval", "--spec", "api.json",
                "--predictions", "p.jsonl", "--sandbox",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        txt = _strip_ansi(result.output)
        assert "sandbox eval" in txt.lower()
        assert "ok: 1" in txt
        assert "tool_error: 1" in txt
        assert "arg_error: 2" in txt
        assert "timeout: 0" in txt
        assert "isolation" in txt.lower()

    def test_sandbox_timeout_via_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_sandbox_fixture(tmp_path)
        import soup_cli.utils.agent_sandbox as m

        monkeypatch.setattr(
            m, "_AGENT_SANDBOX_RUN_OVERRIDE", lambda stub: (None, "", True)
        )
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app,
            [
                "agent", "eval", "--spec", "api.json",
                "--predictions", "p.jsonl", "--sandbox",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        txt = _strip_ansi(result.output)
        # Both heuristic-passing rows forced to timeout; the 2 arg_error stay.
        assert "timeout: 2" in txt
        assert "arg_error: 2" in txt


class TestAgentSandboxNoTopLevelHeavyImport:
    def test_no_top_level_torch(self):
        src = (REPO_ROOT / "src/soup_cli/utils/agent_sandbox.py").read_text(
            encoding="utf-8"
        )
        assert "\nimport torch" not in src
        # rewards.py is imported lazily inside run_eval_in_sandbox (not at top).
        assert "\nfrom soup_cli.trainer.rewards import" not in src


# ===========================================================================
# #16 — soup train --cloud modal
# ===========================================================================
_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./out\n"
)


class TestValidateCloud:
    def test_modal(self):
        from soup_cli.cloud.modal import validate_cloud

        assert validate_cloud("modal") == "modal"
        assert validate_cloud("MODAL") == "modal"

    @pytest.mark.parametrize("bad", ["", "runpod", "x" * 40])
    def test_rejects(self, bad):
        from soup_cli.cloud.modal import validate_cloud

        with pytest.raises(ValueError):
            validate_cloud(bad)

    def test_rejects_bool_and_nul(self):
        from soup_cli.cloud.modal import validate_cloud

        with pytest.raises(ValueError):
            validate_cloud(True)
        with pytest.raises(ValueError):
            validate_cloud("mo\x00dal")


class TestValidateGpu:
    @pytest.mark.parametrize(
        "gpu", ["t4", "l4", "a10g", "a100", "a100-80gb", "l40s", "h100", "A100"]
    )
    def test_known(self, gpu):
        from soup_cli.cloud.modal import validate_gpu

        assert validate_gpu(gpu) == gpu.lower()

    @pytest.mark.parametrize("bad", ["", "v100", "gpu\x00", True])
    def test_rejects(self, bad):
        from soup_cli.cloud.modal import validate_gpu

        with pytest.raises(ValueError):
            validate_gpu(bad)


class TestRenderModalStub:
    def test_structure(self):
        from soup_cli.cloud.modal import render_modal_stub

        stub = render_modal_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.18"
        )
        assert "import modal" in stub
        assert "modal.App" in stub
        assert 'gpu="A100"' in stub
        assert "soup-cli[train]==0.71.18" in stub
        assert "base64.b64decode" in stub

    def test_no_raw_config_injection(self):
        from soup_cli.cloud.modal import render_modal_stub

        secret_yaml = _SOUP_YAML + "INJECT_SENTINEL: pwned\n"
        stub = render_modal_stub(
            secret_yaml, gpu="h100", output_dir="./out", soup_version="0.71.18"
        )
        # The raw config text is base64-embedded — the sentinel never appears
        # verbatim in the rendered stub (zero injection surface).
        assert "INJECT_SENTINEL" not in stub
        assert 'gpu="H100"' in stub

    def test_oversize_config_rejected(self):
        from soup_cli.cloud.modal import _MAX_CONFIG_BYTES, render_modal_stub

        with pytest.raises(ValueError, match="exceeds"):
            render_modal_stub(
                "x" * (_MAX_CONFIG_BYTES + 1),
                gpu="a100", output_dir="./out", soup_version="0.71.18",
            )

    def test_bad_gpu_rejected(self):
        from soup_cli.cloud.modal import render_modal_stub

        with pytest.raises(ValueError):
            render_modal_stub(
                _SOUP_YAML, gpu="v100", output_dir="./out", soup_version="0.71.18"
            )

    def test_bad_output_dir_rejected(self):
        from soup_cli.cloud.modal import render_modal_stub

        with pytest.raises(ValueError):
            render_modal_stub(
                _SOUP_YAML, gpu="a100", output_dir="out\nINJECT",
                soup_version="0.71.18",
            )

    @pytest.mark.parametrize(
        "bad_version",
        ['1"; import os; os.system("x")  #', "1\\", '0.1"x', "ver with space"],
    )
    def test_soup_version_injection_rejected(self, bad_version):
        from soup_cli.cloud.modal import render_modal_stub

        with pytest.raises(ValueError, match="soup_version"):
            render_modal_stub(
                _SOUP_YAML, gpu="a100", output_dir="./out",
                soup_version=bad_version,
            )

    def test_soup_version_length_capped(self):
        from soup_cli.cloud.modal import _MAX_VERSION_LEN, render_modal_stub

        with pytest.raises(ValueError, match="soup_version"):
            render_modal_stub(
                _SOUP_YAML, gpu="a100", output_dir="./out",
                soup_version="0." * _MAX_VERSION_LEN,
            )

    def test_soup_version_pep440ish_accepted(self):
        from soup_cli.cloud.modal import render_modal_stub

        # Real-world version shapes must pass (dev / rc / local-version).
        stub = render_modal_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out",
            soup_version="0.71.18.dev0+gabc123",
        )
        assert "soup-cli[train]==0.71.18.dev0+gabc123" in stub


class TestPlanModalRun:
    def test_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud.modal import CloudPlan, plan_modal_run

        plan = plan_modal_run(
            "soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.18"
        )
        assert isinstance(plan, CloudPlan)
        assert plan.cloud == "modal"
        assert plan.gpu == "a100"
        assert plan.run_command == "modal run soup_modal_app.py"
        assert "modal.App" in plan.stub_text

    def test_outside_cwd_config_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cloud.modal import plan_modal_run

        with pytest.raises(ValueError):
            plan_modal_run(
                "../escape.yaml", gpu="a100", output_dir="./out",
                soup_version="0.71.18",
            )

    def test_missing_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.cloud.modal import plan_modal_run

        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            plan_modal_run(
                "nope.yaml", gpu="a100", output_dir="./out",
                soup_version="0.71.18",
            )


class TestWriteStub:
    def test_writes_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud.modal import plan_modal_run, write_stub

        plan = plan_modal_run(
            "soup.yaml", gpu="t4", output_dir="./out", soup_version="0.71.18"
        )
        path = write_stub(plan)
        assert (tmp_path / "soup_modal_app.py").exists()
        assert "modal.App" in Path(path).read_text(encoding="utf-8")


class TestSubmitModalRun:
    def test_override_seam(self, monkeypatch):
        import soup_cli.cloud.modal as m

        plan = m.CloudPlan(
            cloud="modal", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="modal run x.py",
        )
        monkeypatch.setattr(m, "_MODAL_SUBMIT_OVERRIDE", lambda p: 7)
        assert m.submit_modal_run(plan) == 7

    def test_no_token_raises(self, monkeypatch):
        import soup_cli.cloud.modal as m

        plan = m.CloudPlan(
            cloud="modal", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="modal run x.py",
        )
        # No env token AND no ~/.modal.toml.
        monkeypatch.setattr(m.os.path, "exists", lambda p: False)
        with pytest.raises(RuntimeError, match="authenticated"):
            m.submit_modal_run(plan, env={})

    def test_non_plan_rejected(self):
        from soup_cli.cloud.modal import submit_modal_run

        with pytest.raises(TypeError, match="CloudPlan"):
            submit_modal_run({"stub_path": "x"})

    def test_modal_sdk_missing_raises(self, monkeypatch):
        import sys

        import soup_cli.cloud.modal as m

        plan = m.CloudPlan(
            cloud="modal", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="modal run x.py",
        )
        # Token present (passes the auth gate) but the Modal SDK is absent.
        monkeypatch.setitem(sys.modules, "modal", None)  # forces ImportError
        with pytest.raises(RuntimeError, match="not installed"):
            m.submit_modal_run(
                plan, env={"MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"}
            )


class TestTrainCloudCli:
    def test_flags_in_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        out = _strip_ansi(CliRunner().invoke(app, ["train", "--help"]).stdout)
        flat = out.replace(" ", "")
        assert "--cloud" in flat
        assert "--gpu" in flat
        assert "cloud-submit" in flat

    def test_cloud_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "modal", "--gpu", "a100"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "soup_modal_app.py").exists()
        txt = _strip_ansi(result.output)
        assert "modal run" in txt
        assert "plan-only" in txt.lower()

    def test_cloud_unknown_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "aws"]
        )
        assert result.exit_code == 2

    def test_cloud_bad_gpu_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "modal", "--gpu", "v100"]
        )
        assert result.exit_code == 2

    def test_cloud_submit_via_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        import soup_cli.cloud.modal as m

        monkeypatch.setattr(m, "_MODAL_SUBMIT_OVERRIDE", lambda plan: 0)
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app,
            ["train", "--config", "soup.yaml", "--cloud", "modal",
             "--gpu", "h100", "--cloud-submit"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))


class TestCloudNoTopLevelModal:
    def test_no_top_level_modal_import(self):
        src = (REPO_ROOT / "src/soup_cli/cloud/modal.py").read_text(encoding="utf-8")
        assert "\nimport modal" not in src

    def test_modal_extra_in_pyproject(self):
        pp = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "modal = [" in pp
