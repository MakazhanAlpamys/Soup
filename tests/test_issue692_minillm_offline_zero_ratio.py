"""Regression tests for issue #692: offline MiniLLM mix 0 is a silent no-op."""

from __future__ import annotations

import types

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig


def _distill_yaml(**training: object) -> str:
    extra = "".join(f"  {key}: {value}\n" for key, value in training.items())
    return f"""
base: hf-internal-testing/tiny-random-gpt2
task: distill
data:
  train: data.jsonl
  format: chatml
training:
  teacher_model: hf-internal-testing/tiny-random-gpt2
{extra}output: ./out
"""


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
            hidden_states = self.emb(input_ids)
            logits = self.head(hidden_states)
            return types.SimpleNamespace(logits=logits)

    return _FakeLM()


class TestOfflineMiniLLMZeroMixRejected:
    def test_enabled_offline_default_mix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="minillm_teacher_mix_ratio"):
            load_config_from_string(_distill_yaml(minillm_enabled=True))

    def test_error_names_on_policy_and_enabled(self) -> None:
        with pytest.raises(ValueError, match="minillm_on_policy"):
            load_config_from_string(_distill_yaml(minillm_enabled=True))
        with pytest.raises(ValueError, match="minillm_enabled"):
            load_config_from_string(_distill_yaml(minillm_enabled=True))

    def test_enabled_offline_nonzero_mix_is_accepted(self) -> None:
        cfg = load_config_from_string(
            _distill_yaml(minillm_enabled=True, minillm_teacher_mix_ratio=0.3)
        )
        assert cfg.training.minillm_enabled is True
        assert cfg.training.minillm_on_policy is False
        assert cfg.training.minillm_teacher_mix_ratio == 0.3

    def test_on_policy_zero_mix_is_accepted(self) -> None:
        cfg = load_config_from_string(
            _distill_yaml(minillm_enabled=True, minillm_on_policy=True)
        )
        assert cfg.training.minillm_on_policy is True
        assert cfg.training.minillm_teacher_mix_ratio == 0.0

    def test_help_distinguishes_offline_blend_from_on_policy_sampling(self) -> None:
        help_text = TrainingConfig.model_fields["minillm_teacher_mix_ratio"].description
        assert help_text is not None
        assert "stopgrad" in help_text
        assert "student-only sampling" in help_text
        assert "KL(student || teacher)" in help_text


class TestOfflineMiniLLMTeacherSignal:
    def test_nonzero_offline_mix_changes_when_teacher_changes(self) -> None:
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_distill_term

        torch.manual_seed(0)
        student = torch.randn(2, 4, 8, requires_grad=True)
        teacher_a = torch.randn(2, 4, 8)
        # Softmax is shift-invariant, so a constant add is a no-op. Scale
        # changes the teacher distribution and therefore the reverse-KL.
        teacher_b = teacher_a * 4.0
        labels = torch.ones(2, 4, dtype=torch.long)
        cfg = MiniLLMConfig(teacher_mix_ratio=0.3)
        loss_a = minillm_distill_term(student, teacher_a, labels, config=cfg)
        loss_b = minillm_distill_term(student, teacher_b, labels, config=cfg)
        assert abs(loss_a.item() - loss_b.item()) > 1e-4
        assert abs(loss_a.item()) > 1e-6
        assert abs(loss_b.item()) > 1e-6

    def test_zero_offline_mix_ignores_the_teacher(self) -> None:
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_distill_term

        torch.manual_seed(1)
        student = torch.randn(2, 4, 8, requires_grad=True)
        teacher_a = torch.randn(2, 4, 8)
        teacher_b = teacher_a + 5.0
        labels = torch.ones(2, 4, dtype=torch.long)
        cfg = MiniLLMConfig(teacher_mix_ratio=0.0)
        loss_a = minillm_distill_term(student, teacher_a, labels, config=cfg)
        loss_b = minillm_distill_term(student, teacher_b, labels, config=cfg)
        assert loss_a.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_b.item() == pytest.approx(0.0, abs=1e-6)

    def test_on_policy_zero_mix_still_uses_teacher(self) -> None:
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.minillm import MiniLLMConfig, minillm_on_policy_rollout

        student = _make_fake_lm(seed=1)
        teacher_a = _make_fake_lm(seed=2)
        teacher_b = _make_fake_lm(seed=99)
        for param in list(teacher_a.parameters()) + list(teacher_b.parameters()):
            param.requires_grad_(False)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        cfg = MiniLLMConfig(teacher_mix_ratio=0.0, on_policy=True)

        def _greedy(probs):
            return probs.argmax(dim=-1, keepdim=True)

        loss_a, _ = minillm_on_policy_rollout(
            student,
            teacher_a,
            ids,
            mask,
            config=cfg,
            max_new_tokens=3,
            temperature=1.0,
            sample_fn=_greedy,
        )
        loss_b, _ = minillm_on_policy_rollout(
            student,
            teacher_b,
            ids,
            mask,
            config=cfg,
            max_new_tokens=3,
            temperature=1.0,
            sample_fn=_greedy,
        )
        assert torch.isfinite(loss_a).item()
        assert torch.isfinite(loss_b).item()
        assert loss_a.item() != pytest.approx(loss_b.item())
