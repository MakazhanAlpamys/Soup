"""v0.53.2 — Modality II live trainers.

Covers:
- #137 ``apply_reasoning_effort_prefix`` system-prompt injector + ``train_on_eot``
  parameter on :func:`build_assistant_only_labels` + ``build_format_row`` wiring.
- #135 Live ``apply_ebft_loss`` (structured / strided) and ``apply_gdpo_loss``
  (standard / length_normalized / margin) pure-tensor kernels +
  ``attach_ebft_compute_loss`` (SFT) / ``attach_gdpo_compute_loss`` (DPO) hooks.
- #133 ``DistillTrainerWrapper`` + ``build_distill_trainer`` factory +
  ``commands/train.py`` ``task='distill'`` routing.
- #132 ``ClassifierTrainerWrapper`` + ``build_classifier_trainer`` factory +
  ``commands/train.py`` ``task in {classifier, reranker, cross_encoder}`` routing.

Deferred to a follow-up patch:
- #71 ONNX export GPU-bound smoke — recorded in ``tests/qa/v053_qa.md``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# #137 — reasoning_effort prompt-prefix injector
# ---------------------------------------------------------------------------


class TestReasoningEffortPrefix:
    def test_inserts_system_message_when_none(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        messages = [{"role": "user", "content": "hi"}]
        out = apply_reasoning_effort_prefix(messages, "high")
        assert out[0]["role"] == "system"
        assert "<|reasoning_effort|>high<|/reasoning_effort|>" in out[0]["content"]
        # User message preserved at index 1
        assert out[1] == {"role": "user", "content": "hi"}

    def test_prepends_to_existing_system_message(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        out = apply_reasoning_effort_prefix(messages, "low")
        assert out[0]["role"] == "system"
        # Both the tag AND original content are present
        assert "<|reasoning_effort|>low<|/reasoning_effort|>" in out[0]["content"]
        assert "You are helpful." in out[0]["content"]
        # User message unchanged
        assert out[1] == {"role": "user", "content": "hi"}

    def test_does_not_mutate_input(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        original = [{"role": "user", "content": "hi"}]
        snapshot = [dict(m) for m in original]
        _ = apply_reasoning_effort_prefix(original, "medium")
        assert original == snapshot

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_accepts_canonical_levels(self, level: str) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        out = apply_reasoning_effort_prefix([{"role": "user", "content": "x"}], level)
        assert f"<|reasoning_effort|>{level}<|/reasoning_effort|>" in out[0]["content"]

    def test_case_insensitive_level(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        out = apply_reasoning_effort_prefix([{"role": "user", "content": "x"}], "HIGH")
        # Canonical lower-case is emitted
        assert "<|reasoning_effort|>high<|/reasoning_effort|>" in out[0]["content"]

    def test_rejects_unknown_level(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(ValueError, match="not supported"):
            apply_reasoning_effort_prefix([{"role": "user", "content": "x"}], "extreme")

    def test_rejects_bool_level(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(TypeError, match="must not be bool"):
            apply_reasoning_effort_prefix([{"role": "user", "content": "x"}], True)  # type: ignore[arg-type]

    def test_rejects_non_list_messages(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(TypeError, match="messages must be a list"):
            apply_reasoning_effort_prefix("hi", "low")  # type: ignore[arg-type]

    def test_rejects_empty_messages(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(ValueError, match="empty"):
            apply_reasoning_effort_prefix([], "low")

    def test_rejects_non_dict_message(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(TypeError, match="must be dict"):
            apply_reasoning_effort_prefix(["not a dict"], "low")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# #137 — train_on_eot loss-mask extension
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer with chat_template + EOS token for loss-mask tests.

    Renders each message as ``role:content<EOT>`` joined by a single space,
    where ``<EOT>`` is token id 9. Each character becomes a token id (its
    Unicode code point) so we can verify mask positions exactly. ``role:``
    prefix is the role's first char + colon (codepoints).
    """

    chat_template = "fake"
    eos_token = "<EOT>"
    eos_token_id = 9

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_assistant_tokens_mask: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Any:
        ids: list[int] = []
        mask: list[int] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            is_assistant = role == "assistant"
            chunk_ids = [ord(c) for c in f"{role[0]}:{content}"]
            ids.extend(chunk_ids)
            mask.extend([1 if is_assistant else 0] * len(chunk_ids))
            # EOT token after each message
            ids.append(self.eos_token_id)
            mask.append(1 if is_assistant else 0)
        if not tokenize:
            return "".join(chr(i) for i in ids)
        if return_assistant_tokens_mask:
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


class _EotBoundaryTokenizer:
    """Preferred-path fake whose assistant_mask EXCLUDES the trailing EOT.

    Mirrors a real chat template whose ``{% generation %}`` block wraps only
    the assistant content, leaving the EOT token outside the mask — which is
    exactly the case ``train_on_eot`` is designed to handle.
    """

    chat_template = "fake"
    eos_token = "<EOT>"
    eos_token_id = 9

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_assistant_tokens_mask: bool = False,
        return_dict: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Any:
        ids: list[int] = []
        mask: list[int] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            is_assistant = role == "assistant"
            chunk_ids = [ord(c) for c in f"{role[0]}:{content}"]
            ids.extend(chunk_ids)
            mask.extend([1 if is_assistant else 0] * len(chunk_ids))
            # EOT token after each message — ALWAYS unmasked (0) in this fake.
            ids.append(self.eos_token_id)
            mask.append(0)
        if not tokenize:
            return "".join(chr(i) for i in ids)
        if return_assistant_tokens_mask and return_dict:
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


class TestTrainOnEot:
    def test_include_eot_default_false_masks_eot(self) -> None:
        """Existing behaviour — EOT *outside* the assistant mask is IGNORE."""
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _EotBoundaryTokenizer()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        result = build_assistant_only_labels(messages, tok, max_length=64)
        labels = result["labels"]
        # Layout from _EotBoundaryTokenizer: u:hi<EOT>a:ok<EOT>
        # mask:                              0 0 0 0  0  0 0 1 1  0
        # With include_eot=False the trailing EOT (id=9) at the end is
        # IGNORE_INDEX.
        assert labels[-1] == IGNORE_INDEX

    def test_include_eot_true_extends_label_to_eos(self) -> None:
        from soup_cli.data.loss_mask import (
            IGNORE_INDEX,
            build_assistant_only_labels,
        )

        tok = _EotBoundaryTokenizer()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        without_eot = build_assistant_only_labels(messages, tok, max_length=64)
        with_eot = build_assistant_only_labels(
            messages, tok, max_length=64, include_eot=True
        )
        # With include_eot=True, the EOT (id=9) immediately following an
        # assistant span is KEPT (not IGNORE), so the unmasked count is
        # strictly higher by exactly the number of assistant turns.
        n_kept_no = sum(1 for x in without_eot["labels"] if x != IGNORE_INDEX)
        n_kept_yes = sum(1 for x in with_eot["labels"] if x != IGNORE_INDEX)
        assert n_kept_yes - n_kept_no == 1
        # Last token (the trailing EOT) is now KEPT.
        assert with_eot["labels"][-1] != IGNORE_INDEX
        assert with_eot["labels"][-1] == 9

    def test_include_eot_must_be_bool(self) -> None:
        from soup_cli.data.loss_mask import build_assistant_only_labels

        with pytest.raises(TypeError, match="include_eot must be bool"):
            build_assistant_only_labels(
                [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                _FakeTokenizer(),
                max_length=64,
                include_eot="yes",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# #135 — EBFT live loss kernel
# ---------------------------------------------------------------------------


def _torch_or_skip():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:  # pragma: no cover - CI without torch
        pytest.skip("torch not available")


class TestEbftLossLive:
    def test_structured_returns_finite_scalar(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        logits = torch.randn(2, 4, 8, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, -100], [3, 4, -100, -100]])
        loss = apply_ebft_loss(logits, labels, variant="structured", temperature=1.0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        # Gradient flows
        loss.backward()
        assert logits.grad is not None

    def test_strided_returns_finite_scalar(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        loss = apply_ebft_loss(
            logits, labels, variant="strided", temperature=0.5, stride=2
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_temperature_scales_loss(self) -> None:
        """Lower temperature sharpens the energy distribution → different loss."""
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        torch.manual_seed(0)
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[0, 1, 2]])
        hot = apply_ebft_loss(logits, labels, variant="structured", temperature=2.0)
        cold = apply_ebft_loss(logits, labels, variant="structured", temperature=0.5)
        assert not math.isclose(float(hot), float(cold), rel_tol=1e-6)

    def test_unknown_variant_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(ValueError, match="not supported"):
            apply_ebft_loss(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 1]]),
                variant="bogus",
                temperature=1.0,
            )

    def test_temperature_validated(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(ValueError):
            apply_ebft_loss(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 1]]),
                variant="structured",
                temperature=float("nan"),
            )

    def test_all_ignore_labels_returns_zero(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        logits = torch.randn(2, 3, 4)
        labels = torch.full((2, 3), -100, dtype=torch.int)
        loss = apply_ebft_loss(logits, labels, variant="structured", temperature=1.0)
        assert float(loss) == 0.0

    def test_shape_mismatch_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(ValueError, match="shape"):
            apply_ebft_loss(
                torch.randn(2, 4, 8),
                torch.tensor([[0, 1, 2]]),  # seq=3 vs logits seq=4
                variant="structured",
                temperature=1.0,
            )


# ---------------------------------------------------------------------------
# #135 — GDPO live loss kernel
# ---------------------------------------------------------------------------


class TestGdpoLossLive:
    def test_standard_returns_finite_scalar(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        pol_chosen = torch.tensor([-2.0, -1.5, -3.0], requires_grad=True)
        pol_rejected = torch.tensor([-3.5, -2.5, -4.0], requires_grad=True)
        ref_chosen = torch.tensor([-2.2, -1.8, -3.1])
        ref_rejected = torch.tensor([-3.2, -2.4, -3.9])
        loss = apply_gdpo_loss(
            policy_chosen_logps=pol_chosen,
            policy_rejected_logps=pol_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            variant="standard",
            beta=0.1,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert pol_chosen.grad is not None

    def test_length_normalized_uses_lengths(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        pol_chosen = torch.tensor([-10.0, -8.0])
        pol_rejected = torch.tensor([-12.0, -10.0])
        # Length-normalized uses chosen_lens / rejected_lens
        loss = apply_gdpo_loss(
            policy_chosen_logps=pol_chosen,
            policy_rejected_logps=pol_rejected,
            variant="length_normalized",
            beta=0.5,
            chosen_lens=torch.tensor([5.0, 4.0]),
            rejected_lens=torch.tensor([6.0, 5.0]),
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_margin_includes_margin_term(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        pol_chosen = torch.tensor([-2.0])
        pol_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-3.0])
        no_margin = apply_gdpo_loss(
            policy_chosen_logps=pol_chosen,
            policy_rejected_logps=pol_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            variant="margin",
            beta=0.1,
            margin=0.0,
        )
        with_margin = apply_gdpo_loss(
            policy_chosen_logps=pol_chosen,
            policy_rejected_logps=pol_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            variant="margin",
            beta=0.1,
            margin=1.0,
        )
        # Larger margin → larger loss (harder to satisfy)
        assert float(with_margin) > float(no_margin)

    def test_unknown_variant_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(ValueError, match="not supported"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                variant="bogus",
                beta=0.1,
            )

    def test_standard_requires_reference(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(ValueError, match="reference"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                variant="standard",
                beta=0.1,
            )

    def test_length_normalized_requires_lengths(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(ValueError, match="chosen_lens"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                variant="length_normalized",
                beta=0.1,
            )

    def test_beta_validated(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(ValueError, match="beta"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                ref_chosen_logps=t,
                ref_rejected_logps=t,
                variant="standard",
                beta=-0.1,
            )

    def test_beta_rejects_bool(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(TypeError, match="bool"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                ref_chosen_logps=t,
                ref_rejected_logps=t,
                variant="standard",
                beta=True,  # type: ignore[arg-type]
            )

    def test_shape_mismatch_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        with pytest.raises(ValueError, match="shape"):
            apply_gdpo_loss(
                policy_chosen_logps=torch.zeros(2),
                policy_rejected_logps=torch.zeros(3),
                ref_chosen_logps=torch.zeros(2),
                ref_rejected_logps=torch.zeros(2),
                variant="standard",
                beta=0.1,
            )


# ---------------------------------------------------------------------------
# Cross-cutting — stubs lifted
# ---------------------------------------------------------------------------


def test_apply_ebft_loss_no_longer_raises_not_implemented() -> None:
    torch = _torch_or_skip()
    from soup_cli.utils.ebft_gdpo import apply_ebft_loss

    # Should succeed (not raise NotImplementedError) — confirms stub lifted.
    apply_ebft_loss(
        torch.randn(1, 2, 3),
        torch.tensor([[0, 1]]),
        variant="structured",
        temperature=1.0,
    )


def test_apply_gdpo_loss_no_longer_raises_not_implemented() -> None:
    torch = _torch_or_skip()
    from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

    t = torch.zeros(2)
    apply_gdpo_loss(
        policy_chosen_logps=t,
        policy_rejected_logps=t,
        ref_chosen_logps=t,
        ref_rejected_logps=t,
        variant="standard",
        beta=0.1,
    )


# ---------------------------------------------------------------------------
# #135 — attach hooks (live wiring into SFT + DPO trainers)
# ---------------------------------------------------------------------------


class _StubTrainer:
    """Minimal trainer surface for hook tests — captures compute_loss calls."""

    def __init__(self) -> None:
        self.call_log: list[tuple] = []

        def _compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            self.call_log.append(("compute_loss", return_outputs))
            torch = _torch_or_skip()
            outputs = type("Out", (), {"logits": torch.zeros(1, 2, 3)})()
            loss = torch.tensor(0.5)
            return loss, outputs if return_outputs else loss

        self.compute_loss = _compute_loss


class TestAttachEbftComputeLoss:
    def test_no_op_when_variant_unset(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

        trainer = _StubTrainer()
        tcfg = type("Tcfg", (), {"ebft_variant": None})()
        assert attach_ebft_compute_loss(trainer, tcfg) is False

    def test_wraps_when_variant_set(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

        trainer = _StubTrainer()
        original = trainer.compute_loss
        tcfg = type(
            "Tcfg", (), {"ebft_variant": "structured", "ebft_temperature": 1.0}
        )()
        assert attach_ebft_compute_loss(trainer, tcfg) is True
        assert trainer.compute_loss is not original

        # Call the wrapped function — verify it returns a scalar tensor and
        # calls the original (loss should be CE + EBFT term).
        labels = torch.tensor([[0, 1]])
        loss = trainer.compute_loss(None, {"labels": labels}, return_outputs=False)
        assert torch.is_tensor(loss)
        # Original was called with return_outputs=True.
        assert trainer.call_log[-1] == ("compute_loss", True)

    def test_invalid_variant_rejected(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

        trainer = _StubTrainer()
        tcfg = type("Tcfg", (), {"ebft_variant": "bogus"})()
        with pytest.raises(ValueError, match="not supported"):
            attach_ebft_compute_loss(trainer, tcfg)


class _StubDpoTrainer:
    """Stand-in for TRL DPOTrainer's dpo_loss surface."""

    def __init__(self) -> None:
        def _dpo_loss(*args, **kwargs):
            return None  # Original returns 3-tuple; stub returns sentinel.

        self.dpo_loss = _dpo_loss


class TestAttachGdpoComputeLoss:
    def test_no_op_when_variant_unset(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = _StubDpoTrainer()
        tcfg = type("Tcfg", (), {"gdpo_variant": None})()
        assert attach_gdpo_compute_loss(trainer, tcfg) is False

    def test_no_op_when_trainer_lacks_dpo_loss(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = object()
        tcfg = type("Tcfg", (), {"gdpo_variant": "standard"})()
        assert attach_gdpo_compute_loss(trainer, tcfg) is False

    def test_wraps_and_returns_trl_shape(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = _StubDpoTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {"gdpo_variant": "standard", "dpo_beta": 0.1, "dpo_margin": 0.0},
        )()
        assert attach_gdpo_compute_loss(trainer, tcfg) is True

        pol_c = torch.tensor([-1.0, -2.0])
        pol_r = torch.tensor([-2.0, -3.0])
        ref_c = torch.tensor([-1.1, -2.1])
        ref_r = torch.tensor([-2.1, -3.1])
        losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
            pol_c, pol_r, ref_c, ref_r
        )
        assert losses.shape == pol_c.shape
        assert chosen_rewards.shape == pol_c.shape
        assert rejected_rewards.shape == pol_r.shape


# ---------------------------------------------------------------------------
# #133 — DistillTrainerWrapper + distill divergence kernel
# ---------------------------------------------------------------------------


class TestDistillDivergenceKernel:
    @pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
    def test_divergence_returns_finite_scalar(self, divergence: str) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        student = torch.randn(2, 4, 8, requires_grad=True)
        teacher = torch.randn(2, 4, 8)
        out = _compute_distill_term(student, teacher, divergence, temperature=2.0)
        assert out.ndim == 0
        assert torch.isfinite(out)
        out.backward()
        assert student.grad is not None

    def test_unknown_divergence_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        with pytest.raises(ValueError, match="Unknown divergence"):
            _compute_distill_term(
                torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), "bogus", 1.0
            )

    def test_identical_logits_zero_kl(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        logits = torch.randn(1, 3, 5)
        out = _compute_distill_term(logits, logits.clone(), "forward_kl", 1.0)
        assert abs(float(out)) < 1e-5


class TestDistillWrapper:
    def test_imports_cleanly(self) -> None:
        from soup_cli.trainer.distill import DistillTrainerWrapper  # noqa: F401

    def test_train_before_setup_raises(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.distill import DistillTrainerWrapper

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: distill
            training:
              teacher_model: sshleifer/tiny-gpt2
              distill_temperature: 2.0
              distill_divergence: forward_kl
            data:
              train: ./fake.jsonl
            """
        )
        wrapper = DistillTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="before setup"):
            wrapper.train()

    def test_build_distill_trainer_factory_lifted(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.distill import build_distill_trainer

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: distill
            training:
              teacher_model: sshleifer/tiny-gpt2
            data:
              train: ./fake.jsonl
            """
        )
        # Lifted from NotImplementedError in v0.53.2 — returns wrapper.
        wrapper = build_distill_trainer(cfg, device="cpu")
        from soup_cli.trainer.distill import DistillTrainerWrapper

        assert isinstance(wrapper, DistillTrainerWrapper)


# ---------------------------------------------------------------------------
# #132 — ClassifierTrainerWrapper + helpers
# ---------------------------------------------------------------------------


class TestClassifierWrapperHelpers:
    def test_row_to_text_with_text_field(self) -> None:
        from soup_cli.trainer.classifier import _row_to_text

        assert _row_to_text({"text": "hello"}) == "hello"

    def test_row_to_text_joins_messages(self) -> None:
        from soup_cli.trainer.classifier import _row_to_text

        out = _row_to_text(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            }
        )
        assert "a" in out and "b" in out

    def test_row_to_text_missing_field_raises(self) -> None:
        from soup_cli.trainer.classifier import _row_to_text

        with pytest.raises(ValueError, match="missing 'text'"):
            _row_to_text({"label": 0})

    def test_row_to_pair_text_ab(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        a, b = _row_to_pair({"text_a": "x", "text_b": "y"})
        assert (a, b) == ("x", "y")

    def test_row_to_pair_question_answer(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        a, b = _row_to_pair({"question": "q", "answer": "a"})
        assert (a, b) == ("q", "a")

    def test_row_to_pair_missing_raises(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        with pytest.raises(ValueError, match="text_a"):
            _row_to_pair({"text": "single"})

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_label_index_int_in_range(self, idx: int) -> None:
        from soup_cli.trainer.classifier import _label_index

        assert _label_index(idx, None, num_labels=3) == idx

    def test_label_index_int_out_of_range_rejected(self) -> None:
        from soup_cli.trainer.classifier import _label_index

        with pytest.raises(ValueError, match="out of range"):
            _label_index(5, None, num_labels=3)

    def test_label_index_string_via_label_names(self) -> None:
        from soup_cli.trainer.classifier import _label_index

        assert _label_index("pos", ["neg", "pos"], num_labels=2) == 1

    def test_label_index_string_without_names_rejected(self) -> None:
        from soup_cli.trainer.classifier import _label_index

        with pytest.raises(ValueError, match="label_names is unset"):
            _label_index("pos", None, num_labels=2)

    def test_label_index_bool_rejected(self) -> None:
        from soup_cli.trainer.classifier import _label_index

        # Project policy (v0.30.0 Candidate / v0.39.0 ReLoRAPolicy / v0.41.0
        # Part B): bool-as-int violations raise TypeError.
        with pytest.raises(TypeError, match="bool"):
            _label_index(True, None, num_labels=2)

    def test_normalise_label_multi_label_from_list(self) -> None:
        from soup_cli.trainer.classifier import _normalise_label

        vec = _normalise_label(
            [0, 2], label_names=None, num_labels=3, multi_label=True
        )
        assert vec == [1.0, 0.0, 1.0]


class TestClassifierWrapper:
    def test_imports_cleanly(self) -> None:
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper  # noqa: F401

    def test_train_before_setup_raises(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            training:
              num_labels: 3
            data:
              train: ./fake.jsonl
            """
        )
        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="before setup"):
            wrapper.train()

    def test_build_classifier_trainer_factory_lifted(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.classifier import build_classifier_trainer

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            training:
              num_labels: 2
            data:
              train: ./fake.jsonl
            """
        )
        wrapper = build_classifier_trainer(cfg, device="cpu")
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        assert isinstance(wrapper, ClassifierTrainerWrapper)


# ---------------------------------------------------------------------------
# commands/train.py routing — source-level audit
# ---------------------------------------------------------------------------


class TestTrainRouting:
    """Source-grep audit — looks for the *instantiation site* of each wrapper.

    Bare ``"DistillTrainerWrapper" in src`` would pass on a comment mentioning
    the class. ``"DistillTrainerWrapper(cfg, **trainer_kwargs)"`` requires the
    actual call expression to be present, which is much harder to satisfy by
    accident.
    """

    def test_distill_routed(self) -> None:
        from soup_cli.commands import train as train_cmd

        src = __import__("inspect").getsource(train_cmd)
        assert 'cfg.task == "distill"' in src
        # Require the actual instantiation expression, not just the bare name.
        assert "DistillTrainerWrapper(cfg, **trainer_kwargs)" in src

    def test_classifier_family_routed(self) -> None:
        from soup_cli.commands import train as train_cmd

        src = __import__("inspect").getsource(train_cmd)
        # Tuple membership in the if-branch is the load-bearing pattern.
        assert (
            'cfg.task in ("classifier", "reranker", "cross_encoder")' in src
        )
        # Instantiation expression — not just the class name.
        assert "ClassifierTrainerWrapper(cfg, **trainer_kwargs)" in src


# ---------------------------------------------------------------------------
# #137 — build_format_row wiring of reasoning_effort + train_on_eot
# ---------------------------------------------------------------------------


class _Tcfg:
    def __init__(
        self,
        reasoning_effort: Optional[str] = None,
        train_on_eot: bool = False,
    ) -> None:
        self.reasoning_effort = reasoning_effort
        self.train_on_eot = train_on_eot


def _make_data_cfg(
    train_on_responses_only: bool = True, chat_template: Optional[str] = None
):
    """Minimal duck-typed DataConfig for sft_format tests."""
    obj = type(
        "DataCfg",
        (),
        {
            "train_on_responses_only": train_on_responses_only,
            "train_on_messages_with_train_field": False,
            "max_length": 64,
            "chat_template": chat_template,
        },
    )()
    return obj


class TestFormatRowReasoningEffort:
    def test_no_reasoning_effort_passthrough(self) -> None:
        from soup_cli.data.sft_format import build_format_row

        format_row = build_format_row(
            tokenizer=_EotBoundaryTokenizer(),
            data_cfg=_make_data_cfg(),
            training_cfg=_Tcfg(reasoning_effort=None),
        )
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        out = format_row(row)
        # Should produce {input_ids, labels, attention_mask}
        assert set(out) >= {"input_ids", "labels", "attention_mask"}

    def test_reasoning_effort_injected_via_format_row(self) -> None:
        from soup_cli.data.sft_format import build_format_row

        captured_messages: list = []

        class _CapturingTok(_EotBoundaryTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                captured_messages.append([dict(m) for m in messages])
                return super().apply_chat_template(messages, **kwargs)

        format_row = build_format_row(
            tokenizer=_CapturingTok(),
            data_cfg=_make_data_cfg(),
            training_cfg=_Tcfg(reasoning_effort="high"),
        )
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        _ = format_row(row)
        # First captured render should have a system message with the tag.
        first = captured_messages[0]
        assert first[0]["role"] == "system"
        assert "<|reasoning_effort|>high<|/reasoning_effort|>" in first[0]["content"]

    def test_does_not_mutate_original_row(self) -> None:
        from soup_cli.data.sft_format import build_format_row

        format_row = build_format_row(
            tokenizer=_EotBoundaryTokenizer(),
            data_cfg=_make_data_cfg(),
            training_cfg=_Tcfg(reasoning_effort="low"),
        )
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        original_msgs = [dict(m) for m in row["messages"]]
        _ = format_row(row)
        assert row["messages"] == original_msgs


class TestFormatRowTrainOnEot:
    def test_train_on_eot_extends_loss_mask(self) -> None:
        from soup_cli.data.loss_mask import IGNORE_INDEX
        from soup_cli.data.sft_format import build_format_row

        tok = _EotBoundaryTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        without = build_format_row(
            tokenizer=tok,
            data_cfg=_make_data_cfg(),
            training_cfg=_Tcfg(train_on_eot=False),
        )(row)
        # Need a fresh tokenizer instance because the override is module-level
        tok2 = _EotBoundaryTokenizer()
        with_eot = build_format_row(
            tokenizer=tok2,
            data_cfg=_make_data_cfg(),
            training_cfg=_Tcfg(train_on_eot=True),
        )(row)
        n_no = sum(1 for x in without["labels"] if x != IGNORE_INDEX)
        n_yes = sum(1 for x in with_eot["labels"] if x != IGNORE_INDEX)
        assert n_yes - n_no == 1


# ---------------------------------------------------------------------------
# Schema gates — classifier and distill still validate
# ---------------------------------------------------------------------------


def test_distill_task_loads_with_teacher() -> None:
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(
        """
        base: sshleifer/tiny-gpt2
        task: distill
        training:
          teacher_model: sshleifer/tiny-gpt2
          distill_temperature: 2.0
          distill_divergence: forward_kl
        data:
          train: ./fake.jsonl
        """
    )
    assert cfg.task == "distill"
    assert cfg.training.teacher_model == "sshleifer/tiny-gpt2"


def test_classifier_task_requires_num_labels() -> None:
    from soup_cli.config.loader import load_config_from_string

    # ``load_config_from_string`` re-raises pydantic validation as ValueError.
    with pytest.raises(ValueError, match="num_labels"):
        load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            data:
              train: ./fake.jsonl
            """
        )


# ---------------------------------------------------------------------------
# Review-fix coverage (v0.53.2 TDD + security review followups)
# ---------------------------------------------------------------------------


class TestEbftAttachLabelsNone:
    """tdd-review C1 — wrapped compute_loss must tolerate inputs without labels."""

    def test_labels_missing_returns_ce_only(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

        trainer = _StubTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {"ebft_variant": "structured", "ebft_temperature": 1.0},
        )()
        attach_ebft_compute_loss(trainer, tcfg)
        # No "labels" key — wrapper must fall through to CE without crashing.
        loss = trainer.compute_loss(None, {}, return_outputs=False)
        assert torch.is_tensor(loss)


class TestEbftAttachIdempotent:
    def test_double_wrap_is_no_op(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_ebft_compute_loss

        trainer = _StubTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {"ebft_variant": "structured", "ebft_temperature": 1.0},
        )()
        first = attach_ebft_compute_loss(trainer, tcfg)
        second = attach_ebft_compute_loss(trainer, tcfg)
        assert first is True
        assert second is False
        assert getattr(trainer, "_soup_ebft_wrapped", False) is True


class TestGdpoAttachIdempotent:
    def test_double_wrap_is_no_op(self) -> None:
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = _StubDpoTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {"gdpo_variant": "standard", "dpo_beta": 0.1, "dpo_margin": 0.0},
        )()
        first = attach_gdpo_compute_loss(trainer, tcfg)
        second = attach_gdpo_compute_loss(trainer, tcfg)
        assert first is True
        assert second is False
        assert getattr(trainer, "_soup_gdpo_wrapped", False) is True


class TestGdpoAttachLengthNormalizedForwardsLens:
    """python-review MEDIUM — length_normalized requires chosen_lens/rejected_lens."""

    def test_length_normalized_via_kwargs(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = _StubDpoTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {
                "gdpo_variant": "length_normalized",
                "dpo_beta": 0.5,
                "dpo_margin": 0.0,
            },
        )()
        attach_gdpo_compute_loss(trainer, tcfg)

        pol_c = torch.tensor([-10.0, -8.0])
        pol_r = torch.tensor([-12.0, -10.0])
        ref_c = torch.zeros(2)
        ref_r = torch.zeros(2)
        losses, _, _ = trainer.dpo_loss(
            pol_c,
            pol_r,
            ref_c,
            ref_r,
            chosen_lens=torch.tensor([5.0, 4.0]),
            rejected_lens=torch.tensor([6.0, 5.0]),
        )
        assert losses.shape == pol_c.shape

    def test_length_normalized_via_positional(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import attach_gdpo_compute_loss

        trainer = _StubDpoTrainer()
        tcfg = type(
            "Tcfg",
            (),
            {
                "gdpo_variant": "length_normalized",
                "dpo_beta": 0.5,
                "dpo_margin": 0.0,
            },
        )()
        attach_gdpo_compute_loss(trainer, tcfg)

        pol_c = torch.tensor([-10.0, -8.0])
        pol_r = torch.tensor([-12.0, -10.0])
        # Positional chosen_lens / rejected_lens (newer TRL signatures may
        # pass these positionally).
        losses, _, _ = trainer.dpo_loss(
            pol_c,
            pol_r,
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([5.0, 4.0]),
            torch.tensor([6.0, 5.0]),
        )
        assert losses.shape == pol_c.shape


class TestExtendMaskIdempotency:
    """tdd-review C2 — _extend_mask_to_eot must be idempotent."""

    def test_second_pass_no_op(self) -> None:
        from soup_cli.data.loss_mask import _extend_mask_to_eot

        ids = [10, 11, 9, 20, 21, 9]
        mask = [1, 1, 0, 0, 0, 0]
        once = _extend_mask_to_eot(ids, mask, eos_token_id=9)
        twice = _extend_mask_to_eot(ids, once, eos_token_id=9)
        assert once == twice

    def test_leading_eos_not_marked(self) -> None:
        from soup_cli.data.loss_mask import _extend_mask_to_eot

        ids = [9, 10, 11, 9]
        mask = [0, 1, 1, 0]
        out = _extend_mask_to_eot(ids, mask, eos_token_id=9)
        assert out[0] == 0  # leading EOS not absorbed
        assert out[3] == 1  # trailing EOS absorbed

    def test_two_assistant_spans_both_get_eos(self) -> None:
        from soup_cli.data.loss_mask import _extend_mask_to_eot

        ids = [1, 2, 9, 9, 3, 4, 9]
        mask = [0, 1, 0, 0, 0, 1, 0]
        out = _extend_mask_to_eot(ids, mask, eos_token_id=9)
        assert out[2] == 1  # consecutive EOS absorbed
        assert out[3] == 1
        assert out[6] == 1


class TestResolveEosTokenId:
    """python-review MEDIUM — handle list/str/None/bool eos_token_id."""

    def test_int(self) -> None:
        from soup_cli.data.loss_mask import _resolve_eos_token_id

        class T:
            eos_token_id = 9

        assert _resolve_eos_token_id(T()) == 9

    def test_list_picks_first_int(self) -> None:
        from soup_cli.data.loss_mask import _resolve_eos_token_id

        class T:
            eos_token_id = [128001, 128009]  # Llama 3 style

        assert _resolve_eos_token_id(T()) == 128001

    def test_str_returns_none(self) -> None:
        from soup_cli.data.loss_mask import _resolve_eos_token_id

        class T:
            eos_token_id = "9"

        assert _resolve_eos_token_id(T()) is None

    def test_none_returns_none(self) -> None:
        from soup_cli.data.loss_mask import _resolve_eos_token_id

        class T:
            pass

        assert _resolve_eos_token_id(T()) is None

    def test_bool_returns_none(self) -> None:
        from soup_cli.data.loss_mask import _resolve_eos_token_id

        class T:
            eos_token_id = True

        assert _resolve_eos_token_id(T()) is None


class TestComputeDistillTermValidation:
    """python-review MEDIUM / tdd-review H1 — temperature validation in kernel."""

    def test_zero_temperature_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        with pytest.raises(ValueError, match="positive"):
            _compute_distill_term(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                "forward_kl",
                0.0,
            )

    def test_negative_temperature_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        with pytest.raises(ValueError, match="positive"):
            _compute_distill_term(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                "forward_kl",
                -1.0,
            )

    def test_nan_temperature_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        with pytest.raises(ValueError, match="finite"):
            _compute_distill_term(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                "forward_kl",
                float("nan"),
            )

    def test_bool_temperature_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.trainer.distill import _compute_distill_term

        with pytest.raises(TypeError, match="bool"):
            _compute_distill_term(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                "forward_kl",
                True,  # type: ignore[arg-type]
            )


class TestReasoningEffortNullByte:
    """tdd-review H3 — null-byte in level rejected."""

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

        with pytest.raises(ValueError, match="null"):
            apply_reasoning_effort_prefix(
                [{"role": "user", "content": "x"}], "lo\x00w"
            )


class TestEbftStrideValidation:
    """tdd-review M2 — stride bool / non-positive rejected."""

    def test_stride_bool_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(TypeError, match="stride"):
            apply_ebft_loss(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 1]]),
                variant="strided",
                temperature=1.0,
                stride=True,  # type: ignore[arg-type]
            )

    def test_stride_zero_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(ValueError, match="stride"):
            apply_ebft_loss(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 1]]),
                variant="strided",
                temperature=1.0,
                stride=0,
            )

    def test_stride_negative_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss

        with pytest.raises(ValueError, match="stride"):
            apply_ebft_loss(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 1]]),
                variant="strided",
                temperature=1.0,
                stride=-1,
            )


class TestGdpoMarginValidation:
    """tdd-review M3 — margin bool / NaN rejected."""

    def test_margin_bool_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(TypeError, match="margin"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                ref_chosen_logps=t,
                ref_rejected_logps=t,
                variant="margin",
                beta=0.1,
                margin=True,  # type: ignore[arg-type]
            )

    def test_margin_nan_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        t = torch.zeros(2)
        with pytest.raises(ValueError, match="finite"):
            apply_gdpo_loss(
                policy_chosen_logps=t,
                policy_rejected_logps=t,
                ref_chosen_logps=t,
                ref_rejected_logps=t,
                variant="margin",
                beta=0.1,
                margin=float("nan"),
            )


class TestGdpoRefShapeMismatch:
    """tdd-review M4 — ref logps shape mismatch vs policy."""

    def test_ref_shape_mismatch_rejected(self) -> None:
        torch = _torch_or_skip()
        from soup_cli.utils.ebft_gdpo import apply_gdpo_loss

        with pytest.raises(ValueError, match="shape"):
            apply_gdpo_loss(
                policy_chosen_logps=torch.zeros(3),
                policy_rejected_logps=torch.zeros(3),
                ref_chosen_logps=torch.zeros(2),
                ref_rejected_logps=torch.zeros(3),
                variant="standard",
                beta=0.1,
            )


class TestRowToTextRejectsNonStrContent:
    """security-review M3 — non-str content raises rather than silent skip."""

    def test_non_str_content_raises(self) -> None:
        from soup_cli.trainer.classifier import _row_to_text

        with pytest.raises(TypeError, match="must be str"):
            _row_to_text(
                {
                    "messages": [
                        {"role": "user", "content": ["multi", "modal"]},
                    ]
                }
            )

    def test_non_dict_message_silently_skipped(self) -> None:
        from soup_cli.trainer.classifier import _row_to_text

        out = _row_to_text(
            {"messages": ["not-a-dict", {"role": "user", "content": "real"}]}
        )
        assert "real" in out


class TestRowToPairRejectsNonStr:
    """security-review M4 — unchecked str() coercion fix."""

    def test_non_str_text_a_rejected(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        with pytest.raises(TypeError, match="text_a"):
            _row_to_pair({"text_a": {"d": "ict"}, "text_b": "y"})

    def test_non_str_text_b_rejected(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        with pytest.raises(TypeError, match="text_b"):
            _row_to_pair({"text_a": "x", "text_b": [1, 2]})

    def test_non_str_question_rejected(self) -> None:
        from soup_cli.trainer.classifier import _row_to_pair

        with pytest.raises(TypeError, match="question"):
            _row_to_pair({"question": 42, "answer": "ok"})


class TestLabelIndexFurtherCoverage:
    """tdd-review M5 / M6 / M9."""

    def test_single_label_list_raises(self) -> None:
        from soup_cli.trainer.classifier import _normalise_label

        with pytest.raises(TypeError):
            _normalise_label(
                [0, 1], label_names=None, num_labels=3, multi_label=False
            )

    def test_string_not_in_label_names(self) -> None:
        from soup_cli.trainer.classifier import _label_index

        with pytest.raises(ValueError, match="not in training.label_names"):
            _label_index("unknown", ["pos", "neg"], num_labels=2)

    @pytest.mark.parametrize("bad", [None, 1.5, object()])
    def test_unsupported_type_rejected(self, bad: object) -> None:
        from soup_cli.trainer.classifier import _label_index

        with pytest.raises(TypeError, match="must be int"):
            _label_index(bad, None, num_labels=3)


class TestMultiLabelListCap:
    """security-review H2 — uncapped multi-label list DoS defense."""

    def test_oversize_list_rejected(self) -> None:
        from soup_cli.trainer.classifier import _normalise_label

        # _MAX_MULTI_LABEL_ENTRIES is 1024.
        big = [0] * 2000
        with pytest.raises(ValueError, match="too int"):
            _normalise_label(
                big, label_names=None, num_labels=3, multi_label=True
            )


class TestFactoryUnknownKwarg:
    """tdd-review L1 — factories must reject unknown kwargs loudly."""

    def test_build_distill_trainer_unknown_kwarg(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.distill import build_distill_trainer

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: distill
            training:
              teacher_model: sshleifer/tiny-gpt2
            data:
              train: ./fake.jsonl
            """
        )
        with pytest.raises(TypeError):
            build_distill_trainer(cfg, device="cpu", nonexistent=True)

    def test_build_classifier_trainer_unknown_kwarg(self) -> None:
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.classifier import build_classifier_trainer

        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            training:
              num_labels: 2
            data:
              train: ./fake.jsonl
            """
        )
        with pytest.raises(TypeError):
            build_classifier_trainer(cfg, device="cpu", nonexistent=True)


class TestDistillSourceLevelGuards:
    """Regression guards for v0.53.2 wave 3 smoke-surfaced bugs."""

    def test_uses_seq2seq_collator(self) -> None:
        """``DataCollatorForLanguageModeling`` does NOT pad labels — would crash
        on variable-length pre-tokenised rows. v0.53.2 wave 3 switched to
        ``DataCollatorForSeq2Seq``. Check the import statement specifically
        (the deprecated class name still appears in an explanatory comment)."""
        import inspect

        from soup_cli.trainer import distill as distill_mod

        src = inspect.getsource(distill_mod)
        # The actual import line must reference the Seq2Seq collator.
        assert "from transformers import DataCollatorForSeq2Seq" in src
        # No import of the wrong collator.
        assert "from transformers import DataCollatorForLanguageModeling" not in src

    def test_compute_loss_bridges_teacher_device(self) -> None:
        """Bug surfaced during CPU smoke on a CUDA-capable box: HF Trainer
        auto-moved the student to CUDA while the teacher stayed where loaded.
        compute_loss must bridge teacher inputs onto the teacher's device and
        teacher_logits back onto the student's device."""
        import inspect

        from soup_cli.trainer import distill as distill_mod

        src = inspect.getsource(distill_mod)
        assert "teacher_device = next(teacher_ref.parameters()).device" in src
        assert ".to(student_logits.device)" in src


# ---------------------------------------------------------------------------
# Step 6f — failure-mode smoke: every cross-validator triggered by bad YAML
# ---------------------------------------------------------------------------


class TestFailureModeSmoke:
    """One YAML per cross-validator path the v0.53.2 surfaces add or trigger.

    Goal: prove the validators name the actual problem so users grep
    successfully, not "ValidationError" generic.
    """

    def test_classifier_missing_num_labels(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="num_labels"):
            load_config_from_string(
                "base: x\ntask: classifier\ndata:\n  train: ./f.jsonl\n"
            )

    def test_classifier_label_names_length_mismatch(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="num_labels"):
            load_config_from_string(
                "base: x\ntask: classifier\ntraining:\n"
                "  num_labels: 3\n  label_names: [a, b]\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_classifier_fields_outside_classifier_task(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="num_labels|classifier"):
            load_config_from_string(
                "base: x\ntask: sft\ntraining:\n  num_labels: 3\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_distill_missing_teacher(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="teacher_model"):
            load_config_from_string(
                "base: x\ntask: distill\ndata:\n  train: ./f.jsonl\n"
            )

    def test_distill_fields_outside_distill_task(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        # teacher_model set on a non-distill task → rejected with named field.
        with pytest.raises(ValueError, match="teacher_model|distill"):
            load_config_from_string(
                "base: x\ntask: sft\ntraining:\n  teacher_model: y\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_reasoning_effort_on_non_sft_family_task(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        # reasoning_effort is SFT-family-only; reject on grpo.
        with pytest.raises(ValueError, match="reasoning_effort"):
            load_config_from_string(
                "base: x\ntask: grpo\ntraining:\n  reasoning_effort: high\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_train_on_eot_on_non_sft_family_task(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="train_on_eot"):
            load_config_from_string(
                "base: x\ntask: grpo\ntraining:\n  train_on_eot: true\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_ebft_temperature_without_variant(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="ebft_variant"):
            load_config_from_string(
                "base: x\ntask: sft\ntraining:\n  ebft_temperature: 1.0\n"
                "data:\n  train: ./f.jsonl\n"
            )

    def test_gdpo_variant_on_sft_rejected(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="gdpo_variant"):
            load_config_from_string(
                "base: x\ntask: sft\ntraining:\n  gdpo_variant: standard\n"
                "data:\n  train: ./f.jsonl\n"
            )
