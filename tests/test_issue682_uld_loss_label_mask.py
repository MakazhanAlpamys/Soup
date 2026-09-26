"""ULD loss must respect the response-only label mask and causal shift (#682).

The SFT cross-entropy term (``_compute_distill_term`` / the CE branch in
``DistillTrainerWrapper.setup``'s ``compute_loss``) restricts itself to
``labels != -100`` after a causal shift (logit position i predicts token
i+1), which excludes prompt tokens under ``train_on_responses_only`` (the
default) and the final, unsupervisable position. The ULD branch
(``wasserstein`` / ``topk_align`` / ``wasserstein_aligned``) received only
``attention_mask``, so it trained on every attended position instead,
including prompt tokens and the shift-boundary position the CE term
explicitly drops.

These tests pin the fix at the tensor level: a position excluded by the
shifted label mask must not move the loss, and a supervised position must.
Perturbations touch a SINGLE vocab entry, never a whole position vector:
adding a constant to every logit at a position is a softmax no-op, which
would make an "excluded position" assertion trivially true either way.
"""

from __future__ import annotations

import pytest


class TestUldDistillLossLabelMask:
    def test_prompt_position_excluded_from_wasserstein_loss(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=6, teacher_vocab_size=6)
        torch.manual_seed(0)
        s = torch.randn(1, 4, 6)
        t = torch.randn(1, 4, 6)
        # Shift drops position 3; shifted labels = [-100, 5, 2], so position 0
        # (predicting label index 1 == -100) is excluded and positions 1-2
        # (predicting labels 5 and 2) are supervised.
        labels = torch.tensor([[-100, -100, 5, 2]])

        base = uld_distill_loss(s, t, config=cfg, labels=labels)

        excluded = s.clone()
        excluded[0, 0, 0] += 50.0  # position 0 predicts labels[1] == -100
        assert torch.allclose(base, uld_distill_loss(excluded, t, config=cfg, labels=labels))

        included = s.clone()
        included[0, 1, 0] += 50.0  # position 1 predicts labels[2] == 5
        assert not torch.allclose(base, uld_distill_loss(included, t, config=cfg, labels=labels))

    def test_final_position_excluded_by_causal_shift(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=6, teacher_vocab_size=6)
        torch.manual_seed(1)
        s = torch.randn(1, 3, 6)
        t = torch.randn(1, 3, 6)
        labels = torch.tensor([[-100, 5, 2]])  # every real label is supervised

        base = uld_distill_loss(s, t, config=cfg, labels=labels)
        last = s.clone()
        last[0, 2, 0] += 50.0  # last position predicts nothing; CE drops it too
        assert torch.allclose(base, uld_distill_loss(last, t, config=cfg, labels=labels))

        earlier = s.clone()
        earlier[0, 1, 0] += 50.0  # position 1 predicts labels[2] == 2, kept
        assert not torch.allclose(base, uld_distill_loss(earlier, t, config=cfg, labels=labels))

    def test_topk_align_respects_the_same_mask(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(
            strategy="topk_align", student_vocab_size=6, teacher_vocab_size=6, top_k=3
        )
        torch.manual_seed(2)
        s = torch.randn(1, 4, 6)
        t = torch.randn(1, 4, 6)
        labels = torch.tensor([[-100, -100, 5, 2]])

        base = uld_distill_loss(s, t, config=cfg, labels=labels)

        excluded = s.clone()
        excluded[0, 0, 0] += 50.0
        assert torch.allclose(base, uld_distill_loss(excluded, t, config=cfg, labels=labels))

        included = s.clone()
        included[0, 1, 0] += 50.0
        assert not torch.allclose(base, uld_distill_loss(included, t, config=cfg, labels=labels))

    def test_fully_masked_batch_is_finite_and_zero(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=5, teacher_vocab_size=5)
        s = torch.randn(1, 3, 5, requires_grad=True)
        t = torch.randn(1, 3, 5)
        labels = torch.full((1, 3), -100)

        loss = uld_distill_loss(s, t, config=cfg, labels=labels)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)
        loss.backward()
        assert torch.all(s.grad == 0)

    def test_no_labels_or_mask_keeps_prior_unmasked_behavior(self):
        """No labels/attention_mask given: unchanged from before #682."""
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_distill_loss

        cfg = ULDConfig(strategy="wasserstein", student_vocab_size=6, teacher_vocab_size=6)
        torch.manual_seed(3)
        s = torch.randn(1, 4, 6)
        t = torch.randn(1, 4, 6)
        with_last = uld_distill_loss(s, t, config=cfg)
        # No shift applied, so perturbing the final position DOES move the
        # loss when there is no labels/attention_mask to trigger the shift.
        perturbed = s.clone()
        perturbed[0, 3, 0] += 50.0
        assert not torch.allclose(with_last, uld_distill_loss(perturbed, t, config=cfg))


class TestUldAlignedLossLabelMask:
    def test_aligned_strategy_applies_the_label_mask(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned", student_vocab_size=5, teacher_vocab_size=5
        )
        torch.manual_seed(4)
        s = torch.randn(1, 3, 5)
        t = torch.randn(1, 3, 5)
        tokens = [["a", "b", "c"]]  # identical decode -> exact offset alignment
        # Shift drops raw position 2; shifted labels = [7, -100], so only
        # raw position 0 (predicting raw label index 1 == 7) is supervised.
        labels = torch.tensor([[-100, 7, -100]])

        base = uld_aligned_loss(s, t, tokens, tokens, config=cfg, labels=labels)

        included = s.clone()
        included[0, 0, 0] += 50.0
        assert not torch.allclose(
            base, uld_aligned_loss(included, t, tokens, tokens, config=cfg, labels=labels)
        )

        excluded = s.clone()
        excluded[0, 1, 0] += 50.0
        assert torch.allclose(
            base, uld_aligned_loss(excluded, t, tokens, tokens, config=cfg, labels=labels)
        )

    def test_attention_mask_only_still_shifts_and_masks(self):
        """No labels: attention_mask alone still drops the final position."""
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned", student_vocab_size=5, teacher_vocab_size=5
        )
        torch.manual_seed(5)
        s = torch.randn(1, 3, 5)
        t = torch.randn(1, 3, 5)
        tokens = [["a", "b", "c"]]
        mask = torch.tensor([[1, 1, 0]])  # position 2 is padding

        base = uld_aligned_loss(s, t, tokens, tokens, config=cfg, attention_mask=mask)
        last = s.clone()
        last[0, 2, 0] += 50.0  # dropped by the shift regardless of the mask
        moved = uld_aligned_loss(last, t, tokens, tokens, config=cfg, attention_mask=mask)
        assert torch.allclose(base, moved)

    def test_fully_masked_batch_is_finite_and_zero(self):
        pytest.importorskip("torch")
        import torch

        from soup_cli.utils.uld import ULDConfig, uld_aligned_loss

        cfg = ULDConfig(
            strategy="wasserstein_aligned", student_vocab_size=5, teacher_vocab_size=5
        )
        s = torch.randn(1, 3, 5, requires_grad=True)
        t = torch.randn(1, 3, 5)
        tokens = [["a", "b", "c"]]
        labels = torch.full((1, 3), -100)

        loss = uld_aligned_loss(s, t, tokens, tokens, config=cfg, labels=labels)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)


class TestDistillPassesLabelsToUld:
    """Wiring: both ULD call sites in compute_loss forward ``labels`` (#682)."""

    def test_source_forwards_labels_to_both_uld_call_sites(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "src/soup_cli/trainer/distill.py").read_text(encoding="utf-8")
        # A bounded window, not split(")")[0]: the call's own arguments
        # contain a closing paren (`inputs.get("attention_mask")`), so
        # splitting on the first ")" cuts the window before "labels=labels".
        window = 400
        uld_at = src.index("distill_loss = _uld_projection(")
        assert "labels=labels" in src[uld_at : uld_at + window]
        aligned_at = src.index("distill_loss = uld_aligned_loss(")
        assert "labels=labels" in src[aligned_at : aligned_at + window]
