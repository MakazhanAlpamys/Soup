"""Tests for sequence-level Group Sequence Policy Optimization (GSPO) — Issue #723.

Validates the published sequence-level objective (Qwen, arXiv:2507.18071):
1. Masked token mutation invariance (masked tokens do not shift loss).
2. Exact zero gradient at all masked completion tokens.
3. Reference mathematical calculation parity with variable completion lengths.
4. Batch row permutation invariance.
5. Padding position invariance (left-padding vs right-padding).
6. Clip boundary regimes (unclipped, upper-clipped with A > 0, lower-clipped with A < 0).
7. All-masked tokens / rows numerical stability (no NaN / ZeroDivisionError).
8. Custom delta parameter for clipping radius.
9. Advantage shape compatibility (1D [B], 2D column [B, 1], 2D broadcast [B, T]).
10. Runtime GRPOTrainer variant execution without fallback warnings.
"""

from __future__ import annotations

import math

import pytest


def _torch_or_skip():
    return pytest.importorskip("torch")


class TestSequenceLevelGSPO:
    def test_masked_token_mutation_leaves_loss_invariant(self) -> None:
        """Mutating a masked token's log-ratio must not change the GSPO loss (#723)."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        torch.manual_seed(42)
        batch, seq = 4, 6
        logp_old = torch.randn(batch, seq)
        adv = torch.tensor([1.5, -0.8, 0.4, -1.2])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        base_logp_new = logp_old.clone() + 0.1
        loss_baseline = apply_variant_loss(
            "gspo",
            logp_new=base_logp_new,
            logp_old=logp_old,
            advantages=adv,
            completion_mask=mask,
        )

        for mutant_val in (-100.0, -10.0, 0.0, 25.0, 1000.0):
            mutant_logp_new = base_logp_new.clone()
            # Mutate only masked positions (row 0 col 4, row 1 col 3, row 3 col 5)
            mutant_logp_new[0, 4] = mutant_val
            mutant_logp_new[1, 3] = mutant_val
            mutant_logp_new[3, 5] = mutant_val

            loss_mutant = apply_variant_loss(
                "gspo",
                logp_new=mutant_logp_new,
                logp_old=logp_old,
                advantages=adv,
                completion_mask=mask,
            )
            assert abs(loss_baseline.item() - loss_mutant.item()) < 1e-6

    def test_masked_tokens_receive_zero_gradient(self) -> None:
        """Every masked token position must receive strictly zero gradient (#723)."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        torch.manual_seed(101)
        batch, seq = 3, 5
        logp_old = torch.randn(batch, seq)
        adv = torch.tensor([2.0, -1.0, 0.5])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ])

        logp_new = (logp_old + 0.05).detach().clone().requires_grad_(True)
        loss = apply_variant_loss(
            "gspo",
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=adv,
            completion_mask=mask,
        )
        loss.backward()

        assert logp_new.grad is not None
        # Unmasked tokens must receive non-zero gradients
        assert (logp_new.grad[mask == 1.0].abs() > 0.0).all()
        # All masked positions must have exactly zero gradient
        assert (logp_new.grad[mask == 0.0] == 0.0).all()

    def test_reference_sequence_objective_matches_manual_calculation(self) -> None:
        """Loss must match a direct reference calculation of length-normalized GSPO (#723)."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new = torch.tensor([
            [1.2, 0.8, -0.4, 0.0],
            [-0.5, 0.3, 0.0, 0.0],
        ])
        logp_old = torch.tensor([
            [1.0, 0.5, -0.2, 0.0],
            [-0.7, 0.1, 0.0, 0.0],
        ])
        adv = torch.tensor([1.2, -0.8])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],  # length 3
            [1.0, 1.0, 0.0, 0.0],  # length 2
        ])

        # Manual calculation:
        # Row 0: token log-ratios = [0.2, 0.3, -0.2], sum = 0.3, length = 3 -> seq_log_ratio = 0.1
        # s_0 = exp(0.1) ~ 1.1051709
        # surr1 = 1.1051709 * 1.2 ~ 1.3262051
        # surr2 = clip(1.1051709, 0.8, 1.2) * 1.2 = 1.1051709 * 1.2 ~ 1.3262051 (within clip)
        # loss_0 = -1.3262051
        #
        # Row 1: token log-ratios = [0.2, 0.2], sum = 0.4, length = 2 -> seq_log_ratio = 0.2
        # s_1 = exp(0.2) ~ 1.2214028
        # surr1 = 1.2214028 * (-0.8) ~ -0.9771222
        # surr2 = clip(1.2214028, 0.8, 1.2) * (-0.8) = 1.2 * (-0.8) = -0.96
        # min(surr1, surr2) = min(-0.9771222, -0.96) = -0.9771222
        # loss_1 = -(-0.9771222) = 0.9771222
        #
        # Expected mean loss = 0.5 * (-1.3262051 + 0.9771222) = -0.17454145
        out = apply_variant_loss(
            "gspo",
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=adv,
            completion_mask=mask,
        )

        r0 = (0.2 + 0.3 - 0.2) / 3.0
        s0 = math.exp(r0)
        l0 = -min(s0 * 1.2, min(max(s0, 0.8), 1.2) * 1.2)

        r1 = (0.2 + 0.2) / 2.0
        s1 = math.exp(r1)
        l1 = -min(s1 * -0.8, min(max(s1, 0.8), 1.2) * -0.8)

        expected = 0.5 * (l0 + l1)
        assert abs(out.item() - expected) < 1e-6

    def test_batch_permutation_invariance(self) -> None:
        """Permuting batch rows must leave the GSPO loss invariant (#723)."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        torch.manual_seed(202)
        s = torch.randn(5, 8)
        o = torch.randn(5, 8)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.float32)
        adv = torch.tensor([1.4, -0.7, 0.3, -1.5, 0.9])

        loss_orig = apply_variant_loss(
            "gspo",
            logp_new=s,
            logp_old=o,
            advantages=adv,
            completion_mask=mask,
        )

        perm = [3, 0, 4, 1, 2]
        loss_perm = apply_variant_loss(
            "gspo",
            logp_new=s[perm],
            logp_old=o[perm],
            advantages=adv[perm],
            completion_mask=mask[perm],
        )

        assert abs(loss_orig.item() - loss_perm.item()) < 1e-6

    def test_padding_position_invariance(self) -> None:
        """Left-padded and right-padded completions must yield identical loss (#723)."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        # Right-padded sequence (content at [:3], padding at [3:])
        s_right = torch.tensor([[1.0, 2.0, 3.0, -99.0, 42.0]])
        o_right = torch.tensor([[0.5, 1.5, 2.5, 12.0, -18.0]])
        m_right = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        adv = torch.tensor([1.5])

        # Left-padded sequence (padding at [:2], content at [2:])
        s_left = torch.tensor([[-99.0, 42.0, 1.0, 2.0, 3.0]])
        o_left = torch.tensor([[12.0, -18.0, 0.5, 1.5, 2.5]])
        m_left = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

        loss_right = apply_variant_loss(
            "gspo",
            logp_new=s_right,
            logp_old=o_right,
            advantages=adv,
            completion_mask=m_right,
        )
        loss_left = apply_variant_loss(
            "gspo",
            logp_new=s_left,
            logp_old=o_left,
            advantages=adv,
            completion_mask=m_left,
        )

        assert abs(loss_right.item() - loss_left.item()) < 1e-6

    def test_unequal_sequence_lengths_normalized_independently(self) -> None:
        """Each sequence in a batch must be normalized by its own completion length."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        # Row 0: 1 token with diff=0.1 -> length=1, seq_log_ratio = 0.1 / 1 = 0.1
        # Row 1: 5 tokens each with diff=0.1 -> length=5, seq_log_ratio = 0.5 / 5 = 0.1
        # Both have s = exp(0.1) ~ 1.105 (within [0.8, 1.2], unclipped).
        # Because both have the same length-normalized ratio and same advantage,
        # their per-sequence loss must be identical.
        s = torch.tensor([
            [1.1, 0.0, 0.0, 0.0, 0.0],
            [1.1, 1.1, 1.1, 1.1, 1.1],
        ])
        o = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        mask = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        adv = torch.tensor([1.0, 1.0])

        loss = apply_variant_loss(
            "gspo",
            logp_new=s,
            logp_old=o,
            advantages=adv,
            completion_mask=mask,
        )
        expected = -math.exp(0.1) * 1.0
        assert abs(loss.item() - expected) < 1e-6

    @pytest.mark.parametrize(
        "diff,adv_val,is_clipped",
        [
            (0.0, 2.0, False),    # s = 1.0 -> within [0.8, 1.2], unclipped
            (0.1, 2.0, False),    # s = exp(0.1) ~ 1.105 -> unclipped
            (0.5, 2.0, True),     # s = exp(0.5) > 1.2, A > 0 -> upper clipped to 1.2 * A
            (-0.5, -2.0, True),   # s = exp(-0.5) < 0.8, A < 0 -> lower clipped to 0.8 * A
        ],
    )
    def test_clip_boundary_regimes(
        self, diff: float, adv_val: float, is_clipped: bool
    ) -> None:
        """Test clip boundary regimes: unclipped vs upper-clipped vs lower-clipped."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        s = (torch.ones(1, 2) * diff).requires_grad_(True)
        o = torch.zeros(1, 2)
        adv = torch.tensor([adv_val])

        loss = apply_variant_loss(
            "gspo",
            logp_new=s,
            logp_old=o,
            advantages=adv,
        )
        loss.backward()

        assert s.grad is not None
        if is_clipped:
            # Gradient through clipped surrogate must be zero
            assert (s.grad == 0.0).all()
        else:
            # Unclipped surrogate maintains non-zero policy gradient
            assert (s.grad.abs() > 0.0).all()

    def test_custom_delta_sets_clipping_radius(self) -> None:
        """Custom delta parameter sets [1-delta, 1+delta] clipping radius."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        # s = exp(0.25) ~ 1.284
        # With default eps=0.2 (max=1.2), this is clipped.
        # With delta=0.35 (max=1.35), this is UNCLIPPED.
        s = (torch.ones(1, 2) * 0.25).requires_grad_(True)
        o = torch.zeros(1, 2)
        adv = torch.tensor([1.0])

        loss_custom = apply_variant_loss(
            "gspo",
            logp_new=s,
            logp_old=o,
            advantages=adv,
            delta=0.35,
        )
        loss_custom.backward()

        assert s.grad is not None
        assert (s.grad.abs() > 0.0).all()  # unclipped under delta=0.35

    def test_all_masked_tokens_or_empty_completions(self) -> None:
        """All-masked rows or completely masked batches must not raise or produce NaN."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        # Row 1 is completely masked out (length 0)
        s = torch.randn(3, 4, requires_grad=True)
        o = torch.randn(3, 4)
        adv = torch.tensor([1.0, -0.5, 0.8])
        mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # empty completion
            [1.0, 1.0, 1.0, 1.0],
        ])

        loss = apply_variant_loss(
            "gspo",
            logp_new=s,
            logp_old=o,
            advantages=adv,
            completion_mask=mask,
        )
        assert math.isfinite(loss.item())
        loss.backward()

        assert s.grad is not None
        # Row 1 (all masked) receives strictly zero gradient
        assert (s.grad[1] == 0.0).all()

        # Completely masked batch (all zeros)
        s2 = torch.randn(2, 2, requires_grad=True)
        o2 = torch.randn(2, 2)
        m2 = torch.zeros(2, 2)
        loss_empty = apply_variant_loss(
            "gspo",
            logp_new=s2,
            logp_old=o2,
            advantages=torch.tensor([1.0, 2.0]),
            completion_mask=m2,
        )
        assert loss_empty.item() == 0.0
        loss_empty.backward()
        assert (s2.grad == 0.0).all()

    def test_advantages_shape_compatibility(self) -> None:
        """1D [B], 2D column [B, 1], and 2D broadcast [B, T] advantages yield identical loss."""
        torch = _torch_or_skip()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        s = torch.randn(4, 3)
        o = torch.randn(4, 3)
        m = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        adv_1d = torch.tensor([1.0, -0.5, 0.2, 0.8])
        adv_col = adv_1d.unsqueeze(-1)
        adv_tok = adv_1d.unsqueeze(-1).expand(-1, 3)

        l1 = apply_variant_loss(
            "gspo", logp_new=s, logp_old=o, advantages=adv_1d, completion_mask=m
        )
        l2 = apply_variant_loss(
            "gspo", logp_new=s, logp_old=o, advantages=adv_col, completion_mask=m
        )
        l3 = apply_variant_loss(
            "gspo", logp_new=s, logp_old=o, advantages=adv_tok, completion_mask=m
        )

        assert abs(l1.item() - l2.item()) < 1e-6
        assert abs(l1.item() - l3.item()) < 1e-6

    def test_runtime_grpo_trainer_integration(self) -> None:
        """Real trainer wrapper executes the sequence-level gspo kernel without fallback warning."""
        torch = _torch_or_skip()
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _StandaloneMockTrainer:
            def __init__(self, beta=0.0):
                self.args = type("Args", (), {"beta": beta})()
                self.model = type("Model", (), {"training": True})()
                self.current_gradient_accumulation_steps = 1
                self._get_logps_called = 0

            def _get_per_token_logps_and_entropies(
                self, model, input_ids, attention_mask, logits_to_keep, **kwargs
            ):
                self._get_logps_called += 1
                b = input_ids.size(0)
                t = logits_to_keep
                logps = torch.tensor(
                    [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], requires_grad=True
                )
                return logps, torch.zeros(b, t)

            def _compute_loss(self, model, inputs):
                return torch.tensor(777.0)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return self._compute_loss(model, inputs)

        variant_cls = make_grpo_trainer_variant(_StandaloneMockTrainer, "gspo")
        trainer = variant_cls()

        inputs = {
            "prompt_ids": torch.tensor([[1, 2], [3, 4]]),
            "prompt_mask": torch.tensor([[1, 1], [1, 1]]),
            "completion_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
            "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "per_token_logps": torch.tensor([[-0.2, -0.3, 0.0], [-0.4, -0.1, -0.5]]),
            "old_per_token_logps": torch.tensor([[-0.25, -0.35, 0.0], [-0.45, -0.15, -0.55]]),
            "advantages": torch.tensor([1.0, -0.5]),
        }

        loss = trainer.compute_loss(model=None, inputs=inputs)
        assert not trainer._soup_fallback_warned
        assert math.isfinite(float(loss.detach()))
        assert float(loss.detach()) != 777.0
