"""#1232: every non-standard ``grpo_variant`` dropped the KL penalty.

``apply_variant_loss`` range-checked ``beta`` and accepted ``reference_logp``,
then used neither. ``_GRPOTrainerVariant._compute_loss`` replaces trl's
``_compute_loss``, which is the one place trl adds ``beta * per_token_kl``, so
under gspo / dapo / dr_grpo / bnpo / two_sided / rft neither ``grpo_beta`` nor
the reward-hack controller's beta changed the loss or the gradient, while trl
still ran the reference forward on every step.

The fix follows trl 0.29's ``GRPOTrainer._compute_loss``: the per-token
estimator ``exp(ref - logp) - (ref - logp) - 1`` is weighted by ``beta`` and
added to the per-token loss before the variant's own reduction. ``rft`` has no
trl counterpart; it applies the term to the tokens it trains on (the accepted
completions) with the same denominator.

Three layers, each with the control that must not move:

1. the kernel: the beta response equals the masked KL estimate for every
   variant, in value and in gradient; ``beta == 0`` reproduces the pre-fix
   kernel bit for bit; a policy equal to its reference adds exactly zero;
2. the variant trainer on a stand-in base: which beta it reads, and what a
   missing reference does;
3. the real ``GRPOTrainerWrapper`` on a tiny offline checkpoint: the issue's
   repro (the gradient norm was constant across beta 0.1 / 1 / 10), parity
   with trl's own loss on the same generated batch, and the reward-hack
   controller's beta reaching the loss.
"""

import logging
import math
import types

import pytest

VARIANTS = ("gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft")


def _torch():
    return pytest.importorskip("torch")


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _delta(variant):
    return 0.2 if variant == "two_sided" else None


# --------------------------------------------------------------------------
# kernel fixtures
# --------------------------------------------------------------------------
def _batch(torch):
    """Three completions of different lengths, padded to five tokens.

    float64 so the tolerance can be far tighter than the issue's 1e-6. The old
    log-probs are perturbed so the ratio is not 1 and the clip branches of the
    variants are exercised by the bit-for-bit control.
    """
    gen = torch.Generator().manual_seed(1232)
    dtype = torch.float64
    logp_new = (-2.0 * torch.rand(3, 5, generator=gen, dtype=dtype)).requires_grad_(True)
    logp_old = logp_new.detach() + 0.2 * torch.randn(3, 5, generator=gen, dtype=dtype)
    offset = torch.empty(3, 5, dtype=dtype).uniform_(-1.5, 1.5, generator=gen)
    return {
        "logp_new": logp_new,
        "logp_old": logp_old,
        "reference": logp_new.detach() + offset,
        "advantages": torch.tensor([1.0, -0.5, 0.75], dtype=dtype),
        "mask": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=dtype
        ),
    }


def _loss(variant, batch, *, beta, reference="batch", mask="batch"):
    """``apply_variant_loss`` on ``batch``; pass ``reference`` / ``mask`` to override."""
    from soup_cli.utils.grpo_variants import apply_variant_loss

    return apply_variant_loss(
        variant,
        logp_new=batch["logp_new"],
        logp_old=batch["logp_old"],
        advantages=batch["advantages"],
        beta=beta,
        delta=_delta(variant),
        completion_mask=batch["mask"] if isinstance(mask, str) else mask,
        reference_logp=batch["reference"] if isinstance(reference, str) else reference,
    )


def _grad(torch, loss, wrt):
    (grad,) = torch.autograd.grad(loss, wrt)
    return grad


def _expected_kl(torch, variant, logp_new, reference, mask, advantages):
    """The KL term each variant should add, written out independently.

    trl's per-token estimator, reduced the way the variant reduces its policy
    term: a masked token mean over the batch (dapo, two_sided), a per-sequence
    length-normalised mean (bnpo), a per-token sum averaged over the batch
    (dr_grpo), a per-sequence mean over non-empty sequences (gspo), and a mean
    over the accepted tokens only (rft).
    """
    kl = torch.exp(reference - logp_new) - (reference - logp_new) - 1
    if variant in ("dapo", "two_sided"):
        return (kl * mask).sum() / mask.sum()
    if variant == "bnpo":
        return ((kl * mask).sum(-1) / mask.sum(-1)).mean()
    if variant == "dr_grpo":
        return (kl * mask).sum(-1).mean()
    if variant == "gspo":
        lengths = mask.sum(-1)
        per_sequence = (kl * mask).sum(-1) / lengths.clamp(min=1.0)
        return per_sequence[lengths > 0].mean()
    if variant == "rft":
        accepted = (advantages.unsqueeze(-1) > 0).to(mask.dtype) * mask
        return (kl * accepted).sum() / accepted.sum().clamp(min=1.0)
    raise AssertionError(variant)


# --------------------------------------------------------------------------
# the pre-#1232 kernel, verbatim, as the oracle for beta == 0
# --------------------------------------------------------------------------
def _pre_1232_masked_mean(token_loss, completion_mask=None, *, normalize_by_length=False):
    if completion_mask is None:
        return token_loss.mean()
    masked = token_loss * completion_mask
    if normalize_by_length:
        lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
        per_sample = masked.sum(dim=-1) / lengths
        return per_sample.mean()
    denom = completion_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def _pre_1232_loss(torch, name, *, logp_new, logp_old, advantages, delta, completion_mask):
    """``apply_variant_loss`` as it was before #1232 (it ignored beta and the reference)."""
    advantages_2d = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    if name == "gspo":
        eps = delta if delta is not None else 0.2
        if advantages.dim() == 1:
            adv_seq = advantages
        elif advantages.dim() == 2 and advantages.size(-1) == 1:
            adv_seq = advantages.squeeze(-1)
        elif completion_mask is not None:
            adv_seq = (advantages * completion_mask).sum(dim=-1) / completion_mask.sum(
                dim=-1
            ).clamp(min=1.0)
        else:
            adv_seq = advantages.mean(dim=-1)
        if completion_mask is not None:
            mask = completion_mask.to(dtype=log_ratio.dtype)
            lengths = mask.sum(dim=-1)
            valid_seq_mask = (lengths > 0).to(dtype=log_ratio.dtype)
            seq_log_ratio = (log_ratio * mask).sum(dim=-1) / lengths.clamp(min=1.0)
        else:
            seq_log_ratio = log_ratio.mean(dim=-1)
            valid_seq_mask = torch.ones_like(seq_log_ratio)
        seq_ratio = torch.exp(seq_log_ratio)
        surr1 = seq_ratio * adv_seq
        surr2 = torch.clamp(seq_ratio, min=1.0 - eps, max=1.0 + eps) * adv_seq
        seq_loss = -torch.min(surr1, surr2)
        denom = valid_seq_mask.sum().clamp(min=1.0)
        return (seq_loss * valid_seq_mask).sum() / denom
    if name == "dapo":
        clipped = torch.clamp(ratio, min=1 - 0.2, max=1 + 0.28)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _pre_1232_masked_mean(token_loss, completion_mask)
    if name == "dr_grpo":
        token_loss = -(ratio * advantages_2d)
        if completion_mask is not None:
            token_loss = token_loss * completion_mask
        return token_loss.sum(dim=-1).mean()
    if name == "bnpo":
        clipped = torch.clamp(ratio, min=1 - 0.2, max=1 + 0.2)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _pre_1232_masked_mean(token_loss, completion_mask, normalize_by_length=True)
    if name == "two_sided":
        clipped = torch.clamp(ratio, min=1 - delta, max=1 + delta)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _pre_1232_masked_mean(token_loss, completion_mask)
    if name == "rft":
        positive_mask = (advantages_2d > 0).to(logp_new.dtype)
        if completion_mask is not None:
            positive_mask = positive_mask * completion_mask
        token_loss = -(logp_new * positive_mask)
        denom = positive_mask.sum().clamp(min=1.0)
        return token_loss.sum() / denom
    raise AssertionError(name)


# ==========================================================================
# 1. the kernel
# ==========================================================================
class TestTheKernelAppliesTheKlTerm:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_beta_adds_the_masked_kl_estimate(self, variant):
        """Acceptance: loss(beta) - loss(0) is beta times the masked KL estimate."""
        torch = _torch()
        batch = _batch(torch)
        expected = _expected_kl(
            torch, variant, batch["logp_new"].detach(), batch["reference"],
            batch["mask"], batch["advantages"],
        ).item()
        assert expected > 0.01, "the fixture must put the reference visibly away"
        base = _loss(variant, batch, beta=0.0).item()
        for beta in (0.1, 1.0, 5.0):
            moved = _loss(variant, batch, beta=beta).item() - base
            assert moved == pytest.approx(beta * expected, rel=1e-9, abs=1e-12), (
                f"{variant}: beta={beta} moved the loss by {moved}, "
                f"expected {beta * expected}"
            )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_gradient_gains_the_kl_gradient(self, variant):
        """The gradient moves with beta, by exactly beta times the KL term's gradient."""
        torch = _torch()
        batch = _batch(torch)
        logp = batch["logp_new"]
        grad_0 = _grad(torch, _loss(variant, batch, beta=0.0), logp)
        kl_grad = _grad(
            torch,
            _expected_kl(
                torch, variant, logp, batch["reference"], batch["mask"], batch["advantages"]
            ),
            logp,
        )
        assert kl_grad.abs().max() > 1e-3, "the KL term must have a non-trivial gradient"
        for beta in (0.1, 1.0, 5.0):
            grad_b = _grad(torch, _loss(variant, batch, beta=beta), logp)
            assert torch.allclose(grad_b - grad_0, beta * kl_grad, rtol=1e-9, atol=1e-12), (
                f"{variant}: beta={beta} gradient does not carry beta * dKL/dlogp"
            )
        # padding carries no gradient from the KL term
        grad_1 = _grad(torch, _loss(variant, batch, beta=1.0), logp)
        padding = batch["mask"] == 0
        assert torch.equal((grad_1 - grad_0)[padding], torch.zeros_like(grad_1[padding]))

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_issue_sweep_now_moves(self, variant):
        """The issue's own kernel snippet: a reference 3 nats away on every token.

        trl's estimator is exp(-3) + 2 = 2.0498 per token there, so beta=0.1
        must add 0.2050 and beta=5 must add 10.2489 (x4 tokens for dr_grpo,
        which sums over tokens). Before the fix all four losses were equal.
        """
        torch = _torch()
        from soup_cli.utils.grpo_variants import apply_variant_loss

        torch.manual_seed(0)
        logp = torch.randn(2, 4)
        kw = dict(
            logp_new=logp,
            logp_old=logp - 0.1,
            advantages=torch.tensor([1.0, -0.5]),
            completion_mask=torch.ones(2, 4),
        )
        reference = logp - 3.0
        per_token = math.exp(-3.0) + 2.0
        factor = 4.0 if variant == "dr_grpo" else 1.0
        losses = [
            apply_variant_loss(
                variant, beta=b, reference_logp=reference, delta=_delta(variant), **kw
            ).item()
            for b in (0.0, 0.1, 5.0, 100.0)
        ]
        assert len(set(losses)) == 4, f"{variant}: loss did not move with beta: {losses}"
        for b, value in zip((0.1, 5.0, 100.0), losses[1:]):
            assert value - losses[0] == pytest.approx(b * factor * per_token, rel=1e-5), (
                variant, b, value - losses[0],
            )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_no_completion_mask_uses_the_unmasked_reduction(self, variant):
        """Without a mask the KL is reduced like the unmasked policy term."""
        torch = _torch()
        batch = _batch(torch)
        logp = batch["logp_new"].detach()
        kl = torch.exp(batch["reference"] - logp) - (batch["reference"] - logp) - 1
        if variant == "dr_grpo":
            expected = kl.sum(-1).mean()
        elif variant == "rft":
            # without a mask rft's denominator counts accepted ROWS, and the
            # KL follows the same reduction as its policy term
            accepted = (batch["advantages"].unsqueeze(-1) > 0).to(kl.dtype)
            expected = (kl * accepted).sum() / accepted.sum()
        else:
            expected = kl.mean()
        base = _loss(variant, batch, beta=0.0, mask=None).item()
        moved = _loss(variant, batch, beta=2.0, mask=None).item() - base
        assert moved == pytest.approx(2.0 * expected.item(), rel=1e-9), variant


class TestTheKernelControls:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_beta_zero_is_bit_identical_to_the_pre_fix_kernel(self, variant):
        """CONTROL: at beta == 0 the loss and gradient are the pre-#1232 ones, bit for bit.

        The reference is ignored completely at beta == 0: the third one is 1000
        nats ABOVE the policy, where exp() overflows float64 to inf, so
        computing the estimator at all (even to multiply it by 0) would turn the
        loss into NaN. The last assertion is what makes the comparison mean
        something: the frozen copy is not the current code, because at beta=1
        they differ.
        """
        torch = _torch()
        batch = _batch(torch)
        logp = batch["logp_new"]
        old = _pre_1232_loss(
            torch, variant, logp_new=logp, logp_old=batch["logp_old"],
            advantages=batch["advantages"], delta=_delta(variant),
            completion_mask=batch["mask"],
        )
        old_grad = _grad(torch, old, logp)
        overflowing = logp.detach() + 1000.0
        for reference in (None, batch["reference"], overflowing):
            new = _loss(variant, batch, beta=0.0, reference=reference)
            assert torch.equal(new, old), (variant, reference is None, new, old)
            assert torch.equal(_grad(torch, new, logp), old_grad), variant
        assert not torch.equal(_loss(variant, batch, beta=1.0), old), (
            f"{variant}: beta=1 still reproduces the pre-fix loss, so beta is ignored"
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_policy_equal_to_reference_adds_exactly_zero(self, variant):
        """A policy identical to its reference has KL exactly 0: exp(0) - 0 - 1."""
        torch = _torch()
        batch = _batch(torch)
        logp = batch["logp_new"]
        same = logp.detach().clone()
        base = _loss(variant, batch, beta=0.0)
        base_grad = _grad(torch, base, logp)
        for beta in (0.1, 1.0, 100.0):
            loss = _loss(variant, batch, beta=beta, reference=same)
            assert torch.equal(loss, base), (variant, beta, loss, base)
            assert torch.equal(_grad(torch, loss, logp), base_grad), (variant, beta)
        # ... and one nat of distance is enough to register
        shifted = _loss(variant, batch, beta=1.0, reference=same - 1.0)
        assert shifted.item() > base.item(), (variant, shifted, base)


class TestTheKernelRefusesWhatItCannotCompute:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_positive_beta_without_reference_raises(self, variant):
        torch = _torch()
        batch = _batch(torch)
        with pytest.raises(ValueError, match=r"beta=0\.1 needs reference_logp"):
            _loss(variant, batch, beta=0.1, reference=None)

    def test_reference_shape_must_match_the_policy(self):
        """A [B, 1] reference would broadcast silently into a wrong estimate."""
        torch = _torch()
        batch = _batch(torch)
        with pytest.raises(ValueError, match=r"reference_logp shape \(3, 1\).*\(3, 5\)"):
            _loss("dapo", batch, beta=0.1, reference=batch["reference"][:, :1])

    def test_standard_still_defers_to_trl(self):
        """``standard`` returns None whatever beta is: trl's own loss applies it.

        It is exempt from the reference requirement every other variant now has.
        """
        torch = _torch()
        batch = _batch(torch)
        assert _loss("standard", batch, beta=0.1, reference=None) is None
        with pytest.raises(ValueError, match="needs reference_logp"):
            _loss("dapo", batch, beta=0.1, reference=None)


class TestTheKernelReadsOnlyTheTokensItScores:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_masked_reference_values_are_ignored(self, variant):
        """The reference is read on scored tokens and ignored on padding."""
        torch = _torch()
        batch = _batch(torch)
        base = _loss(variant, batch, beta=1.0)
        padded = batch["reference"].clone()
        padded[batch["mask"] == 0] += 7.0  # finite garbage in the padding
        assert torch.equal(_loss(variant, batch, beta=1.0, reference=padded), base), variant
        scored = batch["reference"].clone()
        scored[0, 0] += 0.5  # row 0 is accepted and non-empty for every variant
        assert _loss(variant, batch, beta=1.0, reference=scored).item() != base.item(), (
            f"{variant}: a scored token's reference did not reach the loss"
        )

    def test_rft_applies_kl_only_to_accepted_completions(self):
        """rft trains on accepted completions only; its KL term covers the same tokens.

        A rejected row's reference must not matter, and a batch in which no
        completion was accepted still contributes a zero loss at any beta.
        """
        torch = _torch()
        batch = _batch(torch)  # advantages [1.0, -0.5, 0.75]: row 1 is rejected
        base = _loss("rft", batch, beta=1.0)
        rejected = batch["reference"].clone()
        rejected[1] -= 2.0
        assert torch.equal(_loss("rft", batch, beta=1.0, reference=rejected), base)
        accepted = batch["reference"].clone()
        accepted[2, 0] -= 2.0
        assert _loss("rft", batch, beta=1.0, reference=accepted).item() != base.item()

        none_accepted = dict(batch, advantages=torch.tensor([-1.0, 0.0, -0.3],
                                                            dtype=torch.float64))
        loss = _loss("rft", none_accepted, beta=5.0)
        assert loss.item() == 0.0

    def test_gspo_skips_the_kl_of_an_empty_completion(self):
        """gspo averages over non-empty sequences; an empty one adds no KL either."""
        torch = _torch()
        batch = _batch(torch)
        mask = batch["mask"].clone()
        mask[2] = 0.0
        loss = _loss("gspo", batch, beta=1.0, mask=mask)
        base = _loss("gspo", batch, beta=0.0, mask=mask)
        expected = _expected_kl(
            torch, "gspo", batch["logp_new"].detach(), batch["reference"], mask,
            batch["advantages"],
        ).item()
        assert math.isfinite(loss.item())
        assert loss.item() - base.item() == pytest.approx(expected, rel=1e-9)


# ==========================================================================
# 2. the variant trainer on a stand-in base
# ==========================================================================
class _StandInGRPOBase:
    """Only what ``_GRPOTrainerVariant._compute_variant_loss`` touches."""

    def __init__(self, *, logps, beta=None, args_beta=0.0):
        if beta is not None:
            self.beta = beta
        self.args = types.SimpleNamespace(beta=args_beta)
        self.model = types.SimpleNamespace(training=True)
        self.current_gradient_accumulation_steps = 1
        self._logps = logps
        self.stock_calls = 0

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                           logits_to_keep, **kwargs):
        return self._logps, None

    def _compute_loss(self, model, inputs):
        import torch

        self.stock_calls += 1
        return torch.tensor(-1234.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return self._compute_loss(model, inputs)


def _trainer_inputs(torch, batch, *, with_reference=True):
    inputs = {
        "prompt_ids": torch.ones(3, 2, dtype=torch.long),
        "prompt_mask": torch.ones(3, 2, dtype=torch.long),
        "completion_ids": torch.ones(3, 5, dtype=torch.long),
        "completion_mask": batch["mask"],
        "advantages": batch["advantages"],
    }
    if with_reference:
        inputs["ref_per_token_logps"] = batch["reference"]
    return inputs


def _variant_trainer(variant, **kwargs):
    from soup_cli.trainer.grpo import make_grpo_trainer_variant

    trainer = make_grpo_trainer_variant(_StandInGRPOBase, variant)(**kwargs)
    if variant == "two_sided":
        trainer._soup_grpo_delta = 0.2
    return trainer


class TestTheVariantTrainerWiresBeta:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_loss_follows_trls_live_beta(self, variant):
        """The variant weights the KL by trl's own ``self.beta``.

        That is the attribute trl consults before running the reference forward
        and the one its stock loss uses, so the reference exists exactly when
        the loss needs it. A base that exposes only ``args.beta`` still works.
        """
        torch = _torch()
        batch = _batch(torch)
        logp = batch["logp_new"]
        kl = _expected_kl(
            torch, variant, logp.detach(), batch["reference"], batch["mask"],
            batch["advantages"],
        ).item()
        # ratio 1 inside the trainer (no old_per_token_logps): the policy term
        # is the beta=0 loss evaluated at logp_old = logp_new
        policy = _pre_1232_loss(
            torch, variant, logp_new=logp, logp_old=logp.detach(),
            advantages=batch["advantages"], delta=_delta(variant),
            completion_mask=batch["mask"],
        ).item()

        trainer = _variant_trainer(variant, logps=logp, beta=0.5, args_beta=0.1)
        loss = trainer._compute_loss(None, _trainer_inputs(torch, batch)).item()
        assert loss == pytest.approx(policy + 0.5 * kl, rel=1e-9), (variant, loss)
        assert trainer.stock_calls == 0

        args_only = _variant_trainer(variant, logps=logp, args_beta=0.1)
        loss = args_only._compute_loss(None, _trainer_inputs(torch, batch)).item()
        assert loss == pytest.approx(policy + 0.1 * kl, rel=1e-9), (variant, loss)

    def test_a_missing_reference_falls_back_loudly_instead_of_dropping_kl(self, caplog):
        """beta > 0 with no reference must not train the variant without its KL.

        The kernel refuses, and the trainer takes its existing one-shot
        fallback (#159) to trl's stock loss, which carries trl's own KL term.
        At beta == 0 no reference is needed and the variant runs.
        """
        torch = _torch()
        batch = _batch(torch)
        no_reference = _trainer_inputs(torch, batch, with_reference=False)

        kl_free = _variant_trainer("dapo", logps=batch["logp_new"], beta=0.0)
        loss = kl_free._compute_loss(None, no_reference)
        assert kl_free.stock_calls == 0
        assert loss.item() != -1234.0

        trainer = _variant_trainer("dapo", logps=batch["logp_new"], beta=0.1)
        with caplog.at_level(logging.WARNING, logger="soup_cli.trainer.grpo"):
            loss = trainer._compute_loss(None, no_reference)
        assert trainer.stock_calls == 1
        assert loss.item() == -1234.0
        messages = [r.getMessage() for r in caplog.records if "fell back" in r.getMessage()]
        assert len(messages) == 1, [r.getMessage() for r in caplog.records]
        assert "needs reference_logp" in messages[0]


# ==========================================================================
# 3. the real trainer on a tiny offline checkpoint
# ==========================================================================
def _tiny_llama_dir(tmp_path, vocab=64, hidden=64):
    """Duplicated from test_issue353_seed_every_task.py so this file stands alone."""
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(directory))


def _build_real(tmp_path, monkeypatch, variant, **training_over):
    """``GRPOTrainerWrapper.setup()`` for real, with ``grpo_beta: 0.1``.

    Gradient accumulation 1, so trl generates one micro-batch per step and its
    ``num_items_in_batch`` is that micro-batch's token count: the global-token
    normaliser of trl's ``dapo`` and the local one of Soup's coincide.
    """
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.grpo import GRPOTrainerWrapper

    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    training = {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "reward_fn": "format",
        "quantization": "none",
        "epochs": 1,
        "grpo_variant": variant,
        "grpo_beta": 0.1,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if variant == "two_sided":
        training["grpo_delta"] = 0.2
    training.update(training_over)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "grpo",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = GRPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"prompt": "hi"} for _ in range(4)]})
    return wrapper


def _generated_batch_off_reference(trainer):
    """One real trl batch, then the policy moved away from its reference.

    trl computes ``ref_per_token_logps`` (adapter disabled) because its beta is
    non-zero. Advantages are set to a winner and a loser, as in the issue, and
    the LoRA B weights are nudged so the policy differs from that reference.
    """
    import torch

    torch.manual_seed(0)
    batch = next(iter(trainer.get_train_dataloader()))
    inputs = trainer._prepare_inputs(batch)
    assert inputs.get("ref_per_token_logps") is not None, (
        "trl did not compute the reference log-probs although beta > 0"
    )
    rows = inputs["advantages"].shape[0]
    inputs["advantages"] = torch.tensor([1.0, -1.0] * (rows // 2))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in trainer.model.named_parameters():
            if "lora_B" in name:
                param.add_(0.5 * torch.randn(param.shape, generator=generator))
    trainer.model.train()
    # set by the HF training loop; the variant and trl's grpo/bnpo/dr_grpo read it
    trainer.current_gradient_accumulation_steps = 1
    return inputs


def _loss_and_grad(trainer, compute):
    import torch

    model = trainer.model
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    loss = compute()
    loss.backward()
    grads = [p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]
    return loss.detach(), torch.cat(grads)


def _set_beta(trainer, beta):
    """Write beta the way the reward-hack controller's ``_apply_coefficient`` does."""
    trainer.beta = beta
    trainer.args.beta = beta


# variant -> (trl loss_type, trl importance_sampling_level): the trl reduction
# that is the same as the variant's own. dr_grpo is scaled: trl divides its token
# sum by max_completion_length, Soup's dr_grpo does not.
_TRL_REDUCTION = {
    "dapo": ("dapo", "token"),
    "two_sided": ("dapo", "token"),
    "bnpo": ("grpo", "token"),
    "gspo": ("grpo", "sequence"),
    "dr_grpo": ("dr_grpo", "token"),
}


def _trl_stock_loss(trainer, variant, inputs):
    stock = type(trainer).__mro__[1]
    loss_type, level = _TRL_REDUCTION[variant]
    saved = (trainer.loss_type, trainer.importance_sampling_level)
    trainer.loss_type, trainer.importance_sampling_level = loss_type, level
    try:
        return stock._compute_loss(trainer, trainer.model, inputs)
    finally:
        trainer.loss_type, trainer.importance_sampling_level = saved


def _rft_reference_loss(trainer, inputs):
    """rft has no trl counterpart: its loss written out on the model's log-probs."""
    import torch

    input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
    attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
    logp, _ = trainer._get_per_token_logps_and_entropies(
        trainer.model, input_ids, attention_mask, inputs["completion_ids"].size(1)
    )
    ref = inputs["ref_per_token_logps"]
    kl = torch.exp(ref - logp) - (ref - logp) - 1
    accepted = (inputs["advantages"].unsqueeze(-1) > 0).to(logp.dtype) * inputs["completion_mask"]
    return (-(logp * accepted) + trainer.beta * kl * accepted).sum() / accepted.sum().clamp(min=1.0)


class TestTheRealTrainer:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_loss_and_gradient_respond_to_beta_like_trl(self, tmp_path, monkeypatch, variant):
        """The issue's repro, for every variant.

        Before the fix the variant loss and its gradient norm were identical at
        beta 0.1, 1 and 10 while trl's own loss moved. Now they move, and on the
        same generated batch they equal trl's loss with the same reduction (the
        importance ratio is 1 here, so the policy terms coincide too).
        """
        _requires_train_extra()
        import torch

        wrapper = _build_real(tmp_path, monkeypatch, variant)
        trainer = wrapper.trainer
        assert type(trainer).__name__ == f"_GRPOTrainerVariant_{variant}"
        assert trainer.beta == 0.1
        inputs = _generated_batch_off_reference(trainer)
        scale = float(trainer.max_completion_length) if variant == "dr_grpo" else 1.0

        def variant_loss():
            return trainer._compute_loss(trainer.model, inputs)

        losses, norms = [], []
        for beta in (0.1, 1.0, 10.0):
            _set_beta(trainer, beta)
            loss, grad = _loss_and_grad(trainer, variant_loss)
            if variant == "rft":
                ref_loss, ref_grad = _loss_and_grad(
                    trainer, lambda: _rft_reference_loss(trainer, inputs)
                )
            else:
                ref_loss, ref_grad = _loss_and_grad(
                    trainer, lambda: _trl_stock_loss(trainer, variant, inputs)
                )
                ref_loss, ref_grad = ref_loss * scale, ref_grad * scale
            assert loss.item() == pytest.approx(ref_loss.item(), rel=1e-5, abs=1e-7), (
                f"{variant}: beta={beta} loss {loss.item()} != reference {ref_loss.item()}"
            )
            assert torch.allclose(grad, ref_grad, rtol=1e-4, atol=1e-7), (
                f"{variant}: beta={beta} gradient differs from the reference's"
            )
            losses.append(loss.item())
            norms.append(grad.norm().item())
        assert losses[0] < losses[1] < losses[2], (variant, losses)
        assert len(set(norms)) == 3, f"{variant}: gradient norm constant across beta: {norms}"

    def test_the_reward_hack_controllers_beta_reaches_the_variant_loss(
        self, tmp_path, monkeypatch
    ):
        """``kl_control`` raises beta through ``_apply_coefficient``; the variant sees it.

        The controller is the one ``setup()`` attached. One tripping vote
        doubles beta (``kl_gain: 2``); the variant's loss then moves by exactly
        as much as trl's stock loss moves between the same two betas.
        """
        _requires_train_extra()
        wrapper = _build_real(
            tmp_path,
            monkeypatch,
            "dapo",
            reward_hack_detector="info_rm",
            reward_hack_mitigation="kl_control",
            reward_hack_dwell_steps=1,
            reward_hack_kl_gain=2.0,
        )
        trainer = wrapper.trainer
        controllers = [
            cb for cb in trainer.callback_handler.callbacks
            if type(cb).__name__ == "RewardHackMitigationCallback"
        ]
        assert len(controllers) == 1, trainer.callback_handler.callbacks
        controller = controllers[0]
        inputs = _generated_batch_off_reference(trainer)

        def variant_loss():
            return _loss_and_grad(trainer, lambda: trainer._compute_loss(trainer.model, inputs))

        def stock_loss():
            return _loss_and_grad(trainer, lambda: _trl_stock_loss(trainer, "dapo", inputs))

        before, _ = variant_loss()
        stock_before, _ = stock_loss()
        controller._run_bang_bang({}, {"info_rm": 1.0})
        assert trainer.beta == pytest.approx(0.2)
        assert trainer.args.beta == pytest.approx(0.2)
        after, _ = variant_loss()
        stock_after, _ = stock_loss()
        moved = after.item() - before.item()
        assert moved > 0.0, "the controller's beta did not reach the variant loss"
        assert moved == pytest.approx(stock_after.item() - stock_before.item(), rel=1e-4), (
            moved, stock_after.item() - stock_before.item(),
        )
