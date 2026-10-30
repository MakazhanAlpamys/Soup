"""Native GRPO on MLX (#1976), Step 1: a dense base.

The objective is the one Soup's transformers path trains on TRL 0.29.1, whose
``GRPOConfig`` defaults Soup never overrides: ``loss_type="dapo"``,
``scale_rewards="group"``, ``epsilon=0.2``, ``num_iterations=1``,
``importance_sampling_level="token"``. Layers:

- units, prompt order, advantages, completion masks, reward kwargs and the data
  contract: pure Python, checked against independent references (no MLX);
- the loss and its gradients against a float64 NumPy oracle on fixed logits with
  synthetic old / reference log-probs (needs MLX), and the oracle against TRL's
  own ``GRPOTrainer._compute_loss`` (needs torch + trl, no MLX);
- generation, the adapter-off reference and training through the real wrapper
  on a tiny random Llama (needs MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tests.conftest import strip_ansi


def _plain(text):
    """Console output without SGR escapes, whitespace collapsed: what a reader sees
    under FORCE_COLOR=1 or a narrow COLUMNS, where rich colours and wraps it."""
    return " ".join(strip_ansi(text).split())

# --------------------------------------------------------------------------
# A digit tokenizer with HF's surface. Ids: 0 pad, 1 BOS, 2 EOS, 3.. chars.
# Small enough that a tiny random model's completions often state a number.
# --------------------------------------------------------------------------

PAD, BOS, EOS = 0, 1, 2
_CHARS = "0123456789 :?<>/abcdefghijklmnopqrstuvwxyz|"
_IDS = {c: i + 3 for i, c in enumerate(_CHARS)}
VOCAB = len(_CHARS) + 3


class DigitTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS
    pad_token_id = PAD
    eos_token = "$"
    chat_template = "<digit template>"

    def __init__(self, chars=_CHARS):
        self.chars = chars
        self.ids = {c: i + 3 for i, c in enumerate(chars)}

    def encode(self, text, add_special_tokens=True):
        return [self.ids[c] for c in text if c in self.ids]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_):
        text = "".join(f"<{m['role'][0]}>{m['content']}|" for m in messages)
        text += "<a>" if add_generation_prompt else ""
        return self.encode(text) if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if i in (PAD, BOS, EOS):
                if not skip_special_tokens:
                    out.append({PAD: "_", BOS: "^", EOS: "$"}[i])
            else:
                out.append(self.chars[i - 3])
        return "".join(out)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------


def _np_advantages(rewards, group):
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, ddof=1, keepdims=True)
    return ((r - mean) / (std + 1e-4)).reshape(-1)


def _np_logps(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1))
    picked = np.take_along_axis(shifted, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return picked - log_z


def _np_loss(logps, old, ref, adv, mask, *, beta, num_items, eps_low=0.2, eps_high=0.2):
    """TRL 0.29.1 ``_compute_loss`` for loss_type='dapo', token-level IS, in float64."""
    logps, mask = np.asarray(logps, np.float64), np.asarray(mask, np.float64)
    old = np.asarray(old, np.float64)
    adv = np.asarray(adv, np.float64)[:, None]
    coef_1 = np.exp(logps - old)
    coef_2 = np.clip(coef_1, 1 - eps_low, 1 + eps_high)
    per_token = -np.minimum(coef_1 * adv, coef_2 * adv)
    if beta != 0.0:
        d = np.asarray(ref, np.float64) - logps
        per_token = per_token + beta * (np.exp(d) - d - 1)
    return float((per_token * mask).sum() / num_items)


# --------------------------------------------------------------------------
# Units and prompt order
# --------------------------------------------------------------------------


class TestUnits:
    def test_batch_size_counts_completions_like_trl(self):
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=4, accum=2, num_generations=4)
        assert (units.completions_per_step, units.accum) == (4, 2)
        assert (units.prompts_per_update, units.completions_per_update) == (2, 8)
        assert not units.bumped

    def test_a_batch_smaller_than_one_group_is_raised_to_it(self):
        """grpo.py raises per_device_train_batch_size to num_generations."""
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=2, accum=1, num_generations=4)
        assert units.bumped and units.completions_per_step == 4
        assert units.prompts_per_update == 1

    @pytest.mark.parametrize("batch_size, accum, g", [(6, 1, 4), (5, 3, 4), (4, 1, 3)])
    def test_whole_groups_per_update_are_required(self, batch_size, accum, g):
        from soup_cli.trainer.mlx_grpo import resolve_units

        total = batch_size * accum
        with pytest.raises(ValueError, match=rf"\({total}\) must be divisible by "
                                             rf"num_generations \({g}\)"):
            resolve_units(batch_size=batch_size, accum=accum, num_generations=g)

    def test_odd_micro_batches_that_sum_to_whole_groups_are_accepted(self):
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=6, accum=2, num_generations=4)
        assert (units.prompts_per_update, units.completions_per_update) == (3, 12)

    def test_updates_per_epoch_drop_the_trailing_partial_chunk(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        plan = plan_prompt_updates(n_prompts=7, prompts_per_update=3, epochs=2, seed=0)
        assert len(plan) == 2 * (7 // 3)
        assert all(len(chunk) == 3 for chunk in plan)
        for epoch in (plan[:2], plan[2:]):
            seen = [i for chunk in epoch for i in chunk]
            assert len(set(seen)) == 6 and set(seen) <= set(range(7))

    def test_order_is_seeded_and_reshuffled_each_epoch(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        a = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=3)
        b = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=3)
        c = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=4)
        assert a == b and a != c
        assert a[:6] != a[6:]

    def test_exactly_one_update_of_prompts_trains(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        plan = plan_prompt_updates(n_prompts=3, prompts_per_update=3, epochs=2, seed=0)
        assert len(plan) == 2 and all(sorted(chunk) == [0, 1, 2] for chunk in plan)

    def test_fewer_prompts_than_one_update_are_refused(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        with pytest.raises(ValueError, match="2 prompt"):
            plan_prompt_updates(n_prompts=2, prompts_per_update=3, epochs=1, seed=0)


# --------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------


class TestAdvantages:
    @pytest.mark.parametrize("rewards, group", [
        ([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0], 4),
        ([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 3),
        ([3.0, -2.0, 7.5, 0.25], 2),
        ([1e6, 1e6 + 3.0, -1e6, 5e5], 2),
        ([1e-7, 2e-7, 0.0, 3e-7], 4),
    ])
    def test_match_an_independent_reference(self, rewards, group):
        from soup_cli.trainer.mlx_grpo import group_advantages

        np.testing.assert_allclose(group_advantages(rewards, group),
                                   _np_advantages(rewards, group), rtol=1e-12, atol=1e-12)

    def test_equal_rewards_give_zero_advantage(self):
        from soup_cli.trainer.mlx_grpo import group_advantages

        assert group_advantages([0.5] * 4 + [1.0] * 4, 4) == [0.0] * 8

    def test_groups_are_normalised_separately(self):
        """Shifting one group's rewards must not move another group's advantages."""
        from soup_cli.trainer.mlx_grpo import group_advantages

        a = group_advantages([1.0, 0.0, 0.0, 0.0, 1.0, 2.0], 3)
        b = group_advantages([1.0, 0.0, 0.0, 100.0, 101.0, 102.0], 3)
        assert a[:3] == b[:3]
        np.testing.assert_allclose(a[3:], b[3:], rtol=1e-9)

    def test_sample_std_with_the_epsilon_outside_the_root(self):
        from soup_cli.trainer.mlx_grpo import group_advantages

        # Group (0, 1): mean 0.5, sample std sqrt(0.5) (population std would be 0.5).
        adv = group_advantages([0.0, 1.0], 2)
        assert adv[1] == pytest.approx(0.5 / (math.sqrt(0.5) + 1e-4), rel=1e-12)
        # Tiny spread: std 1e-6, so the epsilon dominates -- inside the root it would not.
        tiny = group_advantages([0.0, math.sqrt(2) * 1e-6], 2)
        assert tiny[1] == pytest.approx((math.sqrt(2) * 0.5e-6) / (1e-6 + 1e-4), rel=1e-9)

    @pytest.mark.parametrize("rewards, match", [
        ([1.0, 0.0, 1.0], "3 reward"),
        ([1.0, float("nan"), 0.0, 1.0], "non-finite"),
        ([1.0, float("inf"), 0.0, 1.0], "non-finite"),
        ([True, 0.0, 0.0, 1.0], "boolean"),
        ([None, 0.0, 0.0, 1.0], "non-numeric"),
    ])
    def test_bad_rewards_are_refused(self, rewards, match):
        from soup_cli.trainer.mlx_grpo import group_advantages

        with pytest.raises(ValueError, match=match):
            group_advantages(rewards, 2)

    def test_reward_statistics_match_trl_logging(self):
        from soup_cli.trainer.mlx_grpo import reward_statistics

        rewards = [1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        stats = reward_statistics(rewards, 4)
        assert stats["reward"] == pytest.approx(np.mean(rewards))
        assert stats["reward_std"] == pytest.approx(np.std(rewards, ddof=1))
        assert stats["frac_reward_zero_std"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Completions and masks (pure)
# --------------------------------------------------------------------------


class TestCompletionMasks:
    def test_kept_through_the_first_eos(self):
        from soup_cli.trainer.mlx_grpo import cut_at_eos

        assert cut_at_eos([5, 6, EOS, 7, EOS], EOS) == ([5, 6, EOS], True)
        assert cut_at_eos([EOS, 5], EOS) == ([EOS], True)

    def test_truncated_completion_keeps_every_token(self):
        from soup_cli.trainer.mlx_grpo import cut_at_eos

        assert cut_at_eos([5, 6, 7], EOS) == ([5, 6, 7], False)

    def test_eos_equal_to_the_pad_id_is_still_a_completion_token(self):
        from soup_cli.trainer.mlx_grpo import collate_completions, cut_at_eos

        kept, done = cut_at_eos([5, 0, 6], 0)
        assert (kept, done) == ([5, 0], True)
        tokens, mask, start = collate_completions([([9, 9, 9], kept), ([9], [7])])
        # Row 0: the pad-valued EOS at position 4 is supervised; row 1's padding is not.
        assert start == 0
        assert mask[0] == [0, 0, 1, 1]
        assert mask[1] == [1, 0, 0, 0]
        assert tokens[0] == [9, 9, 9, 5, 0] and tokens[1] == [9, 7, 0, 0, 0]

    def test_mask_starts_at_each_rows_own_prompt_end(self):
        from soup_cli.trainer.mlx_grpo import collate_completions

        tokens, mask, start = collate_completions([([1, 2, 3, 4], [5, EOS]), ([1, 2], [6, 7, 8])])
        # Logit positions start at the shortest prompt's last token.
        assert start == 1
        width = len(tokens[0]) - 1 - start
        assert all(len(row) == width for row in mask)
        # Column j scores sequence position start + j + 1, i.e. positions 2..5.
        assert width == 4
        assert mask[0] == [0, 0, 1, 1]
        assert mask[1] == [1, 1, 1, 0]
        assert sum(map(sum, mask)) == 5


# --------------------------------------------------------------------------
# Rewards: TRL's kwargs shape, gold kept with its prompt
# --------------------------------------------------------------------------


class TestRewardKwargs:
    def test_trl_shape_and_every_non_prompt_column(self):
        from soup_cli.trainer.mlx_grpo import score_completions

        seen = {}

        def reward(completions, **kwargs):
            seen.update(kwargs, completions=completions)
            return [0.0] * len(completions)

        rows = [{"prompt": [{"role": "user", "content": "a"}], "answer": "1", "tag": "x"},
                {"prompt": [{"role": "user", "content": "b"}], "answer": "2", "tag": "y"}]
        score_completions(reward, rows, ["c1", "c2", "c3", "c4"], [[3], [4], [5], [6]], 2)
        assert seen["completions"] == [[{"role": "assistant", "content": c}]
                                       for c in ("c1", "c2", "c3", "c4")]
        assert seen["answer"] == ["1", "1", "2", "2"]
        assert seen["tag"] == ["x", "x", "y", "y"]
        assert seen["prompts"] == [rows[0]["prompt"]] * 2 + [rows[1]["prompt"]] * 2
        assert seen["completion_ids"] == [[3], [4], [5], [6]]

    def test_accuracy_scores_each_completion_against_its_own_gold(self):
        """Shuffled prompts, two of them identical with different golds."""
        from soup_cli.trainer.mlx_grpo import score_completions
        from soup_cli.trainer.rewards import accuracy_reward, validate_reward_funcs

        rows = [{"prompt": [{"role": "user", "content": "q"}], "answer": str(g)}
                for g in (3, 5, 7, 9)]
        rows.append({"prompt": [{"role": "user", "content": "q"}], "answer": "4"})
        rng = random.Random(0)
        order = list(range(len(rows)))
        rng.shuffle(order)
        chunk = [rows[i] for i in order]
        texts = [f"#### {row['answer']}" if k % 2 == 0 else "#### 0"
                 for row in chunk for k in range(2)]
        scores = score_completions(validate_reward_funcs(accuracy_reward), chunk, texts,
                                   [[3]] * len(texts), 2)
        assert scores == [1.0, 0.0] * len(chunk)

    def test_a_wrong_count_or_nonfinite_score_is_refused(self):
        from soup_cli.trainer.mlx_grpo import score_completions
        from soup_cli.trainer.rewards import validate_reward_funcs

        rows = [{"prompt": [{"role": "user", "content": "q"}]}]

        def short(completions, **_):
            return [1.0]

        def nan(completions, **_):
            return [float("nan")] * len(completions)

        with pytest.raises(ValueError, match="returned 1 scores for 2 completions"):
            score_completions(validate_reward_funcs(short), rows, ["a", "b"], [[3], [4]], 2)
        with pytest.raises(ValueError, match="non-finite"):
            score_completions(validate_reward_funcs(nan), rows, ["a", "b"], [[3], [4]], 2)


# --------------------------------------------------------------------------
# The loss: a fixed-logits fixture with synthetic old / reference log-probs
# --------------------------------------------------------------------------

#: (prompt length, completion length) per row: unequal lengths, one length-1.
_ROWS = [(3, 3), (2, 4), (3, 1), (2, 2), (3, 2)]
#: Advantages: positive, negative, zero, negative, positive.
_ADV = [1.3, -0.7, 0.0, -1.1, 0.9]
#: Ratio new/old per row: inside the trust region, above it, below it, below it,
#: above it. Row 4 (A>0, 1.3 > 1.2) clips high and row 3 (A<0, 0.6 < 0.8) clips
#: low; row 1 (A<0, 1.5) is above the region on the side that does not clip.
#: 1.3 sits below 1 + 2 * epsilon, so a widened upper bound would not clip it.
_RATIO = [1.1, 1.5, 0.5, 0.6, 1.3]


def _fixture(seed=0, vocab=6):
    """Tokens, the collated mask, fixed logits and per-row synthetic logps."""
    from soup_cli.trainer.mlx_grpo import collate_completions

    rng = np.random.default_rng(seed)
    samples = [(list(rng.integers(3, vocab, p)), list(rng.integers(3, vocab, c)))
               for p, c in _ROWS]
    tokens, mask, start = collate_completions(samples)
    tokens, mask = np.array(tokens), np.array(mask, dtype=np.float64)
    width = tokens.shape[1] - 1
    logits = rng.normal(0.0, 1.5, (len(samples), width, vocab))
    return tokens, mask, start, logits


def _synthetic(logits, tokens, start):
    logps = _np_logps(logits[:, start:], tokens[:, 1:][:, start:])
    old = logps - np.log(np.array(_RATIO))[:, None]
    ref = logps + np.random.default_rng(9).normal(0.0, 0.3, logps.shape)
    return logps, old, ref


class _FixedLogits:
    def __new__(cls, logits):
        nn = pytest.importorskip("mlx.nn")
        mx = pytest.importorskip("mlx.core")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = mx.array(logits.astype(np.float32))

            def __call__(self, x):
                return self.logits

        return Model()


def _mlx_loss(model, tokens, mask, start, *, beta, old=None, ref=None, num_items=None):
    mx = pytest.importorskip("mlx.core")
    from soup_cli.trainer.mlx_grpo import completion_token_logps, grpo_token_loss

    logps = completion_token_logps(model, mx.array(tokens), start)
    return grpo_token_loss(
        logps, mx.array(np.array(_ADV, np.float32)), mx.array(mask.astype(np.float32)),
        num_items=float(mask.sum() if num_items is None else num_items), beta=beta,
        ref_logps=None if ref is None else mx.array(ref.astype(np.float32)),
        old_logps=None if old is None else mx.array(old.astype(np.float32)),
    )


def _fd_grad(f, logits, eps=1e-6):
    grad = np.zeros_like(logits)
    for idx in np.ndindex(*logits.shape):
        up, down = logits.copy(), logits.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2 * eps)
    return grad


class TestTheLossMatchesTheOracle:
    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_value_with_synthetic_old_and_reference(self, beta):
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        loss, _ = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=beta, old=old, ref=ref)
        expected = _np_loss(logps, old, ref, _ADV, mask, beta=beta, num_items=mask.sum())
        assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    @pytest.mark.parametrize("synthetic_old", [True, False])
    def test_gradients_match_finite_differences(self, beta, synthetic_old):
        """Both clip sides, A of either sign and zero, unequal lengths, masked tokens.

        Without a synthetic old policy the ratio is 1 (old = new detached), so the
        oracle holds old fixed at the current log-probs while differentiating.
        """
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture(seed=1)
        logps, old, ref = _synthetic(logits, tokens, start)
        if not synthetic_old:
            old = logps
        model = _FixedLogits(logits)

        def f(m):
            return _mlx_loss(m, tokens, mask, start, beta=beta,
                             old=old if synthetic_old else None, ref=ref)[0]

        _, grads = nn.value_and_grad(model, f)(model)
        targets = tokens[:, 1:][:, start:]

        def oracle(z):
            return _np_loss(_np_logps(z[:, start:], targets), old, ref, _ADV, mask,
                            beta=beta, num_items=mask.sum())

        expected = _fd_grad(oracle, logits)
        np.testing.assert_allclose(np.array(grads["logits"]), expected, atol=2e-5, rtol=1e-3)
        assert np.abs(expected).max() > 1e-3
        mx.eval(grads)

    def test_clipped_tokens_carry_no_policy_gradient(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture()
        logps, old, _ = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        _, grads = nn.value_and_grad(
            model, lambda m: _mlx_loss(m, tokens, mask, start, beta=0.0, old=old)[0]
        )(model)
        g = np.array(grads["logits"])
        # Row 1 (A<0, ratio 1.5) is unclipped; rows 2 (A=0), 3 (A<0, ratio 0.6 < 0.8,
        # clipped low) and 4 (A>0, ratio 1.3 > 1.2, clipped high) give nothing, and
        # row 0 (ratio 1.1) is inside the region.
        assert np.abs(g[0]).max() > 0 and np.abs(g[1]).max() > 0
        assert np.abs(g[2]).max() == 0 and np.abs(g[3]).max() == 0
        assert np.abs(g[4]).max() == 0
        mx.eval(grads)

    def test_clip_metrics_count_each_side(self):
        tokens, mask, start, logits = _fixture()
        _, old, _ = _synthetic(logits, tokens, start)
        _, sums = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=0.0, old=old)
        # Row 3: A<0 and ratio 0.6 < 0.8 -> low; row 4: A>0 and ratio 1.3 > 1.2 -> high;
        # row 1: A<0, ratio 1.5 -> neither side.
        assert sums["clip_low"].item() == _ROWS[3][1]
        assert sums["clip_high"].item() == _ROWS[4][1]
        assert sums["clip_region"].item() == _ROWS[3][1] + _ROWS[4][1]
        assert sums["tokens"].item() == sum(c for _, c in _ROWS)

    def test_beta_zero_needs_no_reference_and_beta_adds_the_k3_kl(self):
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        plain, _ = _mlx_loss(model, tokens, mask, start, beta=0.0, old=old)
        with_kl, sums = _mlx_loss(model, tokens, mask, start, beta=0.25, old=old, ref=ref)
        d = ref - logps
        kl = ((np.exp(d) - d - 1) * mask).sum()
        assert with_kl.item() - plain.item() == pytest.approx(0.25 * kl / mask.sum(), rel=1e-4)
        assert sums["kl"].item() == pytest.approx(kl, rel=1e-4)
        with pytest.raises(ValueError, match="reference"):
            _mlx_loss(model, tokens, mask, start, beta=0.1)

    def test_the_normaliser_is_the_given_token_count_not_a_row_mean(self):
        """DAPO sums tokens; TRL's 'grpo' would average per row first."""
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        loss, _ = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=0.0, old=old,
                            num_items=100.0)
        assert loss.item() == pytest.approx(
            _np_loss(logps, old, ref, _ADV, mask, beta=0.0, num_items=100.0), rel=1e-5
        )
        per_row = -np.minimum(np.exp(logps - old) * np.array(_ADV)[:, None],
                              np.clip(np.exp(logps - old), 0.8, 1.2) * np.array(_ADV)[:, None])
        grpo_style = ((per_row * mask).sum(1) / mask.sum(1)).mean()
        assert abs(loss.item() * 100.0 / mask.sum() - grpo_style) > 1e-3

    def test_prompt_padding_and_post_completion_logits_move_nothing(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture()
        _, old, ref = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        base, _ = _mlx_loss(model, tokens, mask, start, beta=0.1, old=old, ref=ref)
        # Every logit position whose target is not a completion token: the prompt
        # (before `start` and inside longer prompts) and the right padding.
        unused = np.ones(logits.shape[:2], dtype=bool)
        unused[:, start:] = mask == 0
        noisy = logits.copy()
        noisy[unused] += np.random.default_rng(5).normal(0, 50.0, noisy[unused].shape)
        moved, _ = _mlx_loss(_FixedLogits(noisy), tokens, mask, start, beta=0.1, old=old, ref=ref)
        assert moved.item() == pytest.approx(base.item(), abs=1e-6)
        _, grads = nn.value_and_grad(
            model, lambda m: _mlx_loss(m, tokens, mask, start, beta=0.1, old=old, ref=ref)[0]
        )(model)
        assert np.abs(np.array(grads["logits"])[unused]).max() == 0
        mx.eval(grads)


class TestTheOracleIsTrl:
    """The NumPy oracle against TRL 0.29.1's own ``GRPOTrainer._compute_loss``,
    bound to a stub that supplies the per-token log-probs (torch + trl, no MLX)."""

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_loss_and_gradient(self, beta):
        torch = pytest.importorskip("torch")
        pytest.importorskip("trl")
        from collections import defaultdict
        from types import SimpleNamespace

        from trl.trainer.grpo_trainer import GRPOTrainer

        tokens, mask, start, logits = _fixture(seed=2)
        logps, old, ref = _synthetic(logits, tokens, start)
        leaf = torch.tensor(logps, dtype=torch.float64, requires_grad=True)
        stub = SimpleNamespace(
            _get_per_token_logps_and_entropies=lambda *a, **k: (leaf, torch.zeros_like(leaf)),
            top_entropy_quantile=1.0, off_policy_mask_threshold=None,
            importance_sampling_level="token", beta=beta, loss_type="dapo",
            epsilon_low=0.2, epsilon_high=0.2, use_vllm=False,
            vllm_importance_sampling_correction=True,
            args=SimpleNamespace(use_bias_correction_kl=False, delta=None),
            model=SimpleNamespace(training=True),
            accelerator=SimpleNamespace(num_processes=1, gather=lambda t: t),
            _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        )
        width = mask.shape[1]
        inputs = {
            "prompt_ids": torch.zeros((len(_ROWS), 1), dtype=torch.long),
            "prompt_mask": torch.ones((len(_ROWS), 1), dtype=torch.long),
            "completion_ids": torch.zeros((len(_ROWS), width), dtype=torch.long),
            "completion_mask": torch.tensor(mask, dtype=torch.float64),
            "advantages": torch.tensor(_ADV, dtype=torch.float64),
            "old_per_token_logps": torch.tensor(old, dtype=torch.float64),
            "ref_per_token_logps": torch.tensor(ref, dtype=torch.float64),
            "num_items_in_batch": torch.tensor(float(mask.sum()) + 3.0, dtype=torch.float64),
        }
        loss = GRPOTrainer._compute_loss(stub, None, inputs)
        loss.backward()
        num_items = float(mask.sum()) + 3.0
        assert loss.item() == pytest.approx(
            _np_loss(logps, old, ref, _ADV, mask, beta=beta, num_items=num_items), rel=1e-12
        )

        def oracle(z):
            return _np_loss(z, old, ref, _ADV, mask, beta=beta, num_items=num_items)

        np.testing.assert_allclose(leaf.grad.numpy(), _fd_grad(oracle, logps), atol=1e-7)
        if beta:
            d = ref - logps
            assert stub._metrics["train"]["kl"][0] == pytest.approx(
                ((np.exp(d) - d - 1) * mask).sum() / mask.sum(), rel=1e-9
            )


#: GRPOConfig fields mlx_grpo.py trains at TRL's default, so the transformers path
#: must never pass them (review of #2182).
_CONTRACT_FIELDS = frozenset({
    "loss_type", "scale_rewards", "epsilon", "epsilon_high", "num_iterations",
    "importance_sampling_level", "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "mask_truncated_completions", "delta",
})


class TestTheTrlContract:
    """mlx_grpo.py's constants are TRL 0.29.1's ``GRPOConfig`` defaults, and Soup's
    transformers path overrides none of them, as tests/test_mlx_kto.py pins KTO."""

    def test_the_contract_is_grpoconfigs_defaults(self):
        trl = pytest.importorskip("trl")
        from soup_cli.trainer import mlx_grpo

        fields = trl.GRPOConfig.__dataclass_fields__
        assert fields["loss_type"].default == mlx_grpo.LOSS_TYPE == "dapo"
        assert fields["scale_rewards"].default == mlx_grpo.SCALE_REWARDS == "group"
        assert fields["epsilon"].default == mlx_grpo.EPSILON == 0.2
        assert fields["epsilon_high"].default is None  # None means epsilon
        assert mlx_grpo.EPSILON_HIGH == fields["epsilon"].default
        assert fields["num_iterations"].default == mlx_grpo.NUM_ITERATIONS == 1
        assert (fields["importance_sampling_level"].default
                == mlx_grpo.IMPORTANCE_SAMPLING_LEVEL == "token")
        assert fields["temperature"].default == mlx_grpo.TEMPERATURE == 1.0
        assert fields["top_p"].default == mlx_grpo.TOP_P == 1.0
        assert fields["top_k"].default == mlx_grpo.TOP_K == 0
        assert fields["min_p"].default is None and mlx_grpo.MIN_P is None
        assert fields["repetition_penalty"].default == mlx_grpo.REPETITION_PENALTY == 1.0
        assert fields["mask_truncated_completions"].default is False
        assert mlx_grpo.MASK_TRUNCATED_COMPLETIONS is False
        assert fields["delta"].default is None  # no two-sided clipping by default
        assert set(_CONTRACT_FIELDS) <= set(fields)

    def test_the_recorded_contract_is_the_constants(self):
        from soup_cli.trainer import mlx_grpo

        contract = mlx_grpo.grpo_contract(beta=0.1, num_generations=4,
                                          max_completion_length=64)
        assert (contract["epsilon"], contract["epsilon_high"]) == (0.2, 0.2)
        assert (contract["loss_type"], contract["scale_rewards"], contract["num_iterations"],
                contract["importance_sampling_level"]) == ("dapo", "group", 1, "token")

    def test_the_transformers_path_passes_none_of_them(self):
        import ast
        from pathlib import Path

        import soup_cli.trainer.grpo as grpo

        tree = ast.parse(Path(grpo.__file__).read_text(encoding="utf-8"))
        passed: set = set()
        built = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and target.id == "grpo_kwargs"
                            and isinstance(node.value, ast.Dict)):
                        built += 1
                        passed |= {k.value for k in node.value.keys
                                   if isinstance(k, ast.Constant)}
                    elif (isinstance(target, ast.Subscript)
                          and getattr(target.value, "id", "") == "grpo_kwargs"):
                        passed.add(target.slice.value)
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "GRPOConfig":
                passed |= {kw.arg for kw in node.keywords if kw.arg}
        assert built == 1
        assert {"beta", "num_generations", "max_completion_length"} <= passed
        assert not passed & _CONTRACT_FIELDS


# --------------------------------------------------------------------------
# A tiny random Llama
# --------------------------------------------------------------------------


def _tiny_llama(seed=0, vocab=VOCAB):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    mx.random.seed(seed)
    return llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=vocab,
        tie_word_embeddings=True,
    ))


def _attach_lora(model, rank=4, scale=2.0, dropout=0.0):
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {
        "rank": rank, "scale": scale, "dropout": dropout,
        "keys": {"self_attn.q_proj", "self_attn.v_proj"},
    })
    return model


def _flat(params):
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(params)}


class _Scripted:
    """A sampler that emits a fixed token per row per step and counts its calls."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def __call__(self, logprobs):
        mx = pytest.importorskip("mlx.core")
        step = self.calls
        self.calls += 1
        return mx.array([row[step] if step < len(row) else 9 for row in self.script])


class TestGeneration:
    def test_a_group_per_prompt_cut_at_the_first_eos(self):
        from soup_cli.trainer.mlx_grpo import generate_group

        script = _Scripted([[5, EOS, 7, 7], [5, 6, 7, 8], [EOS, 9, 9, 9]])
        out = generate_group(_tiny_llama(), [3, 4, 5], num_generations=3, max_tokens=4,
                             eos_token_id=EOS, sampler=script)
        assert out == [([5, EOS], True), ([5, 6, 7, 8], False), ([EOS], True)]
        assert script.calls == 4

    def test_stops_once_every_row_has_ended(self):
        from soup_cli.trainer.mlx_grpo import generate_group

        script = _Scripted([[5, EOS], [EOS, 3]])
        out = generate_group(_tiny_llama(), [3, 4], num_generations=2, max_tokens=50,
                             eos_token_id=EOS, sampler=script)
        assert out == [([5, EOS], True), ([EOS], True)] and script.calls == 2

    def test_seeded_rows_reproduce_and_differ_within_a_group(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_grpo import generate_group

        model = _tiny_llama()

        def run(seed):
            mx.random.seed(seed)
            return generate_group(model, [3, 4, 5], num_generations=4, max_tokens=12,
                                  eos_token_id=EOS)

        first = run(0)
        assert len(first) == 4 and first == run(0) and first != run(1)
        assert len({tuple(t) for t, _ in first}) > 1

    def test_the_kv_cache_loop_matches_a_full_forward(self):
        """Each sampled token is drawn from the model's distribution given the prefix."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_grpo import generate_group

        model = _tiny_llama()
        model.eval()
        seen = []

        def greedy(logprobs):
            seen.append(np.array(logprobs))
            return mx.argmax(logprobs, axis=-1)

        prompt = [3, 4, 5, 6]
        (tokens, _), = generate_group(model, prompt, num_generations=1, max_tokens=5,
                                      eos_token_id=EOS, sampler=greedy)
        full = model(mx.array([prompt + tokens]))
        full = full - mx.logsumexp(full, axis=-1, keepdims=True)
        for k, step in enumerate(seen):
            np.testing.assert_allclose(step[0], np.array(full[0, len(prompt) - 1 + k]),
                                       atol=1e-4)

    def test_the_gold_never_reaches_the_prompt(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        rows = _prepare_grpo_dataset([{"messages": [
            {"role": "user", "content": "what is 2 3?"},
            {"role": "assistant", "content": "the answer is 97531"},
        ]}])
        tok = DigitTokenizer()
        ids = MLXGRPOTrainerWrapper._render(tok, rows[0]["prompt"], "train", 0)
        text = tok.decode(ids)
        assert rows[0]["answer"] == "the answer is 97531"
        assert "97531" not in text and text.endswith("<a>")


class TestReference:
    def _model(self):
        mx = pytest.importorskip("mlx.core")
        model = _tiny_llama()
        x = mx.array([[3, 4, 5, 6, 7]])
        base = np.array(model(x))
        _attach_lora(model)
        for _, m in model.named_modules():
            if hasattr(m, "lora_b"):
                m.lora_b = mx.random.normal(m.lora_b.shape) * 0.5
        return model, x, base

    def test_adapter_off_is_the_base_model_exactly(self):
        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model, x, base = self._model()
        assert np.abs(np.array(model(x)) - base).max() > 1e-4
        with adapter_disabled(model) as count:
            np.testing.assert_array_equal(np.array(model(x)), base)
        assert count == 2 * len(model.layers)
        assert np.abs(np.array(model(x)) - base).max() > 1e-4

    def test_scales_come_back_after_an_exception(self):
        from soup_cli.trainer.mlx_grpo import _lora_modules, adapter_disabled

        model, _, _ = self._model()
        before = [m.scale for m in _lora_modules(model)]
        with pytest.raises(KeyError):
            with adapter_disabled(model):
                raise KeyError("boom")
        assert [m.scale for m in _lora_modules(model)] == before == [2.0] * len(before)

    def test_parameters_and_optimizer_state_are_untouched(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten

        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model, x, _ = self._model()
        opt = optim.AdamW(learning_rate=1e-2)
        _, grads = nn.value_and_grad(model, lambda m: m(x).sum())(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        params = _flat(model.parameters())
        state = {k: np.array(v) for k, v in tree_flatten(opt.state)}
        trainable = set(_flat(model.trainable_parameters()))
        with adapter_disabled(model):
            mx.eval(model(x))
        assert set(_flat(model.trainable_parameters())) == trainable
        for k, v in _flat(model.parameters()).items():
            np.testing.assert_array_equal(v, params[k])
        for k, v in tree_flatten(opt.state):
            np.testing.assert_array_equal(np.array(v), state[k])

    def test_a_model_without_an_adapter_is_refused(self):
        from soup_cli.trainer.mlx_grpo import adapter_disabled

        with pytest.raises(RuntimeError, match="no LoRA module"):
            with adapter_disabled(_tiny_llama()):
                pass


# --------------------------------------------------------------------------
# Training through the real wrapper
# --------------------------------------------------------------------------

#: The training vocabulary: ``<think>`` and ``</think>`` are single tokens, so a
#: random model's completions earn 0, 0.5 or 1.0 from the ``format`` reward and
#: rewards differ inside a group. Prompt characters and digits fill the rest.
_TRAIN_CHARS = ["<think>", "</think>", "x", " ", "<", ">", "|", "u", "a", "q", "?",
                *"0123456789"]
_TRAIN_VOCAB = len(_TRAIN_CHARS) + 3

_GRPO_YAML = """
base: tiny-base
task: grpo
backend: mlx
data:
  train: ./x.jsonl
  max_length: 64
training:
  epochs: 1
  lr: 0.05
  batch_size: 4
  gradient_accumulation_steps: 1
  num_generations: 4
  grpo_beta: 0.1
  reward_fn: format
  warmup_ratio: 0.0
  logging_steps: 1
  quantization: none
  lora: {r: 4, alpha: 8, dropout: 0.0}
output: ./output
"""


_WITH_CHATML = "  max_length: 64\n  chat_template: chatml\n"


def _prompt_rows(n=8, prefix="q"):
    return [{"prompt": f"{prefix}{i}?", "answer": str(i % 10)} for i in range(n)]


def _config(**training):
    from soup_cli.config.loader import load_config_from_string

    text = _GRPO_YAML
    for key, value in training.items():
        line = f"  {key}: {value}\n"
        own = [x for x in text.splitlines(keepends=True) if x.startswith(f"  {key}:")]
        text = text.replace(own[0], line) if own else text.replace(
            "  quantization: none\n", "  quantization: none\n" + line)
    return load_config_from_string(text)


def _wrapper(tmp_path, monkeypatch, *, train=None, val=None, model_seed=0, **training):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    wrapper = MLXGRPOTrainerWrapper(_config(**training))
    wrapper.model = _tiny_llama(seed=model_seed, vocab=_TRAIN_VOCAB)
    wrapper.tokenizer = DigitTokenizer(_TRAIN_CHARS)
    wrapper._dataset = {"train": _prompt_rows() if train is None else train,
                        "val": [] if val is None else val}
    return wrapper


class _RecordingOptimizer:
    """Delegates to the real (clipping) optimizer; snapshots after each update."""

    def __init__(self, inner):
        self.__dict__["inner"] = inner
        self.__dict__["snapshots"] = []

    def update(self, model, grads):
        mx = pytest.importorskip("mlx.core")
        from mlx.utils import tree_flatten

        out = self.inner.update(model, grads)
        mx.eval(model.parameters(), self.inner.state)
        self.snapshots.append((_flat(model.trainable_parameters()),
                               {k: np.array(v) for k, v in tree_flatten(self.inner.state)}))
        return out

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)


def _record_optimizer(monkeypatch):
    from soup_cli.trainer import mlx_grpo

    made = []
    real = mlx_grpo._clipping_optimizer

    def build(inner, max_norm):
        made.append(_RecordingOptimizer(real(inner, max_norm)))
        return made[-1]

    monkeypatch.setattr(mlx_grpo, "_clipping_optimizer", build)
    return made


def _policy_logps(model, samples, adapter=True):
    mx = pytest.importorskip("mlx.core")
    from soup_cli.trainer.mlx_grpo import (
        adapter_disabled,
        collate_completions,
        completion_token_logps,
    )

    tokens, mask, start = collate_completions(samples)
    model.eval()
    if adapter:
        out = completion_token_logps(model, mx.array(tokens), start)
    else:
        with adapter_disabled(model):
            out = completion_token_logps(model, mx.array(tokens), start)
    return np.array(out), np.array(mask, dtype=np.float64)


class TestTraining:
    def test_lora_trains_the_base_stays_frozen_and_rewards_differ(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        before = _flat(wrapper.model.parameters())
        result = wrapper.train()
        after = _flat(wrapper.model.parameters())
        lora_b = [v for k, v in after.items() if k.endswith("lora_b")]
        assert lora_b and any(np.abs(v).sum() > 0 for v in lora_b)
        for name, value in before.items():
            key = name if name in after else name.replace(".weight", ".linear.weight")
            np.testing.assert_array_equal(after[key], value, err_msg=name)
        assert result["total_steps"] == 8 and len(result["losses"]) == 8
        assert all(math.isfinite(x) for x in result["losses"])
        assert any(h["frac_reward_zero_std"] < 1.0 for h in result["history"])
        assert result["skipped_steps"] == 0
        assert not list((tmp_path / "output").glob("*.partial*"))

    def test_one_update_raises_the_advantage_weighted_log_likelihood(self, tmp_path, monkeypatch):
        """beta 0, SGD: the update ascends sum_i A_i log p(completion_i)."""
        from soup_cli.trainer import mlx_grpo

        seen = {}
        real_rollout = mlx_grpo.MLXGRPOTrainerWrapper._rollout
        real_adv = mlx_grpo.group_advantages

        def rollout(self, hf, prompt_ids, eos, max_new):
            out = real_rollout(self, hf, prompt_ids, eos, max_new)
            g = self._units.num_generations
            seen["samples"] = [(prompt_ids[k // g], c) for k, c in enumerate(out[0])]
            return out

        def advantages(rewards, group):
            seen["adv"] = real_adv(rewards, group)
            return seen["adv"]

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", rollout)
        monkeypatch.setattr(mlx_grpo, "group_advantages", advantages)
        rows = _prompt_rows(1)
        wrapper = _wrapper(tmp_path, monkeypatch, train=rows, grpo_beta=0.0, optimizer="sgd",
                           scheduler="constant", lr=0.02, seed=6)
        wrapper.train()
        adv = np.array(seen["adv"])
        assert np.abs(adv).max() > 0.5, "fixture must give unequal rewards in the group"
        after, mask = _policy_logps(wrapper.model, seen["samples"])
        # The adapter's B started at zero: with it switched off this is the policy
        # the completions were sampled from.
        before, _ = _policy_logps(wrapper.model, seen["samples"], adapter=False)
        objective = lambda lp: float((adv[:, None] * lp * mask).sum())  # noqa: E731
        assert objective(after) > objective(before) + 1e-3
        moved = ((after - before) * mask).sum(axis=1)
        assert all(np.sign(moved[i]) == np.sign(a) for i, a in enumerate(adv) if abs(a) > 0.5)

    def test_micro_batches_accumulate_to_one_batch_without_dividing(self, tmp_path, monkeypatch):
        """DAPO: one normaliser, the token count of the whole update; no /accum."""
        def run(path, batch_size, accum):
            wrapper = _wrapper(path, monkeypatch, train=_prompt_rows(2), batch_size=batch_size,
                               gradient_accumulation_steps=accum, optimizer="sgd",
                               scheduler="constant", seed=7)
            result = wrapper.train()
            return result, _flat(wrapper.model.trainable_parameters())

        one, w1 = run(tmp_path / "a", 8, 1)
        two, w2 = run(tmp_path / "b", 4, 2)
        assert len(one["losses"]) == len(two["losses"]) == 1
        assert one["num_items"] == two["num_items"]
        assert one["losses"][0] == pytest.approx(two["losses"][0], abs=1e-6)
        assert max(np.abs(v).max() for k, v in w1.items() if k.endswith("lora_b")) > 1e-4
        for name in w1:
            np.testing.assert_allclose(w1[name], w2[name], atol=1e-6, err_msg=name)

    def test_a_fixed_seed_reproduces_the_run(self, tmp_path, monkeypatch):
        def run(path, seed):
            wrapper = _wrapper(path, monkeypatch, seed=seed)
            return wrapper.train(), _flat(wrapper.model.trainable_parameters())

        a, wa = run(tmp_path / "a", 3)
        b, wb = run(tmp_path / "b", 3)
        c, _ = run(tmp_path / "c", 4)
        assert a["losses"] == b["losses"] and a["losses"] != c["losses"]
        for name in wa:
            np.testing.assert_array_equal(wa[name], wb[name])

    @pytest.mark.parametrize("poison", ["loss", "gradient"])
    def test_a_nonfinite_update_is_skipped_and_leaves_everything(self, tmp_path, monkeypatch,
                                                                 poison):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_grpo

        made = _record_optimizer(monkeypatch)
        real = mlx_grpo.grpo_token_loss
        calls = []

        def poisoned(logps, *args, **kwargs):
            loss, sums = real(logps, *args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                if poison == "loss":
                    loss = loss * mx.array(float("nan"))
                else:  # a finite value with a NaN gradient
                    loss = loss + mx.where(mx.array(False), mx.sqrt(-1.0 - mx.abs(logps.sum())),
                                           mx.array(0.0))
            return loss, sums

        monkeypatch.setattr(mlx_grpo, "grpo_token_loss", poisoned)
        wrapper = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(2))
        result = wrapper.train()
        assert result["skipped_steps"] == 1 and len(result["losses"]) == 1
        snapshots = made[0].snapshots
        assert len(snapshots) == 1, "the poisoned update must not reach the optimizer"
        weights, state = snapshots[0]
        for name, value in _flat(wrapper.model.trainable_parameters()).items():
            np.testing.assert_array_equal(value, weights[name], err_msg=name)
        from mlx.utils import tree_flatten

        final_state = {k: np.array(v) for k, v in tree_flatten(made[0].state)}
        assert final_state.keys() == state.keys()
        for name, value in state.items():
            np.testing.assert_array_equal(final_state[name], value, err_msg=name)
        assert int(final_state["step"]) == 1
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["skipped_steps"] == 1

    def test_beta_zero_runs_no_reference_forward(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_grpo

        def refuse(*args, **kwargs):
            raise AssertionError("reference forward with beta = 0")

        monkeypatch.setattr(mlx_grpo, "adapter_disabled", refuse)
        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_reference_logps", refuse)
        result = _wrapper(tmp_path, monkeypatch, grpo_beta=0.0).train()
        assert "kl" not in result and len(result["losses"]) == 8

    def test_beta_runs_the_reference_once_per_micro_batch(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_grpo

        calls = []
        real = mlx_grpo.MLXGRPOTrainerWrapper._reference_logps

        def counted(self, tokens, start):
            calls.append(int(tokens.shape[0]))
            return real(self, tokens, start)

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_reference_logps", counted)
        result = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(4), batch_size=4,
                          gradient_accumulation_steps=2).train()
        assert calls == [4, 4, 4, 4] and result["kl"] >= 0.0

    def test_the_reference_stays_the_base_model_after_updates(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_grpo import completion_token_logps

        wrapper = _wrapper(tmp_path, monkeypatch, lr=0.1)
        samples = [([3, 4, 5], [6, 7, EOS]), ([3, 4], [8, 9, 10, EOS])]
        from soup_cli.trainer.mlx_grpo import collate_completions

        tokens, _, start = collate_completions(samples)
        wrapper.model.eval()
        base = np.array(completion_token_logps(wrapper.model, mx.array(tokens), start))
        wrapper.train()
        ref, _ = _policy_logps(wrapper.model, samples, adapter=False)
        policy, _ = _policy_logps(wrapper.model, samples)
        np.testing.assert_array_equal(ref, base)
        assert np.abs(policy - base).max() > 1e-3

    def test_validation_reward_is_reported_and_changes_nothing(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, val=_prompt_rows(3, prefix="v"))
        result = wrapper.train()
        assert 0.0 <= result["val_reward"] <= 1.0 and "val_reward_std" in result
        saved = mx.load(str(tmp_path / "output" / "adapters.safetensors"))
        for name, value in _flat(wrapper.model.trainable_parameters()).items():
            np.testing.assert_array_equal(np.array(saved[name]), value, err_msg=name)
        again = _wrapper(tmp_path / "again", monkeypatch, val=_prompt_rows(3, prefix="v")).train()
        assert again["val_reward"] == result["val_reward"]

    def test_metadata_is_complete_and_written_before_the_first_update(self, tmp_path,
                                                                     monkeypatch):
        from soup_cli.trainer import mlx_grpo

        early = []
        real = mlx_grpo.MLXGRPOTrainerWrapper._rollout

        def rollout(self, *args):
            path = tmp_path / "output" / "adapter_config.json"
            if not early:
                early.append(json.loads(path.read_text()))
            return real(self, *args)

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", rollout)
        _wrapper(tmp_path, monkeypatch, train=_prompt_rows(2)).train()
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        expected = {
            "task": "grpo", "loss_type": "dapo", "scale_rewards": "group", "epsilon": 0.2,
            "epsilon_high": 0.2, "num_iterations": 1, "importance_sampling_level": "token",
            "beta": 0.1, "num_generations": 4, "max_completion_length": 64,
            "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": None,
            "repetition_penalty": 1.0, "mask_truncated_completions": False,
            "advantage_epsilon": 1e-4, "reward_fn": "format", "skipped_steps": 0,
            "prompts_per_update": 1, "completions_per_update": 4, "fine_tune_type": "lora",
        }
        assert {k: meta[k] for k in expected} == expected
        for key in ("advantage_std", "reference", "generation", "watchdog", "seed"):
            assert meta[key] not in (None, "")
        assert "sample" in meta["advantage_std"] and "LoRA scale 0" in meta["reference"]
        assert early and early[0]["task"] == "grpo" and early[0]["loss_type"] == "dapo"

    def test_data_seed_is_recorded_as_configured_and_moves_nothing(self, tmp_path,
                                                                   monkeypatch):
        """The order follows training.seed, as TRL's RepeatSampler does; data_seed
        is written as the user set it (None when unset), not as the seed."""
        def run(path, **training):
            wrapper = _wrapper(path, monkeypatch, train=_prompt_rows(2), seed=3, **training)
            result = wrapper.train()
            meta = json.loads((path / "output" / "adapter_config.json").read_text())
            return result, meta

        plain, meta = run(tmp_path / "a")
        assert (meta["seed"], meta["data_seed"]) == (3, None)
        seeded, meta = run(tmp_path / "b", data_seed=11)
        assert (meta["seed"], meta["data_seed"]) == (3, 11)
        assert seeded["losses"] == plain["losses"]

    def test_beta_zero_metadata_names_no_reference(self, tmp_path, monkeypatch):
        _wrapper(tmp_path, monkeypatch, train=_prompt_rows(1), grpo_beta=0.0).train()
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert meta["beta"] == 0.0 and meta["reference"].startswith("none")

    def test_prompts_keep_their_own_gold_in_the_loop(self, tmp_path, monkeypatch):
        """Shuffled order, identical prompt text with different golds."""
        rows = [{"prompt": "q?", "answer": str(i), "rid": i} for i in range(4)]
        wrapper = _wrapper(tmp_path, monkeypatch, train=rows, epochs=2, reward_fn="accuracy")
        wrapper._prepare(wrapper._dataset)
        seen = []

        def reward(completions, answer, rid, **_):
            seen.extend(zip(answer, rid))
            return [0.0] * len(completions)

        from soup_cli.trainer.rewards import validate_reward_funcs

        wrapper._reward_fn = validate_reward_funcs(reward)
        wrapper.train()
        assert len(seen) == 2 * 4 * 4
        assert all(answer == str(rid) for answer, rid in seen)
        order = [rid for _, rid in seen[::4]]
        assert sorted(order[:4]) == [0, 1, 2, 3] and order != [0, 1, 2, 3, 0, 1, 2, 3]

    def test_a_prompt_too_long_for_the_context_is_refused(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper.model.args.max_position_embeddings = 70
        with pytest.raises(ValueError, match="exceeds the model's context of 70"):
            wrapper.train()

    def test_resume_is_refused(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="resume is not implemented for MLX GRPO"):
            wrapper.train(resume_from_checkpoint=str(tmp_path / "x.safetensors"))

    def test_a_quantized_base_is_refused(self, tmp_path, monkeypatch):
        """`quantization: 4bit` is the default and only names a pre-quantized base on
        MLX, so the base itself is checked once loaded (Step 2 adds 4-bit)."""
        nn = pytest.importorskip("mlx.nn")
        wrapper = _wrapper(tmp_path, monkeypatch)
        nn.quantize(wrapper.model, group_size=32, bits=4)
        with pytest.raises(ValueError, match="trains a dense base"):
            wrapper.train()

    def test_setup_refuses_a_quantized_base_right_after_loading(self, tmp_path, monkeypatch):
        nn = pytest.importorskip("mlx.nn")
        import soup_cli.utils.mlx as mlx_utils
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        model = _tiny_llama(vocab=_TRAIN_VOCAB)
        nn.quantize(model, group_size=32, bits=4)
        monkeypatch.setattr(mlx_utils, "load_mlx_model",
                            lambda *a, **k: (model, DigitTokenizer(_TRAIN_CHARS)))
        wrapper = MLXGRPOTrainerWrapper(_config(quantization="4bit"))
        with pytest.raises(ValueError, match="is quantized"):
            wrapper.setup({"train": _prompt_rows(2)})


class TestFreshProcessReload:
    def test_logits_reproduce_and_the_adapter_has_an_effect(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")

        wrapper = _wrapper(tmp_path, monkeypatch, lr=0.1)
        base_path = tmp_path / "base.safetensors"
        wrapper.model.save_weights(str(base_path))
        wrapper.train()
        tokens = np.array([[3, 4, 5, 6, 7, 8, 9]])
        wrapper.model.eval()
        logits = np.array(wrapper.model(mx.array(tokens)))
        script = textwrap.dedent(f"""
            import json, sys
            import numpy as np
            import mlx.core as mx
            from mlx_lm.models import llama
            from mlx_lm.tuner.utils import load_adapters
            args = llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-5, vocab_size={_TRAIN_VOCAB}, tie_word_embeddings=True)
            model = llama.Model(args)
            model.load_weights({str(base_path)!r})
            if sys.argv[1] == "adapter":
                load_adapters(model, {str(tmp_path / "output")!r})
            model.eval()
            out = model(mx.array({tokens.tolist()!r}))
            print(json.dumps(np.array(out).tolist()))
        """)

        def run(mode):
            out = subprocess.run([sys.executable, "-c", script, mode], capture_output=True,
                                 text=True, check=True, timeout=300)
            return np.array(json.loads(out.stdout.strip().splitlines()[-1]))

        reloaded, base = run("adapter"), run("base")
        np.testing.assert_allclose(reloaded, logits, rtol=1e-5, atol=1e-5)
        assert np.abs(base - reloaded).max() > 1e-3


# --------------------------------------------------------------------------
# Data and setup: everything refused before the model loads (no MLX needed)
# --------------------------------------------------------------------------


class _NoLoad:
    def __init__(self, monkeypatch):
        import soup_cli.utils.mlx as mlx_utils

        def load(*args, **kwargs):
            raise AssertionError("the model must not load")

        monkeypatch.setattr(mlx_utils, "load_mlx_model", load)


def _setup_wrapper(monkeypatch, **training):
    from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

    wrapper = MLXGRPOTrainerWrapper(_config(**training))
    monkeypatch.setattr(wrapper, "_require_mlx", lambda: None)
    _NoLoad(monkeypatch)
    return wrapper


class TestData:
    def test_string_messages_and_alpaca_rows_become_prompts_with_golds(self, monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, reward_fn="accuracy", batch_size=4)
        rows = [
            {"prompt": "add 1 2", "answer": "3"},
            {"messages": [{"role": "user", "content": "add 2 2"},
                          {"role": "assistant", "content": "#### 4"}]},
            {"messages": [{"role": "system", "content": "be brief"},
                          {"role": "user", "content": "add 2 3"},
                          {"role": "assistant", "content": "5"}], "answer": "5"},
            {"instruction": "add 3 3", "output": "6"},
        ]
        wrapper._prepare({"train": rows})
        prompts = wrapper._prompts["train"]
        assert prompts[0]["prompt"] == [{"role": "user", "content": "add 1 2"}]
        assert prompts[1]["prompt"] == [{"role": "user", "content": "add 2 2"}]
        assert prompts[1]["answer"] == "#### 4"
        assert prompts[2]["prompt"][0]["role"] == "system" and len(prompts[2]["prompt"]) == 2
        assert [p["answer"] for p in prompts] == ["3", "#### 4", "5", "6"]

    def test_a_missing_gold_is_refused_before_the_model_loads(self, monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, reward_fn="accuracy")
        rows = _prompt_rows(4) + [{"prompt": "q?"}]
        with pytest.raises(ValueError, match="row 4 is missing or empty 'answer'"):
            wrapper.setup({"train": rows})

    def test_an_unreadable_validation_gold_is_refused_before_the_model_loads(self, monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, reward_fn="accuracy")
        val = [{"prompt": "q?", "answer": "either 41 or 42 is the answer is 41 or 42"}]
        with pytest.raises(ValueError, match="GRPO validation row 0"):
            wrapper.setup({"train": _prompt_rows(4), "val": val})

    def test_too_few_prompts_for_one_update_are_refused_before_the_model_loads(self,
                                                                             monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, batch_size=4, gradient_accumulation_steps=2)
        with pytest.raises(ValueError, match="1 prompt.*fewer than the 2"):
            wrapper.setup({"train": _prompt_rows(1)})

    def test_partial_groups_are_refused_before_the_model_loads(self, monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, batch_size=6)
        with pytest.raises(ValueError, match="must be divisible by num_generations"):
            wrapper.setup({"train": _prompt_rows(8)})

    @pytest.mark.parametrize("spec, name", [("accuracy", "accuracy"),
                                            ("' accuracy'", "accuracy"),
                                            ("'accuracy '", "accuracy"),
                                            ("format", "format")])
    def test_the_reward_name_is_stripped_as_at_load(self, monkeypatch, spec, name):
        """The load check strips reward_fn, and so does the transformers path."""
        wrapper = _setup_wrapper(monkeypatch, reward_fn=spec)
        wrapper._prepare({"train": _prompt_rows(4)})
        assert wrapper._reward_name == name

    def test_batch_size_auto_is_one_group(self, monkeypatch):
        wrapper = _setup_wrapper(monkeypatch, batch_size="auto", gradient_accumulation_steps=1)
        wrapper._prepare({"train": _prompt_rows(2)})
        assert wrapper._units.completions_per_step == 4 and wrapper._units.bumped

    @pytest.mark.parametrize("kwarg, flag", [("deepspeed_config", "--deepspeed"),
                                             ("fsdp_config", "--fsdp")])
    def test_distributed_launch_options_are_refused(self, kwarg, flag):
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        with pytest.raises(ValueError, match=flag):
            MLXGRPOTrainerWrapper(_config(), **{kwarg: {"x": 1}})
        MLXGRPOTrainerWrapper(_config(), deepspeed_config=None, fsdp_config=None, device="mps")

    def test_the_chat_template_override_and_fallback_are_applied(self):
        from soup_cli.trainer.mlx_grpo import FALLBACK_CHAT_TEMPLATE, MLXGRPOTrainerWrapper

        class Bare(DigitTokenizer):
            chat_template = None

        class Wrapped:  # mlx-lm's TokenizerWrapper keeps the HF tokenizer here
            def __init__(self, inner):
                self._tokenizer = inner

        wrapper = MLXGRPOTrainerWrapper(_config())
        wrapper.tokenizer = Wrapped(Bare())
        assert wrapper._template().chat_template == FALLBACK_CHAT_TEMPLATE
        from soup_cli.config.loader import load_config_from_string

        text = _GRPO_YAML.replace("  max_length: 64\n", _WITH_CHATML)
        wrapper = MLXGRPOTrainerWrapper(load_config_from_string(text))
        wrapper.tokenizer = Wrapped(Bare())
        assert "<|im_start|>" in wrapper._template().chat_template


# --------------------------------------------------------------------------
# Config, dispatch, doctor and the CLI
# --------------------------------------------------------------------------


#: One load-time cell per ``_MLX_GRPO_UNAPPLIED_PREFIXES`` entry, on a field the
#: older gates do not refuse first.
_PREFIX_CELLS = ["reward_hack_halt: true", "echo_trap_threshold: 0.7",
                 "rl_checkpoint_keep_last: 2", "curriculum_buckets: 8"]


class TestConfig:
    def test_grpo_on_mlx_loads_with_inactive_defaults_and_dispatches(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper
        from soup_cli.trainer.mlx_routing import get_mlx_trainer

        minimal = """
base: tiny-base
task: grpo
backend: mlx
data: {train: ./x.jsonl}
training:
  quantization: none
output: ./output
"""
        cfg = load_config_from_string(minimal)
        assert (cfg.task, cfg.backend, cfg.training.grpo_beta) == ("grpo", "mlx", 0.1)
        assert get_mlx_trainer("grpo") is MLXGRPOTrainerWrapper

    @pytest.mark.parametrize("line, match", [
        ("quantization: 8bit", "quantization='8bit' is not supported by task='grpo'"),
        ("lora: {r: 0}", "lora.r must be at least 1 for task='grpo'"),
        ("lora: {r: 4, use_dora: true}", "MLX DoRA requires task='sft'"),
        ("reward_fn: ./my_reward.py", "does not load custom reward files"),
        ("reward_fn: accuracy,format", "is a reward ensemble"),
        ("eval_steps: 5", "training.eval_steps is not applied by task='grpo' on backend='mlx'"),
        ("echo_trap_enabled: true", "training.echo_trap_enabled is not applied"),
        ("rl_checkpoint_save_every_steps: 5", "rl_checkpoint_save_every_steps is not applied"),
        ("curriculum: true", "training.curriculum is not applied"),
        ("grpo_variant: gspo", "grpo_variant is not supported on backend=mlx: MLX GRPO trains "
         "TRL's default objective (loss_type='dapo') only. Remove training.grpo_variant, or "
         "use backend: transformers."),
        ("prm_reward: ./prm", "prm_reward requires backend='transformers'"),
        ("vllm_sleep_mode: true", "vllm_sleep_mode requires backend"),
        ("use_lorafa: true", "use_lorafa requires backend='transformers'"),
        ("loraplus_lr_ratio: 16", "loraplus_lr_ratio is not implemented on backend='mlx'"),
        ("relora_steps: 10", "relora_steps is not supported on the mlx backend"),
        ("stream_layers: true", "stream_layers cannot be used with task='grpo'"),
    ])
    def test_each_unsupported_setting_is_refused_at_load(self, line, match):
        key, value = line.split(": ", 1)
        with pytest.raises(Exception, match=match.replace("[", r"\[").replace("]", r"\]")
                           .replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")):
            _config(**{key: value})

    @pytest.mark.parametrize("line", [
        "tis_threshold: 2.0", "skip_zero_advantage: true", "off_policy_mask_threshold: 0.5",
        "grpo_fp16: true", "async_grpo_prefetch: true", "ref_model_ema_alpha: 0.9",
        "replay_buffer_size: 8", "defer_rerolling: true", "rollout_backend: openenv",
    ])
    def test_older_gates_defer_to_the_mlx_grpo_refusal_with_its_remedy(self, line):
        """The v0.50.0 gates used to answer first, with no remedy (review of #1976)."""
        key, value = line.split(": ", 1)
        with pytest.raises(Exception) as info:
            _config(**{key: value})
        message = " ".join(str(info.value).split())
        assert f"training.{key} is not applied by task='grpo' on backend='mlx'" in message
        assert f"Remove training.{key}, or use backend: transformers." in message
        assert "v0.50.0" not in message

    @pytest.mark.parametrize("key", ["long_context_grpo", "vision_grpo", "quantize_ref_model"])
    def test_fields_read_by_nothing_get_the_staged_refusal(self, key):
        """#808 refuses these on every backend as of v0.77, so MLX GRPO adds no
        refusal of its own and they are not in _MLX_GRPO_UNAPPLIED."""
        from soup_cli.config.schema import _MLX_GRPO_UNAPPLIED

        with pytest.raises(Exception) as info:
            _config(**{key: "true"})
        message = " ".join(str(info.value).split())
        assert f"training.{key} is accepted but read by nothing" in message
        assert key not in _MLX_GRPO_UNAPPLIED

    @pytest.mark.parametrize("key", ["tis_threshold", "skip_zero_advantage",
                                     "off_policy_mask_threshold", "mask_truncated_completions"])
    def test_record_only_fields_say_the_transformers_path_only_records_them(self, key):
        from soup_cli.config.schema import _MLX_GRPO_UNAPPLIED

        assert _MLX_GRPO_UNAPPLIED[key].startswith("the transformers path only")

    @pytest.mark.parametrize("line", _PREFIX_CELLS)
    def test_each_prefix_is_refused_with_its_reason(self, line):
        from soup_cli.config.schema import _MLX_GRPO_UNAPPLIED_PREFIXES

        key, value = line.split(": ", 1)
        why = next(r for p, r in _MLX_GRPO_UNAPPLIED_PREFIXES.items() if key.startswith(p))
        with pytest.raises(Exception) as info:
            _config(**{key: value})
        assert (f"training.{key} is not applied by task='grpo' on backend='mlx': {why}. "
                f"Remove training.{key}, or use backend: transformers.") in _plain(
                    str(info.value))

    def test_every_prefix_has_a_cell_and_every_entry_names_a_field(self):
        from soup_cli.config.schema import (
            _MLX_GRPO_UNAPPLIED,
            _MLX_GRPO_UNAPPLIED_PREFIXES,
            TrainingConfig,
        )

        fields = set(TrainingConfig.model_fields)
        assert set(_MLX_GRPO_UNAPPLIED) <= fields
        for prefix in _MLX_GRPO_UNAPPLIED_PREFIXES:
            assert any(name.startswith(prefix) for name in fields), prefix
        assert {next(p for p in _MLX_GRPO_UNAPPLIED_PREFIXES if c.startswith(p))
                for c in _PREFIX_CELLS} == set(_MLX_GRPO_UNAPPLIED_PREFIXES)

    def test_the_older_gates_are_unchanged_elsewhere(self):
        from soup_cli.config.loader import load_config_from_string

        transformers = _GRPO_YAML.replace("backend: mlx", "backend: transformers")
        cfg = load_config_from_string(transformers.replace(
            "  quantization: none\n", "  quantization: none\n  tis_threshold: 2.0\n"))
        assert cfg.training.tis_threshold == 2.0
        dpo = _GRPO_YAML.replace("task: grpo", "task: dpo").replace(
            "  reward_fn: format\n", "")
        for extra, match in [("  tis_threshold: 2.0\n", "require task='grpo'"),
                             ("  rollout_backend: openenv\n", "requires task='grpo'"),
                             ("  long_context_grpo: true\n", "grpo")]:
            with pytest.raises(Exception, match=match):
                load_config_from_string(dpo.replace("  quantization: none\n",
                                                    "  quantization: none\n" + extra))
        from soup_cli.utils.grpo_long_context import validate_long_context_grpo_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_long_context_grpo_compat(task="grpo", backend="mlx",
                                              use_ring_attention=False)

    def test_reward_verifiable_is_refused(self):
        with pytest.raises(Exception) as info:
            _config(reward_fn="verifiable", verifiable_domain="math")
        message = _plain(str(info.value))
        assert ("training.reward_fn='verifiable' is not supported by task='grpo' on "
                "backend='mlx': MLX GRPO scores with the built-in 'accuracy' or 'format' "
                "reward only. Use one of them, or backend: transformers.") in message
        assert "does not load custom reward files" not in message

    @pytest.mark.parametrize("extra", [""])
    def test_vision_modality_is_refused(self, extra):
        """The MLX GRPO refusal, not the v0.50.0 gate's."""
        from soup_cli.config.loader import load_config_from_string

        text = _GRPO_YAML.replace("backend: mlx\n", "backend: mlx\nmodality: vision\n")
        with pytest.raises(Exception) as info:
            load_config_from_string(text.replace("  quantization: none\n",
                                                 "  quantization: none\n" + extra))
        message = " ".join(str(info.value).split())
        assert ("modality='vision' is not supported by task='grpo' on backend='mlx': MLX "
                "GRPO trains on text prompts only. Use backend=transformers, or modality: "
                "text.") in message
        assert "v0.50.0" not in message

    def test_the_transformers_route_is_unchanged(self):
        from soup_cli.config.loader import load_config_from_string

        text = _GRPO_YAML.replace("backend: mlx", "backend: transformers")
        cfg = load_config_from_string(text.replace("  quantization: none\n",
                                                   "  quantization: 4bit\n  eval_steps: 5\n"))
        assert (cfg.backend, cfg.training.quantization, cfg.training.eval_steps) == (
            "transformers", "4bit", 5)

    def test_doctor_reports_ignored_settings_but_not_the_applied_template(self):
        from soup_cli.config.backend_support import check_config
        from soup_cli.config.loader import load_config_from_string

        text = _GRPO_YAML.replace("  max_length: 64\n", _WITH_CHATML)
        cfg = load_config_from_string(text.replace("  quantization: none\n",
                                                   "  quantization: none\n  use_galore: true\n"))
        assert {e.field for e in check_config(cfg)} == {"training.use_galore"}

    def test_every_doctor_row_can_load_and_is_reported(self):
        """Doctor rows only for settings a GRPO config on MLX can carry (#755, #1330)."""
        from soup_cli.config.backend_support import check_config, unsupported_for

        fields = [e.field for e in unsupported_for("grpo", "mlx")]
        assert fields
        for field in fields:
            cfg = _config(**{field.split(".", 1)[1]: "true"})
            assert [e.field for e in check_config(cfg)] == [field]

    @pytest.mark.parametrize("line", ["use_lorafa: true", "use_flash_attn: true",
                                      "use_liger: true", "neftune_alpha: 5.0",
                                      "quantization_aware: true"])
    def test_settings_refused_at_load_have_no_doctor_row(self, line):
        from soup_cli.config.backend_support import unsupported_for

        key, value = line.split(": ", 1)
        with pytest.raises(Exception, match=key):
            _config(**{key: value})
        assert f"training.{key}" not in {e.field for e in unsupported_for("grpo", "mlx")}

    def test_the_ignored_settings_are_printed_without_the_template(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer import mlx_dpo
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        buffer = StringIO()
        monkeypatch.setattr(mlx_dpo, "console", Console(file=buffer, width=400))
        text = _GRPO_YAML.replace("  max_length: 64\n", _WITH_CHATML)
        cfg = load_config_from_string(text.replace("  quantization: none\n",
                                                   "  quantization: none\n  use_galore: true\n"))
        MLXGRPOTrainerWrapper(cfg)._check_unsupported()
        printed = _plain(buffer.getvalue())
        assert "MLX backend ignores: GaLore" in printed and "chat_template" not in printed

    def test_mlx_trainers_on_the_dpo_helpers_declare_what_those_read(self):
        """Every MLX task whose declared modules include mlx_dpo.py also declares the
        trainer modules mlx_dpo.py imports (KTO lost two when GRPO's row landed)."""
        import ast
        from pathlib import Path

        from soup_cli.config.backend_support import TRAINER_MODULES
        from soup_cli.trainer import mlx_dpo

        tree = ast.parse(Path(mlx_dpo.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module.replace(".", "/") + ".py" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            and node.module.startswith("soup_cli.trainer.")
        }
        assert {"soup_cli/trainer/_trl_compat.py", "soup_cli/trainer/loss_summary.py"} <= imported
        users = {key: set(mods) for key, mods in TRAINER_MODULES.items()
                 if "soup_cli/trainer/mlx_dpo.py" in mods}
        assert {task for task, _, _ in users} == {"dpo", "simpo", "orpo", "kto", "grpo"}
        missing = {key: sorted(imported - mods) for key, mods in users.items()
                   if imported - mods}
        assert missing == {}

    def test_cli_resume_is_refused_before_anything_loads(self, capsys):
        import typer

        from soup_cli.commands.train import _resolve_resume_or_exit

        with pytest.raises(typer.Exit):
            _resolve_resume_or_exit("auto", _config())
        assert ("--resume is not supported for task 'grpo' on backend 'mlx': resume is not "
                "implemented for MLX GRPO.") in _plain(capsys.readouterr().out)

    @pytest.mark.parametrize("args, flag", [
        (["--resume", "auto"], "--resume"),
        (["--hf-resume", "--push-as", "org/repo"], "--hf-resume"),
    ])
    def test_train_refuses_each_resume_flag(self, tmp_path, monkeypatch, args, flag):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.trainer.mlx_grpo import RESUME_REFUSAL

        monkeypatch.chdir(tmp_path)
        (tmp_path / "x.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in _prompt_rows(8)), encoding="utf-8")
        (tmp_path / "soup.yaml").write_text(_GRPO_YAML, encoding="utf-8")
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", *args, "--yes"])
        output = _plain(result.output)
        assert result.exit_code == 1, output
        assert (f"{flag} is not supported for task 'grpo' on backend 'mlx': resume is not "
                "implemented for MLX GRPO.") in output
        assert _plain(RESUME_REFUSAL.replace("--resume", flag, 1)) in output
        assert "#1977" not in RESUME_REFUSAL


# --------------------------------------------------------------------------
# Review of #1976: config values reach the loop; coverage checks
# --------------------------------------------------------------------------


class TestConfigValuesReachTheLoop:
    def test_num_generations_beta_and_the_completion_cap(self, tmp_path, monkeypatch):
        """num_generations=3, grpo_beta=0.3 and data.max_length=70 are the values the
        loop runs with: whole groups of 3, completions capped at 70, and a loss
        that matches the oracle at beta 0.3 once the policy has moved off the base."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_grpo

        rollouts, calls = [], []
        real_rollout = mlx_grpo.MLXGRPOTrainerWrapper._rollout
        real_loss = mlx_grpo.grpo_token_loss

        def rollout(self, hf, prompt_ids, eos, max_new):
            out = real_rollout(self, hf, prompt_ids, eos, max_new)
            rollouts.append((len(prompt_ids), out[0], out[1]))
            return out

        def loss(logps, advantages, mask, **kwargs):
            value, sums = real_loss(logps, advantages, mask, **kwargs)
            calls.append({"logps": mx.stop_gradient(logps), "adv": advantages, "mask": mask,
                          "ref": kwargs["ref_logps"], "num_items": kwargs["num_items"],
                          "loss": mx.stop_gradient(value), "kl": mx.stop_gradient(sums["kl"])})
            return value, sums

        monkeypatch.setattr(mlx_grpo.MLXGRPOTrainerWrapper, "_rollout", rollout)
        monkeypatch.setattr(mlx_grpo, "grpo_token_loss", loss)
        from soup_cli.config.loader import load_config_from_string

        text = (_GRPO_YAML.replace("  max_length: 64\n", "  max_length: 70\n")
                .replace("  num_generations: 4\n", "  num_generations: 3\n")
                .replace("  grpo_beta: 0.1\n", "  grpo_beta: 0.3\n")
                .replace("  batch_size: 4\n", "  batch_size: 3\n")
                .replace("  gradient_accumulation_steps: 1\n",
                         "  gradient_accumulation_steps: 2\n")
                .replace("  lr: 0.05\n", "  lr: 0.2\n"))
        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper.config = load_config_from_string(text)
        result = wrapper.train()

        assert len(rollouts) == 4 and len(calls) == 8  # 8 prompts / 2 per update
        truncated = 0
        for n_prompts, completions, terminated in rollouts:
            assert n_prompts == 2 and len(completions) == 2 * 3
            for tokens, done in zip(completions, terminated):
                assert len(tokens) <= 70
                if done:
                    assert tokens[-1] == EOS and EOS not in tokens[:-1]
                else:
                    assert len(tokens) == 70
                    truncated += 1
        assert truncated > 0, "fixture must reach the cap"
        last = rollouts[-1]
        assert result["completions/max_length"] <= 70
        assert result["completions/clipped_ratio"] == pytest.approx(
            last[2].count(False) / len(last[2]))
        meta = json.loads((tmp_path / "output" / "adapter_config.json").read_text())
        assert (meta["num_generations"], meta["beta"], meta["max_completion_length"]) == (
            3, 0.3, 70)

        moved = 0
        for call in calls:
            logps, mask = np.array(call["logps"]), np.array(call["mask"], dtype=np.float64)
            ref, adv = np.array(call["ref"]), np.array(call["adv"])
            expected = _np_loss(logps, logps, ref, adv, mask, beta=0.3,
                                num_items=call["num_items"])
            assert call["loss"].item() == pytest.approx(expected, rel=1e-4, abs=1e-6)
            d = ref - logps
            kl = ((np.exp(d) - d - 1) * mask).sum()
            assert call["kl"].item() == pytest.approx(kl, rel=1e-3, abs=1e-7)
            moved += kl > 1e-4
        assert moved >= 2, "the policy must move off the reference for beta to matter"


class TestHeldOutPrompts:
    def test_the_hold_out_is_announced_with_its_remedy(self, tmp_path, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_grpo

        buffer = StringIO()
        monkeypatch.setattr(mlx_grpo, "console", Console(file=buffer, width=400))
        _wrapper(tmp_path, monkeypatch, val=_prompt_rows(3, prefix="v")).train()
        text = _plain(buffer.getvalue())
        assert ("MLX GRPO: 3 prompt(s) are held out for validation (data.val_split=0.1; "
                "on backend: mlx the loader always holds the split out)") in text
        assert ("Set data.val_split: 0 to train on every prompt, as transformers GRPO does "
                "without training.eval_steps.") in text

    def test_nothing_is_announced_without_a_validation_split(self, tmp_path, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_grpo

        buffer = StringIO()
        monkeypatch.setattr(mlx_grpo, "console", Console(file=buffer, width=400))
        _wrapper(tmp_path, monkeypatch, train=_prompt_rows(1)).train()
        assert "held out" not in _plain(buffer.getvalue())

    def test_batch_size_auto_names_the_group(self, monkeypatch):
        from io import StringIO

        from rich.console import Console

        from soup_cli.trainer import mlx_grpo

        buffer = StringIO()
        monkeypatch.setattr(mlx_grpo, "console", Console(file=buffer, width=400))
        _setup_wrapper(monkeypatch, batch_size="auto")._prepare({"train": _prompt_rows(2)})
        assert ("training.batch_size auto (1 on MLX) is below num_generations=4; using one "
                "group, 4 completions per micro-step") in _plain(buffer.getvalue())


class TestSavingAndTheRunRecord:
    def test_a_failed_write_leaves_no_partial_file(self, tmp_path):
        from pathlib import Path

        from soup_cli.trainer.mlx_grpo import write_atomically

        target = tmp_path / "adapters.safetensors"

        def fail(path):
            Path(path).write_bytes(b"half")
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            write_atomically(target, fail)
        assert list(tmp_path.iterdir()) == []
        write_atomically(target, lambda path: Path(path).write_bytes(b"whole"))
        assert [p.name for p in tmp_path.iterdir()] == ["adapters.safetensors"]
        assert target.read_bytes() == b"whole"

    def test_a_failed_adapter_save_in_training_leaves_no_partial_file(self, tmp_path,
                                                                       monkeypatch):
        mx = pytest.importorskip("mlx.core")
        wrapper = _wrapper(tmp_path, monkeypatch)
        real = mx.save_safetensors

        def fail(path, weights):
            real(path, weights)
            raise OSError("disk full")

        monkeypatch.setattr(mx, "save_safetensors", fail)
        with pytest.raises(OSError, match="disk full"):
            wrapper.train()
        assert not list((tmp_path / "output").glob("*.safetensors"))

    def test_the_val_reward_row(self):
        from soup_cli.trainer.mlx_grpo import log_val_reward

        rows: list = []

        class Tracker:
            def log_metrics(self, **row):
                rows.append(row)

        log_val_reward(Tracker(), "run-1", step=4, epoch=1.0, lr=0.01, val_reward=0.5)
        log_val_reward(Tracker(), None, step=4, epoch=1.0, lr=0.01, val_reward=0.5)
        log_val_reward(None, "run-1", step=4, epoch=1.0, lr=0.01, val_reward=0.5)
        assert rows == [{"run_id": "run-1", "step": 4, "epoch": 1.0, "loss": None,
                         "lr": 0.01, "val_reward": 0.5}]

        class Broken:
            def log_metrics(self, **row):
                raise RuntimeError("db locked")

        log_val_reward(Broken(), "run-1", step=4, epoch=1.0, lr=0.01, val_reward=0.5)

    def test_training_logs_the_val_reward_to_the_tracker(self, tmp_path, monkeypatch):
        rows: list = []

        class Tracker:
            def log_metrics(self, **row):
                rows.append(row)

        wrapper = _wrapper(tmp_path, monkeypatch, val=_prompt_rows(3, prefix="v"))
        result = wrapper.train(tracker=Tracker(), run_id="run-1")
        last = rows[-1]
        assert (last["step"], last["loss"], last["val_reward"]) == (
            result["total_steps"], None, result["val_reward"])
        assert all(row.get("val_reward") is None for row in rows[:-1])
        assert sum(row["loss"] is not None for row in rows) >= 1


class TestAdapterCoverage:
    def _base_then_lora(self, model, keys):
        mx = pytest.importorskip("mlx.core")
        from mlx_lm.tuner.utils import linear_to_lora_layers

        x = mx.array([[3, 4, 5, 6]])
        base = np.array(model(x))
        model.freeze()
        linear_to_lora_layers(model, len(model.layers),
                              {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": keys})
        for _, m in model.named_modules():
            if hasattr(m, "lora_b"):
                m.lora_b = mx.random.normal(m.lora_b.shape)
        return x, base

    def test_an_embedding_adapter_is_switched_off(self):
        pytest.importorskip("mlx_lm")
        from mlx_lm.models import llama

        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model = llama.Model(llama.ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=40,
            tie_word_embeddings=False))
        x, base = self._base_then_lora(model, {"model.embed_tokens", "self_attn.q_proj"})
        kinds = {type(m).__name__ for _, m in model.named_modules() if hasattr(m, "lora_a")}
        assert kinds == {"LoRAEmbedding", "LoRALinear"}
        assert np.abs(np.array(model(x)) - base).max() > 1e-3
        with adapter_disabled(model) as count:
            np.testing.assert_array_equal(np.array(model(x)), base)
        assert count == 3
        assert np.abs(np.array(model(x)) - base).max() > 1e-3

    def test_a_switch_adapter_is_switched_off(self):
        pytest.importorskip("mlx_lm")
        from mlx_lm.models import mixtral

        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model = mixtral.Model(mixtral.ModelArgs(
            model_type="mixtral", vocab_size=40, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-5))
        x, base = self._base_then_lora(model, {"block_sparse_moe.switch_mlp.gate_proj"})
        with adapter_disabled(model) as count:
            np.testing.assert_array_equal(np.array(model(x)), base)
        assert count == 2

    def test_an_adapter_kind_the_switch_misses_is_refused(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        from soup_cli.trainer.mlx_grpo import _lora_modules, adapter_disabled

        model = _attach_lora(_tiny_llama())

        class Unknown(nn.Module):  # e.g. DoRA, or a future mlx-lm adapter layer
            def __init__(self):
                super().__init__()
                self.lora_a = mx.zeros((2, 2))
                self.scale = 2.0

        model.extra = Unknown()
        before = [m.scale for m in _lora_modules(model)]
        with pytest.raises(RuntimeError, match="the reference would not be the base model"):
            with adapter_disabled(model):
                pass
        assert [m.scale for m in _lora_modules(model)] == before and model.extra.scale == 2.0


class TestDenseCheck:
    def test_a_quantized_switch_layer_is_refused(self, tmp_path, monkeypatch):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx.nn")
        pytest.importorskip("mlx_lm")
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear

        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper.model.experts = QuantizedSwitchLinear(64, 64, 2, group_size=32)
        with pytest.raises(ValueError, match="first: experts"):
            wrapper._check_dense(wrapper.model)

    def test_any_module_storing_quantization_scales_is_refused(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")

        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = mx.ones((1,))

        wrapper = _wrapper(tmp_path, monkeypatch)
        wrapper.model.odd = Stub()
        with pytest.raises(ValueError, match="first: odd"):
            wrapper._check_dense(wrapper.model)
        del wrapper.model["odd"]
        wrapper._check_dense(wrapper.model)


# --------------------------------------------------------------------------
# Second review of #1976: the loop's bookkeeping, pure where it can be
# --------------------------------------------------------------------------


class TestUpdateBookkeeping:
    """The pieces of one update that need no MLX: micro-steps, the token
    normaliser, the reported metrics, decoding and the written metadata."""

    def test_micro_steps_cover_the_update_in_order(self):
        from soup_cli.trainer.mlx_grpo import micro_slices, resolve_units

        units = resolve_units(batch_size=4, accum=3, num_generations=4)
        completions = list(range(units.completions_per_update))
        assert [completions[part] for part in micro_slices(units)] == [
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def test_the_token_count_includes_each_eos(self):
        from soup_cli.trainer.mlx_grpo import update_token_count

        assert update_token_count([[5, EOS], [5, 6, 7], [EOS]]) == 6.0

    def test_kl_and_clip_ratios_are_token_means_over_the_update(self):
        from soup_cli.trainer.mlx_grpo import update_metrics

        sums = {"tokens": 10.0, "kl": 2.0, "clip_low": 1.0, "clip_high": 2.0,
                "clip_region": 3.0}
        common = dict(loss=0.5, rewards=[1.0, 0.0, 0.5, 0.5], group=2, reward_name="format",
                      completions=[[5, EOS], [5, 6, 7], [EOS], [5, 6, 7, 8]],
                      terminated=[True, False, False, False], sums=sums, num_items=10.0,
                      skipped=1)
        metrics = update_metrics(beta=0.1, **common)
        assert metrics["kl"] == pytest.approx(0.2)
        assert (metrics["clip_ratio/low_mean"], metrics["clip_ratio/high_mean"],
                metrics["clip_ratio/region_mean"]) == pytest.approx((0.1, 0.2, 0.3))
        assert metrics["completions/clipped_ratio"] == 0.75  # completions without EOS
        assert (metrics["completions/mean_length"], metrics["completions/min_length"],
                metrics["completions/max_length"]) == (2.5, 1.0, 4.0)
        assert metrics["frac_reward_zero_std"] == 0.5 and metrics["skipped_steps"] == 1
        assert metrics["rewards/format/mean"] == metrics["reward"] == 0.5
        assert "kl" not in update_metrics(beta=0.0, **common)

    def test_zero_std_groups_use_torch_isclose_tolerance(self):
        """TRL counts ``isclose(std, 0)``: |std| <= 1e-8, not exact zero."""
        from soup_cli.trainer.mlx_grpo import reward_statistics

        # Sample stds: 3.5e-9 (zero), 0 (zero), 3.5e-8 (not), float round-off (zero).
        stats = reward_statistics([0.0, 5e-9, 1.0, 1.0, 0.0, 5e-8, 0.1 + 0.2, 0.3], 2)
        assert stats["frac_reward_zero_std"] == pytest.approx(3 / 4)

    def test_completions_are_decoded_without_special_tokens(self):
        from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

        texts = MLXGRPOTrainerWrapper._decode(DigitTokenizer(), [[_IDS["4"], EOS],
                                                                 [BOS, _IDS["2"], PAD]])
        assert texts == ["4", "2"]

    @pytest.mark.parametrize("beta", [0.1, 0.0])
    def test_objective_metadata(self, beta):
        from soup_cli.trainer.mlx_grpo import (
            MLXGRPOTrainerWrapper,
            grpo_contract,
            resolve_units,
        )

        wrapper = MLXGRPOTrainerWrapper(_config(grpo_beta=beta))
        wrapper._units = resolve_units(batch_size=4, accum=2, num_generations=4)
        wrapper._reward_name = "format"
        meta = wrapper._objective_metadata()
        contract = grpo_contract(beta=beta, num_generations=4, max_completion_length=64)
        assert {k: meta[k] for k in contract} == contract
        assert (meta["task"], meta["reward_fn"], meta["skipped_steps"]) == ("grpo", "format", 0)
        assert (meta["prompts_per_update"], meta["completions_per_update"]) == (2, 8)
        assert meta["reference"].startswith("none" if beta == 0.0 else "base model")


def _scripted_group(model, prompt_ids, *, num_generations, max_tokens, eos_token_id,
                    sampler=None):
    """Completions of 2, 3 and 4 tokens ending in EOS, then one truncated at 3."""
    out = [([4 + i] * (i + 1) + [eos_token_id], True) for i in range(num_generations - 1)]
    return out + [([5, 6, 7], False)]


class TestTheLoopsBookkeeping:
    def test_num_items_and_rewards_see_every_token_and_no_special_text(self, tmp_path,
                                                                        monkeypatch):
        """The normaliser is every completion token, EOS included; rewards read the
        text with special tokens skipped and the ids with EOS kept."""
        from soup_cli.trainer import mlx_grpo
        from soup_cli.trainer.rewards import validate_reward_funcs

        monkeypatch.setattr(mlx_grpo, "generate_group", _scripted_group)
        seen = []

        def reward(completions, completion_ids, **_):
            seen.extend(zip([c[0]["content"] for c in completions], completion_ids))
            return [float(len(ids)) for ids in completion_ids]

        wrapper = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(2), grpo_beta=0.0)
        wrapper._prepare(wrapper._dataset)
        wrapper._reward_fn = validate_reward_funcs(reward)
        result = wrapper.train()
        assert [h["num_items"] for h in result["history"]] == [2 + 3 + 4 + 3] * 2
        assert all(math.isfinite(x) for x in result["losses"])
        assert len(seen) == 8 and sum(ids[-1] == EOS for _, ids in seen) == 6
        assert all("$" not in text and text for text, _ in seen)

    def test_the_schedule_spans_updates_not_micro_steps(self, tmp_path, monkeypatch):
        from soup_cli.trainer import mlx_optim

        seen = []
        real = mlx_optim.plan_optimizer

        def plan(**kwargs):
            seen.append(kwargs["total_updates"])
            return real(**kwargs)

        monkeypatch.setattr(mlx_optim, "plan_optimizer", plan)
        result = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(4), batch_size=4,
                          gradient_accumulation_steps=2, scheduler="linear").train()
        assert result["total_steps"] == 2 and seen == [2]

    def test_kl_is_the_token_mean_over_every_micro_step(self, tmp_path, monkeypatch):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_grpo

        calls = []
        real = mlx_grpo.grpo_token_loss

        def loss(logps, advantages, mask, **kwargs):
            value, sums = real(logps, advantages, mask, **kwargs)
            calls.append((mx.stop_gradient(sums["kl"]), mask.sum()))
            return value, sums

        monkeypatch.setattr(mlx_grpo, "grpo_token_loss", loss)
        result = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(4), batch_size=4,
                          gradient_accumulation_steps=2, lr=0.2).train()
        last = calls[-2:]
        kl = sum(k.item() for k, _ in last) / sum(n.item() for _, n in last)
        assert kl > 1e-6 and result["kl"] == pytest.approx(kl, rel=1e-5)

    def test_validation_samples_in_eval_mode(self, tmp_path, monkeypatch):
        """With LoRA dropout on, validation is the eval-mode model's sampling."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer import mlx_grpo

        real = mlx_grpo.generate_group
        calls = []

        def spy(model, prompt_ids, **kwargs):
            out = real(model, prompt_ids, **kwargs)
            calls.append((bool(model.training), list(prompt_ids), out))
            return out

        monkeypatch.setattr(mlx_grpo, "generate_group", spy)
        wrapper = _wrapper(tmp_path, monkeypatch, train=_prompt_rows(2),
                           val=_prompt_rows(2, prefix="v"), seed=5,
                           lora="{r: 4, alpha: 8, dropout: 0.3}")
        wrapper.train()
        val = [c for c in calls if not c[0]]
        assert len(calls) == 4 and calls[2:] == val, "rollouts train, validation evaluates"

        def replay(train_mode):
            wrapper.model.train(train_mode)
            mx.random.seed(5)
            return [real(wrapper.model, ids, num_generations=4, max_tokens=64,
                         eos_token_id=EOS) for _, ids, _ in val]

        assert replay(False) == [out for *_, out in val]
        assert replay(True) != replay(False), "dropout must change the sampling"

    def test_gradient_checkpointing_trains_the_same_update(self, tmp_path, monkeypatch):
        pytest.importorskip("mlx_lm")
        from mlx_lm.models import llama

        # mlx-lm's grad_checkpoint patches the block class process-wide; the
        # monkeypatch restores it after this test.
        original = llama.TransformerBlock.__call__
        monkeypatch.setattr(llama.TransformerBlock, "__call__", original)

        def run(path, **training):
            wrapper = _wrapper(path, monkeypatch, train=_prompt_rows(2), seed=3, **training)
            return wrapper.train(), _flat(wrapper.model.trainable_parameters())

        plain, w_plain = run(tmp_path / "a")
        assert llama.TransformerBlock.__call__ is original
        ckpt, w_ckpt = run(tmp_path / "b", gradient_checkpointing="true")
        assert llama.TransformerBlock.__call__ is not original
        np.testing.assert_allclose(ckpt["losses"], plain["losses"], rtol=1e-4, atol=1e-6)
        for name in w_plain:
            np.testing.assert_allclose(w_ckpt[name], w_plain[name], atol=1e-5, err_msg=name)
