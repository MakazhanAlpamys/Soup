"""Offline CPU controls for selected expressions in the pinned #342 workflow."""

import json
import math
import sys


def layout_control():
    import torch

    prompt_lengths = (48, 60, 56, 44)
    completion_lengths = (120, 126, 132, 125)

    def layout(lengths):
        def mask(counts, left):
            width = max(counts)
            return torch.tensor([
                [False] * (width - n) + [True] * n if left else [True] * n + [False] * (width - n)
                for n in counts
            ])

        joined = torch.cat((mask(lengths, True), mask(completion_lengths, False)), dim=1)
        return joined, joined[:2]

    rollout, microbatch = layout(prompt_lengths)
    other_rollout, _ = layout((48, 61, 56, 44))
    assert rollout.size(1) == microbatch.size(1) == 192
    assert max(p + c for p, c in zip(prompt_lengths, completion_lengths, strict=True)) == 188
    assert other_rollout.size(1) == 193
    return {
        "padded_length": rollout.size(1),
        "microbatch_length": microbatch.size(1),
        "maximum_unpadded_row": 188,
        "same_completion_lengths_other_prompt_padding": other_rollout.size(1),
    }


def loss_control(behavior_logp, reference_logp, *, policy_logp=-10.0):
    import torch

    # One positive-advantage active token, unit denominator, no correction.
    policy = torch.tensor(policy_logp, requires_grad=True)
    behavior = policy.detach() if behavior_logp is None else torch.tensor(behavior_logp)
    reference = torch.tensor(reference_logp)
    ratio = torch.exp(policy - behavior)
    delta = reference - policy
    kl = torch.exp(delta) - delta - 1
    loss = -torch.minimum(ratio, ratio.clamp(0.8, 1.2)) + 0.02 * kl
    loss.backward()
    return {
        "ratio": classify(ratio),
        "kl": classify(kl),
        "loss": classify(loss),
        "logp_grad": classify(policy.grad),
    }


def classify(tensor):
    value = tensor.detach().item()
    return value if math.isfinite(value) else str(value)


def clipped_step(inject_nan, watchdog):
    import torch

    bad, good = (torch.nn.Parameter(torch.tensor(v)) for v in (0.0, 1.0))
    model = torch.nn.ParameterList([bad, good])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = good + bad.sqrt() * (0.0 if inject_nan else 1.0)
    loss.backward()
    before = [classify(p.grad) for p in model.parameters()]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = [classify(p.grad) for p in model.parameters()]
    if watchdog and any(not torch.isfinite(p.grad) for p in model.parameters()):
        model.zero_grad(set_to_none=True)
    optimizer.step()
    probabilities = torch.stack((bad, good)).detach().softmax(dim=0)
    return {
        "loss": classify(loss),
        "raw_grads": before,
        "norm": classify(norm),
        "post_clip_grads": after,
        "weights": [classify(p) for p in model.parameters()],
        "optimizer_state_entries": len(optimizer.state),
        "sampling_probabilities_finite": bool(torch.isfinite(probabilities).all()),
    }


def finite_grad_norm_control():
    import torch

    parameter = torch.nn.Parameter(torch.zeros(16))
    parameter.grad = torch.full_like(parameter, 1e20)
    assert torch.isfinite(parameter.grad).all()
    reference_norm = parameter.grad.double().norm().item()
    norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    assert torch.isinf(norm) and torch.count_nonzero(parameter.grad).item() == 0
    return {"float64_norm": reference_norm, "logged_float32_norm": classify(norm),
            "post_clip_nonzero_grad_elements": int(torch.count_nonzero(parameter.grad))}


def logging_control():
    import torch

    # Trainer at the first step since the last log, before global_step increments.
    def accumulate(rows, filter_enabled):
        running = torch.zeros(())
        for value in rows:
            step_loss = torch.tensor(value)
            if filter_enabled and not torch.isfinite(step_loss):
                running = running + running / 1
            else:
                running = running + step_loss
        return classify(running)

    return {
        "one_nan_at_reset": accumulate([math.nan], True),
        "finite_then_nan_same_step": accumulate([0.125, math.nan], True),
        "finite_then_nan_without_filter": accumulate([0.125, math.nan], False),
    }


def checkpoint_control(reentrant):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint

    calls = []

    def attention(hidden):
        query = hidden * 2
        calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "q_requires_grad": query.requires_grad,
        })
        return torch.nn.functional.scaled_dot_product_attention(
            query, hidden * 3, hidden, is_causal=True,
        )

    hidden = torch.ones((1, 1, 4, 4), requires_grad=True)
    with sdpa_kernel(SDPBackend.MATH):
        checkpoint(attention, hidden, use_reentrant=reentrant).sum().backward()
    assert len(calls) == 2 and torch.isfinite(hidden.grad).all()
    assert [row["q_requires_grad"] for row in calls] == [not reentrant, True]
    return calls


def lse_witness_control():
    import torch

    query = torch.tensor([1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    key = torch.tensor([0.0, -120.0, 120.0, -120.0], dtype=torch.float64)
    scores = torch.outer(query, key)
    mask = torch.ones((4, 4), dtype=torch.bool).tril()
    mask[:, 0] = False
    lse = scores.masked_fill(~mask, -math.inf).logsumexp(dim=1)
    threshold = -math.log(torch.finfo(torch.float32).max)
    candidates = mask.diagonal() & (scores.diagonal() < threshold)
    low_lse = torch.isfinite(lse) & (lse < threshold)
    assert candidates.tolist() == [False, True, True, True]
    assert low_lse.tolist() == [False, True, False, False]
    return {"rows_below_threshold_on_diagonal": int(candidates.sum()),
            "rows_below_threshold_after_lse": int(low_lse.sum()),
            "fully_masked_rows": int((~mask.any(dim=1)).sum())}


def reward_control():
    import torch

    def group_advantages(rows):
        totals = torch.tensor(rows).nansum(dim=1)
        advantages = (totals - totals.mean()) / (totals.std() + 1e-4)
        return {"totals": [classify(x) for x in totals],
                "advantages": [classify(x) for x in advantages]}

    missing = group_advantages([[math.nan], [math.nan]])
    infinite = group_advantages([[math.inf], [2.0]])
    assert missing["advantages"] == [0.0, 0.0]
    assert infinite["advantages"] == ["nan", "nan"]
    return {"all_missing": missing, "one_infinite": infinite}


def dapo_denominator_control():
    import torch

    def masked_loss(normalizer):
        values = torch.tensor([0.2, 0.3], requires_grad=True)
        loss = (values * torch.zeros_like(values)).sum() / normalizer
        loss.backward()
        return {"loss": classify(loss), "grads": [classify(x) for x in values.grad]}

    nonempty = masked_loss(2)
    empty = masked_loss(0)
    assert nonempty == {"loss": 0.0, "grads": [0.0, 0.0]}
    assert empty == {"loss": "nan", "grads": ["nan", "nan"]}
    return {"all_masked_with_positive_rollout_count": nonempty, "zero_rollout_count": empty}


def sampling_control():
    import torch

    def softmax_is_valid(scores):
        probabilities = torch.tensor(scores).softmax(dim=-1)
        return bool(torch.isfinite(probabilities).all() and (probabilities >= 0).all())

    results = {
        "one_forbidden": softmax_is_valid([1.0, -math.inf]),
        "all_forbidden": softmax_is_valid([-math.inf, -math.inf]),
        "one_positive_inf": softmax_is_valid([1.0, math.inf]),
        "one_nan": softmax_is_valid([1.0, math.nan]),
    }
    assert results == {"one_forbidden": True, "all_forbidden": False,
                       "one_positive_inf": False, "one_nan": False}
    return results


def attention_storage_control():
    import torch

    def span_bytes(tensor):
        last_element = sum((size - 1) * stride for size, stride in
                           zip(tensor.shape, tensor.stride(), strict=True))
        return (last_element + 1) * tensor.element_size()

    # Meta tensors describe only sizes/strides. No large CPU or GPU allocations.
    packed = torch.empty((128, 704, 28, 128), dtype=torch.bfloat16, device="meta").transpose(1, 2)
    gapped = torch.empty_strided((2, 1, 4, 128), (2**31, 512, 128, 1),
                                 dtype=torch.bfloat16, device="meta")
    result = {"illustrative_packed_span_bytes": span_bytes(packed),
              "synthetic_gapped_span_bytes": span_bytes(gapped)}
    assert result["illustrative_packed_span_bytes"] < 2**32 < result["synthetic_gapped_span_bytes"]
    return result


def gqa_layout_control():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    generator = torch.Generator().manual_seed(342)

    def projections(heads):
        return (torch.randn((1, 8, heads, 128), dtype=torch.float64, generator=generator)
                .transpose(1, 2).requires_grad_(True))

    def repeat_kv(value):
        batch, heads, length, dim = value.shape
        return value[:, :, None].expand(batch, heads, 7, length, dim).reshape(
            batch, heads * 7, length, dim,
        )

    attention = torch.nn.functional.scaled_dot_product_attention
    q, k, v = projections(28), projections(4), projections(4)
    expanded_k, expanded_v = repeat_kv(k), repeat_kv(v)
    mask = torch.ones((1, 1, 8, 8), dtype=torch.bool).tril()
    with sdpa_kernel(SDPBackend.MATH):
        implicit = attention(q, k, v, is_causal=True, enable_gqa=True)
        explicit = attention(q, expanded_k, expanded_v, attn_mask=mask)
        grad = torch.randn(implicit.shape, dtype=implicit.dtype, generator=generator)
        implicit_grads = torch.autograd.grad(implicit, (q, k, v), grad)
        explicit_grads = torch.autograd.grad(explicit, (q, k, v), grad)
    assert torch.allclose(implicit, explicit, rtol=1e-12, atol=1e-12)
    assert all(torch.allclose(x, y, rtol=1e-12, atol=1e-12)
               for x, y in zip(implicit_grads, explicit_grads, strict=True))
    return {"native_kv_shape": list(k.shape), "expanded_kv_shape": list(expanded_k.shape),
            "q_stride": list(q.stride()), "native_k_stride": list(k.stride()),
            "expanded_k_stride": list(expanded_k.stride()),
            "forward_max_abs_delta": float((implicit - explicit).detach().abs().max()),
            "grad_max_abs_deltas": [float((x - y).abs().max())
                                    for x, y in zip(implicit_grads, explicit_grads, strict=True)]}


def main():
    import torch

    ratio = loss_control(-100.0, -10.0)
    on_policy = loss_control(None, -10.0)
    kl = loss_control(None, -1.0)
    kl_overflow = loss_control(None, -10.0, policy_logp=-100.0)
    assert ratio["ratio"] == "inf" and ratio["logp_grad"] == "nan"
    assert math.isclose(ratio["loss"], -1.2, rel_tol=1e-6)
    assert on_policy == {"ratio": 1.0, "kl": 0.0, "loss": -1.0, "logp_grad": -1.0}
    assert math.isfinite(kl["loss"]) and math.isfinite(kl["logp_grad"])
    assert kl_overflow["kl"] == kl_overflow["loss"] == "inf"

    raw_inf = clipped_step(inject_nan=False, watchdog=False)
    raw_nan = clipped_step(inject_nan=True, watchdog=False)
    guarded = clipped_step(inject_nan=True, watchdog=True)
    assert raw_inf["raw_grads"] == ["inf", 1.0] and raw_inf["post_clip_grads"] == ["nan", 0.0]
    assert raw_nan["raw_grads"] == ["nan", 1.0] and raw_nan["post_clip_grads"] == ["nan", "nan"]
    assert not raw_nan["sampling_probabilities_finite"]
    assert guarded["sampling_probabilities_finite"] and guarded["weights"] == [0.0, 1.0]
    assert guarded["optimizer_state_entries"] == 0
    logging = logging_control()
    assert logging == {"one_nan_at_reset": 0.0, "finite_then_nan_same_step": 0.25,
                       "finite_then_nan_without_filter": "nan"}

    sys.stdout.write(json.dumps({
        "torch": torch.__version__,
        "layout": layout_control(),
        "stock_loss_expressions": {"explicit_old_logp_overflow": ratio, "on_policy": on_policy,
                                   "finite_kl": kl, "kl_overflow": kl_overflow},
        "clipping_optimizer": {"raw_inf": raw_inf, "raw_nan": raw_nan, "raw_nan_guarded": guarded,
                               "finite_grads_norm_overflow": finite_grad_norm_control()},
        "logging_branch": logging,
        "checkpoint_calls": {"reentrant": checkpoint_control(True),
                             "non_reentrant": checkpoint_control(False)},
        "lse_witness": lse_witness_control(),
        "reward": reward_control(),
        "dapo_denominator": dapo_denominator_control(),
        "sampling_probabilities_valid": sampling_control(),
        "attention_storage": attention_storage_control(),
        "gqa_layout": gqa_layout_control(),
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
