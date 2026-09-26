"""Fast-LoRA shared-X Q/K/V autograd path for issue #838.

The q projection computes Q/K/V together and caches K/V only until the next
two projection calls with the identical input object. Cross-attention calls
that use a different key/value tensor delegate to PEFT; the speculative K/V
computed by q_proj are then discarded. This avoids copying architecture-
specific attention forwards while still sharing one X@A_cat.T GEMM.

A lone q_proj call intentionally retains one {x, k, v} graph until the next q
call or unpatch. That is bounded to one attention module rather than a leak,
but callers doing projection-only probes should unpatch or issue the normal
k/v calls so the short-lived cache is drained.
"""

from __future__ import annotations

import types
from typing import Any

from soup_cli.utils.fast_lora import (
    _FORWARD_OWNER_MARKER,
    _GROUP_PATCH_OWNER_MARKER,
    _as_dtype,
    _dense_weight,
    _flatten,
    _is_supported_lora_projection,
    _projection_state,
)

_PATCH_MARKER = "_soup_fast_lora_qkv"
_SINGLE_PROJECTION_PATCH_MARKER = "_soup_fast_lora_single_projection"
_CACHE_MARKER = "_soup_fast_lora_qkv_cache"
_CACHE_HIT_MARKER = "_soup_fast_lora_qkv_last_cache_hits"
_RESTORE_MARKER = "_soup_fast_lora_qkv_restore"
_OWNER = "qkv"
_FUNCTION: Any = None

__all__ = ["patch_fast_lora_qkv", "unpatch_fast_lora_qkv"]


def _qkv_function() -> Any:
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION

    import torch
    import torch.nn.functional as functional

    class _FastLoraQKV(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, wq, bq, wk, bk, wv, bv,
            aq, bql, ak, bkl, av, bvl,
            sq, sk, sv, qq_meta, qk_meta, qv_meta, *qparts,
        ):
            metas = (qq_meta, qk_meta, qv_meta)
            counts = [0 if meta is None else int(meta["_count"]) for meta in metas]
            starts = (0, counts[0], counts[0] + counts[1])
            qlists = [
                list(qparts[starts[i] : starts[i] + counts[i]]) for i in range(3)
            ]
            weights = (wq, wk, wv)
            biases = (bq, bk, bv)
            outs = []
            for weight, bias, meta, parts in zip(weights, biases, metas, qlists):
                dense = _dense_weight(weight, meta, parts, x.dtype)
                outs.append(
                    functional.linear(
                        x, dense, _as_dtype(bias, x.dtype) if bias is not None else None
                    )
                )
                del dense

            adapters = ((aq, bql, sq), (ak, bkl, sk), (av, bvl, sv))
            ranks = [a.shape[0] if a.numel() else 0 for a, _b, _s in adapters]
            present = [i for i, rank in enumerate(ranks) if rank]
            if present:
                a_cat = torch.cat([adapters[i][0] for i in present], dim=0)
                h = functional.linear(_as_dtype(x, a_cat.dtype), a_cat)
                cursor = 0
                for i in present:
                    a, b, scaling = adapters[i]
                    rank = a.shape[0]
                    hi = h[..., cursor : cursor + rank]
                    outs[i] = torch.add(
                        outs[i],
                        torch.matmul(hi, b.t()).to(outs[i].dtype),
                        alpha=float(scaling),
                    )
                    cursor += rank
            else:
                h = x.new_empty((*x.shape[:-1], 0))

            ctx.ranks = ranks
            ctx.present = present
            ctx.scalings = (float(sq), float(sk), float(sv))
            ctx.qmetas = metas
            ctx.qcounts = counts
            ctx.qparts_len = len(qparts)

            ctx.save_for_backward(
                x, wq, wk, wv, aq, bql, ak, bkl, av, bvl, h, *qparts
            )
            return tuple(outs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_q, grad_k, grad_v):
            saved = ctx.saved_tensors
            x, wq, wk, wv, aq, bql, ak, bkl, av, bvl, h = saved[:11]
            qparts = list(saved[11:])
            grads_out = (grad_q, grad_k, grad_v)
            weights = (wq, wk, wv)
            adapters = ((aq, bql), (ak, bkl), (av, bvl))
            sq, sk, sv = ctx.scalings
            scalings = (sq, sk, sv)
            cq, ck, cv = ctx.qcounts
            qlists = (
                qparts[:cq],
                qparts[cq : cq + ck],
                qparts[cq + ck : cq + ck + cv],
            )

            grad_x = None
            for grad, weight, meta, parts in zip(
                grads_out, weights, ctx.qmetas, qlists
            ):
                dense = _dense_weight(weight, meta, parts, grad.dtype)
                term = torch.matmul(grad, dense)

                grad_x = term if grad_x is None else torch.add(grad_x, term)
                del dense

            grad_as = [None, None, None]
            grad_bs = [None, None, None]
            if ctx.present:
                dh_parts = []
                cursor = 0
                for i in ctx.present:
                    a, b = adapters[i]
                    grad = grads_out[i]
                    rank = ctx.ranks[i]
                    hi = h[..., cursor : cursor + rank]
                    grad_bs[i] = (
                        _flatten(_as_dtype(grad, hi.dtype)).t() @ _flatten(hi)
                        * scalings[i]
                    )
                    dhi = torch.matmul(_as_dtype(grad, b.dtype), b) * scalings[i]
                    dh_parts.append(dhi)
                    cursor += rank

                dh = torch.cat(dh_parts, dim=-1)
                a_cat = torch.cat([adapters[i][0] for i in ctx.present], dim=0)
                grad_a_cat = _flatten(dh).t() @ _as_dtype(_flatten(x), dh.dtype)
                grad_x = torch.add(
                    grad_x,
                    torch.matmul(_as_dtype(dh, grad_x.dtype), _as_dtype(a_cat, grad_x.dtype)),
                )
                cursor = 0

                for i in ctx.present:
                    rank = ctx.ranks[i]
                    grad_as[i] = grad_a_cat[cursor : cursor + rank]
                    cursor += rank

            result = [
                grad_x,
                None, None, None, None, None, None,
                grad_as[0], grad_bs[0],
                grad_as[1], grad_bs[1],
                grad_as[2], grad_bs[2],
                None, None, None, None, None, None,
            ]
            result.extend([None] * ctx.qparts_len)
            return tuple(result)

    _FUNCTION = _FastLoraQKV
    return _FUNCTION


def _install_attention_patch(attn: Any) -> None:
    fast = _qkv_function()
    projections = {name: getattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")}
    originals = {name: proj.forward for name, proj in projections.items()}
    restore = {
        name: ("forward" in proj.__dict__, proj.__dict__.get("forward"))
        for name, proj in projections.items()
    }

    def q_forward(_proj, x, *args, **kwargs):
        setattr(attn, _CACHE_MARKER, None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        if args or kwargs:
            return originals["q_proj"](x, *args, **kwargs)
        states = [
            _projection_state(getattr(attn, name), x, allow_unadapted=True)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        if any(state is None for state in states):
            return originals["q_proj"](x)
        q, k, v = states
        compute_dtypes = {
            state.compute_dtype for state in states if state.compute_dtype is not None
        }
        if len(compute_dtypes) > 1:
            return originals["q_proj"](x)

        input_dtype = x.dtype
        work_x = x
        if compute_dtypes:
            compute_dtype = next(iter(compute_dtypes))
            if work_x.dtype != compute_dtype:
                work_x = work_x.to(compute_dtype)

        qparts = [*q.qparts, *k.qparts, *v.qparts]
        q_out, k_out, v_out = fast.apply(
            work_x,
            q.weight, q.bias, k.weight, k.bias, v.weight, v.bias,
            q.lora_a, q.lora_b, k.lora_a, k.lora_b, v.lora_a, v.lora_b,
            q.scaling, k.scaling, v.scaling,
            q.qmeta, k.qmeta, v.qmeta,
            *qparts,
        )
        if work_x is not x:
            q_out = q_out.to(input_dtype)
            k_out = k_out.to(input_dtype)
            v_out = v_out.to(input_dtype)

        setattr(
            attn,
            _CACHE_MARKER,
            {
                "x": x,
                "k": k_out,
                "v": v_out,
                "hits": 0,
                "grad_fns": [type(q_out.grad_fn).__name__],
            },
        )
        return q_out

    def k_forward(_proj, x, *args, **kwargs):
        cache = getattr(attn, _CACHE_MARKER, None)
        if not args and not kwargs and cache is not None and cache.get("x") is x and "k" in cache:
            out = cache.pop("k")
            cache["hits"] += 1
            cache["grad_fns"].append(type(out.grad_fn).__name__)
            return out
        setattr(attn, _CACHE_MARKER, None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        return originals["k_proj"](x, *args, **kwargs)

    def v_forward(_proj, x, *args, **kwargs):
        cache = getattr(attn, _CACHE_MARKER, None)
        if not args and not kwargs and cache is not None and cache.get("x") is x and "v" in cache:
            out = cache.pop("v")
            cache["hits"] += 1
            cache["grad_fns"].append(type(out.grad_fn).__name__)
            setattr(attn, _CACHE_HIT_MARKER, (cache["hits"], tuple(cache["grad_fns"])))
            setattr(attn, _CACHE_MARKER, None)
            return out
        setattr(attn, _CACHE_MARKER, None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        return originals["v_proj"](x, *args, **kwargs)

    for forward in (q_forward, k_forward, v_forward):
        setattr(forward, _FORWARD_OWNER_MARKER, _OWNER)
    for name, forward in (
        ("q_proj", q_forward),
        ("k_proj", k_forward),
        ("v_proj", v_forward),
    ):
        projection = projections[name]
        setattr(projection, _GROUP_PATCH_OWNER_MARKER, _OWNER)
        projection.forward = types.MethodType(forward, projection)
    setattr(attn, "_soup_fast_lora_qkv_originals", originals)
    setattr(attn, _RESTORE_MARKER, restore)
    setattr(attn, _PATCH_MARKER, True)


def patch_fast_lora_qkv(model: Any) -> int:
    """Patch structural q_proj/k_proj/v_proj attention modules."""
    count = 0
    for module in model.modules():
        if getattr(module, _PATCH_MARKER, False):
            continue
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        if not any(
            hasattr(getattr(module, name), "lora_A")
            for name in ("q_proj", "k_proj", "v_proj")
        ):
            continue
        projections = [
            getattr(module, name) for name in ("q_proj", "k_proj", "v_proj")
        ]
        if any(hasattr(proj, "modules_to_save") for proj in projections):
            continue
        if any(
            hasattr(proj, "lora_A") and not _is_supported_lora_projection(proj)
            for proj in projections
        ):
            continue
        if any(
            getattr(proj, _GROUP_PATCH_OWNER_MARKER, None) is not None
            for proj in projections
        ):
            continue
        if any(
            getattr(proj, _SINGLE_PROJECTION_PATCH_MARKER, False)
            for proj in projections
        ):
            continue
        _install_attention_patch(module)
        count += 1
    return count


def unpatch_fast_lora_qkv(model: Any) -> int:
    restored = 0
    for module in model.modules():
        originals = getattr(module, "_soup_fast_lora_qkv_originals", None)
        restore = getattr(module, _RESTORE_MARKER, None)
        if originals is None or restore is None or not getattr(module, _PATCH_MARKER, False):
            continue
        for name, (had_instance_forward, instance_forward) in restore.items():
            proj = getattr(module, name)
            if getattr(proj.forward, _FORWARD_OWNER_MARKER, None) == _OWNER:
                if had_instance_forward:
                    proj.forward = instance_forward
                elif "forward" in proj.__dict__:
                    delattr(proj, "forward")
            if getattr(proj, _GROUP_PATCH_OWNER_MARKER, None) == _OWNER:
                delattr(proj, _GROUP_PATCH_OWNER_MARKER)
        for attr in (
            _CACHE_MARKER,
            _CACHE_HIT_MARKER,
            "_soup_fast_lora_qkv_originals",
            _RESTORE_MARKER,
            _PATCH_MARKER,
        ):
            if hasattr(module, attr):
                delattr(module, attr)
        restored += 1
    return restored
