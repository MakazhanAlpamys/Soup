"""Fast-LoRA fused SwiGLU MLP autograd path for issue #837.

Targets gate_proj/up_proj/down_proj blocks with SiLU/swish. This is the
correctness slice from tracker #792; it makes no throughput claim.
"""

from __future__ import annotations

import logging
import types
from typing import Any

from soup_cli.utils.fast_lora import (
    _as_dtype,
    _dense_weight,
    _flatten,
    _is_supported_lora_projection,
    _projection_state,
)

logger = logging.getLogger(__name__)
_PATCH_MARKER = "_soup_fast_lora_mlp"
_ORIGINAL_FORWARD_MARKER = "_soup_fast_lora_mlp_original_forward"
_HAD_INSTANCE_FORWARD_MARKER = "_soup_fast_lora_mlp_had_instance_forward"
_FUNCTION: Any = None

__all__ = ["patch_fast_lora_mlp", "unpatch_fast_lora_mlp"]


def _mlp_function() -> Any:
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION

    import torch
    import torch.nn.functional as functional

    class _FastLoraSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, wg, bg, wu, bu, wd, bd,
            ag, bgl, au, bul, ad, bdl,
            sg, su, sd, qg_meta, qu_meta, qd_meta, *qparts,
        ):
            metas = (qg_meta, qu_meta, qd_meta)
            counts = [0 if meta is None else int(meta["_count"]) for meta in metas]
            starts = (0, counts[0], counts[0] + counts[1])
            qlists = [
                list(qparts[starts[i] : starts[i] + counts[i]]) for i in range(3)
            ]

            dense_g = _dense_weight(wg, qg_meta, qlists[0], x.dtype)
            dense_u = _dense_weight(wu, qu_meta, qlists[1], x.dtype)
            g = functional.linear(x, dense_g, _as_dtype(bg, x.dtype) if bg is not None else None)
            u = functional.linear(x, dense_u, _as_dtype(bu, x.dtype) if bu is not None else None)
            del dense_g, dense_u

            has_g, has_u, has_d = ag.numel() != 0, au.numel() != 0, ad.numel() != 0
            gu_parts = []
            if has_g:
                gu_parts.append(ag)
            if has_u:
                gu_parts.append(au)
            if gu_parts:
                # #837's advertised fusion: gate/up LoRA A projections share X,
                # so concatenate A and perform one GEMM, then split by rank.
                a_gu = torch.cat(gu_parts, dim=0)
                h_gu = functional.linear(_as_dtype(x, a_gu.dtype), a_gu)
                cursor = 0
                if has_g:
                    rank = ag.shape[0]
                    hg = h_gu[..., cursor : cursor + rank]
                    g = torch.add(g, torch.matmul(hg, bgl.t()).to(g.dtype), alpha=float(sg))
                    cursor += rank
                if has_u:
                    rank = au.shape[0]
                    hu = h_gu[..., cursor : cursor + rank]
                    u = torch.add(u, torch.matmul(hu, bul.t()).to(u.dtype), alpha=float(su))
            else:
                h_gu = x.new_empty((*x.shape[:-1], 0))

            m = functional.silu(g) * u
            dense_d = _dense_weight(wd, qd_meta, qlists[2], m.dtype)
            y = functional.linear(m, dense_d, _as_dtype(bd, m.dtype) if bd is not None else None)
            del dense_d
            if has_d:
                hd = functional.linear(_as_dtype(m, ad.dtype), ad)
                y = torch.add(y, torch.matmul(hd, bdl.t()).to(y.dtype), alpha=float(sd))

            ctx.has_adapters = (has_g, has_u, has_d)
            ctx.gu_ranks = (ag.shape[0] if has_g else 0, au.shape[0] if has_u else 0)
            ctx.scalings = (float(sg), float(su), float(sd))
            ctx.qmetas = metas
            ctx.qcounts = counts
            ctx.qparts_len = len(qparts)
            ctx.save_for_backward(
                x, wg, wu, wd, g, u, ag, bgl, au, bul, ad, bdl, h_gu, *qparts
            )
            return y

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_y):
            saved = ctx.saved_tensors
            x, wg, wu, wd, g, u, ag, bgl, au, bul, ad, bdl, h_gu = saved[:13]
            qparts = list(saved[13:])
            qg_meta, qu_meta, qd_meta = ctx.qmetas
            cg, cu, cd = ctx.qcounts
            qg = qparts[:cg]
            qu = qparts[cg : cg + cu]
            qd = qparts[cg + cu : cg + cu + cd]
            has_g, has_u, has_d = ctx.has_adapters
            sg, su, sd = ctx.scalings

            silu_g = torch.nn.functional.silu(g)
            m = silu_g * u
            grad_ad = grad_bdl = None

            dense_d = _dense_weight(wd, qd_meta, qd, grad_y.dtype)
            grad_m = torch.matmul(grad_y, dense_d)
            del dense_d
            if has_d:
                hd = torch.matmul(_as_dtype(m, ad.dtype), ad.t())
                grad_bdl = _flatten(_as_dtype(grad_y, hd.dtype)).t() @ _flatten(hd) * sd
                grad_hd = torch.matmul(_as_dtype(grad_y, bdl.dtype), bdl) * sd
                grad_ad = _flatten(grad_hd).t() @ _as_dtype(_flatten(m), grad_hd.dtype)
                grad_m = torch.add(
                    grad_m,
                    torch.matmul(_as_dtype(grad_hd, grad_m.dtype), _as_dtype(ad, grad_m.dtype)),
                )

            sig = torch.sigmoid(g)
            grad_u = grad_m * silu_g
            grad_g = grad_m * u * sig * (1 + g * (1 - sig))

            grad_x = None
            if ctx.needs_input_grad[0]:
                dense_g = _dense_weight(wg, qg_meta, qg, grad_g.dtype)
                grad_x = torch.matmul(grad_g, dense_g)
                del dense_g
                dense_u = _dense_weight(wu, qu_meta, qu, grad_u.dtype)
                grad_x = torch.add(grad_x, torch.matmul(grad_u, dense_u))
                del dense_u

            grad_ag = grad_bgl = grad_au = grad_bul = None
            dh_parts = []
            a_parts = []
            cursor = 0
            if has_g:
                rank = ctx.gu_ranks[0]
                hg = h_gu[..., cursor : cursor + rank]
                grad_bgl = _flatten(_as_dtype(grad_g, hg.dtype)).t() @ _flatten(hg) * sg
                grad_hg = torch.matmul(_as_dtype(grad_g, bgl.dtype), bgl) * sg
                dh_parts.append(grad_hg)
                a_parts.append(ag)
                cursor += rank
            if has_u:
                rank = ctx.gu_ranks[1]
                hu = h_gu[..., cursor : cursor + rank]
                grad_bul = _flatten(_as_dtype(grad_u, hu.dtype)).t() @ _flatten(hu) * su
                grad_hu = torch.matmul(_as_dtype(grad_u, bul.dtype), bul) * su
                dh_parts.append(grad_hu)
                a_parts.append(au)

            if dh_parts:
                # One dA GEMM for the gate/up pair, mirroring the one forward
                # X @ A_gu^T GEMM above; split the concatenated result by rank.
                dh_gu = dh_parts[0] if len(dh_parts) == 1 else torch.cat(dh_parts, dim=-1)
                a_gu = a_parts[0] if len(a_parts) == 1 else torch.cat(a_parts, dim=0)
                grad_a_gu = _flatten(dh_gu).t() @ _as_dtype(_flatten(x), dh_gu.dtype)
                if grad_x is not None:
                    grad_x = torch.add(
                        grad_x,
                        torch.matmul(
                            _as_dtype(dh_gu, grad_x.dtype),
                            _as_dtype(a_gu, grad_x.dtype),
                        ),
                    )
                cursor = 0
                if has_g:
                    rank = ctx.gu_ranks[0]
                    grad_ag = grad_a_gu[cursor : cursor + rank]
                    cursor += rank
                if has_u:
                    rank = ctx.gu_ranks[1]
                    grad_au = grad_a_gu[cursor : cursor + rank]

            result = [
                grad_x,
                None, None, None, None, None, None,
                grad_ag, grad_bgl, grad_au, grad_bul, grad_ad, grad_bdl,
                None, None, None, None, None, None,
            ]
            result.extend([None] * ctx.qparts_len)
            return tuple(result)

    _FUNCTION = _FastLoraSwiGLU
    return _FUNCTION


def _make_mlp_forward(original_forward: Any) -> Any:
    fast = _mlp_function()

    def _forward(self, x, *args, **kwargs):
        if args or kwargs:
            return original_forward(x, *args, **kwargs)
        states = [
            _projection_state(getattr(self, name), x, allow_unadapted=True)
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(state is None for state in states):
            return original_forward(x)
        gate, up, down = states
        compute_dtypes = {
            state.compute_dtype for state in states if state.compute_dtype is not None
        }
        if len(compute_dtypes) > 1:
            return original_forward(x)
        input_dtype = x.dtype
        work_x = x
        if compute_dtypes:
            compute_dtype = next(iter(compute_dtypes))
            if work_x.dtype != compute_dtype:
                work_x = work_x.to(compute_dtype)

        qparts = [*gate.qparts, *up.qparts, *down.qparts]
        out = fast.apply(
            work_x,
            gate.weight, gate.bias, up.weight, up.bias, down.weight, down.bias,
            gate.lora_a, gate.lora_b, up.lora_a, up.lora_b, down.lora_a, down.lora_b,
            gate.scaling, up.scaling, down.scaling,
            gate.qmeta, up.qmeta, down.qmeta,
            *qparts,
        )
        return out if work_x is x else out.to(input_dtype)

    return _forward


def _module_uses_silu(module: Any) -> bool:
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        return False
    name = (getattr(act_fn, "__name__", "") or type(act_fn).__name__).lower()
    return "silu" in name or "swish" in name


def _looks_like_moe_expert(path: str, module: Any) -> bool:
    parts = path.lower().split(".")
    cls = type(module).__name__.lower()
    return (
        any(part == "experts" or part.startswith("expert") for part in parts)
        or "moe" in cls
        or "expert" in cls
        or "blocksparse" in cls
    )


def patch_fast_lora_mlp(model: Any) -> int:
    """Patch supported dense SiLU MLP blocks and return the number patched."""
    hidden_act = str(getattr(getattr(model, "config", None), "hidden_act", "")).lower()
    if hidden_act not in {"silu", "swish"}:
        logger.info(
            "Fast-LoRA MLP skipped: hidden_act=%r is not SiLU/swish", hidden_act or None
        )
        return 0

    patched = 0
    for path, module in model.named_modules():
        if getattr(module, _PATCH_MARKER, False):
            continue
        if _looks_like_moe_expert(path, module):
            continue
        if not all(
            hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        if not _module_uses_silu(module):
            continue
        if not any(
            hasattr(getattr(module, name), "lora_A")
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        projections = [
            getattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(hasattr(proj, "modules_to_save") for proj in projections):
            continue
        if any(
            hasattr(proj, "lora_A") and not _is_supported_lora_projection(proj)
            for proj in projections
        ):
            continue
        setattr(module, _ORIGINAL_FORWARD_MARKER, module.forward)
        setattr(module, _HAD_INSTANCE_FORWARD_MARKER, "forward" in vars(module))
        setattr(module, _PATCH_MARKER, True)
        module.forward = types.MethodType(_make_mlp_forward(module.forward), module)
        patched += 1
    return patched


def unpatch_fast_lora_mlp(model: Any) -> int:
    """Restore original MLP forwards. Returns the number restored."""
    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL_FORWARD_MARKER, None)
        if original is None or not getattr(module, _PATCH_MARKER, False):
            continue
        if getattr(module, _HAD_INSTANCE_FORWARD_MARKER, False):
            module.forward = original
        else:
            del module.forward
        delattr(module, _ORIGINAL_FORWARD_MARKER)
        delattr(module, _HAD_INSTANCE_FORWARD_MARKER)
        delattr(module, _PATCH_MARKER)
        restored += 1
    return restored
