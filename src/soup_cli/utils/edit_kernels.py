"""v0.71.9 #194 — live ROME / MEMIT / AlphaEdit weight-edit kernels.

Surgical locate-and-edit: patch one factual association with a rank-1 weight
update at an MLP down-projection, WITHOUT a full fine-tuning loop. These are
real, working kernels (validated on SmolLM2-135M) — a deliberately simplified
covariance-free (``C = I``) variant of the ROME family that is well-defined,
tractable on a 4 GB box, and genuinely changes the target fact.

Method differences:

* ``rome``  — single-layer rank-1 update at the recommended layer.
* ``memit`` — distribute the same optimised residual across a small band of
  layers (each layer gets its own key + a 1/N share of the residual).
* ``alphaedit`` — ROME update projected orthogonal to the down-proj's top
  left-singular direction (a light null-space projection that reduces
  interference with the model's dominant features; survives sequential edits
  better than vanilla ROME).

Every heavy import (``torch``) is local so importing this module stays cheap
(project lazy-import policy).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import List, Optional, Sequence

_LOG = logging.getLogger("soup.edit_kernels")

# Optimisation hyper-parameters (tuned for tiny models; deliberately modest so
# a single edit runs in a few seconds on CPU/tiny-GPU).
_DEFAULT_GRAD_STEPS = 25
_DEFAULT_LR = 0.5
_MEMIT_BAND = 3  # number of layers MEMIT distributes the residual across
_MAX_PROMPT_TOKENS = 256

# v0.71.16 #250 — covariance-preconditioned ROME defaults.
_DEFAULT_COV_RIDGE = 0.01  # λ in C = E[k k^T] + λI (keeps C invertible)
_DEFAULT_COV_MAX_PROMPTS = 256
_DEFAULT_COV_MAX_TOKENS = 64


@dataclass(frozen=True)
class EditKernelResult:
    """Outcome of a single weight edit."""

    method: str
    layer: int
    norm_delta: float
    layers_edited: tuple[int, ...]


def _is_transposed_proj(module: object) -> bool:
    """Return True for a GPT-2 ``Conv1D`` down-projection.

    transformers' ``Conv1D`` stores a transposed weight (``[in, out]`` rather
    than nn.Linear's ``[out, in]``) and an ``nf`` int (the output feature
    count); nn.Linear has neither. Detecting via the ``nf`` attribute is robust
    to the transformers internal class path (v0.71.16 #251).
    """
    nf = getattr(module, "nf", None)
    return isinstance(nf, int) and not isinstance(nf, bool)


def _proj_out_dim(module: object) -> int:
    """Output (hidden) dimension of a down-projection module.

    * nn.Linear: weight is ``[out, in]`` → out = ``weight.shape[0]``.
    * GPT-2 Conv1D: weight is ``[in, out]`` → out = ``nf`` (= ``weight.shape[1]``).
    """
    if _is_transposed_proj(module):
        return int(module.nf)  # type: ignore[attr-defined]
    return int(module.weight.shape[0])  # type: ignore[attr-defined]


def _candidate_models(model: object) -> Iterator[object]:
    """Yield ``model`` then its PEFT base (if any) for layer lookup."""
    yield model
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            yield get_base()
        except Exception as exc:  # noqa: BLE001 — best-effort PEFT unwrap
            # Don't mask a real get_base_model() crash silently — surface it at
            # DEBUG so it's inspectable; _locate_decoder_layers still raises a
            # clear ValueError downstream when no layers are found.
            _LOG.debug("get_base_model() failed during layer lookup: %s", exc)


def _layers_from(model: object) -> object:
    """Return the decoder-layer container on ``model`` or ``None``.

    Llama-family: ``model.model.layers``. GPT-2-family: ``model.transformer.h``.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is not None:
        return layers
    transformer = getattr(model, "transformer", None)
    h = getattr(transformer, "h", None) if transformer is not None else None
    if h is not None:
        return h
    return None


def _locate_decoder_layers(model: object) -> object:
    """Return the decoder-layer ``ModuleList`` for a supported model.

    Supports the Llama-family ``model.model.layers`` and the GPT-2-family
    ``model.transformer.h`` (v0.71.16 #251). PEFT-wrapped models are unwrapped
    via ``get_base_model``. Raises ``ValueError`` for unsupported architectures.
    """
    for candidate in _candidate_models(model):
        layers = _layers_from(candidate)
        if layers is not None:
            return layers
    raise ValueError(
        "could not locate decoder layers (expected a Llama-family "
        "model.model.layers or a GPT-2-family model.transformer.h); "
        "ROME/MEMIT/AlphaEdit support the mlp.down_proj (Llama) and "
        "mlp.c_proj (GPT-2) architectures"
    )


def _down_proj(layers: object, layer: int) -> object:
    """Return the MLP down-projection for decoder ``layer``.

    Llama: ``mlp.down_proj`` (nn.Linear, weight ``[out, in]``). GPT-2:
    ``mlp.c_proj`` (Conv1D, weight ``[in, out]``) — v0.71.16 #251.
    """
    try:
        block = layers[layer]  # type: ignore[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"layer index {layer} out of range") from exc
    mlp = getattr(block, "mlp", None)
    down = getattr(mlp, "down_proj", None) if mlp is not None else None
    if down is None and mlp is not None:
        down = getattr(mlp, "c_proj", None)  # GPT-2 Conv1D
    if down is None or not hasattr(down, "weight"):
        raise ValueError(
            f"decoder layer {layer} has no mlp.down_proj / mlp.c_proj weight "
            "(unsupported architecture for ROME-family edits)"
        )
    return down


def _capture_key(model, tokenizer, down, prompt: str, device: str):
    """Capture the key vector ``k*`` = input to ``down`` at the last token."""
    import torch

    captured: List[object] = []

    def _pre_hook(_mod, args):
        # args[0]: [batch, seq, intermediate]
        captured.append(args[0][0, -1, :].detach().clone())

    handle = down.register_forward_pre_hook(_pre_hook)
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=_MAX_PROMPT_TOKENS
        ).to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    if not captured:
        raise ValueError("failed to capture key vector at the target layer")
    return captured[-1]


def _optimise_residual(
    model,
    tokenizer,
    down,
    *,
    subject: str,
    target: str,
    device: str,
    grad_steps: int,
    lr: float,
):
    """Optimise a residual ``delta`` added to ``down``'s output at the last
    subject token so the model produces ``target``.

    Returns the learned ``delta`` tensor (shape ``[hidden]``).
    """
    import torch

    # Build subject+target ids; supervise only the target span.
    subj_ids = tokenizer(subject, add_special_tokens=True)["input_ids"]
    tgt_ids = tokenizer(
        (" " + target) if not target.startswith(" ") else target,
        add_special_tokens=False,
    )["input_ids"]
    if not tgt_ids:
        raise ValueError("target tokenised to an empty sequence")
    input_ids = torch.tensor([subj_ids + tgt_ids], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(subj_ids) + tgt_ids], dtype=torch.long, device=device
    )
    # Inject delta at the LAST subject token position (the position whose
    # residual feeds the prediction of the first target token).
    inject_pos = len(subj_ids) - 1

    # Delta lives in the OUTPUT (hidden) space — for a GPT-2 Conv1D that is
    # ``nf``, NOT ``weight.shape[0]`` (which is the input dim there).
    hidden = _proj_out_dim(down)
    delta = torch.zeros(
        hidden, device=device, dtype=down.weight.dtype, requires_grad=True
    )

    def _hook(_mod, _args, output):
        # output: [batch, seq, hidden]; add delta at inject_pos only.
        if output.shape[1] > inject_pos:
            output = output.clone()
            output[0, inject_pos, :] = output[0, inject_pos, :] + delta
        return output

    optimizer = torch.optim.Adam([delta], lr=lr)
    handle = down.register_forward_hook(_hook)
    was_training = model.training
    model.eval()
    try:
        for _ in range(grad_steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
    finally:
        handle.remove()
        if was_training:
            model.train()
    return delta.detach()


def _rank1_update(down, key, delta, *, cov=None) -> float:
    """Apply a (covariance-preconditioned) rank-1 update in place.

    Makes ``down(key) == old + delta`` exactly, so the down-proj now produces
    the optimised residual for this key. Returns the update's Frobenius norm.

    ``cov`` (optional, v0.71.16 #250): the key covariance matrix ``C``. When
    given, the update uses the preconditioned key ``u = C^{-1} k*`` (computed
    via a stable linear solve) so the update mass is spread per the ROME
    closed form, reducing collateral interference with other keys. The exact
    post-condition ``down(k*) += delta`` is preserved either way because
    ``denom = u·k*`` normalises it.

    Handles both weight layouts (v0.71.16 #251):

    * nn.Linear ``[out, in]`` — ``W += outer(delta, u) / denom``.
    * GPT-2 Conv1D ``[in, out]`` — ``W += outer(u, delta) / denom`` (transposed).
    """
    import torch

    key_w = key.to(down.weight.dtype)
    if cov is not None:
        # u = C^{-1} k* via a linear solve (cheaper + more stable than a full
        # inverse — we only need one column). Solve in fp32 then cast back. A
        # singular / non-finite covariance makes solve raise a torch LinAlgError
        # (a RuntimeError subclass) — surface it as a clean ValueError.
        try:
            u = torch.linalg.solve(
                cov.to(torch.float32), key.to(torch.float32)
            ).to(down.weight.dtype)
        except RuntimeError as exc:
            raise ValueError(
                f"covariance solve failed (singular / non-finite C): {exc}"
            ) from exc
    else:
        u = key_w
    denom = float(torch.dot(u, key_w).item())
    # Reject zero norm AND non-finite denom — with the #250 covariance path a
    # pathological solve can yield NaN, and ``NaN <= 0.0`` is False, which would
    # otherwise let a NaN update silently corrupt the weights.
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError(
            "key vector has zero norm (or degenerate covariance); "
            "cannot apply rank-1 update"
        )
    delta_w = delta.to(down.weight.dtype)
    if _is_transposed_proj(down):
        update = torch.outer(u, delta_w) / denom
    else:
        update = torch.outer(delta_w, u) / denom
    with torch.no_grad():
        down.weight.add_(update)
    return float(torch.linalg.norm(update).item())


def _alphaedit_project(down, update):
    """Project a logical ``[out, in]`` update orthogonal to ``W``'s top
    left-singular direction.

    A light null-space projection (covariance-free AlphaEdit) that reduces
    interference with the down-proj's dominant feature direction. Operates in
    the logical ``[out, in]`` orientation regardless of weight layout: for a
    GPT-2 Conv1D (weight ``[in, out]``) the weight is transposed to ``[out, in]``
    before the power iteration so the projection removes the OUTPUT-direction
    component consistently. The returned update is ``[out, in]`` — the caller
    transposes it back when applying to a Conv1D (v0.71.16 #251).
    """
    import torch

    with torch.no_grad():
        w = down.weight.detach().to(torch.float32)
        if _is_transposed_proj(down):
            w = w.t()  # Conv1D [in, out] -> logical [out, in]
        # Top left-singular vector via a couple of power iterations (cheap).
        # Seed the generator on CPU so the projection is deterministic /
        # reproducible across runs (review MEDIUM M3).
        gen = torch.Generator(device="cpu").manual_seed(0)
        u = torch.randn(w.shape[0], generator=gen).to(
            device=w.device, dtype=torch.float32
        )
        u = u / (torch.linalg.norm(u) + 1e-8)
        for _ in range(8):
            v = w.t() @ u
            v = v / (torch.linalg.norm(v) + 1e-8)
            u = w @ v
            u = u / (torch.linalg.norm(u) + 1e-8)
        upd32 = update.to(torch.float32)
        proj = upd32 - u.unsqueeze(1) * (u.unsqueeze(0) @ upd32)
    return proj.to(down.weight.dtype)


def estimate_key_covariance(
    model,
    tokenizer,
    down,
    corpus: Sequence[str],
    *,
    device: str,
    ridge: float = _DEFAULT_COV_RIDGE,
    max_prompts: int = _DEFAULT_COV_MAX_PROMPTS,
    max_tokens: int = _DEFAULT_COV_MAX_TOKENS,
):
    """Estimate the key covariance ``C = E[k k^T] + ridge*I`` over ``corpus``.

    v0.71.16 #250 — captures the down-projection INPUT (the key) at every token
    position for each corpus prompt and accumulates the second moment. The
    ridge term keeps ``C`` strictly positive-definite (invertible) for the ROME
    ``C^{-1} k*`` solve. Returns a ``[in, in]`` float32 matrix where ``in`` is
    the down-proj input (intermediate) dim.
    """
    import torch

    # Range-guard the caps (defence-in-depth for direct callers — the CLI only
    # passes the corpus, so these stay at their defaults in the normal flow).
    for _name, _val in (("max_prompts", max_prompts), ("max_tokens", max_tokens)):
        if isinstance(_val, bool) or not isinstance(_val, int) or _val < 1:
            raise ValueError(f"{_name} must be a positive int")
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not math.isfinite(float(ridge))
        or ridge < 0.0
    ):
        raise ValueError("ridge must be a finite non-negative number")

    captured: List[object] = []

    def _pre_hook(_mod, args):
        captured.append(args[0][0].detach().to(torch.float32))  # [seq, in]

    handle = down.register_forward_pre_hook(_pre_hook)
    cov = None
    count = 0
    try:
        for prompt in list(corpus)[:max_prompts]:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            captured.clear()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            with torch.no_grad():
                model(**inputs)
            if not captured:
                continue
            ks = captured[-1]  # [seq, in]
            if cov is None:
                cov = torch.zeros(
                    ks.shape[1], ks.shape[1], dtype=torch.float32, device=ks.device
                )
            cov += ks.t() @ ks
            count += ks.shape[0]
    finally:
        handle.remove()
    if cov is None or count == 0:
        raise ValueError("covariance corpus produced no key vectors")
    cov = cov / float(count)
    cov = cov + ridge * torch.eye(
        cov.shape[0], dtype=torch.float32, device=cov.device
    )
    return cov


def apply_rome_edit(
    model,
    tokenizer,
    *,
    subject: str,
    target: str,
    layer: int,
    device: str,
    grad_steps: int = _DEFAULT_GRAD_STEPS,
    lr: float = _DEFAULT_LR,
    cov_corpus: Optional[Sequence[str]] = None,
    cov_ridge: float = _DEFAULT_COV_RIDGE,
) -> EditKernelResult:
    """Single-layer rank-1 ROME edit. Mutates ``model`` in place.

    When ``cov_corpus`` is supplied (v0.71.16 #250) the rank-1 update is
    preconditioned with the key covariance ``C`` estimated over the corpus
    (``C^{-1} k*``), reducing collateral interference with other facts. Falls
    back to ``C = I`` (the v0.71.9 covariance-free path) otherwise.
    """
    layers = _locate_decoder_layers(model)
    down = _down_proj(layers, layer)
    key = _capture_key(model, tokenizer, down, subject, device)
    delta = _optimise_residual(
        model, tokenizer, down,
        subject=subject, target=target, device=device,
        grad_steps=grad_steps, lr=lr,
    )
    cov = None
    if cov_corpus:
        cov = estimate_key_covariance(
            model, tokenizer, down, cov_corpus, device=device, ridge=cov_ridge,
        )
    norm = _rank1_update(down, key, delta, cov=cov)
    return EditKernelResult(
        method="rome", layer=layer, norm_delta=norm, layers_edited=(layer,),
    )


def apply_memit_edit(
    model,
    tokenizer,
    *,
    subject: str,
    target: str,
    layer: int,
    device: str,
    grad_steps: int = _DEFAULT_GRAD_STEPS,
    lr: float = _DEFAULT_LR,
) -> EditKernelResult:
    """MEMIT edit — distribute the optimised residual across a layer band.

    The residual is optimised once at ``layer``; each layer in the band
    ``[layer-band+1, layer]`` receives its own key + a 1/N share of the
    residual. Mutates ``model`` in place.
    """
    layers = _locate_decoder_layers(model)
    top_down = _down_proj(layers, layer)
    delta = _optimise_residual(
        model, tokenizer, top_down,
        subject=subject, target=target, device=device,
        grad_steps=grad_steps, lr=lr,
    )
    band = [idx for idx in range(layer - _MEMIT_BAND + 1, layer + 1) if idx >= 0]
    if not band:
        band = [layer]
    share = delta / float(len(band))
    total_norm = 0.0
    edited: List[int] = []
    for idx in band:
        down = _down_proj(layers, idx)
        # Re-capture the key at THIS layer (its intermediate dim matches its W).
        key = _capture_key(model, tokenizer, down, subject, device)
        if share.shape[0] != _proj_out_dim(down):
            # Hidden (output) dims must match across layers for a residual
            # share; skip any layer whose output width differs (defensive).
            # ``_proj_out_dim`` handles the Conv1D transposed layout (#251).
            continue
        total_norm += _rank1_update(down, key, share)
        edited.append(idx)
    if not edited:
        raise ValueError("MEMIT could not edit any layer in the band")
    return EditKernelResult(
        method="memit",
        layer=layer,
        norm_delta=total_norm,
        layers_edited=tuple(edited),
    )


def apply_alphaedit_edit(
    model,
    tokenizer,
    *,
    subject: str,
    target: str,
    layer: int,
    device: str,
    grad_steps: int = _DEFAULT_GRAD_STEPS,
    lr: float = _DEFAULT_LR,
) -> EditKernelResult:
    """AlphaEdit — ROME update projected orthogonal to W's top singular dir.

    Mutates ``model`` in place.
    """
    import torch

    layers = _locate_decoder_layers(model)
    down = _down_proj(layers, layer)
    key = _capture_key(model, tokenizer, down, subject, device)
    delta = _optimise_residual(
        model, tokenizer, down,
        subject=subject, target=target, device=device,
        grad_steps=grad_steps, lr=lr,
    )
    key_t = key.to(down.weight.dtype)
    denom = float(torch.dot(key_t, key_t).item())
    if denom <= 0.0:
        raise ValueError("key vector has zero norm; cannot apply AlphaEdit update")
    # Logical [out, in] ROME update (delta is OUTPUT-space, key is INPUT-space).
    rome_update = torch.outer(delta.to(down.weight.dtype), key_t) / denom
    projected = _alphaedit_project(down, rome_update)  # logical [out, in]
    with torch.no_grad():
        if _is_transposed_proj(down):
            down.weight.add_(projected.t())  # Conv1D: back to [in, out] (#251)
        else:
            down.weight.add_(projected)
    norm = float(torch.linalg.norm(projected).item())
    return EditKernelResult(
        method="alphaedit", layer=layer, norm_delta=norm, layers_edited=(layer,),
    )


def run_edit_kernel(
    model,
    tokenizer,
    *,
    method: str,
    subject: str,
    target: str,
    layer: int,
    device: str,
    grad_steps: int = _DEFAULT_GRAD_STEPS,
    lr: float = _DEFAULT_LR,
    cov_corpus: Optional[Sequence[str]] = None,
) -> EditKernelResult:
    """Dispatch to the per-method kernel. ``method`` must already be canonical.

    ``cov_corpus`` (v0.71.16 #250) is only consumed by the ROME kernel — the
    caller (``apply_edit``) rejects it for other methods before reaching here.
    """
    if method == "rome":
        return apply_rome_edit(
            model, tokenizer, subject=subject, target=target, layer=layer,
            device=device, grad_steps=grad_steps, lr=lr, cov_corpus=cov_corpus,
        )
    if method == "memit":
        return apply_memit_edit(
            model, tokenizer, subject=subject, target=target, layer=layer,
            device=device, grad_steps=grad_steps, lr=lr,
        )
    if method == "alphaedit":
        return apply_alphaedit_edit(
            model, tokenizer, subject=subject, target=target, layer=layer,
            device=device, grad_steps=grad_steps, lr=lr,
        )
    raise ValueError(
        f"run_edit_kernel does not handle method={method!r}; "
        "expected rome / memit / alphaedit"
    )


def measure_target_prob(
    model, tokenizer, *, subject: str, target: str, device: str,
) -> float:
    """Return the model's mean probability of the ``target`` tokens after
    ``subject`` (a cheap correctness probe for tests + smoke)."""
    import torch

    subj_ids = tokenizer(subject, add_special_tokens=True)["input_ids"]
    tgt_ids = tokenizer(
        (" " + target) if not target.startswith(" ") else target,
        add_special_tokens=False,
    )["input_ids"]
    if not tgt_ids:
        return 0.0
    input_ids = torch.tensor([subj_ids + tgt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    probs: List[float] = []
    for i, tok in enumerate(tgt_ids):
        pos = len(subj_ids) - 1 + i
        if pos < 0 or pos >= logits.shape[0]:
            continue
        dist = torch.softmax(logits[pos].float(), dim=-1)
        probs.append(float(dist[tok].item()))
    if not probs:
        return 0.0
    return sum(probs) / len(probs)


_SUPPORTED_KERNEL_METHODS = ("rome", "memit", "alphaedit")
