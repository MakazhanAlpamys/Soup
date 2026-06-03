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

from dataclasses import dataclass
from typing import List

# Optimisation hyper-parameters (tuned for tiny models; deliberately modest so
# a single edit runs in a few seconds on CPU/tiny-GPU).
_DEFAULT_GRAD_STEPS = 25
_DEFAULT_LR = 0.5
_MEMIT_BAND = 3  # number of layers MEMIT distributes the residual across
_MAX_PROMPT_TOKENS = 256


@dataclass(frozen=True)
class EditKernelResult:
    """Outcome of a single weight edit."""

    method: str
    layer: int
    norm_delta: float
    layers_edited: tuple[int, ...]


def _locate_decoder_layers(model: object) -> object:
    """Return the decoder-layer ``ModuleList`` for a Llama-family model.

    Raises ``ValueError`` for architectures we cannot locate (e.g. GPT-2 style
    ``transformer.h`` — out of scope for v0.71.9; the ROME family is wired for
    the Llama-shape ``mlp.down_proj`` linear).
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        # PEFT-wrapped or unusual nesting — try get_base_model.
        get_base = getattr(model, "get_base_model", None)
        if callable(get_base):
            base = get_base()
            inner = getattr(base, "model", None)
            layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise ValueError(
            "could not locate decoder layers (expected a Llama-family "
            "model.model.layers); ROME/MEMIT/AlphaEdit support the "
            "mlp.down_proj architecture in v0.71.9"
        )
    return layers


def _down_proj(layers: object, layer: int) -> object:
    """Return the ``mlp.down_proj`` linear for decoder ``layer``."""
    try:
        block = layers[layer]  # type: ignore[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"layer index {layer} out of range") from exc
    mlp = getattr(block, "mlp", None)
    down = getattr(mlp, "down_proj", None) if mlp is not None else None
    if down is None or not hasattr(down, "weight"):
        raise ValueError(
            f"decoder layer {layer} has no mlp.down_proj weight "
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

    hidden = down.weight.shape[0]
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


def _rank1_update(down, key, delta) -> float:
    """Apply ``W += delta @ key^T / ||key||^2`` in place. Returns Frobenius norm.

    Makes ``W @ key == (old W @ key) + delta`` exactly, so the down-proj now
    produces the optimised residual for this key.
    """
    import torch

    key = key.to(down.weight.dtype)
    denom = float(torch.dot(key, key).item())
    if denom <= 0.0:
        raise ValueError("key vector has zero norm; cannot apply rank-1 update")
    update = torch.outer(delta.to(down.weight.dtype), key) / denom
    with torch.no_grad():
        down.weight.add_(update)
    return float(torch.linalg.norm(update).item())


def _alphaedit_project(down, update):
    """Project ``update`` orthogonal to ``W``'s top left-singular direction.

    A light null-space projection (covariance-free AlphaEdit) that reduces
    interference with the down-proj's dominant feature direction.
    """
    import torch

    with torch.no_grad():
        w = down.weight.detach().to(torch.float32)
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
) -> EditKernelResult:
    """Single-layer rank-1 ROME edit. Mutates ``model`` in place."""
    layers = _locate_decoder_layers(model)
    down = _down_proj(layers, layer)
    key = _capture_key(model, tokenizer, down, subject, device)
    delta = _optimise_residual(
        model, tokenizer, down,
        subject=subject, target=target, device=device,
        grad_steps=grad_steps, lr=lr,
    )
    norm = _rank1_update(down, key, delta)
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
        if share.shape[0] != down.weight.shape[0]:
            # Hidden dims must match across layers for a residual share; skip
            # any layer whose output width differs (defensive).
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
    rome_update = torch.outer(delta.to(down.weight.dtype), key_t) / denom
    projected = _alphaedit_project(down, rome_update)
    with torch.no_grad():
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
) -> EditKernelResult:
    """Dispatch to the per-method kernel. ``method`` must already be canonical."""
    if method == "rome":
        return apply_rome_edit(
            model, tokenizer, subject=subject, target=target, layer=layer,
            device=device, grad_steps=grad_steps, lr=lr,
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
