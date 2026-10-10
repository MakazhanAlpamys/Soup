#!/usr/bin/env python3
"""One layer-major (L2L) step over the SHIPPED streamed model, and its GA reference.

L2L (plan.md § "Giant MoE", step 0; Pudipeddi et al., arXiv 2002.05645): load a
decoder layer once per step, run all k micro-batches through it, park each
micro-batch's boundary activation in host RAM, and walk the backward the same
way in reverse. The harness drives the shipped modules and nothing else:

* ``StreamedDecoderLayer._body`` calls ``pool.wait(idx)`` and then
  ``prefetcher.advance(idx)``. A second call on the same index leaves the
  direction alone and finds the next layer already owned, so k calls on a loaded
  layer load nothing. The L2L load sequence (forward 0..L-1, then L-1..0) is
  today's, and ``AsyncDiskSource`` infers its direction from that order.
* The decoder container's forward pre-hook (``runtime.hook``, the shipped
  ``_prime``) is called directly, because the harness bypasses
  ``LlamaModel.forward``.
* All k embedding lookups run before layer 0's first ``advance`` queues the head
  into the shared large slot; ``LargeLayerBufferPool.load_async`` orders that
  refill after them on the compute stream.
* The backward turns the wrapper's checkpoint off for its per-(layer,
  micro-batch) forward+backward, which would otherwise pay a third forward.
* The top-level forward is re-implemented for Llama/Mistral only:
  ``embed_tokens`` -> layers with the kwargs HF passed to layer 0 -> ``norm`` ->
  ``lm_head(h[:, slice(0, None), :])`` -> the model's own ``loss_function``
  with ``num_items_in_batch``. ``capture_layer_call`` records the layer kwargs
  from a real forward instead of rebuilding the rotary embeddings and the mask.

Benchmark harness code, not shipped. torch is imported inside functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUPPORTED_ARCHS = ("llama", "mistral")


@dataclass(frozen=True)
class LayerCall:
    """What HF passes to every decoder layer after ``hidden_states``."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


def model_parts(model: Any) -> Tuple[Any, Any]:
    """(the CausalLM under any PEFT wrapper, the container that owns ``.layers``)."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    owner = decoder_owner(model)
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    arch = str(getattr(getattr(causal, "config", None), "model_type", ""))
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"the L2L probe re-implements the top-level forward of {SUPPORTED_ARCHS} only; "
            f"got {arch!r}"
        )
    for name in ("lm_head", "loss_function"):
        if getattr(causal, name, None) is None:
            raise ValueError(f"the causal LM has no {name!r}")
    return causal, owner


def label_tokens(micro_batches: Sequence[Any]) -> int:
    """HF's ``num_items_in_batch`` when labels == input_ids: seq - 1 per row."""
    return sum(int(ids.shape[0]) * (int(ids.shape[1]) - 1) for ids in micro_batches)


def capture_layer_call(model: Any, input_ids: Any) -> LayerCall:
    """Run one forward without grad and record what layer 0 receives."""
    import torch

    _causal, owner = model_parts(model)
    seen: Dict[str, Any] = {}

    def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if "call" not in seen:
            seen["call"] = (tuple(args[1:]), dict(kwargs))

    handle = owner.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    if "call" not in seen:
        raise RuntimeError("decoder layer 0 was never called by the forward")
    args, kwargs = seen["call"]
    kwargs.pop("num_items_in_batch", None)
    return LayerCall(args=args, kwargs=kwargs)


def ga_step(
    model: Any, micro_batches: Sequence[Any], *, n_items: Optional[int] = None
) -> List[Any]:
    """Gradient accumulation through the shipped forward: k forward/backward calls,
    each micro-batch's summed loss divided by the label tokens of ALL k."""
    total = label_tokens(micro_batches) if n_items is None else int(n_items)
    losses = []
    for ids in micro_batches:
        out = model(input_ids=ids, labels=ids, num_items_in_batch=total)
        out.loss.backward()
        losses.append(out.loss.detach())
    return losses


class L2LStep:
    """One layer-major step: forward, head, backward. No optimizer step."""

    def __init__(self, model: Any, runtime: Any, call: LayerCall):
        self.causal, self.owner = model_parts(model)
        self.layers = list(self.owner.layers)
        self.call = call
        hook = getattr(runtime, "hook", None)
        prime = self.owner._forward_pre_hooks.get(getattr(hook, "id", None))
        if prime is None:
            raise RuntimeError(
                "the streaming prime hook is not installed on the decoder container; "
                "the L2L step cannot load the embedding and prime the prefetcher"
            )
        self._prime = prime

    def _layer(self, layer: Any, hidden: Any) -> Any:
        out = layer(hidden, *self.call.args, **self.call.kwargs)
        return out[0] if isinstance(out, tuple) else out

    def run(
        self, micro_batches: Sequence[Any], store: Any, *, n_items: Optional[int] = None
    ) -> List[Any]:
        import torch

        k = len(micro_batches)
        if k < 1:
            raise ValueError("an L2L step needs at least one micro-batch")
        if k > store.k:
            raise ValueError(f"{k} micro-batches, but the activation store holds {store.k}")
        total = label_tokens(micro_batches) if n_items is None else int(n_items)
        vocab = int(self.causal.config.vocab_size)

        self._prime(self.owner, ())
        with torch.no_grad():
            hidden = [self.owner.embed_tokens(ids) for ids in micro_batches]
            for idx, layer in enumerate(self.layers):
                for mb in range(k):
                    store.put(idx, mb, hidden[mb])
                    hidden[mb] = self._layer(layer, hidden[mb])

        losses: List[Any] = []
        grads: List[Any] = []
        for mb, ids in enumerate(micro_batches):
            last = hidden[mb].detach().requires_grad_(True)
            hidden[mb] = None
            with torch.enable_grad():
                normed = self.owner.norm(last)
                logits = self.causal.lm_head(normed[:, slice(0, None), :])
                loss = self.causal.loss_function(
                    logits=logits, labels=ids, vocab_size=vocab, num_items_in_batch=total
                )
            loss.backward()
            losses.append(loss.detach())
            grads.append(last.grad)

        store.begin_backward()
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            saved = layer.use_checkpoint
            layer.use_checkpoint = False
            try:
                for mb in range(k):
                    inputs = store.get(idx, mb).requires_grad_(True)
                    with torch.enable_grad():
                        out = self._layer(layer, inputs)
                    torch.autograd.backward(out, grad_tensors=grads[mb])
                    grads[mb] = inputs.grad
            finally:
                layer.use_checkpoint = saved
        store.end_step()
        return losses


# --------------------------------------------------------------------------
# adapters and gradients
# --------------------------------------------------------------------------
def lora_parameters(model: Any) -> Dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    params = {
        name: param
        for name, param in canonical_named_parameters(model)
        if "lora_" in name and param.requires_grad
    }
    if not params:
        raise RuntimeError("the model has no trainable LoRA parameters")
    return params


def make_non_vacuous(model: Any, seed: int = 23, scale: float = 0.02) -> None:
    """PEFT initialises lora_B to zero, which makes every lora_A gradient zero."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(lora_parameters(model).items()):
            if "lora_B" in name:
                values = torch.randn(param.shape, generator=generator) * scale
                param.copy_(values.to(param.device, param.dtype))


def snapshot_adapters(model: Any) -> Dict[str, Any]:
    return {name: param.detach().clone() for name, param in lora_parameters(model).items()}


def zero_grads(model: Any) -> None:
    for param in lora_parameters(model).values():
        param.grad = None


def restore_adapters(model: Any, snapshot: Dict[str, Any]) -> None:
    import torch

    params = lora_parameters(model)
    if set(params) != set(snapshot):
        raise RuntimeError("the adapter snapshot does not match the model's LoRA parameters")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(snapshot[name])
    zero_grads(model)


def collect_grads(model: Any) -> Dict[str, Any]:
    out = {}
    for name, param in lora_parameters(model).items():
        if param.grad is None:
            raise RuntimeError(f"{name} received no gradient")
        out[name] = param.grad.detach().to("cpu", copy=True)
    return out


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    if set(reference) != set(candidate):
        missing = sorted(set(reference) ^ set(candidate))[:4]
        raise ValueError(f"the two gradient sets name different tensors, e.g. {missing}")
    unequal, zero, diffs = [], [], []
    for name in sorted(reference):
        left, right = reference[name], candidate[name]
        if not int(torch.count_nonzero(left)):
            zero.append(name)
        if not torch.equal(left, right):
            unequal.append(name)
            diffs.append((float((left.float() - right.float()).abs().max()), name))
    diffs.sort(reverse=True)
    return {
        "tensors": len(reference),
        "unequal": unequal,
        "max_abs": diffs[0][0] if diffs else 0.0,
        "worst": [{"name": name, "max_abs": value} for value, name in diffs[:5]],
        "all_zero": zero,
        "equal": not unequal,
    }


def losses_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    import torch

    return len(left) == len(right) and all(
        torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(left, right)
    )
