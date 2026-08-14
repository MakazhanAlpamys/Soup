"""training.stream_layers — streaming runtime (v0.72.0 BETA).

The torch half: pre-allocated VRAM buffer pool, the CPU-RAM weight source, the
prefetch scheduler, the layer wrapper, and the meta-device model build.

Data flow per step (plan 5.2)::

    FORWARD   layer i: wait(i) -> prefetch(i+1) -> checkpoint(body_i)
    BACKWARD  layer i: wait(i) -> prefetch(i-1) -> recompute + backward

Each layer is read TWICE per step and that cannot be optimised away:
``dL/dx = W^T . dL/dy``, so the backward pass needs W to reach lower layers and
their adapters. This is physics, not an implementation detail.

**No top-level torch / peft / transformers** — all lazy, so the light CLI keeps
importing without the training stack.
"""

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)

#: Storage dtypes a shard may hold. ``uint8`` is not a *base* dtype — it is what
#: NF4 packs its nibbles and (under double quant) its absmax into.
_DTYPE_NAMES = ("bfloat16", "float16", "float32", "uint8")

#: safetensors header dtype strings -> our names.
_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "U8": "uint8",
}


# ==========================================================================
# model-graph navigation
# ==========================================================================
def decoder_owner(model: Any) -> Any:
    """Return the module that owns ``.layers`` (LlamaModel / Qwen2Model).

    This is deliberately NOT the CausalLM wrapper. PEFT's ``LoraModel.forward``
    calls ``self.model.forward(...)`` directly, bypassing ``__call__`` and
    therefore every forward hook registered on the wrapper. transformers always
    reaches the layer container through ``__call__``, so hooks land here.
    """
    node = model
    for _ in range(8):
        if hasattr(node, "layers"):
            return node
        for attr in ("base_model", "model", "transformer"):
            child = getattr(node, attr, None)
            if child is not None and child is not node:
                node = child
                break
        else:
            break
    raise ValueError(
        "could not locate the decoder-layer container (a module with .layers) — "
        "layer streaming supports Llama/Qwen-shaped models only"
    )


def _set_module_param(root: Any, full_name: str, tensor: Any) -> None:
    import torch.nn as nn

    parts = full_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    module._parameters[parts[-1]] = nn.Parameter(tensor, requires_grad=False)


def _torch_dtype(name: str):
    import torch

    if name not in _DTYPE_NAMES:
        raise ValueError(f"unsupported dtype {name!r}; supported: {_DTYPE_NAMES}")
    return getattr(torch, name)


# ==========================================================================
# NF4 — Params4bit views rebuilt over the pooled buffers (plan P3, v0.72.2)
# ==========================================================================
def rebuild_quant_state(key: str, buffers: Mapping[str, Any], spec: Any, codes: Mapping[str, Any]):
    """Reassemble one weight's ``QuantState`` from streamed tensors.

    ``Params4bit`` carries its ``quant_state`` alongside the packed bytes and
    quantises on transfer to CUDA, so NF4 weights cannot simply be ``copy_``d
    into a plain buffer (plan P3). The packed nibbles and the per-block absmax
    ARE streamed; the two code tables are constant across every weight and are
    held resident once.
    """
    import torch
    from bitsandbytes.functional import QuantState

    from soup_cli.utils.layer_shard import (
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
    )

    code = codes.get(NF4_CODE_KEY)
    if code is None:
        raise ValueError(
            "the NF4 code table is missing from the shard extras — reshard the "
            "checkpoint (a stale v0.72.0 cache has no quantisation data)"
        )
    state2 = None
    offset = None
    if spec.nested:
        nested_code = codes.get(NF4_NESTED_CODE_KEY)
        if nested_code is None:
            raise ValueError(f"nested NF4 code table missing for {key}")
        state2 = QuantState(
            absmax=buffers[key + NESTED_ABSMAX_SUFFIX],
            code=nested_code,
            blocksize=spec.nested_blocksize,
            dtype=torch.float32,
        )
        offset = buffers[key + NESTED_OFFSET_SUFFIX]
    return QuantState(
        absmax=buffers[key + ABSMAX_SUFFIX],
        shape=torch.Size(spec.shape),
        dtype=_torch_dtype(spec.dtype),
        blocksize=spec.blocksize,
        code=code,
        quant_type=spec.quant_type,
        offset=offset,
        state2=state2,
    )


def rebuild_params4bit(key: str, buffers: Mapping[str, Any], spec: Any, codes: Mapping[str, Any]):
    """A ``Params4bit`` VIEW over the pooled buffer — no copy, no re-quantise.

    ``bnb_quantized=True`` is what stops ``Params4bit`` from trying to quantise
    the already-packed bytes again on its next ``.cuda()``.
    """
    import bitsandbytes as bnb

    return bnb.nn.Params4bit(
        data=buffers[key],
        requires_grad=False,
        quant_state=rebuild_quant_state(key, buffers, spec, codes),
        blocksize=spec.blocksize,
        compress_statistics=spec.nested,
        quant_type=spec.quant_type,
        bnb_quantized=True,
    )


def install_dequant_forward(module: Any) -> int:
    """#331 — keep a STREAMED NF4 weight out of ``bitsandbytes``' ``MatMul4Bit``.

    ``MatMul4Bit.forward`` stashes the packed weight and the ``quant_state`` on
    ``ctx`` as plain attributes rather than through ``save_for_backward``::

        ctx.state = quant_state
        ctx.tensors = (None, B)

    ``torch.utils.checkpoint`` discards and recomputes *saved tensors*. These are
    not saved tensors, so it cannot see them: the reference taken in the forward
    survives, it ALIASES the buffer pool, and the backward reads it after that slot
    has been refilled with a different layer. Measured on 8xH100 against a resident
    NF4 reference, that is a bit-exact forward, a healthy-looking loss curve, and
    gradients wrong on every layer but the last ``stream_buffers``.

    De-aliasing was measured and rejected: bnb holds the reference across the whole
    forward-to-backward span, so any copy keeps one layer alive for that span and
    costs O(model). On real 32B, peak VRAM 4 220 -> 19 720 MiB.

    So the weight never enters that autograd Function. It is dequantised inside the
    checkpointed region and multiplied natively; ``F.linear`` saves the dequantised
    tensor through the ordinary mechanism, which checkpointing DOES discard and
    recompute, and the transient lives only inside the recomputed block — O(window).

    This is not a numerics change at training shapes. ``bitsandbytes::gemm_4bit``
    dispatches on M with ``_gemm_4bit_custom_max_m = 1536``, and on real projection
    shapes it takes ``_dequant_linear_fallback`` at every M measured from 8 to 2048 —
    i.e. bitsandbytes already does exactly this. The dtype handling below therefore
    mirrors ``Linear4bit.forward`` line for line, replacing only the final call: a
    dequantisation into a different dtype than bnb's own would introduce a rounding
    difference precisely where there is none today.

    Returns the number of modules patched, so a caller can assert it patched
    something. Zero would mean the model carries no 4-bit linears at all.
    """
    import types

    import bitsandbytes as bnb
    import torch.nn.functional as functional
    from bitsandbytes.functional import dequantize_4bit

    def _dequant_forward(self, x):
        if getattr(self, "quant_state", None) is not None and self.weight.quant_state is None:
            self.weight.quant_state = self.quant_state
        quant_state = self.weight.quant_state

        if not getattr(self, "compute_type_is_set", True):
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if getattr(self, "compute_dtype", None) is not None:
            x = x.to(self.compute_dtype)

        bias = self.bias
        if bias is not None:
            bias = bias.to(x.dtype)

        # THE repair: dequantise here, inside whatever checkpointed region this
        # forward is running in, and let F.linear save the dense weight properly.
        weight = dequantize_4bit(self.weight, quant_state).to(x.dtype)
        return functional.linear(x, weight, bias).to(inp_dtype)

    patched = 0
    for child in module.modules():
        if isinstance(child, bnb.nn.Linear4bit):
            child.forward = types.MethodType(_dequant_forward, child)
            patched += 1
    return patched


def validate_quant_shape(key: str, spec: Any, shard_spec: Mapping[str, Any]) -> None:
    """The index's claimed shape must be backed by the bytes actually on disk.

    ``dequantize_4bit`` allocates ``prod(spec.shape)`` outputs and reads that
    many values out of the packed buffer and the absmax with no bounds check, so
    an index that OVERSTATES a tensor turns into an out-of-bounds read in native
    code. Checks are one-sided — under-claiming is fine, since bitsandbytes pads
    a non-block-aligned tensor up to the next block.
    """
    from soup_cli.utils.layer_shard import ABSMAX_SUFFIX, NESTED_ABSMAX_SUFFIX

    elements = math.prod(spec.shape)
    packed_shape, _dtype = shard_spec[key]
    packed = math.prod(packed_shape) if packed_shape else 0
    if elements > packed * 2:
        raise ValueError(
            f"shard is inconsistent with its index at {key}: the index claims "
            f"{elements} elements but only {packed} packed bytes are stored "
            f"(NF4 holds two values per byte). Reshard the checkpoint."
        )
    absmax_shape, _ = shard_spec[key + ABSMAX_SUFFIX]
    absmax = math.prod(absmax_shape) if absmax_shape else 0
    blocks = -(-elements // spec.blocksize)
    if blocks > absmax:
        raise ValueError(
            f"shard is inconsistent with its index at {key}: the index implies "
            f"{blocks} absmax blocks but only {absmax} are stored. Reshard the "
            f"checkpoint."
        )
    if spec.nested:
        nested_shape, _ = shard_spec[key + NESTED_ABSMAX_SUFFIX]
        nested = math.prod(nested_shape) if nested_shape else 0
        nested_blocks = -(-absmax // spec.nested_blocksize)
        if nested_blocks > nested:
            raise ValueError(
                f"shard is inconsistent with its index at {key}: the index "
                f"implies {nested_blocks} nested absmax blocks but only "
                f"{nested} are stored. Reshard the checkpoint."
            )


def quant_sidecar_keys(key: str, spec: Any) -> Tuple[str, ...]:
    """Every shard key that must be streamed for ``key`` to be rebuildable."""
    from soup_cli.utils.layer_shard import (
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
    )

    keys = [key, key + ABSMAX_SUFFIX]
    if spec.nested:
        keys.append(key + NESTED_ABSMAX_SUFFIX)
        keys.append(key + NESTED_OFFSET_SUFFIX)
    return tuple(keys)


# ==========================================================================
# Tier 1 — the whole frozen base in CPU RAM (plan 5.5)
# ==========================================================================
class RamSource:
    """The base held in CPU RAM, allocated ONCE and filled by ``copy_``.

    The obvious ``load_file -> .to(dtype) -> .pin_memory()`` costs three
    transient copies of every layer. Measured on the dev box, that transient —
    not the store — is what pushed a 5.55 GB base past the 7.12 GB page-locked
    ceiling and made a 3B run impossible. So the store is pre-allocated at its
    final dtype and each source tensor is streamed into it one at a time.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Mapping[str, Tuple[Tuple[int, ...], str]],
        *,
        pin: bool = True,
    ):
        import torch
        from safetensors import safe_open

        from soup_cli.utils.layer_shard import layer_shard_path

        self.store: list = []
        self.nbytes = 0
        self.pinned = bool(pin)
        for idx in range(n_layers):
            held: Dict[str, Any] = {}
            with safe_open(layer_shard_path(shard_dir, idx), framework="pt") as handle:
                for name, (shape, dtype) in spec.items():
                    dst = torch.empty(
                        tuple(shape), dtype=_torch_dtype(dtype), pin_memory=self.pinned
                    )
                    src = handle.get_tensor(name)
                    dst.copy_(src)
                    del src
                    held[name] = dst
                    self.nbytes += dst.numel() * dst.element_size()
            self.store.append(held)

    @staticmethod
    def spec_from_shard(shard_dir: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        """Shape AND dtype for ONE decoder layer, read from the shard header.

        The dtype is read per tensor rather than taken from ``index.dtype``: an
        NF4 shard is deliberately mixed — packed nibbles and (under double
        quant) absmax are ``uint8`` while the nested absmax, the offset and the
        layernorms are floats. Allocating one dtype across the pool would
        reinterpret packed bytes as floats.
        """
        from safetensors import safe_open

        from soup_cli.utils.layer_shard import layer_shard_path

        spec: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        with safe_open(layer_shard_path(shard_dir, 0), framework="pt") as handle:
            for name in handle.keys():
                sliced = handle.get_slice(name)
                shape = tuple(int(d) for d in sliced.get_shape())
                raw = sliced.get_dtype()
                if raw not in _SAFETENSORS_DTYPES:
                    raise ValueError(
                        f"shard tensor {name} has unsupported dtype {raw!r}; "
                        f"supported: {', '.join(sorted(_SAFETENSORS_DTYPES))}"
                    )
                spec[name] = (shape, _SAFETENSORS_DTYPES[raw])
        return spec

    def get(self, idx: int, name: str):
        return self.store[idx][name]


class DiskSource:
    """Tier 2 — the base stays on disk and each layer is read on demand.

    Same interface as :class:`RamSource` (``get(idx, name)`` + ``nbytes``), so
    the buffer pool, the prefetcher and the layer wrapper are untouched: the
    only thing that changes is where the bytes come from. That is what lets the
    disk tier inherit v0.72.0's correctness gates rather than needing new ones
    for the scheduler.

    Handles are opened ONCE and held. safetensors memory-maps them, so the
    operating system's page cache — not this class — decides what stays
    resident. Two honest consequences:

    * On a machine with spare RAM the "disk" tier is partly a RAM tier, because
      the pages survive between steps. That makes a like-for-like RAM-vs-disk
      comparison hard to construct, and it is why this release publishes the
      code path but no gap measurement (see the v0.72.3 gate notes).
    * ``nbytes`` is 0: nothing is deliberately held resident. The on-disk size
      is reported separately so the pre-flight can still show the operator what
      the model costs.

    NVMe only — ``choose_tier`` refuses spinning disks, where each step costs
    two seeks per layer (plan P11) and the run thrashes rather than merely
    running slower.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Mapping[str, Tuple[Tuple[int, ...], str]],
    ):
        import contextlib

        from safetensors import safe_open

        from soup_cli.utils.layer_shard import layer_shard_path

        self._stack = contextlib.ExitStack()
        # ExitStack, not a comprehension of __enter__(): if shard N fails to
        # open, everything opened before it must still be closed.
        try:
            self._handles = [
                self._stack.enter_context(
                    safe_open(layer_shard_path(shard_dir, idx), framework="pt")
                )
                for idx in range(n_layers)
            ]
        except BaseException:
            self._stack.close()
            raise
        self.nbytes = 0  # nothing is held resident by design
        self.disk_bytes = sum(
            math.prod(shape) * _dtype_size(dtype) for shape, dtype in spec.values()
        ) * n_layers
        self.pinned = False

    def get(self, idx: int, name: str):
        return self._handles[idx].get_tensor(name)

    def close(self) -> None:
        """Release every shard handle. Idempotent — ``ExitStack.close`` is."""
        self._stack.close()
        self._handles = []

    def __del__(self) -> None:
        # Backstop only. The trainer closes the runtime explicitly; this exists
        # so a caller that drops the source without doing so still releases one
        # file handle per decoder layer (80+ on a large model) rather than
        # holding them until the process exits.
        try:
            self.close()
        except Exception:  # noqa: BLE001 — never raise from a finaliser
            pass


def _dtype_size(name: str) -> int:
    from soup_cli.utils.layer_stream import dtype_bytes

    return dtype_bytes(name)


def extras_resident_bytes(shard_dir: str) -> int:
    """Bytes the non-layer weights occupy on the GPU (embeddings, final norm,
    an untied head). Read from the extras header — no tensor is materialised.

    The shared NF4 code tables live in the same file but are 272 floats of
    machinery, not model weights, so they are excluded.
    """
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
    )
    from soup_cli.utils.layer_stream import dtype_bytes

    skip = {NF4_CODE_KEY, NF4_NESTED_CODE_KEY}
    total = 0
    with safe_open(extras_shard_path(shard_dir), framework="pt") as handle:
        for name in handle.keys():
            if name in skip:
                continue
            sliced = handle.get_slice(name)
            raw = sliced.get_dtype()
            if raw not in _SAFETENSORS_DTYPES:
                # Raise, don't skip: this number feeds the RAM-tier decision, so
                # silently under-reporting it would let a base that does not fit
                # be accepted. Mirrors spec_from_shard.
                raise ValueError(
                    f"extras tensor {name} has unsupported dtype {raw!r}; "
                    f"supported: {', '.join(sorted(_SAFETENSORS_DTYPES))}"
                )
            total += math.prod(int(dim) for dim in sliced.get_shape()) * dtype_bytes(
                _SAFETENSORS_DTYPES[raw]
            )
    return total


# ==========================================================================
# Tier 0 — pre-allocated VRAM buffers (plan 5.4)
# ==========================================================================
class LayerBufferPool:
    """N pre-allocated per-layer buffers. Never allocates inside the loop —
    that is what keeps the allocator from fragmenting (plan P7)."""

    def __init__(
        self,
        layer_spec: Mapping[str, Tuple[Tuple[int, ...], str]],
        n_buffers: int = 2,
        device: str = "cuda",
    ):
        import torch

        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.n = int(n_buffers)
        self.buffers = [
            {
                name: torch.empty(tuple(shape), dtype=_torch_dtype(dtype), device=device)
                for name, (shape, dtype) in layer_spec.items()
            }
            for _ in range(self.n)
        ]
        self.events = [torch.cuda.Event() for _ in range(self.n)] if self.is_cuda else []
        self.owner: list = [None] * self.n
        self.loads = 0
        self.nbytes = sum(
            buf.numel() * buf.element_size() for buf in self.buffers[0].values()
        ) * self.n

    def slot_for(self, idx: int) -> int:
        return idx % self.n

    def load_async(self, idx: int, source: RamSource, stream: Any = None) -> int:
        import torch

        slot = self.slot_for(idx)
        if self.is_cuda and stream is not None:
            # The slot's previous owner may still be in flight on the compute
            # stream; the prefetch must not clobber it (plan P1).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for name, dst in self.buffers[slot].items():
                    dst.copy_(source.get(idx, name), non_blocking=True)
                self.events[slot].record(stream)
        else:
            for name, dst in self.buffers[slot].items():
                dst.copy_(source.get(idx, name))
        self.owner[slot] = idx
        self.loads += 1
        return slot

    def wait(self, idx: int) -> Dict[str, Any]:
        """Block the compute stream until layer ``idx`` is resident.

        The ownership check is the plan-P1 tripwire: a buffer recycled while an
        autograd node still references it produces silently WRONG gradients, not
        a crash. Failing loudly here is the whole point.
        """
        import torch

        slot = self.slot_for(idx)
        if self.owner[slot] != idx:
            raise RuntimeError(
                f"layer-stream scheduler bug: buffer slot {slot} holds layer "
                f"{self.owner[slot]}, but layer {idx} was requested. Raise "
                f"training.stream_buffers (currently {self.n}) or report this."
            )
        if self.is_cuda:
            torch.cuda.current_stream().wait_event(self.events[slot])
        return self.buffers[slot]


class StreamPrefetcher:
    """Drives the prefetch. Forward walks 0..L-1; backward recompute walks
    L-1..0, so the direction is inferred from the call order."""

    def __init__(self, pool: Any, source: Any, n_layers: int, stream: Any = None):
        self.pool = pool
        self.source = source
        self.n_layers = int(n_layers)
        self.stream = stream
        self.prev: Optional[int] = None
        self.direction = 1
        self.primes = 0

    def prime(self) -> None:
        """Start of a forward pass: layer 0, walking upward."""
        self.prev = None
        self.direction = 1
        self.primes += 1
        self.pool.load_async(0, self.source, self.stream)

    def advance(self, idx: int) -> None:
        # Direction is explicit state, not re-derived per call. It only ever
        # flips downward, at the forward/backward turnaround, and is reset by
        # prime() at the start of the next step. Inferring it fresh each call
        # happens to work today only because the turnaround index is the last
        # layer; a future deeper lookahead would break that assumption
        # silently.
        if self.prev is not None and idx < self.prev:
            self.direction = -1
        self.prev = idx
        nxt = idx + self.direction
        if 0 <= nxt < self.n_layers and self.pool.owner[self.pool.slot_for(nxt)] != nxt:
            self.pool.load_async(nxt, self.source, self.stream)


# ==========================================================================
# the streamed layer (plan 5.6)
# ==========================================================================
def _build_streamed_layer_class():
    import torch
    import torch.nn as nn
    from torch.func import functional_call
    from torch.utils.checkpoint import checkpoint

    class StreamedDecoderLayer(nn.Module):
        def __init__(
            self,
            inner,
            idx,
            pool,
            prefetcher,
            name_map=None,
            use_checkpoint=True,
            quant_specs=None,
            codes=None,
        ):
            super().__init__()
            self.inner = inner
            self.idx = int(idx)
            self.pool = pool
            self.prefetcher = prefetcher
            self.name_map = dict(name_map or {})
            self.use_checkpoint = bool(use_checkpoint)
            self.quant_specs = dict(quant_specs or {})
            self.codes = dict(codes or {})
            # v0.72.5 (#331) — a streamed NF4 weight must not reach MatMul4Bit,
            # which captures it outside save_for_backward and so aliases the pool
            # across the checkpoint boundary. See install_dequant_forward.
            if self.quant_specs:
                self.n_dequant_forward = install_dequant_forward(self.inner)
            else:
                self.n_dequant_forward = 0
            # v0.72.3 — the mirror of the state_dict() override below.
            self._register_load_state_dict_pre_hook(self._redirect_canonical_keys)

        def _redirect_canonical_keys(
            self, state_dict, prefix, *_args: Any, **_kwargs: Any
        ) -> None:
            # v0.72.1 made SAVING canonical; this makes LOADING accept the same
            # keys, which is what `--resume` needs.
            #
            # ``nn.Module.load_state_dict`` narrows the dict by CHILD NAME as it
            # descends: at this wrapper it keeps only keys under `<prefix>`, then
            # recurses into each child with `<prefix><child>.`. Our sole child is
            # `inner`, so a canonical `...layers.0.self_attn...` key matches no
            # child prefix and is silently dropped. Measured before this hook:
            # `load_adapter` landed **0 of 12** tensors, PEFT emitted only a
            # UserWarning, and the resumed loss curve was byte-identical to a
            # from-scratch one.
            #
            # This pre-hook runs at the START of our own `_load_from_state_dict`
            # and may mutate `state_dict` in place; the child loop reads that
            # same object afterwards, so the redirected copies are visible when
            # torch descends into `inner`. Nothing is deleted (the originals
            # match no child and are inert) and nothing is double-added.
            #
            # Deliberately load-side only. It redirects keys rather than
            # re-parenting the module tree, so the forward path is untouched and
            # v0.72.0's bit-exactness gates remain valid without being re-run.
            #
            # The original key is MOVED, not copied. Leaving it behind makes
            # this wrapper's own strict scan flag it `unexpected` — `self_attn`
            # is not one of our children, only `inner` is — so a caller passing
            # ``strict=True`` would get "Unexpected key(s)" on a load that in
            # fact succeeded. (PEFT always loads with ``strict=False`` and never
            # inspects the list, so this was inert in practice; it is still a
            # landmine for any direct caller.) Moving is safe: torch rebuilds a
            # fresh filtered dict at every recursion level, so the dict mutated
            # here is an intermediate one, never the caller's own.
            inner_prefix = prefix + "inner."
            for key in [
                k
                for k in state_dict
                if k.startswith(prefix) and not k.startswith(inner_prefix)
            ]:
                redirected = inner_prefix + key[len(prefix):]
                value = state_dict.pop(key)
                if redirected in state_dict:
                    # A checkpoint carrying BOTH spellings of one weight is
                    # malformed, and silently keeping one would load a tensor
                    # the file does not unambiguously specify.
                    raise ValueError(
                        f"checkpoint contains both {key!r} and {redirected!r} "
                        f"for the same weight — it is malformed; re-save the "
                        f"adapter"
                    )
                state_dict[redirected] = value

        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            # `.to(device)` / `.to(dtype)` walk the module tree via _apply. The
            # wrapped layer's weights are META PLACEHOLDERS, substituted per
            # call from the buffer pool, and moving a meta tensor raises
            # NotImplementedError — which transformers' Trainer.__init__ AND
            # accelerate's prepare_model both trigger. Pass meta tensors
            # through untouched; everything real (the LoRA adapters, which live
            # inside this same subtree) still moves and casts normally.
            def _skip_meta(tensor: Any) -> Any:
                if getattr(tensor, "is_meta", False):
                    return tensor
                return fn(tensor)

            return super()._apply(_skip_meta, recurse=recurse)

        def state_dict(
            self,
            *args: Any,
            destination: Any = None,
            prefix: str = "",
            keep_vars: bool = False,
        ) -> Any:
            # v0.72.1 — serialise as though this wrapper were not in the tree.
            #
            # The wrapper holds the real layer as a child named `inner`, so
            # every adapter parameter would otherwise be written as
            # `...layers.0.inner.self_attn.q_proj.lora_A.weight`. That file
            # loads as ZERO tensors into any normal model — PEFT reports the
            # keys as missing and returns the untuned base, with no exception.
            # Every adapter artifact (the final `trainer.save_model()`, each
            # `save_steps` checkpoint, and therefore everything downstream:
            # `soup merge` / `serve` / `chat` / `adapters *` / the Registry)
            # reaches disk through this method, so delegating at OUR prefix is
            # what makes a streamed adapter indistinguishable from a normal
            # LoRA run.
            #
            # Serialisation-only, deliberately: the forward path is untouched,
            # so v0.72.0's bit-exactness gates remain valid. The cost is that
            # `named_parameters()` still shows `.inner.`, which is why
            # `canonical_named_parameters()` below exists. `--resume` /
            # `--hf-resume` load INTO a streamed model fine (v0.72.3,
            # `train.py`): a separate load-side pre-hook redirects canonical
            # keys at load time, mirroring this save-side delegation.
            #
            # The wrapper owns no parameters or buffers of its own — they all
            # live on `inner` — so nothing is lost by not serialising it. It
            # also means bypassing nn.Module.state_dict skips only hooks
            # registered on the WRAPPER itself, of which there are none (the
            # prefetch hook lives on the decoder container, not here).
            if args:
                # torch's legacy positional form: (destination, prefix, keep_vars)
                if destination is None:
                    destination = args[0]
                if len(args) > 1 and prefix == "":
                    prefix = args[1]
                if len(args) > 2 and keep_vars is False:
                    keep_vars = args[2]
            return self.inner.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        def __getattr__(self, name: str) -> Any:
            # transformers reads contract attributes straight off the layer
            # object (this version reads `decoder_layer.attention_type`). The
            # wrapper must be attribute-transparent or the model breaks at
            # forward time — and a wrapper returning a DEFAULT instead would
            # silently pick the wrong attention path.
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                inner = self._modules.get("inner")
                if inner is None:
                    raise
                return getattr(inner, name)

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            if self.use_checkpoint and torch.is_grad_enabled():
                return checkpoint(
                    self._body, hidden_states, *args, use_reentrant=False, **kwargs
                )
            return self._body(hidden_states, *args, **kwargs)

        def _substituted_weights(self, buffers: Any) -> Any:
            # Weights arrive with requires_grad=False, so autograd allocates no
            # grad buffers for them — but W STAYS IN THE GRAPH for W^T . dL/dy,
            # which is how the lower adapters receive gradient at all.
            if not self.quant_specs:
                return {meta: buffers[ckpt] for meta, ckpt in self.name_map.items()}
            # NF4: a Params4bit VIEW is rebuilt over the pooled buffer on every
            # call (plan P3). The packed bytes are never copied or re-quantised
            # — only the small Python wrapper is reconstructed.
            weights = {}
            for meta, ckpt in self.name_map.items():
                spec = self.quant_specs.get(ckpt)
                if spec is None:
                    weights[meta] = buffers[ckpt]
                else:
                    weights[meta] = rebuild_params4bit(ckpt, buffers, spec, self.codes)
            return weights

        def _body(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            buffers = self.pool.wait(self.idx)
            self.prefetcher.advance(self.idx)
            return functional_call(
                self.inner,
                self._substituted_weights(buffers),
                (hidden_states, *args),
                kwargs,
            )

    return StreamedDecoderLayer


_STREAMED_LAYER_CLASS = None


def _streamed_layer_class():
    global _STREAMED_LAYER_CLASS
    if _STREAMED_LAYER_CLASS is None:
        _STREAMED_LAYER_CLASS = _build_streamed_layer_class()
    return _STREAMED_LAYER_CLASS


class _StreamedDecoderLayerProxy:
    """Callable shim so ``StreamedDecoderLayer(...)`` works as a name."""

    def __call__(self, *args, **kwargs):
        return _streamed_layer_class()(*args, **kwargs)

    def __instancecheck__(self, instance):
        return isinstance(instance, _streamed_layer_class())


StreamedDecoderLayer = _StreamedDecoderLayerProxy()


def canonical_named_parameters(model: Any) -> Iterator[Tuple[str, Any]]:
    """Yield ``(name, param)`` with the wrapper's ``.inner.`` segment stripped,
    matching the spelling ``StreamedDecoderLayer.state_dict()`` already uses
    (v0.72.1). ``named_parameters()`` on a streamed model is not canonical on
    its own: every wrapped layer's parameters carry an extra ``.inner.``
    segment that a resident (non-streamed) model of the same checkpoint does
    not, so a name-keyed comparison between the two sees no overlap unless it
    goes through this function first.
    """
    for name, param in model.named_parameters():
        yield name.replace(".inner.", "."), param


def assert_canonical_parameters_intersect(model_a: Any, model_b: Any) -> FrozenSet[str]:
    """Raise if two models (either may be streamed) share no canonical
    parameter name. A per-parameter comparison (gradients, weights, any other
    property) can only walk names both sides have; if the intersection is
    empty, a loop over it is vacuously satisfied and reports success on
    nothing compared. Returns the intersecting name set on success so a
    caller need not recompute it.
    """
    names_a = frozenset(name for name, _ in canonical_named_parameters(model_a))
    names_b = frozenset(name for name, _ in canonical_named_parameters(model_b))
    shared = names_a & names_b
    if not shared:
        raise ValueError(
            f"no canonical parameter name is shared between the two models "
            f"({len(names_a)} vs {len(names_b)} parameters); a comparison "
            f"over an empty intersection is not a comparison"
        )
    return shared


# ==========================================================================
# allocator
# ==========================================================================
@dataclass(frozen=True)
class GemmCeiling:
    """A dense bf16 GEMM rate measured on THIS card, in THIS session."""

    tflops: float
    sm_clock_mhz: Optional[int]
    size: int


def sm_clock_mhz() -> Optional[int]:
    """Current SM clock, or None when it cannot be read.

    Quoted next to every throughput number: a fraction-of-ceiling without a
    stated clock is meaningless, and this box's boost clock moved 442..952 MHz
    inside one measurement run.
    """
    import subprocess

    from soup_cli.utils.layer_stream import _resolve_tool

    # Absolute path, never the bare name: on Windows CreateProcess searches the
    # CURRENT DIRECTORY before PATH, so a planted nvidia-smi.exe in a cloned
    # project would run instead of the real one (CWE-427).
    tool = _resolve_tool("nvidia-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [tool, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


#: 4096^3 per matmul (~137 GFLOP, ~20 ms here) and best-of-3. Both were measured,
#: not chosen for tidiness: at 2048 the same probe spread **38%** across five
#: repeats and ramped monotonically upward (3.19 -> 4.41 TFLOPS) because the
#: sample is too short for the boost clock to engage; at 4096 the spread fell to
#: 9%. Taking the BEST repeat is not cherry-picking — a ceiling's noise is
#: one-sided, since contention, a cold clock and thermal throttling can only ever
#: make an achievable rate look slower than it is.
_GEMM_SIZE = 4096
_GEMM_REPS = 3
_GEMM_ITERS = 8


def measure_gemm_tflops(
    device: str = "cuda",
    *,
    size: int = _GEMM_SIZE,
    iters: int = _GEMM_ITERS,
    reps: int = _GEMM_REPS,
) -> Optional[GemmCeiling]:
    """Benchmark a dense bf16 matmul to get this card's achievable rate.

    Returns None off CUDA rather than inventing a number — the forecast rests
    entirely on a measurement, so there is nothing honest to return when no
    measurement is possible. A per-card constant compiled into the source would
    be a fabrication: this box alone produced 3.5 and 6.75 TFLOPS in two sessions
    at the same *reported* clock, which is precisely why the number is taken now
    and quoted as a bound rather than a promise.

    Costs ~0.7 s and ~100 MB, once, before the model is built.
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return None
    if not torch.cuda.is_available():
        return None
    best = 0.0
    try:
        left = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        right = torch.randn(size, size, device=device, dtype=torch.bfloat16)
        for _ in range(max(1, reps)):
            for _ in range(3):  # warm up: the first matmul pays kernel selection
                left @ right
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                left @ right
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            if seconds > 0:
                best = max(best, 2.0 * (size**3) * iters / seconds / 1e12)
        del left, right
        torch.cuda.empty_cache()
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None
    if best <= 0:
        return None
    return GemmCeiling(tflops=best, sm_clock_mhz=sm_clock_mhz(), size=size)


@dataclass(frozen=True)
class StepPeak:
    """Peak VRAM of ONE real forward+backward, measured on THIS model and shape.

    Three outcomes, deliberately not two. ``oom`` is a measurement RESULT ("this
    shape does not fit"). ``failed`` means the probe ran a real CUDA op and that
    op raised — the fit is unknown AND the context may no longer be usable, so
    proceeding is not safe merely because the formula was happy. A ``None``
    return from :func:`measure_step_peak_bytes` is the third: the probe was never
    attempted (no CUDA, no torch), nothing was touched, and falling back to the
    prediction is correct. Collapsing ``failed`` into ``None`` would let a run
    continue past a broken CUDA context on the strength of arithmetic.
    """

    peak_bytes: int
    reserved_bytes: int
    seconds: float
    rows: int
    seq_len: int
    oom: bool = False
    failed: bool = False
    error: Optional[str] = None


def measure_step_peak_bytes(
    model: Any,
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> Optional[StepPeak]:
    """Run one forward+backward at the configured shape and read the real peak.

    This is the #349 instrument. The pre-flight's formula is fitted and, measured
    here, under-predicts past seq 4096 (0.830x at seq 5120) — a formula cannot
    model a term nobody has identified, so past that point only a measurement is
    honest. Synthetic ``input_ids`` are used rather than a real batch because the
    quantity being bounded is the CONFIGURED shape, which is the worst case any
    real batch can pad up to; a real batch would measure whatever length it
    happened to have. Validated against a full ``soup train`` run of the same
    config: probe 4.3117 GB reserved vs the real run's 4.3159 GB, 0.1% apart.

    Costs one step. Measured on an RTX 3050 Laptop: 1.02-1.15 s for
    SmolLM2-135M at 1x1024, 5.09 s at 2x2048, and 5.33 s for Llama-3.1-8B NF4 at
    1x512 — against a training run of minutes to hours.

    Returns ``None`` off CUDA or when the probe itself breaks, rather than
    raising: a pre-flight that dies because its own instrument failed is worse
    than one that falls back to the shipped prediction. That is the same contract
    :func:`~soup_cli.utils.layer_stream.measure_logits_loss_bytes_per_element`
    already keeps.
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return None
    if not torch.cuda.is_available():
        return None
    if rows < 1 or seq_len < 1 or vocab_size < 1:
        raise ValueError(
            f"rows/seq_len/vocab_size must all be >= 1; got "
            f"{rows}/{seq_len}/{vocab_size}"
        )
    ids = None
    out = None
    # Bound before the try: `synchronize()` and `reset_peak_memory_stats()` can
    # both surface an earlier async CUDA error as an OOM, and the OOM handler
    # reads `started`. Assigning it inside the try would turn that into an
    # UnboundLocalError escaping setup() instead of the clean refusal below.
    started = time.perf_counter()
    try:  # pragma: no cover - exercised only where torch + CUDA exist
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        ids = torch.randint(0, vocab_size, (rows, seq_len), device=device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    except torch.cuda.OutOfMemoryError:  # pragma: no cover - needs a real OOM
        # A result, not a failure: the shape provably does not fit. Linux raises
        # here; Windows/WDDM spills to host memory instead and reaches the
        # success path with a peak above free VRAM, which the caller refuses just
        # the same.
        return StepPeak(
            peak_bytes=0,
            reserved_bytes=0,
            seconds=time.perf_counter() - started,
            rows=rows,
            seq_len=seq_len,
            oom=True,
        )
    except Exception as exc:  # pragma: no cover - a real CUDA op raised
        # NOT `return None`. None means "never attempted"; this op ran and broke,
        # which can leave the CUDA context poisoned (an illegal access or device
        # assert surfaces exactly here). Reporting that as "cannot tell" would
        # let the caller proceed on the prediction alone, into a context that may
        # no longer work.
        logger.warning("stream VRAM probe raised during the step: %r", exc)
        return StepPeak(
            peak_bytes=0,
            reserved_bytes=0,
            seconds=time.perf_counter() - started,
            rows=rows,
            seq_len=seq_len,
            failed=True,
            error=type(exc).__name__,
        )
    finally:  # pragma: no cover - exercised only where torch + CUDA exist
        del ids, out
        _zero_probe_grads(model)
        torch.cuda.empty_cache()
    return StepPeak(
        peak_bytes=peak,
        reserved_bytes=reserved,
        seconds=elapsed,
        rows=rows,
        seq_len=seq_len,
    )


def _zero_probe_grads(model: Any) -> None:
    """Drop the gradients the probe's backward left on the adapter.

    Without this the first real optimizer step would apply a gradient computed
    from random token ids — training on noise for one step, silently, and only
    when the probe is enabled.

    A failure here is never re-raised (it would mask the measurement the caller
    came for) but it is never silent either: swallowing it would leave exactly
    the corruption this function exists to prevent, with nothing to find it by.
    """
    try:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
    except Exception as exc:  # pragma: no cover - never let cleanup mask the result
        logger.warning(
            "could not clear the VRAM probe's gradients (%r); the first "
            "optimizer step may include a gradient computed from random tokens",
            exc,
        )


def expandable_segments_status() -> tuple[bool, str]:
    """``(enabled, why_not)`` for the ``expandable_segments:True`` hint.

    Split out from :func:`probe_expandable_segments` because the caller used to
    print "unavailable on this platform (silently ignored on Windows)" for every
    False — and on a Colab T4, i.e. Linux, that sentence is simply untrue. There
    the hint fails for a different reason: the allocator reads
    ``PYTORCH_CUDA_ALLOC_CONF`` when the CUDA context is created, so once
    anything has touched CUDA it is too late to set it. Reporting the wrong
    cause is worse than reporting none, and this panel is the most-read output
    the feature has.
    """
    if sys.platform.startswith("win"):
        return (False, "Windows ignores it — torch warns and carries on")
    try:
        import torch
    except ImportError:
        return (False, "torch is not importable")
    if not torch.cuda.is_available():
        return (False, "no CUDA device")
    if torch.cuda.is_initialized():
        already = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if already:
            return (True, "")
        return (
            False,
            "CUDA was already initialised before Soup could set it — the "
            "allocator reads PYTORCH_CUDA_ALLOC_CONF once, at context creation",
        )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return (True, "")
    return (False, "PYTORCH_CUDA_ALLOC_CONF is set to something else")


def probe_expandable_segments() -> bool:
    """Attempt ``expandable_segments:True`` (plan P7) and report whether it took.

    Never claim it is active when it is not; :func:`expandable_segments_status`
    carries the reason.
    """
    return expandable_segments_status()[0]


# ==========================================================================
# model construction — the resident load must NEVER happen (plan P14)
# ==========================================================================
def build_nf4_config(dtype: str, *, double_quant: bool = True):
    """The BitsAndBytesConfig the streamed skeleton and the sharder share.

    ``double_quant`` defaults to True to match
    ``quant_menu.build_quantization_config_for_loader``, which hardcodes it for
    every resident 4-bit load in this repo. That agreement is load-bearing, not
    cosmetic: the release's central claim is that a streamed NF4 run is
    bit-exact against a resident one, and it holds only while both sides
    quantise with the same settings. ``training.bnb_4bit_use_double_quant`` is
    therefore deliberately NOT threaded in here — honouring it on the streamed
    side alone would silently break that parity. (That the flag is ignored
    repo-wide is a pre-existing gap, tracked separately.)
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_torch_dtype(dtype),
        bnb_4bit_use_double_quant=bool(double_quant),
    )


def _stamp_4bit_markers(model: Any, quant_config: Any) -> None:
    """Stamp what ``from_pretrained`` stamps — PEFT and Trainer both read these.

    ``is_loaded_in_4bit`` is THE most important finding of the NF4 gate. PEFT
    checks it to choose ``lora.bnb.Linear4bit``; a ``meta`` skeleton carries no
    such marker, so PEFT silently falls back to the generic
    ``lora.layer.Linear``. That still *runs* against a ``Linear4bit`` base — it
    just casts and accumulates differently, measured as a **9.375e-01** logit
    divergence versus resident NF4 with byte-identical weights AND adapters. No
    crash, no warning, and a loss curve that looks perfectly healthy.

    ``hf_quantizer`` is not optional either, and its absence fails LOUDLY rather
    than silently: ``transformers.Trainer.__init__`` treats
    ``is_quantized and not hf_quantizer.is_trainable`` as "this quantization
    method cannot be fine-tuned" and then formats the error message from
    ``model.hf_quantizer.quantization_config.quant_method`` — so every streaming
    run would die at trainer construction with an AttributeError about an
    attribute the user has never heard of.
    """
    from transformers.quantizers import AutoHfQuantizer
    from transformers.utils.quantization_config import QuantizationMethod

    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.BITS_AND_BYTES
    # pre_quantized=True is the truth here: the shards were quantised offline by
    # the sharder, not on load. Both settings report is_trainable=True, which is
    # the property Trainer actually gates on.
    model.hf_quantizer = AutoHfQuantizer.from_config(quant_config, pre_quantized=True)
    if getattr(model, "config", None) is not None:
        model.config.quantization_config = quant_config


def build_meta_skeleton(
    model_id: str,
    *,
    dtype: str,
    quant: str = "none",
    double_quant: bool = True,
    trust_remote_code: bool = False,
):
    """Build the model structure on ``meta``: no weight storage is allocated.

    Under ``quant='nf4'`` the decoder linears are replaced with
    ``bnb.nn.Linear4bit`` (still on ``meta``) so the streamed ``Params4bit``
    views land in modules that know how to consume them.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.layer_shard import QUANT_NF4, SUPPORTED_STREAM_QUANTS

    if quant not in SUPPORTED_STREAM_QUANTS:
        raise ValueError(
            f"unsupported quant {quant!r} for layer streaming; supported: "
            f"{', '.join(SUPPORTED_STREAM_QUANTS)}"
        )
    torch_dtype = _torch_dtype(dtype)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config.use_cache = False
    quant_config = None
    if quant == QUANT_NF4:
        quant_config = build_nf4_config(dtype, double_quant=double_quant)
    with init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(
                config, dtype=torch_dtype, trust_remote_code=trust_remote_code
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
            )
        if quant_config is not None:
            from transformers.integrations.bitsandbytes import replace_with_bnb_linear

            # lm_head stays at the base dtype: it lives in extras (and is tied
            # to the embeddings on most small Llamas), and quantising the output
            # projection costs accuracy where it is felt most.
            model = replace_with_bnb_linear(
                model,
                modules_to_not_convert=["lm_head"],
                quantization_config=quant_config,
            )
    if quant_config is not None:
        _stamp_4bit_markers(model, quant_config)
    return model


def quantised_layer_suffixes(model: Any) -> FrozenSet[str]:
    """Per-layer short keys that ``replace_with_bnb_linear`` actually converted.

    Authoritative by construction — derived from the skeleton rather than from a
    hard-coded name list that would drift the moment an architecture names its
    projections differently.
    """
    import bitsandbytes as bnb

    try:
        layers = decoder_owner(model).layers
    except ValueError:
        return frozenset()
    if not len(layers):
        return frozenset()
    return frozenset(
        name.replace(".base_layer.", ".")
        for name, param in layers[0].named_parameters()
        if isinstance(param, bnb.nn.Params4bit)
    )


@dataclass(frozen=True)
class ExtrasLoad:
    """What ``materialize_extras`` produced."""

    placed: int
    #: The shared NF4 code tables, moved to the training device. Empty for bf16.
    codes: Mapping[str, Any]


def materialize_extras(
    model: Any, shard_dir: str, index: Any, *, device: str, dtype: str
) -> ExtrasLoad:
    """Give real storage to everything that is NOT a decoder layer.

    Also lifts out the two shared NF4 code tables: they are constant across
    every weight (the sharder asserts it), so one resident copy serves the whole
    model instead of streaming 16 + 256 floats per layer.
    """
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
    )

    extras = load_file(extras_shard_path(shard_dir))
    codes = {
        key: extras.pop(key).to(device)
        for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY)
        if key in extras
    }
    torch_dtype = _torch_dtype(dtype)
    placed = 0
    pending_tied = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta or ".layers." in name:
            continue
        if name in extras:
            _set_module_param(model, name, extras[name].to(device=device, dtype=torch_dtype))
            placed += 1
        else:
            pending_tied.append(name)
    if pending_tied:
        # tie_word_embeddings=True -> lm_head.weight is absent from the
        # checkpoint by design and is restored from the input embeddings.
        model.tie_weights()
        still_meta = [n for n, p in model.named_parameters() if p.is_meta and ".layers." not in n]
        if still_meta:
            raise RuntimeError(
                f"non-layer weights left unmaterialised after tying: {still_meta[:4]}"
            )
    for name, buf in list(model.named_buffers()):
        if buf is not None and str(buf.device) != str(device):
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], buf.to(device))
    return ExtrasLoad(placed=placed, codes=codes)


def materialize_meta_adapters(model: Any, *, seed: int = 0, device: str = "cuda") -> int:
    """Give real storage to LoRA adapters PEFT initialised on ``meta``.

    PEFT creates adapter weights on the base layer's device. In a streaming
    build that device is ``meta``, so the adapters have no storage: the
    optimizer happily accepts them and the run trains nothing. Re-initialise
    with PEFT's own scheme (A ~ kaiming_uniform, B = 0).
    """
    import torch
    import torch.nn as nn

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = 0
    for module_name, module in model.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            full = f"{module_name}.{pname}" if module_name else pname
            if not param.is_meta or "lora_" not in full:
                continue
            data = torch.empty(param.shape, dtype=torch.float32)
            if "lora_B" in full:
                data.zero_()
            else:
                nn.init.kaiming_uniform_(data, a=5**0.5, generator=generator)
            module._parameters[pname] = nn.Parameter(data.to(device), requires_grad=True)
            count += 1
    return count


# ==========================================================================
# installation
# ==========================================================================
@dataclass
class StreamRuntime:
    """Live streaming state for one model."""

    pool: Any
    source: Any
    prefetcher: Any
    n_layers: int
    pinned: bool
    device: str
    hook: Any = None
    #: "ram" or "disk" — which weight source is feeding the buffer pool.
    tier: str = "ram"
    #: True source parameter count, from the shard index. PEFT's own total
    #: is ~6.5x too high for a streamed NF4 model (see trainer/sft.py).
    total_params: int = 0

    def close(self) -> None:
        """Release the weight source and detach the prefetch hook.

        The disk tier holds one open shard handle per decoder layer, so a run
        that finishes without closing leaks 80+ descriptors on a large model.
        A no-op for the RAM tier, which owns no handles.
        """
        source_close = getattr(self.source, "close", None)
        if callable(source_close):
            source_close()
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def stats(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "buffers": self.pool.n,
            "buffer_bytes": self.pool.nbytes,
            "store_bytes": self.source.nbytes,
            "pinned": self.pinned,
            "tier": self.tier,
            "disk_bytes": getattr(self.source, "disk_bytes", 0),
            "layer_loads": self.pool.loads,
            "device": self.device,
            "total_params": self.total_params,
        }


def _device_map_value(device: Any) -> Union[str, int]:
    """The ``hf_device_map`` value ``from_pretrained(device_map=...)`` records.

    It must carry a device INDEX. For a 4-bit model ``accelerate.prepare_model``
    reads ``set(model.hf_device_map.values())`` and does
    ``torch.device(value).index`` — a bare ``"cuda"`` has index ``None``, and
    the very next line calls ``torch.device(None)``, which raises a TypeError
    naming nothing the user could act on. A resident load stores ``0``, so this
    stores ``0``.
    """
    text = str(device)
    if not text.startswith("cuda"):
        return text
    import torch

    if ":" in text:
        return int(text.split(":", 1)[1])
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def _layer_name_map(layer: Any) -> Dict[str, str]:
    """meta parameter name inside one layer -> its shard key."""
    return {
        pname: pname.replace(".base_layer.", ".")
        for pname, param in layer.named_parameters()
        if param.is_meta
    }


def install_streaming(
    model: Any,
    *,
    shard_dir: str,
    index: Any,
    buffers: int = 2,
    pin: bool = True,
    device: str = "cuda",
    console: Any = None,
    codes: Optional[Mapping[str, Any]] = None,
    tier: str = "ram",
) -> StreamRuntime:
    """Wrap every decoder layer and wire the buffer pool + prefetch scheduler."""
    import torch

    from soup_cli.utils.layer_shard import QUANT_NF4

    # Validate the index BEFORE touching the model: an index whose `quant` and
    # `quant_specs` disagree passes the cache key (which reads `quant`) and
    # would then reconstruct NF4 against a skeleton never converted to
    # Linear4bit.
    quant_specs = dict(getattr(index, "quant_specs", None) or {})
    if bool(quant_specs) != (getattr(index, "quant", None) == QUANT_NF4):
        raise ValueError(
            f"shard index is inconsistent: quant={getattr(index, 'quant', None)!r} "
            f"but {len(quant_specs)} quant_specs — reshard the checkpoint"
        )

    owner = decoder_owner(model)
    layers = owner.layers
    n_layers = len(layers)
    if n_layers != index.n_layers:
        raise ValueError(
            f"model has {n_layers} decoder layers but the shard index has "
            f"{index.n_layers} — reshard the checkpoint"
        )

    name_map = _layer_name_map(layers[0])
    if not name_map:
        raise RuntimeError(
            "no meta decoder weights found — the base was materialised, which "
            "defeats layer streaming entirely"
        )
    shard_spec = RamSource.spec_from_shard(shard_dir)
    missing = sorted(set(name_map.values()) - set(shard_spec))
    if missing:
        raise ValueError(f"shard is missing decoder weights: {missing[:4]}")

    # NF4 streams the packed nibbles AND the statistics needed to rebuild the
    # QuantState; the two code tables are shared and stay resident.
    needed: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for ckpt in name_map.values():
        spec_q = quant_specs.get(ckpt)
        wanted = quant_sidecar_keys(ckpt, spec_q) if spec_q is not None else (ckpt,)
        for key in wanted:
            if key not in shard_spec:
                raise ValueError(
                    f"shard is missing the NF4 sidecar {key!r} — reshard the "
                    f"checkpoint"
                )
            needed[key] = shard_spec[key]
        if spec_q is not None:
            validate_quant_shape(ckpt, spec_q, shard_spec)
    spec = needed

    source, pinned = _build_source(shard_dir, n_layers, spec, pin, console, tier)
    pool = LayerBufferPool(spec, n_buffers=buffers, device=device)
    stream = torch.cuda.Stream() if str(device).startswith("cuda") else None
    prefetcher = StreamPrefetcher(pool, source, n_layers, stream)

    layer_cls = _streamed_layer_class()
    for idx in range(n_layers):
        layers[idx] = layer_cls(
            layers[idx],
            idx,
            pool,
            prefetcher,
            name_map,
            quant_specs=quant_specs,
            codes=codes,
        )

    handle = owner.register_forward_pre_hook(lambda *_a, **_k: prefetcher.prime())

    # transformers' Trainer.__init__ calls _move_model_to_device -> model.to(),
    # and .to() on a module holding meta parameters raises NotImplementedError.
    # The decoder weights stay on meta BY DESIGN, so declare that this model
    # manages its own placement — exactly the marker a device_map-sharded model
    # carries, and the one _move_model_to_device short-circuits on.
    model.hf_device_map = {"": _device_map_value(device)}

    return StreamRuntime(
        pool=pool,
        source=source,
        prefetcher=prefetcher,
        n_layers=n_layers,
        pinned=pinned,
        device=str(device),
        hook=handle,
        total_params=int(getattr(index, "total_params", 0) or 0),
        tier=tier,
    )


def _build_source(shard_dir, n_layers, spec, pin, console, tier="ram"):
    """Build the weight source for the chosen tier.

    On the RAM tier, page-locking is bounded by the box rather than by free RAM
    (the dev box topped out at 7.12 GB with 9.1 GB "available"). Falling back to
    a pageable store is correct; hiding the ~97% -> ~79% GPU-utilisation cost is
    not, so the fallback says so out loud.
    """
    if tier == "disk":
        return DiskSource(shard_dir, n_layers, spec), False
    if not pin:
        return RamSource(shard_dir, n_layers, spec, pin=False), False
    try:
        return RamSource(shard_dir, n_layers, spec, pin=True), True
    except (RuntimeError, MemoryError) as exc:
        message = (
            "layer streaming could not page-lock the base "
            f"({type(exc).__name__}); falling back to a PAGEABLE RAM store. "
            "Host-to-device copies become synchronous, which costs overlap — "
            "measured GPU utilisation drops from ~97% to ~79%. Free RAM or use "
            "a smaller base to keep the pinned store."
        )
        if console is not None:
            console.print(f"[yellow]{message}[/]")
        else:
            logger.warning(message)
        return RamSource(shard_dir, n_layers, spec, pin=False), False


def build_streamed_model(
    *,
    model_id: str,
    shard_dir: str,
    index: Any,
    lora_config: Any,
    device: str = "cuda",
    dtype: str = "bfloat16",
    buffers: int = 2,
    pin: bool = True,
    seed: int = 0,
    trust_remote_code: bool = False,
    console: Any = None,
    quant: str = "none",
    double_quant: bool = True,
    tier: str = "ram",
) -> Tuple[Any, StreamRuntime]:
    """Meta skeleton -> extras -> LoRA -> streaming. No resident base load."""
    from peft import get_peft_model

    model = build_meta_skeleton(
        model_id,
        dtype=dtype,
        quant=quant,
        double_quant=double_quant,
        trust_remote_code=trust_remote_code,
    )
    extras = materialize_extras(model, shard_dir, index, device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, lora_config)
    materialize_meta_adapters(model, seed=seed, device=device)
    runtime = install_streaming(
        model,
        shard_dir=shard_dir,
        index=index,
        buffers=buffers,
        pin=pin,
        device=device,
        console=console,
        codes=extras.codes,
        tier=tier,
    )
    return model, runtime
