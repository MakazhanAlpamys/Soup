"""Expert-granularity streaming for MoE decoder layers.

Layer streaming moves one decoder layer at a time, so the largest layer bounds
the card. A DeepSeek-V3-class MoE layer is mostly routed experts: one Kimi K2
layer (384 experts of 3 x 7168 x 2048) is ~9 GB at NF4, larger than an 8 GB
card on its own. Here the routed experts of a MoE layer live in their own file,
one tensor pair per expert, and the layer's experts module is replaced by
``StreamedExperts``: it loads each expert the router picked, uses it, and drops
it. The decoder-layer pool then carries only the dense part (attention, norms,
router, shared expert).

Numerics: ``StreamedExperts`` follows transformers' eager experts loop op for
op (the same body in ``DeepseekV3Experts``, ``OlmoeExperts``,
``Qwen3MoeExperts``, ``MixtralExperts``), so it equals a resident module run
with ``experts_implementation="eager"`` bit for bit, forward and backward.
Expert weights are frozen; the backward needs only the gradient of the
expert's input, so it fetches the expert again and recomputes the expert MLP
on a local graph. Inside a checkpointed decoder layer that is three reads of
every routed-to expert per step (forward, recompute, backward).

What it buys: the VRAM bound, nothing else. ``benchmarks/probe-moe-expert-coverage.md``
measured that a training step routes to 96.7-100% of the experts of a layer, so
an expert-streamed step reads at least the bytes a layer-streamed step reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

#: Tensor names inside a per-layer experts file: ``"<expert>.gate_up"`` is the
#: ``[2I, H]`` matrix (gate rows first, as transformers' fused ``gate_up_proj``
#: row ``e``), ``"<expert>.down"`` the ``[H, I]`` one.
GATE_UP = "gate_up"
DOWN = "down"
EXPERT_PARTS = (GATE_UP, DOWN)
#: The index's marker for a cache whose MoE layers keep their routed experts
#: in per-layer experts files instead of inside the decoder-layer file.
EXPERT_LAYOUT_PER_EXPERT = "per_expert"
_MAX_EXPERTS = 4096
#: One expert's stored bytes may not exceed this: the reader allocates its staging
#: from the header, so an unbounded header would size an unbounded allocation. A
#: Kimi K2 expert is ~25 MB at NF4, ~88 MB at bf16.
_MAX_EXPERT_BYTES = 2**30


def expert_key(expert: int, part: str) -> str:
    """Shard key of one expert's matrix."""
    return f"{expert}.{part}"


def expert_shard_name(layer: int) -> str:
    """File name of a MoE layer's experts file."""
    return f"experts_{layer:03d}.safetensors"


@dataclass(frozen=True)
class ExpertShardInfo:
    """What :func:`write_expert_shard` wrote for one MoE layer."""

    count: int
    #: part -> stored dense shape, e.g. {"gate_up": (2I, H), "down": (H, I)}
    shapes: Mapping[str, Tuple[int, ...]]
    #: part -> NF4WeightSpec; empty when unquantised
    quant_specs: Mapping[str, Any] = field(default_factory=dict)
    total_params: int = 0
    #: the NF4 code tables seen while quantising (``code``, ``nested_code``)
    codes: Tuple[Any, Any] = (None, None)


def write_expert_shard(
    path: str,
    experts: Iterable[Tuple[Any, Any]],
    *,
    quantise: bool,
    double_quant: bool = True,
    device: str = "cpu",
) -> ExpertShardInfo:
    """Write one MoE layer's routed experts, one ``(gate_up, down)`` pair each.

    ``experts`` yields the dense matrices in expert order, already at the
    stream dtype. Under ``quantise`` each matrix is NF4-quantised on its own,
    exactly as the decoder-layer sharder quantises one ``nn.Linear`` weight.
    Every expert must have the shapes of the first one.
    """
    from soup_cli.utils.layer_shard import _atomic_save, _quantize_nf4

    blob: Dict[str, Any] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    specs: Dict[str, Any] = {}
    code = nested_code = None
    count = 0
    total = 0
    for expert, pair in enumerate(experts):
        if expert >= _MAX_EXPERTS:
            raise ValueError(f"a MoE layer may hold at most {_MAX_EXPERTS} experts")
        for part, tensor in zip(EXPERT_PARTS, pair):
            shape = tuple(int(dim) for dim in tensor.shape)
            if len(shape) != 2:
                raise ValueError(f"expert {expert} {part} must be 2-D, got {shape}")
            prior = shapes.setdefault(part, shape)
            if prior != shape:
                raise ValueError(
                    f"expert {expert} {part} has shape {shape}; expert 0 has {prior}"
                )
            total += tensor.numel()
            key = expert_key(expert, part)
            if not quantise:
                blob[key] = tensor.detach().contiguous()
                continue
            sidecars, spec, seen_code, seen_nested = _quantize_nf4(
                tensor, double_quant=double_quant, device=device
            )
            prior_spec = specs.setdefault(part, spec)
            if prior_spec != spec:
                raise ValueError(f"expert {expert} {part} has inconsistent NF4 metadata")
            code = seen_code if code is None else code
            nested_code = seen_nested if nested_code is None else nested_code
            for suffix, value in sidecars.items():
                blob[key + suffix] = value
        count += 1
    if count == 0:
        raise ValueError("a MoE layer needs at least one expert")
    if shapes[GATE_UP][1] != shapes[DOWN][0] or shapes[GATE_UP][0] != 2 * shapes[DOWN][1]:
        raise ValueError(
            f"gate_up {shapes[GATE_UP]} and down {shapes[DOWN]} are not one expert MLP "
            f"([2I, H] and [H, I])"
        )
    _atomic_save(blob, path)
    return ExpertShardInfo(
        count=count,
        shapes=shapes,
        quant_specs=specs,
        total_params=total,
        codes=(code, nested_code),
    )


def expert_bytes(
    shapes: Mapping[str, Tuple[int, ...]], *, dtype_bytes: int, quantised: bool
) -> Tuple[int, int]:
    """(stored bytes, dense compute bytes) of ONE expert.

    Stored NF4 is half a byte per weight plus one fp32 absmax per 64-weight
    block (the nested statistics are smaller still and not counted).
    """
    weights = sum(shape[0] * shape[1] for shape in shapes.values())
    dense = weights * dtype_bytes
    if not quantised:
        return dense, dense
    stored = weights // 2 + (weights // 64) * 4
    return stored, dense


def expert_stream_vram_bytes(
    shapes: Mapping[str, Tuple[int, ...]], *, dtype_bytes: int, quantised: bool
) -> int:
    """Peak VRAM the expert path adds to a step, for the pre-flight.

    One expert's packed copy plus its dense weights, twice: the backward holds
    the expert it recomputes while the caching allocator may still keep the
    previous one's blocks.
    """
    stored, dense = expert_bytes(shapes, dtype_bytes=dtype_bytes, quantised=quantised)
    packed = stored if quantised else 0
    return 2 * (packed + dense)


class ExpertShardReader:
    """Reads one MoE layer's experts file, one expert at a time.

    Each stored tensor of the expert is one positioned read of its own
    sector-aligned span, through the shared ``safetensors_reader`` (its header
    is validated against the file), so the file's tensor order does not matter.
    No memory map: a live safetensors map charges Windows commit for the whole
    file, and a 1T-class store has ~60 files of ~9 GB. Direct (unbuffered) I/O
    when the platform has it, so a store far larger than RAM does not churn the
    page cache of the whole box; buffered reads otherwise.

    The header is read through the identity of a handle opened for it, and the
    handle is closed again before ``__init__`` returns: a reader holds no OS
    handle until its first fetch, so an install that fails after the readers
    were built leaks nothing. The first fetch reopens the file and refuses it
    unless its identity is still the one the header was read from, so a file
    swapped in between is never read at stale offsets. From then on the handle
    stays open, which on Windows also keeps the file from being replaced.
    Every stored tensor's size is checked against the index (dense shape, NF4
    spec), so a header-valid file cannot steer a read or a dequantisation past
    the bytes it really holds.
    """

    def __init__(
        self,
        path: str,
        *,
        count: int,
        shapes: Mapping[str, Tuple[int, ...]],
        quant_specs: Mapping[str, Any],
        dtype: str,
        direct: bool = True,
    ) -> None:
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys
        from soup_cli.utils.safetensors_reader import (
            aligned_span,
            identity_of,
            read_header_with_identity,
        )

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= _MAX_EXPERTS:
            raise ValueError(f"expert count must be an int in [1, {_MAX_EXPERTS}], got {count!r}")
        self.path = path
        self.count = count
        self.shapes = {part: _checked_shape(part, shapes[part]) for part in EXPERT_PARTS}
        self.quant_specs = dict(quant_specs)
        self.dtype = dtype
        self.direct = direct
        self._handle: Optional[Any] = None
        name = os.path.basename(path)
        self._identity: Optional[Any] = None
        handle = self._open()
        header, identity = read_header_with_identity(path)
        if identity_of(handle) != identity:
            self.close()
            raise ValueError(f"experts file {name} changed while it was being opened")
        self._identity = identity
        self._size = identity.size
        expected = {
            part: _expected_tensors(
                expert_key(0, part), self.shapes[part], self.quant_specs.get(part)
            )
            for part in EXPERT_PARTS
        }
        stored_dtype = _stored_torch_dtype(dtype)
        #: per expert: ((tensor name, entry), ...)
        self._entries: list = []
        largest = 0
        for expert in range(count):
            entries = []
            for part in EXPERT_PARTS:
                key = expert_key(expert, part)
                spec = self.quant_specs.get(part)
                wanted = quant_sidecar_keys(key, spec) if spec is not None else (key,)
                for tensor_name, (want_shape, want_dtype) in zip(wanted, expected[part]):
                    entry = header.get(tensor_name)
                    if entry is None:
                        self.close()
                        raise ValueError(
                            f"experts file {name} has no tensor {tensor_name!r} "
                            f"— reshard the checkpoint"
                        )
                    got_dtype = _stored_torch_dtype(entry.dtype)
                    if want_dtype is None:
                        want_dtype = stored_dtype
                    if tuple(entry.shape) != want_shape or got_dtype != want_dtype:
                        self.close()
                        raise ValueError(
                            f"experts file {name}: {tensor_name!r} is {entry.dtype} "
                            f"{tuple(entry.shape)}, the index expects {want_dtype} "
                            f"{want_shape} — reshard the checkpoint"
                        )
                    if entry.start % got_dtype.itemsize:
                        self.close()
                        raise ValueError(
                            f"experts file {name}: {tensor_name!r} is not aligned to its dtype"
                        )
                    entries.append((tensor_name, entry))
            need = 0
            for _, entry in entries:
                lo, hi = aligned_span(entry.start, entry.end)
                need += hi - lo
            if need > _MAX_EXPERT_BYTES:
                self.close()
                raise ValueError(
                    f"experts file {name}: expert {expert} needs {need} bytes, over the "
                    f"{_MAX_EXPERT_BYTES}-byte limit for one expert"
                )
            largest = max(largest, need)
            self._entries.append(tuple(entries))
        if expert_key(count, GATE_UP) in header or expert_key(count, DOWN) in header:
            self.close()
            raise ValueError(
                f"experts file {name} holds more than {count} experts — reshard the checkpoint"
            )
        #: Staging bytes one expert needs (each tensor's span on sector bounds).
        self.max_span = largest
        self.fetches = 0
        self.bytes_read = 0
        self.close()

    def _open(self) -> Any:
        if self._handle is None:
            if self.direct:
                from soup_cli.utils.safetensors_reader import open_direct

                try:
                    self._handle = open_direct(self.path)
                except OSError:
                    # No direct I/O on this platform / filesystem: buffered reads.
                    self.direct = False
            if self._handle is None:
                self._handle = open(self.path, "rb", buffering=0)
            if self._identity is not None:
                from soup_cli.utils.safetensors_reader import identity_of

                if identity_of(self._handle) != self._identity:
                    self.close()
                    raise ValueError(
                        f"experts file {os.path.basename(self.path)} changed on disk since "
                        f"its header was read — reshard the checkpoint and restart"
                    )
        return self._handle

    def read_host(self, expert: int, staging: Any) -> Dict[str, Any]:
        """``expert``'s stored tensors as CPU views into ``staging``.

        ``staging`` is a contiguous CPU uint8 tensor of at least ``max_span``
        bytes whose address is sector-aligned (``aligned_staging``). The views
        are valid until the next read into it.
        """
        from soup_cli.utils.safetensors_reader import aligned_span, read_range_into

        if not 0 <= expert < self.count:
            raise IndexError(f"expert {expert} outside [0, {self.count})")
        if staging.numel() < self.max_span:
            raise ValueError(f"staging holds {staging.numel()} bytes, a read needs {self.max_span}")
        handle = self._open()
        out: Dict[str, Any] = {}
        cursor = 0
        for tensor_name, entry in self._entries[expert]:
            lo, hi = aligned_span(entry.start, entry.end)
            region = staging[cursor : cursor + hi - lo]
            read_range_into(handle, lo, region, min(hi, self._size) - lo)
            raw = region[entry.start - lo : entry.end - lo]
            out[tensor_name] = raw.view(_stored_torch_dtype(entry.dtype)).view(entry.shape)
            cursor += hi - lo
            self.bytes_read += entry.nbytes
        self.fetches += 1
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best effort at interpreter teardown
            pass


def _checked_shape(part: str, shape: Any) -> Tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 2 or any(dim < 1 for dim in dims) or dims[0] * dims[1] > 2**31:
        raise ValueError(f"expert {part} shape must be two positive dims, got {tuple(shape)!r}")
    return dims


def _expected_tensors(key: str, shape: Tuple[int, int], spec: Any) -> Tuple[Tuple[Any, Any], ...]:
    """(shape, torch dtype) of every stored tensor of one expert matrix, in the
    order ``quant_sidecar_keys`` names them. ``None`` dtype = the stream dtype.

    NF4 (bitsandbytes ``quantize_4bit``): packed nibbles ``[ceil(n/2), 1]`` uint8,
    one absmax per ``blocksize`` weights (fp32, or uint8 codes under double
    quantisation with an fp32 ``nested_absmax`` per ``nested_blocksize`` and a
    scalar fp32 ``nested_offset``).
    """
    import torch

    if spec is None:
        return ((shape, None),)
    if tuple(int(dim) for dim in spec.shape) != shape:
        raise ValueError(f"{key}: NF4 spec shape {tuple(spec.shape)} is not the index's {shape}")
    nested_ok = not spec.nested or int(spec.nested_blocksize) >= 1
    if not 1 <= int(spec.blocksize) <= 4096 or not nested_ok:
        raise ValueError(f"{key}: NF4 spec has an impossible block size")
    weights = shape[0] * shape[1]
    blocks = -(-weights // int(spec.blocksize))
    out = [(((weights + 1) // 2, 1), torch.uint8)]
    if spec.nested:
        out.append(((blocks,), torch.uint8))
        out.append(((-(-blocks // int(spec.nested_blocksize)),), torch.float32))
        out.append(((), torch.float32))
    else:
        out.append(((blocks,), torch.float32))
    return tuple(out)


def aligned_staging(size: int) -> Tuple[Any, Any]:
    """``(owner, view)``: ``size`` bytes of CPU memory starting on a sector."""
    import torch

    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    owner = torch.empty(size + SECTOR_BYTES, dtype=torch.uint8, device="cpu")
    pad = (-owner.data_ptr()) % SECTOR_BYTES
    return owner, owner[pad : pad + size]


def _stored_torch_dtype(name: str) -> Any:
    import torch

    # ``safetensors_reader`` reports torch names ("bfloat16"); the file format
    # itself spells them "BF16". Either is accepted, nothing else is.
    table = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "U8": torch.uint8,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "uint8": torch.uint8,
    }
    if name not in table:
        raise ValueError(f"experts file stores unsupported dtype {name!r}")
    return table[name]


class ExpertFetcher:
    """Turns stored expert tensors into dense weights on the compute device.

    Owns the one host staging buffer every reader it serves reads into, so a
    60-layer store costs one expert's span of host RAM, not sixty. Not
    thread-safe: one fetch at a time (the streamed forward and backward are
    sequential); a concurrent fetch would overwrite the staging under another.
    """

    def __init__(self, *, device: str, dtype: str, codes: Optional[Mapping[str, Any]] = None):
        self.device = device
        self.dtype = dtype
        self.codes = dict(codes or {})
        self._staging_owner: Any = None
        self._staging: Any = None

    def _staging_for(self, reader: ExpertShardReader) -> Any:
        if self._staging is None or self._staging.numel() < reader.max_span:
            self._staging_owner, self._staging = aligned_staging(reader.max_span)
        return self._staging

    def fetch(self, reader: ExpertShardReader, expert: int) -> Tuple[Any, Any]:
        from soup_cli.utils.layer_stream_runtime import _torch_dtype, rebuild_quant_state

        host = reader.read_host(expert, self._staging_for(reader))
        target = _torch_dtype(self.dtype)
        dense = []
        for part in EXPERT_PARTS:
            key = expert_key(expert, part)
            spec = reader.quant_specs.get(part)
            if spec is None:
                # copy=True: on a CPU target `.to` would hand back the staging
                # view, which the next read overwrites.
                dense.append(host[key].to(device=self.device, dtype=target, copy=True))
                continue
            from bitsandbytes.functional import dequantize_4bit

            moved = {
                name: tensor.to(self.device, copy=True)
                for name, tensor in host.items()
                if name == key or name.startswith(key + "::")
            }
            state = rebuild_quant_state(key, moved, spec, self.codes)
            dense.append(dequantize_4bit(moved[key], quant_state=state).to(target))
        return dense[0], dense[1]


def _build_streamed_experts_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def expert_mlp(hidden: Any, gate_up: Any, down: Any, act_fn: Any) -> Any:
        # transformers' eager experts body, line for line.
        gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
        current = act_fn(gate) * up
        return functional.linear(current, down)

    class _ExpertMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, owner, expert):  # noqa: D401 - autograd API
            ctx.device_type = hidden.device.type
            ctx.autocast_kwargs = {
                "enabled": torch.is_autocast_enabled(ctx.device_type),
                "dtype": torch.get_autocast_dtype(ctx.device_type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            gate_up, down = owner.fetch(expert)
            out = expert_mlp(hidden, gate_up, down, owner.act_fn)
            ctx.owner = owner
            ctx.expert = expert
            ctx.save_for_backward(hidden)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (hidden,) = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return None, None, None
            gate_up, down = ctx.owner.fetch(ctx.expert)
            # Backward usually runs outside the forward's autocast region.
            with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast_kwargs):
                local = hidden.detach().requires_grad_(True)
                out = expert_mlp(local, gate_up, down, ctx.owner.act_fn)
                (grad_hidden,) = torch.autograd.grad(out, local, grad_out)
            return grad_hidden, None, None

    class StreamedExperts(nn.Module):
        """Drop-in for a transformers experts module; fetches each expert on use."""

        def __init__(self, reader, fetcher, act_fn, layer_idx):
            super().__init__()
            self.reader = reader
            self.fetcher = fetcher
            self.act_fn = act_fn
            self.layer_idx = layer_idx
            self.num_experts = reader.count

        def fetch(self, expert):
            return self.fetcher.fetch(self.reader, expert)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                current = _ExpertMLP.apply(current_state, self, int(expert_idx))
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
            return final_hidden_states

    return StreamedExperts


_STREAMED_EXPERTS_CLASS = None


def streamed_experts_class() -> Any:
    global _STREAMED_EXPERTS_CLASS
    if _STREAMED_EXPERTS_CLASS is None:
        _STREAMED_EXPERTS_CLASS = _build_streamed_experts_class()
    return _STREAMED_EXPERTS_CLASS


def find_experts_module(layer: Any) -> Optional[Tuple[Any, str, Any]]:
    """``(parent, attribute, module)`` of a layer's fused routed-experts module.

    Matched by shape of contract, not by class name: a child named ``experts``
    holding 3-D ``gate_up_proj`` and ``down_proj`` parameters, the layout every
    transformers 5.x MoE uses. ``None`` for a dense layer.
    """
    for name, module in layer.named_modules():
        parent_name, _, leaf = name.rpartition(".")
        if leaf != "experts":
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if getattr(gate_up, "dim", lambda: 0)() == 3 and getattr(down, "dim", lambda: 0)() == 3:
            parent = layer.get_submodule(parent_name) if parent_name else layer
            return parent, leaf, module
    return None


def install_expert_streaming(
    layers: Sequence[Any],
    readers: Mapping[int, ExpertShardReader],
    fetcher: ExpertFetcher,
) -> int:
    """Replace each MoE layer's experts module with a ``StreamedExperts``.

    Runs before the decoder-layer name maps are taken, so the fused expert
    parameters leave the layer and the layer pool never allocates them.
    Returns the number of layers converted.
    """
    cls = streamed_experts_class()
    converted = 0
    for idx, layer in enumerate(layers):
        reader = readers.get(idx)
        found = find_experts_module(layer)
        if reader is None:
            if found is not None:
                raise ValueError(
                    f"decoder layer {idx} has a routed-experts module but the cache has "
                    f"no experts file for it — reshard the checkpoint"
                )
            continue
        if found is None:
            raise ValueError(
                f"the cache has an experts file for decoder layer {idx}, which has no "
                f"routed-experts module — reshard the checkpoint"
            )
        parent, attr, module = found
        expected_gate_up = tuple(int(dim) for dim in module.gate_up_proj.shape)
        expected_down = tuple(int(dim) for dim in module.down_proj.shape)
        stored_gate_up = (reader.count, *reader.shapes[GATE_UP])
        stored_down = (reader.count, *reader.shapes[DOWN])
        if (expected_gate_up, expected_down) != (stored_gate_up, stored_down):
            raise ValueError(
                f"decoder layer {idx}: the model's experts are {expected_gate_up} / "
                f"{expected_down} but the experts file holds {reader.count} x "
                f"{reader.shapes[GATE_UP]} / {reader.shapes[DOWN]} — reshard the checkpoint"
            )
        names = [name for name, _ in module.named_parameters()]
        if any(name.startswith("lora_") or ".lora_" in name for name in names):
            raise ValueError(
                f"decoder layer {idx}: adapters on the routed experts cannot be trained "
                f"while the experts are streamed one at a time; target attention or the "
                f"shared expert instead"
            )
        setattr(parent, attr, cls(reader, fetcher, module.act_fn, idx))
        converted += 1
    return converted
