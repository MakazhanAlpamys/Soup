# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: ``prepare`` raises :class:`UnsupportedError` instead of returning None with a
# global reason; LoRA dropout, the 4-bit layer streaming and the debug capture list are not
# ported; the state a step needs (stash, host tables, plan, int8 switch, rounding) is read from
# the run object (``runtime.PackedRun``) instead of module singletons.
"""The training forward of Soup's 4-bit LoRA Qwen2 over packed tokens: padding removed inside the
batch, every row on its own positions 0..len-1, attention never crossing rows.

Base Soup pads each batch to its longest row and runs every layer on the pads too. Real tokens give
the same results either way (right padding, causal attention, pads are never keys), so dropping
the pads is exact; it only changes how much work and memory a step takes. The collator's batch, its
attention_mask (TRL counts num_tokens from it) and the order of batches are untouched.

Shapes (hidden size, heads, kv heads, head size, intermediate size, vocabulary, layer count, tied or
separate lm_head), the compute dtype (fp16 on a T4, bf16 on Ampere and newer) and the LoRA targets
(any subset of q, k, v, o, gate, up, down) are read from the model; nothing assumes a particular
size. :func:`prepare` raises :class:`UnsupportedError`, with the reason, for a model it does not
recognise; the caller then runs the stock forward (or stops, as ``on_unsupported`` says).
"""

import torch
from torch.nn import functional

from .layers import PROJECTIONS, Adapter, DecoderLayer, NormHead, prefetch_saved, uses_int8
from .offload import layer_bytes
from .support import OPEN_ARCHITECTURES

FP_DTYPES = (torch.float16, torch.bfloat16)


class Layer:
    """References into one decoder layer, taken once: 4-bit weights, biases, LoRA, norms."""


class Step:
    """Per-step packing: positions, row spans, rotary tables, and the run's numerics."""


class Packed:
    """What :func:`prepare` found in a model; the input of :func:`forward`."""


class UnsupportedError(Exception):
    """The model is not the shape the packed forward handles; the message says why."""


def unwrap(model):
    """The causal LM inside a peft wrapper (a plain causal LM is returned as it is)."""
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _lora(proj):
    """(lora_A weight, lora_B weight, scaling) of a plain single-adapter peft LoRA layer with no
    dropout; raises UnsupportedError otherwise."""
    adapters = list(getattr(proj, "active_adapters", None) or [])
    if len(adapters) != 1 or getattr(proj, "merged", False):
        raise UnsupportedError("a LoRA layer with several (or merged) adapters")
    name = adapters[0]
    try:
        lora_a, lora_b = proj.lora_A[name], proj.lora_B[name]
        dropout, scaling = proj.lora_dropout[name], proj.scaling[name]
    except (AttributeError, KeyError):
        raise UnsupportedError("an unrecognised LoRA layer") from None
    if getattr(proj, "use_dora", {}).get(name, False) or lora_b.bias is not None:
        raise UnsupportedError("DoRA / a LoRA-B bias")
    if isinstance(dropout, torch.nn.Dropout):
        p = float(dropout.p)
    elif isinstance(dropout, torch.nn.Identity):
        p = 0.0
    else:
        raise UnsupportedError(f"LoRA dropout of type {type(dropout).__name__}")
    if p != 0.0:
        raise UnsupportedError(f"LoRA dropout {p} (the kernels need lora.dropout: 0)")
    return lora_a.weight, lora_b.weight, float(scaling)


def _four_bit(module, dt):
    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    if state is None or state.dtype != dt or len(state.shape) != 2:
        return None
    return weight


def _exact_in(weight, dtype):
    """True if an fp32 tensor only holds values of ``dtype`` (fp16 / bf16) (the base loads the model
    in that dtype and prepare_model_for_kbit_training upcasts it), checked in slices to avoid a big
    temporary."""
    step = 8192
    for start in range(0, weight.shape[0], step):
        part = weight[start:start + step]
        if not torch.equal(part.to(dtype).float(), part):
            return False
    return True


def _adapters_elsewhere(model):
    """Names of modules outside the seven projections that carry a LoRA (the kernels would
    train around them in silence)."""
    allowed = {f"{block}.{name}" for block in ("self_attn", "mlp") for name in (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            if not any(name.endswith(suffix) for suffix in allowed):
                found.append(name)
    return found


def prepare(model, dt, architectures=OPEN_ARCHITECTURES):
    """Collect what the packed forward needs from a ``Qwen2ForCausalLM`` under peft, for compute
    dtype ``dt``. Raises :class:`UnsupportedError` (the message is the reason) if the model is not
    the shape this code handles. ``architectures``: the class names to accept."""
    import bitsandbytes as bnb

    from .triton import nf4

    model = unwrap(model)
    if type(model).__name__ not in architectures:
        raise UnsupportedError(f"architecture {type(model).__name__} "
                          f"(supported: {', '.join(architectures)})")
    config = model.config
    if getattr(config, "use_sliding_window", False):
        raise UnsupportedError("sliding-window attention")
    if config.hidden_act != "silu":
        raise UnsupportedError(f"activation {config.hidden_act}")
    if getattr(config, "_attn_implementation", "sdpa") not in ("sdpa", "eager"):
        raise UnsupportedError(f"attention implementation {config._attn_implementation}")
    inner = model.model
    q_heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // q_heads
    if head_dim % 2 or q_heads % kv_heads or kv_heads & (kv_heads - 1):
        raise UnsupportedError(f"{q_heads} heads / {kv_heads} kv heads / head size {head_dim}")
    elsewhere = _adapters_elsewhere(model)
    if elsewhere:
        raise UnsupportedError(f"a LoRA on {elsewhere[0]} (the kernels run the seven "
                               f"projections of each layer only)")
    layers, names = [], None
    for block in inner.layers:
        attn, mlp = block.self_attn, block.mlp
        if hasattr(attn, "q_norm") != hasattr(attn, "k_norm"):
            raise UnsupportedError("a q norm without a k norm")
        projections = dict(zip(PROJECTIONS, (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                                             mlp.gate_proj, mlp.up_proj, mlp.down_proj)))
        bases, found = {}, {}
        for name, proj in projections.items():
            base = getattr(proj, "base_layer", None)
            if base is None:  # a projection without a LoRA
                base = proj
            if not isinstance(base, bnb.nn.Linear4bit):
                raise UnsupportedError(f"{name}_proj is not a 4-bit layer")
            bases[name] = base
            if base is not proj:
                found[name] = _lora(proj)
        here = tuple(n for n in PROJECTIONS if n in found)
        if names is None:
            names = here
        elif here != names:
            raise UnsupportedError("LoRA targets differ between layers")
        scalings = {found[n][2] for n in found if n in "qkv"}
        if len(scalings) > 1:
            raise UnsupportedError("different LoRA scalings on q, k, v")
        weights = [_four_bit(bases[n], dt) for n in PROJECTIONS]
        biases = [bases["q"].bias, bases["k"].bias, bases["v"].bias]
        norms = (block.input_layernorm.weight, block.post_attention_layernorm.weight)
        if any(w is None for w in weights):
            raise UnsupportedError(f"4-bit weights not quantized for compute dtype {dt}")
        if len({b is None for b in biases}) != 1 or any(
                bases[n].bias is not None for n in ("o", "gate", "up", "down")):
            raise UnsupportedError("unexpected biases")
        if any(n.dtype != torch.float32 for n in norms):
            raise UnsupportedError("norm weights are not fp32")
        layer = Layer()
        if not nf4.Pack.supported(weights):
            raise UnsupportedError("4-bit weights in a layout the decoder does not handle")
        layer.pack = nf4.Pack(weights)  # q, k, v, o, gate, up, down
        layer.dt = dt
        layer.qk_norm = None
        if hasattr(attn, "q_norm"):  # Qwen3: RMSNorm of every q / k head, weight [head_dim]
            eps = attn.q_norm.variance_epsilon
            weights_qk = [attn.q_norm.weight.detach().float().contiguous(),
                          attn.k_norm.weight.detach().float().contiguous()]
            if (attn.k_norm.variance_epsilon != eps
                    or any(w.shape != (head_dim,) for w in weights_qk)):
                raise UnsupportedError(
                    "q / k norms of a shape or epsilon the kernels do not handle")
            layer.qk_norm = (weights_qk[0], weights_qk[1], float(eps))
        layer.qkv_bias = (None if biases[0] is None
                          else torch.cat([b.detach() for b in biases]).to(dt))
        layer.ln1, layer.ln2 = (n.detach() for n in norms)
        layer.eps = block.input_layernorm.variance_epsilon
        layer.lora = {n: Adapter(found[n][0], found[n][1], found[n][2]) for n in names}
        layer.names = names
        layer.scaling_qkv = next((found[n][2] for n in "qkv" if n in found), 1.0)
        layer.q_heads, layer.kv_heads, layer.head_dim = q_heads, kv_heads, head_dim
        # their LoRA input comes back from the stash
        layer.save_x = "gate" in found or "up" in found
        # saved tensors in save order: r, q, k, v, o [, x_hat q, x_hat k (Qwen3)] | x | gate, up
        # (DecoderLayer.forward)
        n_rest = 7 if layer.qk_norm is not None else 5
        layer.groups = {"rest": (0, n_rest)}
        if layer.save_x:
            layer.groups["x"] = (n_rest, n_rest + 1)
        at = n_rest + 1 if layer.save_x else n_rest
        layer.groups["gate_up"] = (at, at + 2)
        layer.index = len(layers)
        layers.append(layer)
    if len({layer.qk_norm is None for layer in layers}) > 1:
        raise UnsupportedError("q / k norms in some layers only")

    embed, head = inner.embed_tokens, model.lm_head
    if getattr(head, "bias", None) is not None:
        raise UnsupportedError("an lm_head bias")
    if embed.weight.requires_grad or head.weight.requires_grad:
        raise UnsupportedError("a trainable embedding or lm_head (the kernels read them as frozen)")
    tied = head.weight is embed.weight
    converted = False
    for module in (embed,) if tied else (embed, head):
        weight = module.weight
        if weight.dtype == torch.float32 and _exact_in(weight, dt):
            # The frozen table holds `dt` values in fp32 storage (prepare_model_for_kbit_training
            # upcast it). Keep the same values in `dt` storage: the embedding lookup upcasts them
            # exactly, and the lm_head matmul used their `dt` cast anyway (autocast).
            module.weight = torch.nn.Parameter(weight.detach().to(dt), requires_grad=False)
            converted = True
    if tied:
        head.weight = embed.weight
    if converted:
        model._packed_lora_fp32_embeds = True  # the stock forward upcasts its embedding output
    if embed.weight.dtype != dt or head.weight.dtype != dt:
        raise UnsupportedError(f"embedding / lm_head dtype {embed.weight.dtype} / "
                               f"{head.weight.dtype} for compute dtype {dt}")

    packed = Packed()
    packed.layers = layers
    packed.names = names
    packed.dtype = dt
    # compute-dtype copies of every LoRA weight (what the matmuls use), refreshed once per step by
    # one multi-tensor copy instead of two casts per adapter and layer
    adapters = [layer.lora[n] for layer in layers for n in names]
    packed.lora32 = [t for ad in adapters for t in (ad.a32, ad.b32)]
    packed.lora16 = [torch.empty(t.shape, device=t.device, dtype=dt) for t in packed.lora32]
    for i, ad in enumerate(adapters):
        ad.a16, ad.b16 = packed.lora16[2 * i], packed.lora16[2 * i + 1]
    for layer in layers:
        layer.params = [t for n in names for t in (layer.lora[n].a32, layer.lora[n].b32)]
    packed.lora_bytes = sum(t.numel() * 4 for t in packed.lora32)
    packed.embed = embed.weight
    packed.head = head.weight
    packed.tied = tied
    packed.norm, packed.norm_eps = inner.norm.weight.detach(), inner.norm.variance_epsilon
    packed.rotary = inner.rotary_emb
    packed.q_heads, packed.kv_heads, packed.head_dim = q_heads, kv_heads, head_dim
    packed.q_dim, packed.kv_dim = q_heads * head_dim, kv_heads * head_dim
    packed.hidden = config.hidden_size
    packed.intermediate = config.intermediate_size
    packed.save_x = layers[0].save_x
    packed.qk_norm = layers[0].qk_norm is not None
    packed.groups = layers[0].groups
    packed.config = config
    packed.resident_layers = 0
    return packed


def embed_for_stock_forward(model, input_ids):
    """The embedding the stock forward would compute, for a model whose tables ``prepare`` (or
    :func:`keep_tables_in_compute_dtype`) kept in the compute dtype: the stock forward keeps an
    fp32 residual stream only from an fp32 embedding, so it is fed ``inputs_embeds`` as fp32."""
    return model.get_input_embeddings()(input_ids).float()


def keep_tables_in_compute_dtype(model):
    """Context manager for ``prepare_model_for_kbit_training``: it upcasts every fp16 / bf16
    parameter that is not 4-bit to fp32, the embedding and the lm_head included: a 0.9 GB fp32
    copy of values that are all fp16 for a 1.5B model, 2 x 2.2 GB for 7B, made while the original
    is still alive. The packed forward reads the tables in the compute dtype anyway, so inside this
    context they are left in their loaded storage (same values, half the memory, no transient).
    Norms and biases are upcast as before. Afterwards the stock forward needs
    :func:`embed_for_stock_forward` (``model._packed_lora_fp32_embeds`` says so)."""
    import contextlib

    @contextlib.contextmanager
    def context():
        tables = []
        if getattr(model, "is_loaded_in_4bit", False):
            for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                weight = getattr(module, "weight", None)
                if (weight is not None and weight.dtype in FP_DTYPES and weight.is_cuda
                        and not any(weight is t for t in tables)):
                    tables.append(weight)
        stored = [(w, w.data) for w in tables]
        for w, data in stored:
            w.data = torch.empty(0, device=data.device, dtype=torch.float32)
        try:
            yield
        finally:
            for w, data in stored:
                w.data = data
            if stored:
                model._packed_lora_fp32_embeds = True

    return context()


def describe(packed):
    """One line for the log: which path runs and with what."""
    layer = packed.layers[0]
    rank = layer.lora[packed.names[0]].a32.shape[0] if packed.names else 0
    return (f"packed forward: {type(packed.config).__name__.replace('Config', '')} "
            f"{len(packed.layers)} layers, hidden {packed.hidden}, "
            f"{packed.q_heads}/{packed.kv_heads} heads x {packed.head_dim}, "
            f"intermediate {packed.intermediate}, "
            f"{'tied' if packed.tied else 'separate'} lm_head, dtype {str(packed.dtype)[6:]}, "
            f"LoRA on {','.join(packed.names) or 'nothing'} (r={rank})"
            + (", per-head q/k norms" if packed.qk_norm else ""))


def forward(packed, run, input_ids, attention_mask, labels, n_items):
    """(loss, stats) for one training batch, or None if the batch is not right-padded. ``run``:
    the :class:`~soup_cli.kernels.packed_lora.runtime.PackedRun` that owns the stash, the host
    tables, the plan and the numerics of this training run."""
    rows, width = input_ids.shape
    dev = input_ids.device
    dt = packed.dtype
    lengths = attention_mask.sum(1)
    positions = torch.arange(width, device=dev)
    real = positions[None, :] < lengths[:, None]
    targets_padded = functional.pad(labels[:, 1:], (0, 1), value=-100)
    # one GPU sync for everything the host needs: row lengths, labelled count, right padding
    # (with a table in host memory, also the tokens and targets, to gather its rows)
    tables = run.tables
    embed_host = tables is not None and tables.active_embed()
    head_host = tables is not None and tables.active_head()
    on_host = embed_host or head_host
    needed = torch.cat([lengths, (targets_padded != -100).sum()[None],
                        (real != attention_mask.bool()).sum()[None]])
    if on_host:
        blob = torch.cat([needed, input_ids.reshape(-1), targets_padded.reshape(-1)]).cpu()
        host = blob[:rows + 2].tolist()
    else:
        host = needed.tolist()
    row_lengths, n_labelled, misplaced = host[:rows], host[rows], host[rows + 1]
    if misplaced:
        return None
    flat = real.reshape(-1).nonzero().squeeze(1)
    ids = input_ids.reshape(-1).index_select(0, flat)
    targets = targets_padded.reshape(-1).index_select(0, flat)
    if n_items is None:
        n_items = (targets != -100).sum()
    embedded = target_rows = None
    if on_host:
        # the same tokens in the same order as flat, gathered from the host tables in one transfer
        cpu_flat = torch.cat([torch.arange(n) + r * width for r, n in enumerate(row_lengths)])
        cpu_ids = blob[rows + 2:rows + 2 + rows * width][cpu_flat]
        cpu_targets = blob[rows + 2 + rows * width:][cpu_flat]
        cpu_targets = cpu_targets[cpu_targets != -100]
        if embed_host and head_host:
            embedded, target_rows = tables.gather_pair(cpu_ids, cpu_targets, dev)
        elif embed_host:
            embedded = tables.gather(cpu_ids, dev)
        else:
            target_rows = tables.head.gather(cpu_targets, dev)

    step = Step()
    step.q_heads, step.kv_heads, step.head_dim = packed.q_heads, packed.kv_heads, packed.head_dim
    step.attn_scale = packed.head_dim ** -0.5
    step.int8, step.rounding = run.int8, run.rounding
    step.pos = positions.expand(rows, width).reshape(-1).index_select(0, flat).to(torch.int32)
    spans, start = [], 0
    for n in row_lengths:
        spans.append((start, start + n))
        start += n
    step.spans = spans
    # row boundaries for the varlen attention kernel, one small copy per step for all layers
    step.cu = torch.tensor([0] + [stop for _, stop in spans], dtype=torch.int32, device=dev)
    step.max_len = max(row_lengths)
    probe = torch.empty(1, device=dev, dtype=torch.float32)
    cos, sin = packed.rotary(probe, positions[None, :max(row_lengths)])
    step.cos, step.sin = cos[0].float().contiguous(), sin[0].float().contiguous()

    n_layers = len(packed.layers)
    int8_stash = uses_int8(packed.layers[0], len(ids), run.int8)
    per_layer = layer_bytes(len(ids), packed.hidden, packed.intermediate, packed.q_dim,
                            packed.kv_dim, packed.save_x, int8_stash, packed.qk_norm)
    stash = run.stash_for_step(packed, len(ids), per_layer, int8_stash, dev)
    stash.begin_step(dev)
    last = n_layers - 1
    on_backward = lambda: prefetch_saved(stash, packed.layers[last], last, dev)  # noqa: E731
    packed.resident_layers = n_layers - min(stash.resident_from, n_layers)

    stats = {}
    with torch.no_grad():  # the same values as aq.half() etc., in one launch for all layers
        torch._foreach_copy_(packed.lora16, packed.lora32)
    with torch.autocast("cuda", enabled=False):
        r = (embedded if embed_host else functional.embedding(ids, packed.embed)).float()
        try:
            for layer in packed.layers:
                r = DecoderLayer.apply(r, layer, step, stash, *layer.params)
        finally:
            stash.release()  # the last offloaded layer's tensors
        head_weight = tables.head.host if head_host else packed.head
        loss = NormHead.apply(r, packed.norm, packed.norm_eps, head_weight, targets, n_items,
                              n_labelled, stats, on_backward, target_rows, dt)

    token_loss = torch.zeros(rows * width, device=dev, dtype=torch.float32)
    token_loss[flat] = stats["token_loss"]
    stats["token_loss"] = token_loss.view(rows, width)
    stats["labelled"] = (targets != -100).sum()
    return loss, stats
