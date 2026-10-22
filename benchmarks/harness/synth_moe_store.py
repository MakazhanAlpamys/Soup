"""Write a SYNTHETIC DeepSeek-V3-family shard cache directly: random weights, shape only.

A Kimi K2-shaped checkpoint is ~1 T parameters: ~2 TB at bf16, more than the
disk that would hold its ~0.58 TB NF4 shard cache. So this skips the bf16
source and writes the cache ``layer_shard.shard_checkpoint`` would have made
with ``quant="nf4", expert_layout="per_expert"``: an ``index.json`` the runtime
accepts, NF4 decoder-layer files, one experts file per MoE layer, the extras
and the untied embedding / head. The NF4 layout is the sharder's default,
double quantisation (uint8 absmax codes, an fp32 nested absmax per 256 blocks, a
scalar offset). Every packed NF4 byte comes from one seeded random pool reused
at rotating offsets; every block's absmax decodes to ``1/sqrt(fan_in)``. Numbers
measured on it are labelled "random weights, shape only" wherever they appear.

    python benchmarks/harness/synth_moe_store.py --out D:/soup-bench/moe-k2-synth --shape k2
    python benchmarks/harness/synth_moe_store.py --out <tmp> --shape tiny   # the test shape
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

SHAPES = {
    # moonshotai/Kimi-K2-Instruct config.json, model_type rewritten to deepseek_v3
    # (transformers registers no kimi_k2) and its FP8 quantization_config dropped.
    "k2": dict(
        hidden_size=7168, intermediate_size=18432, moe_intermediate_size=2048,
        num_hidden_layers=61, first_k_dense_replace=1, num_attention_heads=64,
        num_key_value_heads=64, n_routed_experts=384, n_shared_experts=1,
        num_experts_per_tok=8, n_group=1, topk_group=1, kv_lora_rank=512,
        q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        vocab_size=163840, routed_scaling_factor=2.827, norm_topk_prob=True,
        rope_theta=50000.0, max_position_embeddings=131072,
        rope_scaling={
            "type": "yarn", "factor": 32.0, "beta_fast": 1.0, "beta_slow": 1.0,
            "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096,
        },
    ),
    # Small enough for a unit test; same structure.
    "tiny": dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=4, n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=2, n_group=1, topk_group=1, kv_lora_rank=16,
        q_lora_rank=24, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        vocab_size=128, routed_scaling_factor=1.0, norm_topk_prob=True,
        rope_theta=10000.0, max_position_embeddings=128,
    ),
}

_POOL_BYTES = 2**28


def config_dict(shape: dict) -> dict:
    out = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_act": "silu",
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 0,
        "torch_dtype": "bfloat16",
        "use_cache": False,
    }
    out.update(shape)
    return out


def _linears(shape: dict, layer: int) -> dict:
    """short key -> (out_features, in_features) of every nn.Linear in one layer."""
    hidden = shape["hidden_size"]
    heads = shape["num_attention_heads"]
    qk_head = shape["qk_nope_head_dim"] + shape["qk_rope_head_dim"]
    out = {
        "self_attn.q_a_proj.weight": (shape["q_lora_rank"], hidden),
        "self_attn.q_b_proj.weight": (heads * qk_head, shape["q_lora_rank"]),
        "self_attn.kv_a_proj_with_mqa.weight": (
            shape["kv_lora_rank"] + shape["qk_rope_head_dim"], hidden,
        ),
        "self_attn.kv_b_proj.weight": (
            heads * (shape["qk_nope_head_dim"] + shape["v_head_dim"]), shape["kv_lora_rank"],
        ),
        "self_attn.o_proj.weight": (hidden, heads * shape["v_head_dim"]),
    }
    if layer < shape["first_k_dense_replace"]:
        inter = shape["intermediate_size"]
        prefix = "mlp."
    else:
        inter = shape["moe_intermediate_size"] * shape["n_shared_experts"]
        prefix = "mlp.shared_experts."
    out[prefix + "gate_proj.weight"] = (inter, hidden)
    out[prefix + "up_proj.weight"] = (inter, hidden)
    out[prefix + "down_proj.weight"] = (hidden, inter)
    return out


def _dense(shape: dict, layer: int) -> dict:
    """short key -> (shape, safetensors dtype, fill) of the unquantised layer tensors."""
    hidden = shape["hidden_size"]
    out = {
        "input_layernorm.weight": ((hidden,), "BF16", 1.0),
        "post_attention_layernorm.weight": ((hidden,), "BF16", 1.0),
        "self_attn.q_a_layernorm.weight": ((shape["q_lora_rank"],), "BF16", 1.0),
        "self_attn.kv_a_layernorm.weight": ((shape["kv_lora_rank"],), "BF16", 1.0),
    }
    if layer >= shape["first_k_dense_replace"]:
        experts = shape["n_routed_experts"]
        out["mlp.gate.weight"] = ((experts, hidden), "BF16", None)
        out["mlp.gate.e_score_correction_bias"] = ((experts,), "F32", 0.0)
    return out


class _Pool:
    """Seeded random bytes, handed out at rotating offsets."""

    def __init__(self, seed: int, size: int = _POOL_BYTES):
        import numpy as np

        self.data = np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8)
        self.offset = 0

    def take(self, nbytes: int):
        import numpy as np

        parts = []
        while nbytes:
            if self.offset >= self.data.size:
                self.offset = 0
            chunk = min(nbytes, self.data.size - self.offset)
            parts.append(self.data[self.offset : self.offset + chunk])
            self.offset += chunk + 4099  # rotate: consecutive takes do not repeat
            nbytes -= chunk
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


_NESTED_BLOCK = 256


def _nf4_entries(name: str, out_features: int, in_features: int):
    n = out_features * in_features
    if n % 64:
        raise ValueError(f"{name}: {n} weights is not a multiple of the NF4 block (64)")
    blocks = n // 64
    nested = -(-blocks // _NESTED_BLOCK)
    return [
        (name, "U8", (n // 2, 1), n // 2),
        (name + "::absmax", "U8", (blocks,), blocks),
        (name + "::nested_absmax", "F32", (nested,), nested * 4),
        (name + "::nested_offset", "F32", (), 4),
    ]


def _nf4_payload(pool: _Pool, out_features: int, in_features: int, code_index: int):
    """Packed bytes, then the double-quantised absmax: every block's uint8 code
    points at the nested table's 1.0 entry, so absmax = nested_absmax = 1/sqrt(fan_in)."""
    import numpy as np

    n = out_features * in_features
    blocks = n // 64
    nested = -(-blocks // _NESTED_BLOCK)
    return [
        pool.take(n // 2),
        np.full(blocks, code_index, dtype=np.uint8),
        np.full(nested, 1.0 / math.sqrt(in_features), dtype=np.float32),
        np.zeros((), dtype=np.float32),
    ]


def _write_streaming(path: str, entries, payloads) -> None:
    """A safetensors file written tensor by tensor; never more than one in RAM.

    ``entries``: (name, dtype, shape, nbytes) in file order; ``payloads``
    yields one numpy array per entry, in the same order.
    """
    header = {}
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {
            "dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]
        }
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        written = 0
        for (name, _dtype, _shape, nbytes), payload in zip(entries, payloads):
            data = memoryview(payload).cast("B")
            if len(data) != nbytes:
                raise ValueError(f"{name}: payload is {len(data)} bytes, header says {nbytes}")
            handle.write(data)
            written += nbytes
    if written != offset:
        raise ValueError(f"{path}: wrote {written} data bytes, header says {offset}")
    os.replace(tmp, path)


def _nf4_code():
    """bitsandbytes' NF4 code table, as the sharder stores it in the extras."""
    import torch
    from bitsandbytes.functional import get_4bit_type

    return get_4bit_type("nf4", device="cpu").to(torch.float32)


def _nested_code():
    """The table double quantisation codes absmax with (``quantize_blockwise``)."""
    import torch
    from bitsandbytes.functional import create_dynamic_map

    return create_dynamic_map(signed=True).to(torch.float32)


def _large_matrix(vocab: int, hidden: int, generator):
    """A bf16 [vocab, hidden] N(0, 0.02) matrix built in row slices, so at most
    one fp32 slice exists at once (the K2 vocabulary is 163840 x 7168)."""
    import torch

    out = torch.empty(vocab, hidden, dtype=torch.bfloat16)
    for start in range(0, vocab, 8192):
        stop = min(vocab, start + 8192)
        out[start:stop] = torch.randn(stop - start, hidden, generator=generator) * 0.02
    return out


def write_store(out: str, shape_name: str, *, seed: int = 1234, log=print) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from soup_cli import __version__
    from soup_cli.utils.expert_stream import EXPERT_LAYOUT_PER_EXPERT, expert_shard_name
    from soup_cli.utils.layer_shard import (
        _SHARD_FORMAT_VERSION,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        NF4WeightSpec,
        ShardIndex,
        _atomic_write_index,
        extras_shard_path,
        large_shard_path,
        layer_shard_path,
    )

    shape = SHAPES[shape_name]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_dict(shape), handle, indent=2)
    pool = _Pool(seed)
    nested_code = _nested_code()
    one = int(nested_code.argmax())
    if float(nested_code[one]) != 1.0:
        raise ValueError("the nested code table has no exact 1.0 entry")
    hidden = shape["hidden_size"]
    vocab = shape["vocab_size"]
    n_layers = shape["num_hidden_layers"]
    experts = shape["n_routed_experts"]
    inter = shape["moe_intermediate_size"]

    def spec(out_features, in_features):
        return NF4WeightSpec(
            shape=(out_features, in_features), dtype="bfloat16", blocksize=64,
            quant_type="nf4", nested=True, nested_blocksize=_NESTED_BLOCK,
        )

    quant_specs = {}
    layer_keys = set()
    total = 0
    started = time.perf_counter()
    written_bytes = 0
    for layer in range(n_layers):
        entries, payloads = [], []
        for name, (o, i) in _linears(shape, layer).items():
            quant_specs[name] = spec(o, i)
            entries += _nf4_entries(name, o, i)
            payloads += _nf4_payload(pool, o, i, one)
            total += o * i
        for name, (dims, dtype, fill) in _dense(shape, layer).items():
            count = math.prod(dims)
            if dtype == "F32":
                payload = np.full(count, fill, dtype=np.float32)
            elif fill is None:
                payload = (torch.randn(count, generator=torch.Generator().manual_seed(layer))
                           * 0.02).to(torch.bfloat16).view(torch.int16).numpy()
            else:
                payload = torch.full((count,), fill, dtype=torch.bfloat16).view(torch.int16).numpy()
            entries.append((name, dtype, dims, payload.nbytes))
            payloads.append(payload)
            total += count
        layer_keys.update(entry[0] for entry in entries)
        _write_streaming(layer_shard_path(out, layer), entries, payloads)
        written_bytes += sum(entry[3] for entry in entries)
        if layer >= shape["first_k_dense_replace"]:
            expert_entries = []
            for expert in range(experts):
                expert_entries += _nf4_entries(f"{expert}.gate_up", 2 * inter, hidden)
                expert_entries += _nf4_entries(f"{expert}.down", hidden, inter)

            def expert_payloads():
                for _ in range(experts):
                    yield from _nf4_payload(pool, 2 * inter, hidden, one)
                    yield from _nf4_payload(pool, hidden, inter, one)

            _write_streaming(os.path.join(out, expert_shard_name(layer)), expert_entries,
                             expert_payloads())
            written_bytes += sum(entry[3] for entry in expert_entries)
            total += experts * 3 * inter * hidden
        log(f"layer {layer + 1}/{n_layers}: {written_bytes / 1e9:.1f} GB written, "
            f"{time.perf_counter() - started:.0f} s")

    generator = torch.Generator().manual_seed(seed)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        matrix = _large_matrix(vocab, hidden, generator)
        save_file({key: matrix}, large_shard_path(out, key))
        written_bytes += matrix.numel() * 2
        total += matrix.numel()
        del matrix
    extras = {
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        NF4_CODE_KEY: _nf4_code(),
        NF4_NESTED_CODE_KEY: nested_code,
    }
    save_file(extras, extras_shard_path(out))
    total += hidden

    moe_layers = tuple(range(shape["first_k_dense_replace"], n_layers))
    index = ShardIndex(
        n_layers=n_layers,
        layer_keys=tuple(sorted(layer_keys)),
        extra_keys=tuple(sorted(extras)),
        dtype="bfloat16",
        total_params=total,
        arch="deepseek_v3",
        soup_version=__version__,
        large_keys=("lm_head.weight", "model.embed_tokens.weight"),
        format_version=_SHARD_FORMAT_VERSION,
        source_fingerprint=f"synthetic-{shape_name}-seed{seed}",
        source_files=(),
        quant="nf4",
        double_quant=True,
        quant_device="cuda",
        quant_specs=quant_specs,
        expert_layout=EXPERT_LAYOUT_PER_EXPERT,
        expert_layers=moe_layers,
        expert_count=experts,
        expert_shapes={"gate_up": (2 * inter, hidden), "down": (hidden, inter)},
        expert_quant_specs={"gate_up": spec(2 * inter, hidden), "down": spec(hidden, inter)},
    )
    _atomic_write_index(index, out)
    summary = {
        "shape": shape_name,
        "params": total,
        "bytes_written": written_bytes,
        "seconds": round(time.perf_counter() - started, 1),
        "label": "random weights, shape only",
    }
    with open(os.path.join(out, "synth.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="k2")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if os.path.exists(os.path.join(args.out, "index.json")):
        print(f"refusing: {args.out} already holds a shard cache")
        return 2
    print(json.dumps(write_store(args.out, args.shape, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
