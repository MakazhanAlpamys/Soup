#!/usr/bin/env python3
"""Write a synthetic Llama- or Qwen2-shaped HF checkpoint for TIMING and MEMORY measurements.

Layer streaming's disk tier can only be measured cold on a store that does not
fit the page cache, and a real 70B checkpoint needs a Hub token plus a 140 GB
download. A synthetic checkpoint of the same SHAPE gives the same bytes per
layer, the same number of layers and the same GEMM shapes, so every timing
number is real; only the values are random. Nothing here is evidence about
correctness or quality: the bit-exactness gates run on real models.

The same argument holds for VRAM: peak memory depends on shapes, not values, so
a shaped checkpoint reproduces an out-of-memory report without the download.
``--shape qwen2.5-14b --arch qwen2`` is the shape behind #901 (Qwen2.5-14B on
an 8 GB card): 48 layers, hidden 5120, a 152064-row untied head, and the
q/k/v biases Qwen2 carries.

Weights are N(0, --std) so activations stay finite through 80 layers (N(0, 1)
weights blow up to inf and NaN, which still time correctly but read badly).

Typical invocation::

    python benchmarks/harness/synth_checkpoint.py --shape llama-70b \
        --out C:/models/synth-llama-70b

The ``llama-*`` shapes are the published Llama-3.1 configs (vocab 128256,
untied head); ``qwen2.5-14b`` is the published Qwen2.5-14B config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

SHAPES: Dict[str, Dict[str, int]] = {
    "llama-8b": {
        "hidden": 4096,
        "intermediate": 14336,
        "layers": 32,
        "heads": 32,
        "kv_heads": 8,
        "vocab": 128256,
    },
    "llama-70b": {
        "hidden": 8192,
        "intermediate": 28672,
        "layers": 80,
        "heads": 64,
        "kv_heads": 8,
        "vocab": 128256,
    },
    # Qwen/Qwen2.5-14B(-Instruct): the #901 shape. Pair it with --arch qwen2.
    "qwen2.5-14b": {
        "hidden": 5120,
        "intermediate": 13824,
        "layers": 48,
        "heads": 40,
        "kv_heads": 8,
        "vocab": 152064,
    },
}

ARCHS = ("llama", "qwen2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="checkpoint directory to create")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="llama-70b")
    parser.add_argument(
        "--arch",
        choices=ARCHS,
        default="llama",
        help="architecture to write: llama (no attention biases) or qwen2 (q/k/v biases)",
    )
    parser.add_argument("--layers", type=int, default=None, help="override the layer count")
    parser.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="override the vocabulary (a smaller head keeps a 70B-shaped decoder inside 8 GB)",
    )
    parser.add_argument("--std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--layers-per-file", type=int, default=1, help="decoder layers per safetensors file"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing directory")
    return parser.parse_args()


def _transformers_version() -> str:
    """A PARSEABLE version string. ``"synthetic"`` here made ``AutoTokenizer``
    (not ``AutoConfig``, which the streaming harness uses) die with
    ``InvalidVersion`` the moment a tokenizer was dropped beside the checkpoint
    for an end-to-end ``soup train``; ``_soup_synthetic`` is the marker."""
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # the generator needs only torch
        return "4.45.0"


def config_dict(shape: Dict[str, int], arch: str = "llama") -> Dict[str, Any]:
    common = {
        "hidden_size": shape["hidden"],
        "intermediate_size": shape["intermediate"],
        "num_hidden_layers": shape["layers"],
        "num_attention_heads": shape["heads"],
        "num_key_value_heads": shape["kv_heads"],
        "vocab_size": shape["vocab"],
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": False,
        "transformers_version": _transformers_version(),
        "_soup_synthetic": True,
    }
    if arch == "qwen2":
        # The published Qwen2.5-14B config values; the token ids are Qwen2.5's.
        return {
            **common,
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": shape["layers"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
        }
    if arch != "llama":
        raise ValueError(f"unsupported arch {arch!r}; choose one of {ARCHS}")
    return {
        **common,
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "head_dim": shape["hidden"] // shape["heads"],
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    }


def layer_tensors(shape: Dict[str, int], index: int, arch: str = "llama") -> Dict[str, tuple]:
    hidden = shape["hidden"]
    inter = shape["intermediate"]
    head_dim = hidden // shape["heads"]
    q_out = shape["heads"] * head_dim
    kv_out = shape["kv_heads"] * head_dim
    prefix = f"model.layers.{index}."
    tensors = {
        prefix + "self_attn.q_proj.weight": (q_out, hidden),
        prefix + "self_attn.k_proj.weight": (kv_out, hidden),
        prefix + "self_attn.v_proj.weight": (kv_out, hidden),
        prefix + "self_attn.o_proj.weight": (hidden, q_out),
        prefix + "mlp.gate_proj.weight": (inter, hidden),
        prefix + "mlp.up_proj.weight": (inter, hidden),
        prefix + "mlp.down_proj.weight": (hidden, inter),
        prefix + "input_layernorm.weight": (hidden,),
        prefix + "post_attention_layernorm.weight": (hidden,),
    }
    if arch == "qwen2":
        # Qwen2 carries biases on the q/k/v projections and nowhere else.
        tensors[prefix + "self_attn.q_proj.bias"] = (q_out,)
        tensors[prefix + "self_attn.k_proj.bias"] = (kv_out,)
        tensors[prefix + "self_attn.v_proj.bias"] = (kv_out,)
    return tensors


def main() -> int:
    args = parse_args()
    import torch
    from safetensors.torch import save_file

    shape = dict(SHAPES[args.shape])
    if args.layers is not None:
        shape["layers"] = int(args.layers)
    if args.vocab is not None:
        shape["vocab"] = int(args.vocab)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: {out} exists and is not empty; pass --force to overwrite")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def draw(dims: tuple, std: float) -> torch.Tensor:
        tensor = torch.randn(dims, generator=generator, device=device, dtype=torch.float32)
        return (tensor * std).to(torch.bfloat16).cpu().contiguous()

    def ones(dims: tuple) -> torch.Tensor:
        return torch.ones(dims, dtype=torch.bfloat16)

    weight_map: Dict[str, str] = {}
    total_bytes = 0
    started = time.perf_counter()

    extras_file = "model-extras.safetensors"
    extras = {
        "model.embed_tokens.weight": draw((shape["vocab"], shape["hidden"]), args.std),
        "model.norm.weight": ones((shape["hidden"],)),
        "lm_head.weight": draw((shape["vocab"], shape["hidden"]), args.std),
    }
    save_file(extras, str(out / extras_file), metadata={"format": "pt"})
    for name, tensor in extras.items():
        weight_map[name] = extras_file
        total_bytes += tensor.numel() * tensor.element_size()
    del extras

    per_file = max(1, int(args.layers_per_file))
    for first in range(0, shape["layers"], per_file):
        last = min(shape["layers"], first + per_file)
        file_name = f"model-layers-{first:03d}-{last - 1:03d}.safetensors"
        blob: Dict[str, torch.Tensor] = {}
        for index in range(first, last):
            for name, dims in layer_tensors(shape, index, args.arch).items():
                is_norm = name.endswith("layernorm.weight")
                blob[name] = ones(dims) if is_norm else draw(dims, args.std)
        save_file(blob, str(out / file_name), metadata={"format": "pt"})
        for name, tensor in blob.items():
            weight_map[name] = file_name
            total_bytes += tensor.numel() * tensor.element_size()
        del blob
        elapsed = time.perf_counter() - started
        print(
            f"layers {first:3d}-{last - 1:3d} written  "
            f"{total_bytes / 1e9:7.2f} GB  {elapsed:6.1f} s"
        )

    (out / "config.json").write_text(
        json.dumps(config_dict(shape, args.arch), indent=2), encoding="utf-8"
    )
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=1),
        encoding="utf-8",
    )
    params = total_bytes // 2
    print(
        f"done          {out}  {params / 1e9:.2f}B params  {total_bytes / 1e9:.2f} GB bf16  "
        f"{time.perf_counter() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
