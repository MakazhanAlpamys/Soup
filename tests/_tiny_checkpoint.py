"""A tiny Llama-shaped checkpoint on disk: a real ``config.json`` and safetensors
headers over zero-filled data, for the VRAM estimate's tests (#2049). No torch."""

from __future__ import annotations

import json
from pathlib import Path

#: grouped-query attention and untied embeddings, so every term has its own size.
TINY_LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 160,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "tie_word_embeddings": False,
}


def tensors_for(config: dict) -> dict[str, list[int]]:
    hidden, inter = config["hidden_size"], config["intermediate_size"]
    head_dim = hidden // config["num_attention_heads"]
    kv = config["num_key_value_heads"] * head_dim
    out = {"model.embed_tokens.weight": [config["vocab_size"], hidden]}
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        out[prefix + "self_attn.q_proj.weight"] = [hidden, hidden]
        out[prefix + "self_attn.k_proj.weight"] = [kv, hidden]
        out[prefix + "self_attn.v_proj.weight"] = [kv, hidden]
        out[prefix + "self_attn.o_proj.weight"] = [hidden, hidden]
        out[prefix + "mlp.gate_proj.weight"] = [inter, hidden]
        out[prefix + "mlp.up_proj.weight"] = [inter, hidden]
        out[prefix + "mlp.down_proj.weight"] = [hidden, inter]
        out[prefix + "input_layernorm.weight"] = [hidden]
        out[prefix + "post_attention_layernorm.weight"] = [hidden]
    out["model.norm.weight"] = [hidden]
    if not config.get("tie_word_embeddings"):
        out["lm_head.weight"] = [config["vocab_size"], hidden]
    return out


def write_safetensors(path: Path, tensors: dict[str, list[int]]) -> int:
    """A safetensors file with a real header and zero data; returns the parameter count."""
    header, offset, count = {}, 0, 0
    for name, shape in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + 2 * numel],
        }
        offset += 2 * numel
        count += numel
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))
    return count


def write_tiny_model(directory: Path, **config_overrides) -> int:
    """``config.json`` and ``model.safetensors`` in ``directory``; returns the parameter count."""
    config = {**TINY_LLAMA, **config_overrides}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return write_safetensors(directory / "model.safetensors", tensors_for(config))
