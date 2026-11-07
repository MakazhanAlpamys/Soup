"""Convert a bf16 DeepSeek-V3-family checkpoint to the FP8 block format it is published in.

DeepSeek-V3 ships its decoder linears as ``float8_e4m3fn`` with a float32
``<name>.weight_scale_inv`` per 128 x 128 block (``[ceil(rows/128), ceil(cols/128)]``,
the last block row and column cropped), and ``config.json`` names the format in
``quantization_config``. Embeddings, the head, the norms, the router and the router's
correction bias stay as they are. This writes that format from a bf16 checkpoint, so the
FP8 path of Soup's layer-stream sharder can be exercised on real weights
(``benchmarks/gate-fp8-shard.md``, G3) without downloading a 671B checkpoint.

The arithmetic is transformers' own ``Fp8Quantize`` (per-block ``448 / amax`` scale,
clamp, cast; an all-zero block gets scale 1), extended to ragged edges the way DeepSeek's
format has them: a block cut by the matrix edge takes its amax over the elements that
exist. One tensor at a time, one source file at a time: peak memory is one source file's
output, never the model.

    python benchmarks/harness/fp8_block_quantize.py --src <bf16 dir> --dst <new dir> \
        --report <stats.json>

``--dst`` must not exist. The report holds, per FP8 tensor, the relative Frobenius error
``||deq - orig|| / ||orig||`` and ``max|deq - orig| / max|orig|`` of the dequantised weight
against its bf16 original, aggregated per tensor class.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time

#: Every decoder ``nn.Linear`` weight of a DeepSeek-V3 layer: the tensors the published
#: checkpoints store as FP8. Router (``mlp.gate.weight``), norms, the correction bias and
#: buffers are not linears and stay as they are.
_FP8_LINEAR_RE = re.compile(
    r"^model\.layers\.\d+\."
    r"(?P<cls>self_attn\.(?:q_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)"
    r"|mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
#: What DeepSeek-V3's own config.json says (Kimi K2's is the same).
DEEPSEEK_QUANTIZATION_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
_COPIED_SUFFIXES = (".json", ".model", ".py", ".txt", ".tiktoken")
_SKIPPED = ("config.json", "model.safetensors.index.json")


def tensor_class(key: str) -> str | None:
    """The report class of an FP8 target (``routed.gate_proj``, ...), ``None`` otherwise."""
    match = _FP8_LINEAR_RE.match(key)
    if match is None:
        return None
    cls = match.group("cls")
    projection = cls.rsplit(".", 1)[1]
    if cls.startswith("mlp.experts."):
        return "routed." + projection
    if cls.startswith("mlp.shared_experts."):
        return "shared." + projection
    if cls.startswith("mlp."):
        return "dense." + projection
    return projection


def quantize_fp8_block(weight, block):
    """``(float8_e4m3fn weight, float32 weight_scale_inv)`` for one 2-D weight."""
    import torch

    fp8 = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8).max
    fp8_min = torch.finfo(fp8).min
    rows, cols = weight.shape
    # A block past the matrix edge is that edge: the same grid, without padding to its size.
    block_rows, block_cols = min(block[0], rows), min(block[1], cols)
    grid_rows, grid_cols = -(-rows // block_rows), -(-cols // block_cols)
    value = weight.to(torch.float32)
    padded = torch.zeros(grid_rows * block_rows, grid_cols * block_cols, dtype=torch.float32)
    padded[:rows, :cols] = value
    tiles = padded.reshape(grid_rows, block_rows, grid_cols, block_cols)
    max_abs = tiles.abs().amax(dim=(-3, -1))
    safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = fp8_max / safe_max_abs
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
    inv_scales = (1.0 / scales).to(torch.float32)
    scaled = tiles * scales.unsqueeze(-1).unsqueeze(-3)
    quantized = torch.clamp(scaled, min=fp8_min, max=fp8_max).to(fp8)
    quantized = quantized.reshape(grid_rows * block_rows, grid_cols * block_cols)
    return quantized[:rows, :cols].contiguous(), inv_scales.contiguous()


def reference_dequantize(weight_fp8, scale_inv, block):
    """One fp32 multiply per element, block by block: the independent reference."""
    import torch

    rows, cols = weight_fp8.shape
    block_rows, block_cols = block
    out = torch.empty(rows, cols, dtype=torch.float32)
    for i in range(-(-rows // block_rows)):
        for j in range(-(-cols // block_cols)):
            r0, c0 = i * block_rows, j * block_cols
            r1, c1 = min(r0 + block_rows, rows), min(c0 + block_cols, cols)
            out[r0:r1, c0:c1] = weight_fp8[r0:r1, c0:c1].to(torch.float32) * scale_inv[i, j]
    return out


def _dequantize_fast(weight_fp8, scale_inv, block):
    """The same products as :func:`reference_dequantize`, without the Python loop."""
    import torch

    rows, cols = weight_fp8.shape
    expanded = scale_inv.repeat_interleave(min(block[0], rows), dim=0)[:rows]
    expanded = expanded.repeat_interleave(min(block[1], cols), dim=1)[:, :cols]
    return weight_fp8.to(torch.float32) * expanded


def _error(original, dequantized) -> tuple[float, float]:
    import torch

    orig = original.to(torch.float32)
    diff = dequantized - orig
    norm = float(torch.linalg.vector_norm(orig))
    peak = float(orig.abs().max())
    rel_frobenius = float(torch.linalg.vector_norm(diff)) / norm if norm else 0.0
    rel_max = float(diff.abs().max()) / peak if peak else 0.0
    return rel_frobenius, rel_max


def _source_files(src: str) -> tuple[list[str], dict | None]:
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        return sorted(set(index["weight_map"].values())), index
    if os.path.isfile(os.path.join(src, "model.safetensors")):
        return ["model.safetensors"], None
    raise FileNotFoundError(f"no safetensors checkpoint in {src}")


def convert_checkpoint(
    src: str,
    dst: str,
    *,
    block: tuple[int, int] = (128, 128),
    quantization_config: dict | None = None,
    log=print,
) -> dict:
    """Write the FP8 block copy of ``src`` to the new folder ``dst``; return the report."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if os.path.exists(dst):
        raise FileExistsError(f"refusing: {dst} already exists")
    files, index = _source_files(src)
    os.makedirs(dst)
    started = time.perf_counter()
    weight_map: dict[str, str] = {}
    total_size = 0
    per_tensor: dict[str, dict] = {}
    kept: dict[str, int] = {}
    for position, name in enumerate(files, start=1):
        out = {}
        with safe_open(os.path.join(src, name), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                cls = tensor_class(key) if tensor.ndim == 2 else None
                if cls is None:
                    out[key] = tensor.clone()
                    kept[str(tensor.dtype)] = kept.get(str(tensor.dtype), 0) + 1
                else:
                    weight, scale = quantize_fp8_block(tensor, block)
                    rel_frobenius, rel_max = _error(
                        tensor, _dequantize_fast(weight, scale, block)
                    )
                    per_tensor[key] = {
                        "class": cls,
                        "shape": list(tensor.shape),
                        "scale_shape": list(scale.shape),
                        "rel_frobenius": rel_frobenius,
                        "rel_max": rel_max,
                    }
                    out[key] = weight
                    out[key + "_scale_inv"] = scale
                del tensor
        for key, value in out.items():
            weight_map[key] = name
            total_size += value.numel() * value.element_size()
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
        del out
        log(f"[{position}/{len(files)}] {name} ({time.perf_counter() - started:.0f} s)")
    if index is not None:
        payload = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        index_path = os.path.join(dst, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    quant = dict(quantization_config or DEEPSEEK_QUANTIZATION_CONFIG)
    quant["weight_block_size"] = [int(block[0]), int(block[1])]
    config["quantization_config"] = quant
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    for entry in sorted(os.listdir(src)):
        path = os.path.join(src, entry)
        if entry in _SKIPPED or not entry.endswith(_COPIED_SUFFIXES):
            continue
        if os.path.isfile(path) and not os.path.islink(path):
            shutil.copyfile(path, os.path.join(dst, entry))
    elapsed = time.perf_counter() - started
    classes: dict[str, dict] = {}
    for stats in per_tensor.values():
        entry = classes.setdefault(stats["class"], {"n": 0, "rel_frobenius": [], "rel_max": []})
        entry["n"] += 1
        entry["rel_frobenius"].append(stats["rel_frobenius"])
        entry["rel_max"].append(stats["rel_max"])
    summary = {
        cls: {
            "tensors": entry["n"],
            "rel_frobenius_max": max(entry["rel_frobenius"]),
            "rel_frobenius_mean": math.fsum(entry["rel_frobenius"]) / entry["n"],
            "rel_max_max": max(entry["rel_max"]),
            "rel_max_mean": math.fsum(entry["rel_max"]) / entry["n"],
        }
        for cls, entry in sorted(classes.items())
    }
    copy_bytes = sum(
        os.path.getsize(os.path.join(dst, name)) for name in os.listdir(dst)
        if name.endswith(".safetensors")
    )
    source_bytes = sum(os.path.getsize(os.path.join(src, name)) for name in files)
    return {
        "src": src,
        "dst": dst,
        "block": list(block),
        "quantization_config": quant,
        "torch": torch.__version__,
        "wall_seconds": elapsed,
        "source_safetensors_bytes": source_bytes,
        "fp8_safetensors_bytes": copy_bytes,
        "fp8_tensors": len(per_tensor),
        "kept_tensors_by_dtype": kept,
        "classes": summary,
        "per_tensor": per_tensor,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--block", type=int, nargs=2, default=(128, 128))
    args = parser.parse_args(argv)
    if os.path.exists(args.dst):
        print(f"refusing: {args.dst} already exists")
        return 2
    report = convert_checkpoint(args.src, args.dst, block=tuple(args.block))
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in report if key != "per_tensor"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
