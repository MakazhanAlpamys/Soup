"""Gate harness: StreamedExperts vs the resident OLMoE experts block, bit for bit.

Rule and verdict: ``benchmarks/gate-expert-streaming.md`` (the rule was
committed before this ran). Reads ONE decoder layer's experts from a local
OLMoE checkpoint at a time, so the whole 13.8 GB model is never resident.

    python benchmarks/harness/expert_stream_gate.py \
        --model-dir <HF snapshot of allenai/OLMoE-1B-7B-0924> \
        --scratch <empty folder for the experts files> --out result.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time

CELLS = ("cpu-fp32", "cuda-bf16", "cuda-bf16-nf4")


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    return parser.parse_args(argv)


def _layer_tensors(model_dir, layer):
    """{short key: tensor} for one decoder layer, read file by file."""
    from safetensors import safe_open

    prefix = f"model.layers.{layer}."
    found = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(prefix) and (".mlp." in key or "post_attention" in key):
                    found[key[len(prefix):]] = handle.get_tensor(key).clone()
    return found


def _experts_module(config, tensors, n_experts, device, dtype):
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    config._experts_implementation = "eager"
    module = OlmoeExperts(config)
    with torch.no_grad():
        for expert in range(n_experts):
            gate = tensors[f"mlp.experts.{expert}.gate_proj.weight"]
            up = tensors[f"mlp.experts.{expert}.up_proj.weight"]
            module.gate_up_proj[expert].copy_(torch.cat((gate, up)))
            module.down_proj[expert].copy_(tensors[f"mlp.experts.{expert}.down_proj.weight"])
    module.requires_grad_(False)
    return module.to(device=device, dtype=dtype)


def _nf4_roundtrip(module):
    """Replace every expert matrix by its NF4 dequantisation (the NF4 control)."""
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    with torch.no_grad():
        for param in (module.gate_up_proj, module.down_proj):
            for expert in range(param.shape[0]):
                packed, state = quantize_4bit(
                    param[expert].contiguous(), blocksize=64, compress_statistics=True,
                    quant_type="nf4",
                )
                param[expert].copy_(dequantize_4bit(packed, quant_state=state))


def _run_cell(cell, layer, config, tensors, args):
    import torch

    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    device = "cpu" if cell.startswith("cpu") else "cuda"
    dtype = torch.float32 if cell.endswith("fp32") else torch.bfloat16
    quantise = cell.endswith("nf4")
    n_experts = int(config.num_experts)
    top_k = int(config.num_experts_per_tok)
    module = _experts_module(config, tensors, n_experts, device, dtype)

    path = os.path.join(args.scratch, f"experts_{layer:03d}_{cell}.safetensors")
    pairs = (
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach())
        for e in range(n_experts)
    )
    info = write_expert_shard(path, pairs, quantise=quantise, double_quant=True, device=device)
    codes = {}
    if quantise:
        _nf4_roundtrip(module)
        codes = {
            NF4_CODE_KEY: info.codes[0].to(device),
            NF4_NESTED_CODE_KEY: info.codes[1].to(device),
        }
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype="float32" if dtype == torch.float32 else "bfloat16",
    )
    streamed = streamed_experts_class()(
        reader,
        ExpertFetcher(device=device, dtype=reader.dtype, codes=codes),
        module.act_fn,
        layer,
    )

    hidden = config.hidden_size
    generator = torch.Generator().manual_seed(1234)
    norm = tensors["post_attention_layernorm.weight"].float()
    states = (torch.randn(args.tokens, hidden, generator=generator) * norm).to(device, dtype)
    grad_out = torch.randn(args.tokens, hidden, generator=torch.Generator().manual_seed(4321))
    grad_out = grad_out.to(device, dtype)
    router = tensors["mlp.gate.weight"].to(device, dtype)
    with torch.no_grad():
        probs = torch.softmax(torch.nn.functional.linear(states, router).float(), dim=-1)
        weights, index = torch.topk(probs, top_k, dim=-1)
        weights = weights.to(dtype)

    results = {}
    outputs = {}
    for arm, net in (("resident", module), ("streamed", streamed)):
        x = states.clone().requires_grad_(True)
        w = weights.clone().requires_grad_(True)
        started = time.perf_counter()
        out = net(x, index, w)
        out.backward(grad_out)
        if device == "cuda":
            torch.cuda.synchronize()
        outputs[arm] = (out.detach(), x.grad, w.grad)
        results[f"{arm}_seconds"] = round(time.perf_counter() - started, 4)
    hit = int(torch.unique(index).numel())
    for name, slot in (("output", 0), ("grad_input", 1), ("grad_routing", 2)):
        want, got = outputs["resident"][slot], outputs["streamed"][slot]
        equal = bool(torch.equal(got, want))
        results[name] = {
            "equal": equal,
            "max_abs_diff": 0.0 if equal else float((got.float() - want.float()).abs().max()),
        }
    results["experts_hit"] = hit
    results["fetches"] = reader.fetches
    results["fetches_ok"] = reader.fetches == 2 * hit
    reader.close()
    os.remove(path)
    return results


def _box():
    import torch

    stamp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if torch.cuda.is_available():
        stamp["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        stamp["gpu_free_mib"] = free // 2**20
        stamp["gpu_total_mib"] = total // 2**20
    for name in ("transformers", "bitsandbytes", "safetensors"):
        try:
            stamp[name] = __import__(name).__version__
        except ImportError:
            stamp[name] = None
    return stamp


def main(argv=None):
    import torch
    from transformers import OlmoeConfig

    args = _args(argv)
    os.makedirs(args.scratch, exist_ok=True)
    if any(cell.startswith("cuda") for cell in args.cells) and not torch.cuda.is_available():
        print("SKIP: a CUDA cell was asked for and CUDA is not available")
        return 0
    config = OlmoeConfig.from_pretrained(args.model_dir)
    report = {"box": _box(), "args": vars(args), "layers": {}}
    passed = True
    for layer in args.layers:
        tensors = _layer_tensors(args.model_dir, layer)
        report["layers"][str(layer)] = {}
        for cell in args.cells:
            cell_result = _run_cell(cell, layer, config, tensors, args)
            report["layers"][str(layer)][cell] = cell_result
            passed &= all(
                cell_result[name]["equal"] for name in ("output", "grad_input", "grad_routing")
            ) and cell_result["fetches_ok"]
            print(layer, cell, json.dumps(cell_result))
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        del tensors
    report["verdict"] = "PASS" if passed else "FAIL"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(report["verdict"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
