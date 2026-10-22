"""How far apart are the weights the two arms of the FP8-source loss check stream? CPU only.

Pre-run evidence for the thresholds of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md``.
Read-only: it opens safetensors files and writes only ``--json``. Run with
``CUDA_VISIBLE_DEVICES=-1`` and ``PYTHONPATH`` at the tree under test:

    python benchmarks/harness/fp8_loss_check_weights.py --source <bf16 checkpoint> \
        --cache-a <fused NF4 shard dir from the bf16 source> \
        --cache-b <fused NF4 shard dir from the FP8-block copy> \
        --cache-e <per_expert NF4 shard dir from the bf16 source> \
        --layers 1 13 26 --experts 0 21 42 63 --json <report.json>

For each decoder layer named, against the bf16 source weight ``w``:

- the 7 NF4 linears (4 attention, 3 shared expert): dequantised from A and from B with
  bitsandbytes on the CPU; reported ``||A - w|| / ||w||``, ``||B - w|| / ||w||`` and
  ``||A - B|| / ||w||``;
- the fused routed experts (bf16 in both caches), for the experts named: A must equal the
  source exactly; reported ``||A - B|| / ||A||``;
- the per_expert cache E of section 4.2 (routed experts NF4, the reference perturbation
  between sections 4.1 and 4.2): ``||deq(E) - w|| / ||w||`` for the same experts; and E's NF4
  linears must equal A's byte for byte (so 4.1 and 4.2 differed only in the routed experts);
- every other tensor of the layer (norms, router, correction bias), plus the extras, must be
  byte-identical in A and B, which shows that the cache A was written by older code stores
  what this code stores for everything not derived from an FP8 weight.

Then a per-token perturbation budget: squared Frobenius distance summed over the matrices one
token passes through in a MoE layer (attention, shared expert, and ``top_k`` routed experts at
the sampled experts' mean), for B against A and for E against A. Its square root ratio is a
crude scale of the A/B perturbation against the 4.1/4.2 one; it weights every matrix equally
per unit of squared error, which the record states as an assumption.

B in the record is the gate's CPU-quantised cache; ``soup train`` quantises on CUDA, which
agrees on the packed nibbles and absmax and differs only in the float32 nested statistic.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

NF4_LINEARS = (
    "self_attn.q_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)
FUSED = {"gate_up": "mlp.experts.gate_up_proj", "down": "mlp.experts.down_proj"}


def _free_gib() -> float:
    import psutil

    return psutil.virtual_memory().available / 2**30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--cache-e", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 13, 26])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 21, 42, 63])
    parser.add_argument("--json", required=True)
    parser.add_argument("--min-free-gib", type=float, default=6.0)
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        print("refusing: set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
        return 2
    if _free_gib() < args.min_free_gib:
        print(f"refusing: {_free_gib():.2f} GiB available < {args.min_free_gib} GiB")
        return 2

    import torch
    from bitsandbytes import functional as bnb_functional
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY, read_shard_index
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    if torch.cuda.is_available():
        print("refusing: CUDA is visible")
        return 2
    started = time.time()

    with open(os.path.join(args.source, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    top_k = int(config["num_experts_per_tok"])
    with open(os.path.join(args.source, "model.safetensors.index.json"), encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]

    def source(key: str):
        with safe_open(os.path.join(args.source, weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    caches = {"A": args.cache_a, "B": args.cache_b, "E": args.cache_e}
    index = {name: read_shard_index(path) for name, path in caches.items()}
    codes = {}
    extras_equal = {}
    for name, path in caches.items():
        with safe_open(os.path.join(path, "extras.safetensors"), framework="pt") as handle:
            codes[name] = {key: handle.get_tensor(key) for key in handle.keys()}
    for key in sorted(set(codes["A"]) | set(codes["B"])):
        left, right = codes["A"].get(key), codes["B"].get(key)
        extras_equal[key] = bool(
            left is not None and right is not None and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    for name in ("A", "B", "E"):
        table = codes[name]
        codes[name] = {k: table[k] for k in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY) if k in table}

    def dequant(handle, key: str, spec, cache: str):
        buffers = {
            name: handle.get_tensor(name)
            for name in handle.keys()
            if name == key or name.startswith(key + "::")
        }
        state = rebuild_quant_state(key, buffers, spec, codes[cache])
        return bnb_functional.dequantize_4bit(buffers[key], state).float()

    def rel(diff, ref) -> float:
        return float(diff.norm() / ref.norm())

    rows = []
    kept = {"compared": 0, "equal": 0, "unequal": []}
    e_nf4_equal_a = {"compared": 0, "equal": 0}
    budget = {"BA": 0.0, "EA": 0.0, "BA_nf4_linears": 0.0, "BA_routed": 0.0}
    for layer in args.layers:
        name = f"layer_{layer:03d}.safetensors"
        handle_a = safe_open(os.path.join(args.cache_a, name), framework="pt")
        handle_b = safe_open(os.path.join(args.cache_b, name), framework="pt")
        handle_e = safe_open(os.path.join(args.cache_e, name), framework="pt")
        keys_a, keys_b = set(handle_a.keys()), set(handle_b.keys())
        if keys_a != keys_b:
            raise SystemExit(f"layer {layer}: key sets differ: {sorted(keys_a ^ keys_b)[:5]}")
        prefix = f"model.layers.{layer}."
        for key in sorted(keys_a):
            if "::" in key or key in NF4_LINEARS or key.startswith("mlp.experts."):
                continue
            left, right = handle_a.get_tensor(key), handle_b.get_tensor(key)
            kept["compared"] += 1
            if left.dtype == right.dtype and torch.equal(left, right):
                kept["equal"] += 1
            else:
                kept["unequal"].append(f"{layer}:{key}")
        for key in NF4_LINEARS:
            weight = source(prefix + key).float()
            deq_a = dequant(handle_a, key, index["A"].quant_specs[key], "A")
            deq_b = dequant(handle_b, key, index["B"].quant_specs[key], "B")
            same_e = all(
                torch.equal(handle_a.get_tensor(name), handle_e.get_tensor(name))
                for name in handle_a.keys()
                if name == key or name.startswith(key + "::")
            )
            e_nf4_equal_a["compared"] += 1
            e_nf4_equal_a["equal"] += int(same_e)
            budget["BA"] += float((deq_a - deq_b).norm() ** 2)
            budget["BA_nf4_linears"] += float((deq_a - deq_b).norm() ** 2)
            rows.append({
                "layer": layer,
                "tensor": key,
                "numel": weight.numel(),
                "a_vs_w": rel(deq_a - weight, weight),
                "b_vs_w": rel(deq_b - weight, weight),
                "a_vs_b": rel(deq_a - deq_b, weight),
            })
            del weight, deq_a, deq_b
        experts_file = os.path.join(args.cache_e, f"experts_{layer:03d}.safetensors")
        handle_x = safe_open(experts_file, framework="pt")
        routed_ba = {part: [] for part in FUSED}
        routed_ea = {part: [] for part in FUSED}
        for expert in args.experts:
            base = f"{prefix}mlp.experts.{expert}."
            originals = {
                "gate_up": torch.cat(
                    [source(base + "gate_proj.weight"), source(base + "up_proj.weight")], dim=0
                ),
                "down": source(base + "down_proj.weight"),
            }
            for part, fused_key in FUSED.items():
                slab_a = handle_a.get_slice(fused_key)[expert]
                slab_b = handle_b.get_slice(fused_key)[expert]
                if not torch.equal(slab_a, originals[part]):
                    raise SystemExit(f"layer {layer} expert {expert} {part}: A != source")
                ref = slab_a.float()
                deq_e = dequant(
                    handle_x, f"{expert}.{part}", index["E"].expert_quant_specs[part], "E"
                )
                ba = float((ref - slab_b.float()).norm() ** 2)
                ea = float((ref - deq_e).norm() ** 2)
                routed_ba[part].append(ba)
                routed_ea[part].append(ea)
                rows.append({
                    "layer": layer,
                    "tensor": f"routed.{expert}.{part}",
                    "numel": ref.numel(),
                    "a_vs_b": (ba ** 0.5) / float(ref.norm()),
                    "e_vs_w": (ea ** 0.5) / float(ref.norm()),
                })
                del slab_a, slab_b, ref, deq_e
        # top_k routed experts per token, at the sampled experts' mean squared distance.
        for part in FUSED:
            budget["BA"] += top_k * statistics.mean(routed_ba[part])
            budget["BA_routed"] += top_k * statistics.mean(routed_ba[part])
            budget["EA"] += top_k * statistics.mean(routed_ea[part])
        del handle_a, handle_b, handle_e, handle_x

    def classes(metric: str) -> dict:
        grouped: dict = {}
        for row in rows:
            if metric not in row:
                continue
            tensor = row["tensor"]
            label = tensor.split(".", 2)[2] if tensor.startswith("routed.") else tensor
            grouped.setdefault(label, []).append(row[metric])
        return {
            label: {"n": len(values), "mean": statistics.mean(values), "max": max(values)}
            for label, values in sorted(grouped.items())
        }

    report = {
        "layers": args.layers,
        "experts": args.experts,
        "top_k": top_k,
        "kept_tensors_a_equal_b": kept,
        "extras_a_equal_b": extras_equal,
        "nf4_linears_e_equal_a": e_nf4_equal_a,
        "a_vs_w": classes("a_vs_w"),
        "b_vs_w": classes("b_vs_w"),
        "a_vs_b": classes("a_vs_b"),
        "e_vs_w": classes("e_vs_w"),
        "per_token_sq_distance": budget,
        "amplitude_ratio_ba_over_ea": (budget["BA"] / budget["EA"]) ** 0.5,
        "rows": rows,
        "seconds": round(time.time() - started, 1),
        "min_free_gib_seen_at_end": round(_free_gib(), 2),
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
