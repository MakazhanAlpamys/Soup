"""Two untimed full-model updates for native auto versus residual QKVO.

Uses the selected default attention dispatch; never substitutes SDPA math.
Fresh processes bind to an existing census, preserve native K and refuse a
changed projection plan. Hashes bind every logit, loss, gradient and update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from rich.console import Console


def tensor_record(tensor: Any) -> dict:
    import torch

    if tensor is None:
        raise RuntimeError("missing tensor or adapter gradient")
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError("nonfinite tensor in exactness gate")
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return dict(
        shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(value).hexdigest(),
    )


def check_steps(reference: list, candidate: list) -> None:
    if len(reference) != 2 or len(candidate) != 2:
        raise RuntimeError("exactness requires two complete updates")
    for step in [*reference, *candidate]:
        if (
            set(step) != {"logits", "loss", "gradients", "parameters"}
            or not step["gradients"]
            or set(step["gradients"]) != set(step["parameters"])
        ):
            raise RuntimeError("incomplete exactness step or adapter roster")
    if reference != candidate:
        raise RuntimeError("full-model bit-exact gate failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("auto", "qkvo"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the previous exactness attempt")
    census = json.loads(args.census.read_text())
    cap = 8 * 2**30
    if census["cap_bytes"] != cap or not census["cap_binding"]["verified"]:
        raise ValueError("verified 8 GiB census required")
    if census["shape"] not in ([1, 2048], [1, 4096]):
        raise ValueError("unexpected census shape")
    seq = census["shape"][1]
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if "torch" in sys.modules:
        raise RuntimeError("GPU selection must precede torch import")
    from stream_projection_census import GPU_FIELDS, check_cap_binding, check_inventory

    inventory = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    admission = check_inventory(
        inventory.stdout, inventory.stderr, inventory.returncode, args.gpu, cap,
    )
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import LoraConfig, TaskType
    from stream_projection_retention import install_subset, observe_auto_plan, projection_plan

    import soup_cli
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, streamed_decoder_layers

    root = os.path.normcase(os.path.realpath(Path(__file__).parents[2] / "src"))
    if os.path.commonpath([root, os.path.normcase(os.path.realpath(soup_cli.__file__))]) != root:
        raise RuntimeError("soup_cli came from another checkout")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap / total_memory)
    config = json.loads((args.weights / "config.json").read_text())
    if config["model_type"] != "mistral" or config["num_hidden_layers"] != 32:
        raise ValueError("expected the existing 32-layer Mistral model")
    index_hash = hashlib.sha256((args.shards / "index.json").read_bytes()).hexdigest()
    if index_hash != census["shard_index_sha256"]:
        raise RuntimeError("shard index changed after census")
    model, runtime = build_streamed_model(
        model_id=str(args.weights), shard_dir=str(args.shards),
        index=read_shard_index(str(args.shards)),
        lora_config=LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        ), device="cuda", dtype="bfloat16", buffers=2, pin=True, seed=42,
        quant="nf4", double_quant=True, tier="ram",
    )
    state = dict(status="running", arm=args.arm, shape=[1, seq], profile=args.profile,
                 steps=[], admission=admission, optimizer_updates=0,
                 census_sha256=hashlib.sha256(args.census.read_bytes()).hexdigest(),
                 shard_index_sha256=index_hash, cap_bytes=cap)
    controller = None
    console = Console()
    try:
        if not runtime.pinned or runtime.tier != "ram":
            raise RuntimeError("planned pinned RAM source is unavailable")
        model.train()
        generator = torch.Generator(device="cuda").manual_seed(7)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and "lora_B" in name:
                    param.copy_(torch.randn(param.shape, generator=generator,
                                device=param.device, dtype=param.dtype) * 0.01)
        messages = []
        native = observe_auto_plan(model, rows=1, seq_len=seq,
                                   vocab_size=config["vocab_size"], override_bytes=cap,
                                   notify=messages.append)
        state.update(native=native, messages=messages,
                     cap_binding=check_cap_binding(native, messages, cap))
        layers = streamed_decoder_layers(model)
        plan = projection_plan(layers, native, batch=1, seq=seq, element_bytes=2)
        for key in ("k", "retained_indices"):
            if native.get(key) != census["native"][key]:
                raise RuntimeError("native K changed after census")
        if plan != census["projections"]:
            raise RuntimeError("native residual projection plan changed after census")
        state["plan"] = plan
        retained = [(layer, layer.forward) for layer in layers if not layer.recompute]
        if args.arm == "qkvo":
            controller = install_subset(model, plan["selected_indices"])
            if controller is None:
                raise RuntimeError("no QKVO layers fit")
        if any(layer.forward != forward for layer, forward in retained):
            raise RuntimeError("native retained layer was overridden")
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if len(parameters) != 256 or any("lora_" not in name for name in parameters):
            raise RuntimeError("unexpected trainable parameter roster")
        optimizer = PagedAdamW8bit(list(parameters.values()), lr=2e-4)
        torch.manual_seed(1234)
        ids = torch.randint(0, config["vocab_size"], (1, seq), device="cuda")
        labels = ids.clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        with profile if args.profile else nullcontext():
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=ids, labels=labels)
                cache = 0 if controller is None else controller.cache_bytes()
                expected = plan["raw_bytes"] if controller is not None else 0
                count = 0 if controller is None else len(controller.cached_tensors())
                expected_count = 4 * len(plan["selected_indices"]) if controller else 0
                if cache != expected or count != expected_count:
                    raise RuntimeError("actual QKVO storage does not match predicted bytes/count")
                step = dict(logits=tensor_record(output.logits), loss=tensor_record(output.loss))
                output.loss.backward()
                step["gradients"] = {name: tensor_record(p.grad) for name, p in parameters.items()}
                if any(not p.grad.count_nonzero().item() for p in parameters.values()):
                    raise RuntimeError("zero adapter gradient despite nonzero B")
                optimizer.step()
                step["parameters"] = {name: tensor_record(p) for name, p in parameters.items()}
                state["steps"].append(step)
                state["optimizer_updates"] += 1
                state["cache_bytes"] = cache
                state["cache_tensors"] = count
                del output
                torch.cuda.synchronize()
        state.update(peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved(),
                     attention_implementation=model.config._attn_implementation)
        if state["peak_allocated"] > cap or state["peak_reserved"] > cap:
            raise RuntimeError("combined memory exceeds the shared cap")
        if args.profile:
            state["attention_ops"] = {event.key: event.count for event in profile.key_averages()
                                      if "scaled_dot_product" in event.key}
            if not state["attention_ops"]:
                raise RuntimeError("actual attention dispatch was not observed")
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if reference["status"] != "ok" or reference["shape"] != state["shape"]:
                raise RuntimeError("invalid exactness reference")
            check_steps(reference["steps"], state["steps"])
            state["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        state["status"] = "ok"
        console.print({key: state[key] for key in (
            "status", "arm", "shape", "optimizer_updates", "cache_bytes", "peak_reserved",
        )})
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if controller is not None:
                controller.remove()
            runtime.close()
        finally:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
