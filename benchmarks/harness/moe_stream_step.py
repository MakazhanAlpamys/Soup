"""One measured LoRA training step on an expert-streamed DeepSeek-V3-family store.

Loads a shard cache built with ``expert_layout="per_expert"`` (by the sharder
or by ``synth_moe_store.py``), installs layer + expert streaming, and runs
``--steps`` optimizer steps of LoRA on the attention projections: forward,
backward, AdamW. Writes step seconds, tokens per second, peak VRAM, the
process's peak host memory, and the bytes the expert readers read. On a
synthetic store every number is "random weights, shape only".

    python benchmarks/harness/moe_stream_step.py --store D:/soup-bench/moe-k2-synth \
        --seq 512 --steps 1 --out step.json

With ``--weights`` a real checkpoint is sharded into ``--store`` first (NF4 on
CUDA, the way ``soup train`` does it: native transformers class, the config's
layer count, ``--expert-layout``), unless ``--store`` already holds an index.

``--loss-tokens N`` takes the loss over the last N positions only (transformers'
``logits_to_keep``). Every position still runs forward and backward through every
decoder layer; only the vocabulary projection and the cross entropy shrink. At a
163,840-entry vocabulary they hold about 1.9 MB of VRAM per loss position, which
is what stops a long step from fitting (``probe-rtx5070-giant-moe-seq-scaling.md``).
Each step also records ``os_io``: the bytes the process read and the bytes each
physical drive read during the step, from the OS counters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ATTENTION = ["q_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def _loss_window(text):
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"--loss-tokens must be a whole number >= 0, got {text!r}")
    return value


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, help="shard cache dir")
    parser.add_argument("--weights", default=None, help="checkpoint to shard into --store")
    parser.add_argument(
        "--model", default=None, help="dir with config.json (default: --weights, else --store)"
    )
    parser.add_argument("--expert-layout", choices=("fused", "per_expert"), default="per_expert")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--loss-tokens", type=_loss_window, default=0,
        help="loss over the last N positions of each row only (0, or N >= --seq: all)",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--label", default="random weights, shape only")
    parser.add_argument(
        "--route", choices=("model", "uniform"), default="model",
        help="uniform: every MoE router sends token t to experts 8t..8t+k-1 (mod E), so a "
        "step routes to every expert the way a trained model nearly does (probe #992); "
        "random weights otherwise collapse routing onto a few experts",
    )
    return parser.parse_args(argv)


def _shard(args):
    """Shard ``--weights`` into ``--store`` the way the streamed trainer does."""
    from transformers import AutoConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    config = AutoConfig.from_pretrained(args.weights, trust_remote_code=False)
    probe = build_meta_skeleton(args.weights, dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    started = time.perf_counter()
    shard_checkpoint(
        args.weights, args.store, dtype="bfloat16", arch="deepseek_v3", quant="nf4",
        quant_suffixes=suffixes, double_quant=True, quant_device="cuda",
        num_hidden_layers=int(config.num_hidden_layers),
        expert_layout="" if args.expert_layout == "fused" else "per_expert",
        notify=print,
    )
    return round(time.perf_counter() - started, 1)


def _uniform_routing(model):
    """Replace every DeepSeek-V3 router's choice with a fixed round robin.

    Returns the number of routers replaced. The router's own weight still
    streams (it is in the layer file) but is not used; the expert MLPs, the
    shared expert and the attention are untouched.
    """
    import types

    import torch

    replaced = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV3TopkRouter":
            continue
        top_k, experts = int(module.top_k), int(module.num_experts)

        def forward(self, hidden_states, top_k=top_k, experts=experts):
            rows = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
            device = hidden_states.device
            index = (
                torch.arange(rows, device=device)[:, None] * top_k
                + torch.arange(top_k, device=device)[None, :]
            ) % experts
            weights = torch.full((rows, top_k), 1.0 / top_k, device=device)
            logits = torch.zeros(rows, experts, device=device)
            return logits, weights, index

        module.forward = types.MethodType(forward, module)
        replaced += 1
    return replaced


def _loss_inputs(ids, loss_tokens):
    """``(labels, extra forward kwargs)`` for a loss over the last ``loss_tokens`` positions.

    0, or a window at least as long as the row, is the full loss: ``labels=ids`` and no
    extra kwargs, the harness's behaviour before ``--loss-tokens``. Otherwise
    ``logits_to_keep=N`` projects only the last N hidden states and the labels are the
    last N ids, so the loss's own shift keeps next-token targets: position T-N+j
    predicts id T-N+j+1, and the window's last position is ignored as the row's is.
    """
    if loss_tokens <= 0 or loss_tokens >= int(ids.shape[-1]):
        return ids, {}
    return ids[:, -loss_tokens:], {"logits_to_keep": int(loss_tokens)}


def _forward_loss(model, ids, loss_tokens):
    """The step's loss and the number of loss positions per row."""
    labels, extra = _loss_inputs(ids, loss_tokens)
    return model(input_ids=ids, labels=labels, **extra).loss, int(labels.shape[-1])


def _io_snapshot():
    """Bytes read so far by this process and by each physical drive (psutil).

    ``None`` for a counter the platform does not give. Drive read TIMES are not
    recorded: psutil's Windows ``read_time`` was not credible on this box (seconds
    for terabytes).
    """
    try:
        import psutil
    except ImportError:
        return {"process_read_bytes": None, "disk_read_bytes": None}
    refused = (AttributeError, OSError, RuntimeError, psutil.Error)
    try:
        process = int(psutil.Process().io_counters().read_bytes)
    except refused:
        process = None
    try:
        counters = psutil.disk_io_counters(perdisk=True) or {}
        disks = {name: int(value.read_bytes) for name, value in counters.items()}
    except refused:
        disks = None
    return {"process_read_bytes": process, "disk_read_bytes": disks}


def _io_delta(before, after):
    """What was read between two snapshots: the process's bytes and each drive's."""
    process = None
    if before["process_read_bytes"] is not None and after["process_read_bytes"] is not None:
        process = after["process_read_bytes"] - before["process_read_bytes"]
    disks = None
    if before["disk_read_bytes"] is not None and after["disk_read_bytes"] is not None:
        disks = {
            name: value - before["disk_read_bytes"][name]
            for name, value in after["disk_read_bytes"].items()
            if name in before["disk_read_bytes"]
        }
    return {"process_read_bytes": process, "disk_read_bytes": disks}


def _peak_host_bytes():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))
    except ImportError:
        return None


def main(argv=None):
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import read_shard_index, shard_cache_disk_bytes
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    args = _args(argv)
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    model_dir = args.model or args.weights or args.store
    report = {"label": args.label, "args": vars(args), "steps": []}
    if args.weights and not os.path.exists(os.path.join(args.store, "index.json")):
        report["shard_seconds"] = _shard(args)
    index = read_shard_index(args.store)
    report["store_bytes"] = shard_cache_disk_bytes(args.store, index)
    # What the store holds, not what was asked for: an existing store is reused.
    report["store_expert_layout"] = index.expert_layout or "fused"
    report["store_quant"] = index.quant
    report["store_double_quant"] = bool(index.double_quant)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
        target_modules=_ATTENTION, task_type=TaskType.CAUSAL_LM,
    )
    started = time.perf_counter()
    model, runtime = build_streamed_model(
        model_id=model_dir, shard_dir=args.store, index=index, lora_config=lora,
        device="cuda", dtype=index.dtype, buffers=args.buffers, pin=not args.no_pin,
        tier=args.tier, seed=args.seed, quant=index.quant,
        double_quant=bool(index.double_quant),
    )
    report["build_seconds"] = round(time.perf_counter() - started, 2)
    model.train()
    if args.route == "uniform":
        report["routers_overridden"] = _uniform_routing(model)
        report["label"] += "; routing forced uniform (token t -> experts kt..kt+k-1 mod E)"
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    report["trainable_params"] = sum(param.numel() for param in trainable)
    vocab = model.config.vocab_size
    generator = torch.Generator().manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.steps):
            ids = torch.randint(0, vocab, (args.batch, args.seq), generator=generator).cuda()
            before = runtime.stats()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            io_before = _io_snapshot()
            tick = time.perf_counter()
            loss, loss_positions = _forward_loss(model, ids, args.loss_tokens)
            loss.backward()
            torch.cuda.synchronize()
            seconds_to_grads = time.perf_counter() - tick
            # Checked here, before the optimizer consumes and zeroes the grads.
            grads = [param.grad for param in trainable]
            grads_present = sum(grad is not None for grad in grads)
            grads_finite = grads_present > 0 and all(
                bool(torch.isfinite(grad).all()) for grad in grads if grad is not None
            )
            tick_opt = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds = seconds_to_grads + (time.perf_counter() - tick_opt)
            io_after = _io_snapshot()
            after = runtime.stats()
            report["steps"].append(
                {
                    "step": step,
                    "seconds": round(seconds, 2),
                    "tokens_per_second": round(args.batch * args.seq / seconds, 3),
                    "loss": float(loss.detach()),
                    "loss_positions": loss_positions,
                    "loss_finite": bool(torch.isfinite(loss.detach())),
                    "grads_finite": grads_finite,
                    "grads_present": grads_present,
                    "expert_fetches": after["expert_fetches"] - before["expert_fetches"],
                    "expert_bytes_read": after["expert_bytes_read"] - before["expert_bytes_read"],
                    "layer_loads": after["layer_loads"] - before["layer_loads"],
                    # OS counters over the timed region: every byte this process read
                    # (experts, dense layer files, vocabulary matrices) and what each
                    # physical drive read, whoever asked for it.
                    "os_io": _io_delta(io_before, io_after),
                    # This step's peaks (reset at its start); the host figure is the
                    # process's peak working set so far (Windows ``peak_wset``).
                    "peak_vram_mib": torch.cuda.max_memory_allocated() // 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
                    "peak_host_bytes_so_far": _peak_host_bytes(),
                }
            )
            print(json.dumps(report["steps"][-1]))
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
    finally:
        runtime.close()
    report["stats"] = runtime.stats()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
