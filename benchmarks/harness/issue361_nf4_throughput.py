#!/usr/bin/env python3
"""Reproduce the v0.72.2 streamed-NF4 throughput protocol (#361).

The published headline — Llama-3.1-8B-Instruct NF4 streamed at 119.6 tok/s,
3.32 GB peak, RTX 3050 Laptop 4 GB (``benchmarks/gate-v0.72.2-nf4.md``, step-6
reproduction) — was measured BEFORE the #331 repair. This harness re-runs that
protocol unchanged on the released code so the post-repair number is a
measurement, not an extrapolation:

    shard -> stream -> LoRA -> PagedAdamW8bit ->
    10 warm-up steps + 50 measured steps, batch 1, S=512,
    tok/s, peak VRAM, GPU util + SM clock sampled during the measured window,
    dense bf16 GEMM ceiling taken IN THE SAME SESSION (the gate's own
    methodological rule: a ceiling is only comparable to a throughput
    measured at the same clock).

Fidelity guards, because a wrong number here is worse than no number:

- ``require_pin=True``: if the store cannot be page-locked the harness
  REFUSES rather than recording the pageable lower-bound shape the v0.72.0
  honesty caveat warned about.
- every step's loss must be finite: a NaN run (#342's shape) is not a
  throughput row and exits non-zero.
- GFLOP/token is computed from the checkpoint's own safetensors headers
  with the gate's numerator convention (decoder C=6, ``lm_head`` C=4,
  ``embed_tokens`` 0 — a tied head is not double-counted), and printed
  with its inputs so the derived TFLOPS can be audited. The formula
  reproduces both published rows: 43.98 at Llama-3.1-8B, 2.69 at
  Qwen2.5-0.5B.

Typical invocation
------------------
    python benchmarks/harness/issue361_nf4_throughput.py \
        --weights meta-llama/Llama-3.1-8B-Instruct \
        --shards ./shards-llama8b-nf4 \
        --json ./rows/issue361-post-repair.json

``--json`` re-writes the measured row after EVERY measured step (and marks the
partial evidence if the run aborts on a non-finite loss), so a long session is
not lost to a bad tail — a hard kill runs no handler at all.

Requirements: CUDA GPU, ~4 GB VRAM at S=512 batch 1 for the 8B NF4 row,
host RAM for the pinned store (3.60 GB at 8B) plus shard-build headroom,
disk for the source checkpoint and the ~5.7 GB shard set, and
bitsandbytes for NF4 + PagedAdamW8bit. A machine without CUDA is an
intentional skip and exits 0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

DEFAULT_SEQ = 512
DEFAULT_BATCH = 1
DEFAULT_WARMUP = 10
DEFAULT_STEPS = 50
DEFAULT_BUFFERS = 2
DEFAULT_LORA_R = 16
DEFAULT_TARGETS = "q_proj,v_proj"
SEED = 3
INPUT_SEED = 17


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
    except Exception:
        return "unknown"
    try:
        found = version(name)
    except Exception:
        return "unknown"
    return str(found) if found else "unknown"


def _soup_version() -> str:
    try:
        import soup_cli
    except Exception:
        return "unknown"
    return str(getattr(soup_cli, "__version__", "?") or "?")


def _source_sha() -> str:
    """The tree the row was produced from, or ``unknown``.

    ``soup-cli`` reports a release version that does not move between commits,
    so without this a row cannot name its own tree — and this harness measures a
    path that three post-#361-base commits changed (#989 above all).
    """
    import shutil
    import subprocess

    tool = shutil.which("git")
    if tool is None:
        return "unknown"
    try:
        out = subprocess.run(
            [tool, "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _versions() -> dict:
    """Tooling versions plus the source commit, printed and written with the row."""
    import platform

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is the [train] extra
        torch = None

    return {
        "python": platform.python_version(),
        "soup-cli": _soup_version(),
        "commit": _source_sha(),
        "torch": getattr(torch, "__version__", "unknown"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "trl": _package_version("trl"),
        "accelerate": _package_version("accelerate"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the v0.72.2 streamed-NF4 throughput protocol (#361)."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="checkpoint path or model id resolvable by Soup's weight resolver",
    )
    parser.add_argument(
        "--shards",
        required=True,
        help="directory in which Soup should create or reuse layer shards",
    )
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--no-ceiling",
        action="store_true",
        help="skip the same-session GEMM ceiling (not the published protocol)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        default=None,
        help="also write the row as JSON to PATH, re-written after every measured step",
    )
    return parser.parse_args()


def _dump_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _partial_row(status: str, step_times: list, losses: list, versions: dict) -> dict:
    """A run that has not finished, as far as it got.

    Written after every measured step, because a 60-step 8B run on a 4 GB laptop
    is exactly the run that dies at step 40 — and an OOM or a power cut runs no
    handler at all.
    """
    return {
        "status": status,
        "steps_completed": len(step_times),
        "step_times_s": [round(t, 4) for t in step_times],
        "losses": losses,
        "versions": versions,
    }


def read_param_split(weights_dir: str, config: dict | None = None) -> dict:
    """Split checkpoint parameters from the safetensors headers themselves.

    The gate's rule: counts come from headers, not ``model.parameters()`` on a
    meta skeleton (which double-counted a tied ``lm_head`` at 3B). Returns
    embed/lm_head/decoder element counts plus vocab*hidden from config.
    ``config`` accepts a pre-loaded mapping so main() parses config.json once;
    when omitted the file is read here (which is what the tests rely on).
    """
    from safetensors import safe_open

    directory = Path(weights_dir)
    if config is None:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    elif not isinstance(config, dict):
        config = config.to_dict()  # a transformers PretrainedConfig
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])

    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {directory}")

    embed = lm_head = other = 0
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for name in handle.keys():
                sliced = handle.get_slice(name)
                numel = 1
                for dim in sliced.get_shape():
                    numel *= int(dim)
                if "embed_tokens" in name:
                    embed += numel
                elif "lm_head" in name:
                    lm_head += numel
                else:
                    other += numel
    return {
        "embed": embed,
        "lm_head_file": lm_head,
        "decoder": other,
        "lm_head_effective": vocab * hidden,
        "tied": bool(config.get("tie_word_embeddings", False)),
    }


def gflop_per_token(split: dict) -> float:
    """The gate's numerator convention: decoder C=6, lm_head C=4, embed 0."""
    return (6 * split["decoder"] + 4 * split["lm_head_effective"]) / 1e9


class GpuSampler:
    """Sample util/SM-clock/temp while the measured window runs.

    The gate protocol reads SM occupancy from ``nvidia-smi dmon -s u``; this
    polls the same driver counters over the query interface so it works on
    Windows too, and reports the clock the throughput was taken at — the
    ~13% between-session boost spread is the whole reason the gate demands it.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self.samples: list[tuple[int, int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        import shutil
        import subprocess

        tool = shutil.which("nvidia-smi")
        if tool is None:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        tool,
                        "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                util, clock, temp = (int(x.strip()) for x in out.splitlines()[0].split(","))
                self.samples.append((util, clock, temp))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"util": None, "clock": None, "temp_max": None, "n": 0}
        utils = sorted(s[0] for s in self.samples)
        clocks = sorted(s[1] for s in self.samples)
        temps = sorted(s[2] for s in self.samples)
        mid = (len(utils) - 1) // 2
        return {
            "util": utils[mid],
            "clock": clocks[mid],
            "temp_max": temps[-1],
            "n": len(self.samples),
        }


def main() -> int:  # noqa: C901 — a protocol is a sequence, not a branch tree
    args = parse_args()
    if args.seq <= 0 or args.batch <= 0 or args.steps <= 0 or args.warmup < 0:
        print("ERROR: --seq/--batch/--steps must be positive, --warmup >= 0")
        return 2
    if not 2 <= args.buffers <= 8:
        print("ERROR: --buffers must be between 2 and 8")
        return 2
    if args.lora_r <= 0:
        print("ERROR: --lora-r must be positive")
        return 2
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("ERROR: --targets must name at least one module (e.g. q_proj,v_proj)")
        return 2
    if not cuda_available():
        print("CUDA unavailable — intentional skip (the protocol needs the card).")
        return 0

    import torch
    from peft import LoraConfig, TaskType

    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import resolve_stream_dtype, stream_arch_of
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        measure_gemm_tflops,
        quantised_layer_suffixes,
        sm_clock_mhz,
    )
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    versions = _versions()
    if psutil is not None:
        vm = psutil.virtual_memory()
        print(
            f"host RAM available at start: {vm.available / 1e9:.2f} GB "
            f"of {vm.total / 1e9:.1f} GB"
        )
    else:
        print("host RAM available at start: unknown (psutil not installed)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for name, ver in versions.items():
        print(f"  {name}: {ver}")

    t0 = time.time()
    weights_dir = resolve_model_weights(args.weights)
    print(f"weights resolved -> {weights_dir} ({time.time() - t0:.1f} s)")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(weights_dir)
    split = read_param_split(weights_dir, config)
    gflops = gflop_per_token(split)
    print(
        f"param split (safetensors headers): decoder {split['decoder'] / 1e9:.3f} B, "
        f"embed {split['embed'] / 1e9:.3f} B, lm_head-in-file "
        f"{split['lm_head_file'] / 1e9:.3f} B, tied={split['tied']}; "
        f"GFLOP/token = 6*decoder + 4*lm_head = {gflops:.2f}"
    )

    arch = stream_arch_of(config)
    dtype = resolve_stream_dtype("cuda")
    quant_suffixes = ()
    if args.quant == "nf4":
        probe = build_meta_skeleton(weights_dir, dtype=dtype, quant=args.quant)
        quant_suffixes = quantised_layer_suffixes(probe)
        del probe

    t0 = time.time()
    index = shard_checkpoint(
        weights_dir,
        args.shards,
        dtype=dtype,
        arch=arch,
        quant=args.quant,
        quant_suffixes=quant_suffixes,
        double_quant=True,
        quant_device="cuda",
        notify=lambda msg: print(f"  {msg}"),
    )
    shard_secs = time.time() - t0
    print(f"shards ready in {shard_secs:.1f} s -> {args.shards}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )

    t0 = time.time()
    model, runtime = build_streamed_model(
        model_id=weights_dir,
        shard_dir=args.shards,
        index=index,
        lora_config=lora_config,
        device="cuda",
        dtype=dtype,
        buffers=args.buffers,
        pin=True,
        # Fidelity guard: a silently pageable store is the v0.72.0 lower-bound
        # shape; the protocol's number is a PINNED-store number or it refuses.
        require_pin=True,
        seed=SEED,
        tier="ram",
        quant=args.quant,
        weights_dir=weights_dir,
    )
    build_secs = time.time() - t0
    assert runtime.source.pinned, "require_pin=True but the source is not pinned"
    store_gb = runtime.source.nbytes / 1e9
    print(f"streamed model built in {build_secs:.1f} s | store {store_gb:.2f} GB pinned")

    from bitsandbytes.optim import PagedAdamW8bit

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = PagedAdamW8bit(trainable, lr=2e-4)
    print(f"trainable LoRA params: {n_trainable:,}")

    torch.manual_seed(INPUT_SEED)
    vocab_size = int(config.vocab_size)
    input_ids = torch.randint(
        0, vocab_size, (args.batch, args.seq), device="cuda"
    )
    labels = input_ids.clone()

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def one_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return float(loss.detach().item())

    for step in range(args.warmup):
        loss_val = one_step()
        if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
            print(f"ERROR: non-finite loss during warm-up at step {step}: {loss_val}")
            return 1
    if args.warmup:
        print(f"warm-up done ({args.warmup} steps, last loss {loss_val:.4f}) — measuring")
    else:
        print("no warm-up — measuring")

    step_times: list[float] = []
    losses: list[float] = []
    with GpuSampler() as sampler:
        for _ in range(args.steps):
            t_step = time.time()
            loss_val = one_step()
            step_times.append(time.time() - t_step)
            losses.append(loss_val)
            if args.json_path:
                _dump_json(
                    args.json_path, _partial_row("measuring", step_times, losses, versions)
                )
    gpu = sampler.summary()

    if any(v != v or v in (float("inf"), float("-inf")) for v in losses):
        print("ERROR: non-finite loss inside the measured window — not a valid row")
        if args.json_path:
            _dump_json(
                args.json_path,
                _partial_row("aborted_nonfinite_loss", step_times, losses, versions),
            )
        return 1

    measured_s = sum(step_times)
    tokens = args.steps * args.batch * args.seq
    tok_s = tokens / measured_s
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    eff_tflops = gflops * tok_s / 1000.0

    ceiling = None
    if not args.no_ceiling:
        ceiling = measure_gemm_tflops("cuda")

    print()
    print("================ MEASURED ROW (#361 protocol) ================")
    print(f"weights:            {args.weights}")
    print(f"source commit:      {versions['commit']}")
    print(f"quant/dtype:        {args.quant} / {dtype} / double_quant=True")
    print(f"LoRA:               r={args.lora_r} alpha={2 * args.lora_r} on {args.targets}")
    print(f"seq x batch:        {args.seq} x {args.batch} | buffers {args.buffers}")
    print(f"steps:              {args.warmup} warm-up + {args.steps} measured")
    print(f"store:              {store_gb:.2f} GB pinned")
    print(f"tok/s:              {tok_s:.1f}")
    print(
        f"step time:          median {statistics.median(step_times) * 1e3:.0f} ms | "
        f"min {min(step_times) * 1e3:.0f} | max {max(step_times) * 1e3:.0f} | "
        f"stdev {statistics.pstdev(step_times) * 1e3:.0f} ms"
    )
    print(f"loss:               first {losses[0]:.4f} -> last {losses[-1]:.4f}")
    print(f"peak VRAM:          {peak_alloc:.2f} GB allocated ({peak_reserved:.2f} GB reserved)")
    print(
        f"GPU util / SM clock: {gpu['util']}% / {gpu['clock']} MHz "
        f"(median of {gpu['n']} samples, max temp {gpu['temp_max']} C)"
    )
    print(f"SM clock now:       {sm_clock_mhz()} MHz")
    print(f"GFLOP/token:        {gflops:.2f} (decoder C=6 + lm_head C=4, embed 0)")
    print(f"eff TFLOPS:         {eff_tflops:.2f}")
    if ceiling is not None:
        pct = 100.0 * eff_tflops / ceiling.tflops if ceiling.tflops else float("nan")
        print(
            f"GEMM ceiling:       {ceiling.tflops:.2f} TFLOPS bf16 "
            f"@ {ceiling.sm_clock_mhz} MHz (same session) -> {pct:.0f}% of ceiling"
        )
    print(f"shard build:        {shard_secs:.1f} s | model build: {build_secs:.1f} s")
    print("versions:           " + json.dumps(versions))
    print("================================================================")
    if args.json_path:
        _dump_json(
            args.json_path,
            {
                "status": "ok",
                "issue": 361,
                "weights": args.weights,
                "quant": args.quant,
                "stream_dtype": dtype,
                "seq": args.seq,
                "batch": args.batch,
                "warmup_steps": args.warmup,
                "measured_steps": args.steps,
                "buffers": args.buffers,
                "lora_r": args.lora_r,
                "targets": args.targets,
                "pin_required": True,
                "store_gb_pinned": round(store_gb, 3),
                "param_split": {
                    "decoder": split["decoder"],
                    "embed": split["embed"],
                    "lm_head_file": split["lm_head_file"],
                    "lm_head_effective": split["lm_head_effective"],
                    "tied": split["tied"],
                },
                "tokens_per_s": round(tok_s, 2),
                "step_time_ms": {
                    "median": round(statistics.median(step_times) * 1e3, 1),
                    "min": round(min(step_times) * 1e3, 1),
                    "max": round(max(step_times) * 1e3, 1),
                    "stdev": round(statistics.pstdev(step_times) * 1e3, 1),
                },
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "peak_vram_gb": {
                    "allocated": round(peak_alloc, 3),
                    "reserved": round(peak_reserved, 3),
                },
                "gpu": gpu,
                "sm_clock_mhz_end": sm_clock_mhz(),
                "gflop_per_token": round(gflops, 4),
                "effective_tflops": round(eff_tflops, 3),
                "gemm_ceiling": None
                if ceiling is None
                else {
                    "tflops": round(ceiling.tflops, 3),
                    "sm_clock_mhz": ceiling.sm_clock_mhz,
                },
                "shard_build_s": round(shard_secs, 1),
                "model_build_s": round(build_secs, 1),
                "versions": versions,
            },
        )
        print(f"row written -> {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
