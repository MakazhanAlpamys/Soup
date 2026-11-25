#!/usr/bin/env python3
"""Exactness gate for training.stream_recompute on a REAL streamed model: ON vs OFF.

One process. Deterministic algorithms and the MATH sdpa backend, the same input ids, and
the loss and every LoRA gradient compared bit for bit. Two controls make "equal" mean
something: ON vs ON again must be equal (the harness is deterministic), and a NEGATIVE
control, one input token changed, must differ. LoRA B starts at zero, which makes every A
gradient zero, so B is given small random values first. The run refuses if the switch
found a different number of streamed layers than the config has: a first version compared
ON with ON and reported "equal" because the wrapper is transparent to named_modules().

Typical invocation::

    python benchmarks/harness/stream_recompute_exactness.py \
        --weights unsloth/mistral-7b-instruct-v0.3 --seq 512

A machine without CUDA is an intentional skip and exits 0.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]


def _stream_probe() -> Any:
    """Import the sibling probe, whichever directory this ran from."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import stream_probe

    return stream_probe


def _assert_this_tree() -> str:
    """The measured code must be this checkout's, not a same-version copy elsewhere."""
    import soup_cli

    origin = os.path.normcase(os.path.realpath(soup_cli.__file__))
    src = os.path.normcase(os.path.realpath(REPO / "src"))
    assert origin.startswith(src), (
        f"soup_cli is imported from {soup_cli.__file__}, not from {src}; put this checkout's "
        "src first on PYTHONPATH"
    )
    return soup_cli.__file__


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default="unsloth/mistral-7b-instruct-v0.3")
    parser.add_argument("--shards", default=None, help="an already-sharded directory")
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--out", default=None, help="JSON results file (default: stdout only)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    cli = parse_args(argv)
    # Before the first CUDA context: cuBLAS reads it at handle creation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stream_probe = _stream_probe()
    if not stream_probe.cuda_available():
        print("SKIP: no CUDA device")
        return 0
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute

    print("soup_cli from", _assert_this_tree())
    device = "cuda"
    args = argparse.Namespace(
        weights=cli.weights,
        shards=cli.shards,
        quant=cli.quant,
        tier=cli.tier,
        no_pin=False,
        buffers=2,
        read_ahead=2,
        lora_r=cli.lora_r,
        lora_targets=cli.lora_targets,
        seed=cli.seed,
        lazy_shard_handles=False,
        control_sync_source=False,
    )
    model, _runtime, config, *_ = stream_probe.build(args, device, resolve_stream_dtype(device))
    model.train()
    lora = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    b_generator = torch.Generator(device=device).manual_seed(5)
    with torch.no_grad():
        for name, param in lora:
            if "lora_B" in name:
                noise = torch.randn(
                    param.shape, generator=b_generator, device=device, dtype=param.dtype
                )
                param.copy_(noise * 0.01)
    n_layers = int(config.num_hidden_layers)
    switched = set_stream_recompute(model, False)
    assert switched == n_layers, f"the switch found {switched} streamed layers of {n_layers}"
    generator = torch.Generator(device=device).manual_seed(cli.input_seed)
    vocab = int(config.vocab_size)
    ids = torch.randint(0, vocab, (cli.batch, cli.seq), generator=generator, device=device)
    ids_negative = ids.clone()
    position = min(100, cli.seq - 1)
    ids_negative[0, position] = (ids_negative[0, position] + 1) % vocab
    torch.use_deterministic_algorithms(True)

    def run(recompute: bool, inputs: Any) -> Dict[str, Any]:
        set_stream_recompute(model, recompute)
        for _, param in lora:
            param.grad = None
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with sdpa_kernel([SDPBackend.MATH]):
            base = torch.cuda.memory_allocated()
            out = model(input_ids=inputs, labels=inputs)
            torch.cuda.synchronize()
            kept = torch.cuda.memory_allocated() - base
            out.loss.backward()
        torch.cuda.synchronize()
        return {
            "loss": out.loss.detach().clone(),
            "grads": [param.grad.detach().clone() for _, param in lora],
            "seconds": time.perf_counter() - started,
            "kept_after_forward_gb": kept / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        differing = sum(
            0 if torch.equal(x, y) else 1 for x, y in zip(a["grads"], b["grads"])
        )
        return {
            "loss_equal": bool(torch.equal(a["loss"], b["loss"])),
            "grads_differing": differing,
            "n_grads": len(a["grads"]),
        }

    on_1 = run(True, ids)
    on_2 = run(True, ids)
    off_1 = run(False, ids)
    off_2 = run(False, ids)
    negative = run(False, ids_negative)
    results = {
        "gpu": torch.cuda.get_device_name(),
        "weights": cli.weights,
        "shape": [cli.batch, cli.seq],
        "switched_layers": switched,
        "on_vs_on": compare(on_1, on_2),
        "off_vs_on": compare(off_1, on_1),
        "off_vs_off": compare(off_1, off_2),
        "negative_control_vs_on (must differ)": compare(negative, on_1),
        "seconds": {
            "on_1": on_1["seconds"],
            "on_2": on_2["seconds"],
            "off_1": off_1["seconds"],
            "off_2": off_2["seconds"],
        },
        "kept_after_forward_gb": {
            "on": on_2["kept_after_forward_gb"],
            "off": off_2["kept_after_forward_gb"],
        },
        "peak_gb": {"on": on_2["peak_gb"], "off": off_2["peak_gb"]},
        "nonzero_grads": sum(int(bool(grad.abs().sum() > 0)) for grad in on_1["grads"]),
    }
    text = json.dumps(results, indent=1)
    if cli.out:
        Path(cli.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    ok = (
        results["on_vs_on"]["grads_differing"] == 0
        and results["off_vs_on"]["grads_differing"] == 0
        and results["off_vs_on"]["loss_equal"]
        and results["negative_control_vs_on (must differ)"]["grads_differing"] > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
