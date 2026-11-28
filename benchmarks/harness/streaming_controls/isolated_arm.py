"""One isolated streaming calibration arm; historical compare.py remains frozen.

The workload mirrors its synthetic pilot. Each process builds its own model,
source and optimizer; the parent must verify identity and A/B numerical parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from benchmarks.harness.streaming_controls.compare import (
    ReadCounter,
    resident_shards,
    save,
    source_stats,
    tensor_digest,
)


def run(destination: Path, mode: str, arm: str, prepared: Path | None = None) -> dict[str, Any]:
    if arm not in ("A", "B"):
        raise ValueError("arm must be A or B")
    if mode not in ("cpu-smoke", "cuda-pilot", "cuda-representative"):
        raise ValueError("unknown comparison mode")
    if (
        mode == "cuda-representative"
        and prepared is None
        or mode == "cuda-pilot"
        and prepared is not None
    ):
        raise ValueError("representative mode needs prepared inputs; synthetic pilot forbids them")
    if destination.exists():
        raise FileExistsError("output must be a new directory; no retry or overwrite")
    destination.mkdir(parents=True)
    try:
        return _run(destination, mode, arm, prepared)
    except BaseException as error:
        receipt = destination / "result.json"
        result = json.loads(receipt.read_text()) if receipt.exists() else {"mode": mode}
        result.update(status="FAILED", error=repr(error), performance_verdict="NO_VERDICT")
        save(receipt, result)
        raise


def _run(destination: Path, mode: str, arm: str, prepared: Path | None = None) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM

    from benchmarks.harness.arm_validity import ArmWatch, confirm_gpu_owner, gpu_compute_pids
    from benchmarks.harness.streaming_controls.io_calibration import difference, snapshot
    from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    cuda = mode != "cpu-smoke"
    workload = None
    digest_tensors = tensor_digest
    if prepared is not None:
        from benchmarks.harness.streaming_controls.representative_workload import load_prepared
        from benchmarks.harness.streaming_controls.representative_workload import (
            tensor_digest as digest_tensors,
        )

        workload = load_prepared(prepared, cuda=cuda)
    device = "cuda:0" if cuda else "cpu"
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("the pilot requires exactly one CUDA device")
    ownership = None
    if cuda:
        before = gpu_compute_pids()
        if before != []:
            raise RuntimeError(f"GPU must be observably idle before initialization: {before!r}")
        context_probe = torch.empty(1, device=device)
        torch.cuda.synchronize()
        ownership = confirm_gpu_owner(before, context_active=True, single_device=True)
        del context_probe
        if ownership["method"] != "exclusive-local-CUDA-initialization":
            raise RuntimeError(f"cannot establish GPU ownership: {ownership!r}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=256 if cuda else 64,
        hidden_size=1024 if cuda else 32,
        intermediate_size=4096 if cuda else 64,
        num_hidden_layers=8 if cuda else 4,
        num_attention_heads=8 if cuda else 4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        max_position_embeddings=256,
    )
    if workload is None:
        base = LlamaForCausalLM(config).to(torch.float32).eval()
        weights = destination / "weights"
        base.save_pretrained(weights, safe_serialization=True)
        del base
        directory = destination / "shards"
        index = shard_checkpoint(str(weights), str(directory), dtype="float32", arch="llama")
        identity = {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in weights.iterdir()
            if p.is_file()
        }
    else:
        weights = Path(workload["weights"])
        config = LlamaConfig.from_pretrained(str(weights), local_files_only=True)
        directory = prepared.parent / "shards"
        index = read_shard_index(str(directory))
        identity = workload["identity"]
    records = []
    order = [arm]
    steps, warmup = (6, 2) if cuda else (3, 0)
    cache = 128 << 20 if cuda else 45_000
    envelope = 512 << 20 if cuda else 10_000_000
    reserved = 64 << 20 if cuda else 0
    if workload is not None:
        steps, warmup = workload["steps"], workload["warmup"]
        cache, envelope = workload["cache_bytes"], workload["limit_bytes"]
        reserved = workload["reserved_bytes"]
    result: dict[str, Any] = {
        "mode": mode,
        "source_model": workload["kind"] if workload else "synthetic random dense Llama",
        "workload_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest() if workload else None,
        "effective_tokens_per_step": workload["effective_tokens_per_step"] if workload else None,
        "config": config.to_dict(),
        "weight_sha256": identity,
        "steps_per_arm": steps,
        "warmup_steps": warmup,
        "order": order,
        "torch": torch.__version__,
        "records": records,
        "gpu_ownership": ownership,
        "performance_verdict": "NO_VERDICT",
        "cache_scope": "benchmark-local decoder shard bytes",
        "status": "RUNNING",
        "pid": os.getpid(),
        "scope": "single fresh process including source generation and cache fill",
    }
    save(destination / "result.json", result)
    for position, arm in enumerate(order):
        row: dict[str, Any] = {"position": position, "arm": arm, "updates": []}
        records.append(row)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        setup_io = snapshot(directory) if cuda else None
        try:
            indices, resident = resident_shards(directory, index, cache if arm == "B" else 0)
            with ReadCounter(resident) as reader, ExitStack() as runtime_cleanup:
                model, runtime = build_streamed_model(
                    model_id=str(weights),
                    shard_dir=str(directory),
                    index=index,
                    lora_config=LoraConfig(
                        r=4,
                        lora_alpha=8,
                        lora_dropout=0,
                        bias="none",
                        target_modules=["q_proj", "v_proj"],
                        task_type=TaskType.CAUSAL_LM,
                    ),
                    device=device,
                    dtype=workload["dtype"] if workload else "float32",
                    pin=cuda,
                    seed=3,
                    tier="disk",
                    read_ahead=1,
                )
                runtime_cleanup.callback(runtime.close)
                row["direct_io"] = runtime.source.direct_io
                row["resident_indices"] = indices
                row["setup_seconds"] = time.perf_counter() - started
                row["initial_stats"] = source_stats(runtime, reader, reserved=reserved)
                if cuda:
                    row["setup_physical_io"] = difference(setup_io, snapshot(directory))
                if arm == "B" and not 0 < len(row["resident_indices"]) < config.num_hidden_layers:
                    raise RuntimeError("the comparison requires strictly partial residency")
                if row["initial_stats"]["host_allocated_bytes"] > envelope - reserved:
                    raise RuntimeError("owned source allocations exceed the common envelope")
                if cuda and not row["direct_io"]:
                    raise RuntimeError("pilot refuses buffered disk fallback")
                params = {
                    name: param for name, param in model.named_parameters() if param.requires_grad
                }
                row["initial_adapter_sha256"] = digest_tensors(params)
                optimizer = torch.optim.AdamW(params.values(), lr=0.001)
                model.train()
                watch = (
                    ArmWatch(arm=arm, round=position // 4, run=position, model_path=str(directory))
                    if cuda
                    else None
                )
                with watch if watch is not None else nullcontext():
                    for step in range(steps):
                        if workload:
                            ids = torch.tensor([workload["tokens"][step]], device=device)
                        else:
                            ids = (
                                (torch.arange(128 if cuda else 8).reshape(1, -1) + step)
                                % config.vocab_size
                            ).to(device)
                        optimizer.zero_grad()
                        if cuda:
                            torch.cuda.synchronize()
                        before = reader.snapshot()
                        io_before = snapshot(directory) if cuda else None
                        tick = time.perf_counter()
                        loss = model(input_ids=ids, labels=ids).loss
                        loss.backward()
                        gradients = {name: param.grad for name, param in params.items()}
                        if not torch.isfinite(loss) or not all(
                            value is not None and torch.isfinite(value).all()
                            for value in gradients.values()
                        ):
                            raise RuntimeError("non-finite loss or missing/non-finite gradient")
                        # Numerical hashing is outside the reported training time.
                        if cuda:
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - tick
                        grad_hash = digest_tensors(gradients)
                        tick = time.perf_counter()
                        optimizer.step()
                        if cuda:
                            torch.cuda.synchronize()
                        elapsed += time.perf_counter() - tick
                        after = reader.snapshot()
                        row["updates"].append(
                            {
                                "step": step,
                                "warmup": step < warmup,
                                "loss": float(loss.detach().cpu()),
                                "gradient_sha256": grad_hash,
                                "training_seconds": elapsed,
                                "range_read_calls": after["calls"] - before["calls"],
                                "range_read_bytes": after["bytes"] - before["bytes"],
                            }
                        )
                        if cuda:
                            io_after = snapshot(directory)
                            row["updates"][-1]["physical_io"] = difference(io_before, io_after)
                            if workload:
                                # Shared boundary for physical I/O and instrumented time.
                                wall = io_after["monotonic_s"] - io_before["monotonic_s"]
                                row["updates"][-1].update(
                                    instrumented_window_seconds=wall,
                                    effective_tokens=workload["effective_tokens_per_step"],
                                    instrumented_tokens_per_second=(
                                        workload["effective_tokens_per_step"] / wall
                                    ),
                                )
                row["validity"] = watch.record if watch else None
                row["final_adapter_sha256"] = digest_tensors(params)
                row["changed_adapter"] = (
                    row["final_adapter_sha256"] != row["initial_adapter_sha256"]
                )
                row["final_stats"] = source_stats(runtime, reader, reserved=reserved)
                if cuda:
                    row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    row["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                if cuda:
                    import resource

                    row["process_lifetime_maxrss_kib"] = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                runtime_cleanup.close()
                row["range_reads_total"] = reader.snapshot()
                row["status"] = "COMPLETED"
                if position and (
                    row["final_adapter_sha256"] != records[0]["final_adapter_sha256"]
                    or [(x["loss"], x["gradient_sha256"]) for x in row["updates"]]
                    != [(x["loss"], x["gradient_sha256"]) for x in records[0]["updates"]]
                ):
                    raise RuntimeError("exact numerical comparison failed; no further arms")
                del optimizer, params, gradients, model
        except BaseException as error:
            row["status"] = "FAILED"
            row["error"] = repr(error)
            result["status"] = "FAILED"
            save(destination / "result.json", result)
            raise
        save(destination / "result.json", result)
    # A single arm cannot establish A/B parity; the parent compares all records.
    reference = records[0]
    matches = all(
        row["changed_adapter"]
        and row["final_adapter_sha256"] == reference["final_adapter_sha256"]
        and [(x["loss"], x["gradient_sha256"]) for x in row["updates"]]
        == [(x["loss"], x["gradient_sha256"]) for x in reference["updates"]]
        for row in records
    )
    result["adapter_changed"] = matches
    result["status"] = "ARM_COMPLETE" if matches else "NUMERICAL_FAIL"
    # Arm validity, repetitions, physical device attribution and memory/clock
    # review remain independent gates; a numerical pass cannot approve speed.
    save(destination / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["cpu-smoke", "cuda-pilot", "cuda-representative"], default="cpu-smoke"
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    args = parser.parse_args()
    result = run(args.output, args.mode, args.arm, args.prepared)
    from rich.console import Console

    Console().print(f"{result['status']}; performance={result['performance_verdict']}")
    return 0 if result["adapter_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
