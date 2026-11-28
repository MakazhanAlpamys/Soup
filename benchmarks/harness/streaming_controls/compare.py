"""Bounded synthetic streaming comparison. CPU smoke is never a hardware performance verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

_RANGE_READER_PATCH = threading.Lock()


def tensor_digest(tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ReadCounter:
    """Count successful readinto return bytes, not dispatched work or physical sectors.

    Scope: async shard data-range reads only; excludes header I/O and initial RAM
    cache filling. Keep the wrapper active until all reader threads have closed.
    """

    def __init__(self, resident: dict[Any, bytes] | None = None) -> None:
        self.bytes = 0
        self.calls = 0
        self.by_file: dict[str, dict[str, int]] = {}
        self.resident_bytes_served = 0
        self.resident = resident or {}
        self._lock = threading.Lock()

    def __enter__(self) -> ReadCounter:
        from soup_cli.utils import async_disk_source
        from soup_cli.utils.safetensors_reader import identity_of

        counter = self

        class Handle:
            def __init__(self, wrapped: Any) -> None:
                self.wrapped = wrapped
                stat = os.fstat(wrapped.fileno())
                self.key = f"{stat.st_dev}:{stat.st_ino}"

            def seek(self, offset: int) -> Any:
                return self.wrapped.seek(offset)

            def readinto(self, buffer: Any) -> int:
                got = self.wrapped.readinto(buffer)
                with counter._lock:
                    counter.calls += 1
                    counter.bytes += got or 0
                    row = counter.by_file.setdefault(self.key, {"calls": 0, "bytes": 0})
                    row["calls"] += 1
                    row["bytes"] += got or 0
                return got

        def read(handle: Any, start: int, view: Any, expected: int) -> None:
            cached = counter.resident.get(identity_of(handle)) if counter.resident else None
            if cached is not None:
                memoryview(view.numpy())[:expected] = memoryview(cached)[start : start + expected]
                with counter._lock:
                    counter.resident_bytes_served += expected
                return
            original(Handle(handle), start, view, expected)

        _RANGE_READER_PATCH.acquire()
        original = self._original = async_disk_source.read_range_into
        async_disk_source.read_range_into = read
        return self

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "bytes": self.bytes,
                "by_file": {key: dict(value) for key, value in self.by_file.items()},
            }

    def __exit__(self, *_args: Any) -> None:
        from soup_cli.utils import async_disk_source

        async_disk_source.read_range_into = self._original
        _RANGE_READER_PATCH.release()


def resident_shards(
    directory: Path, index: Any, cache_bytes: int
) -> tuple[list[int], dict[Any, bytes]]:
    """Fill a benchmark-local cache with complete decoder shard bytes within the fixed cap."""
    from soup_cli.utils.layer_shard import layer_paths
    from soup_cli.utils.safetensors_reader import identity_of

    indices, files = [], {}
    if not cache_bytes:
        return indices, files
    for layer, path in enumerate(layer_paths(str(directory), index)):
        with open(path, "rb") as handle:
            identity = identity_of(handle)
            if identity.size > cache_bytes:
                continue
            data = handle.read(identity.size)
            if len(data) != identity.size or identity_of(handle) != identity:
                raise ValueError(f"decoder shard changed during cache fill: {path}")
        files[identity] = data
        indices.append(layer)
        cache_bytes -= len(data)
    return indices, files


def source_stats(runtime: Any, reader: ReadCounter, *, reserved: int) -> dict[str, Any]:
    staging = max(runtime.source.staging_bytes, runtime.source.pinned_bytes)
    resident_bytes = sum(map(len, reader.resident.values()))
    return {
        **runtime.stats(),
        "resident_bytes": resident_bytes,
        "host_allocated_bytes": staging + resident_bytes,
        "accounted_peak_bytes": staging + resident_bytes + reserved,
        "resident_range_bytes_served": reader.resident_bytes_served,
    }


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(destination: Path, mode: str) -> dict[str, Any]:
    if mode not in ("cpu-smoke", "cuda-pilot"):
        raise ValueError("unknown comparison mode")
    if destination.exists():
        raise FileExistsError("output must be a new directory; no retry or overwrite")
    destination.mkdir(parents=True)
    settings = None
    try:
        import torch

        settings = (
            torch.get_num_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        return _run(destination, mode)
    except BaseException as error:
        receipt = destination / "result.json"
        result = json.loads(receipt.read_text()) if receipt.exists() else {"mode": mode}
        result.update(status="FAILED", error=repr(error), performance_verdict="NO_VERDICT")
        save(receipt, result)
        raise
    finally:
        if settings is not None:
            threads, deterministic, warn_only, matmul_tf32, cudnn_tf32 = settings
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _run(destination: Path, mode: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM

    from benchmarks.harness.arm_validity import ArmWatch, confirm_gpu_owner, gpu_compute_pids
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    cuda = mode == "cuda-pilot"
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
    records = []
    order = list("ABBA") * 5 if cuda else list("AB")
    steps, warmup = (6, 2) if cuda else (3, 0)
    cache = 128 << 20 if cuda else 45_000
    envelope = 512 << 20 if cuda else 10_000_000
    reserved = 64 << 20 if cuda else 0
    result: dict[str, Any] = {
        "mode": mode,
        "source_model": "synthetic random dense Llama",
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
    }
    save(destination / "result.json", result)
    for position, arm in enumerate(order):
        row: dict[str, Any] = {"position": position, "arm": arm, "updates": []}
        records.append(row)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
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
                    dtype="float32",
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
                if arm == "B" and not 0 < len(row["resident_indices"]) < config.num_hidden_layers:
                    raise RuntimeError("the comparison requires strictly partial residency")
                if row["initial_stats"]["host_allocated_bytes"] > envelope - reserved:
                    raise RuntimeError("owned source allocations exceed the common envelope")
                if cuda and not row["direct_io"]:
                    raise RuntimeError("pilot refuses buffered disk fallback")
                params = {
                    name: param for name, param in model.named_parameters() if param.requires_grad
                }
                row["initial_adapter_sha256"] = tensor_digest(params)
                optimizer = torch.optim.AdamW(params.values(), lr=0.001)
                model.train()
                watch = (
                    ArmWatch(arm=arm, round=position // 4, run=position, model_path=str(directory))
                    if cuda
                    else None
                )
                with watch if watch is not None else nullcontext():
                    for step in range(steps):
                        ids = (
                            (torch.arange(128 if cuda else 8).reshape(1, -1) + step)
                            % config.vocab_size
                        ).to(device)
                        optimizer.zero_grad()
                        if cuda:
                            torch.cuda.synchronize()
                        before = reader.snapshot()
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
                        grad_hash = tensor_digest(gradients)
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
                row["validity"] = watch.record if watch else None
                row["final_adapter_sha256"] = tensor_digest(params)
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
    reference = records[0]
    matches = all(
        row["changed_adapter"]
        and row["final_adapter_sha256"] == reference["final_adapter_sha256"]
        and [(x["loss"], x["gradient_sha256"]) for x in row["updates"]]
        == [(x["loss"], x["gradient_sha256"]) for x in reference["updates"]]
        for row in records
    )
    result["exact_training_match"] = matches
    result["status"] = "NUMERICAL_PASS" if matches else "NUMERICAL_FAIL"
    # Arm validity, repetitions, physical device attribution and memory/clock
    # review remain independent gates; a numerical pass cannot approve speed.
    save(destination / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu-smoke", "cuda-pilot"], default="cpu-smoke")
    args = parser.parse_args()
    result = run(args.output, args.mode)
    from rich.console import Console

    Console().print(f"{result['status']}; performance={result['performance_verdict']}")
    return 0 if result["exact_training_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
