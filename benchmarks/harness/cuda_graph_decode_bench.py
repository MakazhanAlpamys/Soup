"""Reproducible, local Soup request/decode benchmark; run from the workspace root.

    . ./experiments/env.ps1
    ./.venv/Scripts/python.exe experiments/bench_decode.py --output experiments/baseline.json

The primary measurement invokes the shipped ``infer._generate`` and includes
chat tokenization, input transfers, generation, and output decoding. CUDA is
synchronized at both request boundaries. Forward hooks and profilers run in
SEPARATE requests and never contribute to the primary throughput measurement.
The baseline leaves generation settings alone. Explicit experimental variants
change cache/compilation settings, recorded separately from the workload; no
variant changes EOS, precision, sampling, prompt, or requested output length.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_PROMPT = (
    "Explain in detail how database indexes work. Cover B-tree structure, how an "
    "index lookup differs from a full table scan, composite indexes and column "
    "order, the effect on INSERT and UPDATE operations, and a concrete SQL example. "
    "Write a thorough explanation with several paragraphs."
)
EXPERIMENTAL_GRAPH_VARIANTS = ("cudagraphs", "cudagraphs_static_inputs")
GRAPH_VARIANTS = (*EXPERIMENTAL_GRAPH_VARIANTS, "production_cuda_graphs")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).absolute().parent.parent
    parser.add_argument("--source", default=str(root / "Soup"), help="Local Soup source checkout")
    parser.add_argument(
        "--cpu-affinity",
        type=int,
        help="Pin this benchmark process to one logical CPU before torch import",
    )
    parser.add_argument("--model", default=str(root / "experiments/models/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--phase-repeats", type=int, default=None)
    parser.add_argument(
        "--prefill-phase-repeats",
        type=int,
        default=0,
        help="Separate graph-safe _prefill/request event samples after one discarded warmup",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "inference_mode",
            "static",
            "cudagraphs",
            "cudagraphs_static_inputs",
            "production_cuda_graphs",
            "no_padding_mask",
            "mask_callback",
            "inference_mode_mask_callback",
        ),
        default="baseline",
    )
    parser.add_argument(
        "--reference", help="Baseline JSON; require identical workload and token IDs"
    )
    parser.add_argument("--profile", action="store_true", help="Profile separate complete requests")
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Trace one separate request without shapes/memory/aggregation/cProfile",
    )
    parser.add_argument(
        "--telemetry", action="store_true", help="Sample nvidia-smi once per second"
    )
    args = parser.parse_args()
    if args.cpu_affinity is not None and args.cpu_affinity < 0:
        parser.error("--cpu-affinity cannot be negative")
    if args.phase_repeats is None:
        args.phase_repeats = 0 if args.trace_only else 3
    for name in ("max_tokens", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmups", "phase_repeats", "prefill_phase_repeats"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def contained_path(path: str | Path) -> Path:
    root = os.path.realpath(Path(__file__).absolute().parent.parent)
    target = os.path.realpath(path)
    if os.path.normcase(os.path.commonpath((root, target))) != os.path.normcase(root):
        raise ValueError(f"All benchmark paths must stay inside the workspace: {path}")
    return Path(target)


def command(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def apply_cpu_affinity(requested: int | None) -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        original = process.cpu_affinity()
        if requested is not None:
            process.cpu_affinity([requested])
        effective = process.cpu_affinity()
        if requested is not None and effective != [requested]:
            raise RuntimeError(f"Requested CPU {requested}, but actual affinity is {effective}")
        return {"requested": requested, "original": original, "effective": effective}
    except (ImportError, AttributeError, OSError) as exc:
        if requested is not None:
            raise RuntimeError("Cannot apply the requested benchmark CPU affinity") from exc
        return {"requested": None, "original": None, "effective": None, "error": str(exc)}


def metadata(torch: Any, model_path: Path, affinity: dict[str, Any]) -> dict[str, Any]:
    import soup_cli

    imported_source = contained_path(Path(soup_cli.__file__).absolute().parents[2])
    versions = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "numpy",
        "psutil",
        "soup-cli",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    properties = torch.cuda.get_device_properties(0)
    file_fingerprints = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        path = model_path / name
        if path.is_file():
            file_fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        },
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,pstate",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git_sha": command(["git", "rev-parse", "HEAD"], imported_source),
        "git_status": command(["git", "status", "--short"], imported_source),
        "git_diff": command(["git", "diff", "--no-ext-diff"], imported_source),
        "source_checkout": str(imported_source),
        "soup_import_path": soup_cli.__file__,
        "config_fingerprints": file_fingerprints,
        "source_quantization_config": config.get("quantization_config"),
        "relevant_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
                "PYTHONPATH",
            )
        },
    }


class Telemetry:
    """Optional, low-rate external sampler; same setting must be used for A/B."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            row = command(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
                    "memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate",
                    "--format=csv,noheader,nounits",
                ]
            )
            row["host_perf_counter"] = time.perf_counter()
            self.rows.append(row)
            self.stop_event.wait(1.0)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=25)


class CaptureGeneration:
    """Keep CUDA tensor references; perform no device-to-host copy during timing."""

    def __init__(self, model: Any, kwargs_transform: Callable | None = None) -> None:
        self.model = model
        self.original = model.generate
        self.output: Any = None
        self.input_ids: Any = None
        self.attention_mask: Any = None
        self.kwargs_transform = kwargs_transform

    def __enter__(self) -> CaptureGeneration:
        def generate(*args: Any, **kwargs: Any) -> Any:
            self.input_ids = kwargs["input_ids"]
            self.attention_mask = kwargs.get("attention_mask")
            if self.kwargs_transform is not None:
                kwargs = self.kwargs_transform(kwargs)
            self.output = self.original(*args, **kwargs)
            return self.output

        self.model.generate = generate
        return self

    def reset(self) -> None:
        self.output = self.input_ids = self.attention_mask = None

    def snapshot(self) -> dict[str, Any]:
        ids = self.input_ids.detach().cpu().tolist()
        output = self.output.detach().cpu().tolist()
        return {
            "input_ids": ids,
            "output_ids": [row[len(ids[idx]) :] for idx, row in enumerate(output)],
            "attention_mask": self.attention_mask.detach().cpu().tolist()
            if self.attention_mask is not None
            else None,
        }

    def __exit__(self, *args: Any) -> None:
        self.model.generate = self.original


class ForwardPhases:
    """Record stream intervals, not pure-kernel time, without synchronizing per token."""

    def __init__(self, model: Any, torch: Any) -> None:
        self.model = model
        self.torch = torch
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def __enter__(self) -> ForwardPhases:
        def before(module: Any, args: Any, kwargs: Any) -> None:
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            ids = kwargs.get("input_ids")
            start.record()
            self.rows.append(
                {
                    "cpu_start": time.perf_counter(),
                    "cuda_start": start,
                    "cuda_end": end,
                    "input_shape": list(ids.shape) if ids is not None else None,
                }
            )

        def after(module: Any, args: Any, kwargs: Any, output: Any) -> None:
            row = self.rows[-1]
            row["cuda_end"].record()
            row["cpu_end"] = time.perf_counter()

        self.handles = [
            self.model.register_forward_pre_hook(before, with_kwargs=True),
            self.model.register_forward_hook(after, with_kwargs=True),
        ]
        return self

    def summary(self, request_start: float) -> dict[str, Any]:
        spans = []
        for row in self.rows:
            spans.append(
                {
                    "cpu_submit_ms": (row["cpu_end"] - row["cpu_start"]) * 1000,
                    "cuda_stream_interval_ms": row["cuda_start"].elapsed_time(row["cuda_end"]),
                    "start_after_request_ms": (row["cpu_start"] - request_start) * 1000,
                    "input_shape": row["input_shape"],
                }
            )
        if not spans:
            raise RuntimeError("No model.forward calls observed; phase timing is invalid")
        decode = spans[1:]
        return {
            "forward_calls": len(spans),
            "prefill": spans[0],
            "decode_forwards": decode,
            "decode_forward_cuda_sum_ms": sum(row["cuda_stream_interval_ms"] for row in decode),
            "decode_forward_cpu_submit_sum_ms": sum(row["cpu_submit_ms"] for row in decode),
            "first_forward_start_to_last_forward_end_cuda_ms": self.rows[0][
                "cuda_start"
            ].elapsed_time(self.rows[-1]["cuda_end"]),
            "decode_phase_cuda_ms": self.rows[0]["cuda_end"].elapsed_time(
                self.rows[-1]["cuda_end"]
            ),
            "decode_phase_tokens": len(decode),
            "decode_phase_tokens_per_second": len(decode)
            / (self.rows[0]["cuda_end"].elapsed_time(self.rows[-1]["cuda_end"]) / 1000)
            if decode
            else None,
        }

    def __exit__(self, *args: Any) -> None:
        for handle in self.handles:
            handle.remove()


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values)
        if len(values) > 1 and statistics.mean(values)
        else 0.0,
    }


def save(output: Path, result: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate_tokens_per_second": sum(row["tokens"] for row in samples)
        / sum(row["seconds"] for row in samples),
        "tokens_per_second": distribution([row["tokens_per_second"] for row in samples]),
        "request_seconds": distribution([row["seconds"] for row in samples]),
        "latency_per_token_ms": distribution([row["latency_per_token_ms"] for row in samples]),
        "cpu_utilization_percent": distribution(
            [row["cpu_utilization_percent"] for row in samples]
        ),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in samples),
    }


def dynamo_counters() -> dict[str, dict[str, int]]:
    from torch._dynamo.utils import counters

    return {
        str(group): {str(key): value for key, value in entries.items()}
        for group, entries in counters.items()
    }


def main() -> int:
    args = arguments()
    output = contained_path(args.output)
    model_path = contained_path(args.model)
    reference_path = contained_path(args.reference) if args.reference else None
    source_path = contained_path(args.source)
    if not (source_path / "src/soup_cli/__init__.py").is_file():
        raise FileNotFoundError(f"Not a Soup source checkout: {source_path}")
    sys.path.insert(0, str(source_path / "src"))
    affinity = apply_cpu_affinity(args.cpu_affinity)
    import torch
    from rich.console import Console

    from soup_cli.commands.infer import _generate, _load_model

    console = Console()
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires CUDA; refusing to substitute a CPU benchmark")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    messages = [{"role": "user", "content": args.prompt}]
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "variant": args.variant,
        "environment": metadata(torch, model_path, affinity),
        "workload": {
            "model_path": str(model_path),
            "messages": messages,
            "max_new_tokens": args.max_tokens,
            "temperature": 0.0,
            "batch_size": 1,
            "loader": "soup_cli.commands.infer._load_model",
            "request": "soup_cli.commands.infer._generate",
            "quantization_override": None,
            "dtype_override": None,
        },
        "protocol": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "phase_repeats": args.phase_repeats,
            "prefill_phase_repeats": args.prefill_phase_repeats,
            "prefill_phase_warmups": 1 if args.prefill_phase_repeats else 0,
            "telemetry": args.telemetry,
            "primary_timing": "perf_counter with CUDA synchronize before and after request",
            "phases": "Separate requests with top-level forward CUDA events; includes stream gaps",
            "cpu_utilization": "process CPU seconds / elapsed seconds * 100; may exceed 100",
        },
        "samples": [],
        "phases": [],
        "prefill_phases": [],
        "prefill_phase_warmups": [],
        "warmups": [],
        "profiles": {},
    }
    save(output, result)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, tokenizer = _load_model(str(model_path), None, "cuda", is_local=True)
    torch.cuda.synchronize()
    result["load"] = {
        "seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result["model"] = {
        "class": type(model).__name__,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "hf_device_map": {
            str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()
        },
        "parameter_bytes": sum(
            param.numel() * param.element_size() for param in model.parameters()
        ),
        "use_cache": getattr(model.config, "use_cache", None),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "generation_config": model.generation_config.to_dict(),
    }
    console.print(f"Loaded {model_path.name} in {result['load']['seconds']:.2f}s; {args.variant}")
    expected: dict[str, Any] | None = None

    def invoke() -> tuple[str, int]:
        scope = (
            torch.inference_mode()
            if args.variant in ("inference_mode", "inference_mode_mask_callback")
            else contextlib.nullcontext()
        )
        with scope:
            production_kwargs = (
                {"cuda_graphs": True} if args.variant == "production_cuda_graphs" else {}
            )
            return _generate(
                model,
                tokenizer,
                messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
                **production_kwargs,
            )

    def run_request(capture: CaptureGeneration) -> dict[str, Any]:
        nonlocal expected
        capture.reset()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        response, count = invoke()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        sample = {
            "seconds": seconds,
            "tokens": count,
            "tokens_per_second": count / seconds,
            "latency_per_token_ms": seconds * 1000 / count if count else None,
            "cpu_seconds": cpu_seconds,
            "cpu_utilization_percent": cpu_seconds / seconds * 100,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "request_start_perf_counter": wall_start,
        }
        # Copies and comparisons are deliberately AFTER the synchronized timing boundary.
        snapshot = capture.snapshot()
        if count != len(snapshot["output_ids"][0]):
            raise AssertionError("Soup's reported count differs from captured generated token IDs")
        if expected is None:
            expected = snapshot
            result["correctness"] = {
                **snapshot,
                "response_text": response,
                "prompt_tokens": len(snapshot["input_ids"][0]),
                "generated_tokens": count,
                "all_requests_identical": True,
            }
        elif snapshot != expected:
            result["correctness"]["all_requests_identical"] = False
            result["correctness"]["mismatching_snapshot"] = snapshot
            save(output, result)
            raise AssertionError("Repeated requests produced different prompt/output token IDs")
        return sample

    result["experimental_settings"] = {}
    if args.variant == "production_cuda_graphs":
        result["experimental_settings"]["production_generate_cuda_graphs"] = True
    if args.variant == "static" or args.variant in EXPERIMENTAL_GRAPH_VARIANTS:
        model.generation_config.cache_implementation = "static"
        # Static cache alone should isolate allocation/layout, not implicitly
        # opt into Transformers' default compile backend.
        model.generation_config.disable_compile = args.variant == "static"
        result["experimental_settings"].update(
            cache_implementation="static", disable_compile=args.variant == "static"
        )
    if args.variant in EXPERIMENTAL_GRAPH_VARIANTS:
        from transformers import CompileConfig

        backend = "cudagraphs"
        if args.variant == "cudagraphs_static_inputs":
            from static_cudagraph_backend import static_cudagraphs
            from torch._dynamo.backends.registry import register_backend

            backend = "soup_experiment_static_inputs"
            register_backend(static_cudagraphs, name=backend)
        model.generation_config.compile_config = CompileConfig(
            backend=backend, fullgraph=True, dynamic=False
        )
        result["experimental_settings"]["compile_config"] = {
            "backend": backend,
            "fullgraph": True,
            "dynamic": False,
        }
    extra_context = contextlib.nullcontext()
    kwargs_transform = None
    callback_counts = {"requests": 0}
    if args.variant in ("no_padding_mask", "mask_callback", "inference_mode_mask_callback"):
        from soup_cli.utils.vllm import encode_chat_prompt

        validated_inputs = encode_chat_prompt(
            messages, tokenizer, fallback_on_error=False, return_tensors="pt"
        )
    if args.variant == "no_padding_mask":
        from mask_experiment import omit_redundant_qwen2_padding_mask

        extra_context = omit_redundant_qwen2_padding_mask(model, validated_inputs["attention_mask"])
        result["experimental_settings"]["omit_redundant_qwen2_padding_mask"] = True
    elif args.variant in ("mask_callback", "inference_mode_mask_callback"):
        from mask_callback_experiment import no_padding_generation_kwargs

        def kwargs_transform(kwargs: dict[str, Any]) -> dict[str, Any]:
            prepared = no_padding_generation_kwargs(
                model, validated_inputs["attention_mask"], kwargs
            )
            if "custom_generate" not in prepared:
                raise RuntimeError("The experimental mask callback eligibility gate did not pass")
            callback_counts["requests"] += 1
            return prepared

        result["experimental_settings"]["no_padding_custom_generate"] = True
    if args.variant in GRAPH_VARIANTS and args.phase_repeats:
        console.print("Skipping forward-hook phase instrumentation for the compiled variant.")
        result["protocol"]["phase_repeats_requested"] = args.phase_repeats
        result["protocol"]["phase_repeats"] = 0
        result["protocol"]["phases_skip_reason"] = (
            "Python hooks and CUDA event construction can invalidate compiled graph capture"
        )
        args.phase_repeats = 0
    with extra_context as experiment_stats, CaptureGeneration(model, kwargs_transform) as capture:
        for index in range(args.warmups):
            sample = run_request(capture)
            result["warmups"].append(sample)
            console.print(
                f"Warmup {index + 1}/{args.warmups}: {sample['tokens_per_second']:.2f} tok/s"
            )
        telemetry = Telemetry(args.telemetry)
        telemetry.start()
        try:
            for index in range(args.repeats):
                sample = run_request(capture)
                result["samples"].append(sample)
                save(output, result)
                console.print(
                    f"Measured {index + 1}/{args.repeats}: "
                    f"{sample['tokens_per_second']:.2f} tok/s, {sample['tokens']} tokens"
                )
        finally:
            telemetry.stop()
            result["telemetry"] = {
                "enabled": args.telemetry,
                "rows": telemetry.rows,
                "fields": [
                    "timestamp",
                    "gpu_util_percent",
                    "memory_util_percent",
                    "memory_used_mib",
                    "power_w",
                    "power_limit_w",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                    "temperature_c",
                    "pstate",
                ],
            }

        result["summary"] = summarize(result["samples"])
        for kind in ("allocated", "reserved"):
            key = f"peak_{kind}_bytes"
            result["summary"][f"{key}_including_load_and_warmup"] = max(
                row[key] for row in [result["load"], *result["warmups"], *result["samples"]]
            )
        result["status"] = "measurements_complete"
        if args.variant in GRAPH_VARIANTS:
            result["dynamo_counters_after_measured"] = dynamo_counters()
        if args.variant == "cudagraphs_static_inputs":
            from static_cudagraph_backend import COMPILE_RECORDS

            result["backend_compile_records_after_measured"] = list(COMPILE_RECORDS)
        save(output, result)
        for _ in range(args.phase_repeats):
            with ForwardPhases(model, torch) as phases:
                sample = run_request(capture)
            sample.update(phases.summary(sample["request_start_perf_counter"]))
            result["phases"].append(sample)
            save(output, result)

        if args.prefill_phase_repeats:
            from validate_cuda_graphs import _request_phases

            for index in range(args.prefill_phase_repeats + 1):
                capture.reset()
                sample = _request_phases(model, invoke)
                # The helper has synchronized and closed its timing boundary.
                # Capture evidence afterwards, as in the uninstrumented requests.
                snapshot = capture.snapshot()
                if snapshot != expected:
                    result["correctness"]["all_requests_identical"] = False
                    result["correctness"]["mismatching_prefill_phase_snapshot"] = snapshot
                    save(output, result)
                    raise AssertionError("Prefill-instrumented request changed prompt/output IDs")
                count = sample["generated_tokens"]
                if count != len(snapshot["output_ids"][0]):
                    raise AssertionError("Prefill diagnostic token count mismatch")
                sample.update(
                    input_ids=snapshot["input_ids"],
                    output_ids=snapshot["output_ids"],
                    attention_mask=snapshot["attention_mask"],
                    prompt_tokens=len(snapshot["input_ids"][0]),
                    identical_to_measured_requests=True,
                    decode_forward_tokens=max(0, count - 1),
                    pre_prefill_cuda_event_span_ms=sample["request_cuda_event_span_ms"]
                    - sample["prefill_cuda_event_span_ms"]
                    - sample["post_prefill_cuda_event_span_ms"],
                )
                sample["post_prefill_decode_forward_tokens_per_second"] = sample[
                    "decode_forward_tokens"
                ] / (sample["post_prefill_cuda_event_span_ms"] / 1000)
                bucket = "prefill_phase_warmups" if index == 0 else "prefill_phases"
                result[bucket].append(sample)
                save(output, result)
                console.print(
                    f"Prefill phase {'warmup' if index == 0 else index}: "
                    f"{sample['prefill_cuda_event_span_ms']:.3f} ms prefill; "
                    f"{sample['post_prefill_cuda_event_span_ms']:.3f} ms post-prefill"
                )
            result["prefill_phase_summary"] = {
                key: distribution([sample[key] for sample in result["prefill_phases"]])
                for key in (
                    "synchronized_request_wall_ms",
                    "request_cuda_event_span_ms",
                    "pre_prefill_cuda_event_span_ms",
                    "prefill_cuda_event_span_ms",
                    "post_prefill_cuda_event_span_ms",
                    "post_prefill_decode_forward_tokens_per_second",
                )
            }
            result["prefill_phase_summary"]["aggregate_post_prefill_decode_tokens_per_second"] = (
                sum(sample["decode_forward_tokens"] for sample in result["prefill_phases"])
                / (
                    sum(
                        sample["post_prefill_cuda_event_span_ms"]
                        for sample in result["prefill_phases"]
                    )
                    / 1000
                )
            )
            save(output, result)

        if args.trace_only:
            console.print("Tracing a separate complete request without shape/memory aggregation...")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                run_request(capture)
            trace_path = Path(str(output.with_suffix("")) + ".trace.json")
            profiler.export_chrome_trace(str(trace_path))
            result["profiles"]["torch"] = {
                "trace": str(trace_path),
                "trace_only": True,
                "record_shapes": False,
                "profile_memory": False,
                "key_averages_processed": False,
                "cprofile_run": False,
            }
            save(output, result)
        elif args.profile:
            import cProfile
            import io
            import pstats

            prefix = output.with_suffix("")
            console.print("Profiling a separate complete request with torch.profiler...")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                run_request(capture)
            trace_path = Path(str(prefix) + ".trace.json")
            profiler.export_chrome_trace(str(trace_path))
            table_path = Path(str(prefix) + ".torch-profile.txt")
            averages = profiler.key_averages()
            table_path.write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=60)
                + "\n\n"
                + averages.table(sort_by="self_device_time_total", row_limit=60),
                encoding="utf-8",
            )
            profiler_rows = []
            for event in averages:
                profiler_rows.append(
                    {
                        "key": event.key,
                        "count": event.count,
                        "self_cpu_time_total_us": event.self_cpu_time_total,
                        "cpu_time_total_us": event.cpu_time_total,
                        "self_device_time_total_us": getattr(event, "self_device_time_total", None),
                        "device_time_total_us": getattr(event, "device_time_total", None),
                    }
                )
            result["profiles"]["torch"] = {
                "trace": str(trace_path),
                "table": str(table_path),
                "operators": profiler_rows,
            }
            console.print("Profiling another separate complete request with cProfile...")
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
            run_request(capture)
            cpu_profiler.disable()
            profile_path = Path(str(prefix) + ".pstats")
            cpu_profiler.dump_stats(str(profile_path))
            buffer = io.StringIO()
            pstats.Stats(cpu_profiler, stream=buffer).sort_stats("cumulative").print_stats(70)
            text_path = Path(str(prefix) + ".cprofile.txt")
            text_path.write_text(buffer.getvalue(), encoding="utf-8")
            result["profiles"]["cprofile"] = {
                "stats": str(profile_path),
                "table": str(text_path),
            }
        if experiment_stats is not None:
            from dataclasses import asdict

            result["experimental_call_counts"] = asdict(experiment_stats)
        if kwargs_transform is not None:
            result["experimental_call_counts"] = callback_counts
        if args.variant in GRAPH_VARIANTS:
            result["dynamo_counters_after_diagnostics"] = dynamo_counters()
        if args.variant == "cudagraphs_static_inputs":
            result["backend_compile_records_after_diagnostics"] = list(COMPILE_RECORDS)
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        identical_workload = reference["workload"] == result["workload"]
        identical_tokens = all(
            reference["correctness"][key] == result["correctness"][key]
            for key in ("input_ids", "output_ids", "attention_mask")
        )
        same_config_files = (
            reference["environment"]["config_fingerprints"]
            == result["environment"]["config_fingerprints"]
        )
        same_versions = reference["environment"]["versions"] == result["environment"]["versions"]
        reference_affinity = reference["environment"].get("cpu_affinity", {}).get("effective")
        current_affinity = result["environment"]["cpu_affinity"]["effective"]
        same_affinity = (
            reference_affinity == current_affinity
            if reference_affinity is not None and current_affinity is not None
            else None
        )
        result["reference_comparison"] = {
            "path": str(reference_path),
            "identical_workload": identical_workload,
            "identical_tokens": identical_tokens,
            "same_config_files": same_config_files,
            "same_versions": same_versions,
            "same_effective_cpu_affinity": same_affinity,
            "improvement_percent": (
                result["summary"]["aggregate_tokens_per_second"]
                / reference["summary"]["aggregate_tokens_per_second"]
                - 1
            )
            * 100,
        }
        if not all(
            (
                identical_workload,
                identical_tokens,
                same_config_files,
                same_versions,
                same_affinity is not False,
            )
        ):
            result["status"] = "reference_mismatch"
            save(output, result)
            raise AssertionError("Baseline comparison is invalid; see reference_comparison")
    result["status"] = "complete"
    result["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(output, result)
    console.print(
        f"Aggregate: {result['summary']['aggregate_tokens_per_second']:.3f} tok/s; "
        f"peak allocated: {result['summary']['peak_allocated_bytes'] / 1024**3:.3f} GiB"
    )
    console.print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
