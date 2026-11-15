"""Run the #2878 preference-loss VRAM grid in isolated CUDA processes.

The source tree must be an exact checkout of the #1141 probe commit (or a
reviewed application of that diff onto a named ``main`` commit).  The schema
gate intentionally remains closed: probe workers set the already-validated
internal flag only for this measurement harness.  No production config or gate
is changed by this script.

Example (one task per thermally monitored stage)::

    python benchmarks/measure_preference_vram_grid.py --task dpo \
      --source-tree ../soup-1141-cuda --model-path <local-qwen-snapshot> \
      --main-sha <main-commit> --output-dir benchmarks/results/gate-2878-rtx3050 \
      --report benchmarks/gate-2878-rtx3050.md

Repeat for ``orpo``, ``simpo`` and ``kto``.  Every probe and real training
step runs in its own fresh worker process.  A row is emitted even on OOM or
other failure; rows shorter than ``max_length`` are refused before measurement.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASKS = ("dpo", "orpo", "simpo", "kto")
SEQUENCE_LENGTHS = (2048, 4096)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_OVERREAD_BAND = (1.125, 1.143)


def parse_nvidia_smi_sample(line: str) -> dict[str, Any]:
    """Parse one `nvidia-smi` CSV sample without units."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise ValueError(f"expected five nvidia-smi fields, got {len(fields)}")
    timestamp, used, free, temperature, utilization = fields
    return {
        "timestamp": timestamp,
        "memory_used_mib": int(used),
        "memory_free_mib": int(free),
        "temperature_c": int(temperature),
        "utilization_percent": int(utilization),
    }


class _NvidiaSmiSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.free,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = result.stdout.splitlines()[0]
            self.samples.append(parse_nvidia_smi_sample(line))
        except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)


def classify_ratio(probe_bytes: int | None, step_bytes: int | None) -> str:
    """Classify probe/step evidence without tuning or weakening the gate."""
    if not probe_bytes or not step_bytes or probe_bytes < 0 or step_bytes < 0:
        return "incomplete"
    ratio = probe_bytes / step_bytes
    if ratio < 1.0:
        return "under-read"
    if SFT_OVERREAD_BAND[0] <= ratio <= SFT_OVERREAD_BAND[1]:
        return "over-read-in-SFT-band"
    if ratio > SFT_OVERREAD_BAND[1]:
        return "over-read-above-SFT-band"
    return "over-read-below-SFT-band"


def _git_sha(source_tree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _gpu_metadata(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    driver = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()[0]
        .strip()
    )
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_bytes": int(props.total_memory),
        "driver_version": driver,
        "torch_cuda": torch.version.cuda,
    }


def _long_rows(task: str, seq_len: int, batch_size: int) -> list[dict[str, Any]]:
    # Deliberately exceed the configured length. Soup/TRL must cap and pad the
    # actual dataset batch; the worker refuses any result that is not exact.
    prompt = "Question:" + " explain" * seq_len
    answer = " answer" * (seq_len * 4)
    rows = []
    for index in range(max(8, batch_size * 2)):
        if task == "kto":
            rows.append({"prompt": prompt, "completion": answer, "label": index % 2 == 0})
        else:
            rows.append(
                {"prompt": prompt, "chosen": answer, "rejected": " alternative" * (seq_len * 4)}
            )
    return rows


def _make_config(model_path: Path, out_dir: Path, task: str, seq_len: int, batch_size: int):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    document = {
        "base": str(model_path),
        "task": task,
        "backend": "transformers",
        "modality": "text",
        "data": {
            "train": "synthetic-in-memory.jsonl",
            "max_length": seq_len,
        },
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "stream_layers": True,
            "stream_vram_probe": False,
            "epochs": 1,
            "logging_steps": 1,
            "save_steps": 1000,
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        },
        "output": str(out_dir),
    }
    return load_config_from_string(yaml.safe_dump(document, sort_keys=False))


def _wrapper_class(task: str):
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.trainer.kto import KTOTrainerWrapper
    from soup_cli.trainer.orpo import ORPOTrainerWrapper
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    return {
        "dpo": DPOTrainerWrapper,
        "orpo": ORPOTrainerWrapper,
        "simpo": SimPOTrainerWrapper,
        "kto": KTOTrainerWrapper,
    }[task]


def _sequence_fields(task: str) -> tuple[str, ...]:
    if task == "kto":
        return ("completion_input_ids", "KL_completion_input_ids")
    return ("input_ids",)


def _assert_full_batch(
    task: str, batch: dict[str, Any], seq_len: int, batch_size: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {"fields": {}, "configured_batch_size": batch_size}
    for key in _sequence_fields(task):
        value = batch.get(key)
        if value is None or getattr(value, "ndim", 0) != 2:
            raise ValueError(f"required full-sequence field {key!r} is absent")
        shape = [int(value.shape[0]), int(value.shape[1])]
        expected_rows = batch_size * (2 if key == "input_ids" else 1)
        if shape != [expected_rows, seq_len]:
            raise ValueError(
                f"under-read refused: {key} realized {shape}, expected [{expected_rows}, {seq_len}]"
            )
        summary["fields"][key] = shape
    if task == "kto":
        if batch_size < 2:
            raise ValueError("KTO acceptance row requires batch_size >= 2")
        if "KL_completion_input_ids" not in batch:
            raise ValueError("KTO acceptance row is missing its KL forward inputs")
    return summary


def _install_forward_shape_hook(model, seen: list[list[int]]):
    def record(_module, _args, kwargs):
        value = kwargs.get("input_ids")
        if value is not None and getattr(value, "ndim", 0) == 2:
            seen.append([int(value.shape[0]), int(value.shape[1])])

    return model.register_forward_pre_hook(record, with_kwargs=True)


def _worker(
    *,
    task: str,
    seq_len: int,
    mode: str,
    source_tree: Path,
    model_path: Path,
    main_sha: str,
    probe_sha: str,
    tree_sha: str,
    output_file: Path,
) -> int:
    row: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "task": task,
        "sequence_length": seq_len,
        "mode": mode,
        "status": "failed",
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "probe_commit": probe_sha,
        "main_commit": main_sha,
        "measurement_tree": tree_sha,
        "versions": _versions(),
        "probe_peak_bytes": None,
        "real_step_peak_bytes": None,
        "predicted_bytes": None,
        "available_bytes": None,
        "reserved_peak_bytes": None,
        "probe_seconds": None,
        "forward_shapes": [],
        "measurement_free_bytes_before": None,
        "measurement_total_bytes": None,
        "nvidia_smi_samples": [],
        "nvidia_smi_errors": [],
    }
    temp_dir = Path(tempfile.mkdtemp(prefix=f"soup-2878-{task}-{seq_len}-"))
    wrapper = None
    original_budget = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this grid requires a CUDA GPU")
        row["hardware"] = _gpu_metadata(torch)
        batch_size = 2 if task == "kto" else 1
        row["batch_size"] = batch_size
        os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(temp_dir / "stream-cache")

        cfg = _make_config(model_path, temp_dir / "out", task, seq_len, batch_size)
        # This harness measures a proposed internal path while production schema
        # keeps preference probes disabled until the grid is accepted.
        if mode == "probe":
            object.__setattr__(cfg.training, "stream_vram_probe", True)

        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from soup_cli.utils import layer_stream_runtime

        original_budget = StreamingSetupMixin._stream_budget_lines
        captured: dict[str, Any] = {}

        def capture_budget(self, conf, train_conf, *args, **kwargs):
            prior = train_conf.stream_vram_probe
            object.__setattr__(train_conf, "stream_vram_probe", True)
            try:
                lines, plan = original_budget(self, conf, train_conf, *args, **kwargs)
            finally:
                object.__setattr__(train_conf, "stream_vram_probe", prior)
            if plan is not None:
                captured["predicted_bytes"] = plan.predicted_bytes
                captured["available_bytes"] = plan.available_bytes
            return lines, plan if mode == "probe" else None

        StreamingSetupMixin._stream_budget_lines = capture_budget
        wrapper = _wrapper_class(task)(cfg, device="cuda")
        rows = _long_rows(task, seq_len, batch_size)
        forward_shapes: list[list[int]] = []

        if mode == "probe":
            original_measure = layer_stream_runtime.measure_loss_step_peak_bytes
            original_finish = wrapper._finish_stream_vram_probe

            def capture_measure(model, *, step, rows, seq_len, device):
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                sampler = _NvidiaSmiSampler()
                sampler.start()
                handle = _install_forward_shape_hook(model, forward_shapes)
                try:
                    measured = original_measure(
                        model, step=step, rows=rows, seq_len=seq_len, device=device
                    )
                finally:
                    handle.remove()
                    sampler.stop()
                    captured["measurement_free_bytes_before"] = int(free_bytes)
                    captured["measurement_total_bytes"] = int(total_bytes)
                    captured["nvidia_smi_samples"] = sampler.samples
                    captured["nvidia_smi_errors"] = sampler.errors
                captured["probe_peak_bytes"] = measured.peak_bytes
                captured["reserved_peak_bytes"] = measured.reserved_bytes
                captured["probe_seconds"] = measured.seconds
                captured["realized_batch"] = {"model_rows": rows, "sequence_length": seq_len}
                return measured

            def capture_finish(peak, plan):
                captured["predicted_bytes"] = plan.predicted_bytes
                captured["available_bytes"] = plan.available_bytes
                captured["probe_peak_bytes"] = peak.peak_bytes
                captured["reserved_peak_bytes"] = peak.reserved_bytes
                captured["probe_seconds"] = peak.seconds

            layer_stream_runtime.measure_loss_step_peak_bytes = capture_measure
            wrapper._finish_stream_vram_probe = capture_finish
            try:
                wrapper.setup({"train": rows})
            finally:
                layer_stream_runtime.measure_loss_step_peak_bytes = original_measure
                wrapper._finish_stream_vram_probe = original_finish
            if "probe_peak_bytes" not in captured:
                raise RuntimeError("probe path did not produce a measured loss-step peak")
            row.update(captured)
            row["status"] = "success"

        else:
            wrapper.setup({"train": rows})
            trainer = wrapper.trainer
            batch = next(iter(trainer.get_train_dataloader()))
            row["realized_batch"] = _assert_full_batch(task, batch, seq_len, batch_size)
            original_step = trainer.training_step
            calls = 0
            measured: dict[str, Any] = {}

            def capture_training_step(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    measured["baseline_allocated_bytes"] = int(torch.cuda.memory_allocated())
                    measured["baseline_reserved_bytes"] = int(torch.cuda.memory_reserved())
                    sampler = _NvidiaSmiSampler()
                    sampler.start()
                    started = time.perf_counter()
                    handle = _install_forward_shape_hook(trainer.model, forward_shapes)
                    try:
                        value = original_step(*args, **kwargs)
                        torch.cuda.synchronize()
                    finally:
                        handle.remove()
                        sampler.stop()
                        measured["measurement_free_bytes_before"] = int(free_bytes)
                        measured["measurement_total_bytes"] = int(total_bytes)
                        measured["nvidia_smi_samples"] = sampler.samples
                        measured["nvidia_smi_errors"] = sampler.errors
                    measured["real_step_seconds"] = time.perf_counter() - started
                    measured["real_step_peak_bytes"] = int(torch.cuda.max_memory_allocated())
                    measured["reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
                    return value
                return original_step(*args, **kwargs)

            trainer.training_step = capture_training_step
            trainer.args.max_steps = 1
            trainer.args.num_train_epochs = 1
            trainer.args.save_steps = 1000
            trainer.train()
            if calls != 1:
                raise RuntimeError(f"expected exactly one training step, observed {calls}")
            row.update(captured)
            row.update(measured)
            row["forward_shapes"] = forward_shapes
            row["status"] = "success"

        row["forward_shapes"] = forward_shapes
        row["verdict"] = "pending-paired-process-row"
        return_code = 0
    except Exception as exc:
        is_oom = "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"
        row["status"] = "oom" if is_oom else "failed"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()[-5000:]
        row["forward_shapes"] = forward_shapes if "forward_shapes" in locals() else []
        return_code = 0  # Failures are recorded rows; the parent continues the grid.
    finally:
        if original_budget is not None:
            from soup_cli.trainer.stream_setup import StreamingSetupMixin

            StreamingSetupMixin._stream_budget_lines = original_budget
        if wrapper is not None:
            try:
                wrapper._close_stream_runtime()
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        _write_json(output_file, row)
    return return_code


def _row_path(output_dir: Path, task: str, seq_len: int, mode: str) -> Path:
    return output_dir / "rows" / f"{task}-seq{seq_len}-{mode}.json"


def _paired_verdict(
    probe: dict[str, Any] | None, step: dict[str, Any] | None
) -> tuple[str, float | None]:
    if not probe or not step or probe.get("status") != "success" or step.get("status") != "success":
        return "incomplete", None
    probe_peak = probe.get("probe_peak_bytes")
    real_peak = step.get("real_step_peak_bytes")
    if not probe_peak or not real_peak:
        return "incomplete", None
    return classify_ratio(int(probe_peak), int(real_peak)), float(probe_peak / real_peak)


def _render_report(
    output_dir: Path,
    report_path: Path,
    *,
    main_sha: str,
    probe_sha: str,
    tree_sha: str,
) -> None:
    rows_dir = output_dir / "rows"
    rows = {}
    for path in sorted(rows_dir.glob("*.json")) if rows_dir.exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows[(value.get("task"), value.get("sequence_length"), value.get("mode"))] = value

    lines = [
        "# #2878 preference VRAM grid — RTX 3050 Laptop GPU",
        "",
        f"- Main commit: `{main_sha}`",
        f"- #1141 code under test: `{probe_sha}`",
        f"- Applied measurement tree on `main`: `{tree_sha}`",
        f"- Base model: `{MODEL_ID}` at revision `7ae557604adf67be50417f59c2c2f167def9a775`",
        "- Required grid: DPO, ORPO, SimPO, KTO at 2048 and 4096 tokens; KTO batch size 2.",
        "- Probe and real steps run in fresh processes; short batches are refused.",
        "- Any under-read keeps that loss refused. No thresholds are tuned here.",
        "",
        "| Loss | Seq | Batch | Probe MiB | Step MiB | Ratio | Probe s | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    verdicts: dict[str, list[str]] = {task: [] for task in TASKS}
    for task in TASKS:
        for seq_len in SEQUENCE_LENGTHS:
            probe = rows.get((task, seq_len, "probe"))
            step = rows.get((task, seq_len, "step"))
            verdict, ratio = _paired_verdict(probe, step)
            verdicts[task].append(verdict)
            batch = 2 if task == "kto" else 1
            probe_peak = None if not probe else probe.get("probe_peak_bytes")
            real_peak = None if not step else step.get("real_step_peak_bytes")
            cost = None if not probe else probe.get("probe_seconds")
            probe_mib = f"{probe_peak / 1048576:.2f}" if probe_peak else "—"
            step_mib = f"{real_peak / 1048576:.2f}" if real_peak else "—"
            ratio_text = f"{ratio:.3f}" if ratio is not None else "—"
            cost_text = f"{cost:.2f}" if cost is not None else "—"
            lines.append(
                f"| {task.upper()} | {seq_len} | {batch} | "
                f"{probe_mib} | {step_mib} | {ratio_text} | {cost_text} | {verdict} |"
            )
    lines.extend(["", "## Per-loss result", ""])
    for task in TASKS:
        states = verdicts[task]
        if "under-read" in states:
            statement = "**Under-read observed; keep this loss refused.**"
        elif all(state.startswith("over-read") for state in states):
            statement = (
                "No under-read in completed rows; review the margin before changing the gate."
            )
        else:
            statement = "Grid incomplete; no conclusion about the gate."
        lines.append(f"- **{task.upper()}:** {statement}")
    lines.extend(
        [
            "",
            "## Raw rows",
            "",
            "Each `rows/*.json` records allocated/reserved peaks, prediction, available VRAM, "
            "free VRAM before measurement, timestamped nvidia-smi samples, batch and forward "
            "shapes, duration, device, driver and package versions.",
            "",
            "## GPU regression rows from #1141",
            "",
            "The six-file `pytest -m gpu` run is recorded separately in "
            "`benchmarks/gpu-test-runs.md`; all 21 rows must pass.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        output_dir / "summary.json",
        {
            "main_commit": main_sha,
            "probe_commit": probe_sha,
            "rows": len(rows),
            "verdicts": verdicts,
        },
    )


def _run_task(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    source_tree = args.source_tree.resolve()
    probe_sha = args.probe_sha or _git_sha(source_tree)
    tree_sha = args.tree_sha or _git_sha(source_tree)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_tree / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = Path(__file__).resolve()
    for seq_len in SEQUENCE_LENGTHS:
        for mode in ("probe", "step"):
            row_file = _row_path(output_dir, args.task, seq_len, mode)
            log_file = output_dir / "logs" / f"{args.task}-seq{seq_len}-{mode}.log"
            row_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(script),
                "--worker-mode",
                mode,
                "--task",
                args.task,
                "--seq-len",
                str(seq_len),
                "--source-tree",
                str(source_tree),
                "--model-path",
                str(args.model_path.resolve()),
                "--main-sha",
                args.main_sha,
                "--probe-sha",
                probe_sha,
                "--tree-sha",
                tree_sha,
                "--output-file",
                str(row_file),
            ]
            with log_file.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        command,
                        env=env,
                        cwd=source_tree,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.worker_timeout,
                    )
                except subprocess.TimeoutExpired:
                    result = None
            if result is None or result.returncode != 0 or not row_file.exists():
                detail = (
                    "worker timed out" if result is None else f"worker exit {result.returncode}"
                )
                _write_json(
                    row_file,
                    {
                        "schema_version": 1,
                        "task": args.task,
                        "sequence_length": seq_len,
                        "mode": mode,
                        "status": "failed",
                        "error": detail,
                        "probe_commit": probe_sha,
                        "main_commit": args.main_sha,
                    },
                )
    _render_report(
        output_dir,
        report_path,
        main_sha=args.main_sha,
        probe_sha=probe_sha,
        tree_sha=tree_sha,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--probe-sha")
    parser.add_argument("--tree-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=int, default=3600)
    parser.add_argument("--worker-mode", choices=("probe", "step"))
    parser.add_argument("--seq-len", type=int, choices=SEQUENCE_LENGTHS)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if args.worker_mode:
        if args.seq_len is None or args.output_file is None:
            parser.error("worker mode requires --seq-len and --output-file")
        source_tree = args.source_tree.resolve()
        sys.path.insert(0, str(source_tree / "src"))
        return _worker(
            task=args.task,
            seq_len=args.seq_len,
            mode=args.worker_mode,
            source_tree=source_tree,
            model_path=args.model_path.resolve(),
            main_sha=args.main_sha,
            probe_sha=args.probe_sha or _git_sha(source_tree),
            tree_sha=args.tree_sha or _git_sha(source_tree),
            output_file=args.output_file.resolve(),
        )
    return _run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
