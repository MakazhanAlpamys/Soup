#!/usr/bin/env python3
"""Streaming arms S0..S8 and SB: write the soup.yaml, run ``soup train``, write the run json.

``soup train`` is run unmodified (``soup train --config <yaml> --yes``,
src/soup_cli/commands/train.py:681-690 and :819-826). Soup itself writes no train_runtime and
no token count to disk (its ``duration_secs``, src/soup_cli/trainer/sft.py:2543 and 2604, is
the wall time of the whole ``train()`` call up to the end of ``trainer.train()``), so the same
passive recorder that watches ``hf_baseline.py`` is loaded into the training process through
``sitecustomize.py`` (``SOUP_BENCH_TRACE_OUT``); it reads the trainer's own summary log
(``train_runtime``), TRL's cumulative ``num_tokens``, per-step timestamps and the torch peak
reserved memory. Soup's tracker database (``SOUP_DB_PATH`` in the work directory) is read as
a cross-check, and the console log is parsed for the stream plan (tier, store size, pinned or
pageable) and the ``stream_recompute: auto`` planner line.

Arm table (protocol section 2). Config keys are under ``training:``.

    S0  Soup at commit 2c9a078 (separate venv, ``--soup-bin``); S1's yaml with the keys that
        did not exist then dropped (listed in the json)
    S1  main, defaults
    S2  stream_recompute: auto          S3  stream_recompute: off
    S4  stream_pin: false               S5  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    S6  stream_buffers: 3               S7  stream_loss_chunk: 128
    S8  fast_lora: true + lora.dropout: 0   (informational only)
    SB  S1 + the single changes the derivation rule selected (``--overrides-json``)

Usage::

    python soup_arm.py --arm S2 --shape A --workdir run/ --out run/rep1.raw.json
    python soup_arm.py --arm S1 --shape A --workdir run/ --out run/x.json --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_trace import collect_versions, derive_metrics  # noqa: E402
from ds_common import (  # noqa: E402
    HERE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR,
    MAX_GRAD_NORM,
    MODEL_ID,
    MODEL_REVISION,
    ROWS_PER_SHAPE,
    RUN_SCHEMA,
    SEED,
    SHAPES,
    WEIGHT_DECAY,
    atomic_write_json,
    read_json,
    rows_path,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_OOM = 3

STREAM_ARMS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SB")

#: Single-change definitions of the ablation arms (protocol section 2): what each adds to S1.
ARM_CHANGES: dict[str, dict[str, Any]] = {
    "S2": {"training": {"stream_recompute": "auto"}},
    "S3": {"training": {"stream_recompute": "off"}},
    "S4": {"training": {"stream_pin": False}},
    "S5": {"env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}},
    "S6": {"training": {"stream_buffers": 3}},
    "S7": {"training": {"stream_loss_chunk": 128}},
    "S8": {"training": {"fast_lora": True, "lora": {"dropout": 0.0}}},
}

#: ``training:`` keys that do not exist at Soup 2c9a078 (found by loading that tree's
#: ``TrainingConfig.model_fields``): an unknown key is ignored silently there, so S0 drops
#: them from the yaml instead and lists them in the json. ``seed`` is in every generated
#: yaml; the others only appear when an arm sets them.
S0_UNSUPPORTED_TRAINING_KEYS = (
    "seed",
    "stream_pin",
    "stream_recompute",
    "stream_loss_chunk",
    "fast_lora",
    "stream_vram_probe",
)

_OOM_PATTERN = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX = re.compile("[" + chr(0x2500) + "-" + chr(0x257F) + "]")


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_config(
    arm: str,
    shape: str,
    model: str,
    data_file: str,
    output_dir: str,
    overrides: dict[str, Any] | None = None,
    epochs: int | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """(soup.yaml as a dict, extra child environment, notes) for one streaming arm.

    ``overrides`` is the SB derivation (``{"training": {...}, "env": {...}}``) or the
    Shape C rerun (``{"training": {"stream_recompute": "on"}}``); it is merged on top of the
    arm's own changes.
    """
    if arm not in STREAM_ARMS:
        raise ValueError(f"not a streaming arm: {arm}")
    spec = SHAPES[shape]
    config: dict[str, Any] = {
        "base": model,
        "task": "sft",
        "data": {
            "train": data_file,
            "format": "alpaca",
            "val_split": 0,
            "max_length": spec["max_length"],
        },
        "training": {
            "epochs": epochs or spec["epochs"],
            "lr": LR,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
            "optimizer": "adamw_torch",
            "scheduler": "constant",
            "warmup_ratio": 0.0,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "logging_steps": 1,
            "save_steps": 1000000,
            "seed": SEED,
            "stream_layers": True,
        },
        "output": output_dir,
    }
    env: dict[str, str] = {}
    notes: dict[str, Any] = {"dropped_keys": [], "applied_changes": []}

    if arm in ARM_CHANGES:
        change = ARM_CHANGES[arm]
        config = deep_merge(config, {"training": change.get("training", {})})
        env.update(change.get("env", {}))
        notes["applied_changes"].append(arm)
    if overrides:
        config = deep_merge(config, {"training": overrides.get("training", {})})
        env.update(overrides.get("env", {}))
        notes["applied_changes"].append("overrides")

    if arm == "S0":
        for key in S0_UNSUPPORTED_TRAINING_KEYS:
            if key in config["training"]:
                del config["training"][key]
                notes["dropped_keys"].append(f"training.{key}")
        notes["unavailable_keys_at_2c9a078"] = [
            f"training.{key}" for key in S0_UNSUPPORTED_TRAINING_KEYS
        ]
    return config, env, notes


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    plain = re.fullmatch(r"[A-Za-z0-9_./\:-]+", text) and text.lower() not in (
        "true",
        "false",
        "null",
        "yes",
        "no",
        "on",
        "off",
    )
    try:
        float(text)
        plain = False
    except ValueError:
        pass
    return text if plain else '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dump_plain(data: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.extend(_dump_plain(value, indent + 2))
        else:
            lines.append(f"{' ' * indent}{key}: {_scalar(value)}")
    return lines


def dump_yaml(config: dict[str, Any]) -> str:
    """The soup.yaml text. PyYAML when installed (it is a Soup dependency), else a plain writer
    for the nested scalar mappings this harness generates."""
    try:
        import yaml
    except ImportError:
        return "\n".join(_dump_plain(config)) + "\n"
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def soup_command(soup_bin: str) -> list[str]:
    """The ``soup`` executable, or the module form when no executable is on PATH."""
    if soup_bin != "soup" or shutil.which("soup"):
        return [soup_bin]
    return [sys.executable, "-m", "soup_cli.cli"]


def child_environment(
    extra: dict[str, str], trace_out: str, db_path: str, cap_gb: float | None
) -> dict[str, str]:
    env = dict(os.environ)
    # An inherited value would leak into arms that must not have one: S1 is "whatever soup
    # train exports itself", S5 is the explicit opposite. Only the arm's own value is set.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("SOUP_BENCH_VRAM_CAP_GB", None)
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["SOUP_BENCH_TRACE_OUT"] = trace_out
    env["SOUP_DB_PATH"] = db_path
    if cap_gb is not None:
        env["SOUP_BENCH_VRAM_CAP_GB"] = f"{cap_gb:g}"
    # A wide, colourless console: the stream panel is parsed from the log.
    env["COLUMNS"] = "250"
    env["NO_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def clean_log(text: str) -> str:
    return _BOX.sub(" ", _ANSI.sub("", text))


def parse_soup_log(text: str) -> dict[str, Any]:
    """Facts Soup prints: stream tier, store size and pinning, the auto-recompute planner."""
    flat = re.sub(r"\s+", " ", clean_log(text))
    info: dict[str, Any] = {}
    match = re.search(r"arch (\S+), tier (\w+)", flat)
    if match:
        info["arch"], info["tier"] = match.group(1), match.group(2)
    match = re.search(r"base store ([\d.]+) GB across (\d+) layers \((pinned|pageable)\)", flat)
    if match:
        info["store_gb"] = float(match.group(1))
        info["store_layers"] = int(match.group(2))
        info["store_pinned"] = match.group(3) == "pinned"
    match = re.search(r"the last (\d+) of (\d+) streamed layers keep their activations", flat)
    if match:
        info["recompute_auto_kept_layers"] = int(match.group(1))
        info["recompute_auto_total_layers"] = int(match.group(2))
    match = re.search(r"expandable_segments allocator hint not enabled: (.{0,160}?)(?: \[|$)", flat)
    if match:
        info["expandable_segments_not_enabled"] = match.group(1).strip()
    info["oom_in_log"] = bool(_OOM_PATTERN.search(text))
    info["vram_cap_applied_in_log"] = "VRAM cap applied" in text
    return info


def read_soup_db(db_path: str) -> dict[str, Any] | None:
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT run_id, status, initial_loss, final_loss, total_steps, duration_secs "
            "FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE run_id = ?", (row["run_id"],)
        ).fetchone()[0]
        conn.close()
        return {**{key: row[key] for key in row.keys()}, "metric_rows": count}
    except sqlite3.Error:
        return None


def read_adapter_targets(output_dir: str) -> list[str] | None:
    path = Path(output_dir) / "adapter_config.json"
    if not path.exists():
        return None
    targets = read_json(path).get("target_modules")
    return sorted(targets) if isinstance(targets, list) else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True, choices=STREAM_ARMS)
    parser.add_argument("--shape", required=True, choices=sorted(SHAPES))
    parser.add_argument("--workdir", required=True, help="soup.yaml, output, log and db go here")
    parser.add_argument("--out", required=True, help="result json path")
    parser.add_argument("--model", default=MODEL_ID, help="local model directory (or repo id)")
    parser.add_argument("--soup-bin", default="soup")
    parser.add_argument("--overrides-json", default=None, help="SB derivation / rerun overrides")
    parser.add_argument("--cap-gb", type=float, default=None)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override (rehearsals only)")
    parser.add_argument("--dry-run", action="store_true", help="write the yaml, print, run nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rows_copy = workdir / "rows.jsonl"
    shutil.copyfile(rows_path(args.shape), rows_copy)

    overrides = read_json(args.overrides_json) if args.overrides_json else None
    config, extra_env, notes = build_config(
        args.arm,
        args.shape,
        args.model,
        "rows.jsonl",
        "./output",
        overrides=overrides,
        epochs=args.epochs,
    )
    yaml_text = dump_yaml(config)
    yaml_path = workdir / "soup.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    command = soup_command(args.soup_bin) + ["train", "--config", "soup.yaml", "--yes"]

    out_path = Path(args.out)
    trace_path = str(out_path.with_suffix(".trace.json"))
    db_path = str(workdir / "experiments.db")
    env = child_environment(extra_env, trace_path, db_path, args.cap_gb)

    print(f"[soup_arm] arm {args.arm} shape {args.shape} config {yaml_path}")
    print(f"[soup_arm] cwd {workdir}")
    print(f"[soup_arm] command {' '.join(command)}")
    if extra_env:
        print(f"[soup_arm] extra env {extra_env}")
    if args.dry_run:
        return EXIT_OK

    log_path = out_path.with_suffix(".log")
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.run(
            command, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    wall = time.time() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_info = parse_soup_log(log_text)

    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "kind": "soup",
        "arm": args.arm,
        "shape": args.shape,
        "cap_gb": args.cap_gb,
        "rep": args.rep,
        "argv": sys.argv,
        "command": command,
        "model": {"id": args.model, "revision": MODEL_REVISION},
        "wall_seconds_child": wall,
        "exit_code": process.returncode,
        "soup": {
            "yaml_sha256": sha256_text(yaml_text),
            "yaml": yaml_text,
            "applied_changes": notes["applied_changes"],
            "dropped_keys": notes["dropped_keys"],
            "unavailable_keys_at_2c9a078": notes.get("unavailable_keys_at_2c9a078"),
            "extra_env": extra_env,
            "stream_log": log_info,
            "tracker_db": read_soup_db(db_path),
            "adapter_target_modules": read_adapter_targets(str(workdir / "output")),
            "expected_target_modules": sorted(LORA_TARGET_MODULES),
        },
        "error": None,
    }
    soup_info = result["soup"]
    soup_info["target_modules_match"] = (
        soup_info["adapter_target_modules"] == soup_info["expected_target_modules"]
        if soup_info["adapter_target_modules"] is not None
        else None
    )

    raw_trace = read_json(trace_path) if os.path.exists(trace_path) else None
    if raw_trace is not None:
        expected_tokens = (
            SHAPES[args.shape]["target_tokens"]
            * (args.epochs or SHAPES[args.shape]["epochs"])
            * ROWS_PER_SHAPE
        )
        result.update(derive_metrics(raw_trace, expected_total_tokens=expected_tokens))
        result["versions"] = raw_trace.get("versions")
        result["trace_args"] = raw_trace.get("args")
        result["pytorch_cuda_alloc_conf"] = raw_trace.get("pytorch_cuda_alloc_conf")
        counted = result.get("tokens_trained")
        result["tokens_check"] = {
            "trainer_or_counted": counted,
            "expected": expected_tokens,
            "equal": counted == expected_tokens,
            "source": result.get("tokens_source"),
        }
    else:
        result["versions"] = collect_versions()

    if process.returncode == 0 and raw_trace is not None and raw_trace.get("complete"):
        result["status"] = "ok"
        code = EXIT_OK
    elif log_info["oom_in_log"]:
        result["status"] = "oom"
        result["error"] = "out of memory (see the log)"
        code = EXIT_OOM
    else:
        result["status"] = "failed"
        tail = "\n".join(clean_log(log_text).splitlines()[-25:])
        result["error"] = f"soup train exited {process.returncode}; log tail:\n{tail}"[-3000:]
        code = EXIT_ERROR
    atomic_write_json(out_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
