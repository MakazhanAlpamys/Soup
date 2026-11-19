"""Run the frozen #2050 correctness/profile gates and four paired comparisons.

Example: python benchmarks/harness/issue2050_campaign.py --root D:/Soup-2050-v1
  --gpu GPU-... . The root contains source/, model/, and model-manifest.json.
The runner refuses an existing output/lock and never retries a failed process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

FILES = (
    "benchmarks/harness/issue2050_campaign.py",
    "benchmarks/harness/issue2050_selective_checkpoint.py",
    "benchmarks/harness/stream_selective_checkpoint.py",
    "benchmarks/harness/stream_ab_pairs.py",
    "benchmarks/harness/issue361_nf4_throughput.py",
    "benchmarks/harness/source_provenance.py",
    "benchmarks/protocol-2050-selective-checkpoint.md",
    "tests/test_stream_selective_checkpoint_harness.py",
    "src/soup_cli/utils/layer_stream_runtime.py",
    "src/soup_cli/utils/layer_stream.py",
    "src/soup_cli/utils/layer_shard.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    tree = root / "source"
    out = root / "evidence-v1"
    out.mkdir(exist_ok=False)
    lock = root / "campaign-v1.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    started = time.monotonic()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=str(tree / "src"),
        PYTHONUNBUFFERED="1",
    )
    console = Console()
    files = {name: digest(tree / name) for name in FILES}
    model_manifest = json.loads((root / "model-manifest.json").read_text())
    for item in model_manifest["files"]:
        if digest(root / "model" / item["file"]) != item["sha256"]:
            raise RuntimeError("model manifest mismatch")
    freeze: dict[str, Any] = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_uuid": args.gpu,
        "files": files,
        "model": model_manifest,
        "base": subprocess.check_output(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "environment": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "gpu_inventory": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    status_path = out / "status.json"
    ledger = []

    def status(phase: str, **more: Any) -> None:
        status_path.write_text(
            json.dumps({"phase": phase, "elapsed_s": time.monotonic() - started, **more}, indent=2),
            encoding="utf-8",
        )

    def run(tag: str, command: list[str]) -> None:
        if time.monotonic() - started > 6 * 3600:
            raise RuntimeError("six-hour campaign budget exhausted")
        if any(digest(tree / name) != value for name, value in files.items()):
            raise RuntimeError("frozen source changed")
        status("running", current=tag)
        (out / f"{tag}-command.json").write_text(json.dumps(command), encoding="utf-8")
        with (out / f"{tag}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=tree,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=min(
                    1800 if "pairs" not in tag else 6 * 3600,
                    6 * 3600 - (time.monotonic() - started),
                ),
            )
        ledger.append({"tag": tag, "rc": result.returncode})
        (out / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        console.print(f"{tag}: exit {result.returncode}")
        if result.returncode:
            raise RuntimeError(f"process failed: {tag}")

    harness = str(tree / "benchmarks/harness/issue2050_selective_checkpoint.py")
    shards = str(tree / ".local2050/mistral-nf4-shards")
    try:
        # Finish all correctness gates before any timing result exists.
        for seq in (512, 2048):
            reference = None
            for mode in (
                "full",
                "full-repeat",
                "qkv",
                "qkvo",
                "profile-full",
                "profile-qkv",
                "profile-qkvo",
            ):
                policy = mode.removeprefix("profile-") if mode.startswith("profile-") else mode
                policy = "full" if policy == "full-repeat" else policy
                tag = f"s{seq}-{mode}"
                command = [
                    sys.executable,
                    harness,
                    "--weights",
                    str(root / "model"),
                    "--shards",
                    shards,
                    "--policy",
                    policy,
                    "--seq",
                    str(seq),
                    "--audit",
                    str(out / f"{tag}-audit.json"),
                    "--json",
                    str(out / f"{tag}-row.json"),
                    "--warmup",
                    "0",
                    "--steps",
                    "2",
                    "--no-ceiling",
                ]
                if mode.startswith("profile-"):
                    command.append("--profile")
                run(tag, command)
                receipt = json.loads((out / f"{tag}-audit.json").read_text())
                if receipt["status"] != "ok" or len(receipt["steps"]) != 2:
                    raise RuntimeError(f"incomplete correctness receipt: {tag}")
                if reference is None:
                    reference = receipt["steps"]
                elif reference != receipt["steps"]:
                    raise RuntimeError(f"bit-exact gate failed: {tag}")
            (out / f"s{seq}-exactness.json").write_text(
                json.dumps(
                    {
                        "verified": True,
                        "processes": 7,
                        "updates_each": 2,
                        "checked": [
                            "logits",
                            "loss",
                            "all gradients",
                            "all updated LoRA parameters",
                        ],
                    }
                ),
                encoding="utf-8",
            )
        for seq in (512, 2048):
            for policy in ("qkv", "qkvo"):
                pair_out = out / f"s{seq}-{policy}-pairs"
                run(
                    f"s{seq}-{policy}-pairs",
                    [
                        sys.executable,
                        str(tree / "benchmarks/harness/stream_ab_pairs.py"),
                        "--tree",
                        str(tree),
                        "--weights",
                        str(root / "model"),
                        "--shards",
                        shards,
                        "--out",
                        str(pair_out),
                        "--pairs",
                        "5",
                        "--label",
                        f"2050-s{seq}-{policy}",
                        "--harness",
                        "issue2050_selective_checkpoint.py",
                        "--a-args=--policy full",
                        f"--b-args=--policy {policy}",
                        f"--common=--seq {seq} --warmup 10 --steps 50 --no-ceiling",
                    ],
                )
        status("completed", ledger_entries=len(ledger))
        return 0
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
