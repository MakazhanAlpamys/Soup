"""Bounded native lifecycle evidence, not a domain-quality or CUDA comparison.

From a source checkout with .[mlx] installed:
    python benchmarks/qualify_mlx_pretrain.py --config soup.yaml --report evidence.json

Uses the public config/dispatch, actual local/hub MLX base, corpus files and a
fresh Python process for reload. Intended for tiny qualification models, since
the frozen-base check retains the pre-update arrays. The report stores hashes
and numeric counts, never raw corpus text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def qualify(config_path: Path, report_path: Path) -> None:
    started = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.dispatch import build_trainer
    from soup_cli.utils.mlx import get_chip_info, get_unified_memory_bytes

    cfg = load_config(config_path)
    if (cfg.backend, cfg.task) != ("mlx", "pretrain"):
        raise ValueError("qualification requires backend: mlx / task: pretrain")
    wrapper = build_trainer(cfg)
    wrapper.setup(load_dataset(cfg.data, native_pretrain=True))
    model = wrapper.model
    base = dict(tree_flatten(model.parameters()))
    mx.eval(base)
    probe = [list(wrapper.train_windows[0].tokens[:-1])]
    model.eval()
    base_logits = np.array(model(mx.array(probe)))
    result = wrapper.train()
    live_logits = np.array(model(mx.array(probe)))
    after = dict(tree_flatten(model.parameters()))
    unchanged = 0
    for name, value in base.items():
        parent, leaf = name.rsplit(".", 1)
        mapped = name if name in after else f"{parent}.linear.{leaf}"
        if mapped not in after or not mx.array_equal(value, after[mapped]).item():
            raise AssertionError(f"base tensor changed: {name}")
        unchanged += 1
    adapter_delta = max(float(mx.abs(value).max().item()) for name, value in
                        tree_flatten(model.trainable_parameters()) if name.endswith("lora_b"))
    script = """
import json, sys
import mlx.core as mx
import numpy as np
from soup_cli.utils.mlx_generation import load_native_model
model, tokenizer = load_native_model(sys.argv[1])
print(json.dumps(np.array(model(mx.array(json.loads(sys.argv[2])))).tolist()))
"""
    fresh_started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-c", script, cfg.output, json.dumps(probe)],
                               capture_output=True, text=True, timeout=300, check=True)
    reloaded = np.array(json.loads(completed.stdout.strip().splitlines()[-1]))
    np.testing.assert_allclose(reloaded, live_logits, atol=1e-5, rtol=1e-5)
    base_difference = float(np.abs(live_logits - base_logits).max())
    if adapter_delta <= 1e-6 or base_difference <= 1e-5:
        raise AssertionError("adapter must update and differ meaningfully from its base")
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True,
                         text=True, check=True).stdout.strip()
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((source / "src").rglob("*.py"))}
    meta = json.loads((Path(cfg.output) / "adapter_config.json").read_text())
    evidence = {
        "soup_base_sha": sha, "source_files_sha256": manifest,
        "working_source_tree_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy")},
        "hardware": get_chip_info(), "unified_memory_bytes": get_unified_memory_bytes(),
        "device": str(mx.default_device()), "metal_available": mx.metal.is_available(),
        "training_packages_present": [name for name in ("torch", "trl", "peft", "datasets")
                                      if __import__("importlib.util").util.find_spec(name)],
        "adapter_metadata": meta, "result": result,
        "base_tensor_dtypes": sorted({str(value.dtype) for value in base.values()}),
        "adapter_tensor_dtypes": sorted({str(value.dtype) for _, value in
                                         tree_flatten(model.trainable_parameters())}),
        "unchanged_base_tensors": unchanged, "max_lora_b_change_from_zero": adapter_delta,
        "live_vs_fresh_reload_max_abs": float(np.abs(live_logits - reloaded).max()),
        "live_vs_base_max_abs": base_difference,
        "fresh_process_reload_seconds": time.perf_counter() - fresh_started,
        "end_to_end_seconds": time.perf_counter() - started,
        "limits": "tiny lifecycle fixture; no domain-quality or CUDA parity claim; "
                  "peak MLX allocation excludes model loading; throughput excludes eval/save",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    from rich.console import Console

    Console().print(f"Saved native MLX pretrain qualification: {os.fspath(report_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.config, args.report)
