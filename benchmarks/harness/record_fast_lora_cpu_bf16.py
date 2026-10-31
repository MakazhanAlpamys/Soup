"""Observe #2187's existing CPU test without changing its tensors or assertions.

Run from the repository root, with no coverage/debugger trace active:
  PYTHONPATH=benchmarks/harness:src:. SOUP_BF16_CAPTURE=run.json \
    python -m pytest -p record_fast_lora_cpu_bf16 -o addopts='' -m 'not gpu' \
    tests/test_issue837_fast_lora_mlp.py -k twice_pefts -q

Set ATEN_CPU_CAPABILITY before starting Python to select a CPU instruction path.
Windows PYTHONPATH uses semicolons. There are no model downloads or GPU operations.
The pytest exit status remains the original criterion's result, including failure.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import FrameType
from typing import Any

TARGET = "test_bf16_mlp_stays_within_twice_pefts_error_against_float64"
ROWS: list[dict[str, Any]] = []
_TRACE_ACTIVE = False


def _trace(frame: FrameType, event: str, _arg: object) -> Any:
    if event == "call":
        return _trace if frame.f_code.co_name == TARGET else None
    if frame.f_code.co_name != TARGET or event != "return":
        return _trace
    values = frame.f_locals
    if not all(key in values for key in ("fast", "reference", "oracle")):
        return _trace
    import torch

    names = ["output", "input_gradient"] + [
        name for name, _ in sorted(values["model"].named_parameters()) if "lora_" in name
    ]
    measurements = []
    for name, got, ref, exact in zip(
        names, values["fast"], values["reference"], values["oracle"], strict=True
    ):
        fast_error = (got - exact).abs().max().item()
        peft_error = (ref - exact).abs().max().item()
        measurements.append({
            "name": name, "shape": list(got.shape),
            "fast_error": fast_error, "peft_error": peft_error,
            "ratio": fast_error / peft_error if peft_error else None,
            "within_2x": math.isfinite(fast_error) and math.isfinite(peft_error)
                         and fast_error <= 2 * peft_error + 1e-8,
        })
    ROWS.append({
        "seed": values["seed"], "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "fixture_hashes": {
            name: hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8)
                                 .numpy().tobytes()).hexdigest()
            for name, tensor in [("x0", values["x0"]),
                                 *sorted(values["model"].named_parameters())]
        },
        "measurements": measurements,
        "output_within_1_05x": measurements[0]["fast_error"]
                              <= 1.05 * measurements[0]["peft_error"] + 1e-8,
    })
    return _trace


def pytest_sessionstart(session: Any) -> None:
    global _TRACE_ACTIVE
    if sys.gettrace() is not None:
        raise RuntimeError("Run this recorder without coverage or another tracing tool")
    if not os.environ.get("SOUP_BF16_CAPTURE"):
        raise RuntimeError("SOUP_BF16_CAPTURE must name the output JSON file")
    sys.settrace(_trace)
    _TRACE_ACTIVE = True


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    global _TRACE_ACTIVE
    if not _TRACE_ACTIVE:
        return
    sys.settrace(None)
    _TRACE_ACTIVE = False
    import torch

    output = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "requested_capability": os.environ.get("ATEN_CPU_CAPABILITY", "unset"),
        "threads": torch.get_num_threads(), "exitstatus": int(exitstatus),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("peft", "transformers", "numpy", "pytest")},
        "source_sha256": hashlib.sha256(
            Path("tests/test_issue837_fast_lora_mlp.py").read_bytes()
        ).hexdigest(),
        "rows": sorted(ROWS, key=lambda row: row["seed"]),
    }
    Path(os.environ["SOUP_BF16_CAPTURE"]).write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
