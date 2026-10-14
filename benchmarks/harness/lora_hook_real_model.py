"""Does llama.cpp apply a Soup-shaped adapter, unmerged, on the real Qwen3.5-35B-A3B? (Part B)

The comparison of ``lora_hook_parity.py`` (probe-lora-hook-engines.md, Part A) on
the real test model instead of tiny synthetic ones. SYNTHETIC seeded adapters
with a non-zero ``lora_B`` go on Soup's Qwen3.5 target policy, with and without
the shared expert. They are exported adapter-only by llama.cpp's own converter
(which reads only the base config), served by llama.cpp from the base GGUF with
the weights memory-mapped, and compared with transformers + PEFT. ``--precision``
sets both the GGUF type and the reference dtype: bf16 (the record's §4c, run 1)
or f32 (§4g, run 2). The tolerance is Part B's rule in the record, so the allowed
error scales with the base models' own disagreement.

CPU only. Wall times are logged for budgeting and are not a speed claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lora_hook_parity as parity  # noqa: E402

log = logging.getLogger("lora_hook_real_model")

MODEL_REPO = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
MODEL_FILES = ("*.json", "*.safetensors", "*.txt", "*.jinja")

# Part B of the record. Changing any of these after the run voids it.
TOP1_MIN = 0.9
N_BASE_MAX = 0.05
APPLIED_FLOOR_FACTOR = 3.0
APPLIED_SLACK = 0.02
RHO_DROPPED_MAX = 0.1
# Above this the APPLIED band would admit a dropped adapter (r = 1), so no verdict.
TOLERANCE_MAX = 0.5
SEED = 17
# --precision: the base GGUF's --outtype and the reference's torch dtype, kept equal.
TORCH_DTYPES = {"bf16": "bfloat16", "f32": "float32"}


def variants() -> Dict[str, Tuple[str, ...]]:
    auto = parity.soup_policy("qwen3_5_moe_text")
    return {"soup-auto": auto, "soup-auto+shared": auto + parity.QWEN35_SHARED}


# =====================================================================
# The rule (Part B), as pure functions
# =====================================================================
def top1_agreement(z_a: np.ndarray, z_b: np.ndarray) -> float:
    return float((z_a.argmax(axis=-1) == z_b.argmax(axis=-1)).mean())


def relative_gap(z_eng: np.ndarray, z_ref: np.ndarray) -> float:
    """``‖z_eng − z_ref‖ / ‖z_ref‖`` in float64."""
    diff = z_eng.astype(np.float64) - z_ref.astype(np.float64)
    return float(np.linalg.norm(diff) / np.linalg.norm(z_ref.astype(np.float64)))


def noise_floor(z_eng0: np.ndarray, z_ref0: np.ndarray, z_ref1: np.ndarray) -> float:
    """The base disagreement in units of the adapter's effect: ``‖z_eng⁰ − z_ref⁰‖ / ‖Δ_ref‖``."""
    base = np.linalg.norm(z_eng0.astype(np.float64) - z_ref0.astype(np.float64))
    effect = np.linalg.norm(z_ref1.astype(np.float64) - z_ref0.astype(np.float64))
    return float(base / effect)


def verdict_b(
    *,
    base_ok: bool,
    s_ref: Optional[float],
    convert_ok: Optional[bool],
    load_ok: Optional[bool],
    r: Optional[float] = None,
    rho: Optional[float] = None,
    floor: Optional[float] = None,
) -> str:
    """The first matching row of Part B's table."""
    if not base_ok:
        return "VOID"
    if s_ref is None or s_ref < parity.S_REF_MIN:
        return "TOO WEAK"
    if not convert_ok:
        return "CONVERT-FAILED"
    if not load_ok:
        return "LOAD-FAILED"
    if floor is None or tolerance(floor) > TOLERANCE_MAX:
        return "TOO NOISY"
    if r is not None and r <= tolerance(floor):
        return "APPLIED"
    if rho is not None and rho <= RHO_DROPPED_MAX:
        return "DROPPED"
    return "WRONG"


def tolerance(floor: float) -> float:
    """Allow three base gaps in effect units, plus the preregistered 0.02 slack."""
    return APPLIED_FLOOR_FACTOR * floor + APPLIED_SLACK


def rule_constants() -> Dict[str, Any]:
    """The constants verdict_b and the adapter recipe use, for the result JSON."""
    return {
        "TOP1_MIN": TOP1_MIN, "N_BASE_MAX": N_BASE_MAX, "S_REF_MIN": parity.S_REF_MIN,
        "APPLIED_FLOOR_FACTOR": APPLIED_FLOOR_FACTOR, "APPLIED_SLACK": APPLIED_SLACK,
        "TOLERANCE_MAX": TOLERANCE_MAX, "RHO_DROPPED_MAX": RHO_DROPPED_MAX,
        "LORA_RANK": parity.LORA_RANK, "LORA_ALPHA": parity.LORA_ALPHA,
        "B_STD": parity.B_STD, "B_STD_RETRY_FACTOR": parity.B_STD_RETRY_FACTOR, "SEED": SEED,
    }


# =====================================================================
# Steps
# =====================================================================
def download(out_dir: pathlib.Path) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_dir=out_dir, allow_patterns=list(MODEL_FILES)
    )
    return pathlib.Path(path)


def load_reference(model_dir: pathlib.Path, precision: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(os.cpu_count() or 1)
    dtype = getattr(torch, TORCH_DTYPES[precision])
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    return model.eval()


def logits_of(model: Any, tokens: Sequence[int]) -> np.ndarray:
    import torch

    ids = torch.tensor([list(tokens)], dtype=torch.long)
    with torch.no_grad():
        return model(input_ids=ids).logits[0].float().numpy()


def adapter_on_loaded(
    model: Any, targets: Sequence[str], seed: int, b_std: float, out_dir: pathlib.Path,
    tokens: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[str], Any, np.ndarray]:
    """Attach a seeded adapter, read logits on/off/on, save, detach."""
    import torch
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=parity.LORA_RANK, lora_alpha=parity.LORA_ALPHA, lora_dropout=0.0,
        target_modules=list(targets),
    )
    torch.manual_seed(seed)  # PEFT draws lora_A from the global generator
    with parity.targets_as_named():
        peft_model = get_peft_model(model, config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in sorted(peft_model.named_parameters()):
                if ".lora_B." in name:
                    noise = torch.randn(param.shape, generator=generator) * b_std
                    param.copy_(noise.to(param.dtype))
        on = logits_of(peft_model, tokens)
        with peft_model.disable_adapter():
            off = logits_of(peft_model, tokens)
        on_again = logits_of(peft_model, tokens)
        peft_model.save_pretrained(out_dir)
        base = peft_model.unload()
    return on, off, parity.adapter_modules(out_dir), base, on_again


def machine_state() -> Dict[str, Any]:
    state = parity.host_state()
    commands = (("free_gb", ["free", "-g"]), ("cpu", ["lscpu"]), ("uname", ["uname", "-a"]))
    for key, command in commands:
        try:
            state[key] = subprocess.run(command, capture_output=True, text=True).stdout[-3000:]
        except OSError:
            state[key] = None
    return state


class Clock:
    def __init__(self) -> None:
        self.marks: Dict[str, float] = {}
        self._start = time.time()

    def mark(self, name: str) -> None:
        self.marks[name] = round(time.time() - self._start, 1)
        log.info("[%7.1f s] %s", self.marks[name], name)


# =====================================================================
# The run
# =====================================================================
def engine_base(ctx: parity.Context, gguf: pathlib.Path) -> Dict[str, Any]:
    first, tokens, z0 = parity.engine_logits(ctx, gguf, None, ctx.work / "base1.out.gguf")
    again, tokens2, z0b = parity.engine_logits(ctx, gguf, None, ctx.work / "base2.out.gguf")
    return {
        "step": vars(first), "ok": first.ok and again.ok, "tokens": tokens, "z_eng0": z0,
        "deterministic": bool(first.ok and again.ok and parity.bitwise_equal(z0, z0b)),
        "tokens_stable": bool(first.ok and again.ok and np.array_equal(tokens, tokens2)),
    }


def reference_side(
    model_dir: pathlib.Path, tokens: Sequence[int], work: pathlib.Path, clock: Clock,
    precision: str,
) -> Dict[str, Any]:
    model = load_reference(model_dir, precision)
    clock.mark("reference loaded")
    out: Dict[str, Any] = {"z_ref0": logits_of(model, tokens)}
    out["deterministic"] = parity.bitwise_equal(out["z_ref0"], logits_of(model, tokens))
    clock.mark("reference base logits")
    out["variants"] = {}
    for index, (name, targets) in enumerate(variants().items()):
        for attempt, b_std in enumerate((parity.B_STD, parity.B_STD * parity.B_STD_RETRY_FACTOR)):
            adapter_dir = work / "adapters" / f"{name}-b{attempt}"
            on, off, modules, model, on_again = adapter_on_loaded(
                model, targets, SEED * 1000 + index, b_std, adapter_dir, tokens
            )
            row = {
                "targets": list(targets), "adapter_dir": adapter_dir, "z_ref1": on,
                "adapted_modules": modules, "lora_b_std": b_std,
                "reference_disabled_is_base": parity.bitwise_equal(off, out["z_ref0"]),
                "reference_deterministic": parity.bitwise_equal(on, on_again),
                "digests": {
                    "z_ref1": parity.digest(on), "z_ref1_repeat": parity.digest(on_again),
                },
                "touches_routed_experts": any(".mlp.experts" in m for m in modules),
                "s_ref": parity.effect_size(out["z_ref0"], on),
            }
            if row["s_ref"] >= parity.S_REF_MIN or not parity.reference_controls_ok(row):
                break
        out["variants"][name] = row
        clock.mark(f"reference {name}")
    del model
    gc.collect()
    return out


def engine_variant(
    ctx: parity.Context, model_dir: pathlib.Path, gguf: pathlib.Path, name: str,
    row: Dict[str, Any], base: Dict[str, Any], z_ref0: np.ndarray, base_ok: bool,
) -> Dict[str, Any]:
    adapter_dir = row.pop("adapter_dir")
    z_ref1 = row.pop("z_ref1")
    if not parity.reference_controls_ok(row):
        row.update(verdict="VOID", why="reference sanity, target or determinism control failed")
        return row
    lora_gguf = adapter_dir.with_suffix(".gguf")
    conversion = parity.convert_adapter(ctx, model_dir, adapter_dir, lora_gguf)
    row["convert"] = vars(conversion)
    row["adapter_gguf_bytes"] = lora_gguf.stat().st_size if lora_gguf.exists() else None
    if not conversion.ok:
        row["verdict"] = verdict_b(base_ok=base_ok, s_ref=row["s_ref"], convert_ok=False,
                                   load_ok=None)
        return row
    run, tokens, z_eng1 = parity.engine_logits(ctx, gguf, lora_gguf, ctx.work / f"{name}.out.gguf")
    row["engine"] = vars(run)
    if not run.ok:
        row["verdict"] = verdict_b(base_ok=base_ok, s_ref=row["s_ref"], convert_ok=True,
                                   load_ok=False)
        return row
    again, tokens2, z_eng1b = parity.engine_logits(
        ctx, gguf, lora_gguf, ctx.work / f"{name}.repeat.gguf"
    )
    row["engine_repeat"] = vars(again)
    row["engine_deterministic"] = bool(again.ok and parity.bitwise_equal(z_eng1, z_eng1b))
    row["tokens_match"] = bool(
        again.ok
        and np.array_equal(tokens, base["tokens"])
        and np.array_equal(tokens2, base["tokens"])
    )
    row["digests"].update(z_eng1=parity.digest(z_eng1), z_eng1_repeat=parity.digest(z_eng1b))
    if not base_ok or not row["tokens_match"] or not row["engine_deterministic"]:
        row.update(
            verdict="VOID",
            why="base invalid or engine adapter repeat failed, changed tokens or logits",
        )
        return row
    if row["s_ref"] < parity.S_REF_MIN:
        row["verdict"] = "TOO WEAK"
        return row
    row.update(parity.effect_metrics(z_ref0, z_ref1, base["z_eng0"], z_eng1))
    row["floor"] = noise_floor(base["z_eng0"], z_ref0, z_ref1)
    row["t"] = tolerance(row["floor"])
    row["verdict"] = verdict_b(
        base_ok=True,
        s_ref=row["s_ref"], convert_ok=True,
        load_ok=True, r=row["r"], rho=row["rho"], floor=row["floor"],
    )
    return row


def run(
    ctx: parity.Context, model_dir: pathlib.Path, clock: Clock, precision: str
) -> Dict[str, Any]:
    gguf = ctx.work / f"qwen3.5-35b-a3b.{precision}.gguf"
    if not gguf.exists():
        conversion = parity.convert_base(ctx, model_dir, gguf, ("--no-mtp",), outtype=precision)
        if not conversion.ok:
            return {"status": "VOID", "why": "base conversion failed",
                    "convert_base": vars(conversion)}
    clock.mark("base converted")
    base = engine_base(ctx, gguf)
    clock.mark("engine base logits")
    if not base["ok"]:
        return {"status": "VOID", "why": "engine could not run the base",
                "engine_base": base["step"]}
    reference = reference_side(model_dir, base["tokens"].tolist(), ctx.work, clock, precision)
    z_ref0 = reference["z_ref0"]
    summary: Dict[str, Any] = {
        "tokens": base["tokens"].tolist(),
        "engine_base": base["step"],
        "engine_deterministic": base["deterministic"],
        "engine_tokens_stable": base["tokens_stable"],
        "reference_deterministic": reference["deterministic"],
        "top1_agreement": top1_agreement(base["z_eng0"], z_ref0),
        "n_base": relative_gap(base["z_eng0"], z_ref0),
        "e_base": parity.base_error(base["z_eng0"], z_ref0),
        "digests": {"z_ref0": parity.digest(z_ref0), "z_eng0": parity.digest(base["z_eng0"])},
    }
    base_ok = (
        summary["top1_agreement"] >= TOP1_MIN
        and summary["n_base"] <= N_BASE_MAX
        and base["deterministic"]
        and base["tokens_stable"]
        and reference["deterministic"]
    )
    summary["status"] = "OK" if base_ok else "VOID"
    summary["variants"] = {
        name: engine_variant(ctx, model_dir, gguf, name, row, base, z_ref0, base_ok)
        for name, row in reference["variants"].items()
    }
    if any(row["verdict"] == "VOID" for row in summary["variants"].values()):
        summary["status"] = "VOID"
    for name, row in summary["variants"].items():
        log.info("  %-18s %-15s s_ref=%s rho=%s r=%s floor=%s", name, row["verdict"],
                 row.get("s_ref"), row.get("rho"), row.get("r"), row.get("floor"))
    clock.mark("engine adapters")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llama-bin", type=pathlib.Path, required=True)
    parser.add_argument("--llama-src", type=pathlib.Path, required=True)
    parser.add_argument("--convert-python", required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="where the checkpoint is (or will be) downloaded")
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--log", type=pathlib.Path)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--box", default="", help="free-text box description for the record")
    parser.add_argument("--precision", choices=sorted(TORCH_DTYPES), default="bf16",
                        help="base GGUF type and reference dtype (record §4c: bf16, §4g: f32)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    parity.configure_logging(args.log)
    ctx = parity.Context(
        llama_bin=args.llama_bin, llama_src=args.llama_src, convert_python=args.convert_python,
        work=args.work_dir, prompt=parity.DEFAULT_PROMPT, threads=args.threads,
    )
    ctx.work.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    result: Dict[str, Any] = {"meta": parity.provenance(ctx)}
    # provenance() files the parity module's hash under the harness key, and Part A's rule
    # under "rule"; record both files, and the rule this harness's verdicts apply.
    result["meta"]["parity_sha256_16"] = result["meta"].pop("harness_sha256_16")
    result["meta"]["rule"] = rule_constants()
    result["meta"].update({"harness_sha256_16": parity.file_sha256(pathlib.Path(__file__))[:16],
                           "box": args.box, "machine_before": machine_state(),
                           "model": f"{MODEL_REPO}@{MODEL_REVISION}",
                           "precision": args.precision})
    model_dir = download(args.model_dir)
    clock.mark("model downloaded")
    result["run"] = run(ctx, model_dir, clock, args.precision)
    result["meta"]["machine_after"] = machine_state()
    result["meta"]["wall_seconds"] = clock.marks
    headline = {k: result["run"].get(k) for k in ("status", "top1_agreement", "n_base")}
    log.info("result: %s", json.dumps(headline))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
