"""Can a Soup adapter for the real giant configs be exported on its own, with no weights? (Part B0)

probe-lora-hook-engines.md, Part B0. For each real config (Qwen3.5-35B-A3B,
DeepSeek-V3, Kimi K2) this builds a SYNTHETIC adapter file covering every layer
of Soup's target policy plus the shared expert, with shapes read from a
meta-device model of that config, and runs llama.cpp's ``convert_lora_to_gguf.py``
against a directory that holds only ``config.json``. It then checks what the
record's rule asks for: every pair carries the adapter's rank on both halves,
and for ``deepseek2`` every layer's ``attn_k_b`` and ``attn_v_b`` pairs have the
outer dimensions the config gives the base tensors. The other pairs' outer
dimensions are not checked.

Converter only. Whether the engine then applies the adapter is Parts A and A'.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lora_hook_parity as parity  # noqa: E402

log = logging.getLogger("lora_export_real_configs")

CONFIGS = {
    "qwen3.5-35b-a3b": ("Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
    "deepseek-v3": ("deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb"),
    "kimi-k2": ("moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"),
}
SHARED = {
    "qwen3.5-35b-a3b": parity.QWEN35_SHARED,
    "deepseek-v3": parity.DSV3_SHARED,
    "kimi-k2": parity.DSV3_SHARED,
}

_READ_GGUF = (
    "import sys, json, gguf\n"
    "reader = gguf.GGUFReader(sys.argv[1])\n"
    "shapes = {t.name: [int(x) for x in t.shape] for t in reader.tensors}\n"
    "open(sys.argv[2], 'w').write(json.dumps(shapes))\n"
)


def fetch_config(repo: str, revision: str, out_dir: pathlib.Path) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo, "config.json", revision=revision)
    shutil.copyfile(path, out_dir / "config.json")
    return json.loads((out_dir / "config.json").read_text(encoding="utf-8"))


def meta_model(label: str, config: Dict[str, Any]) -> Any:
    """The config's model on the meta device, for module names and shapes only."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, DeepseekV3Config

    # Shapes only: the FP8 quantisation block of the DeepSeek-family configs is irrelevant here.
    fields = {k: v for k, v in config.items() if k != "quantization_config"}
    if label == "qwen3.5-35b-a3b":
        hf_config = AutoConfig.for_model(**fields)
    else:  # Kimi K2 declares its own model_type but is DeepseekV3ForCausalLM.
        fields = {k: v for k, v in fields.items() if k not in ("model_type", "auto_map")}
        hf_config = DeepseekV3Config(**fields)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(hf_config)


def targets_for(label: str) -> Tuple[str, ...]:
    model_type = "qwen3_5_moe_text" if label == "qwen3.5-35b-a3b" else "deepseek_v3"
    return parity.soup_policy(model_type) + SHARED[label]


def adapter_tensors(model: Any, targets: Sequence[str], seed: int) -> Dict[str, Any]:
    """Seeded LoRA factors for every Linear the targets select, PEFT's key layout."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, module in sorted(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name == t or name.endswith("." + t) for t in targets):
            continue
        key = f"base_model.model.{name}"
        out[f"{key}.lora_A.weight"] = torch.randn(
            parity.LORA_RANK, module.in_features, generator=generator) * 0.02
        out[f"{key}.lora_B.weight"] = torch.randn(
            module.out_features, parity.LORA_RANK, generator=generator) * 0.02
    return out


def write_adapter(tensors: Dict[str, Any], targets: Sequence[str], out_dir: pathlib.Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    config = {"peft_type": "LORA", "r": parity.LORA_RANK, "lora_alpha": parity.LORA_ALPHA,
              "target_modules": list(targets)}
    (out_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def read_shapes(convert_python: str, gguf_path: pathlib.Path) -> Dict[str, List[int]]:
    shapes_path = gguf_path.with_suffix(".shapes.json")
    step = parity.run_step([convert_python, "-c", _READ_GGUF, str(gguf_path), str(shapes_path)])
    if not step.ok:
        raise RuntimeError(f"could not read {gguf_path}: {step.stderr_tail}")
    return json.loads(shapes_path.read_text(encoding="utf-8"))


def pair_problems(shapes: Dict[str, List[int]]) -> List[str]:
    """Factor pairs that llama.cpp's loader would reject on rank: a.ne1 must equal b.ne0."""
    problems = []
    bases = {name[: -len(".lora_a")] for name in shapes if name.endswith(".lora_a")}
    bases |= {name[: -len(".lora_b")] for name in shapes if name.endswith(".lora_b")}
    for base in sorted(bases):
        a, b = shapes.get(base + ".lora_a"), shapes.get(base + ".lora_b")
        if a is None or b is None:
            problems.append(f"{base}: missing half of the pair")
        elif a[1] != parity.LORA_RANK or b[0] != parity.LORA_RANK:
            problems.append(f"{base}: rank mismatch a={a} b={b}")
    return problems


def mla_problems(config: Dict[str, Any], shapes: Dict[str, List[int]]) -> List[str]:
    """For deepseek2: every layer's k_b / v_b pair must match the base tensors' outer dims."""
    n_head, nope = config["num_attention_heads"], config["qk_nope_head_dim"]
    rank_kv, v_head = config["kv_lora_rank"], config["v_head_dim"]
    expected = {"attn_k_b": (nope, rank_kv), "attn_v_b": (rank_kv, v_head)}
    problems = []
    for layer in range(config["num_hidden_layers"]):
        for tensor, (ne0, ne1) in expected.items():
            base = f"blk.{layer}.{tensor}.weight"
            a, b = shapes.get(base + ".lora_a"), shapes.get(base + ".lora_b")
            if a is None or b is None:
                problems.append(f"{base}: absent")
            elif a[0] != ne0 or b[1] != ne1 or (len(b) > 2 and b[2] not in (1, n_head)):
                problems.append(f"{base}: a={a} b={b} vs base ne0={ne0} ne1={ne1}")
    return problems


def check(label: str, config: Dict[str, Any], ctx: parity.Context, adapter_dir: pathlib.Path,
          base_dir: pathlib.Path, tag: str) -> Dict[str, Any]:
    out_path = ctx.work / f"{label}.{tag}.gguf"
    step = parity.convert_adapter(ctx, base_dir, adapter_dir, out_path)
    row: Dict[str, Any] = {"convert": vars(step)}
    if not step.ok:
        row["verdict"] = "FAIL"
        return row
    shapes = read_shapes(ctx.convert_python, out_path)
    problems = pair_problems(shapes)
    if label != "qwen3.5-35b-a3b":
        problems += mla_problems(config, shapes)
    row.update({"pairs": sum(name.endswith(".lora_a") for name in shapes),
                "problems": problems[:20], "problem_count": len(problems),
                "verdict": "PASS" if not problems else "FAIL"})
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--convert-python", required=True)
    parser.add_argument("--llama-src", type=pathlib.Path, action="append", required=True,
                        help="converter tree(s) to test; give --llama-src TAG=PATH")
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--log", type=pathlib.Path)
    args = parser.parse_args(argv)
    parity.configure_logging(args.log)
    trees = [str(item).split("=", 1) for item in args.llama_src]
    # parity.harness_fingerprint() hashes the imported module only; this file decides the run too.
    result: Dict[str, Any] = {
        "harness_sha256_16": parity.file_sha256(pathlib.Path(__file__))[:16],
        "parity_sha256_16": parity.harness_fingerprint(),
        "reference_env": parity.reference_versions(),
        "configs": {},
    }
    for seed, (label, (repo, revision)) in enumerate(CONFIGS.items()):
        base_dir = args.work_dir / f"{label}-config"
        config = fetch_config(repo, revision, base_dir)
        targets = targets_for(label)
        tensors = adapter_tensors(meta_model(label, config), targets, seed)
        adapter_dir = args.work_dir / f"{label}-adapter"
        write_adapter(tensors, targets, adapter_dir)
        entry: Dict[str, Any] = {"model": f"{repo}@{revision}", "targets": list(targets),
                                 "adapted_modules": len(tensors) // 2}
        for tag, path in trees:
            ctx = parity.Context(llama_bin=pathlib.Path("."), llama_src=pathlib.Path(path),
                                 convert_python=args.convert_python, work=args.work_dir,
                                 prompt="", threads=1)
            entry[tag] = check(label, config, ctx, adapter_dir, base_dir, tag)
            entry[tag]["llama_src_head"] = parity.git_head(pathlib.Path(path))
            log.info("%-16s %-6s %s (%s pairs, %s problems)", label, tag, entry[tag]["verdict"],
                     entry[tag].get("pairs"), entry[tag].get("problem_count"))
        result["configs"][label] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
