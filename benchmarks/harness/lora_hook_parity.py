"""Does a MoE serving engine apply a Soup LoRA adapter, unmerged? (probe-lora-hook-engines.md)

Builds two tiny SYNTHETIC models with random weights and fixed seeds: one shaped
like DeepSeek-V3 (MLA attention, routed and shared experts) and one shaped like
Qwen3.5-MoE (the positive control). It attaches SYNTHETIC LoRA adapters with a
non-zero ``lora_B`` to Soup's target modules and compares the adapter's effect
on the logits in llama.cpp against transformers + PEFT. The verdicts and their
thresholds are the record's §2, committed before the first run: APPLIED,
DROPPED, WRONG, CONVERT-FAILED, LOAD-FAILED, TOO WEAK and VOID.

Two Python environments on purpose. This script runs where Soup's training
stack lives (the reference); llama.cpp's converters run in llama.cpp's own
pinned environment (``--convert-python``), because they pin a different
transformers major than Soup does.

No timing is measured. Weights and adapters are random: nothing here is a
statement about a trained model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("lora_hook_parity")

# §2 of the record. Changing any of these after the first run voids the probe.
E_BASE_MAX = 1e-3
S_REF_MIN = 1e-2
R_APPLIED_MAX = 1e-2
RHO_DROPPED_MAX = 1e-2

LORA_RANK = 8
LORA_ALPHA = 16
B_STD = 0.02
B_STD_RETRY_FACTOR = 4.0
MAX_SEEDS = 3

# llama.cpp's CPU defaults are flash attention and an f16 KV cache, which leave a ~3e-4
# base gap on dsv3-tiny (diagnosed after run 1); these make the engine f32 end to end.
ENGINE_F32_FLAGS = ("-fa", "off", "-ctk", "f32", "-ctv", "f32")

DEFAULT_PROMPT = (
    "Soup trains the adapter on a laptop; the engine has to apply it to every "
    "projection it was trained on."
)

DSV3_SHARED = (
    "shared_experts.gate_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
)
QWEN35_SHARED = (
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
)
QWEN35_ATTENTION = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)
# The record's positive control: paths llama.cpp routes through its LoRA helper.
POSITIVE_CONTROL = ("q_proj", "v_proj", "shared")


# =====================================================================
# What is built
# =====================================================================
@dataclass(frozen=True)
class ModelSpec:
    label: str
    tokenizer_repo: str
    tokenizer_revision: str
    tokenizer_files: Tuple[str, ...]
    vocab_size: int
    make_config: Callable[[int], Any]
    # Keys the real checkpoint's config.json carries but transformers does not
    # serialise from its own config class; llama.cpp's converter reads them.
    config_extras: Tuple[Tuple[str, Any], ...] = ()
    convert_flags: Tuple[str, ...] = ()


def dsv3_tiny_config(vocab_size: int) -> Any:
    """DeepSeek-V3 shapes scaled down: MLA with q_lora_rank > 0, routed + shared experts."""
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        # Kimi K2's routing; DeepSeek-V3's expert groups are orthogonal to the adapter.
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )


def qwen35moe_tiny_config(vocab_size: int) -> Any:
    """Qwen3.5-MoE shapes scaled down: 3 Gated DeltaNet + 1 full-attention layer."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    return Qwen3_5MoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        # As in the real config. transformers builds no MTP block from it, while
        # convert_lora_to_gguf.py asserts the config declares one for Qwen3.5.
        mtp_num_hidden_layers=1,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
        },
    )


MODELS: Dict[str, ModelSpec] = {
    "dsv3-tiny": ModelSpec(
        label="dsv3-tiny",
        tokenizer_repo="deepseek-ai/DeepSeek-V3",
        tokenizer_revision="e815299b0bcbac849fa540c768ef21845365c9eb",
        tokenizer_files=("tokenizer.json", "tokenizer_config.json"),
        vocab_size=129280,
        make_config=dsv3_tiny_config,
        # deepseek-ai/DeepSeek-V3@e815299b config.json carries both; without
        # scoring_func the converter writes no gating function and llama.cpp
        # routes with softmax where the checkpoint routes with sigmoid.
        config_extras=(("scoring_func", "sigmoid"), ("topk_method", "noaux_tc")),
    ),
    "qwen35moe-tiny": ModelSpec(
        label="qwen35moe-tiny",
        tokenizer_repo="Qwen/Qwen3.5-35B-A3B",
        tokenizer_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        tokenizer_files=("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
        vocab_size=248320,
        make_config=qwen35moe_tiny_config,
        # The checkpoint holds no MTP weights (see above), so the base converter is
        # told not to look for them.
        convert_flags=("--no-mtp",),
    ),
}


def soup_policy(model_type: str) -> Tuple[str, ...]:
    """The targets Soup's ``target_modules: auto`` resolves to for this model type."""
    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS, QWEN35_TEXT_LORA_TARGETS

    if model_type == "qwen3_5_moe_text":
        return tuple(QWEN35_TEXT_LORA_TARGETS)
    return tuple(MOE_TEXT_LORA_TARGETS[model_type])


def variants_for(label: str) -> Dict[str, Tuple[str, ...]]:
    """Adapter variants of the record's §1 table, by name."""
    if label == "dsv3-tiny":
        attention = soup_policy("deepseek_v3")
        everything = attention + DSV3_SHARED
        out = {"soup-auto": attention, "all": everything}
        out["all-but-kv_b"] = tuple(t for t in everything if t != "kv_b_proj")
        out.update({module: (module,) for module in attention})
        out["shared"] = DSV3_SHARED
        return out
    if label == "qwen35moe-tiny":
        out = {"soup-auto": soup_policy("qwen3_5_moe_text")}
        out["all"] = QWEN35_ATTENTION + QWEN35_SHARED
        out.update({module: (module,) for module in QWEN35_ATTENTION})
        out["shared"] = QWEN35_SHARED
        return out
    raise ValueError(f"unknown model label {label!r}")


# =====================================================================
# The rule (§2), as pure functions
# =====================================================================
def _norm(array: np.ndarray) -> float:
    return float(np.linalg.norm(array.astype(np.float64)))


def base_error(z_eng0: np.ndarray, z_ref0: np.ndarray) -> float:
    """``e_base``: the worst logit gap, relative to the largest reference logit."""
    gap = np.abs(z_eng0.astype(np.float64) - z_ref0.astype(np.float64)).max()
    return float(gap / np.abs(z_ref0.astype(np.float64)).max())


def effect_size(z_ref0: np.ndarray, z_ref1: np.ndarray) -> float:
    """``s_ref``: how much the adapter moves the reference logits."""
    return _norm(z_ref1.astype(np.float64) - z_ref0) / _norm(z_ref0)


def effect_metrics(
    z_ref0: np.ndarray, z_ref1: np.ndarray, z_eng0: np.ndarray, z_eng1: np.ndarray
) -> Dict[str, float]:
    """``rho``: share of the reference's effect the engine reproduces; ``r``: how wrongly."""
    d_ref = z_ref1.astype(np.float64) - z_ref0
    d_eng = z_eng1.astype(np.float64) - z_eng0
    ref_size = _norm(d_ref)
    eng_size = _norm(d_eng)
    cosine = float((d_eng * d_ref).sum() / (eng_size * ref_size)) if eng_size else 0.0
    return {
        "rho": eng_size / ref_size,
        "r": _norm(d_eng - d_ref) / ref_size,
        "cos": cosine,
    }


def verdict(
    *,
    base_ok: bool,
    s_ref: Optional[float],
    convert_ok: Optional[bool],
    load_ok: Optional[bool],
    r: Optional[float] = None,
    rho: Optional[float] = None,
) -> str:
    """The first matching row of §2's table."""
    if not base_ok:
        return "VOID"
    if s_ref is None or s_ref < S_REF_MIN:
        return "TOO WEAK"
    if not convert_ok:
        return "CONVERT-FAILED"
    if not load_ok:
        return "LOAD-FAILED"
    if r is not None and r <= R_APPLIED_MAX:
        return "APPLIED"
    if rho is not None and rho <= RHO_DROPPED_MAX:
        return "DROPPED"
    return "WRONG"


def reference_controls_ok(row: Dict[str, Any]) -> bool:
    """Missing or failed controls cannot support a claim about adapter application."""
    return (
        row.get("reference_disabled_is_base") is True
        and row.get("touches_routed_experts") is False
        and row.get("reference_deterministic") is True
    )


def probe_verdict(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Positive control first; then the hook list is every MLA-model module not APPLIED."""
    for label, model in models.items():
        for name, row in (model.get("variants") or {}).items():
            measured = row.get("verdict") in ("APPLIED", "DROPPED", "WRONG")
            if (
                not reference_controls_ok(row)
                or row.get("verdict") == "VOID"
                or (measured and (
                    row.get("engine_deterministic") is not True
                    or row.get("tokens_match") is not True
                ))
            ):
                return {"verdict": "NO VERDICT", "why": f"failed controls: {label}/{name}"}
    qwen = models.get("qwen35moe-tiny", {}).get("variants")
    if not qwen:
        return {"verdict": "NO VERDICT", "why": "positive-control model has no valid run"}
    failed = [name for name in POSITIVE_CONTROL if qwen.get(name, {}).get("verdict") != "APPLIED"]
    if failed:
        return {"verdict": "NO VERDICT", "why": f"positive control not APPLIED: {failed}"}
    dsv3 = models.get("dsv3-tiny", {}).get("variants")
    if not dsv3:
        return {"verdict": "NO VERDICT", "why": "dsv3-tiny has no valid run"}
    singles = list(soup_policy("deepseek_v3")) + ["shared"]
    hook = {name: dsv3[name]["verdict"] for name in singles if dsv3[name]["verdict"] != "APPLIED"}
    return {"verdict": "HOOK NEEDED" if hook else "NO HOOK NEEDED", "hook_modules": hook}


# =====================================================================
# Steps
# =====================================================================
@dataclass
class Context:
    llama_bin: pathlib.Path
    llama_src: pathlib.Path
    convert_python: str
    work: pathlib.Path
    prompt: str
    threads: int


@dataclass
class Step:
    ok: bool
    returncode: int
    stderr_tail: str


def run_step(command: Sequence[str]) -> Step:
    log.debug("$ %s", " ".join(command))
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    tail = (proc.stderr or "")[-3000:] + (proc.stdout or "")[-1000:]
    return Step(ok=proc.returncode == 0, returncode=proc.returncode, stderr_tail=tail)


def fetch_tokenizer(spec: ModelSpec, out_dir: pathlib.Path) -> None:
    from huggingface_hub import hf_hub_download

    for name in spec.tokenizer_files:
        path = hf_hub_download(spec.tokenizer_repo, name, revision=spec.tokenizer_revision)
        shutil.copyfile(path, out_dir / name)


def build_model(spec: ModelSpec, seed: int, out_dir: pathlib.Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    config = spec.make_config(spec.vocab_size)
    for key, value in spec.config_extras:
        setattr(config, key, value)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(out_dir, safe_serialization=True)
    fetch_tokenizer(spec, out_dir)


@contextlib.contextmanager
def targets_as_named() -> Iterator[None]:
    """Keep LoRA targets exactly as named while PEFT builds or loads an adapter.

    peft 0.21 rewrites, on transformers-v5 MoE model types (``deepseek_v3`` among
    them), every target ending in ``gate_proj``/``up_proj``/``down_proj`` into the
    ROUTED experts' fused parameters. ``shared_experts.gate_proj`` then adapts all
    routed experts and not the shared expert (run 2 of the record). The probe tests
    the engine on the adapter Soup's policy intends, so the rewrite is switched off.
    """
    from peft.utils import transformers_weight_conversion as conversion

    original = conversion.convert_peft_config_for_transformers
    conversion.convert_peft_config_for_transformers = lambda *args, **kwargs: None
    try:
        yield
    finally:
        conversion.convert_peft_config_for_transformers = original


def adapter_modules(adapter_dir: pathlib.Path) -> List[str]:
    """The base modules an adapter file actually carries LoRA factors for."""
    from safetensors import safe_open

    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "np") as handle:
        keys = list(handle.keys())
    prefix = "base_model.model."
    names = {key.split(".lora_")[0].removeprefix(prefix) for key in keys}
    return sorted(names)


def make_adapter(
    model_dir: pathlib.Path, targets: Sequence[str], seed: int, b_std: float, out_dir: pathlib.Path
) -> List[str]:
    """A PEFT adapter whose ``lora_B`` is non-zero; returns the modules it actually adapts."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=list(targets)
    )
    torch.manual_seed(seed)  # PEFT draws lora_A from the global generator
    with targets_as_named():
        peft_model = get_peft_model(model, config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, param in sorted(peft_model.named_parameters()):
                if ".lora_B." in name:
                    param.copy_(torch.randn(param.shape, generator=generator) * b_std)
        peft_model.save_pretrained(out_dir)
    return adapter_modules(out_dir)


def reference_logits(
    model_dir: pathlib.Path, tokens: Sequence[int], adapter_dir: Optional[pathlib.Path] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Logits with the adapter on, and with it disabled through PEFT (None without adapter)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    ids = torch.tensor([list(tokens)], dtype=torch.long)
    if adapter_dir is None:
        with torch.no_grad():
            return model(input_ids=ids).logits[0].numpy(), None
    from peft import PeftModel

    with targets_as_named():
        peft_model = PeftModel.from_pretrained(model, adapter_dir).eval()
    with torch.no_grad():
        on = peft_model(input_ids=ids).logits[0].numpy()
        with peft_model.disable_adapter():
            off = peft_model(input_ids=ids).logits[0].numpy()
    return on, off


def convert_base(
    ctx: Context,
    model_dir: pathlib.Path,
    out: pathlib.Path,
    flags: Sequence[str] = (),
    outtype: str = "f32",
) -> Step:
    script = ctx.llama_src / "convert_hf_to_gguf.py"
    command = [ctx.convert_python, str(script), str(model_dir), *flags]
    return run_step(command + ["--outtype", outtype, "--outfile", str(out)])


def convert_adapter(
    ctx: Context, model_dir: pathlib.Path, adapter_dir: pathlib.Path, out: pathlib.Path
) -> Step:
    script = ctx.llama_src / "convert_lora_to_gguf.py"
    command = [ctx.convert_python, str(script), str(adapter_dir), "--base", str(model_dir)]
    return run_step(command + ["--outtype", "f32", "--outfile", str(out)])


_READ_RESULTS = (
    "import sys, numpy as np, gguf\n"
    "reader = gguf.GGUFReader(sys.argv[1])\n"
    "data = {t.name: np.asarray(t.data).reshape(-1) for t in reader.tensors}\n"
    "np.savez(sys.argv[2], tokens=data['tokens'], logits=data['logits'])\n"
)


def results_binary(llama_bin: pathlib.Path) -> pathlib.Path:
    for name in ("llama-results.exe", "llama-results"):
        if (llama_bin / name).is_file():
            return llama_bin / name
    raise FileNotFoundError(f"no llama-results binary in {llama_bin}")


def engine_logits(
    ctx: Context, gguf: pathlib.Path, lora: Optional[pathlib.Path], out: pathlib.Path
) -> Tuple[Step, Optional[np.ndarray], Optional[np.ndarray]]:
    """Token ids and logits of every prompt position, as llama-results writes them."""
    command = [str(results_binary(ctx.llama_bin)), "-m", str(gguf), "-p", ctx.prompt]
    command += ["-o", str(out), "-t", str(ctx.threads), "-c", "512", *ENGINE_F32_FLAGS]
    if lora is not None:
        command += ["--lora", str(lora)]
    step = run_step(command)
    if not step.ok:
        return step, None, None
    npz = out.with_suffix(".npz")
    read = run_step([ctx.convert_python, "-c", _READ_RESULTS, str(out), str(npz)])
    if not read.ok:
        raise RuntimeError(f"could not read {out}: {read.stderr_tail}")
    data = np.load(npz)
    tokens = data["tokens"].astype(np.int64)
    # results.cpp declares the tensor as (n_tokens, n_vocab) but fills it row-major by token.
    logits = data["logits"].astype(np.float32).reshape(len(tokens), -1)
    return step, tokens, logits


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare logits exactly, including signed zero, rather than with numeric equality."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and bool(np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        ))
    )


def digest(array: Optional[np.ndarray]) -> Optional[str]:
    if array is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


# =====================================================================
# One model, one variant
# =====================================================================
@dataclass
class Base:
    model_dir: pathlib.Path
    gguf: pathlib.Path
    tokens: np.ndarray
    z_ref0: np.ndarray
    z_eng0: np.ndarray


def run_variant(
    ctx: Context, base: Base, name: str, targets: Sequence[str], seed: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"targets": list(targets)}
    for attempt, b_std in enumerate((B_STD, B_STD * B_STD_RETRY_FACTOR)):
        adapter_dir = ctx.work / f"{base.model_dir.name}-adapters" / f"{name}-b{attempt}"
        out["adapted_modules"] = make_adapter(base.model_dir, targets, seed, b_std, adapter_dir)
        # The probe's premise: routed experts stay frozen. A target rewrite would break it.
        out["touches_routed_experts"] = any(".mlp.experts" in m for m in out["adapted_modules"])
        z_ref1, z_off = reference_logits(base.model_dir, base.tokens, adapter_dir)
        out["lora_b_std"] = b_std
        out["reference_disabled_is_base"] = bitwise_equal(z_off, base.z_ref0)
        out["s_ref"] = effect_size(base.z_ref0, z_ref1)
        if (
            out["s_ref"] >= S_REF_MIN
            or not out["reference_disabled_is_base"]
            or out["touches_routed_experts"]
        ):
            break
    z_ref1b, z_offb = reference_logits(base.model_dir, base.tokens, adapter_dir)
    out["reference_deterministic"] = bitwise_equal(z_ref1, z_ref1b)
    out["reference_disabled_is_base"] &= bitwise_equal(z_offb, base.z_ref0)
    out["digests"] = {"z_ref1": digest(z_ref1), "z_ref1_repeat": digest(z_ref1b)}
    if not reference_controls_ok(out):
        out.update(verdict="VOID", why="reference sanity, target or determinism control failed")
        return out
    lora_gguf = adapter_dir.with_suffix(".gguf")
    conversion = convert_adapter(ctx, base.model_dir, adapter_dir, lora_gguf)
    out["convert"] = vars(conversion)
    if not conversion.ok:
        out["verdict"] = verdict(base_ok=True, s_ref=out["s_ref"], convert_ok=False, load_ok=None)
        return out
    results_path = lora_gguf.with_name(name + ".out.gguf")
    run, tokens, z_eng1 = engine_logits(ctx, base.gguf, lora_gguf, results_path)
    out["engine"] = vars(run)
    if not run.ok:
        out["verdict"] = verdict(base_ok=True, s_ref=out["s_ref"], convert_ok=True, load_ok=False)
        return out
    again, tokens2, z_eng1b = engine_logits(
        ctx, base.gguf, lora_gguf, results_path.with_name(name + ".repeat.gguf")
    )
    out["engine_repeat"] = vars(again)
    out["engine_deterministic"] = bool(again.ok and bitwise_equal(z_eng1, z_eng1b))
    out["tokens_match"] = bool(
        again.ok and np.array_equal(tokens, base.tokens) and np.array_equal(tokens2, base.tokens)
    )
    out["digests"].update(z_eng1=digest(z_eng1), z_eng1_repeat=digest(z_eng1b))
    if not out["tokens_match"] or not out["engine_deterministic"]:
        out.update(
            verdict="VOID", why="engine adapter repeat failed, changed tokens or changed logits"
        )
        return out
    if out["s_ref"] < S_REF_MIN:
        out["verdict"] = "TOO WEAK"
        return out
    out.update(effect_metrics(base.z_ref0, z_ref1, base.z_eng0, z_eng1))
    out["verdict"] = verdict(
        base_ok=True,
        s_ref=out["s_ref"],
        convert_ok=True,
        load_ok=True,
        r=out["r"],
        rho=out["rho"],
    )
    return out


def prepare_base(ctx: Context, spec: ModelSpec, seed: int) -> Tuple[Dict[str, Any], Optional[Base]]:
    """Build, convert and compare the base model; returns the attempt record and a Base if valid."""
    model_dir = ctx.work / f"{spec.label}-seed{seed}"
    build_model(spec, seed, model_dir)
    gguf = model_dir.with_suffix(".f32.gguf")
    attempt: Dict[str, Any] = {"seed": seed, "model_dir": model_dir.name}
    conversion = convert_base(ctx, model_dir, gguf, spec.convert_flags)
    attempt["convert_base"] = vars(conversion)
    if not conversion.ok:
        attempt["status"] = "VOID"
        attempt["why"] = "base conversion failed"
        return attempt, None
    first, tokens, z_eng0 = engine_logits(ctx, gguf, None, model_dir.with_suffix(".base1.gguf"))
    again, tokens2, z_eng0b = engine_logits(ctx, gguf, None, model_dir.with_suffix(".base2.gguf"))
    attempt["engine_base"] = vars(first)
    if not (first.ok and again.ok):
        attempt["status"] = "VOID"
        attempt["why"] = "engine could not run the base model"
        return attempt, None
    z_ref0, _ = reference_logits(model_dir, tokens)
    z_ref0b, _ = reference_logits(model_dir, tokens)
    attempt["tokens"] = tokens.tolist()
    attempt["e_base"] = base_error(z_eng0, z_ref0)
    attempt["engine_deterministic"] = bitwise_equal(z_eng0, z_eng0b)
    attempt["engine_tokens_stable"] = bool(np.array_equal(tokens, tokens2))
    attempt["reference_deterministic"] = bitwise_equal(z_ref0, z_ref0b)
    attempt["digests"] = {"z_ref0": digest(z_ref0), "z_eng0": digest(z_eng0)}
    valid = (
        attempt["e_base"] <= E_BASE_MAX
        and attempt["engine_deterministic"]
        and attempt["engine_tokens_stable"]
        and attempt["reference_deterministic"]
    )
    attempt["status"] = "OK" if valid else "VOID"
    if not valid:
        return attempt, None
    return attempt, Base(model_dir, gguf, tokens, z_ref0, z_eng0)


def run_model(ctx: Context, spec: ModelSpec, first_seed: int) -> Dict[str, Any]:
    attempts = []
    for seed in range(first_seed, first_seed + MAX_SEEDS):
        attempt, base = prepare_base(ctx, spec, seed)
        attempts.append(attempt)
        log.info("%s seed %d: base %s (e_base %s)", spec.label, seed, attempt["status"],
                 attempt.get("e_base"))
        if base is None:
            continue
        variants = {}
        for index, (name, targets) in enumerate(variants_for(spec.label).items()):
            variants[name] = run_variant(ctx, base, name, targets, seed * 1000 + index)
            row = variants[name]
            log.info("  %-20s %-15s s_ref=%s rho=%s r=%s", name, row["verdict"],
                     _fmt(row.get("s_ref")), _fmt(row.get("rho")), _fmt(row.get("r")))
        status = "VOID" if any(row["verdict"] == "VOID" for row in variants.values()) else "OK"
        return {"attempts": attempts, "variants": variants, "status": status}
    return {"attempts": attempts, "variants": None}


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3e}"


# =====================================================================
# Provenance
# =====================================================================
def harness_fingerprint() -> str:
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16]


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_versions() -> Dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in ("torch", "transformers", "peft", "numpy")}


def converter_versions(convert_python: str) -> str:
    probe = (
        "import torch, transformers, numpy, gguf, importlib.metadata as m;"
        "print(torch.__version__, transformers.__version__, numpy.__version__,"
        " m.version('gguf'))"
    )
    return run_step([convert_python, "-c", probe]).stderr_tail.strip()


def git_head(path: pathlib.Path) -> str:
    return run_step(["git", "-C", str(path), "rev-parse", "HEAD"]).stderr_tail.strip()


def host_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        state["ram_total_gb"] = round(memory.total / 1e9, 2)
        state["ram_available_gb"] = round(memory.available / 1e9, 2)
    except ImportError:
        state["ram_total_gb"] = None
    return state


def provenance(ctx: Context) -> Dict[str, Any]:
    binary = results_binary(ctx.llama_bin)
    return {
        "harness_sha256_16": harness_fingerprint(),
        "host": host_state(),
        "reference_env": reference_versions(),
        "converter_env": converter_versions(ctx.convert_python),
        "llama_src_head": git_head(ctx.llama_src),
        "llama_version": run_step([str(binary), "--version"]).stderr_tail.strip(),
        "llama_binaries_sha256": {
            path.name: file_sha256(path)
            for path in sorted(ctx.llama_bin.iterdir())
            if path.is_file()
        },
        "prompt": ctx.prompt,
        "threads": ctx.threads,
        "rule": {
            "E_BASE_MAX": E_BASE_MAX,
            "S_REF_MIN": S_REF_MIN,
            "R_APPLIED_MAX": R_APPLIED_MAX,
            "RHO_DROPPED_MAX": RHO_DROPPED_MAX,
            "LORA_RANK": LORA_RANK,
            "LORA_ALPHA": LORA_ALPHA,
            "B_STD": B_STD,
        },
    }


# =====================================================================
# CLI
# =====================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llama-bin", type=pathlib.Path, required=True,
                        help="directory holding the llama-results binary")
    parser.add_argument("--llama-src", type=pathlib.Path, required=True,
                        help="llama.cpp source tree at the tag under test (converters)")
    parser.add_argument("--convert-python", required=True,
                        help="python of llama.cpp's own converter environment")
    parser.add_argument("--work-dir", type=pathlib.Path, required=True,
                        help="scratch directory for models, adapters and GGUF files")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="result JSON")
    parser.add_argument("--log", type=pathlib.Path, help="also write the console log here")
    parser.add_argument("--models", default="qwen35moe-tiny,dsv3-tiny")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args(argv)


def configure_logging(log_path: Optional[pathlib.Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log)
    labels = [label.strip() for label in args.models.split(",") if label.strip()]
    unknown = [label for label in labels if label not in MODELS]
    if unknown:
        raise SystemExit(f"unknown --models {unknown}; choose from {sorted(MODELS)}")
    ctx = Context(
        llama_bin=args.llama_bin,
        llama_src=args.llama_src,
        convert_python=args.convert_python,
        work=args.work_dir,
        prompt=args.prompt,
        threads=args.threads,
    )
    ctx.work.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"meta": provenance(ctx), "models": {}}
    for label in labels:
        result["models"][label] = run_model(ctx, MODELS[label], args.seed)
    result["probe"] = probe_verdict(result["models"])
    log.info("probe: %s", json.dumps(result["probe"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
