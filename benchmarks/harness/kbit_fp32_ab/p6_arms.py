"""Scratch arms for issue #2609 step 1 (MakazhanAlpamys/Soup).  No product change: nothing under src/soup_cli is edited.

apply(arm, out_dir) is called inside the Soup worker process (tco_run.py worker, soup branch) BEFORE `soup train` starts.
soup-cli 0.77.2 imports `prepare_model_for_kbit_training` from peft inside `_setup_transformers` at call time, so replacing
the attribute on the `peft` module is enough (verified by `arm_info.json: prepare_calls`, written by the replacement itself).

  A  nothing patched: soup-cli 0.77.2 as installed from PyPI.
  B  peft's function runs unchanged, then every float32 parameter that is not a bitsandbytes Params4bit and whose name
     contains embed_tokens / lm_head / norm is cast back to bfloat16 and the CUDA cache is emptied (as peft does after its
     own cast).  bf16 -> fp32 -> bf16 is lossless, so the weights are exactly the loaded ones.  Biases are not touched.
  C  peft's function is replaced by what the issue asks for: freeze the base, enable_input_require_grads(), gradient
     checkpointing enabled; no dtype cast at all.  The gating (loaded in 4/8 bit, use_gradient_checkpointing,
     use_reentrant default) is copied from peft 0.20.0 utils/other.py.
  D  nothing patched here: tco_run.soup_yaml adds `use_liger: true` to the yaml (TCO_ARM=D).

Every arm also writes arm_info.json (dtype census at trainer construction and at train end, module classes, forward owner).
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

MIB = 1024.0 * 1024.0
_INFO = {"arm": None, "prepare_calls": 0, "events": []}
_OUT = {"dir": None}


def _flush() -> None:
    if not _OUT["dir"]:
        return
    path = os.path.join(_OUT["dir"], "arm_info.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_INFO, handle, indent=1, default=str)
    os.replace(tmp, path)


def _log(message: str) -> None:
    _INFO["events"].append({"t": time.time(), "msg": message})
    print(f"[p6_arms] {message}", flush=True)
    _flush()


def _cast_back_to_bf16(model) -> dict:
    import torch

    cast = collections.Counter()
    for name, param in model.named_parameters():
        if param.dtype is torch.float32 and param.__class__.__name__ != "Params4bit" \
                and any(key in name for key in ("embed_tokens", "lm_head", "norm")):
            param.data = param.data.to(torch.bfloat16)
            group = "embed_tokens" if "embed_tokens" in name else "lm_head" if "lm_head" in name else "norm"
            cast[group] += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(cast)


def _kbit_b_factory(original):
    def prepare_model_for_kbit_training_b(model, *args, **kwargs):
        _INFO["prepare_calls"] += 1
        out = original(model, *args, **kwargs)
        cast = _cast_back_to_bf16(out)
        _log(f"B: peft prepare_model_for_kbit_training ran, then cast back to bf16: {cast}")
        return out
    return prepare_model_for_kbit_training_b


def kbit_c(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None, auto_clear_cache=True):
    """Arm C: peft 0.20.0 prepare_model_for_kbit_training minus the fp32 cast block (and its empty_cache)."""
    _INFO["prepare_calls"] += 1
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}
    for _name, param in model.named_parameters():
        param.requires_grad = False
    if loaded_in_kbit and use_gradient_checkpointing:
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    _log(f"C: replacement ran: frozen base, loaded_in_kbit={bool(loaded_in_kbit)}, "
         f"gradient_checkpointing_requested={bool(use_gradient_checkpointing)}, no dtype cast")
    return model


def dtype_census(model) -> dict:
    """Parameter bytes by (group, dtype); Params4bit counted as their packed uint8 storage."""
    groups = collections.defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        low = name.lower()
        if "lora_a" in low or "lora_b" in low:
            group = "lora_AB"
        elif "embed_tokens" in low:
            group = "embed_tokens"
        elif "lm_head" in low:
            group = "lm_head"
        elif "norm" in low:
            group = "norm"
        elif low.endswith(".bias"):
            group = "bias"
        elif param.__class__.__name__ == "Params4bit":
            group = "nf4_weight"
        else:
            group = "other"
        key = f"{group} [{str(param.dtype).replace('torch.', '')}]"
        groups[key][0] += 1
        groups[key][1] += param.numel() * param.element_size()
    return {key: {"tensors": val[0], "mib": round(val[1] / MIB, 1)} for key, val in sorted(groups.items())}


def module_census(model) -> dict:
    classes = collections.Counter(type(module).__name__ for module in model.modules())
    keep = {name: count for name, count in classes.items()
            if any(tag in name for tag in ("RMSNorm", "MLP", "Rotary", "Attention", "DecoderLayer", "ForCausalLM"))}
    owner = None
    for module in model.modules():
        if type(module).__name__.endswith("ForCausalLM") and not type(module).__name__.startswith("Peft"):
            fwd = getattr(type(module), "forward", None)
            owner = {"class": type(module).__name__, "forward_module": getattr(fwd, "__module__", None),
                     "forward_name": getattr(fwd, "__name__", None)}
            break
    return {"module_classes": keep, "causal_lm_forward": owner}


def _census_record(model, tag: str) -> None:
    try:
        _INFO[tag] = {"dtypes": dtype_census(model), **module_census(model)}
    except Exception as exc:   # noqa: BLE001 - a census problem must never stop a measured run
        _INFO[tag] = {"error": f"{type(exc).__name__}: {exc}"}
    _flush()


def _install_census() -> None:
    import tco_meter
    from transformers import TrainerCallback

    class CensusCallback(TrainerCallback):
        def on_train_end(self, args, state, control, model=None, **kwargs):
            if model is not None:
                _census_record(model, "census_at_train_end")

    original_attach = tco_meter._attach

    def attach(trainer, out_dir, warmup, max_wall_s):
        original_attach(trainer, out_dir, warmup, max_wall_s)
        _census_record(trainer.model, "census_at_trainer_construction")
        _INFO["use_liger_kernel_arg"] = getattr(trainer.args, "use_liger_kernel", "<absent>")
        trainer.add_callback(CensusCallback())

    tco_meter._attach = attach


def apply(arm: str, out_dir: str) -> None:
    if arm not in ("A", "B", "C", "D"):
        raise SystemExit(f"TCO_ARM must be A, B, C or D; got {arm!r}")
    _OUT["dir"] = out_dir
    _INFO["arm"] = arm
    _INFO["python"] = sys.version.split()[0]
    import peft

    if arm == "B":
        peft.prepare_model_for_kbit_training = _kbit_b_factory(peft.prepare_model_for_kbit_training)
    elif arm == "C":
        peft.prepare_model_for_kbit_training = kbit_c
    _log(f"arm {arm} installed (patched peft.prepare_model_for_kbit_training: {arm in ('B', 'C')})")
    _install_census()
