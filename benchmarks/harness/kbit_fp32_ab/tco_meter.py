"""The ONE meter used for every arm (soup, hf, unsloth).  Imported inside the training process.

Why it exists
    The Kaggle baseline (source/kaggle-notebook) measured Soup through its own panel and the
    other two through a hand-written loop with a different warm-up rule and a different memory
    counter.  Here the training code differs per framework (that is what is being compared) but
    the measuring code is identical and is injected from outside: ``install()`` wraps
    ``transformers.Trainer.__init__`` so that whichever Trainer subclass the framework builds
    (Soup's SFTTrainer wrapper, trl's SFTTrainer, Unsloth's patched one) gets this callback.

Crash safety
    Everything is appended to files and fsynced as it happens (steps.jsonl, log.jsonl), so a
    run killed at step 140 of 205, or a session that died, still yields data: the driver and
    summarize.py rebuild the numbers from those files with the same functions.

Nothing here touches the model, the optimizer or the data; it only reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback

_STATE = {"installed": False, "trainers_seen": 0, "attached": False}


def _fsync_append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _sync() -> None:
    torch = _cuda()
    if torch is not None:
        torch.cuda.synchronize()


def count_tokens(inputs: dict) -> int:
    """Non-pad tokens in one micro-batch.  attention_mask when present; else every input_id."""
    mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if mask is not None:
        return int(mask.sum().item())
    ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    return int(ids.numel()) if ids is not None else 0


def fingerprint(inputs: dict) -> str:
    """Short hash of the token ids in one micro-batch: proves the data ORDER across arms."""
    ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if ids is None:
        return "none"
    return hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def effective_config(trainer) -> dict:
    """What the trainer ACTUALLY ended up with, read after construction (not what we asked for)."""
    args = trainer.args
    names = ("per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
             "lr_scheduler_type", "warmup_steps", "warmup_ratio", "weight_decay", "max_grad_norm",
             "optim", "seed", "data_seed", "bf16", "fp16", "gradient_checkpointing",
             "gradient_checkpointing_kwargs", "num_train_epochs", "max_steps", "max_length",
             "packing", "padding_free", "assistant_only_loss", "completion_only_loss",
             "use_liger_kernel", "torch_compile", "group_by_length", "dataloader_num_workers",
             "per_device_eval_batch_size", "eval_strategy", "save_strategy", "report_to")
    cfg = {n: getattr(args, n, "<absent>") for n in names}
    model = getattr(trainer, "model", None)
    inner = getattr(model, "base_model", model)
    mcfg = getattr(getattr(inner, "model", inner), "config", None) or getattr(model, "config", None)
    info = {
        "trainer_class": type(trainer).__mro__[0].__name__,
        "trainer_mro": [c.__name__ for c in type(trainer).__mro__[:6]],
        "attn_implementation": getattr(mcfg, "_attn_implementation", None),
        "model_dtype": str(getattr(model, "dtype", None)),
        "quantization_config": str(getattr(mcfg, "quantization_config", None))[:400],
        "is_gradient_checkpointing": getattr(model, "is_gradient_checkpointing", None),
        "unsloth_imported": sys.modules.get("unsloth") is not None,
    }
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info["trainable_params"], info["total_params_visible"] = int(trainable), int(total)
    except Exception:  # noqa: BLE001
        pass
    try:
        peft_cfg = getattr(model, "peft_config", None)
        if peft_cfg:
            first = next(iter(peft_cfg.values()))
            info["lora"] = {"r": first.r, "alpha": first.lora_alpha,
                            "dropout": first.lora_dropout,
                            "targets": sorted(first.target_modules)
                            if first.target_modules else None}
    except Exception:  # noqa: BLE001
        pass
    return {"args": cfg, "model": info}


def make_callback_class():
    from transformers import TrainerCallback

    class TcoMeter(TrainerCallback):
        def __init__(self, out_dir: str, warmup: int, max_wall_s: float):
            self.out_dir, self.warmup, self.max_wall_s = out_dir, warmup, max_wall_s
            self.steps_path = os.path.join(out_dir, "steps.jsonl")
            self.log_path = os.path.join(out_dir, "log.jsonl")
            self.t_train_begin = None
            self.cur_tokens = 0
            self.cur_fp = []
            self.t_begin = None
            self.capped = None
            self.torch = _cuda()

        # ---- called by the wrapped training_step -------------------------------------
        def observe(self, inputs: dict) -> None:
            self.cur_tokens += count_tokens(inputs)
            self.cur_fp.append(fingerprint(inputs))

        def on_train_begin(self, args, state, control, **kwargs):
            _sync()
            if self.torch is not None:
                self.torch.cuda.reset_peak_memory_stats()
            self.t_train_begin = time.perf_counter()
            _atomic_write(os.path.join(self.out_dir, "train_begin.json"), {
                "unix": time.time(), "perf": self.t_train_begin,
                "alloc_mib_at_begin": (self.torch.cuda.memory_allocated() / 2**20
                                       if self.torch else None)})

        def on_step_begin(self, args, state, control, **kwargs):
            _sync()
            self.t_begin = time.perf_counter()
            self.cur_tokens, self.cur_fp = 0, []

        def on_step_end(self, args, state, control, **kwargs):
            _sync()
            t_end = time.perf_counter()
            row = {"step": int(state.global_step), "t_begin": self.t_begin, "t_end": t_end,
                   "tokens": self.cur_tokens, "fp": "".join(self.cur_fp)[:16] if len(self.cur_fp) == 1
                   else hashlib.sha256("".join(self.cur_fp).encode()).hexdigest()[:16]}
            if self.torch is not None:
                row["alloc_peak_mib"] = round(self.torch.cuda.max_memory_allocated() / 2**20, 1)
                row["reserved_peak_mib"] = round(self.torch.cuda.max_memory_reserved() / 2**20, 1)
            _fsync_append(self.steps_path, row)
            # stop rule: one wall-clock cap (the $ cap was converted to seconds by the driver)
            if self.t_train_begin is not None and t_end - self.t_train_begin > self.max_wall_s:
                self.capped = "wall"
                control.should_training_stop = True
                _atomic_write(os.path.join(self.out_dir, "capped.json"),
                              {"reason": "wall", "after_s": t_end - self.t_train_begin,
                               "at_step": int(state.global_step)})

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            _fsync_append(self.log_path, {
                "step": int(state.global_step), "loss": float(logs["loss"]),
                "grad_norm": float(logs["grad_norm"]) if logs.get("grad_norm") is not None else None,
                "lr": float(logs["learning_rate"]) if logs.get("learning_rate") is not None else None,
                "t": time.perf_counter()})

        def on_train_end(self, args, state, control, **kwargs):
            _sync()
            _atomic_write(os.path.join(self.out_dir, "train_end.json"), {
                "unix": time.time(), "perf": time.perf_counter(),
                "global_step": int(state.global_step), "capped": self.capped,
                "alloc_peak_mib": (self.torch.cuda.max_memory_allocated() / 2**20
                                   if self.torch else None),
                "reserved_peak_mib": (self.torch.cuda.max_memory_reserved() / 2**20
                                      if self.torch else None)})

    return TcoMeter


def _attach(trainer, out_dir: str, warmup: int, max_wall_s: float) -> None:
    _STATE["trainers_seen"] += 1
    if _STATE["attached"]:
        return  # only the first Trainer built in the process is metered (the training one)
    callback = make_callback_class()(out_dir, warmup, max_wall_s)
    trainer.add_callback(callback)
    original = trainer.training_step

    def metered_training_step(model, inputs, *args, **kwargs):
        callback.observe(inputs)
        return original(model, inputs, *args, **kwargs)

    trainer.training_step = metered_training_step
    _atomic_write(os.path.join(out_dir, "effective_config.json"), effective_config(trainer))
    _STATE["attached"] = True
    if os.environ.get("TCO_PROFILE") == "1":   # diagnostic profiler (tco_profile.py); never set in the TCO record
        try:
            import tco_profile
            trainer.add_callback(tco_profile.attach(trainer, out_dir, warmup, os.environ.get("TCO_FRAMEWORK", "?")))
        except Exception:  # noqa: BLE001
            _atomic_write(os.path.join(out_dir, "profiler_error.json"), {"traceback": traceback.format_exc()[-3000:]})


def install(out_dir: str, warmup: int, max_wall_s: float) -> None:
    """Patch transformers.Trainer.__init__ once.  Call AFTER importing unsloth (it patches too)."""
    import transformers

    if _STATE["installed"]:
        return
    original_init = transformers.Trainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            _attach(self, out_dir, warmup, max_wall_s)
        except Exception:  # noqa: BLE001 - a broken meter must be loud in the result, not fatal
            _atomic_write(os.path.join(out_dir, "meter_error.json"),
                          {"traceback": traceback.format_exc()[-3000:]})

    transformers.Trainer.__init__ = patched_init
    _STATE["installed"] = True
    os.makedirs(out_dir, exist_ok=True)
    _atomic_write(os.path.join(out_dir, "meter_installed.json"),
                  {"unix": time.time(), "transformers": transformers.__version__})


class GpuPoller(threading.Thread):
    """nvidia-smi memory.used of the whole GPU twice a second, in the PARENT driver.

    Whole-GPU, not per-process: the rented box is dedicated to the run.  The baseline reading
    taken before the worker starts is subtracted by the driver.  This is the number the Kaggle
    notebook called 'GPU peak' for hf/unsloth; torch's allocator peak (steps.jsonl) is the other.
    """

    def __init__(self, path: str, gpu_index: int = 0, period_s: float = 0.5):
        super().__init__(daemon=True)
        self.path, self.gpu_index, self.period_s = path, gpu_index, period_s
        self.samples, self.halt, self.available = [], False, True

    def run(self):
        import subprocess
        while not self.halt:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", str(self.gpu_index),
                     "--query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu,"
                     "utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                mem, clk, pwr, tmp, util = [x.strip() for x in out.split(",")]
                self.samples.append((time.time(), float(mem), float(clk) if clk.replace('.', '', 1).isdigit() else None,
                                     float(pwr) if pwr.replace('.', '', 1).isdigit() else None,
                                     float(tmp) if tmp.replace('.', '', 1).isdigit() else None,
                                     float(util) if util.replace('.', '', 1).isdigit() else None))
            except Exception:  # noqa: BLE001
                self.available = False
                return
            time.sleep(self.period_s)

    def summary(self, baseline_mib: float = 0.0) -> dict:
        if not self.samples:
            return {"available": False}
        mem = [s[1] for s in self.samples]
        clocks = [s[2] for s in self.samples if s[2] is not None]
        return {"available": True, "n": len(self.samples),
                "peak_used_mib": max(mem), "baseline_mib": baseline_mib,
                "peak_over_baseline_mib": max(mem) - baseline_mib,
                "sm_clock_mhz_median": sorted(clocks)[len(clocks) // 2] if clocks else None,
                "sm_clock_mhz_min": min(clocks) if clocks else None,
                "power_w_max": max((s[3] for s in self.samples if s[3] is not None), default=None),
                "temp_c_max": max((s[4] for s in self.samples if s[4] is not None), default=None)}
