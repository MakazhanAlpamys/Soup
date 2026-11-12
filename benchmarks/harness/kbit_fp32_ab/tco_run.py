#!/usr/bin/env python3
"""Driver for the Soup-vs-Unsloth TCO.  Usually invoked through run_tco.sh.

    tco_run.py plan      <case> <framework> [--card NAME --usd-per-hour X ...]   print, touch nothing
    tco_run.py preflight <case> <framework>                                      check the rented box
    tco_run.py prepare-data <case> [--synthetic]                                 build the dataset
    tco_run.py run       <case> <framework> --repeat K [--mode window|epoch]      one measured run
    tco_run.py selftest                                                          CPU smoke of the meter

``run`` starts a worker process (the training), watches the GPU from outside, and ends by
writing result.json from the append-only files the worker's meter left behind.  If the worker
dies, or the session does, those files are still there and summarize.py rebuilds from them.
It never creates, stops or destroys a rented instance; that is the operator's step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tco_common as C  # noqa: E402


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------
def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def tco_out() -> str:
    return os.environ.get("TCO_OUT", os.path.join(HERE, "tco-out"))


def sh(cmd: list, timeout: int = 20) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (done.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def pinned_versions() -> dict:
    """name -> pinned version from requirements-tco.txt (only '==' pins)."""
    pins = {}
    with open(os.path.join(HERE, "requirements-tco.txt"), encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;]+)$", line)
            if match:
                pins[match.group(1).lower().replace("_", "-")] = match.group(3)
    pins["torch"] = "2.11.0"
    return pins


def installed_versions() -> tuple[dict, dict]:
    pins, found, drift = pinned_versions(), {}, {}
    names = list(pins) + ["xformers", "triton", "torchao", "typer", "click", "numpy"]
    for name in names:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            found[name] = None
    for name, want in pins.items():
        got = found.get(name)
        if got is None or not (got == want or got.split("+")[0] == want):
            drift[name] = {"pinned": want, "installed": got}
    found["python"] = platform.python_version()
    return found, drift


def hardware_info(cpu_only: bool = False) -> dict:
    smi = "" if cpu_only else sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pcie.link.gen.current,"
              "pcie.link.width.current,power.limit,clocks.max.sm", "--format=csv,noheader,nounits",
              "-i", "0"])
    parts = [p.strip() for p in smi.split(",")] if smi else []
    info = {"gpu_name": parts[0] if parts else None,
            "gpu_total_mib": float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit() else None,
            "driver": parts[2] if len(parts) > 2 else None,
            "pcie_gen": parts[3] if len(parts) > 3 else None,
            "pcie_width": parts[4] if len(parts) > 4 else None,
            "power_limit_w": parts[5] if len(parts) > 5 else None,
            "max_sm_clock_mhz": parts[6] if len(parts) > 6 else None,
            "nvidia_smi_cuda": None, "cpu": None, "cpu_threads": os.cpu_count(),
            "ram_gb": None, "memlock_ulimit": None, "os": platform.platform()}
    header = "" if cpu_only else sh(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    info["nvidia_smi_cuda"] = match.group(1) if match else None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", encoding="utf-8") as handle:
            info["ram_gb"] = round(int(handle.readline().split()[1]) / 1048576, 1)
    except OSError:
        pass
    try:
        import resource
        info["memlock_ulimit"] = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    except Exception:  # noqa: BLE001
        pass
    return info


def sample_host(pid: int, acc: dict) -> None:
    """Worker RSS and system MemAvailable every few seconds (Linux).  Soup's RAM tier pins the whole
    NF4 store in host memory; that is a cost of case 2 that VRAM numbers never show."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    acc["worker_rss_peak_mib"] = max(acc["worker_rss_peak_mib"], int(line.split()[1]) / 1024)
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024
                    cur = acc["mem_available_min_mib"]
                    acc["mem_available_min_mib"] = avail if cur is None else min(cur, avail)
        acc["samples"] += 1
    except OSError:
        pass


def order_path(case_id: str, mode: str, seed: int, rows: int) -> str:
    return os.path.join(tco_out(), "orders", f"{case_id}-{mode}-{rows}rows-seed{seed}.json")


def soup_order_from_rewind(rewind_path: str) -> dict | None:
    """Soup's own per-step record of which rows each batch held (rewind.jsonl, kind=batch).

    A CPU dry run of this protocol showed that, for the same seed and data_seed, Soup 0.77.2 and a
    plain trl SFTTrainer visit the SAME rows in a DIFFERENT order (Soup left data_seed's effect out
    of the sampler it wraps).  So Soup's order is the reference: the other arms replay it.
    """
    rows, header = [], None
    for rec in C.read_jsonl(rewind_path):
        if rec.get("kind") == "header":
            header = rec
        elif rec.get("kind") == "batch":
            rows.append((rec["step"], rec.get("micro", 0), rec["rows"]))
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], r[1]))
    flat = [i for _, _, batch in rows for i in batch]
    return {"rows": flat, "n_rows_header": (header or {}).get("n_rows"), "steps": len(rows),
            "source": "soup rewind.jsonl"}


def card_slug(name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name or "cpu").strip("-").lower()


# ------------------------------------------------------------------------------------------
# configs for each arm
# ------------------------------------------------------------------------------------------
def case_steps(case: dict, mode: str) -> tuple[int, int, int]:
    """(warmup, measured, rows) for window or epoch mode."""
    warm = case["warmup_steps"]
    if mode == "epoch":
        rows = case["epoch_rows"]
        steps = rows // case["batch"]
        return warm, steps - warm, rows
    return warm, case["window_steps"], (warm + case["window_steps"]) * case["batch"]


def soup_yaml(case: dict, seed: int, data_path: str, out_dir: str, batch: int | None = None,
              max_length: int | None = None) -> str:
    h = C.HYPERS
    lines = [
        f"base: {case['model']}",
        "task: sft",
        "data:",
        f"  train: {data_path}",
        "  format: chatml",
        # Deviation D12: Soup's default is data.train_on_responses_only: true (loss on assistant tokens only);
        # the protocol matches "loss on every non-pad token in every arm", so it must be switched off explicitly.
        "  train_on_responses_only: false",
        "  val_split: 0.0",
        f"  max_length: {max_length or h['max_length']}",
        "training:",
        "  epochs: 1",
        f"  lr: {h['lr']}",
        f"  batch_size: {batch or case['batch']}",
        f"  gradient_accumulation_steps: {h['grad_accum']}",
        f"  seed: {seed}",
        f"  data_seed: {seed}",   # explicit: with data_seed unset HF draws the sampler seed from the global RNG
        f"  scheduler: {h['lr_scheduler']}",
        "  warmup_ratio: 0.0",
        f"  weight_decay: {h['weight_decay']}",
        f"  max_grad_norm: {h['max_grad_norm']}",
        f"  optimizer: {h['optimizer']}",
        "  gradient_checkpointing: true",
        "  quantization: 4bit",
        "  logging_steps: 1",
        "  save_steps: 1000000",
        "  lora:",
        f"    r: {h['lora_r']}",
        f"    alpha: {h['lora_alpha']}",
        f"    dropout: {h['lora_dropout']}",
        "    target_modules: [" + ", ".join(h["lora_targets"]) + "]",
    ]
    if os.environ.get("TCO_ARM") == "D":   # P6-2609: arm D = arm A + training.use_liger (the only yaml difference)
        lines.insert(lines.index("  quantization: 4bit") + 1, "  use_liger: true")
    if case["stream"]:
        lines += ["  stream_layers: true", "  stream_buffers: 2", "  stream_source: auto",
                  "  stream_pin: true",
                  # AMENDMENT-1: explicit (the schema default is "on" = the shipped full checkpoint recompute)
                  f"  stream_recompute: {os.environ.get('TCO_STREAM_RECOMPUTE', 'on')}"]
    lines.append(f"output: {out_dir}")
    return "\n".join(lines) + "\n"


def plan_for(case_id: str, framework: str, mode: str, args) -> dict:
    case = C.CASES[case_id]
    warm, measured, rows = case_steps(case, mode)
    cap_wall = case["epoch_max_wall_s"] if mode == "epoch" else case["max_wall_s"]
    cap_usd = case["epoch_max_usd"] if mode == "epoch" else case["max_usd"]
    price = args.usd_per_hour
    tokens_step = case["batch"] * C.HYPERS["max_length"]
    plan = {
        "case": case_id, "label": case["label"], "framework": framework, "mode": mode,
        "model": case["model"], "streaming": bool(case["stream"] and framework == "soup"),
        "batch": case["batch"], "max_length": C.HYPERS["max_length"],
        "rows": rows, "warmup_steps": warm, "measured_steps": measured,
        "tokens_per_step_expected": tokens_step,
        "tokens_measured_expected": tokens_step * measured,
        "tokens_total_expected": tokens_step * (warm + measured),
        "seeds": case["seeds"], "repeats": case["repeats"],
        "dataset": C.DATASET["name"] + " / " + C.DATASET["split"],
        "caps": {"max_wall_s": cap_wall, "max_usd": cap_usd, "usd_per_hour_used": price,
                 "effective_wall_cap_s": C.wall_cap_seconds(cap_wall, cap_usd, price)},
        "matched_hyperparameters": C.HYPERS,
        "pins": pinned_versions(),
    }
    if framework == "soup":
        plan["soup_yaml"] = soup_yaml(case, case["seeds"][0], "<TCO_OUT>/data/train.jsonl",
                                      "<TCO_OUT>/run/soup_out")
    return plan


def print_plan(plan: dict) -> None:
    print(f"== PLAN  case {plan['case']}  framework {plan['framework']}  mode {plan['mode']}")
    print(f"   {plan['label']}")
    for key in ("model", "streaming", "batch", "max_length", "rows", "warmup_steps",
                "measured_steps", "tokens_per_step_expected", "tokens_measured_expected",
                "tokens_total_expected", "seeds", "repeats", "dataset"):
        print(f"   {key:<26} {plan[key]}")
    caps = plan["caps"]
    print(f"   caps: {caps['max_wall_s']} s and ${caps['max_usd']:.2f}; at "
          f"${caps['usd_per_hour_used']}/h the effective wall cap is "
          f"{caps['effective_wall_cap_s']:.0f} s" if caps["usd_per_hour_used"] else
          f"   caps: {caps['max_wall_s']} s and ${caps['max_usd']:.2f} (no hourly price given: "
          f"wall cap only)")
    print("   matched hyperparameters:")
    for key, val in plan["matched_hyperparameters"].items():
        print(f"     {key:<24} {val}")
    print("   pinned versions: " + ", ".join(f"{k}=={v}" for k, v in plan["pins"].items()))
    if "soup_yaml" in plan:
        print("   soup.yaml that would be written:")
        for line in plan["soup_yaml"].splitlines():
            print("     | " + line)


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
def prepare_data(case_id: str, synthetic: bool, out_root: str, model_override: str | None = None,
                 max_length: int | None = None, need_rows: int | None = None) -> dict:
    case = C.CASES[case_id]
    max_len = max_length or C.HYPERS["max_length"]
    warm, measured, window_rows = case_steps(case, "window")
    n_rows = max(need_rows or 0, window_rows, 64)
    data_dir = os.path.join(out_root, "data", case_id)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "train.jsonl")
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path) and os.path.exists(path):
        old = json.load(open(manifest_path, encoding="utf-8"))
        if old.get("rows") >= n_rows and old.get("model") == (model_override or case["model"]) \
                and old.get("synthetic") == synthetic and old.get("max_length") == max_len:
            return old
    t0 = time.time()
    rows, scanned = [], 0
    model = model_override or case["model"]
    if synthetic:
        import random
        rng = random.Random(1234)
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi".split()
        while len(rows) < n_rows:
            scanned += 1
            question = " ".join(rng.choice(words) for _ in range(8))
            answer = " ".join(rng.choice(words) for _ in range(max_len * 2))
            rows.append({"messages": [{"role": "user", "content": question},
                                      {"role": "assistant", "content": answer}]})
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        stream = load_dataset(C.DATASET["name"], split=C.DATASET["split"], streaming=True)
        for row in stream:
            scanned += 1
            msgs = row["messages"]
            enc = tok.apply_chat_template(msgs, tokenize=True)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            head = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
            head_ids = head["input_ids"] if hasattr(head, "keys") else head
            # total >= max_len so every row is exactly max_len after truncation; the first user turn
            # must leave >= 128 tokens for the assistant (Soup refuses a row whose answer is truncated away)
            if len(ids) >= max_len and len(head_ids) <= max_len - 128:
                rows.append({"messages": msgs})
            if len(rows) >= n_rows:
                break
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"path": path, "rows": len(rows), "scanned": scanned, "model": model,
                "synthetic": synthetic, "max_length": max_len, "sha256": C.sha256_file(path),
                "dataset": C.DATASET["name"], "split": C.DATASET["split"], "rule": C.DATASET["rule"],
                "built_utc": now_utc(), "build_s": round(time.time() - t0, 1)}
    atomic_write(manifest_path, manifest)
    return manifest


def slice_rows(src: str, dst: str, n_rows: int) -> None:
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n_rows:
                break
            fout.write(line)


# ------------------------------------------------------------------------------------------
# the worker: one framework's training, in its own process, with the meter installed
# ------------------------------------------------------------------------------------------
def worker(args) -> int:
    case = C.CASES[args.case]
    model_name = args.model or case["model"]
    out_dir = args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TCO_FRAMEWORK"] = args.framework   # tco_profile.py needs to know which arm it is in
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if args.framework in ("soup", "hf"):
        # ARM ISOLATION (deviation D9): the pilot found that `soup train` imports unsloth whenever it is
        # installed (commands/train.py "Tip: unsloth is installed"), which patches trl in the Soup process:
        # padding_free is auto-enabled and Soup's rewind recorder switches itself off.  With one shared venv the
        # Soup arm was therefore "Soup + Unsloth patches".  None in sys.modules makes `import unsloth` raise
        # ImportError and find_spec return None, i.e. the soup and hf arms see a venv without unsloth.
        for _name in ("unsloth", "unsloth_zoo"):
            sys.modules[_name] = None
    h = C.HYPERS
    batch = args.batch or case["batch"]
    max_len = args.max_length or h["max_length"]
    # AMENDMENT-1 knobs (all default to the original protocol behaviour)
    gc_on = os.environ.get("TCO_GC", "on") != "off"                 # "off": no gradient checkpointing in the unsloth/hf arms
    unsloth_exact = os.environ.get("TCO_UNSLOTH_EXACT", "0") == "1"  # load the named repo as-is, no pre-quantised remap

    if args.framework == "soup":
        import tco_meter
        yaml_text = soup_yaml(case, args.seed, args.data, os.path.join(out_dir, "soup_out"),
                              batch=args.batch, max_length=args.max_length)
        if args.model:
            yaml_text = yaml_text.replace(f"base: {case['model']}", f"base: {model_name}")
        if args.no_quant:   # CPU selftest only: 4-bit needs CUDA
            yaml_text = yaml_text.replace("quantization: 4bit", "quantization: none")
        yaml_path = os.path.join(out_dir, "soup.yaml")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        tco_meter.install(out_dir, args.warmup, args.wall_cap_s)
        import p6_arms   # P6-2609: scratch arms B/C patch peft.prepare_model_for_kbit_training; A and D change nothing there
        p6_arms.apply(os.environ.get("TCO_ARM", "A"), out_dir)
        sys.argv = ["soup", "train", "--config", yaml_path, "--yes"]
        from soup_cli.cli import app
        try:
            app()
        except SystemExit as exc:
            return int(exc.code or 0) if isinstance(exc.code, int) or exc.code is None else 1
        return 0

    import torch
    use_cuda = torch.cuda.is_available() and not args.cpu
    quant = not args.no_quant
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=args.data, split="train")
    if args.order_file:
        with open(args.order_file, encoding="utf-8") as handle:
            order = json.load(handle)["rows"]
        dataset = dataset.select(order)    # rows in the order Soup visited them

    if args.framework == "unsloth":
        from unsloth import FastLanguageModel  # noqa: F401  (must precede transformers/trl/peft)
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_len, dtype=None, load_in_4bit=quant,
            use_exact_model_name=unsloth_exact)
        model = FastLanguageModel.get_peft_model(
            model, r=h["lora_r"], target_modules=h["lora_targets"], lora_alpha=h["lora_alpha"],
            lora_dropout=h["lora_dropout"], bias="none",
            use_gradient_checkpointing=("unsloth" if gc_on else False),
            random_state=args.seed)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"dtype": dtype}
        if quant:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        if use_cuda:
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc_on)
        model = get_peft_model(model, LoraConfig(
            r=h["lora_r"], lora_alpha=h["lora_alpha"], lora_dropout=h["lora_dropout"],
            target_modules=h["lora_targets"], task_type="CAUSAL_LM"))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import tco_meter
    from trl import SFTConfig, SFTTrainer
    tco_meter.install(out_dir, args.warmup, args.wall_cap_s)   # AFTER unsloth/trl are imported
    cfg = SFTConfig(
        output_dir=os.path.join(out_dir, "hf_out"), num_train_epochs=1,
        per_device_train_batch_size=batch, gradient_accumulation_steps=h["grad_accum"],
        learning_rate=h["lr"], lr_scheduler_type=h["lr_scheduler"], warmup_steps=0,
        weight_decay=h["weight_decay"], max_grad_norm=h["max_grad_norm"], optim=h["optimizer"],
        logging_steps=1, save_strategy="no", report_to=[], seed=args.seed, data_seed=args.seed,
        bf16=bf16, fp16=bool(use_cuda and not bf16), use_cpu=not use_cuda,
        gradient_checkpointing=gc_on, max_length=max_len, packing=False, padding_free=False,
        assistant_only_loss=False, dataloader_num_workers=0)
    trainer_cls = SFTTrainer
    if args.order_file:
        import torch.utils.data as torch_data

        class OrderedSFTTrainer(SFTTrainer):   # no shuffling: the dataset is already in Soup's order
            def _get_train_sampler(self, *a, **k):
                return torch_data.SequentialSampler(self.train_dataset)

        trainer_cls = OrderedSFTTrainer
    trainer = trainer_cls(model=model, args=cfg, train_dataset=dataset, processing_class=tok)
    if not tco_meter._STATE["attached"]:   # a framework that re-patched Trainer after install(): attach by hand
        tco_meter._attach(trainer, out_dir, args.warmup, args.wall_cap_s)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "adapter"))
    return 0


# ------------------------------------------------------------------------------------------
# run: launch worker, watch from outside, write result.json
# ------------------------------------------------------------------------------------------
def load_price(args) -> dict:
    if args.price_json:
        snap = json.load(open(args.price_json, encoding="utf-8"))
        entry = snap["cards"].get(args.card or "")
        if not entry or entry.get("median_dph_total") is None:
            raise SystemExit(f"no price for card {args.card!r} in {args.price_json}")
        return {"usd_per_hour": entry["median_dph_total"], "card": args.card,
                "definition": snap.get("price_definition"), "retrieved_utc": snap["retrieved_utc"],
                "source_command": entry.get("command"), "n_offers": entry.get("n_used"),
                "snapshot_file": os.path.basename(args.price_json),
                "actual_billed_usd_per_hour": args.actual_usd_per_hour}
    if args.usd_per_hour is not None:
        return {"usd_per_hour": args.usd_per_hour, "card": args.card,
                "definition": "operator-supplied", "retrieved_utc": args.price_date,
                "source_command": args.price_source, "n_offers": None, "snapshot_file": None,
                "actual_billed_usd_per_hour": args.actual_usd_per_hour}
    return {"usd_per_hour": None, "card": args.card, "definition": "none given",
            "actual_billed_usd_per_hour": args.actual_usd_per_hour}


def run(args) -> int:
    case = C.CASES[args.case]
    mode = args.mode
    warm, measured, rows = case_steps(case, mode)
    seed = case["seeds"][(args.repeat - 1) % len(case["seeds"])]
    hw = hardware_info(cpu_only=args.cpu)
    slug = card_slug(hw["gpu_name"])
    base = os.path.join(tco_out(), "runs", args.case + ("-epoch" if mode == "epoch" else ""),
                        args.framework, slug, f"rep{args.repeat}")
    if os.path.exists(os.path.join(base, "result.json")) and not args.force:
        print(f"result already at {base}; use --force to rerun")
        return 0
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base)
    price = load_price(args)
    cap_wall = case["epoch_max_wall_s"] if mode == "epoch" else case["max_wall_s"]
    cap_usd = case["epoch_max_usd"] if mode == "epoch" else case["max_usd"]
    wall_cap = C.wall_cap_seconds(cap_wall, cap_usd, price.get("usd_per_hour"))
    versions, drift = installed_versions()

    t_prep = time.time()
    manifest = prepare_data(args.case, args.synthetic, tco_out(), args.model, args.max_length,
                            need_rows=rows)
    data_prep_s = round(time.time() - t_prep, 1)
    data_file = os.path.join(os.path.dirname(manifest["path"]),
                             f"train_{mode}_{warm + measured}x{case['batch']}.jsonl")
    if not os.path.exists(data_file):
        slice_rows(manifest["path"], data_file, rows)

    order_file = order_path(args.case, mode, seed, rows)
    order_source = None
    if args.framework != "soup":
        if os.path.exists(order_file):
            order_source = "soup rewind order: " + os.path.basename(order_file)
        else:
            order_source = "own default order (no Soup order file: rule C1 will fail)"
    hf_home = os.environ.get("TCO_HF_HOME") or os.path.join(tco_out(), "hf-" + args.framework)   # AMENDMENT-2: one shared cache per box
    cold = not os.path.isdir(os.path.join(hf_home, "hub"))
    expected_step = (args.batch or case["batch"]) * (args.max_length or C.HYPERS["max_length"])
    meta = {
        "case": args.case, "framework": args.framework, "repeat": args.repeat, "seed": seed,
        "mode": mode, "model": args.model or case["model"], "batch": args.batch or case["batch"],
        "max_length": args.max_length or C.HYPERS["max_length"],
        "warmup_steps": warm, "window_steps": measured, "expected_tokens_per_step": expected_step,
        "hardware": hw, "gpu_total_mib": hw["gpu_total_mib"], "versions": versions,
        "version_drift": drift, "price": price, "cold_cache": cold,
        "caps": {"max_wall_s": cap_wall, "max_usd": cap_usd, "effective_wall_cap_s": wall_cap},
        "stream": ({"stream_layers": True, "stream_buffers": 2, "stream_pin": True}
                   if (case["stream"] and args.framework == "soup") else None),
        "data": {k: manifest[k] for k in ("rows", "sha256", "dataset", "synthetic", "scanned")},
        "requested": {"hyperparameters": C.HYPERS, "launched_utc": now_utc(),
                      "amendment1_env": {k: os.environ.get(k) for k in ("TCO_GC", "TCO_STREAM_RECOMPUTE", "TCO_UNSLOTH_EXACT", "TCO_ARM", "TCO_PROFILE")},
                      "tco_run_args": vars(args)},
        "data_prep_s": data_prep_s, "setup_s": args.setup_s, "download_s": None,
        "order_source": order_source,
    }
    atomic_write(os.path.join(base, "run_meta.json"), meta)

    env = dict(os.environ, HF_HOME=hf_home, TCO_OUT=tco_out(), PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0" if not args.cpu else "-1"))
    cmd = [sys.executable, os.path.abspath(__file__), "worker", args.case, args.framework,
           "--run-dir", base, "--data", data_file, "--seed", str(seed), "--warmup", str(warm),
           "--wall-cap-s", str(wall_cap)]
    for flag, val in (("--model", args.model), ("--batch", args.batch),
                      ("--max-length", args.max_length)):
        if val:
            cmd += [flag, str(val)]
    if args.cpu:
        cmd += ["--cpu"]
    if args.no_quant:
        cmd += ["--no-quant"]
    if order_source and order_source.startswith("soup"):
        cmd += ["--order-file", order_file]

    import tco_meter
    baseline = None
    poller = None
    if not args.cpu and shutil.which("nvidia-smi"):
        smi = sh(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        baseline = float(smi) if smi.replace('.', '', 1).isdigit() else 0.0
        poller = tco_meter.GpuPoller(os.path.join(base, "gpu.json"))
        poller.start()
    atomic_write(os.path.join(base, "status.json"), {"phase": "worker_running", "utc": now_utc()})
    host = {"worker_rss_peak_mib": 0.0, "mem_available_min_mib": None, "samples": 0}
    t_start = time.time()
    meta["t_worker_start_unix"] = t_start
    atomic_write(os.path.join(base, "run_meta.json"), meta)   # a stop mid-run still leaves the start time
    log_path = os.path.join(base, "worker.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=base)
        pretrain_cap = args.pretrain_cap_s
        rc = None
        ticks = 0
        while rc is None:
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sample_host(proc.pid, host)
                ticks += 1
                if ticks % 6 == 0:   # every ~30 s: keep the on-disk copy current for a pull after a stop
                    if poller is not None:
                        atomic_write(os.path.join(base, "gpu.json"), {
                            "memory": poller.summary(baseline or 0.0), "partial": True,
                            "clock_power": {k: v for k, v in poller.summary(baseline or 0.0).items()
                                            if k.startswith(("sm_clock", "power", "temp"))}})
                    meta["host_memory"] = host
                    atomic_write(os.path.join(base, "run_meta.json"), meta)
                begun = os.path.exists(os.path.join(base, "train_begin.json"))
                if not begun and time.time() - t_start > pretrain_cap:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    rc = -15
                    meta["pretrain_capped_after_s"] = pretrain_cap
    meta["t_worker_end_unix"] = time.time()
    if args.framework == "soup":
        found = soup_order_from_rewind(os.path.join(base, "soup_out", "rewind.jsonl"))
        if found and found["steps"] >= warm + measured:
            os.makedirs(os.path.dirname(order_file), exist_ok=True)
            atomic_write(order_file, found)
            meta["order_written"] = order_file
        else:
            meta["order_written"] = None   # no complete Soup order: the other arms will report C1 failing
    meta["host_memory"] = host
    meta["worker_rc"] = rc
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        meta["log_tail"] = handle.read()[-4000:]
    if poller is not None:
        poller.halt = True
        poller.join(timeout=3)
        # P6-2609: keep the raw samples (unix time, mem MiB, SM clock MHz, power W, temp C, util %) so the SM clock
        # can be taken over the measured window only; the summary below spans the whole run, idle model load included.
        atomic_write(os.path.join(base, "gpu_samples.json"),
                     {"cols": ["unix", "mem_mib", "sm_mhz", "power_w", "temp_c", "util_pct"], "rows": poller.samples})
        atomic_write(os.path.join(base, "gpu.json"), {
            "memory": poller.summary(baseline or 0.0),
            "clock_power": {k: v for k, v in poller.summary(baseline or 0.0).items()
                            if k.startswith(("sm_clock", "power", "temp"))}})
    atomic_write(os.path.join(base, "run_meta.json"), meta)
    result = C.rebuild_result(base)
    atomic_write(os.path.join(base, "result.json"), result)
    atomic_write(os.path.join(base, "status.json"), {"phase": "done", "utc": now_utc(),
                                                       "status": result["status"]})
    thr = result["throughput"]
    print(f"[{result['status']}] case {args.case} {args.framework} rep{args.repeat} "
          f"tok/s(wall)={thr['tok_s_wall']} peak_alloc_mib={result['memory']['peak_alloc_mib']} "
          f"-> {os.path.join(base, 'result.json')}")
    if result["status"] in ("failed",):
        print(f"   reason: {result['reason']}  (see {log_path})")
    return 0 if result["status"] in ("complete", "capped", "partial") else 2


# ------------------------------------------------------------------------------------------
# preflight: the checks that decide whether a rented box is a VALID host for the case
# ------------------------------------------------------------------------------------------
def preflight(args) -> int:
    case = C.CASES[args.case]
    hw = hardware_info()
    free_gb = round(shutil.disk_usage(tco_out() if os.path.isdir(tco_out()) else HERE).free / 1e9, 1)
    checks = {"gpu_visible": hw["gpu_name"] is not None,
              "ram_gb_ok": (hw["ram_gb"] or 0) >= case["min_ram_gb"],
              "disk_free_gb_ok": free_gb >= case["disk_gb"],
              "python_3_11_to_3_13": (3, 11) <= sys.version_info[:2] <= (3, 13)}
    out = {"case": args.case, "framework": args.framework, "hardware": hw, "disk_free_gb": free_gb,
           "need": {"ram_gb": case["min_ram_gb"], "disk_gb": case["disk_gb"]}, "checks": checks}
    try:
        import torch
        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["bf16"] = torch.cuda.is_bf16_supported()
            out["sm"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            # host<->device bandwidth, the number Soup's streaming lives on (a provider's advertised pcie_bw is a claim)
            buf = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
            dev = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
            for _ in range(2):
                dev.copy_(buf, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(8):
                dev.copy_(buf, non_blocking=True)
            torch.cuda.synchronize()
            out["h2d_pinned_gb_s"] = round(8 * (1 << 30) / (time.perf_counter() - t0) / 1e9, 2)
            checks["pinned_h2d_ge_10_gb_s"] = out["h2d_pinned_gb_s"] >= 10
            if args.framework == "soup" and case["stream"]:
                try:
                    big = torch.empty(4 << 30, dtype=torch.uint8).pin_memory()   # page-lock 4 GiB
                    out["pin_4gib"] = True
                    del big
                except Exception as exc:  # noqa: BLE001
                    out["pin_4gib"] = f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
                checks["can_page_lock_4gib"] = out.get("pin_4gib") is True
    except Exception as exc:  # noqa: BLE001
        out["torch_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["valid_host"] = all(checks.values())
    print(json.dumps(out, indent=1, default=str))
    os.makedirs(tco_out(), exist_ok=True)
    atomic_write(os.path.join(tco_out(), f"preflight-{args.case}-{args.framework}.json"), out)
    return 0 if out["valid_host"] else 3


# ------------------------------------------------------------------------------------------
# selftest: the meter and result writer on CPU with a tiny model (no GPU is touched)
# ------------------------------------------------------------------------------------------
def selftest(args) -> int:
    os.environ["TCO_OUT"] = args.out
    ns = argparse.Namespace(
        case="smoke", framework=args.framework, repeat=1, mode="window", force=True, synthetic=True,
        model=args.model, batch=2, max_length=64, cpu=True, no_quant=True, card="cpu",
        usd_per_hour=0.5, price_json=None, price_source="selftest (fixed)", price_date=now_utc(),
        actual_usd_per_hour=None, setup_s=None, pretrain_cap_s=900)
    C.CASES["smoke"] = dict(C.CASES["smoke"], warmup_steps=2, window_steps=6, batch=2)
    return run(ns)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, framework=True):
        p.add_argument("case", choices=sorted(C.CASES))
        if framework:
            p.add_argument("framework", choices=C.FRAMEWORKS)
        p.add_argument("--mode", choices=("window", "epoch"), default="window")
        p.add_argument("--card", help="GPU name of the rented card, e.g. 'RTX 4090'")
        p.add_argument("--price-json", help="price_snapshot.py output (priced by --card)")
        p.add_argument("--usd-per-hour", type=float, help="price input if no snapshot file")
        p.add_argument("--price-source", help="the command that produced --usd-per-hour")
        p.add_argument("--price-date", help="UTC date of that price")
        p.add_argument("--actual-usd-per-hour", type=float,
                       help="what the instance is really billed (the provider's instance listing)")

    p_plan = sub.add_parser("plan"); common(p_plan)
    p_pf = sub.add_parser("preflight"); common(p_pf)
    p_data = sub.add_parser("prepare-data"); common(p_data, framework=False)
    p_data.add_argument("--synthetic", action="store_true")
    p_run = sub.add_parser("run"); common(p_run)
    for p in (p_run,):
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--force", action="store_true")
        p.add_argument("--synthetic", action="store_true", help="CPU selftest data only")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--no-quant", action="store_true")
        p.add_argument("--model"); p.add_argument("--batch", type=int)
        p.add_argument("--max-length", type=int)
        p.add_argument("--setup-s", type=float)
        p.add_argument("--pretrain-cap-s", type=float, default=3600)
    p_w = sub.add_parser("worker")
    p_w.add_argument("case", choices=sorted(C.CASES)); p_w.add_argument("framework", choices=C.FRAMEWORKS)
    p_w.add_argument("--run-dir", required=True); p_w.add_argument("--data", required=True)
    p_w.add_argument("--seed", type=int, required=True); p_w.add_argument("--warmup", type=int, required=True)
    p_w.add_argument("--wall-cap-s", type=float, required=True)
    p_w.add_argument("--model"); p_w.add_argument("--batch", type=int); p_w.add_argument("--max-length", type=int)
    p_w.add_argument("--cpu", action="store_true"); p_w.add_argument("--no-quant", action="store_true")
    p_w.add_argument("--order-file")
    p_st = sub.add_parser("selftest")
    p_st.add_argument("--out", default=os.path.join(HERE, "evidence", "selftest-out"))
    p_st.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p_st.add_argument("--framework", choices=("hf", "soup"), default="hf")

    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print_plan(plan_for(args.case, args.framework, args.mode, args))
        return 0
    if args.cmd == "preflight":
        return preflight(args)
    if args.cmd == "prepare-data":
        manifest = prepare_data(args.case, args.synthetic, tco_out(),
                                need_rows=(C.CASES[args.case]["epoch_rows"] if args.mode == "epoch" else None))
        print(json.dumps(manifest, indent=1))
        return 0
    if args.cmd == "run":
        return run(args)
    if args.cmd == "worker":
        return worker(args)
    if args.cmd == "selftest":
        return selftest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
