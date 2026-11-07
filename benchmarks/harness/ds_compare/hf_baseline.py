#!/usr/bin/env python3
"""Plain Hugging Face Trainer + PEFT LoRA run: the DeepSpeed and resident arms.

No import of ``soup_cli``. This is the script an engineer would write from the Hugging Face
and DeepSpeed documentation: bf16 base weights, LoRA r=8, batch 1, one optimizer step per
row, and (arms D1..D4) a DeepSpeed config JSON. It writes ONE json per run with the schema
of ``soup_arm.py`` (arm id, argv, versions, trained tokens, train_runtime, tok/s,
steady-state tok/s over steps 33..end, torch peak reserved memory, final loss and the loss
at steps 64/128/192/256, exit status).

How it is launched (``orchestrate.py`` does exactly this)::

    # DeepSpeed arms, the launcher the Hugging Face DeepSpeed guide documents:
    deepspeed --num_gpus 1 hf_baseline.py --arm D1 --deepspeed ds_d1_z3_param_offload.json ...
    # resident arms, no launcher:
    python hf_baseline.py --arm R1 --deepspeed none ...

The launcher appends ``--local_rank=N`` to the script arguments; it is accepted and ignored
(rank and world size come from the environment the launcher sets).

Order matters for ZeRO-3 (Hugging Face DeepSpeed guide): ``TrainingArguments(deepspeed=...)``
is built BEFORE ``from_pretrained`` so the model is created under ``zero.Init`` and its
bf16 weights are partitioned (and offloaded, when configured) while they load.

Exit codes: 0 ok, 3 out of memory (written to the json as status "oom"), 1 any other error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_trace import (  # noqa: E402
    TraceState,
    collect_versions,
    derive_metrics,
    make_callback_class,
)
from ds_common import (  # noqa: E402
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR,
    MAX_GRAD_NORM,
    MODEL_ID,
    MODEL_REVISION,
    RUN_SCHEMA,
    SEED,
    WEIGHT_DECAY,
    atomic_write_json,
    load_rows,
    render_alpaca_row,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_OOM = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL_ID, help="repo id or local directory")
    parser.add_argument("--revision", default=None, help="model revision (ignored for a directory)")
    parser.add_argument("--rows", required=True, help="alpaca-format jsonl")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument(
        "--target-modules",
        default=",".join(LORA_TARGET_MODULES),
        help="comma separated; default = Soup's resolved list for Llama (ds_common.py)",
    )
    parser.add_argument("--deepspeed", default="none", help="DeepSpeed config json, or 'none'")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chat-template-file", default=None, help="override the chat template")
    parser.add_argument("--out", required=True, help="result json path")
    parser.add_argument("--arm", default="HF", help="arm id recorded in the json")
    parser.add_argument("--shape", default=None)
    parser.add_argument("--cap-gb", type=float, default=None)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="launcher")
    return parser.parse_args(argv)


class RowsDataset:
    """Pre-tokenized rows (lists of ints); ``__getitem__`` returns a feature dict."""

    def __init__(self, features: list[dict[str, list[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class CountingCollator:
    """Right-pads a batch and counts the non-padding tokens actually fed to the model."""

    def __init__(self, trace: TraceState, pad_token_id: int) -> None:
        self.trace = trace
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in features)
        input_ids, labels, attention = [], [], []
        for item in features:
            pad = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
        self.trace.add_tokens(int(batch["attention_mask"].sum()))
        return batch


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "outofmemoryerror" in text or "out of memory" in text


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def _resolved_deepspeed_config(trainer: Any) -> Any:
    """The DeepSpeed config after Hugging Face filled every "auto" (best effort)."""
    try:
        plugin = trainer.accelerator.state.deepspeed_plugin
        config = plugin.deepspeed_config
        return dict(config) if config is not None else None
    except Exception:  # noqa: BLE001
        return None


def run(
    args: argparse.Namespace, result: dict[str, Any], trace_path: str, holder: list[TraceState]
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    is_dir = os.path.isdir(args.model)
    revision = None if is_dir else (args.revision or None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if args.chat_template_file:
        tokenizer.chat_template = Path(args.chat_template_file).read_text(encoding="utf-8")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    rows = load_rows(args.rows)
    features = [render_alpaca_row(tokenizer, row, args.max_length) for row in rows]
    lengths = [len(item["input_ids"]) for item in features]
    steps_planned = args.max_steps if args.max_steps > 0 else len(features) * args.epochs
    expected_tokens = sum(lengths[index % len(lengths)] for index in range(steps_planned))
    result["data"] = {
        "rows_file": os.path.basename(args.rows),
        "rows": len(rows),
        "tokens_per_row_min": min(lengths),
        "tokens_per_row_max": max(lengths),
        "expected_total_tokens": expected_tokens,
        "max_length": args.max_length,
        "epochs": args.epochs,
    }

    ds_path = None if args.deepspeed in ("none", "", None) else args.deepspeed
    work_dir = tempfile.mkdtemp(prefix="ds_compare_hf_")
    try:
        train_args = TrainingArguments(
            output_dir=work_dir,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            lr_scheduler_type="constant",
            warmup_steps=0,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            optim="adamw_torch",
            bf16=args.dtype == "bf16",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=args.seed,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if args.gradient_checkpointing else None
            ),
            deepspeed=ds_path,
        )

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype)
        except TypeError:  # transformers 4.x spelling
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=revision, torch_dtype=dtype
            )
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

        targets = [name for name in args.target_modules.split(",") if name]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        wrapped = sum(1 for name, _ in model.named_modules() if name.endswith(".lora_A"))
        result["lora"] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": targets,
            "wrapped_modules": wrapped,
            "trainable_parameters": trainable,
            "total_parameters": total,
        }

        trace = TraceState(trace_path)
        holder.append(trace)
        callback = make_callback_class(trace)()
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=RowsDataset(features),
            data_collator=CountingCollator(trace, pad_token_id),
            callbacks=[callback],
        )
        try:
            trainer.train()
        finally:
            result["resolved_deepspeed_config"] = _resolved_deepspeed_config(trainer)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "kind": "hf",
        "arm": args.arm,
        "shape": args.shape,
        "cap_gb": args.cap_gb,
        "rep": args.rep,
        "argv": sys.argv,
        "model": {"id": args.model, "revision": args.revision or MODEL_REVISION},
        "deepspeed_config_file": None if args.deepspeed == "none" else args.deepspeed,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "status": "failed",
        "exit_code": EXIT_ERROR,
        "error": None,
    }
    out_path = Path(args.out)
    trace_path = str(out_path.with_suffix(".trace.json"))
    holder: list[TraceState] = []
    code = EXIT_ERROR
    try:
        run(args, result, trace_path, holder)
        result["status"] = "ok"
        code = EXIT_OK
    except BaseException as exc:  # noqa: BLE001 - every failure is recorded, then re-raised as a code
        result["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-2000:]
        result["traceback_tail"] = traceback.format_exc()[-4000:]
        if _is_oom(exc):
            result["status"] = "oom"
            code = EXIT_OOM
        print(f"[hf_baseline] {result['status']}: {result['error']}", file=sys.stderr)
    result["exit_code"] = code
    if _is_rank_zero():
        raw_trace = None
        if holder and holder[0].begin_perf is not None:
            holder[0].flush()
            raw_trace = holder[0].to_dict()
        if raw_trace is None:
            result["versions"] = collect_versions()
        else:
            expected = (result.get("data") or {}).get("expected_total_tokens")
            result.update(derive_metrics(raw_trace, expected_total_tokens=expected))
            result["versions"] = raw_trace.get("versions")
            result["trace_args"] = raw_trace.get("args")
            result["pytorch_cuda_alloc_conf"] = raw_trace.get("pytorch_cuda_alloc_conf")
        atomic_write_json(out_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
