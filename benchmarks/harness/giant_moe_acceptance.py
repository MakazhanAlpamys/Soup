"""Run the giant MoE acceptance benchmark against a real DeepSeek-V3 checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Iterable
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from benchmarks.harness.giant_moe_source_plan import (
    MODEL_TYPE,
    meta_construction,
    require_complete_source,
    source_shards,
    validate_plan,
    validate_writer_plan,
    writer_geometry,
)

if TYPE_CHECKING:
    from soup_cli.utils.layer_shard import ShardIndex


def reuse_cache(cache: Path, *, reuse_requested: bool, restart_incomplete: bool) -> bool:
    index = cache / "index.json"
    if reuse_requested:
        if not index.is_file():
            raise ValueError("--reuse-cache and --resume require a committed cache index")
        return True
    if cache.exists() and not (restart_incomplete and cache.is_dir() and not index.exists()):
        raise ValueError("cache already exists: use --reuse-cache")
    return False


def validate_cache(cache: Path, index: ShardIndex, n_layers: int, budget: int) -> int:
    if index.total_params < 500_000_000_000:
        raise ValueError("giant MoE pilot requires at least 500B source parameters")
    if (
        index.arch != MODEL_TYPE
        or index.n_layers != n_layers
        or index.expert_count != 256
        or index.expert_layers != tuple(range(3, n_layers))
    ):
        raise ValueError("unexpected DeepSeek-V3 cache geometry")
    cache_bytes = sum(path.stat().st_size for path in cache.iterdir() if path.is_file())
    if cache_bytes > budget:
        raise ValueError(f"cache exceeds planned budget: {cache_bytes} > {budget} bytes")
    return cache_bytes


def run(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be nonnegative")
    checkpoint_every = args.checkpoint_every or int(bool(args.resume))
    if checkpoint_every and (args.evaluation_protocol or args.evaluation_data):
        raise ValueError("checkpoint and resume options apply to the standalone giant MoE pilot")
    if args.resume and args.restart_incomplete_cache:
        raise ValueError(
            "--resume requires a committed cache and cannot use --restart-incomplete-cache"
        )
    if bool(args.evaluation_protocol) != bool(args.evaluation_data):
        raise ValueError("both evaluation protocol and frozen data are required")

    source, cache, output = Path(args.source), Path(args.cache), Path(args.output)
    reuse = reuse_cache(
        cache,
        reuse_requested=args.reuse_cache or bool(args.resume),
        restart_incomplete=args.restart_incomplete_cache,
    )
    resume_path = None
    if args.resume:
        from benchmarks.harness.giant_moe_training_checkpoint import find_training_checkpoint

        resume_path = find_training_checkpoint(Path(args.resume))

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("giant MoE requires a real CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("giant MoE pilot requires one visible CUDA GPU in the 8 GB class")
    props = torch.cuda.get_device_properties(0)
    if not 7000 * 2**20 <= props.total_memory <= 8500 * 2**20:
        raise RuntimeError("giant MoE pilot requires one visible CUDA GPU in the 8 GB class")

    from peft import LoraConfig, TaskType, load_peft_weights, set_peft_model_state_dict
    from transformers import AutoTokenizer

    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from soup_cli.utils.layer_shard import (
        checkpoint_source_components,
        fingerprint_source_files,
        inspect_shard_cache,
        shard_checkpoint,
        source_identity_includes_config,
    )
    from soup_cli.utils.layer_stream_runtime import (
        assert_streamed_adapter_saved,
        build_streamed_model,
    )
    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS

    if "expert_pool" not in signature(build_streamed_model).parameters:
        raise RuntimeError("giant MoE requires the pooled expert streaming runtime")
    console = Console()
    output.mkdir(parents=True, exist_ok=False)
    events = output / "events.jsonl"

    def emit(kind: str, **values: Any) -> None:
        import psutil

        memory = psutil.Process().memory_info()
        row = {
            "event": kind,
            "unix": time.time(),
            "rss_bytes": memory.rss,
            "host_ram_bytes": psutil.virtual_memory().total,
            "disk_free_bytes_now": shutil.disk_usage(output).free,
            **values,
        }
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    row["peak_rss_bytes"] = int(line.split()[1]) * 1024
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        phase_file = getattr(args, "ram_phase_file", None)
        phase_map = {
            "environment": "preparation",
            "training_start": "training",
            "evaluation_before_start": "evaluation_before",
            "evaluation_after_start": "evaluation_after",
            "reload_start": "reload",
        }
        if phase_file and kind in phase_map:
            Path(phase_file).write_text(phase_map[kind], encoding="utf-8")
        console.print(row)

    evaluation_protocol = None
    evaluation_examples = []
    if args.evaluation_protocol:
        from benchmarks.harness.giant_moe_evaluation import load_evaluation

        evaluation_protocol, evaluation_examples = load_evaluation(
            args.evaluation_protocol,
            args.evaluation_data,
        )
        if (
            evaluation_protocol["model_revision"] != args.revision
            or evaluation_protocol["max_length"] != args.max_length
            or args.steps != 2
        ):
            raise ValueError("model revision or token limit differs from the protocol")

    data_bytes = Path(args.data).read_bytes()
    texts = json.loads(data_bytes.decode("utf-8"))
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    if not isinstance(texts, dict) or any(
        not isinstance(texts.get(split), list)
        or not texts[split]
        or any(not isinstance(item, str) or not item.strip() for item in texts[split])
        for split in ("train", "heldout")
    ):
        raise ValueError("data JSON requires separate train and heldout strings")
    if evaluation_protocol:
        canonical = json.dumps(texts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if (
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            != evaluation_protocol["training_data_canonical_sha256"]
        ):
            raise ValueError("training data differ from the frozen protocol")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261028)
    # Use FP32 on older cards to avoid FP16 range/scaling ambiguity in the new pilot.
    dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float32"
    paths = source_shards(source)
    files = []
    for path in paths:
        stat = path.stat()
        files.append((path.name, stat.st_size, stat.st_mtime_ns))
    require_complete_source([size for _, size, _ in files])
    source_plan = (
        validate_plan(source, Path(args.source_plan), args.revision) if args.source_plan else None
    )
    budget = (
        source_plan["cache_budget_bytes"]
        if source_plan
        else new_expert_cache_budget([str(path) for path in paths])
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache.parent).free
    if free < budget and not reuse:
        raise ValueError(f"new cache requires {budget} free bytes, available {free}")
    emit(
        "environment",
        gpu=props.name,
        vram_bytes=props.total_memory,
        dtype=dtype,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cache_budget_bytes=budget,
        disk_free_bytes=free,
        source_revision=args.revision,
    )
    if evaluation_protocol:
        emit(
            "evaluation_protocol",
            protocol_sha256=hashlib.sha256(Path(args.evaluation_protocol).read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256(Path(args.evaluation_data).read_bytes()).hexdigest(),
            example_ids=evaluation_protocol["example_ids"],
        )

    suffixes, n_layers = writer_geometry(source, dtype)
    if source_plan:
        validate_writer_plan(source_plan, suffixes, n_layers, dtype)
        emit(
            "source_plan",
            sha256=hashlib.sha256(Path(args.source_plan).read_bytes()).hexdigest(),
            cache_budget_bytes=budget,
            generic_cache_budget_bytes=source_plan["generic_cache_budget_bytes"],
            n_layers=n_layers,
            quant_suffixes=sorted(suffixes),
        )
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.shard_threads)
        if reuse:
            components = checkpoint_source_components(
                str(source),
                tuple(files),
                include_config=source_identity_includes_config(MODEL_TYPE),
            )
            index, reason = inspect_shard_cache(
                str(cache),
                dtype,
                fingerprint_source_files(components),
                components,
                "nf4",
                False,
                "cuda",
                arch=MODEL_TYPE,
                expert_layout="per_expert",
            )
            if index is None:
                raise ValueError(f"existing cache failed validation: {reason}")
        else:
            index = shard_checkpoint(
                str(source),
                str(cache),
                dtype=dtype,
                arch=MODEL_TYPE,
                num_hidden_layers=n_layers,
                quant="nf4",
                quant_suffixes=suffixes,
                quant_device="cuda",
                double_quant=False,
                expert_layout="per_expert",
            )
    finally:
        torch.set_num_threads(previous_threads)
    cache_bytes = validate_cache(cache, index, n_layers, budget)
    emit(
        "sharded",
        seconds=time.perf_counter() - started,
        cache_reused=reuse,
        shard_threads=args.shard_threads,
        training_threads=previous_threads,
        source_parameters=index.total_params,
        cache_bytes=cache_bytes,
    )
    lora = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0,
        bias="none",
        target_modules=list(MOE_TEXT_LORA_TARGETS[MODEL_TYPE]),
        task_type=TaskType.CAUSAL_LM,
    )

    def build() -> tuple[Any, Any]:
        with meta_construction():
            model, runtime = build_streamed_model(
                model_id=str(source),
                shard_dir=str(cache),
                index=index,
                lora_config=lora,
                device="cuda",
                dtype=dtype,
                buffers=2,
                pin=False,
                quant="nf4",
                double_quant=False,
                tier="disk",
                read_ahead=1,
                expert_pool=True,
            )
        return model, runtime

    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=False)
    emit("data", sha256=data_sha256, max_length=args.max_length)

    checkpoint_identity = {}
    if checkpoint_every:
        tokenizer_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
        )
        checkpoint_identity = {
            "revision": args.revision,
            "cache_index_sha256": hashlib.sha256((cache / "index.json").read_bytes()).hexdigest(),
            "data_sha256": data_sha256,
            "tokenizer": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in tokenizer_files
                if (source / name).is_file()
            },
            "dtype": dtype,
            "max_length": args.max_length,
            "steps": args.steps,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": props.name,
        }

    def batch(text: str) -> dict[str, Any]:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)[
            "input_ids"
        ].to("cuda")
        return {"input_ids": ids, "labels": ids}

    def adapter_hash(params: Iterable[Any]) -> str:
        digest = hashlib.sha256()
        for parameter in params:
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    validation = batch(texts["heldout"][0])
    model, runtime = build()
    try:
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        parameters = list(trainable.values())
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        start_step = 0
        if resume_path is not None:
            from benchmarks.harness.giant_moe_training_checkpoint import restore_training_checkpoint

            start_step, before = restore_training_checkpoint(
                resume_path,
                trainable,
                optimizer,
                identity=checkpoint_identity,
                device="cuda",
            )
            emit("resume", checkpoint=str(resume_path), completed_steps=start_step)
        else:
            model.eval()
            with torch.no_grad():
                before = float(model(**validation).loss)
            if not math.isfinite(before):
                raise RuntimeError("non-finite heldout loss before training")
        emit("heldout_before", loss=before)
        evaluation_before = []
        if evaluation_examples:
            from benchmarks.harness.giant_moe_evaluation import evaluate_examples

            zero_b = [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and ".lora_B." in name
            ]
            if not zero_b or any(torch.count_nonzero(param).item() for param in zero_b):
                raise RuntimeError("baseline requires zero-initialized adapter B matrices")
            emit("unadapted_baseline", zero_lora_b_tensors=len(zero_b))
            del zero_b
            evaluation_before = evaluate_examples(model, evaluation_examples, batch, emit, "before")
        emit("training_start")
        gradients = []
        new_hash = adapter_hash(parameters) if start_step == args.steps else ""
        for step in range(start_step, args.steps):
            model.train()
            inputs = batch(texts["train"][step % len(texts["train"])])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model(**inputs)
            if not torch.isfinite(result.loss):
                raise RuntimeError("non-finite training loss")
            result.loss.backward()
            gradients = [param.grad for param in parameters if param.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise RuntimeError("missing or non-finite adapter gradients")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("adapter gradient norm is zero or non-finite")
            old_hash = adapter_hash(parameters)
            optimizer.step()
            new_hash = adapter_hash(parameters)
            if new_hash == old_hash:
                raise RuntimeError("optimizer step did not change the adapter")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            emit(
                "step",
                step=step,
                loss=float(result.loss),
                gradient_norm=norm,
                adapter_sha256_before=old_hash,
                adapter_sha256_after=new_hash,
                gradient_tensors=len(gradients),
                seconds=seconds,
                tokens=int(inputs["input_ids"].numel()),
                tokens_per_second=inputs["input_ids"].numel() / seconds,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
            completed = step + 1
            if checkpoint_every and (completed % checkpoint_every == 0 or completed == args.steps):
                from benchmarks.harness.giant_moe_training_checkpoint import (
                    save_training_checkpoint,
                )

                checkpoint = save_training_checkpoint(
                    output / "checkpoints",
                    trainable,
                    optimizer,
                    identity=checkpoint_identity,
                    completed_steps=completed,
                    heldout_before=before,
                    device="cuda",
                )
                emit("checkpoint", path=str(checkpoint), completed_steps=completed)
        model.eval()
        with torch.no_grad():
            result = model(**validation)
            logits = result.logits.cpu()
            if not torch.isfinite(result.loss):
                raise RuntimeError("non-finite heldout loss after training")
            emit("heldout_after", loss=float(result.loss))
        if evaluation_examples:
            from benchmarks.harness.giant_moe_evaluation import paired_summary

            evaluation_after = evaluate_examples(model, evaluation_examples, batch, emit, "after")
            quality = paired_summary(
                evaluation_before,
                evaluation_after,
                expected_ids=evaluation_protocol["example_ids"],
            )
            (output / "evaluation-summary.json").write_text(
                json.dumps(quality, indent=2),
                encoding="utf-8",
            )
            emit("evaluation_summary", **quality)
        from safetensors.torch import save_file

        save_file({"logits": logits.contiguous()}, str(output / "reference-logits.safetensors"))
        adapter = output / "adapter"
        model.save_pretrained(adapter)
        assert_streamed_adapter_saved(model, str(adapter))
        emit("runtime", stats=runtime.stats())
        emit("expert_io", stats=runtime.expert_pool.stats())
    finally:
        runtime.close()
    del model, runtime, optimizer, trainable, parameters, gradients, result
    gc.collect()
    torch.cuda.empty_cache()
    emit("reload_start")
    model, runtime = build()
    try:
        state = load_peft_weights(str(adapter), device="cpu")
        set_peft_model_state_dict(model, state)
        reloaded_hash = adapter_hash([param for param in model.parameters() if param.requires_grad])
        if reloaded_hash != new_hash:
            raise RuntimeError("reloaded adapter parameters differ from the trained parameters")
        model.eval()
        with torch.no_grad():
            reloaded = model(**validation).logits.cpu()
        equal = torch.equal(logits, reloaded)
        error = float((logits.float() - reloaded.float()).abs().max())
        emit(
            "reload",
            equal=equal,
            max_abs_error=error,
            parameter_digest_matches=True,
            parameter_sha256=reloaded_hash,
        )
        if not equal:
            raise RuntimeError("reloaded logits differ")
    finally:
        runtime.close()
    emit("complete")


if __name__ == "__main__":
    import faulthandler
    import signal

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output", "data", "revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--shard-threads", type=int, default=8)
    parser.add_argument("--evaluation-protocol")
    parser.add_argument("--evaluation-data")
    parser.add_argument("--ram-phase-file")
    parser.add_argument("--source-plan")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="STEPS",
        help="Save training state every N optimizer steps and at the final step",
    )
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT_OR_DIR",
        help="Resume from a checkpoint file or a previous run's checkpoints dir",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--restart-incomplete-cache", action="store_true")
    cache_mode.add_argument("--reuse-cache", action="store_true")
    run(parser.parse_args())
