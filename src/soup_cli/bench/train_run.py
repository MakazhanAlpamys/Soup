"""Run a short, fixed-length training job and measure it (#836).

This is the wiring between a real trainer and the two pure halves: the
collector records, :mod:`soup_cli.bench.train_report` judges. Scope is the
transformers SFT trainer, the path every recipe's ``soup train`` takes by
default; other tasks and MLX are refused by name rather than measured through
a path this has not been checked against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from importlib import metadata
from typing import Any, Callable, Optional

_VERSIONED = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate")


def bench_scope_error(cfg: Any) -> Optional[str]:
    """Why this config cannot be benchmarked here, or ``None`` when it can."""
    if cfg.task != "sft":
        return f"task: {cfg.task} -- bench train measures the SFT trainer only"
    if getattr(cfg, "backend", "transformers") != "transformers":
        return (
            f"backend: {cfg.backend} -- bench train measures the transformers "
            f"trainer only"
        )
    return None


def config_hash(resolved: dict) -> str:
    """sha256 of the fully resolved config, so a schema-default change that moves
    the run is visible in the report (#716)."""
    blob = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _versions() -> dict:
    found = {}
    for name in _VERSIONED:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def _device_provenance(torch: Any, device: str) -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {"device": device, "card": None, "cuda_runtime": None, "sm": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "card": props.name,
        "cuda_runtime": torch.version.cuda,
        "sm": f"{props.major}.{props.minor}",
    }


def run_bench_train(
    cfg: Any,
    *,
    steps: int,
    warmup: int,
    device: str,
    load_dataset: Callable[[Any], dict],
    before_train: Optional[Callable[[Any], None]] = None,
) -> dict:
    """Train ``cfg`` for exactly ``steps`` optimizer steps and build the report.

    ``before_train`` receives the wrapper after setup; tests use it to break the
    run on purpose (freeze everything, zero the gradients) so the checks are
    shown to fire through the real trainer and not only on hand-built records.
    """
    import torch

    from soup_cli.bench.collector import BenchCollector, summarize_trainable
    from soup_cli.bench.train_report import check_trainable
    from soup_cli.trainer.sft import SFTTrainerWrapper

    if steps <= warmup:
        raise ValueError(
            f"--steps {steps} leaves nothing after --warmup {warmup}: raise "
            f"--steps or lower --warmup."
        )

    with tempfile.TemporaryDirectory(prefix="soup-bench-") as scratch:
        # Log every step (the grad_norm check reads each one), never save, and
        # never write into the config's own output directory.
        training = cfg.training.model_copy(
            update={"logging_steps": 1, "save_steps": steps + 1}
        )
        cfg = cfg.model_copy(update={"training": training, "output": scratch})

        wrapper = SFTTrainerWrapper(cfg, device=device)
        wrapper.setup(load_dataset(cfg))
        trainer = wrapper.trainer
        trainer.args.max_steps = steps
        if hasattr(trainer.args, "eval_strategy"):
            trainer.args.eval_strategy = "no"

        collector = BenchCollector()
        if device == "cuda" and torch.cuda.is_available():
            collector._sync = torch.cuda.synchronize
        trainer.add_callback(collector)

        # Tokens come from what training_step receives: that is the batch the
        # loss sees, and it runs in this process -- a collator wrapper would run
        # inside dataloader workers and count into their copy.
        original_step = trainer.training_step

        def training_step(model, inputs, *args, **kwargs):
            collector.observe_batch(inputs)
            return original_step(model, inputs, *args, **kwargs)

        trainer.training_step = training_step

        if before_train is not None:
            before_train(wrapper)

        # With nothing to train, the Trainer dies inside autograd with a
        # message that names no cause. The builder's own rule, asked first.
        nothing = check_trainable(summarize_trainable(trainer.model))
        if nothing is not None:
            raise ValueError(nothing["message"])

        on_cuda = device == "cuda" and torch.cuda.is_available()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        trainer.train()
        if on_cuda:
            memory = {
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        else:
            memory = {"max_memory_allocated_bytes": None, "max_memory_reserved_bytes": None}

        resolved = cfg.model_dump(mode="json")
        resolved.pop("output", None)  # the scratch directory, different every run
        model = trainer.model
        provenance = {
            **_device_provenance(torch, device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": _versions(),
            "dtype": str(getattr(model, "dtype", None)),
            "optimizer": str(trainer.args.optim),
            "seed": trainer.args.seed,
            "data_seed": trainer.args.data_seed,
        }
        return collector.build_report(
            warmup_steps=warmup,
            provenance=provenance,
            memory=memory,
            config_hash=config_hash(resolved),
            steps_requested=steps,
            extra={"resolved_config": resolved},
        )
