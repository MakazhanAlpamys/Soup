"""MLX SFT trainer — Apple Silicon fine-tuning via mlx-lm.

Rewritten for mlx-lm >= 0.31 (v0.73.0 fix, PR #362):
- ``mlx_lm.tuner.trainer.train`` no longer takes a ``tokenizer`` argument;
  datasets are built with ``mlx_lm.tuner.datasets.create_dataset``, wrapped in
  ``CacheDataset`` (bare ``ChatDataset`` lacks ``__len__``/``__getitem__``),
  and loss is reported through ``TrainingCallback.on_train_loss_report``.
- LoRA layers are applied with ``mlx_lm.tuner.utils.linear_to_lora_layers``
  after ``model.freeze()``: ``mlx_lm.load`` does NOT freeze, and without it
  every parameter is trainable and the saved "adapter" is actually a full
  fine-tune (172 tensors vs 24 LoRA tensors on a 1.2B model).
- Implements the CLI trainer contract: ``train()`` accepts the common kwargs
  (display/tracker/run_id/resume_from_checkpoint) and returns the metrics
  dict that ``commands/train.py`` expects.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


def _count_safetensors_tensors(path: str) -> int:
    """Number of tensors declared in a ``.safetensors`` file's own header.

    Parsed directly from the format (an 8-byte little-endian length prefix
    followed by that many bytes of JSON metadata) rather than via mlx or
    the ``safetensors`` package, so this runs on any machine regardless of
    whether either is installed. ``__metadata__`` is the one header key
    that isn't a tensor.

    ``path`` is untrusted here: the ``--resume`` direct-path branch accepts
    any existing file, and ``--hf-resume`` can point at a directory. The
    length prefix is bounded against the file's own size before it's used
    to size a read, and every failure mode (garbage length, truncated or
    non-JSON content, a directory instead of a file) collapses to one
    ``ValueError`` naming the path, instead of a raw ``MemoryError``,
    ``JSONDecodeError``, or ``IsADirectoryError`` reaching the caller.
    """
    try:
        file_size = Path(path).stat().st_size
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            if not 0 < header_len <= file_size:
                raise ValueError(
                    f"declared header length {header_len} is invalid for a "
                    f"{file_size}-byte file"
                )
            header = json.loads(f.read(header_len))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"MLX checkpoint {path} is not a readable safetensors file: {exc}"
        ) from exc
    return sum(1 for key in header if key != "__metadata__")


#: What `target_modules: auto` means on MLX. peft resolves `auto` per
#: architecture; mlx-lm has no equivalent, so the streamed-down default is
#: attention Q/V — the modules `_apply_lora` has always actually trained.
MLX_DEFAULT_TARGET_KEYS = ["self_attn.q_proj", "self_attn.v_proj"]


def resolve_mlx_target_keys(lora_cfg: object) -> list[str]:
    """THE answer to "which modules does this MLX run train?" (#392).

    Both the trainer and the ``adapter_config.json`` writer must use it. They
    used to resolve separately — ``_apply_lora`` into a local, the writer not at
    all — so a default run trained Q/V and then shipped ``{"keys": ["auto"]}``.
    ``linear_to_lora_layers`` matches nothing against that and
    ``load_weights(strict=False)`` drops every LoRA tensor in silence, leaving
    an adapter whose generations are bit-identical to the base model.
    """
    raw = getattr(lora_cfg, "target_modules", None)
    if not raw or raw in (["auto"], "auto"):
        return list(MLX_DEFAULT_TARGET_KEYS)
    return list(raw) if isinstance(raw, list) else [raw]


def build_mlx_adapter_config(lora_cfg: object, *, adapter_path: str, **extra: object) -> dict:
    """The ``lora_parameters`` block, keyed off the RESOLVED module list.

    Separate from the writer so a test can assert the file's ``keys`` against
    :func:`resolve_mlx_target_keys` without running a training loop — the
    disagreement between those two values is the whole of #392.
    """
    rank = int(getattr(lora_cfg, "r", 0))
    alpha = float(getattr(lora_cfg, "alpha", 0.0))
    config: dict = {
        "fine_tune_type": "lora",
        "adapter_path": adapter_path,
        "lora_parameters": {
            "rank": rank,
            "scale": alpha / max(1.0, float(rank)),
            "dropout": float(getattr(lora_cfg, "dropout", 0.0) or 0.0),
            "keys": resolve_mlx_target_keys(lora_cfg),
        },
    }
    config.update(extra)
    return config


class MLXSFTTrainerWrapper:
    """High-level wrapper for MLX supervised fine-tuning."""

    def __init__(self, config: SoupConfig, **kwargs) -> None:
        self.config = config
        # Accepted for CLI-contract parity (trainer_kwargs forward). Stored
        # intentionally unused: trust_remote_code has no MLX meaning and
        # load_mlx_model does not take it.
        self.extra_kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _require_mlx(self) -> None:
        try:
            import mlx  # noqa: F401
            import mlx.core  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MLX backend requires the 'mlx' and 'mlx-lm' packages. "
                "Install with: pip install \"soup-cli[mlx]\""
            ) from exc

    def _check_unsupported(self) -> None:
        tcfg = self.config.training
        unsupported = []
        if tcfg.quantization == "8bit":
            unsupported.append("quantization=8bit (use mlx-community 4bit models)")
        if tcfg.use_galore:
            unsupported.append("GaLore")
        if tcfg.use_ring_attention:
            unsupported.append("Ring Attention")
        if tcfg.use_flash_attn:
            unsupported.append("FlashAttention (MLX has its own attention kernels)")
        # #353's fourth criterion. #381 threaded training.seed through every
        # transformers task wrapper; MLX has its own RNG (mx.random) and reads
        # neither field, so a seeded MLX run is silently unseeded. `is not None`
        # rather than truthiness: 0 is a real seed.
        if tcfg.seed is not None:
            unsupported.append("training.seed (MLX seeds through mx.random)")
        if tcfg.data_seed is not None:
            unsupported.append("training.data_seed (MLX seeds through mx.random)")
        if unsupported:
            console.print(
                "[yellow]MLX backend ignores: " + ", ".join(unsupported) + "[/]"
            )

    def setup(self, dataset: dict) -> None:
        """Load MLX model, configure LoRA, prepare dataset."""
        self._require_mlx()
        self._check_unsupported()

        from soup_cli.utils.mlx import load_mlx_model

        cfg = self.config
        console.print(f"[dim]Loading MLX model: {cfg.base}[/]")
        self.model, self.tokenizer = load_mlx_model(
            cfg.base, quantization=cfg.training.quantization
        )
        console.print(
            f"[green]MLX model loaded:[/] {cfg.base} "
            f"(task={cfg.task}, lora_r={cfg.training.lora.r})"
        )
        self._dataset = dataset

    def _apply_lora(self, model) -> None:
        """Convert the configured linear layers to LoRA (mlx-lm >= 0.31).

        The model must be frozen first: ``mlx_lm.load`` does NOT freeze, and
        without it every parameter is trainable and the saved "adapter" is
        actually a full fine-tune (172 tensors instead of ~64 LoRA weights).
        """
        model.freeze()
        from mlx_lm.tuner.utils import linear_to_lora_layers

        cfg = self.config
        lora_cfg = cfg.training.lora
        target_keys = resolve_mlx_target_keys(lora_cfg)
        if target_keys == MLX_DEFAULT_TARGET_KEYS and lora_cfg.target_modules in (
            None,
            [],
            "auto",
            ["auto"],
        ):
            console.print(
                "[yellow]MLX backend: target_modules: auto cannot be resolved for "
                f"all architectures; defaulting to {target_keys}[/]"
            )
        keys = set(target_keys)
        num_layers = len(getattr(model, "layers", []))
        linear_to_lora_layers(
            model,
            num_layers,
            {
                "rank": int(lora_cfg.r),
                "scale": float(lora_cfg.alpha) / max(1.0, float(lora_cfg.r)),
                "dropout": float(getattr(lora_cfg, "dropout", 0.0) or 0.0),
                "keys": keys,
            },
        )

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """Warm-start LoRA weights from a saved MLX checkpoint (#634).

        Must run after ``_apply_lora`` — the saved file holds only the
        LoRA-shaped tensors, which don't exist on the model until the linear
        layers have been converted.

        This restores adapter WEIGHTS only. mlx-lm's LoRA trainer exposes no
        optimizer state and no step/iteration count, so it is a warm start,
        not a resume of training state: the step count and data position
        both restart from zero regardless of how far the checkpoint got.
        Say so rather than implying a full resume. Replaying the dataset
        from the saved iteration is a separate, harder claim — it needs a
        reproducible iteration order tied to training.seed/data_seed, which
        the MLX path does not thread yet (#353) — and is out of scope here.

        ``strict=False`` means a checkpoint saved under a different
        ``lora.r`` or ``target_modules`` drops every tensor in silence —
        exactly the #392 failure mode this file's own
        ``resolve_mlx_target_keys`` docstring records. What's checked here
        is the narrower, MLX-independent half of that: the checkpoint FILE
        itself declares at least one tensor before ``load_weights`` ever
        runs, so an empty or corrupt checkpoint fails loudly instead of
        producing a warm start from nothing that still prints two green
        messages. Confirming that the declared tensors actually match this
        model's LoRA-shaped parameter names — the other half — needs mlx
        itself to introspect, which isn't available to verify here.
        """
        tensor_count = _count_safetensors_tensors(checkpoint_path)
        if tensor_count == 0:
            raise ValueError(
                f"MLX checkpoint {checkpoint_path} declares no tensors; refusing "
                "to warm-start from an empty or corrupt checkpoint file"
            )
        console.print(
            f"[green]MLX: loading checkpoint weights from[/] {checkpoint_path} "
            f"({tensor_count} tensors)"
        )
        self.model.load_weights(checkpoint_path, strict=False)

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        """Run MLX training loop via mlx-lm (mlx-lm >= 0.31 API).

        ``display`` / ``tracker`` / ``run_id`` drive Soup's live dashboard the
        way they do on the transformers path (#23). This is an adapter, not a
        reuse of ``SoupTrainerCallback``: that is a HuggingFace
        ``TrainerCallback`` wanting ``args, state, control``, while mlx-lm
        offers only ``on_train_loss_report`` / ``on_val_loss_report``, so
        bridging through it would couple this path to HF trainer internals.
        """

        self._require_mlx()

        import mlx.optimizers as optim  # type: ignore
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.datasets import CacheDataset, create_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train  # type: ignore

        cfg = self.config
        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "MLX backend: setup(dataset) must be called before train()"
            )

        self._apply_lora(self.model)

        if resume_from_checkpoint is not None:
            self._load_checkpoint_weights(resume_from_checkpoint)

        batch_size = (
            int(cfg.training.batch_size)
            if isinstance(cfg.training.batch_size, int)
            else 1
        )
        train_rows = list(self._dataset.get("train", []))
        val_rows = list(self._dataset.get("val", []))
        iters = int(
            cfg.training.epochs * max(1, math.ceil(len(train_rows) / batch_size))
        )

        max_seq_length = int(getattr(cfg.data, "max_length", 2048) or 2048)
        steps_per_report = int(getattr(cfg.training, "logging_steps", 10) or 10)
        steps_per_save = int(getattr(cfg.training, "save_steps", 0) or 0)
        if steps_per_save <= 0:
            steps_per_save = iters
        # Real eval cadence when a val split exists; otherwise the value is
        # irrelevant (val_dataset stays None and mlx-lm skips evaluation).
        steps_per_eval = max(1, iters // 4) if val_rows else max(1000, iters + 1)
        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            max_seq_length=max_seq_length,
            steps_per_report=steps_per_report,
            steps_per_eval=steps_per_eval,
            steps_per_save=steps_per_save,
            adapter_file=str(output_dir / "adapters.safetensors"),
        )

        train_dataset = CacheDataset(create_dataset(train_rows, self.tokenizer, args))
        val_dataset = (
            CacheDataset(create_dataset(val_rows, self.tokenizer, args))
            if val_rows
            else None
        )

        optimizer = optim.AdamW(learning_rate=float(cfg.training.lr))

        captured: dict = {}
        total_epochs = float(cfg.training.epochs)

        class _Callback(TrainingCallback):
            """mlx-lm's two hooks, adapted onto Soup's display and tracker.

            Keys are mlx-lm's own, built at ``mlx_lm/tuner/trainer.py``:
            ``iteration``, ``train_loss``, ``learning_rate``,
            ``tokens_per_second``, ``peak_memory``. There is deliberately no
            ``grad_norm`` — mlx-lm does not compute one for the callback, and a
            dashboard field reading a plausible 0.0 on this backend while
            carrying a real value on another is worse than an absent one.
            """

            def on_train_loss_report(self, train_info: dict) -> None:
                loss = train_info.get("train_loss", 0.0)
                captured.setdefault("losses", []).append(loss)
                if display is None:
                    return

                step = int(train_info.get("iteration", 0) or 0)
                epoch = (step / iters * total_epochs) if iters else 0.0
                lr_value = train_info.get("learning_rate", cfg.training.lr)
                # TrainingDisplay hard-labels this "it/s" (display.py:116), and
                # the transformers path feeds it train_steps_per_second. Feeding
                # tokens_per_second here renders a ~50x number under an it/s label.
                speed = train_info.get("iterations_per_second", 0.0)
                peak = train_info.get("peak_memory")
                gpu_mem = f"{peak:.3f} GB" if isinstance(peak, (int, float)) else ""

                display.update(
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    lr=lr_value,
                    speed=speed,
                    gpu_mem=gpu_mem,
                )
                if tracker is not None and run_id:
                    tracker.log_metrics(
                        run_id=run_id,
                        step=step,
                        epoch=epoch,
                        loss=loss,
                        lr=lr_value,
                        speed=speed,
                        gpu_mem=gpu_mem,
                    )

        if display is not None:
            display.start(iters)

        t0 = time.time()
        try:
            train(
                model=self.model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                args=args,
                training_callback=_Callback(),
            )
        finally:
            # A Live display left attached would corrupt the terminal if
            # mlx-lm raises, so this is a finally rather than a trailing call.
            if display is not None:
                display.stop()
        duration = time.time() - t0

        # mlx-lm's tuner only saves adapters.safetensors; write the
        # adapter_config.json so the output dir is directly loadable with
        # mlx_lm.load(..., adapter_path=output).
        lora_cfg = cfg.training.lora
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "model": str(cfg.base),
                    "num_layers": len(getattr(self.model, "layers", [])),
                    "batch_size": batch_size,
                    "iters": iters,
                    "learning_rate": float(cfg.training.lr),
                    "steps_per_report": steps_per_report,
                    "steps_per_eval": steps_per_eval,
                    "steps_per_save": steps_per_save,
                    "max_seq_length": max_seq_length,
                    "adapter_path": str(output_dir),
                    # #392: the RESOLVED module list, never the raw `auto`.
                    "lora_parameters": build_mlx_adapter_config(
                        lora_cfg, adapter_path=str(output_dir)
                    )["lora_parameters"],
                    "mask_prompt": False,
                    "grad_checkpoint": False,
                    "grad_accumulation_steps": 1,
                },
                indent=2,
            )
        )

        losses = captured.get("losses") or [0.0]
        result = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "total_steps": iters,
            "duration_secs": duration,
            "duration": f"{duration:.0f}s",
            "output_dir": str(output_dir),
        }
        console.print(f"[green]MLX training complete:[/] {output_dir}")
        return result
