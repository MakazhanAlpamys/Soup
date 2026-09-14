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

# Chat key for the masked dataset, matching upstream's `chat_feature` default.
_CHAT_KEY = "messages"


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


class _GradientClippingOptimizer:
    """An MLX optimizer that clips the global gradient norm before applying it.

    ``training.max_grad_norm`` is honoured by every transformers trainer and
    reached nothing on this backend: no MLX file read it, ``mlx_lm``'s
    ``TrainingArgs`` has no such field, and its trainer clips nowhere -- so the
    same config trained clipped on one backend and unclipped on the other,
    silently.

    Upstream applies gradients through exactly one call,
    ``optimizer.update(model, grad)`` (``mlx_lm/tuner/trainer.py:259``), so
    clipping is reachable by handing ``train()`` an optimizer that clips first,
    without forking the training loop. That call site is inside a function
    compiled with ``mx.compile(inputs=state, outputs=state)``
    (``trainer.py:246-248``); this proxy was measured through that same
    compiled shape rather than assumed to survive tracing.

    Everything other than ``update`` delegates, because upstream reads
    ``optimizer.state`` (``trainer.py:246``) and ``optimizer.learning_rate``
    (``trainer.py:337``) off the object it is given -- including the callable
    schedule, whose ``.step`` counter lives on the wrapped optimizer.
    """

    def __init__(self, inner: object, max_norm: float) -> None:
        # Assigned through __dict__ so __getattr__ cannot recurse on them.
        self.__dict__["_inner"] = inner
        self.__dict__["_max_norm"] = float(max_norm)

    def update(self, model: object, gradients: object) -> object:
        import mlx.optimizers as optim  # heavy: imported at call time

        gradients, _total_norm = optim.clip_grad_norm(gradients, self._max_norm)
        return self._inner.update(model, gradients)

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self.__dict__["_inner"], name, value)


def _clipping_optimizer(inner: object, max_norm: float) -> object:
    """Wrap ``inner`` so gradients are clipped at ``max_norm`` before they are
    applied. ``max_grad_norm`` is schema-validated ``gt=0``, so there is no
    "clipping off" configuration to represent."""
    return _GradientClippingOptimizer(inner, max_norm)


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
        dcfg = self.config.data
        unsupported = []
        # #683 review: the per-message `train` field has no MLX equivalent --
        # `MaskedChatDataset` supervises every assistant turn and reads no
        # per-message flag. It looked rejected only because the mutual-
        # exclusion validator fires while `train_on_responses_only` is at its
        # `true` default; set that to false and the field was dropped in
        # silence, which is the shape of the defect #683 reports.
        if getattr(dcfg, "train_on_messages_with_train_field", False):
            unsupported.append(
                "data.train_on_messages_with_train_field (MLX supervises "
                "every assistant turn; the per-message flag is not read)"
            )
        if getattr(dcfg, "train_on_prompt", False):
            unsupported.append(
                "data.train_on_prompt (MLX masks the prompt or supervises the "
                "whole sequence; there is no per-field switch)"
            )
        if tcfg.quantization == "8bit":
            unsupported.append("quantization=8bit (use mlx-community 4bit models)")
        if tcfg.use_galore:
            unsupported.append("GaLore")
        if tcfg.use_ring_attention:
            unsupported.append("Ring Attention")
        if tcfg.use_flash_attn:
            unsupported.append("FlashAttention (MLX has its own attention kernels)")
        if tcfg.use_liger:
            unsupported.append(
                "training.use_liger (Liger fused kernels have no MLX implementation)"
            )
        if tcfg.neftune_alpha is not None:
            unsupported.append(
                "training.neftune_alpha (NEFT noise is applied on the "
                "transformers training path, not MLX)"
            )
        # #353's fourth criterion. #381 threaded training.seed through every
        # transformers task wrapper; MLX has its own RNG (mx.random) and reads
        # neither field, so a seeded MLX run is silently unseeded. `is not None`
        # rather than truthiness: 0 is a real seed.
        if tcfg.seed is not None:
            unsupported.append("training.seed (MLX seeds through mx.random)")
        if tcfg.data_seed is not None:
            unsupported.append("training.data_seed (MLX seeds through mx.random)")
        if isinstance(tcfg.gradient_checkpointing, str):
            unsupported.append(
                f"gradient_checkpointing tier {tcfg.gradient_checkpointing!r} "
                "(MLX has a single on/off switch; enabling it)"
            )
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
        # mlx-lm's TrainingArgs.grad_checkpoint is a single bool with no concept
        # of a granularity tier; bool() is the same coercion commands/train.py's
        # hardware-fit predictor and layer_stream.should_enable_hf_gradient_checkpointing
        # already apply to this field on every other backend, so any non-empty
        # tier string ("selective"/"medium"/"full"/"auto") resolves to True here too.
        grad_checkpoint = bool(cfg.training.gradient_checkpointing)
        grad_accumulation_steps = int(cfg.training.gradient_accumulation_steps)
        # mlx-lm updates the optimizer only when it % accum == 0 (trainer.py) and
        # never flushes a partial group, so round iters down to a whole number of
        # groups, keeping at least one group so a small dataset does not train
        # for zero optimizer steps.
        if grad_accumulation_steps > 1:
            iters = max(
                grad_accumulation_steps,
                iters - (iters % grad_accumulation_steps),
            )
        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            max_seq_length=max_seq_length,
            steps_per_report=steps_per_report,
            steps_per_eval=steps_per_eval,
            steps_per_save=steps_per_save,
            adapter_file=str(output_dir / "adapters.safetensors"),
            grad_checkpoint=grad_checkpoint,
            grad_accumulation_steps=grad_accumulation_steps,
        )

        # #683: `data.train_on_responses_only` defaults to True and was reaching
        # nothing here -- no mask was passed, so every MLX SFT run trained on
        # system and user turns against the documented default.
        #
        # The route in is an attribute set rather than a constructor kwarg:
        # upstream reads `getattr(config, "mask_prompt", False)`
        # (`datasets.py:180`) off the args object, and `TrainingArgs` has no
        # such field, so `TrainingArgs(mask_prompt=...)` raises TypeError.
        #
        # But that only gives the right answer for prompt/completion rows.
        # `ChatDataset` masks a single prefix before `messages[-1]`, so on
        # multi-turn chat it supervises the last assistant turn and silently
        # drops the earlier ones -- a different wrong distribution, not a fix.
        # Chat rows therefore go through Soup's own per-token mask, injected
        # via `train(loss=..., iterate_batches=...)`.
        from soup_cli.trainer.mlx_masking import plan_response_masking

        responses_only = bool(getattr(cfg.data, "train_on_responses_only", False))
        plan = plan_response_masking(
            responses_only, train_rows[0] if train_rows else {}
        )
        use_token_mask = plan.token_mask
        args.mask_prompt = plan.mask_prompt
        if plan.warning:
            console.print(f"[yellow]MLX backend ignores: {plan.warning}[/]")

        if use_token_mask:
            from soup_cli.trainer.mlx_masking import (
                MaskedChatDataset,
                ResponseMaskError,
                masked_iterate_batches,
                masked_loss,
            )

            masked_train = MaskedChatDataset(
                train_rows, self.tokenizer, chat_key=_CHAT_KEY
            )
            # Probe row 0 now. `process` is otherwise called lazily by
            # `CacheDataset` from inside `train()`, so a template this cannot
            # mask -- Qwen3's, which injects its thinking block only for the
            # last assistant turn and is therefore not prefix-stable at any
            # earlier one -- surfaced after the model had loaded, LoRA was
            # applied and "Starting training..." had printed. The refusal is
            # correct; its timing was not.
            if train_rows:
                try:
                    masked_train.process(train_rows[0])
                except ResponseMaskError as exc:
                    raise ResponseMaskError(
                        f"{exc}. Set `data.train_on_responses_only: false` to "
                        "train on the full sequence on this model, or run the "
                        "recipe on the transformers backend."
                    ) from exc

            train_dataset = CacheDataset(masked_train)
            val_dataset = (
                CacheDataset(
                    MaskedChatDataset(val_rows, self.tokenizer, chat_key=_CHAT_KEY)
                )
                if val_rows
                else None
            )
            train_hooks = {"loss": masked_loss, "iterate_batches": masked_iterate_batches}
        else:
            train_dataset = CacheDataset(create_dataset(train_rows, self.tokenizer, args))
            val_dataset = (
                CacheDataset(create_dataset(val_rows, self.tokenizer, args))
                if val_rows
                else None
            )
            train_hooks = {}

        # #686: the optimizer was `AdamW(learning_rate=<scalar>)` and nothing
        # else, so `warmup_ratio`, `scheduler`, `weight_decay` and `optimizer`
        # were validated, accepted and dropped -- an MLX run silently trained a
        # different recipe from the configured one.
        #
        # The schedule counts OPTIMIZER UPDATES, not iterations: MLX calls a
        # callable learning_rate with `optimizer.step`, which advances once per
        # `optimizer.update()`, and mlx-lm calls that only every
        # `grad_accumulation_steps` iterations. Building against `iters` would
        # stretch the warmup by that factor and never reach the cosine floor.
        from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer

        total_updates = max(1, iters // max(1, grad_accumulation_steps))
        optimizer_plan = plan_optimizer(
            lr=float(cfg.training.lr),
            optimizer=str(getattr(cfg.training, "optimizer", "adamw_torch")),
            scheduler=str(getattr(cfg.training, "scheduler", "cosine")),
            warmup_ratio=float(getattr(cfg.training, "warmup_ratio", 0.0) or 0.0),
            weight_decay=float(getattr(cfg.training, "weight_decay", 0.0) or 0.0),
            total_updates=total_updates,
        )
        for _warning in optimizer_plan.warnings:
            console.print(f"[yellow]MLX backend: {_warning}[/]")
        # #749: mlx-lm never clips, so the optimizer #686 built is wrapped in
        # one that clips the global gradient norm before delegating. Applied
        # here rather than inside build_optimizer so the plan stays a pure
        # description of the schedule and the two fixes stay separable.
        optimizer = _clipping_optimizer(
            build_optimizer(optimizer_plan), float(cfg.training.max_grad_norm)
        )

        captured: dict = {}
        total_epochs = float(cfg.training.epochs)

        class _Callback(TrainingCallback):
            """mlx-lm's two hooks, adapted onto Soup's display and tracker.

            Keys are mlx-lm's own, built at ``mlx_lm/tuner/trainer.py``:
            ``iteration``, ``train_loss``, ``learning_rate``,
            ``iterations_per_second``, ``peak_memory``. ``speed`` reads
            ``iterations_per_second`` and **not** ``tokens_per_second``: the
            display hard-labels that field ``it/s``, so the token figure would
            render as a ~44x overstatement. mlx-lm reports both; only one
            belongs here. There is deliberately no
            ``grad_norm`` — mlx-lm does not compute one for the callback, and a
            dashboard field reading a plausible 0.0 on this backend while
            carrying a real value on another is worse than an absent one.
            """

            # #23: the last MEASURED validation loss, for the panel only.
            # An evaluation happens every `steps_per_eval` iterations, so the
            # panel must keep showing the last one between passes. It must NOT
            # be written to the tracker or the SSE wire on training steps --
            # that fabricates measurements that never happened, which is the
            # defect #713 was blocked on (9 persisted rows for 2 evaluations).
            _sticky_val_loss = None
            #: Last measured training values, carried into the row an
            #: evaluation creates so it reports no number that was not
            #: measured somewhere. Initial 0.0 matches the transformers
            #: callback's own initialisation (callback.py:61-63).
            _last_loss = 0.0
            _last_lr = 0.0
            _last_speed = 0.0

            @staticmethod
            def _as_float(value):
                """mlx-lm hands back a Python float (`evaluate()` calls
                `.item()`), but a future build returning an mx scalar must not
                put an unserialisable object on the SSE wire."""
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            def on_val_loss_report(self, val_info: dict) -> None:
                """mlx-lm's validation hook — the #23 checklist item.

                Payload is built at ``mlx_lm/tuner/trainer.py:310-315`` as
                ``{"iteration": it - 1, "val_loss": float, "val_time": float}``.
                Note ``it - 1``: upstream reports validation one behind the
                training counter. That is recorded as sent rather than
                corrected, so a row can be matched to upstream's own log line.
                """
                val_loss = self._as_float(val_info.get("val_loss"))
                if val_loss is None:
                    # A payload without the key is not an evaluation; recording
                    # a None row would be indistinguishable from a real one.
                    return
                type(self)._sticky_val_loss = val_loss

                # NOTE the value this can take: upstream's `it - 1` at
                # `it = 0` makes the initial evaluation report **step -1**, and
                # that negative step is written to the tracker and the wire as
                # sent. Recording upstream's own counter is deliberate -- a row
                # can be matched to its log line -- but it does surface in a
                # table users read, so it is said here rather than discovered.
                step = int(val_info.get("iteration", 0) or 0)
                epoch = (step / iters * total_epochs) if iters else 0.0

                if display is not None:
                    display.update(
                        step=step,
                        epoch=epoch,
                        loss=captured.get("losses", [0.0])[-1] if captured.get("losses") else 0.0,
                        lr=cfg.training.lr,
                        val_loss=val_loss,
                    )
                if tracker is not None and run_id:
                    try:
                        # `log_metrics` defaults loss/lr/speed to 0.0, not None,
                        # so a bare val row writes three training columns that
                        # no step measured -- over half the `loss` series at a
                        # realistic cadence. The merged transformers producer
                        # carries `_last_loss` / `_last_lr` into the row an
                        # evaluation creates (callback.py:205-207); this mirrors
                        # it, so the two backends fabricate the same nothing.
                        #
                        # An evaluation before the first training step still
                        # carries the 0.0 initial value -- there is no measured
                        # loss to carry yet -- which is exactly what the
                        # transformers path does at the same point.
                        tracker.log_metrics(
                            run_id=run_id,
                            step=step,
                            epoch=epoch,
                            loss=type(self)._last_loss,
                            lr=type(self)._last_lr,
                            speed=type(self)._last_speed,
                            val_loss=val_loss,
                        )
                    except Exception:  # noqa: BLE001 — telemetry must not kill a run
                        pass
                try:
                    from soup_cli.utils.sse_train_stream import TrainEvent
                    from soup_cli.utils.train_event_buffer import push_train_event

                    push_train_event(
                        TrainEvent(
                            type="metric",
                            step=step,
                            epoch=float(epoch),
                            val_loss=val_loss,
                        )
                    )
                except Exception:  # noqa: BLE001 — telemetry must not kill a run
                    pass

            def on_train_loss_report(self, train_info: dict) -> None:
                loss = train_info.get("train_loss", 0.0)
                captured.setdefault("losses", []).append(loss)
                # Carried for the validation row, which has no training numbers
                # of its own. Recorded BEFORE the display guard on purpose: a
                # run with no display still reaches this hook, and reading them
                # after the early return would leave every carried value at its
                # initial state for exactly the configuration that has no panel
                # to notice. (The same early-return trap that made the SSE test
                # below vacuous.)
                type(self)._last_loss = loss
                type(self)._last_lr = train_info.get("learning_rate", cfg.training.lr)
                type(self)._last_speed = train_info.get("iterations_per_second", 0.0)
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
                    # Sticky, and deliberately only here: the panel keeps the
                    # last measured validation loss between evaluations, while
                    # the tracker and the wire below receive nothing, so no row
                    # or event claims a measurement that did not happen.
                    val_loss=type(self)._sticky_val_loss,
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

                # Feed the SSE buffer so `soup ui` and GET /api/train/stream
                # show an MLX run, not just the terminal panel. Best-effort in
                # the same shape as the transformers path: any exception in
                # here must never take down training. grad_norm stays None --
                # mlx-lm does not compute one, and the event field is Optional.
                try:
                    from soup_cli.utils.sse_train_stream import TrainEvent
                    from soup_cli.utils.train_event_buffer import push_train_event

                    push_train_event(
                        TrainEvent(
                            type="metric",
                            step=step,
                            epoch=float(epoch),
                            loss=float(loss) if loss is not None else None,
                            lr=float(lr_value) if lr_value is not None else None,
                            grad_norm=None,
                        )
                    )
                except Exception:  # noqa: BLE001 — telemetry must not kill a run
                    pass

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
                **train_hooks,
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
                    # #683: the EFFECTIVE masking, not a hardcoded False.
                    # `mask_prompt` stays upstream's meaning (a single masked
                    # prefix); `response_token_mask` is Soup's per-token mask,
                    # which is what a multi-turn chat run actually used.
                    "mask_prompt": bool(args.mask_prompt),
                    "response_token_mask": use_token_mask,
                    "train_on_responses_only": responses_only,
                    "grad_checkpoint": grad_checkpoint,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    # #686: the EFFECTIVE optimizer and schedule, so an adapter
                    # records the recipe that ran rather than the one requested.
                    **optimizer_plan.as_metadata(),
                    # #749: the norm gradients were actually clipped at.
                    # Recorded because MLX honours it through a Soup-side
                    # wrapper rather than through anything mlx-lm writes, so
                    # the output dir is the only place a finished run says
                    # whether it clipped.
                    "max_grad_norm": float(cfg.training.max_grad_norm),
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
