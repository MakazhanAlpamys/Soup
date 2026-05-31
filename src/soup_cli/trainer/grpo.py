"""GRPO (Group Relative Policy Optimization) trainer — wraps trl.GRPOTrainer."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


def make_grpo_trainer_variant(base_cls: type, variant: str) -> type:
    """v0.53.11 #123 — build a ``_GRPOTrainerVariant`` subclass.

    Returns a subclass of ``trl.GRPOTrainer`` whose ``compute_loss`` routes
    through :func:`soup_cli.utils.grpo_variants.apply_variant_loss`. Cached
    so multiple instantiations with the same (base, variant) share one class.

    Pure factory — no torch / trl imports at module load time. Variant
    name is normalised via ``validate_grpo_variant`` BEFORE the cache
    boundary so ``"GSPO"`` and ``"gspo"`` share one class (security review
    MEDIUM fix).
    """
    from soup_cli.utils.grpo_variants import validate_grpo_variant

    variant = validate_grpo_variant(variant)
    return _make_grpo_trainer_variant_cached(base_cls, variant)


@lru_cache(maxsize=8)
def _make_grpo_trainer_variant_cached(base_cls: type, variant: str) -> type:
    """Cached factory body — keyed on already-normalised variant."""
    from soup_cli.utils.grpo_variants import apply_variant_loss

    class _GRPOTrainerVariant(base_cls):  # type: ignore[misc, valid-type]
        """GRPOTrainer subclass that routes compute_loss through Soup's variants."""

        _soup_grpo_variant: str = variant

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # v0.53.11 review fix (code-review HIGH) — read kernel inputs
            # FIRST and only call super() as a fallback. The previous
            # ordering ran an extra forward pass on every step that was
            # discarded when the kernel produced a loss, doubling VRAM
            # peak. The TRL trainer stores per-batch tensors on ``inputs``
            # by the time compute_loss is called, so we can probe them
            # without burning a forward pass.
            #
            # Probe TRL ≥0.9 attribute name ``per_token_logps`` — drop the
            # earlier ``logits_to_keep`` probe (code-review HIGH fix:
            # ``logits_to_keep`` is a position mask, not log-probs).
            logp_new = _read_attr(inputs, "per_token_logps")
            logp_old = _read_attr(inputs, "old_per_token_logps")
            if logp_old is None:
                logp_old = _read_attr(inputs, "ref_per_token_logps")
            advantages = _read_attr(inputs, "advantages")
            completion_mask = _read_attr(inputs, "completion_mask")

            if logp_new is None or logp_old is None or advantages is None:
                # Fall back to the original loss — defence-in-depth so a
                # TRL internal rename does not crash the training loop.
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )

            beta_attr = getattr(self.args, "beta", None)
            beta = float(beta_attr) if beta_attr is not None else 0.0
            delta = getattr(self, "_soup_grpo_delta", None)
            try:
                variant_loss = apply_variant_loss(
                    self._soup_grpo_variant,
                    logp_new=logp_new,
                    logp_old=logp_old,
                    advantages=advantages,
                    beta=beta,
                    delta=delta,
                    completion_mask=completion_mask,
                )
            except (TypeError, ValueError):
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )
            if variant_loss is None:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )
            if return_outputs:
                return variant_loss, None
            return variant_loss

    _GRPOTrainerVariant.__name__ = f"_GRPOTrainerVariant_{variant}"
    return _GRPOTrainerVariant


def _read_attr(obj: Any, name: str) -> Any:
    """Read ``name`` from a mapping OR object — TRL inputs vary in shape."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        return obj.get(name)
    return getattr(obj, name, None)

class GRPOTrainerWrapper:
    """High-level wrapper for GRPO training from SoupConfig.

    GRPO generates multiple completions per prompt, scores them with a reward
    function, and optimizes using group-relative advantages. This is the approach
    used by DeepSeek-R1 for reasoning model training.

    Data format: same as SFT (messages with prompt/response) or DPO-style prompts.
    The reward_fn in config determines how completions are scored.
    """

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.trust_remote_code = trust_remote_code
        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        requires = model_requires_trust_remote_code(config.base) or False
        self._trust_remote_code = resolve_trust_remote_code(
            config.base,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _build_precision_kwargs(self) -> dict[str, bool]:
        """Resolve fp16/bf16 kwargs for GRPOConfig (v0.53.3 #128).

        Priority:
        - Non-CUDA device (CPU / MPS / XPU) → no mixed precision (both
          False). HF Trainer's fp16/bf16 kwargs are CUDA-specific; non-CUDA
          backends must use their own mixed-precision path (MPS Metal,
          XPU IPEX). Documented explicitly so future MPS work doesn't
          regress this branch silently.
        - ``grpo_fp16=True`` (CUDA) → ``fp16=True, bf16=False`` (unsloth
          parity).
        - Default CUDA → ``fp16=False, bf16=True`` (legacy v0.50.0 path).

        ``auto_mixed_precision`` is mutually exclusive with ``grpo_fp16``
        (rejected at schema load via ``_validate_grpo_fp16_amp_exclusive``);
        when only ``auto_mixed_precision`` is set, the v0.32.0 picker runs
        elsewhere in the training loop and overrides this default.
        """
        if self.device != "cuda":
            return {"fp16": False, "bf16": False}
        # grpo_fp16 is a Pydantic field with default=False; direct attribute
        # access (no getattr fallback) so a typo would fail loudly.
        if self.config.training.grpo_fp16:
            return {"fp16": True, "bf16": False}
        return {"fp16": False, "bf16": True}

    def setup(self, dataset: dict):
        """Load model, tokenizer, apply LoRA, create GRPO trainer."""
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        # v0.53.11 #123 — variant subclass override
        variant = self.config.training.grpo_variant
        if variant is not None and variant != "standard":
            GRPOTrainer = make_grpo_trainer_variant(GRPOTrainer, variant)  # noqa: N806

        # Enable Rich progress bar for HuggingFace downloads
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training
        use_unsloth = cfg.backend == "unsloth"

        # --- Load reward function ---
        from soup_cli.trainer.rewards import load_reward_fn

        reward_fn = load_reward_fn(tcfg.reward_fn)

        if use_unsloth:
            self._setup_unsloth(cfg, tcfg)
        else:
            self._setup_transformers(cfg, tcfg)

        # Ensure tokenizer has a chat template — trl's GRPOTrainer calls
        # apply_chat_template() when it detects conversational prompts (message
        # lists) and will raise ValueError if the template is missing.
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for msg in messages %}"
                "{{ msg['content'] }}\n"
                "{% endfor %}"
            )

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        console.print(
            f"[green]LoRA applied:[/] {trainable:,} trainable"
            f" / {total:,} total ({pct:.2f}%)"
        )

        # --- Batch size ---
        batch_size = tcfg.batch_size
        if batch_size == "auto":
            from soup_cli.utils.gpu import get_gpu_info

            gpu_info = get_gpu_info()
            model_size = model_size_from_name(cfg.base)
            batch_size = estimate_batch_size(
                model_params_b=model_size,
                seq_length=cfg.data.max_length,
                gpu_memory_bytes=gpu_info["memory_total_bytes"],
                quantization=tcfg.quantization,
                lora_r=tcfg.lora.r,
            )
            # GRPO generates N completions per prompt → more memory
            batch_size = max(1, batch_size // tcfg.num_generations)
            console.print(f"[green]Auto batch size (GRPO):[/] {batch_size}")

        # Ensure batch_size >= num_generations (trl requires
        # generation_batch_size to be divisible by num_generations)
        num_gen = tcfg.num_generations
        if batch_size < num_gen:
            batch_size = num_gen

        # --- Dataset ---
        # GRPO expects prompts — extract from messages or use prompt field
        train_data = _prepare_grpo_dataset(dataset["train"])
        train_ds = Dataset.from_list(train_data)
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_data = _prepare_grpo_dataset(dataset["val"])
            eval_ds = Dataset.from_list(eval_data)

        # --- Output dir ---
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Calculate warmup steps from ratio ---
        import math

        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps)
            * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        # --- Warn if running on CPU (trl GRPO has known CPU issues) ---
        if self.device == "cpu":
            console.print(
                "[yellow]Warning: GRPO on CPU is experimental. "
                "trl's GRPOTrainer may produce empty generations on CPU, "
                "causing tensor size errors. A CUDA GPU is recommended.[/]"
            )

        # --- GRPO config ---
        grpo_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": tcfg.epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
            "learning_rate": tcfg.lr,
            "warmup_steps": warmup_steps,
            "weight_decay": tcfg.weight_decay,
            "max_grad_norm": tcfg.max_grad_norm,
            "optim": tcfg.optimizer,
            "lr_scheduler_type": tcfg.scheduler,
            "logging_steps": tcfg.logging_steps,
            "save_steps": tcfg.save_steps,
            "save_total_limit": 3,
            **self._build_precision_kwargs(),
            "report_to": self.report_to,
            "remove_unused_columns": False,
            "deepspeed": self.deepspeed_config,
            **(self.fsdp_config or {}),
            "beta": tcfg.grpo_beta,
            "num_generations": tcfg.num_generations,
            "max_completion_length": cfg.data.max_length,
        }

        # CPU support: set use_cpu and prevent empty generations
        if self.device == "cpu":
            import inspect as _inspect

            grpo_params = _inspect.signature(GRPOConfig).parameters
            if "use_cpu" in grpo_params:
                grpo_kwargs["use_cpu"] = True
            # Workaround for trl GRPO CPU bug: model.generate() can produce
            # zero new tokens on CPU, causing tensor size mismatch errors.
            # Setting min_new_tokens=1 ensures at least one token is generated.
            if "generation_kwargs" in grpo_params:
                grpo_kwargs["generation_kwargs"] = {"min_new_tokens": 1}

        grpo_config = GRPOConfig(**grpo_kwargs)

        # Workaround: also set min_new_tokens on model's generation_config directly.
        # GRPOConfig may not forward generation_kwargs to model.generate() in all
        # trl versions, so this ensures the model always generates at least 1 token.
        if self.device == "cpu" and hasattr(self.model, "generation_config"):
            self.model.generation_config.min_new_tokens = 1

        # --- Trainer ---
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            reward_funcs=reward_fn,
            processing_class=self.tokenizer,
        )
        # v0.53.11 #123 — thread grpo_delta into the variant subclass.
        if (
            tcfg.grpo_variant is not None
            and tcfg.grpo_variant != "standard"
            and tcfg.grpo_delta is not None
        ):
            self.trainer._soup_grpo_delta = float(tcfg.grpo_delta)
        # v0.53.11 #127 — wire the live stability callback.
        from soup_cli.utils.peft_wiring import attach_grpo_stability_callback
        attach_grpo_stability_callback(self.trainer, tcfg)

        # v0.40.6 #67 — ReLoRA callback (magnitude-prune LoRA every N steps).
        from soup_cli.utils.peft_wiring import (
            attach_curriculum_callback,
            attach_plugin_callback,
            attach_relora_callback,
        )
        attach_relora_callback(self.trainer, tcfg)
        # v0.53.5 #114/#115 — dynamic curriculum live callback.
        attach_curriculum_callback(self.trainer, tcfg, str(output_dir), console)
        # v0.53.6 #101 — Soup plugin TrainerCallback.
        attach_plugin_callback(self.trainer, console)

        self._output_dir = str(output_dir)

    def _setup_transformers(self, cfg: SoupConfig, tcfg) -> None:
        """Load model via standard transformers + peft pipeline."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization (v0.38.0 Quant Menu — see soup_cli.utils.quant_menu)
        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        quant_config_obj = build_quantization_config_for_loader(
            tcfg=tcfg, base=cfg.base, console=console,
        )

        console.print(f"[dim]Loading model: {cfg.base}[/]")
        # On CPU, use device_map="cpu" to avoid meta tensors from "auto"
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {
            "trust_remote_code": self._trust_remote_code, "device_map": dev_map,
        }
        if quant_config_obj is not None:
            model_kwargs["quantization_config"] = quant_config_obj

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

        if tcfg.quantization in ("4bit", "8bit", "mxfp4"):
            self.model = prepare_model_for_kbit_training(self.model)

        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

        lora_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )
        # v0.40.6 #67 — surgical PEFT patches (Gemma4 ClippableLinear pre-LoRA;
        # 3-D fused-MoE expert dropout strip post-LoRA).
        from soup_cli.utils.peft_wiring import (
            apply_post_lora_patches,
            apply_pre_lora_patches,
        )
        apply_pre_lora_patches(self.model, cfg.base)
        self.model = get_peft_model(self.model, lora_config)
        apply_post_lora_patches(self.model)

        # QAT — insert fake quantization ops after LoRA. The "fp8" variant
        # is FP8 training (handled by apply_v028_speed_memory), not int8 QAT.
        if tcfg.quantization_aware and tcfg.quantization_aware != "fp8":
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

        # v0.35.0 #60 — multi-trainer wiring of v0.28.0 speed/memory features.
        from soup_cli.utils.v028_features import apply_v028_speed_memory
        apply_v028_speed_memory(
            model=self.model, tcfg=tcfg, base_model=cfg.base,
            console=console, device=self.device, backend=cfg.backend,
        )

    def _setup_unsloth(self, cfg, tcfg):
        """Load model via unsloth FastLanguageModel (2-5x faster)."""
        from soup_cli.utils.unsloth import load_model_and_tokenizer

        console.print(f"[dim]Loading model via [bold]unsloth[/]: {cfg.base}[/]")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=cfg.base,
            max_seq_length=cfg.data.max_length,
            quantization=tcfg.quantization,
            lora_r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=tcfg.lora.target_modules,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        """Run GRPO training and return results summary."""
        start = time.time()

        # Add callback for live display and experiment tracking
        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(
                SoupTrainerCallback(
                    display, tracker=tracker, run_id=run_id,
                    loss_watchdog=self.config.training.loss_watchdog,
                    loss_watchdog_threshold=self.config.training.loss_watchdog_threshold,
                    loss_watchdog_patience=self.config.training.loss_watchdog_patience,
                )
            )

        from soup_cli.utils.v028_features import activation_offloading_context

        with activation_offloading_context(
            self.config.training, self._output_dir,
        ):
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        duration = time.time() - start

        # Save final model (LoRA adapter)
        self.trainer.save_model(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

        # Extract metrics
        logs = self.trainer.state.log_history
        train_losses = [entry["loss"] for entry in logs if "loss" in entry]

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        return {
            "initial_loss": train_losses[0] if train_losses else 0,
            "final_loss": train_losses[-1] if train_losses else 0,
            "duration": duration_str,
            "duration_secs": duration,
            "output_dir": self._output_dir,
            "total_steps": self.trainer.state.global_step,
        }


def _prepare_grpo_dataset(data: list[dict]) -> list[dict]:
    """Convert dataset rows to GRPO format.

    GRPO expects each row to have a 'prompt' field (list of messages or string).
    Input can be:
      - messages format: [{"role": "user", "content": "..."}, ...]
      - DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
      - prompt field: {"prompt": "..."}

    Returns list of dicts with 'prompt' as a message list for chat models.
    """
    prepared = []
    for row in data:
        if "prompt" in row and isinstance(row["prompt"], str):
            # DPO or plain prompt format — convert to message list
            entry = {"prompt": [{"role": "user", "content": row["prompt"]}]}
            # Preserve 'answer' field if present (for accuracy reward)
            if "answer" in row:
                entry["answer"] = row["answer"]
            prepared.append(entry)
        elif "messages" in row:
            # Messages format — use the user message(s) as prompt
            messages = row["messages"]
            prompt_msgs = [msg for msg in messages if msg["role"] != "assistant"]
            entry = {"prompt": prompt_msgs}
            prepared.append(entry)
        elif "prompt" in row and isinstance(row["prompt"], list):
            # Already in message list format
            entry = {"prompt": row["prompt"]}
            if "answer" in row:
                entry["answer"] = row["answer"]
            prepared.append(entry)
        else:
            # Fallback: treat any 'instruction' field as prompt
            instruction = row.get("instruction", row.get("input", ""))
            entry = {"prompt": [{"role": "user", "content": str(instruction)}]}
            if "output" in row:
                entry["answer"] = row["output"]
            prepared.append(entry)
    return prepared
