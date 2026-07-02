"""Pretrain (Continued Pre-training) trainer — wraps HuggingFace SFTTrainer for CLM."""

import math
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class PretrainTrainerWrapper:
    """High-level wrapper for continued pre-training from SoupConfig.

    Pre-training uses raw text data (no instruction/response structure).
    Each sample is a plain text document that the model learns via
    causal language modelling (next-token prediction).
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
        self._output_dir = None

    def setup(self, dataset: dict) -> None:
        """Load model, tokenizer, apply LoRA, create trainer for CLM."""
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        # Enable Rich progress bar for HuggingFace downloads
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training
        use_unsloth = cfg.backend == "unsloth"

        if use_unsloth:
            self._setup_unsloth(cfg, tcfg)
        else:
            self._setup_transformers(cfg, tcfg)

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
            if batch_size <= 0:
                batch_size = 1
            console.print(f"[green]Auto batch size:[/] {batch_size}")

        # --- Dataset ---
        # v0.53.7 #86 — short-circuit tokenization when caller pre-tokenized
        # via `soup data preprocess`. Skips the raw-text load entirely.
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        pretok = _maybe_load_pretokenized(cfg.data, cfg.base, console)
        if pretok is not None:
            train_ds, eval_ds = pretok
        else:
            # Plaintext: data already has {"text": "..."} from format conversion
            train_ds = Dataset.from_list(dataset["train"])
            eval_ds = None
            if "val" in dataset and dataset["val"]:
                eval_ds = Dataset.from_list(dataset["val"])

        # --- Output dir ---
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Calculate warmup steps from ratio ---
        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps)
            * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        # --- Training args ---
        training_kwargs = {
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
            "bf16": self.device == "cuda",
            "report_to": self.report_to,
            "remove_unused_columns": False,
            "deepspeed": self.deepspeed_config,
        }

        # FSDP2 — alternative to DeepSpeed
        if self.fsdp_config:
            training_kwargs.update(self.fsdp_config)

        # LoRA+ — different learning rates for A and B matrices
        if tcfg.loraplus_lr_ratio is not None:
            training_kwargs["loraplus_lr_ratio"] = tcfg.loraplus_lr_ratio

        # GaLore — memory-efficient full-parameter training
        if tcfg.use_galore:
            from soup_cli.utils.galore import get_galore_optimizer_and_params

            if tcfg.optimizer != "adamw_torch":
                console.print(
                    f"[yellow]GaLore overrides optimizer '{tcfg.optimizer}' "
                    f"with 'galore_adamw'.[/]"
                )
            galore_kwargs = get_galore_optimizer_and_params(
                galore_rank=tcfg.galore_rank,
                galore_update_proj_gap=tcfg.galore_update_proj_gap,
                galore_scale=tcfg.galore_scale,
            )
            training_kwargs.update(galore_kwargs)
            console.print(
                f"[green]GaLore enabled:[/] rank={tcfg.galore_rank}, "
                f"update_gap={tcfg.galore_update_proj_gap}, scale={tcfg.galore_scale}"
            )

        training_args = TrainingArguments(**training_kwargs)

        # --- Trainer ---
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": self.tokenizer,
        }

        # Sample packing — pack multiple short samples into one sequence
        if tcfg.packing:
            trainer_kwargs["packing"] = True
            console.print("[green]Sample packing enabled[/]")

        # v0.40.4 #65 — multipack live wiring (mirrors sft.py).
        use_multipack = bool(getattr(tcfg, "multipack", False))
        if use_multipack:
            from soup_cli.utils.multipack_sampler import (
                validate_multipack_architecture,
            )
            from soup_cli.utils.multipack_trainer import (
                attach_multipack_state,
                detect_arch_name,
                lengths_from_dataset,
                make_multipack_trainer_class,
            )

            arch = detect_arch_name(self.model)
            if arch:
                validate_multipack_architecture(arch)
            trainer_cls = make_multipack_trainer_class(SFTTrainer)
            self.trainer = trainer_cls(**trainer_kwargs)
            attach_multipack_state(
                self.trainer,
                lengths=lengths_from_dataset(train_ds),
                max_seq_len=cfg.data.max_length,
                batch_size=batch_size,
                seed=getattr(tcfg, "seed", 0) or 0,
            )
            console.print("[green]Multipack FFD bin-packing sampler enabled[/]")
        else:
            self.trainer = SFTTrainer(**trainer_kwargs)

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

        from soup_cli.utils.moe import detect_moe_model, get_moe_target_modules

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

        # MoE aux loss for load balancing
        is_moe = detect_moe_model(self.model)
        if is_moe and tcfg.moe_aux_loss_coeff > 0:
            if hasattr(self.model.config, "router_aux_loss_coef"):
                self.model.config.router_aux_loss_coef = tcfg.moe_aux_loss_coeff
            if hasattr(self.model.config, "output_router_logits"):
                self.model.config.output_router_logits = True
            console.print(
                f"[green]MoE detected:[/] aux_loss_coeff={tcfg.moe_aux_loss_coeff}"
            )

        if tcfg.quantization in ("4bit", "8bit", "mxfp4"):
            self.model = prepare_model_for_kbit_training(self.model)

        # v0.53.4 #83 — LLaMA Pro block expansion (centralised — see SFT).
        from soup_cli.utils.block_expansion import (
            apply_block_expansion_if_configured,
        )

        apply_block_expansion_if_configured(self.model, tcfg, console)

        # LoRA — with MoE-aware target modules if moe_lora is enabled
        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

        if tcfg.moe_lora and is_moe:
            moe_targets = get_moe_target_modules(self.model)
            if moe_targets:
                target_modules = moe_targets
                console.print(
                    f"[green]ScatterMoE LoRA:[/] targeting {len(moe_targets)} module patterns"
                )

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
        # v0.40.6 #67 — surgical PEFT patches.
        from soup_cli.utils.peft_wiring import (
            apply_post_lora_patches,
            apply_pre_lora_patches,
        )
        apply_pre_lora_patches(self.model, cfg.base)
        self.model = get_peft_model(self.model, lora_config)
        apply_post_lora_patches(self.model)

        # v0.71.12 #84 — Mixture-of-Depths selective-token routing (applied
        # after get_peft_model so the routers are trainable).
        from soup_cli.utils.mod import apply_mod_if_configured

        apply_mod_if_configured(self.model, tcfg, cfg.base, console)

        # QAT — insert fake quantization ops after LoRA
        if tcfg.quantization_aware and tcfg.quantization_aware != "fp8":
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

        # v0.33.0 #43 — multi-trainer wiring of v0.28.0 speed/memory features.
        from soup_cli.utils.v028_features import apply_v028_speed_memory
        apply_v028_speed_memory(
            model=self.model, tcfg=tcfg, base_model=cfg.base,
            console=console, device=self.device, backend=cfg.backend,
        )

    def _setup_unsloth(self, cfg: SoupConfig, tcfg) -> None:
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
        """Run continued pre-training and return results summary."""
        if self.trainer is None or self._output_dir is None:
            raise RuntimeError(
                "PretrainTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
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
                    eval_gate_config=self.config.training.eval_gate,
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
