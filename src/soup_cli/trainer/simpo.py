"""SimPO (Simple Preference Optimization) trainer — wraps trl.CPOTrainer."""

import math
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class SimPOTrainerWrapper:
    """High-level wrapper for SimPO training from SoupConfig.

    SimPO is a reference-free preference optimization method that uses
    length-normalized log probabilities as implicit rewards. Implemented
    via trl.CPOTrainer with loss_type='simpo'.

    Data fields (same as DPO):
    - prompt: the input prompt
    - chosen: the preferred response
    - rejected: the less preferred response
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
        """Load model, tokenizer, apply LoRA, create SimPO (CPO) trainer."""
        from datasets import Dataset
        from trl import CPOConfig, CPOTrainer

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
            # SimPO processes pairs → roughly 2x memory per sample
            batch_size = max(1, batch_size // 2)
            console.print(f"[green]Auto batch size (SimPO):[/] {batch_size}")

        # --- Dataset ---
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

        # --- SimPO config (via CPOTrainer with loss_type='simpo') ---
        cpo_config = CPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            warmup_steps=warmup_steps,
            weight_decay=tcfg.weight_decay,
            max_grad_norm=tcfg.max_grad_norm,
            optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            bf16=self.device == "cuda",
            report_to=self.report_to,
            remove_unused_columns=False,
            deepspeed=self.deepspeed_config,
            **(self.fsdp_config or {}),
            loss_type="simpo",
            cpo_alpha=tcfg.cpo_alpha,
            simpo_gamma=tcfg.simpo_gamma,
            max_length=cfg.data.max_length,
            max_prompt_length=cfg.data.max_length // 2,
            **({"neftune_noise_alpha": tcfg.neftune_alpha}
               if tcfg.neftune_alpha is not None else {}),
        )

        # --- Trainer ---
        self.trainer = CPOTrainer(
            model=self.model,
            args=cpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
        )

        # v0.40.6 #67 — ReLoRA callback.
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
        # v0.40.6 #67 — surgical PEFT patches.
        from soup_cli.utils.peft_wiring import (
            apply_post_lora_patches,
            apply_pre_lora_patches,
        )
        apply_pre_lora_patches(self.model, cfg.base)
        self.model = get_peft_model(self.model, lora_config)
        apply_post_lora_patches(self.model)

        # QAT — int8 only; "fp8" handled by apply_v028_speed_memory below.
        if tcfg.quantization_aware and tcfg.quantization_aware != "fp8":
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

        # v0.35.0 #60 — multi-trainer wiring of v0.28.0 speed/memory features.
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
        """Run SimPO training and return results summary."""
        if self.trainer is None:
            raise RuntimeError(
                "SimPOTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
        start = time.time()

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

        self.trainer.save_model(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

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
