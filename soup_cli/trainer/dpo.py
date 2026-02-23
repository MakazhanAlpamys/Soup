"""DPO (Direct Preference Optimization) trainer — wraps trl.DPOTrainer."""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class DPOTrainerWrapper:
    """High-level wrapper for DPO training from SoupConfig.

    DPO requires preference data with three fields:
    - prompt: the input prompt
    - chosen: the preferred response
    - rejected: the less preferred response
    """

    def __init__(self, config: SoupConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, dataset: dict):
        """Load model, tokenizer, apply LoRA, create DPO trainer."""
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import DPOConfig, DPOTrainer

        cfg = self.config
        tcfg = cfg.training

        # --- Tokenizer ---
        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Quantization ---
        bnb_config = None
        if tcfg.quantization == "4bit":
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif tcfg.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # --- Model ---
        console.print(f"[dim]Loading model: {cfg.base}[/]")
        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

        if tcfg.quantization in ("4bit", "8bit"):
            self.model = prepare_model_for_kbit_training(self.model)

        # --- LoRA ---
        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None  # peft will auto-detect

        lora_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
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
            # DPO processes pairs → roughly 2x memory per sample
            batch_size = max(1, batch_size // 2)
            console.print(f"[green]Auto batch size (DPO):[/] {batch_size}")

        # --- Dataset ---
        # DPO expects: prompt, chosen, rejected
        train_ds = Dataset.from_list(dataset["train"])
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_ds = Dataset.from_list(dataset["val"])

        # --- Output dir ---
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- DPO config ---
        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            warmup_ratio=tcfg.warmup_ratio,
            weight_decay=tcfg.weight_decay,
            max_grad_norm=tcfg.max_grad_norm,
            optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            bf16=self.device == "cuda",
            report_to="none",
            remove_unused_columns=False,
            beta=tcfg.dpo_beta,
            max_length=cfg.data.max_length,
            max_prompt_length=cfg.data.max_length // 2,
        )

        # --- Trainer ---
        self.trainer = DPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
        )

        self._output_dir = str(output_dir)

    def train(self, display: Optional[object] = None) -> dict:
        """Run DPO training and return results summary."""
        start = time.time()

        # Add callback for live display
        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(SoupTrainerCallback(display))

        self.trainer.train()
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
            "output_dir": self._output_dir,
            "total_steps": self.trainer.state.global_step,
        }
