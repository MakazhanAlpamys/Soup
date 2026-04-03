"""Reward Model trainer — wraps trl.RewardTrainer.

Trains a reward model from preference data (prompt + chosen + rejected).
The resulting model scores text sequences with a scalar reward, used by
PPO training to align a policy model.

Full RLHF pipeline:  SFT → Reward Model → PPO
"""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class RewardModelTrainerWrapper:
    """High-level wrapper for reward model training from SoupConfig.

    Trains an AutoModelForSequenceClassification on preference data
    (prompt/chosen/rejected) using TRL's RewardTrainer. The trained model
    can then be used as the reward signal for PPO training.

    Data format: same as DPO — requires 'prompt', 'chosen', 'rejected' fields.
    """

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, dataset: dict):
        """Load model, tokenizer, create RewardTrainer."""
        from datasets import Dataset
        from trl import RewardConfig, RewardTrainer

        # Enable Rich progress bar for HuggingFace downloads
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training

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
            # Reward model processes pairs → 2x memory per sample
            batch_size = max(1, batch_size // 2)
            console.print(f"[green]Auto batch size (Reward Model):[/] {batch_size}")

        # --- Dataset ---
        # RewardTrainer expects: chosen, rejected (text columns)
        train_data = _prepare_reward_dataset(dataset["train"])
        train_ds = Dataset.from_list(train_data)
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_data = _prepare_reward_dataset(dataset["val"])
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

        # --- Reward config ---
        reward_config = RewardConfig(
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
            max_length=cfg.data.max_length,
        )

        # --- Trainer ---
        self.trainer = RewardTrainer(
            model=self.model,
            args=reward_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
        )

        self._output_dir = str(output_dir)

    def _setup_transformers(self, cfg, tcfg):
        """Load model as AutoModelForSequenceClassification + LoRA."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if tcfg.quantization == "4bit":
            from soup_cli.utils.gpu import get_compute_dtype

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=get_compute_dtype(),
                bnb_4bit_use_double_quant=True,
            )
        elif tcfg.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        console.print(f"[dim]Loading reward model: {cfg.base}[/]")
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map, "num_labels": 1}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base, **model_kwargs,
        )

        if tcfg.quantization in ("4bit", "8bit"):
            self.model = prepare_model_for_kbit_training(self.model)

        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

        lora_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_CLS,
            bias="none",
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )
        self.model = get_peft_model(self.model, lora_config)

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        """Run reward model training and return results summary."""
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

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        duration = time.time() - start

        # Save final model
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


def _prepare_reward_dataset(data: list[dict]) -> list[dict]:
    """Convert dataset rows to reward model format.

    RewardTrainer expects each row to have 'chosen' and 'rejected' text fields.

    Input can be:
      - DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
      - Messages format with preference: {"chosen": [...messages], "rejected": [...messages]}

    Returns list of dicts with 'chosen' and 'rejected' text strings.
    """
    prepared = []
    for row in data:
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        prompt = row.get("prompt", "")

        # If chosen/rejected are message lists, convert to text
        if isinstance(chosen, list):
            chosen = " ".join(msg.get("content", "") for msg in chosen)
        if isinstance(rejected, list):
            rejected = " ".join(msg.get("content", "") for msg in rejected)

        # Prepend prompt if present
        if prompt:
            if isinstance(prompt, list):
                prompt = " ".join(msg.get("content", "") for msg in prompt)
            chosen = f"{prompt} {chosen}"
            rejected = f"{prompt} {rejected}"

        prepared.append({"chosen": chosen, "rejected": rejected})
    return prepared
