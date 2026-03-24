"""GRPO (Group Relative Policy Optimization) trainer — wraps trl.GRPOTrainer."""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


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
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, dataset: dict):
        """Load model, tokenizer, apply LoRA, create GRPO trainer."""
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

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
            "bf16": self.device == "cuda",
            "report_to": self.report_to,
            "remove_unused_columns": False,
            "deepspeed": self.deepspeed_config,
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

        # --- Trainer ---
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            reward_funcs=reward_fn,
            processing_class=self.tokenizer,
        )

        self._output_dir = str(output_dir)

    def _setup_transformers(self, cfg, tcfg):
        """Load model via standard transformers + peft pipeline."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        console.print(f"[dim]Loading model: {cfg.base}[/]")
        # On CPU, use device_map="cpu" to avoid meta tensors from "auto"
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

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
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

        # QAT — insert fake quantization ops after LoRA
        if tcfg.quantization_aware:
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

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
                SoupTrainerCallback(display, tracker=tracker, run_id=run_id)
            )

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
