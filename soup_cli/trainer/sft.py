"""SFT (Supervised Fine-Tuning) trainer — wraps HuggingFace transformers + peft + trl."""

import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class SFTTrainerWrapper:
    """High-level wrapper that sets up model + tokenizer + trainer from SoupConfig."""

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
        """Load model, tokenizer, apply LoRA, create trainer."""
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        # Enable Rich progress bar for HuggingFace downloads
        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training
        use_unsloth = cfg.backend == "unsloth"
        use_vision = cfg.modality == "vision"

        if use_vision:
            self._setup_vision_transformers(cfg, tcfg)
        elif use_unsloth:
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
            console.print(f"[green]Auto batch size:[/] {batch_size}")

        # --- Dataset ---
        if use_vision:
            train_ds, eval_ds = self._prepare_vision_dataset(dataset)
        else:
            def format_row(example):
                if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                    text = self.tokenizer.apply_chat_template(
                        example["messages"], tokenize=False, add_generation_prompt=False
                    )
                else:
                    # Fallback for models without chat template
                    parts = []
                    for msg in example["messages"]:
                        role = msg["role"]
                        content = msg["content"]
                        parts.append(f"{role}: {content}")
                    text = "\n".join(parts)
                return {"text": text}

            train_ds = Dataset.from_list(dataset["train"]).map(
                format_row, remove_columns=["messages"]
            )
            eval_ds = None
            if "val" in dataset and dataset["val"]:
                eval_ds = Dataset.from_list(dataset["val"]).map(
                    format_row, remove_columns=["messages"]
                )

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

        # --- Training args ---
        training_args = TrainingArguments(
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
        )

        # --- Trainer ---
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
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

        # Quantization
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

        # LoRA
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

    def _setup_vision_transformers(self, cfg, tcfg):
        """Load vision-language model via transformers (LLaMA-Vision, Qwen2-VL, etc.)."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        console.print(f"[dim]Loading vision processor: {cfg.base}[/]")
        self.processor = AutoProcessor.from_pretrained(cfg.base, trust_remote_code=True)
        self.tokenizer = self.processor  # SFTTrainer uses processing_class

        # Quantization
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

        console.print(f"[dim]Loading vision model: {cfg.base}[/]")
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForVision2Seq.from_pretrained(cfg.base, **model_kwargs)

        if tcfg.quantization in ("4bit", "8bit"):
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA — target language model layers only
        target_modules = tcfg.lora.target_modules
        if target_modules == "auto":
            target_modules = None

        lora_config = LoraConfig(
            r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

        # QAT — insert fake quantization ops after LoRA
        if tcfg.quantization_aware:
            from soup_cli.utils.qat import prepare_model_for_qat

            self.model = prepare_model_for_qat(self.model)

    def _prepare_vision_dataset(self, dataset: dict):
        """Prepare dataset for vision fine-tuning with image loading."""
        from datasets import Dataset

        def load_and_format_vision(example):
            from PIL import Image as PILImage

            image_path = example.get("image", "")
            image = None
            if image_path:
                try:
                    image = PILImage.open(image_path).convert("RGB")
                except (FileNotFoundError, OSError):
                    console.print(f"[yellow]Warning: cannot open image: {image_path}[/]")

            messages = example["messages"]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            result = {"text": text}
            if image is not None:
                result["images"] = [image]
            return result

        remove_cols = ["messages", "image"]
        train_ds = Dataset.from_list(dataset["train"]).map(
            load_and_format_vision,
            remove_columns=[c for c in remove_cols if c in dataset["train"][0]],
        )
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_ds = Dataset.from_list(dataset["val"]).map(
                load_and_format_vision,
                remove_columns=[c for c in remove_cols if c in dataset["val"][0]],
            )
        return train_ds, eval_ds

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        """Run training and return results summary."""
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


def _enable_hf_transfer_progress():
    """Enable Rich progress bars for HuggingFace Hub file downloads."""
    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        class RichDownloadProgress:
            """Wraps tqdm calls with Rich progress bars for HF downloads."""

            def __init__(self, *args, **kwargs):
                desc = kwargs.get("desc", "") or (args[0] if args else "Downloading")
                total = kwargs.get("total", None)
                self._progress = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                )
                self._progress.start()
                self._task = self._progress.add_task(str(desc), total=total)
                self._n = 0

            def update(self, n=1):
                self._n += n
                self._progress.update(self._task, advance=n)

            def close(self):
                self._progress.stop()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

        # Patch huggingface_hub's tqdm usage
        import huggingface_hub.utils._http as hf_http

        if hasattr(hf_http, "tqdm"):
            hf_http.tqdm = RichDownloadProgress
    except (ImportError, AttributeError):
        pass  # Silently skip if huggingface_hub internals changed
