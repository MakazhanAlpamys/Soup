"""Embedding model trainer — contrastive/triplet loss for sentence embeddings."""

import math
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.utils.gpu import estimate_batch_size, model_size_from_name

console = Console()


class EmbeddingTrainerWrapper:
    """High-level wrapper for embedding model fine-tuning from SoupConfig.

    Supports contrastive, triplet, and cosine loss for training sentence
    embedding models (BGE, E5, GTE, INSTRUCTOR, etc.).

    Data fields:
    - anchor: the query / anchor text
    - positive: semantically similar text
    - negative: semantically dissimilar text (optional for contrastive, required for triplet)
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
        self._output_dir = None

    def setup(self, dataset: dict) -> None:
        """Load model, tokenizer, apply LoRA, create embedding trainer."""
        from datasets import Dataset
        from transformers import TrainingArguments

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
            # Embedding processes pairs/triplets → roughly 2-3x memory per sample
            batch_size = max(1, batch_size // 3)
            console.print(f"[green]Auto batch size (embedding):[/] {batch_size}")

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

        # --- Determine loss function ---
        loss_type = tcfg.embedding_loss
        margin = tcfg.embedding_margin
        has_negatives = "negative" in train_ds.column_names

        if loss_type == "triplet" and not has_negatives:
            console.print(
                "[yellow]Warning: triplet loss requires 'negative' field. "
                "Falling back to contrastive loss.[/]"
            )
            loss_type = "contrastive"

        console.print(
            f"[green]Embedding config:[/] loss={loss_type}, margin={margin}, "
            f"pooling={tcfg.embedding_pooling}"
        )

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

        if self.fsdp_config:
            training_kwargs.update(self.fsdp_config)

        if tcfg.loraplus_lr_ratio is not None:
            training_kwargs["loraplus_lr_ratio"] = tcfg.loraplus_lr_ratio

        training_args = TrainingArguments(**training_kwargs)

        # --- Custom Trainer with embedding loss ---
        self.trainer = _EmbeddingTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            loss_type=loss_type,
            margin=margin,
            pooling=tcfg.embedding_pooling,
            temperature=tcfg.embedding_temperature,
            max_length=cfg.data.max_length,
        )

        self._output_dir = str(output_dir)

    def _setup_transformers(self, cfg: SoupConfig, tcfg: TrainingConfig) -> None:
        """Load model via standard transformers + peft pipeline."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

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
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Use AutoModel (not AutoModelForCausalLM) for embedding models
        self.model = AutoModel.from_pretrained(cfg.base, **model_kwargs)

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
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )
        self.model = get_peft_model(self.model, lora_config)

    def _setup_unsloth(self, cfg: SoupConfig, tcfg: TrainingConfig) -> None:
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
        """Run embedding training and return results summary."""
        if self.trainer is None or self._output_dir is None:
            raise RuntimeError(
                "EmbeddingTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
        start = time.time()

        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(
                SoupTrainerCallback(display, tracker=tracker, run_id=run_id)
            )

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


def _pool_embeddings(last_hidden_state, attention_mask, pooling: str):
    """Apply pooling strategy to hidden states."""
    import torch

    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    elif pooling == "last":
        # Get last non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_idx, seq_lengths, :]
    else:
        # Mean pooling (default)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class _EmbeddingTrainer:
    """Custom trainer for embedding models with contrastive/triplet loss.

    Wraps HuggingFace Trainer with custom compute_loss for embedding objectives.
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        loss_type: str,
        margin: float,
        pooling: str,
        temperature: float,
        max_length: int,
    ):
        from transformers import Trainer

        self._loss_type = loss_type
        self._margin = margin
        self._pooling = pooling
        self._temperature = temperature
        self._max_length = max_length
        self._tokenizer = processing_class

        # Create a custom Trainer subclass dynamically to inject compute_loss
        embedding_trainer = self

        class _CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return embedding_trainer._compute_embedding_loss(
                    model, inputs, return_outputs
                )

        self._trainer = _CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self._collate_fn,
            processing_class=processing_class,
        )

    def _collate_fn(self, features):
        """Collate embedding pairs/triplets into tokenized batches."""
        anchors = [feat["anchor"] for feat in features]
        positives = [feat["positive"] for feat in features]
        negatives = None
        if "negative" in features[0] and features[0]["negative"]:
            negatives = [feat["negative"] for feat in features]

        batch = {}
        anchor_enc = self._tokenizer(
            anchors, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
        )
        batch["anchor_input_ids"] = anchor_enc["input_ids"]
        batch["anchor_attention_mask"] = anchor_enc["attention_mask"]

        pos_enc = self._tokenizer(
            positives, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
        )
        batch["positive_input_ids"] = pos_enc["input_ids"]
        batch["positive_attention_mask"] = pos_enc["attention_mask"]

        if negatives:
            neg_enc = self._tokenizer(
                negatives, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            )
            batch["negative_input_ids"] = neg_enc["input_ids"]
            batch["negative_attention_mask"] = neg_enc["attention_mask"]

        return batch

    def _compute_embedding_loss(self, model, inputs, return_outputs=False):
        """Compute contrastive, triplet, or cosine loss on embeddings."""
        import torch
        from torch.nn import functional as nn_func

        # Encode anchor
        anchor_out = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
        )
        anchor_emb = _pool_embeddings(
            anchor_out.last_hidden_state,
            inputs["anchor_attention_mask"],
            self._pooling,
        )

        # Encode positive
        pos_out = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
        )
        pos_emb = _pool_embeddings(
            pos_out.last_hidden_state,
            inputs["positive_attention_mask"],
            self._pooling,
        )

        # Normalize embeddings
        anchor_emb = nn_func.normalize(anchor_emb, p=2, dim=-1)
        pos_emb = nn_func.normalize(pos_emb, p=2, dim=-1)

        if self._loss_type == "triplet" and "negative_input_ids" in inputs:
            neg_out = model(
                input_ids=inputs["negative_input_ids"],
                attention_mask=inputs["negative_attention_mask"],
            )
            neg_emb = _pool_embeddings(
                neg_out.last_hidden_state,
                inputs["negative_attention_mask"],
                self._pooling,
            )
            neg_emb = nn_func.normalize(neg_emb, p=2, dim=-1)

            # Triplet margin loss
            pos_dist = (anchor_emb - pos_emb).pow(2).sum(dim=-1)
            neg_dist = (anchor_emb - neg_emb).pow(2).sum(dim=-1)
            loss = nn_func.relu(pos_dist - neg_dist + self._margin).mean()

        elif self._loss_type == "cosine":
            # Cosine similarity loss — maximize similarity of anchor-positive
            cos_sim = (anchor_emb * pos_emb).sum(dim=-1)
            loss = (1.0 - cos_sim).mean()

        else:
            # Contrastive loss (InfoNCE / in-batch negatives)
            similarity = torch.matmul(anchor_emb, pos_emb.T) / self._temperature
            labels = torch.arange(similarity.size(0), device=similarity.device)
            loss = nn_func.cross_entropy(similarity, labels)

        if return_outputs:
            return loss, anchor_out
        return loss

    # Delegate Trainer interface methods
    def train(self, resume_from_checkpoint=None):
        return self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_model(self, output_dir):
        return self._trainer.save_model(output_dir)

    def add_callback(self, callback):
        return self._trainer.add_callback(callback)

    @property
    def state(self):
        return self._trainer.state
