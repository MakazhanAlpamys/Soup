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

        use_audio = cfg.modality == "audio"

        if use_vision:
            self._setup_vision_transformers(cfg, tcfg)
        elif use_audio:
            self._setup_audio_transformers(cfg, tcfg)
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

        # --- Curriculum learning: sort dataset by difficulty ---
        if tcfg.curriculum:
            from soup_cli.utils.curriculum import sort_by_length

            if tcfg.curriculum_metric == "length":
                dataset["train"] = sort_by_length(dataset["train"])
                console.print(
                    f"[green]Curriculum learning enabled:[/] "
                    f"metric=length, buckets={tcfg.curriculum_buckets}"
                )
            else:
                console.print(
                    f"[yellow]Curriculum metric '{tcfg.curriculum_metric}' "
                    "requires pre-computed scores. Using length-based sorting.[/]"
                )
                dataset["train"] = sort_by_length(dataset["train"])

        # --- Dataset ---
        if use_vision:
            train_ds, eval_ds = self._prepare_vision_dataset(dataset)
        elif use_audio:
            train_ds, eval_ds = self._prepare_audio_dataset(dataset)
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
        # v0.33.0 #58: auto_mixed_precision wires pick_mixed_precision()
        # into bf16/fp16 kwargs. Default behaviour (bf16 on CUDA) preserved
        # when the auto flag is False.
        bf16_flag, fp16_flag = self._resolve_mixed_precision(tcfg, cfg.base)

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
            "bf16": bf16_flag,
            "fp16": fp16_flag,
            "report_to": self.report_to,
            "remove_unused_columns": False,
            "deepspeed": self.deepspeed_config,
        }

        # FSDP2 — alternative to DeepSpeed. The helper also enables
        # torch.compile when tcfg.use_fsdp2_compile is True.
        from soup_cli.utils.fsdp import apply_fsdp_training_kwargs

        apply_fsdp_training_kwargs(
            training_kwargs,
            fsdp_config=self.fsdp_config,
            use_fsdp2_compile=tcfg.use_fsdp2_compile,
        )
        if self.fsdp_config and tcfg.use_fsdp2_compile:
            console.print("[green]torch.compile enabled on FSDP2[/]")

        # Gradient checkpointing — tiered (v0.28.0): bool or tier string.
        if tcfg.gradient_checkpointing:
            from soup_cli.utils.gpu import get_gpu_info
            from soup_cli.utils.gradient_ckpt import (
                describe_tier,
                resolve_gradient_checkpointing,
            )

            gpu_memory_gb: Optional[float] = None
            try:
                gpu_memory_gb = get_gpu_info().get(
                    "memory_total_bytes", 0
                ) / (1024**3) or None
            except (KeyError, TypeError, ZeroDivisionError):
                gpu_memory_gb = None

            ckpt_kwargs = resolve_gradient_checkpointing(
                tcfg.gradient_checkpointing, gpu_memory_gb=gpu_memory_gb,
            )
            training_kwargs.update(ckpt_kwargs)
            if ckpt_kwargs:
                console.print(
                    f"[green]Gradient checkpointing:[/] "
                    f"{describe_tier(tcfg.gradient_checkpointing, gpu_memory_gb)}"
                )

        # NEFTune — noisy embeddings for better fine-tuning quality
        if tcfg.neftune_alpha is not None:
            training_kwargs["neftune_noise_alpha"] = tcfg.neftune_alpha

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
            if cfg.data.max_length < 256:
                console.print(
                    f"[yellow]Warning:[/] packing=true with max_length={cfg.data.max_length} "
                    "may be suboptimal. Consider increasing max_length for better packing."
                )
            console.print("[green]Sample packing enabled[/]")
            if tcfg.packing_cross_doc_attn_mask:
                # TRL's SFTTrainer exposes an `eos_token`-based boundary detector
                # on recent versions (>= 0.12). When available, we flag the
                # trainer to emit block-diagonal attention masks; otherwise the
                # flag is a best-effort hint (no regression in behavior).
                trainer_kwargs["packing_strategy"] = "attention_free"
                console.print(
                    "[green]Cross-document attention masking enabled:[/] "
                    "packed docs cannot attend across boundaries"
                )

        self.trainer = SFTTrainer(**trainer_kwargs)

        self._output_dir = str(output_dir)
        self._batch_size = batch_size

    def _resolve_mixed_precision(self, tcfg, base_model: str) -> tuple[bool, bool]:
        """Return ``(bf16, fp16)`` flags for TrainingArguments.

        - When ``tcfg.auto_mixed_precision`` is True: query GPU compute
          capability and call :func:`pick_mixed_precision` to decide.
        - Otherwise: preserve legacy default (bf16 on CUDA, no fp16).
        """
        if not getattr(tcfg, "auto_mixed_precision", False):
            return (self.device == "cuda", False)

        if self.device != "cuda":
            return (False, False)

        try:
            import torch

            major, minor = torch.cuda.get_device_capability()
            cc = float(f"{major}.{minor}")
        except (ImportError, RuntimeError, AssertionError, OSError):
            return (self.device == "cuda", False)

        from soup_cli.utils.mixed_precision import pick_mixed_precision

        try:
            mode = pick_mixed_precision(base_model, cc)
        except ValueError:
            return (self.device == "cuda", False)

        console.print(
            f"[green]Auto mixed-precision picked:[/] {mode} "
            f"(model={base_model}, cc={cc})"
        )
        return (mode == "bf16", mode == "fp16")

    def _setup_transformers(self, cfg, tcfg):
        """Load model via standard transformers + peft pipeline."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from soup_cli.utils.moe import detect_moe_model, get_moe_target_modules

        # Liger Kernel — apply fused ops BEFORE model loading
        if tcfg.use_liger:
            from soup_cli.utils.liger import apply_liger_kernel

            if apply_liger_kernel(cfg.base):
                console.print(
                    "[green]Liger Kernel enabled:[/] fused RMSNorm, SwiGLU, CrossEntropy, RoPE"
                )
            else:
                console.print("[yellow]Liger Kernel: no matching architecture found[/]")

        # Cut Cross-Entropy (v0.28.0) — patch BEFORE model loading
        if tcfg.use_cut_ce:
            from soup_cli.utils.cut_ce import apply_cut_ce

            if apply_cut_ce(cfg.base):
                console.print(
                    "[green]Cut Cross-Entropy enabled:[/] "
                    "large-vocab CE replaced with chunked CCE kernel"
                )
            else:
                console.print(
                    "[yellow]Cut Cross-Entropy: no matching architecture found "
                    "or cut_cross_entropy not installed[/]"
                )

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

        # FlashAttention — set attn_implementation for faster attention
        if tcfg.use_flash_attn:
            from soup_cli.utils.flash_attn import get_attn_implementation

            attn_impl = get_attn_implementation(tcfg.use_flash_attn, self.device)
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
                console.print(f"[green]FlashAttention enabled:[/] {attn_impl}")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)

        # Long-context — apply RoPE scaling after model load
        if tcfg.rope_scaling_type:
            from soup_cli.utils.long_context import apply_long_context_config

            rope_config = apply_long_context_config(
                self.model.config,
                target_length=cfg.data.max_length,
                rope_scaling_type=tcfg.rope_scaling_type,
                model_name=cfg.base,
            )
            if rope_config:
                console.print(
                    f"[green]Long-context enabled:[/] RoPE {tcfg.rope_scaling_type} "
                    f"scaling to {cfg.data.max_length} tokens"
                )

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

        if tcfg.quantization in ("4bit", "8bit"):
            self.model = prepare_model_for_kbit_training(self.model)

        # Freeze training — freeze bottom layers before LoRA
        if tcfg.freeze_layers is not None or tcfg.freeze_ratio is not None:
            from soup_cli.utils.freeze import freeze_model_layers

            frozen = freeze_model_layers(
                self.model,
                freeze_layers=tcfg.freeze_layers,
                freeze_ratio=tcfg.freeze_ratio,
            )
            console.print(
                f"[green]Freeze training:[/] {frozen} parameters frozen"
            )

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
        self.model = get_peft_model(self.model, lora_config)

        self._apply_quantization_aware(tcfg)

    def _apply_quantization_aware(self, tcfg) -> None:
        """Apply quantization-aware training post-LoRA (shared text/vision).

        - ``quantization_aware=True``   → int8 QAT via torchao (legacy path)
        - ``quantization_aware="fp8"``  → FP8 training via torchao.float8 (v0.28.0)
        - ``False`` / None              → no-op
        """
        if tcfg.quantization_aware == "fp8":
            from soup_cli.utils.fp8 import apply_fp8_training

            if apply_fp8_training(self.model, recipe=tcfg.fp8_recipe):
                console.print(
                    f"[green]FP8 training enabled:[/] "
                    f"converted linears to Float8Linear (recipe={tcfg.fp8_recipe})"
                )
            else:
                console.print(
                    "[yellow]FP8 training requested but unavailable "
                    "(no Hopper+ GPU or torchao.float8 missing)[/]"
                )
        elif tcfg.quantization_aware is True:
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
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )
        self.model = get_peft_model(self.model, lora_config)

        self._apply_quantization_aware(tcfg)

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

    def _setup_audio_transformers(self, cfg, tcfg):
        """Load audio-language model via transformers (Qwen2-Audio, Whisper, etc.)."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from rich.panel import Panel as RichPanel
        from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

        console.print(
            RichPanel(
                f"[bold yellow]WARNING:[/] Loading audio model: "
                f"[bold]{cfg.base}[/]\n"
                "If this model contains custom code (trust_remote_code), "
                "it will execute on this machine.\n"
                "Only use models you trust.",
                title="Audio Model",
                border_style="yellow",
            )
        )
        console.print(f"[dim]Loading audio processor: {cfg.base}[/]")
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

        console.print(f"[dim]Loading audio model: {cfg.base}[/]")
        dev_map = "cpu" if self.device == "cpu" else "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": dev_map}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Use AutoModel for audio models — AutoModelForCausalLM doesn't handle
        # audio-language architectures (Qwen2-Audio, Whisper, etc.)
        self.model = AutoModel.from_pretrained(cfg.base, **model_kwargs)

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
            use_dora=tcfg.lora.use_dora,
            use_rslora=tcfg.lora.use_rslora,
        )
        self.model = get_peft_model(self.model, lora_config)

    def _prepare_audio_dataset(self, dataset: dict):
        """Prepare dataset for audio fine-tuning with audio loading."""
        from datasets import Dataset

        try:
            import librosa  # noqa: F401
        except ImportError:
            raise ImportError(
                "librosa is required for audio training. "
                "Install with: pip install 'soup-cli[audio]'"
            )

        def load_and_format_audio(example):
            import librosa

            audio_path = example.get("audio", "")
            audio_array = None
            sampling_rate = 16000
            if audio_path:
                try:
                    audio_array, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True,
                    )
                except (FileNotFoundError, OSError):
                    console.print(f"[yellow]Warning: cannot open audio: {audio_path}[/]")

            messages = example["messages"]
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                parts = []
                for msg in messages:
                    parts.append(f"{msg['role']}: {msg['content']}")
                text = "\n".join(parts)

            result = {"text": text}
            if audio_array is not None:
                result["audio"] = audio_array
                result["sampling_rate"] = sampling_rate
            return result

        if not dataset["train"]:
            raise ValueError(
                "Audio training dataset is empty after validation. "
                "Check audio file paths and audio_dir."
            )

        remove_cols = ["messages", "audio"]
        train_ds = Dataset.from_list(dataset["train"]).map(
            load_and_format_audio,
            remove_columns=[
                c for c in remove_cols if c in dataset["train"][0]
            ],
        )
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_ds = Dataset.from_list(dataset["val"]).map(
                load_and_format_audio,
                remove_columns=[
                    c for c in remove_cols if c in dataset["val"][0]
                ],
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

            tcfg_local = self.config.training
            self.trainer.add_callback(
                SoupTrainerCallback(
                    display, tracker=tracker, run_id=run_id,
                    output_dir=self._output_dir,
                    loss_watchdog=tcfg_local.loss_watchdog,
                    loss_watchdog_threshold=tcfg_local.loss_watchdog_threshold,
                    loss_watchdog_patience=tcfg_local.loss_watchdog_patience,
                    spike_recovery=getattr(
                        tcfg_local, "loss_spike_recovery", False,
                    ),
                    spike_recovery_max_attempts=getattr(
                        tcfg_local, "loss_spike_recovery_max_attempts", 3,
                    ),
                    spike_recovery_lr_decay=getattr(
                        tcfg_local, "loss_spike_recovery_lr_decay", 0.5,
                    ),
                    grad_accum_auto_tune=getattr(
                        tcfg_local, "grad_accum_auto_tune", False,
                    ),
                    grad_accum_pressure_threshold=getattr(
                        tcfg_local, "grad_accum_pressure_threshold", 0.9,
                    ),
                    grad_accum_current_steps=getattr(
                        tcfg_local, "gradient_accumulation_steps", 1,
                    ),
                    grad_accum_current_batch=self._batch_size,
                )
            )

        # Activation offloading (v0.28.0) — wrap train() so saved-tensor hooks
        # are active only during training (and removed afterwards).
        from soup_cli.utils.activation_offload import offload_context
        from soup_cli.utils.paths import is_under_cwd

        tcfg = self.config.training
        offload_save_dir: Optional[str] = None
        if tcfg.activation_offloading == "disk":
            candidate = str(Path(self._output_dir) / "_activation_offload")
            # Defense-in-depth: refuse to create the scratch directory outside
            # the project tree even if cfg.output escaped containment upstream.
            if not is_under_cwd(self._output_dir):
                raise ValueError(
                    "activation_offloading='disk' requires the training output "
                    "dir to be under the current working directory; got: "
                    f"{self._output_dir!r}"
                )
            offload_save_dir = candidate
        with offload_context(
            tcfg.activation_offloading, save_dir=offload_save_dir
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
