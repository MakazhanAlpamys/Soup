"""Native Sesame CSM training: text conditioning and 32 parallel Mimi codebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soup_cli.trainer.sft import SFTTrainerWrapper, console
from soup_cli.utils.csm import validate_csm_config, validate_csm_dataset, validate_csm_row
from soup_cli.utils.eval_schedule import training_eval_kwargs
from soup_cli.utils.seeding import apply_training_seed, training_seed_kwargs


class CSMDataCollator:
    """Let CsmProcessor build native targets; never flatten codebooks into text."""

    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict:
        import numpy as np

        from soup_cli.utils.tts_codec import load_audio_mono

        texts, audios = [], []
        for row in examples:
            path, speaker, transcript = validate_csm_row(row)
            if any(token in transcript for token in (
                self.processor.audio_token, self.processor.audio_eos_token,
            )):
                raise ValueError("CSM transcript must not contain an audio control token")
            audio = load_audio_mono(path, target_sr=24000)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError("CSM audio must be non-empty and finite")
            # Render without tokenize=True: only Soup's bounded local reader
            # opens the file. The processor receives arrays, never paths/URLs.
            texts.append(self.processor.apply_chat_template(
                [{"role": speaker, "content": [
                    {"type": "text", "text": transcript}, {"type": "audio"},
                ]}], tokenize=False, add_generation_prompt=False,
            ))
            audios.append(audio)
        batch = self.processor(
            text=texts, audio=audios, sampling_rate=24000, output_labels=True,
            depth_decoder_labels_ratio=1.0, padding=True, truncation=False,
            return_tensors="pt", add_special_tokens=False,
        )
        if batch["input_ids"].shape[-1] > self.max_length:
            raise ValueError(
                f"CSM sequence exceeds data.max_length={self.max_length}; split the clip "
                "or increase max_length. Truncating audio markers would break Mimi alignment."
            )
        padding = batch["attention_mask"] == 0
        batch["labels"][padding] = -100
        # Some CSM tokenizers pad with the audio marker. The model merges audio
        # based on input_ids rather than attention_mask; use a neutral text id
        # on padding to avoid inventing frames or EOS targets in the model.
        neutral_id = self.processor.tokenizer.bos_token_id
        if neutral_id is None or neutral_id in {
            self.processor.audio_token_id, self.processor.audio_eos_token_id,
        }:
            raise ValueError("CSM tokenizer needs a BOS text token distinct from audio markers")
        batch["input_ids"][padding] = neutral_id
        if not (batch["labels"] == self.processor.audio_token_id).any(dim=-1).all():
            raise ValueError("CSM batch has an example without an audio-frame target")
        return batch


def freeze_csm_codec(model: Any) -> None:
    """Mimi supplies fixed targets, and must never receive optimizer updates."""
    model.codec_model.requires_grad_(False)
    model.codec_model.eval()


def _non_leaf_depth_embeddings(module: Any, inputs: Any, output: Any) -> Any:
    """Allow CSM's in-place backbone-frame insertion with frozen LoRA embeddings.

    Transformers' checkpointing input-grad hook makes frozen embeddings a leaf
    requiring grad. CSM replaces their first frame in place, which autograd
    refuses on that leaf. A differentiable clone preserves gradient flow and
    the upstream forward, without changing any target or trainable parameter.
    """
    if output.requires_grad and output.is_leaf:
        return output.clone()
    return output


def make_csm_trainer(**kwargs: Any) -> Any:
    """Keep Mimi in eval even when Trainer toggles the whole model to train."""
    from transformers import Trainer

    class NativeCSMTrainer(Trainer):
        def __init__(self, **trainer_kwargs: Any) -> None:
            super().__init__(**trainer_kwargs)
            # The model sums two losses with different target cardinalities.
            # HF's count of 2-D processor labels is not their normalization.
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self, model: Any, inputs: dict, return_outputs: bool = False, **loss_kwargs: Any,
        ) -> Any:
            model.codec_model.eval()
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **loss_kwargs,
            )

    return NativeCSMTrainer(**kwargs)


class CSMTrainerWrapper(SFTTrainerWrapper):
    """Dedicated setup with Soup's common logging, resume and final-save lifecycle."""

    def setup(self, dataset: dict) -> None:
        validate_csm_config(self.config)
        from soup_cli.trainer.stream_setup import _distributed_launch

        if self.deepspeed_config or self.fsdp_config or _distributed_launch():
            raise ValueError("Native Sesame CSM currently supports single-device training only")
        if str(self.device).startswith("cuda"):
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                raise ValueError(
                    "Native Sesame CSM supports single-device training only; "
                    "set CUDA_VISIBLE_DEVICES to one GPU to avoid Trainer DataParallel"
                )
        validate_csm_dataset(dataset)

        try:
            import soundfile  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Native Sesame CSM needs soundfile to read audio. "
                "Install pip install 'soup-cli[train,audio]'."
            ) from exc

        try:
            from transformers import CsmForConditionalGeneration, CsmProcessor, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Native Sesame CSM needs Transformers CsmForConditionalGeneration and "
                "CsmProcessor. Install pip install 'soup-cli[train,audio]'."
            ) from exc
        from soup_cli.utils.gpu import bf16_fp16_flags, resolve_base_load_dtype
        from soup_cli.utils.warmup import resolve_trainer_warmup_steps

        cfg, tcfg = self.config, self.config.training
        apply_training_seed(tcfg)
        self.processor = CsmProcessor.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code,
        )
        # Saving the processor, rather than just its tokenizer, preserves the
        # feature extractor + native chat template for reload/inference.
        self.tokenizer = self.processor
        self.model = CsmForConditionalGeneration.from_pretrained(
            cfg.base,
            torch_dtype=resolve_base_load_dtype(self.device, full_finetune=tcfg.lora.r == 0),
            trust_remote_code=self._trust_remote_code,
        )
        if (self.model.config.num_codebooks != 32
                or self.model.codec_model.config.num_quantizers != 32
                or self.model.depth_decoder.config.num_codebooks != 32):
            raise ValueError("Native Sesame CSM requires all 32 Mimi codebooks")
        if self.model.codec_model.config.sampling_rate != 24000:
            raise ValueError("Native Sesame CSM requires a 24 kHz Mimi codec")
        if (self.model.config.audio_token_id != self.processor.audio_token_id
                or self.model.config.audio_eos_token_id != self.processor.audio_eos_token_id):
            raise ValueError("CSM model and processor audio token ids do not match")
        freeze_csm_codec(self.model)
        self.model.config.use_cache = False
        self.model.depth_decoder.config.use_cache = False
        self.model.to(self.device)
        if tcfg.lora.r > 0:
            from peft import get_peft_model

            from soup_cli.utils.peft_wiring import build_lora_config

            # Generic PEFT forwarding preserves audio kwargs. CAUSAL_LM's
            # text-specific wrapper is not the contract of this multimodal LM.
            self.model = get_peft_model(self.model, build_lora_config(
                tcfg.lora, task_type=None,
                target_modules=r"(?:backbone_model|depth_decoder\.model)\.layers\.\d+"
                               r"\.self_attn\.(?:q_proj|v_proj)",
            ))
        if tcfg.gradient_checkpointing:
            # Install HF's input-grad hooks before our clone hook. Trainer can
            # enable checkpointing again: later input-grad hooks then see the
            # already differentiable non-leaf and leave it intact.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            embeddings = self.model.depth_decoder.get_input_embeddings()
            self._depth_embedding_hook = embeddings.register_forward_hook(
                _non_leaf_depth_embeddings,
            )
        self._batch_size = 1 if tcfg.batch_size == "auto" else tcfg.batch_size
        if tcfg.batch_size == "auto":
            console.print("[dim]Native CSM: batch_size=auto uses a conservative batch of 1.[/]")
        output = Path(cfg.output)
        if cfg.experiment_name:
            output /= cfg.experiment_name
        self._output_dir = str(output)
        bf16, fp16 = bf16_fp16_flags(self.device, allow_mps_bf16=True)
        args = TrainingArguments(
            output_dir=self._output_dir, num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr, weight_decay=tcfg.weight_decay, max_grad_norm=tcfg.max_grad_norm,
            warmup_steps=resolve_trainer_warmup_steps(tcfg.warmup_ratio),
            optim=tcfg.optimizer, lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps, save_steps=tcfg.save_steps, save_total_limit=3,
            bf16=bf16, fp16=fp16, use_cpu=self.device == "cpu", report_to=self.report_to,
            gradient_checkpointing=tcfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False, label_names=["labels"], prediction_loss_only=True,
            **training_seed_kwargs(tcfg),
            **training_eval_kwargs(cfg, dataset.get("val"), batch_size=self._batch_size),
        )
        self.trainer = make_csm_trainer(
            model=self.model, args=args, processing_class=self.processor,
            train_dataset=dataset["train"], eval_dataset=dataset.get("val") or None,
            data_collator=CSMDataCollator(self.processor, cfg.data.max_length),
        )
        console.print("[green]Native Sesame CSM:[/] backbone + 31 depth codebooks; Mimi frozen")
