"""ORPO (Odds Ratio Preference Optimization) trainer — wraps trl.ORPOTrainer."""

import math
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.stream_setup import StreamingSetupMixin
from soup_cli.utils.gpu import (
    bf16_fp16_flags,
    estimate_batch_size,
    model_size_from_name,
    resolve_device_map,
)
from soup_cli.utils.seeding import apply_training_seed, training_seed_kwargs

console = Console()


class ORPOTrainerWrapper(StreamingSetupMixin):
    """High-level wrapper for ORPO training from SoupConfig.

    ORPO uses preference data (same format as DPO) but does not
    require a reference model — it combines SFT and alignment in one step.

    Data fields:
    - prompt: the input prompt
    - chosen: the preferred response
    - rejected: the less preferred response
    """

    #: TRL builds this loss's forward through ``concatenated_inputs`` +
    #: ``torch.cat``, so chosen and rejected arrive as ONE tensor of twice
    #: the configured batch. The VRAM pre-flight must budget for that.
    _STREAM_ROWS_PER_EXAMPLE = 2

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
        """Load model, tokenizer, apply LoRA, create ORPO trainer."""
        from datasets import Dataset

        from soup_cli.trainer._trl_compat import (
            prompt_length_kwargs,
            resolve_trl_symbol,
        )
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        # #326 — trl 0.29.0 dropped ORPOConfig / ORPOTrainer from the public
        # `trl` namespace. They were not deleted: they live on under
        # `trl.experimental.orpo`. A plain `from trl import ORPOConfig` is an
        # ImportError there, so the task dies before it can say anything useful.
        orpo_config_cls = resolve_trl_symbol("ORPOConfig", "trl.experimental.orpo")
        orpo_trainer_cls = resolve_trl_symbol("ORPOTrainer", "trl.experimental.orpo")

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training

        # #353: seed before the model and any adapter are built.
        apply_training_seed(tcfg)

        use_unsloth = cfg.backend == "unsloth"
        # v0.72.4 — layer streaming replaces the model-load path entirely (meta
        # skeleton, never a resident load), so it dispatches ahead of the backend
        # branches. The schema already rejects streaming + unsloth/mlx.
        use_streaming = bool(getattr(tcfg, "stream_layers", False))

        if use_streaming:
            self._setup_streaming_transformers(cfg, tcfg)
        elif use_unsloth:
            self._setup_unsloth(cfg, tcfg)
        else:
            self._setup_transformers(cfg, tcfg)

        trainable, total = self.model.get_nb_trainable_parameters()
        # v0.72.4 (mirrors sft.py) — under NF4 streaming PEFT's total is wrong
        # by ~6.5x: it sizes Params4bit as `numel * 2 * quant_storage.itemsize`,
        # right for a RESIDENT one but not for our `meta` placeholder, which
        # still carries the LOGICAL shape. The sharder counted the real source
        # elements, so prefer that.
        stream_total = getattr(self._stream_runtime, "total_params", 0)
        if stream_total:
            total = stream_total
        pct = 100 * trainable / total if total else 0.0
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
            # ORPO processes pairs → roughly 2x memory per sample
            batch_size = max(1, batch_size // 2)
            console.print(f"[green]Auto batch size (ORPO):[/] {batch_size}")

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

        # --- ORPO config ---
        from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

        _bf16, _fp16 = bf16_fp16_flags(self.device)
        orpo_config = orpo_config_cls(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            warmup_steps=warmup_steps,
            # #328 — StreamedDecoderLayer already wraps every layer in
            # checkpoint(use_reentrant=False). Letting HF check-point the INNER
            # decoder layer as well recomputes it after functional_call's
            # reparametrisation context has exited and restored the `meta`
            # placeholders, which on torch 2.13 + CUDA dies with "Tensor on device
            # cuda:0 is not on the expected device meta!". Passing this explicitly
            # also stops the value being decided by TRL's default, which is not
            # stable across versions (False on trl 0.19.1, True on 0.26.2) and was
            # silently dropping the user's own setting on the older one.
            gradient_checkpointing=should_enable_hf_gradient_checkpointing(
                tcfg.gradient_checkpointing, stream_layers=tcfg.stream_layers
            ),
            weight_decay=tcfg.weight_decay,
            max_grad_norm=tcfg.max_grad_norm,
            optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            bf16=_bf16,
            fp16=_fp16,
            report_to=self.report_to,
            remove_unused_columns=False,
            deepspeed=self.deepspeed_config,
            **training_seed_kwargs(tcfg),
            **(self.fsdp_config or {}),
            beta=tcfg.orpo_beta,
            max_length=cfg.data.max_length,
            # #326 — ORPOConfig lost `max_prompt_length` at 0.28.0. See dpo.py.
            **prompt_length_kwargs(orpo_config_cls, cfg.data.max_length // 2),
            **({"neftune_noise_alpha": tcfg.neftune_alpha}
               if tcfg.neftune_alpha is not None else {}),
        )

        # --- Trainer ---
        self.trainer = orpo_trainer_cls(
            model=self.model,
            args=orpo_config,
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
        dev_map = resolve_device_map(self.device)
        model_kwargs = {
            "trust_remote_code": self._trust_remote_code, "device_map": dev_map,
        }
        if quant_config_obj is not None:
            model_kwargs["quantization_config"] = quant_config_obj

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)
        from soup_cli.utils.data_pipeline import apply_vocab_expansion

        apply_vocab_expansion(
            self.tokenizer,
            self.model,
            cfg.data,
        )
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
        """Run ORPO training and return results summary."""
        if self.trainer is None:
            raise RuntimeError(
                "ORPOTrainerWrapper.train() called before setup(). "
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

        # v0.72.4 — the shared context releases the streaming weight source even
        # if training raises (see StreamingSetupMixin._training_context).
        with self._training_context(
            activation_offloading_context(self.config.training, self._output_dir)
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
