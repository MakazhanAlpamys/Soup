"""Knowledge-distillation trainer (v0.53.2 #133).

``task='distill'`` — student model learns from a frozen teacher.

The training loss is a mix of:

* the standard SFT cross-entropy on the student logits, AND
* a token-level divergence loss between student and teacher logits, scaled
  by ``training.distill_temperature`` (T) following Hinton et al. 2015.

Four divergence options (mirroring axolotl's distill plugin):

* ``kl`` (alias of ``forward_kl``) — KL(teacher || student); standard
  distillation.
* ``reverse_kl`` — KL(student || teacher); mode-seeking.
* ``js`` — Jensen-Shannon (symmetric).

The teacher is loaded once, frozen (``requires_grad_(False)``), and
evaluated under ``torch.no_grad()`` to keep VRAM bounded. Student wears
LoRA per the standard PEFT pipeline.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from soup_cli.config.schema import SoupConfig

if TYPE_CHECKING:
    import torch as _torch_typ

console = Console()

# 50/50 CE / distillation blend — matches Hinton et al. 2015.
# Promote to a schema field (distill_ce_weight) in a follow-up patch.
_CE_WEIGHT: float = 0.5
_DISTILL_WEIGHT: float = 1.0 - _CE_WEIGHT


def _compute_distill_term(
    student_logits: "_torch_typ.Tensor",
    teacher_logits: "_torch_typ.Tensor",
    divergence: str,
    temperature: float,
) -> "_torch_typ.Tensor":
    """Pure tensor kernel: divergence between student and teacher logits.

    Both logits are ``(batch, seq, vocab)``. Temperature softens the
    distributions before the divergence is computed (Hinton). The result is
    a scalar mean over the token-level divergences.

    Raises:
        TypeError: ``temperature`` not numeric or is bool.
        ValueError: ``temperature`` non-finite or non-positive; ``divergence``
            outside the supported set.
    """
    import torch

    if isinstance(temperature, bool):
        raise TypeError(f"temperature must not be bool, got {temperature!r}")
    if not isinstance(temperature, (int, float)):
        raise TypeError(
            f"temperature must be float, got {type(temperature).__name__}"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(
            f"temperature must be finite and positive, got {temperature!r}"
        )
    temp = float(temperature)
    s = student_logits / temp
    t = teacher_logits / temp
    if divergence == "forward_kl":
        # KL(teacher || student). Use kl_div which expects log-probs of the
        # student and probs of the teacher.
        log_s = torch.log_softmax(s, dim=-1)
        p_t = torch.softmax(t, dim=-1)
        return torch.nn.functional.kl_div(
            log_s, p_t, reduction="batchmean"
        ) * (temp * temp)
    if divergence == "reverse_kl":
        log_t = torch.log_softmax(t, dim=-1)
        p_s = torch.softmax(s, dim=-1)
        return torch.nn.functional.kl_div(
            log_t, p_s, reduction="batchmean"
        ) * (temp * temp)
    if divergence == "js":
        # Jensen-Shannon: 0.5 (KL(p||m) + KL(q||m)), m = 0.5 (p + q).
        log_s = torch.log_softmax(s, dim=-1)
        log_t = torch.log_softmax(t, dim=-1)
        p_s = log_s.exp()
        p_t = log_t.exp()
        m = 0.5 * (p_s + p_t)
        log_m = m.clamp(min=1e-12).log()
        kl_pm = torch.nn.functional.kl_div(log_m, p_s, reduction="batchmean")
        kl_qm = torch.nn.functional.kl_div(log_m, p_t, reduction="batchmean")
        return 0.5 * (kl_pm + kl_qm) * (temp * temp)
    raise ValueError(f"Unknown divergence {divergence!r}")


class DistillTrainerWrapper:
    """High-level wrapper for student/teacher distillation.

    Mirrors :class:`BCOTrainerWrapper` lifecycle (``__init__`` → ``setup`` →
    ``train``). Teacher loads once in ``setup``; SFT-shaped dataset is
    formatted via the standard ``build_format_row`` factory so the student
    sees ``{input_ids, labels, attention_mask}`` rows. The custom
    ``compute_loss`` injects the distillation term.
    """

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: str | None = None,
        fsdp_config: dict | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        # Raw user-supplied flag — kept so teacher trust_remote_code can be
        # resolved separately against its own model id during setup().
        self._raw_trust_remote_code = trust_remote_code

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

        self.model: Any = None
        self.teacher: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._output_dir: str | None = None

    def setup(self, dataset: dict) -> None:
        """Load student + teacher, build distillation Trainer."""
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        from soup_cli.data.sft_format import build_format_row
        from soup_cli.utils.distill import validate_divergence

        cfg = self.config
        tcfg = cfg.training

        if tcfg.teacher_model is None:
            raise ValueError(
                "task='distill' requires training.teacher_model to be set"
            )
        divergence = validate_divergence(tcfg.distill_divergence or "forward_kl")
        temperature = float(tcfg.distill_temperature or 2.0)

        console.print(f"[dim]Loading tokenizer (student/teacher shared): {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        console.print(f"[dim]Loading student: {cfg.base}[/]")
        dev_map = "cpu" if self.device == "cpu" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base,
            trust_remote_code=self._trust_remote_code,
            device_map=dev_map,
        )

        # LoRA on the student — bracket with v0.40.6 #67 surgical PEFT patches.
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
        from soup_cli.utils.peft_wiring import (
            apply_post_lora_patches,
            apply_pre_lora_patches,
        )
        apply_pre_lora_patches(self.model, cfg.base)
        self.model = get_peft_model(self.model, lora_config)
        apply_post_lora_patches(self.model)

        # Teacher trust_remote_code resolved INDEPENDENTLY against the teacher
        # model id (code-review HIGH-1 fix — student's resolution must not
        # auto-trust the teacher).
        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code as _req,
        )
        from soup_cli.utils.trust_remote import (
            resolve_trust_remote_code as _resolve,
        )

        teacher_requires = _req(tcfg.teacher_model) or False
        teacher_trc = _resolve(
            tcfg.teacher_model,
            requested=self._raw_trust_remote_code,
            console=console,
            requires_remote_code=teacher_requires,
        )

        console.print(f"[dim]Loading teacher (frozen): {tcfg.teacher_model}[/]")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            tcfg.teacher_model,
            trust_remote_code=teacher_trc,
            device_map=dev_map,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Cross-tokenizer distillation is not supported — vocab sizes must
        # match so KL between logit distributions is well-defined.
        teacher_vocab = getattr(self.teacher.config, "vocab_size", None)
        student_vocab = getattr(self.model.config, "vocab_size", None)
        if (
            teacher_vocab is not None
            and student_vocab is not None
            and teacher_vocab != student_vocab
        ):
            raise ValueError(
                f"Teacher vocab size ({teacher_vocab}) != student vocab "
                f"size ({student_vocab}). Cross-tokenizer distillation is "
                "not supported in v0.53.2; use a teacher that shares the "
                "student tokenizer family."
            )

        # Dataset prep — reuse the SFT formatter so distill sees
        # {input_ids, labels, attention_mask}. v0.53.2 #137: pass training_cfg
        # so reasoning_effort + train_on_eot are honored on task='distill'.
        format_row = build_format_row(
            tokenizer=self.tokenizer,
            data_cfg=cfg.data,
            console=console,
            training_cfg=tcfg,
        )
        raw_train = Dataset.from_list(dataset["train"])
        train_ds = raw_train.map(
            format_row, remove_columns=raw_train.column_names
        )
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            raw_val = Dataset.from_list(dataset["val"])
            eval_ds = raw_val.map(
                format_row, remove_columns=raw_val.column_names
            )

        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = tcfg.batch_size if tcfg.batch_size != "auto" else 4
        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps)
            * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        args = TrainingArguments(
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
        )

        teacher_ref = self.teacher

        class _DistillTrainer(Trainer):
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs: bool = False,
                num_items_in_batch=None,
            ):
                import torch

                labels = inputs.get("labels")
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                student_logits = outputs.logits

                ce_loss = torch.tensor(0.0, device=student_logits.device)
                if labels is not None:
                    shift_logits = student_logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    ce_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                # Bridge devices: HF Trainer may auto-move the student to
                # CUDA while the frozen teacher stays on CPU (or vice versa).
                # Move teacher inputs onto the teacher's device, then move
                # teacher_logits back onto the student's device for the
                # KL kernel.
                try:
                    teacher_device = next(teacher_ref.parameters()).device
                except StopIteration:
                    teacher_device = student_logits.device
                teacher_inputs = {
                    k: (v.to(teacher_device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                    if k != "labels"
                }
                with torch.no_grad():
                    teacher_out = teacher_ref(**teacher_inputs)
                    teacher_logits = teacher_out.logits.to(student_logits.device)

                distill_loss = _compute_distill_term(
                    student_logits, teacher_logits, divergence, temperature
                )
                total = _CE_WEIGHT * ce_loss + _DISTILL_WEIGHT * distill_loss
                return (total, outputs) if return_outputs else total

        # ``DataCollatorForSeq2Seq`` pads ``input_ids`` and ``attention_mask``
        # via the tokenizer AND pads ``labels`` with ``label_pad_token_id``
        # (-100 = IGNORE_INDEX). ``DataCollatorForLanguageModeling`` does
        # NOT pad labels — incorrect for our pre-tokenised loss-masked rows.
        from transformers import DataCollatorForSeq2Seq

        self.trainer = _DistillTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                label_pad_token_id=-100,
                padding=True,
            ),
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

    def train(
        self,
        display: object | None = None,
        tracker: object | None = None,
        run_id: str = "",
        resume_from_checkpoint: str | None = None,
    ) -> dict:
        if self.trainer is None:
            raise RuntimeError(
                "DistillTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
        start = time.time()
        if display is not None:
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
