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

        # v0.71.12 #145 — sequence-level KD vs token-level logit KD.
        sequence_mode = getattr(tcfg, "distill_mode", "token") == "sequence"
        if sequence_mode and (
            tcfg.uld_strategy is not None or tcfg.minillm_enabled
        ):
            raise ValueError(
                "distill_mode='sequence' is incompatible with uld_strategy / "
                "minillm_enabled (those are token/logit-level distillation). "
                "Sequence-level KD trains the student with plain CE on "
                "teacher-generated text — drop uld_strategy / minillm_enabled."
            )

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

        teacher_vocab = getattr(self.teacher.config, "vocab_size", None)
        student_vocab = getattr(self.model.config, "vocab_size", None)
        uld_projection = None
        uld_teacher_tokenizer = None
        minillm_cb = None

        if sequence_mode:
            # v0.71.12 #145 — sequence-level KD. The teacher generates a
            # completion per prompt using ITS OWN tokenizer; the student then
            # trains with plain CE on the re-tokenised teacher output. This
            # works across ANY tokenizer pair, so the vocab-mismatch gate and
            # the token-level ULD / MiniLLM paths are skipped entirely.
            from soup_cli.utils.distill import build_sequence_distill_rows

            teacher_tokenizer = AutoTokenizer.from_pretrained(
                tcfg.teacher_model, trust_remote_code=teacher_trc
            )
            if (
                teacher_tokenizer.pad_token is None
                and teacher_tokenizer.eos_token is not None
            ):
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            seq_budget = min(int(cfg.data.max_length), 256)
            console.print(
                "[green]Sequence-level KD[/] — generating teacher "
                f"completions (max_new_tokens={seq_budget})"
            )
            dataset = dict(dataset)
            dataset["train"] = build_sequence_distill_rows(
                dataset["train"],
                self.teacher,
                teacher_tokenizer,
                max_new_tokens=seq_budget,
            )
            if dataset.get("val"):
                dataset["val"] = build_sequence_distill_rows(
                    dataset["val"],
                    self.teacher,
                    teacher_tokenizer,
                    max_new_tokens=seq_budget,
                )
            # Free the teacher — sequence-level KD does NOT need it in the
            # student loss loop (keeps the 4 GB VRAM budget honest).
            self.teacher = None
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        elif tcfg.uld_strategy is not None:
            # v0.71.11 #236 — cross-tokenizer ULD projection. When uld_strategy
            # is set, a vocab-size mismatch is EXPECTED (that's the whole point)
            # and the ULD loss handles it; otherwise vocab sizes must match so
            # the column-wise KL is well-defined.
            from soup_cli.utils.uld import ULDConfig, build_uld_projection

            uld_projection = build_uld_projection(
                ULDConfig(
                    strategy=tcfg.uld_strategy,
                    student_vocab_size=int(student_vocab or 32000),
                    teacher_vocab_size=int(teacher_vocab or 32000),
                    top_k=tcfg.uld_top_k,
                )
            )
            # v0.71.18 #258 — the aligned strategy forwards the teacher on ITS
            # OWN tokenization of the same text, so it needs the teacher's
            # tokenizer to re-encode + decode per-token strings for alignment.
            if tcfg.uld_strategy == "wasserstein_aligned":
                uld_teacher_tokenizer = AutoTokenizer.from_pretrained(
                    tcfg.teacher_model, trust_remote_code=teacher_trc
                )
                if (
                    uld_teacher_tokenizer.pad_token is None
                    and uld_teacher_tokenizer.eos_token is not None
                ):
                    uld_teacher_tokenizer.pad_token = uld_teacher_tokenizer.eos_token
            else:
                uld_teacher_tokenizer = None
            console.print(
                f"[green]Cross-tokenizer ULD enabled[/] "
                f"(strategy={tcfg.uld_strategy})"
            )
        elif (
            teacher_vocab is not None
            and student_vocab is not None
            and teacher_vocab != student_vocab
        ):
            raise ValueError(
                f"Teacher vocab size ({teacher_vocab}) != student vocab "
                f"size ({student_vocab}). Cross-tokenizer distillation needs "
                "training.uld_strategy (wasserstein / topk_align); use a "
                "teacher that shares the student tokenizer family, or set "
                "uld_strategy."
            )

        # v0.71.11 #237 — MiniLLM on-policy distillation modifier. Skipped in
        # sequence mode (mutually exclusive — rejected earlier in setup).
        if not sequence_mode and tcfg.minillm_enabled:
            from soup_cli.utils.minillm import MiniLLMConfig, build_minillm_callback

            # v0.71.18 #257 — honour an explicit training.minillm_rollout_length
            # when set; otherwise derive a tractable length from max_length,
            # capped at 32 so the growing-sequence autoregressive loop (re-
            # forwards the full prefix each step, ~O(L^2) graph) stays within
            # the consumer-GPU budget.
            if tcfg.minillm_rollout_length is not None:
                rollout_len = int(tcfg.minillm_rollout_length)
            else:
                rollout_len = max(1, min(int(cfg.data.max_length), 32))
            minillm_cb = build_minillm_callback(
                MiniLLMConfig(
                    teacher_mix_ratio=float(tcfg.minillm_teacher_mix_ratio),
                    length_normalize=bool(tcfg.minillm_length_normalize),
                    pretrain_anchor_weight=float(
                        tcfg.minillm_pretrain_anchor_weight
                    ),
                    pretrain_anchor_path=tcfg.minillm_pretrain_anchor_path,
                    on_policy=bool(tcfg.minillm_on_policy),
                    rollout_length=rollout_len,
                ),
                tokenizer=self.tokenizer,
                temperature=temperature,
            )
            mode = "on-policy rollout" if tcfg.minillm_on_policy else "offline blend"
            console.print(
                f"[green]MiniLLM distillation enabled[/] ({mode}, "
                f"teacher_mix={tcfg.minillm_teacher_mix_ratio}, "
                f"anchor={tcfg.minillm_pretrain_anchor_weight})"
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
        _teacher_vocab = teacher_vocab
        _uld_projection = uld_projection
        _minillm_cb = minillm_cb
        _sequence_mode = sequence_mode
        # v0.71.18 #257 — when on-policy, the rollout does its own teacher
        # forwards, so the batch-level teacher forward below is skipped.
        _minillm_on_policy = minillm_cb is not None and minillm_cb.config.on_policy
        # v0.71.18 #258 — aligned ULD re-encodes the text with the teacher
        # tokenizer (different vocab + boundaries) and aligns the sequences.
        _uld_aligned = tcfg.uld_strategy == "wasserstein_aligned"
        _uld_teacher_tokenizer = uld_teacher_tokenizer
        _student_tokenizer = self.tokenizer

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

                # v0.71.12 #145 — sequence-level KD trains the student with
                # plain CE on teacher-generated text. The teacher has already
                # been used (and freed) during dataset construction, so there
                # is no teacher forward / logit term here.
                if _sequence_mode:
                    return (ce_loss, outputs) if return_outputs else ce_loss

                # v0.71.18 #257 — true on-policy MiniLLM rollout. Samples a
                # fresh teacher-mixed rollout (doing its own teacher forwards)
                # so the batch-level teacher forward below is skipped entirely.
                if _minillm_on_policy:
                    rollout_loss = _minillm_cb.on_policy_term(
                        model,
                        teacher_ref,
                        inputs["input_ids"],
                        inputs.get("attention_mask"),
                    )
                    anchor = _minillm_cb.anchor_term(model)
                    total = _CE_WEIGHT * ce_loss + _DISTILL_WEIGHT * rollout_loss
                    if anchor is not None:
                        total = total + anchor
                    return (total, outputs) if return_outputs else total

                # v0.71.18 #258 — aligned ULD. Decode the student ids to text,
                # re-encode with the teacher tokenizer (different vocab AND
                # boundaries), forward the teacher on its own ids, then align
                # the two token sequences over their decoded character spans.
                if _uld_aligned and _uld_teacher_tokenizer is not None:
                    from soup_cli.utils.uld import uld_aligned_loss

                    student_ids = inputs["input_ids"]
                    s_mask = inputs.get("attention_mask")
                    texts = _student_tokenizer.batch_decode(
                        student_ids, skip_special_tokens=True
                    )
                    t_enc = _uld_teacher_tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=int(student_ids.shape[1]),
                    )
                    try:
                        t_dev = next(teacher_ref.parameters()).device
                    except StopIteration:
                        t_dev = student_logits.device
                    t_ids = t_enc["input_ids"].to(t_dev)
                    t_mask = t_enc["attention_mask"]
                    with torch.no_grad():
                        t_out = teacher_ref(
                            input_ids=t_ids,
                            attention_mask=t_mask.to(t_dev),
                        )
                        aligned_teacher_logits = t_out.logits.to(
                            student_logits.device
                        )
                    # Per-token decoded strings, trimmed to the real (non-pad)
                    # length so pad tokens don't pollute the alignment.
                    s_strings = []
                    s_ids_list = student_ids.tolist()
                    for bi, row in enumerate(s_ids_list):
                        s_len = (
                            int(s_mask[bi].sum()) if s_mask is not None else len(row)
                        )
                        s_strings.append(
                            [_student_tokenizer.decode([int(i)]) for i in row[:s_len]]
                        )
                    t_strings = []
                    for bi, row in enumerate(t_ids.tolist()):
                        t_len = int(t_mask[bi].sum())
                        t_strings.append(
                            [
                                _uld_teacher_tokenizer.decode([int(i)])
                                for i in row[:t_len]
                            ]
                        )
                    distill_loss = uld_aligned_loss(
                        student_logits,
                        aligned_teacher_logits,
                        s_strings,
                        t_strings,
                        config=_uld_projection.config,
                        attention_mask=s_mask,
                    )
                    total = _CE_WEIGHT * ce_loss + _DISTILL_WEIGHT * distill_loss
                    return (total, outputs) if return_outputs else total

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
                # v0.71.11 #236 — when ULD bridges different vocabs, clamp the
                # student token ids to the teacher's range so the (possibly
                # smaller) teacher embedding never index-errors.
                if (
                    _uld_projection is not None
                    and _teacher_vocab is not None
                    and "input_ids" in teacher_inputs
                ):
                    teacher_inputs["input_ids"] = teacher_inputs[
                        "input_ids"
                    ].clamp(max=int(_teacher_vocab) - 1)
                with torch.no_grad():
                    teacher_out = teacher_ref(**teacher_inputs)
                    teacher_logits = teacher_out.logits.to(student_logits.device)

                anchor = None
                if _uld_projection is not None:
                    # v0.71.11 #236 — cross-tokenizer ULD distillation loss.
                    distill_loss = _uld_projection(
                        student_logits,
                        teacher_logits,
                        attention_mask=inputs.get("attention_mask"),
                    )
                elif _minillm_cb is not None:
                    # v0.71.11 #237 — MiniLLM teacher-mixed reverse-KL +
                    # pretrain anchor.
                    distill_loss = _minillm_cb.distill_term(
                        student_logits, teacher_logits, labels
                    )
                    anchor = _minillm_cb.anchor_term(model)
                else:
                    distill_loss = _compute_distill_term(
                        student_logits, teacher_logits, divergence, temperature
                    )
                total = _CE_WEIGHT * ce_loss + _DISTILL_WEIGHT * distill_loss
                if anchor is not None:
                    total = total + anchor
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

        # v0.71.11 #237 — attach the MiniLLM callback for lifecycle (the
        # loss terms are applied directly in compute_loss above).
        if minillm_cb is not None:
            self.trainer.add_callback(minillm_cb)

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
                    eval_gate_config=self.config.training.eval_gate,
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
