"""Classifier / reranker / cross_encoder trainer (v0.53.2 #132).

Wraps :class:`transformers.AutoModelForSequenceClassification` for the three
classifier-family tasks declared in v0.52.0 Part B:

* ``classifier`` — single-input sequence classification.
* ``reranker`` — single-input reranker (typically score head); shares the
  classifier head topology.
* ``cross_encoder`` — paired-input scoring (passage / query, NLI, etc.).

Single-label uses ``CrossEntropyLoss``; multi-label uses ``BCEWithLogitsLoss``
(automatically when ``training.classifier_kind='multi_label'``).

Data format expected per row:

* ``text`` (or ``messages`` joined into a string) — single-input tasks.
* ``text_a`` + ``text_b`` — paired-input ``cross_encoder``.
* ``label`` — int (single-label), list[int] (multi-label), or string in
  ``training.label_names``.

Mirrors the BCO / pretrain wrapper pattern. Lazy imports for heavy deps;
trust_remote_code threaded through the v0.36.0 resolver.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, List, Union

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()

# Cap on multi-label list entries — defense against malformed dataset rows
# (security review v0.53.2 H2). Matches v0.52.0 ``_MAX_LABELS=1024``.
_MAX_MULTI_LABEL_ENTRIES: int = 1024


def _row_to_text(row: dict) -> str:
    """Extract the single-input text from a row (``text`` or joined messages).

    Raises:
        TypeError: non-string ``content`` inside a messages list — silent skip
            could poison training data with empty strings (security review
            v0.53.2 M3).
        ValueError: neither ``text`` nor ``messages`` present.
    """
    if "text" in row and isinstance(row["text"], str):
        return row["text"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        parts: list[str] = []
        for msg in msgs:
            if not isinstance(msg, dict):
                # Non-dict entries are silently skipped — caller's loader
                # would normally have produced these; keep loud-fail at the
                # row-level (missing text) rather than per-message.
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                raise TypeError(
                    "Classifier row messages[i]['content'] must be str, got "
                    f"{type(content).__name__!r}"
                )
            parts.append(content)
        return "\n".join(parts)
    raise ValueError(
        "Classifier row missing 'text' field and no joinable 'messages' list. "
        f"Row keys: {sorted(row)!r}"
    )


def _row_to_pair(row: dict) -> tuple[str, str]:
    """Extract (text_a, text_b) for paired ``cross_encoder`` rows.

    Raises:
        TypeError: either field present but not a string (security review M4 —
            silent ``str()`` coercion of dicts/lists produced garbage training
            text).
        ValueError: neither pair of fields present.
    """
    if "text_a" in row and "text_b" in row:
        a, b = row["text_a"], row["text_b"]
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError(
                "cross_encoder rows require 'text_a' and 'text_b' to be str; "
                f"got text_a={type(a).__name__}, text_b={type(b).__name__}"
            )
        return a, b
    if "question" in row and "answer" in row:
        q, ans = row["question"], row["answer"]
        if not isinstance(q, str) or not isinstance(ans, str):
            raise TypeError(
                "cross_encoder rows require 'question' and 'answer' to be "
                f"str; got question={type(q).__name__}, answer={type(ans).__name__}"
            )
        return q, ans
    raise ValueError(
        "cross_encoder row requires 'text_a' + 'text_b' (or 'question' + "
        f"'answer'). Row keys: {sorted(row)!r}"
    )


def _normalise_label(
    raw: object,
    label_names: List[str] | None,
    num_labels: int,
    multi_label: bool,
) -> Union[int, list[float]]:
    """Convert a raw label (int / str / list) to the trainer-expected form.

    Single-label → int in [0, num_labels). Multi-label → list[float] of
    length ``num_labels``.

    Raises:
        ValueError: invalid index, oversize multi-label list (defense-in-depth
            against malformed datasets — security review H2).
        TypeError: bool / unsupported scalar type.
    """
    if multi_label:
        if isinstance(raw, list):
            if len(raw) > _MAX_MULTI_LABEL_ENTRIES:
                raise ValueError(
                    f"multi-label list too int: {len(raw)} entries "
                    f"(max {_MAX_MULTI_LABEL_ENTRIES})"
                )
            vec = [0.0] * num_labels
            for entry in raw:
                idx = _label_index(entry, label_names, num_labels)
                vec[idx] = 1.0
            return vec
        # Single label silently broadcast to one-hot multi-label.
        idx = _label_index(raw, label_names, num_labels)
        vec = [0.0] * num_labels
        vec[idx] = 1.0
        return vec
    return _label_index(raw, label_names, num_labels)


def _label_index(
    raw: object, label_names: List[str] | None, num_labels: int
) -> int:
    if isinstance(raw, bool):
        # Project policy (v0.30.0 Candidate / v0.39.0 ReLoRAPolicy / v0.41.0
        # Part B): bool-as-int violations raise TypeError, not ValueError.
        raise TypeError(f"label must not be bool, got {raw!r}")
    if isinstance(raw, int):
        if raw < 0 or raw >= num_labels:
            raise ValueError(
                f"label index {raw} out of range [0, {num_labels})"
            )
        return raw
    if isinstance(raw, str):
        if label_names is None:
            raise ValueError(
                f"label is str {raw!r} but training.label_names is unset"
            )
        try:
            return label_names.index(raw)
        except ValueError as exc:
            raise ValueError(
                f"label {raw!r} not in training.label_names={label_names!r}"
            ) from exc
    raise TypeError(
        f"label must be int / str / list, got {type(raw).__name__}"
    )


class ClassifierTrainerWrapper:
    """High-level wrapper for classifier / reranker / cross_encoder training."""

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

        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._output_dir: str | None = None

    def setup(self, dataset: dict) -> None:
        """Load model + tokenizer, tokenise dataset, build HF Trainer."""
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        cfg = self.config
        tcfg = cfg.training

        if tcfg.num_labels is None:
            raise ValueError(
                f"task={cfg.task!r} requires training.num_labels to be set"
            )
        num_labels = int(tcfg.num_labels)
        multi_label = (tcfg.classifier_kind == "multi_label")
        problem_type = (
            "multi_label_classification" if multi_label else "single_label_classification"
        )

        console.print(f"[dim]Loading tokenizer: {cfg.base}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        console.print(f"[dim]Loading classifier model: {cfg.base}[/]")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base,
            num_labels=num_labels,
            problem_type=problem_type,
            trust_remote_code=self._trust_remote_code,
        )

        is_paired = (cfg.task == "cross_encoder")
        label_names = (
            list(tcfg.label_names) if tcfg.label_names is not None else None
        )

        def encode(row: dict) -> dict:
            if is_paired:
                a, b = _row_to_pair(row)
                enc = self.tokenizer(
                    a, b,
                    truncation=True,
                    max_length=cfg.data.max_length,
                )
            else:
                text = _row_to_text(row)
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=cfg.data.max_length,
                )
            label = _normalise_label(
                row.get("label"), label_names, num_labels, multi_label
            )
            enc["labels"] = label
            return enc

        raw_train = Dataset.from_list(dataset["train"])
        train_ds = raw_train.map(encode, remove_columns=raw_train.column_names)
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            raw_val = Dataset.from_list(dataset["val"])
            eval_ds = raw_val.map(encode, remove_columns=raw_val.column_names)

        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = tcfg.batch_size if tcfg.batch_size != "auto" else 8
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
            deepspeed=self.deepspeed_config,
            **(self.fsdp_config or {}),
        )

        from transformers import DataCollatorWithPadding

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
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
                "ClassifierTrainerWrapper.train() called before setup(). "
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
