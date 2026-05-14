"""PRM (Process Reward Model) trainer wrapper — v0.53.11 #126.

Per-step reward prediction over stepwise-supervised data. Consumes the
v0.42.0 Part A ``data.format='prm'`` rows (segments + per-segment labels)
and trains a scalar reward head on top of a causal LM via LoRA.

Loss: MSE between predicted scalar rewards and the supervised labels at
each step boundary. The math kernel is
:func:`soup_cli.utils.prm.compute_prm_loss` so it can be unit-tested
without instantiating TRL.

Heavy deps (torch / transformers / peft) are lazy-imported inside methods
per project policy — ``python -m soup_cli.cli --help`` must not pull torch.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig

logger = logging.getLogger(__name__)
console = Console()


@lru_cache(maxsize=4)
def make_prm_trainer_class(base_cls: type) -> type:
    """Factory: build a ``_PRMTrainer`` subclass of HF ``Trainer``.

    The subclass overrides ``compute_loss`` to:
      1. Forward the input_ids through the causal LM and grab the last
         hidden state for each step-boundary token.
      2. Project to a scalar via ``self.model.reward_head`` (added by
         the wrapper before training starts).
      3. Compute MSE against the per-step labels via
         :func:`soup_cli.utils.prm.compute_prm_loss`.

    The factory pattern (with ``lru_cache``) lets us subclass whichever
    HF Trainer the caller supplies without an import-time dependency.
    """
    from soup_cli.utils.prm import compute_prm_loss

    class _PRMTrainer(base_cls):  # type: ignore[misc, valid-type]
        """HF Trainer subclass for PRM stepwise reward training."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # ``inputs`` carries input_ids, attention_mask, step_positions
            # (the indices of the per-step boundary tokens), and labels (the
            # per-step scalar rewards).
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            step_positions = inputs["step_positions"]
            labels = inputs["labels"].float()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, T, H]
            # Gather hidden states at step_positions.
            batch_size = last_hidden.size(0)
            # Step positions shape: [B, S] — pad index for missing steps is -1.
            valid_mask = (step_positions >= 0).float()
            # Replace -1 with 0 for gather safety (masked out below).
            safe_positions = step_positions.clamp(min=0)
            # Gather: [B, S, H]
            idx = safe_positions.unsqueeze(-1).expand(
                batch_size, safe_positions.size(1), last_hidden.size(-1)
            )
            step_hidden = last_hidden.gather(1, idx)
            # Project to scalar: [B, S, 1] -> [B, S]
            predictions = self.model.reward_head(step_hidden).squeeze(-1)
            loss = compute_prm_loss(predictions, labels, mask=valid_mask)
            if return_outputs:
                return loss, {"predictions": predictions}
            return loss

    _PRMTrainer.__name__ = f"_PRMTrainer_{base_cls.__name__}"
    return _PRMTrainer


def _prepare_prm_dataset(raw_rows: list[dict], tokenizer: Any, max_length: int) -> list[dict]:
    """Tokenise PRM rows into (input_ids, attention_mask, step_positions, labels).

    Each input row has shape ``{prompt, completions: [step1, step2, ...],
    labels: [r1, r2, ...]}``. We concatenate prompt + completions and
    record the index of the last token of each completion as the step
    boundary.
    """
    prepared: list[dict] = []
    for row in raw_rows:
        prompt = row.get("prompt", "")
        completions = row.get("completions", [])
        raw_labels = row.get("labels", [])
        if not completions or len(completions) != len(raw_labels):
            continue
        # Tokenise prompt — truncate to leave room for at least one step.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Reserve at least 1 token for completion tokens.
        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[: max_length - 1]
        input_ids = list(prompt_ids)
        step_positions: list[int] = []
        for step_text in completions:
            step_ids = tokenizer(step_text, add_special_tokens=False)["input_ids"]
            # Truncate this step to fit if needed.
            remaining = max_length - len(input_ids)
            if remaining <= 0:
                break
            step_ids = step_ids[:remaining]
            input_ids.extend(step_ids)
            step_positions.append(len(input_ids) - 1)
            if len(input_ids) >= max_length:
                break
        if not step_positions:
            continue
        prepared.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "step_positions": step_positions,
            "labels": list(raw_labels[: len(step_positions)]),
        })
    return prepared


def _build_collator(tokenizer: Any):
    """Build a PRM collator that pads input_ids + step_positions + labels."""
    import torch

    pad_id = tokenizer.pad_token_id or 0

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        max_steps = max(len(b["step_positions"]) for b in batch)
        input_ids = []
        attention_mask = []
        step_positions = []
        labels = []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attention_mask.append(b["attention_mask"] + [0] * pad)
            sp_pad = max_steps - len(b["step_positions"])
            step_positions.append(b["step_positions"] + [-1] * sp_pad)
            labels.append(list(b["labels"]) + [0.0] * sp_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "step_positions": torch.tensor(step_positions, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    return collate


class PRMTrainerWrapper:
    """High-level wrapper for PRM training from a SoupConfig.

    PRM (Process Reward Model) trains a scalar reward head to score each
    step of a reasoning chain. Data format: ``data.format='prm'`` rows
    with ``prompt`` + ``completions`` (list of steps) + ``labels`` (list
    of scalar rewards).
    """

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
        self._dataset: Optional[dict] = None

    def setup(self, dataset: dict) -> None:
        """Build the model + scalar reward head + HF Trainer."""
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base,
            trust_remote_code=self._trust_remote_code,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        hidden_size = base_model.config.hidden_size
        base_model.reward_head = nn.Linear(hidden_size, 1, bias=True)
        self.model = base_model
        self._dataset = dataset
        console.print(
            f"[green]PRM trainer ready[/]: base={cfg.base}, hidden={hidden_size}, "
            f"head=Linear({hidden_size}, 1)"
        )

    def train(self, **_kwargs) -> dict:
        """Run training with the PRM Trainer subclass."""
        if self.model is None:
            raise RuntimeError("PRMTrainerWrapper.train() called before setup()")
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments

        cfg = self.config
        tcfg = cfg.training
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_rows = _prepare_prm_dataset(
            self._dataset["train"], self.tokenizer, cfg.data.max_length
        )
        if not train_rows:
            raise RuntimeError(
                "PRM dataset preparation yielded zero usable rows. Check that "
                "rows have 'prompt', 'completions', and 'labels' with matching lengths."
            )
        eval_rows = None
        if "val" in self._dataset and self._dataset["val"]:
            eval_rows = _prepare_prm_dataset(
                self._dataset["val"], self.tokenizer, cfg.data.max_length
            )

        # v0.53.11 review fix (python-review HIGH) — bool is subclass of int,
        # so explicit bool reject before isinstance(int).
        if isinstance(tcfg.batch_size, bool) or not isinstance(tcfg.batch_size, int):
            bs = 1
        else:
            bs = tcfg.batch_size
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            report_to=self.report_to,
            remove_unused_columns=False,
            deepspeed=self.deepspeed_config,
        )

        prm_trainer_cls = make_prm_trainer_class(Trainer)
        collator = _build_collator(self.tokenizer)
        # v0.53.11 review fix (code-review MEDIUM) — wrap list[dict] in
        # datasets.Dataset.from_list for full HF Trainer compatibility.
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows) if eval_rows else None
        self.trainer = prm_trainer_cls(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        console.print("[green]Starting PRM training...[/]")
        result = self.trainer.train()
        self.trainer.save_model(str(output_dir))
        return {"status": "ok", "output_dir": str(output_dir), "metrics": result.metrics}
