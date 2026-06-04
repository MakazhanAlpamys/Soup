"""MoLE per-token routing trainer wrapper — v0.71.12 #222.

Mixture of LoRA Experts (Wu et al. 2024): a small gating network learns to
route each token to a weighted blend of N pre-trained task LoRAs. The base
model and every task adapter stay frozen — only the gate trains.

Data: plain SFT-style chat / text rows (the router learns from the standard
causal-LM objective on the blended output). Adapter paths come from
``training.mole_task_adapters``; the gate config (``num_task_adapters`` =
len(adapters), ``hidden_dim`` = base hidden size, ``top_k``, ``temperature``)
is built from the schema fields.

Heavy deps (torch / transformers / peft) are lazy-imported inside methods per
project policy — ``python -m soup_cli.cli --help`` must not pull torch.
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
def make_mole_trainer_class(base_cls: type) -> type:
    """Factory: build a ``_MoleTrainer`` subclass of HF ``Trainer``.

    ``compute_loss`` runs:
      1. base forward with adapters disabled -> last hidden state -> per-token
         routing weights ``g`` ``[B, T, N]`` from ``model.mole_gate``.
      2. one forward per task adapter -> ``logits_i`` ``[B, T, V]``.
      3. blended logits = ``sum_i g[..., i] * logits_i``.
      4. shifted causal-LM cross-entropy against ``labels`` (pad = -100).

    Only the gate has ``requires_grad`` so the optimizer trains the router and
    leaves the base + every task LoRA frozen.

    The ``lru_cache`` factory pattern (mirrors ``make_prm_trainer_class``) lets
    us subclass whichever HF Trainer the caller supplies without an import-time
    dependency.
    """
    import torch
    import torch.nn.functional as functional

    class _MoleTrainer(base_cls):  # type: ignore[misc, valid-type]
        """HF Trainer subclass for MoLE per-token routing."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            labels = inputs["labels"]
            gate = model.mole_gate
            adapter_names = model._soup_mole_adapter_names

            # Router input: the base model's last hidden state (adapters off).
            # The base + every task LoRA are frozen, so the router hidden and
            # the per-adapter logits carry NO trainable ancestor — compute them
            # under no_grad + detach so autograd never retains the (N+1) frozen
            # forward graphs. Only the gate's weights (applied below) require
            # grad, so the loss still back-props into the router correctly.
            with torch.no_grad(), model.disable_adapter():
                base_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            router_hidden = base_out.hidden_states[-1]  # [B, T, H]
            weights = gate(router_hidden.to(gate.gate.weight.dtype))  # [B, T, N]

            blended = None
            for i, name in enumerate(adapter_names):
                model.set_adapter(name)
                with torch.no_grad():
                    out_i = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                logits_i = out_i.logits.detach()  # [B, T, V] — frozen, no grad
                w_i = weights[..., i : i + 1].to(logits_i.dtype)
                term = w_i * logits_i
                blended = term if blended is None else blended + term
            # Reset to the first adapter so any between-step eval / callback
            # forward does not silently run only the last task adapter.
            model.set_adapter(adapter_names[0])

            if blended is None:  # defensive — adapter_names never empty
                zero = torch.zeros((), device=input_ids.device, requires_grad=True)
                return (zero, {}) if return_outputs else zero

            shift_logits = blended[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if return_outputs:
                return loss, {"logits": blended}
            return loss

    _MoleTrainer.__name__ = f"_MoleTrainer_{base_cls.__name__}"
    return _MoleTrainer


def _row_to_text(row: dict) -> str:
    """Best-effort text extraction from an SFT-style row for MoLE training."""
    if not isinstance(row, dict):
        return ""
    text = row.get("text")
    if isinstance(text, str) and text:
        return text
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [
            str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("content")
        ]
        if parts:
            return "\n".join(parts)
    prompt = row.get("prompt")
    completion = row.get("completion") or row.get("response")
    if isinstance(prompt, str) and prompt:
        return prompt + (str(completion) if completion else "")
    return ""


def _prepare_mole_dataset(
    raw_rows: list[dict], tokenizer: Any, max_length: int
) -> list[dict]:
    """Tokenise rows into (input_ids, attention_mask, labels) for causal LM."""
    prepared: list[dict] = []
    for row in raw_rows:
        text = _row_to_text(row)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=True)["input_ids"][:max_length]
        if not ids:
            continue
        prepared.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": list(ids),
            }
        )
    return prepared


def _build_collator(tokenizer: Any):
    """Pad input_ids / attention_mask / labels (pad label = -100)."""
    import torch

    # Explicit None check — a legitimate pad_token_id of 0 must not collapse
    # into the `or 0` fallback (classic falsy-zero trap).
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attention_mask.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


class MoleRoutingTrainerWrapper:
    """High-level wrapper for MoLE per-token routing from a SoupConfig.

    Loads the base model + N task LoRA adapters (all frozen), attaches a
    trainable gating kernel, and trains the gate via the standard causal-LM
    objective on the per-token blended output.
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
        self._adapter_names: list[str] = []
        self._dataset: Optional[dict] = None

    def setup(self, dataset: dict) -> None:
        """Load base + N task adapters (frozen) + the trainable gating kernel."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from soup_cli.utils.mole_routing import (
            MoleGatingConfig,
            build_gating_kernel,
        )

        if not isinstance(dataset, dict) or "train" not in dataset:
            raise ValueError(
                "MoleRoutingTrainerWrapper.setup() needs a dataset dict with a "
                "'train' key."
            )
        cfg = self.config
        tcfg = cfg.training
        adapters = list(tcfg.mole_task_adapters or [])
        if len(adapters) < 2:
            raise RuntimeError(
                "MoLE training requires >= 2 task adapters in "
                "training.mole_task_adapters."
            )

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

        # Load the first adapter via from_pretrained, then load_adapter for the
        # rest — PEFT multi-adapter pattern (matches v0.71.8 interference_live).
        adapter_names = [f"task_{i}" for i in range(len(adapters))]
        model = PeftModel.from_pretrained(
            base_model, adapters[0], adapter_name=adapter_names[0]
        )
        for name, path in zip(adapter_names[1:], adapters[1:]):
            model.load_adapter(path, adapter_name=name)

        # Freeze the base + every task LoRA — only the gate trains.
        for param in model.parameters():
            param.requires_grad_(False)

        top_k = tcfg.mole_top_k if tcfg.mole_top_k is not None else len(adapters)
        temperature = (
            tcfg.mole_temperature if tcfg.mole_temperature is not None else 1.0
        )
        gate_cfg = MoleGatingConfig(
            num_task_adapters=len(adapters),
            hidden_dim=hidden_size,
            temperature=temperature,
            top_k=top_k,
        )
        gate = build_gating_kernel(gate_cfg)
        gate.requires_grad_(True)
        if self.device == "cuda":
            gate = gate.to("cuda", dtype=torch.bfloat16)
        # nn.Module.__setattr__ registers the gate as a submodule, so its
        # params appear in model.parameters() for the optimizer; the plain
        # list attribute is stored in __dict__ (not registered).
        model.mole_gate = gate
        # Namespaced plain attribute (not a submodule) — avoids colliding with
        # any current/future PEFT-internal ``_``-prefixed attribute.
        model._soup_mole_adapter_names = adapter_names

        self.model = model
        self._adapter_names = adapter_names
        self._dataset = dataset
        n_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        console.print(
            f"[green]MoLE trainer ready[/]: base={cfg.base}, "
            f"adapters={len(adapters)}, top_k={top_k}, temp={temperature}, "
            f"gate=Linear({hidden_size}, {len(adapters)}), "
            f"trainable_params={n_trainable}"
        )

    def train(self, **_kwargs) -> dict:
        """Train the gating kernel with the MoLE Trainer subclass."""
        if self.model is None:
            raise RuntimeError(
                "MoleRoutingTrainerWrapper.train() called before setup()"
            )
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments

        cfg = self.config
        tcfg = cfg.training
        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_rows = _prepare_mole_dataset(
            self._dataset["train"], self.tokenizer, cfg.data.max_length
        )
        if not train_rows:
            raise RuntimeError(
                "MoLE dataset preparation yielded zero usable rows. Check that "
                "rows have a 'text' / 'messages' / 'prompt' field."
            )
        eval_rows = None
        if "val" in self._dataset and self._dataset["val"]:
            eval_rows = _prepare_mole_dataset(
                self._dataset["val"], self.tokenizer, cfg.data.max_length
            )

        # bool is subclass of int — explicit reject before isinstance(int).
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

        mole_trainer_cls = make_mole_trainer_class(Trainer)
        collator = _build_collator(self.tokenizer)
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows) if eval_rows else None
        self.trainer = mole_trainer_cls(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        console.print("[green]Starting MoLE gate training...[/]")
        result = self.trainer.train()
        # Persist the trained gate (the base + adapters are unchanged on disk).
        import torch

        gate_path = output_dir / "mole_gate.pt"
        torch.save(self.model.mole_gate.state_dict(), str(gate_path))
        return {
            "status": "ok",
            "output_dir": str(output_dir),
            "gate_path": str(gate_path),
            "metrics": result.metrics,
        }
