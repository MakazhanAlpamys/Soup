"""v0.71.9 #193 — live unlearning trainer (NPO / SimNPO / RMU).

``UnlearnTrainerWrapper`` runs a compact, self-contained training loop (not a
TRL/HF ``Trainer`` subclass — the two-dataset forget/retain NPO objective is
cleaner with a hand-rolled loop). It loads a LoRA-wrapped policy model, a
frozen reference copy (NPO / RMU), and forget / retain JSONL datasets, then
optimises the per-method loss from :mod:`soup_cli.utils.unlearn_kernels`.

Backends:

* ``npo``    — Negative Preference Optimization: push the policy's forget-set
  log-prob below a frozen reference, with a weighted retain-set CE term.
* ``simnpo`` — length-normalised NPO without a reference model.
* ``rmu``    — Representation Misdirection: steer the residual stream toward a
  fixed control vector on forget inputs while preserving retain activations
  (measured against the frozen reference).

Validated on SmolLM2-135M (RTX 3050 4 GB). Heavy imports (torch / transformers
/ peft) are local.
"""

from __future__ import annotations

import json
import os
import stat
import time
from typing import Any, List, Optional, Tuple

from rich.console import Console

console = Console()

# Bounds — keep the loop bounded on a 4 GB box.
_MAX_STEPS_CAP = 2000
_MAX_ROWS = 5000
_MAX_ROW_BYTES = 1 * 1024 * 1024  # per-line cap (defends against multi-GB row)
_MAX_FILE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB dataset-file cap
_MAX_LENGTH = 256
_RMU_CONTROL_SCALE = 6.0
_DEFAULT_BETA = 0.1


def _validated_output_dir(output_dir: str) -> str:
    """Reject an output dir outside cwd or pointing at a symlink (TOCTOU)."""
    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output dir must be a non-empty string")
    if "\x00" in output_dir:
        raise ValueError("output dir must not contain null bytes")
    if not is_under_cwd(output_dir):
        raise ValueError(f"output dir must stay under cwd: {output_dir!r}")
    if os.path.lexists(output_dir) and stat.S_ISLNK(os.lstat(output_dir).st_mode):
        raise ValueError("output dir must not be a symlink")
    return output_dir


def _load_unlearn_rows(path: str) -> List[Tuple[str, str]]:
    """Load (prompt, target) pairs from a forget / retain JSONL file.

    Accepts ``{messages:[...]}`` (last user→assistant turn), ``{prompt,
    completion}``, ``{prompt, target}``, or ``{text}`` (whole text as target
    with an empty prompt). Skips malformed / empty rows. Cwd-contained.
    """
    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(path, str) or not path:
        raise ValueError("forget_set / retain_set path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("dataset path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"dataset path must stay under cwd: {path!r}")
    # TOCTOU: reject a pre-placed symlink at the dataset path before opening.
    st = os.lstat(path) if os.path.lexists(path) else None
    if st is None:
        raise FileNotFoundError(f"dataset not found: {path!r}")
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("dataset path must not be a symlink")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset not found: {path!r}")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(f"dataset file exceeds {_MAX_FILE_BYTES} bytes")
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            if i >= _MAX_ROWS:
                break
            if len(line) > _MAX_ROW_BYTES:
                continue  # skip a pathologically large single row
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair = _extract_pair(row)
            if pair is not None:
                pairs.append(pair)
    return pairs


def _extract_pair(row: object) -> Optional[Tuple[str, str]]:
    if not isinstance(row, dict):
        return None
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        prompt = ""
        target = ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if role in ("user", "system"):
                prompt = content
            elif role == "assistant":
                target = content
        if target:
            return (prompt, target)
        return None
    prompt = row.get("prompt")
    target = row.get("completion") or row.get("target") or row.get("output")
    if isinstance(prompt, str) and isinstance(target, str) and target:
        return (prompt, target)
    text = row.get("text")
    if isinstance(text, str) and text:
        return ("", text)
    return None


class UnlearnTrainerWrapper:
    """Live unlearning trainer for ``task='unlearn'`` (v0.71.9 #193)."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        try:
            self.config = config
            self.method = config.training.unlearn_method
        except AttributeError as exc:
            raise AttributeError(
                "UnlearnTrainerWrapper requires a SoupConfig with "
                f"training.unlearn_method set; got {exc}"
            ) from exc
        self._kwargs = dict(kwargs)
        self.device = kwargs.get("device")
        self.trust_remote_code = bool(kwargs.get("trust_remote_code", False))
        self._setup_called = False
        self.model: Any = None
        self.tokenizer: Any = None
        self.ref_model: Any = None
        self._dev: Optional[str] = None
        self._forget: List[Tuple[str, str]] = []
        self._retain: List[Tuple[str, str]] = []
        self._optimizer: Any = None
        # Kept None — this wrapper has no HF Trainer object (the push-callback
        # path in commands/train.py gracefully skips when .trainer is None).
        self.trainer: Any = None

    def setup(self, dataset: Any = None) -> None:
        """Load policy + (optional) frozen reference, LoRA, and datasets."""
        import torch
        from peft import LoraConfig, get_peft_model

        from soup_cli.utils.live_eval import load_model_and_tokenizer
        from soup_cli.utils.unlearning import validate_unlearn_method

        cfg = self.config
        tcfg = cfg.training
        method = validate_unlearn_method(self.method)
        self.method = method

        self.model, self.tokenizer, self._dev = load_model_and_tokenizer(
            cfg.base, device=self.device, trust_remote_code=self.trust_remote_code,
        )
        lora_cfg = LoraConfig(
            r=cfg.training.lora.r,
            lora_alpha=cfg.training.lora.alpha,
            lora_dropout=cfg.training.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.train()

        # NPO + RMU need a frozen reference copy of the base.
        if method in ("npo", "rmu"):
            ref, _, _ = load_model_and_tokenizer(
                cfg.base, device=self.device, trust_remote_code=self.trust_remote_code,
            )
            for p in ref.parameters():
                p.requires_grad_(False)
            ref.eval()
            self.ref_model = ref

        forget_path = cfg.data.forget_set
        self._forget = _load_unlearn_rows(forget_path)
        if not self._forget:
            raise ValueError(f"forget_set {forget_path!r} yielded no usable rows")
        if cfg.data.retain_set:
            self._retain = _load_unlearn_rows(cfg.data.retain_set)

        # NPO / SimNPO without a retain set has no utility anchor — the policy
        # is driven down on the forget set with nothing holding general
        # capability. Warn loudly (review HIGH).
        if method in ("npo", "simnpo") and not self._retain:
            console.print(
                f"[yellow]Unlearn warning:[/] method={method} has no "
                "retain_set; general capability is unanchored. Supply "
                "data.retain_set to preserve utility."
            )

        lr = float(tcfg.lr)
        self._optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=lr,
        )
        self._setup_called = True
        console.print(
            f"[green]Unlearn setup:[/] method={method} "
            f"forget={len(self._forget)} retain={len(self._retain)}"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, **kwargs: Any) -> dict:
        """Run the unlearn loop, save the adapter, return a result dict."""
        if not self._setup_called:
            raise RuntimeError(
                "UnlearnTrainerWrapper.train() called before setup(); "
                "call setup() first."
            )
        import torch

        from soup_cli.utils.live_eval import _tokenize_pair

        cfg = self.config
        tcfg = cfg.training
        epochs = max(1, int(tcfg.epochs))
        n_steps = min(_MAX_STEPS_CAP, epochs * len(self._forget))
        started = time.monotonic()
        dev = self._dev
        method = self.method

        # RMU: prepare a fixed control vector once.
        control_vec = None
        rmu_layer = None
        if method == "rmu":
            hidden = int(self.model.config.hidden_size)
            gen = torch.Generator(device="cpu").manual_seed(0)
            control_vec = (
                torch.randn(hidden, generator=gen).to(dev) * _RMU_CONTROL_SCALE
            )
            rmu_layer = self._resolve_rmu_layer()

        initial_loss = None
        final_loss = None
        step = 0
        retain_idx = 0
        while step < n_steps:
            for f_prompt, f_target in self._forget:
                if step >= n_steps:
                    break
                self._optimizer.zero_grad(set_to_none=True)
                if method in ("npo", "simnpo"):
                    loss = self._step_preference(
                        _tokenize_pair, f_prompt, f_target, method, dev,
                    )
                else:  # rmu
                    r_pair = (
                        self._retain[retain_idx % len(self._retain)]
                        if self._retain
                        else None
                    )
                    retain_idx += 1
                    loss = self._step_rmu(
                        _tokenize_pair, f_prompt, f_target, r_pair,
                        control_vec, rmu_layer, dev,
                    )
                if loss is None:
                    continue
                loss.backward()
                self._optimizer.step()
                lval = float(loss.item())
                if initial_loss is None:
                    initial_loss = lval
                final_loss = lval
                step += 1
            if step == 0:
                break

        output_dir = _validated_output_dir(cfg.output)
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        duration = time.monotonic() - started
        console.print(
            f"[green]Unlearn done:[/] {step} steps, "
            f"loss {initial_loss} -> {final_loss}, saved {output_dir}"
        )
        return {
            "initial_loss": initial_loss if initial_loss is not None else 0.0,
            "final_loss": final_loss if final_loss is not None else 0.0,
            "total_steps": step,
            "duration_secs": duration,
            "output_dir": output_dir,
        }

    def _step_preference(self, tokenize_pair, prompt, target, method, dev):
        import torch

        from soup_cli.utils.unlearn_kernels import (
            npo_loss,
            sequence_lengths,
            sequence_logprob,
            simnpo_loss,
        )

        input_ids, labels = tokenize_pair(
            self.tokenizer, prompt, target, max_length=_MAX_LENGTH
        )
        if (labels != -100).sum().item() == 0:
            return None
        input_ids = input_ids.to(dev)
        labels = labels.to(dev)
        retain_ce = self._retain_ce(tokenize_pair, dev)

        beta = _DEFAULT_BETA
        alpha = float(self.config.training.unlearn_alpha or 1.0)
        out = self.model(input_ids=input_ids)
        policy_logps = sequence_logprob(out.logits, labels)
        if method == "npo":
            with torch.no_grad():
                ref_out = self.ref_model(input_ids=input_ids)
                ref_logps = sequence_logprob(ref_out.logits, labels)
            forget_loss = npo_loss(policy_logps, ref_logps, beta=beta)
        else:  # simnpo
            lengths = sequence_lengths(labels)
            forget_loss = simnpo_loss(policy_logps, lengths, beta=beta)
        total = forget_loss
        if retain_ce is not None:
            total = total + alpha * retain_ce
        return total

    def _retain_ce(self, tokenize_pair, dev):
        """Standard CE on one retain pair (None when no retain set)."""
        if not self._retain:
            return None
        # Round-robin a single retain pair per step (cheap, keeps memory flat).
        idx = getattr(self, "_retain_ce_idx", 0)
        self._retain_ce_idx = idx + 1
        prompt, target = self._retain[idx % len(self._retain)]
        input_ids, labels = tokenize_pair(
            self.tokenizer, prompt, target, max_length=_MAX_LENGTH
        )
        if (labels != -100).sum().item() == 0:
            return None
        input_ids = input_ids.to(dev)
        labels = labels.to(dev)
        out = self.model(input_ids=input_ids, labels=labels)
        return out.loss

    def _resolve_rmu_layer(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        layers = _locate_decoder_layers(self.model)
        # Steer a mid layer.
        idx = min(len(layers) // 2, len(layers) - 1)
        return layers[idx]

    def _step_rmu(
        self, tokenize_pair, f_prompt, f_target, r_pair, control_vec, layer, dev,
    ):
        import torch

        from soup_cli.utils.unlearn_kernels import rmu_loss

        captured: List[object] = []

        def _hook(_mod, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden)

        f_ids, f_labels = tokenize_pair(
            self.tokenizer, f_prompt, f_target, max_length=_MAX_LENGTH
        )
        f_ids = f_ids.to(dev)

        handle = layer.register_forward_hook(_hook)
        try:
            captured.clear()
            self.model(input_ids=f_ids)
            if not captured:
                return None
            forget_acts = captured[-1][0].mean(dim=0)  # [hidden]

            retain_acts = None
            retain_frozen = None
            if r_pair is not None and self._retain:
                r_ids, _ = tokenize_pair(
                    self.tokenizer, r_pair[0], r_pair[1], max_length=_MAX_LENGTH
                )
                r_ids = r_ids.to(dev)
                captured.clear()
                self.model(input_ids=r_ids)
                if captured:
                    retain_acts = captured[-1][0].mean(dim=0)
                # Frozen reference activation for the same retain input.
                ref_layer = self._resolve_ref_rmu_layer()
                ref_captured: List[object] = []

                def _ref_hook(_mod, _args, output):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    ref_captured.append(hidden)

                ref_handle = ref_layer.register_forward_hook(_ref_hook)
                try:
                    with torch.no_grad():
                        self.ref_model(input_ids=r_ids)
                finally:
                    ref_handle.remove()
                if ref_captured:
                    retain_frozen = ref_captured[-1][0].mean(dim=0).detach()
        finally:
            handle.remove()

        if retain_acts is None or retain_frozen is None:
            # No retain set: pure forget steering.
            return torch.mean((forget_acts - control_vec) ** 2)
        alpha = float(self.config.training.unlearn_alpha or 1.0)
        return rmu_loss(
            forget_acts, control_vec, retain_acts, retain_frozen, alpha=alpha,
        )

    def _resolve_ref_rmu_layer(self):
        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        layers = _locate_decoder_layers(self.ref_model)
        idx = min(len(layers) // 2, len(layers) - 1)
        return layers[idx]
