"""MLX DPO trainer — native preference alignment on Apple Silicon.

``mlx-lm`` ships no DPO helper, so the loop is Soup's own, built on the same
pieces as MLX SFT: ``linear_to_lora_layers`` for the adapter, the #686
optimizer plan and the #749 gradient clipping wrapper.

Three decisions shape it:

- **The reference model is never loaded.** ``mlx-lm`` initialises LoRA ``B`` to
  zeros, so before the first update the policy *is* the base model. Reference
  log-probs are therefore computed once, in eval mode, before training starts
  -- TRL's ``precompute_ref_log_probs`` -- which halves memory against a second
  frozen copy. A warm start (``--resume``) computes them before the
  checkpoint's weights are loaded, so the reference stays the base model.
- **Tokenization mirrors TRL 0.29's DPO trainer** (text rows get EOS appended,
  conversational rows go through the chat template, the prompt is capped at
  ``max_length // 2`` and completions at what remains), so a ``soup.yaml``
  trains on the same tokens on either backend. Where TRL only *warns* that a
  template is not prefix-stable and then slices at the wrong offset, this
  refuses: a wrong slice supervises the wrong tokens.
- **The run is seeded.** ``training.seed`` seeds ``mx.random`` (LoRA ``A`` init,
  dropout) and the batch order; ``training.data_seed``, when set, owns the batch
  order instead, as it does on the transformers path.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from soup_cli.trainer._trl_compat import _truncate_tokens
from soup_cli.trainer.loss_summary import summarize_training_loss
from soup_cli.trainer.mlx_sft import (
    MLXSFTTrainerWrapper,
    _clipping_optimizer,
    build_mlx_adapter_config,
)

console = Console()


class PreferenceRowError(ValueError):
    """A preference row this trainer cannot turn into supervised tokens."""


class IdenticalPairError(PreferenceRowError):
    """Chosen and rejected are the same: the pair has zero DPO gradient.

    Real preference sets contain ties (UltraFeedback does). TRL trains on them
    at a constant ``log 2``; the trainer skips them and says how many.
    """


@dataclass(frozen=True)
class PreferenceExample:
    """One tokenized pair: a shared prompt and the two completions after it."""

    prompt_ids: list
    chosen_ids: list
    rejected_ids: list


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


def _as_ids(value: Any) -> list:
    """``apply_chat_template`` returns ids, or a dict of them under
    ``return_dict`` on some tokenizer builds; normalise to a flat list."""
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _split_conversations(row: dict) -> tuple[list, list, list]:
    """Return (prompt messages, chosen completion, rejected completion).

    Accepts TRL's explicit layout (prompt messages, completion messages) and
    the implicit one Soup's bundled sample uses (full conversations that repeat
    the prompt turns, with or without a string ``prompt`` alongside).
    """
    prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise PreferenceRowError(
            "dpo row mixes formats: chosen and rejected must both be message "
            "lists or both be strings"
        )
    if isinstance(prompt, list):
        n = len(prompt)
        if chosen[:n] == prompt and rejected[:n] == prompt:
            chosen, rejected = chosen[n:], rejected[n:]
        return list(prompt), list(chosen), list(rejected)

    shared = 0
    while (
        shared < min(len(chosen), len(rejected))
        and chosen[shared] == rejected[shared]
    ):
        shared += 1
    if shared == len(chosen) and shared == len(rejected):
        raise IdenticalPairError(
            "dpo row has identical chosen and rejected conversations; there is "
            "no completion to prefer"
        )
    if shared:
        return chosen[:shared], chosen[shared:], rejected[shared:]
    return [{"role": "user", "content": str(prompt)}], chosen, rejected


def build_preference_example(
    row: dict,
    tokenizer: Any,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str = "keep_start",
) -> PreferenceExample:
    """Tokenize one ``{prompt, chosen, rejected}`` row the way TRL 0.29 does."""
    for name in ("prompt", "chosen", "rejected"):
        if row.get(name) is None:
            raise PreferenceRowError(f"dpo.{name} must not be null")

    conversational = any(
        isinstance(row[name], list) for name in ("prompt", "chosen", "rejected")
    )
    if conversational:
        prompt_msgs, chosen_msgs, rejected_msgs = _split_conversations(row)
        prompt_ids = _as_ids(
            tokenizer.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=True
            )
        )
        full_chosen = _as_ids(
            tokenizer.apply_chat_template(prompt_msgs + chosen_msgs, tokenize=True)
        )
        full_rejected = _as_ids(
            tokenizer.apply_chat_template(prompt_msgs + rejected_msgs, tokenize=True)
        )
    else:
        if not isinstance(row["chosen"], str) or not isinstance(row["rejected"], str):
            raise PreferenceRowError(
                "dpo row mixes formats: chosen and rejected must both be message "
                "lists or both be strings"
            )
        eos = getattr(tokenizer, "eos_token", None) or ""
        prompt = str(row["prompt"])

        def with_eos(text: str) -> str:
            return text if not eos or text.endswith(eos) else text + eos

        prompt_ids = list(tokenizer.encode(prompt))
        full_chosen = list(tokenizer.encode(prompt + with_eos(row["chosen"])))
        full_rejected = list(tokenizer.encode(prompt + with_eos(row["rejected"])))

    if full_chosen == full_rejected:
        raise IdenticalPairError(
            "dpo row has identical chosen and rejected completions; there is "
            "no completion to prefer"
        )
    n = len(prompt_ids)
    if full_chosen[:n] != prompt_ids or full_rejected[:n] != prompt_ids:
        raise PreferenceRowError(
            "the tokenized prompt is not a prefix of the tokenized prompt + "
            "completion, so the completion cannot be located; this tokenizer or "
            "chat template is not prefix-stable for this row"
        )

    prompt_ids = _truncate_tokens(prompt_ids, max_prompt_length, truncation_mode)
    room = max(0, max_length - len(prompt_ids))
    return PreferenceExample(
        prompt_ids=prompt_ids,
        chosen_ids=full_chosen[n:][:room],
        rejected_ids=full_rejected[n:][:room],
    )


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def plan_batches(n_rows: int, batch_size: int, epochs: int, seed: int) -> list:
    """Every epoch's full batches, shuffled reproducibly from ``seed``.

    The order is drawn from a private ``random.Random`` so nothing else that
    touches the global RNGs can change it. A trailing partial batch is
    dropped, as ``mlx-lm`` does.
    """
    if n_rows < batch_size:
        raise ValueError(
            f"dpo dataset has {n_rows} row(s), fewer than batch_size={batch_size}"
        )
    rng = random.Random(seed)
    plan = []
    for _ in range(max(1, int(epochs))):
        order = list(range(n_rows))
        rng.shuffle(order)
        plan.extend(
            order[i : i + batch_size]
            for i in range(0, n_rows - batch_size + 1, batch_size)
        )
    return plan


def collate_pairs(examples: list) -> tuple[list, list]:
    """Stack chosen sequences, then rejected ones, right-padded with 0.

    The mask marks completion tokens -- the only ones whose log-probs DPO sums.
    Padding id 0 is never read: attention is causal and the mask is 0 there.
    """
    seqs, masks = [], []
    for side in ("chosen_ids", "rejected_ids"):
        for ex in examples:
            completion = getattr(ex, side)
            seqs.append(list(ex.prompt_ids) + list(completion))
            masks.append([0] * len(ex.prompt_ids) + [1] * len(completion))
    width = max(len(s) for s in seqs)
    tokens = [s + [0] * (width - len(s)) for s in seqs]
    mask = [m + [0] * (width - len(m)) for m in masks]
    return tokens, mask


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


#: Positions per vocab-head chunk. A chunk's float32 logits are
#: ``2 * batch_size * chunk_size * vocab * 4`` bytes: 640 MB for Qwen2.5's
#: 152k vocab at batch_size 2. Only one chunk is alive at a time.
DEFAULT_LOGIT_CHUNK = 128


def resolve_head_split(model: Any) -> Optional[tuple]:
    """``(backbone, head)`` for an mlx-lm causal LM, or None if unrecognised.

    mlx-lm models compute ``head(self.model(x))`` where the head is the tied
    embedding (``embed_tokens.as_linear``) or ``lm_head``. Some architectures
    do more after the head (Gemma's logit soft-capping), so the trainer checks
    the split against the full model on real data before relying on it.
    """
    from mlx.utils import tree_flatten

    backbone = getattr(model, "model", None)
    if backbone is None or not callable(backbone):
        return None
    args = getattr(model, "args", None)
    embed = getattr(backbone, "embed_tokens", None)
    if getattr(args, "tie_word_embeddings", False) and embed is not None:
        owner, head = embed, embed.as_linear
    elif getattr(model, "lm_head", None) is not None:
        owner = head = model.lm_head
    else:
        return None
    # The checkpointed chunk closes over the head's weights as constants, so a
    # head that trains (LoRA on lm_head, a full fine-tune) would get no
    # gradient through it. Only a frozen head may take the chunked path.
    if tree_flatten(owner.trainable_parameters()):
        return None
    return backbone, head


def _chunk_logps(head: Any, hidden: Any, targets: Any, mask: Any) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    logits = head(hidden).astype(mx.float32)
    nll = nn.losses.cross_entropy(logits, targets, reduction="none")
    return -(nll * mask).sum(axis=-1)


def sequence_logps(
    model: Any,
    tokens: Any,
    mask: Any,
    *,
    head_split: Optional[tuple] = None,
    chunk_size: int = DEFAULT_LOGIT_CHUNK,
) -> Any:
    """Summed log-prob of every masked token, one value per sequence.

    Without ``head_split`` the whole model runs and every position's logits are
    materialised (and kept for the backward pass): about 17 MB per padded
    position at batch_size 2 on a 152k vocab, which put a 1024-token batch past
    38 GB. With it, the backbone runs once and the head runs in checkpointed
    chunks from the first completion position on -- prompt positions never
    reach the head, and each chunk's logits are recomputed in the backward pass
    instead of stored.
    """
    import mlx.core as mx

    targets = tokens[:, 1:]
    target_mask = mask[:, 1:].astype(mx.float32)
    if head_split is None:
        import mlx.nn as nn

        logits = model(tokens[:, :-1]).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, targets, reduction="none")
        return -(nll * target_mask).sum(axis=-1)

    backbone, head = head_split
    hidden = backbone(tokens[:, :-1])
    active = target_mask.max(axis=0)
    width = int(active.shape[0])
    start = int(mx.argmax(active).item()) if width else 0
    total = mx.zeros((tokens.shape[0],), dtype=mx.float32)
    for lo in range(start, width, max(1, int(chunk_size))):
        hi = min(width, lo + int(chunk_size))
        # Only the hidden states are a checkpoint input: mx.checkpoint
        # differentiates every argument, and targets are not differentiable.
        chunk = mx.checkpoint(
            lambda h, t=targets[:, lo:hi], m=target_mask[:, lo:hi]: _chunk_logps(
                head, h, t, m
            )
        )
        total = total + chunk(hidden[:, lo:hi])
    return total


def dpo_loss(
    policy_chosen: Any,
    policy_rejected: Any,
    ref_chosen: Any,
    ref_rejected: Any,
    beta: float,
) -> tuple[Any, dict]:
    """Sigmoid DPO loss (Rafailov et al.), TRL's ``loss_type="sigmoid"``.

    ``-log σ(z)`` is computed as ``logaddexp(0, -z)`` so a large margin in
    either direction stays finite.
    """
    import mlx.core as mx

    chosen_rewards = beta * (policy_chosen - ref_chosen)
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    z = chosen_rewards - rejected_rewards
    loss = mx.logaddexp(mx.zeros_like(z), -z).mean()
    metrics = {
        "reward_accuracy": (chosen_rewards > rejected_rewards).astype(mx.float32).mean(),
        "reward_margin": z.mean(),
    }
    return loss, metrics


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


def _clear_allocator_cache() -> None:
    import mlx.core as mx

    if mx.get_cache_memory() > 0:
        mx.clear_cache()


class MLXDPOTrainerWrapper(MLXSFTTrainerWrapper):
    """DPO on MLX. Reuses MLX SFT's model loading, LoRA attach and warm start."""

    def _check_unsupported(self) -> None:
        tcfg = self.config.training
        unsupported = []
        if tcfg.quantization == "8bit":
            unsupported.append("quantization=8bit (use mlx-community 4bit models)")
        if self.config.data.chat_template:
            unsupported.append(
                "data.chat_template (MLX DPO renders with the model's own template)"
            )
        for name, label in (
            ("use_galore", "GaLore"),
            ("use_lorafa", "training.use_lorafa"),
            ("use_ring_attention", "Ring Attention"),
            ("use_flash_attn", "FlashAttention (MLX has its own attention kernels)"),
            ("use_liger", "training.use_liger"),
            ("use_mod", "training.use_mod"),
            ("moe_lora", "training.moe_lora"),
            ("quantization_aware", "training.quantization_aware"),
            ("use_fsdp2_compile", "training.use_fsdp2_compile"),
            ("loss_watchdog", "training.loss_watchdog"),
            ("loss_spike_recovery", "training.loss_spike_recovery"),
            ("grad_accum_auto_tune", "training.grad_accum_auto_tune"),
        ):
            if getattr(tcfg, name, False):
                unsupported.append(label)
        if tcfg.neftune_alpha is not None:
            unsupported.append("training.neftune_alpha")
        if isinstance(tcfg.gradient_checkpointing, str):
            unsupported.append(
                f"gradient_checkpointing tier {tcfg.gradient_checkpointing!r} "
                "(MLX has a single on/off switch; enabling it)"
            )
        if unsupported:
            console.print(
                "[yellow]MLX backend ignores: " + ", ".join(unsupported) + "[/]"
            )

    # -- data -------------------------------------------------------------

    def _examples(self, rows: list) -> tuple[list, int]:
        """Tokenized pairs, and how many tied rows were skipped."""
        cfg = self.config
        max_length = int(getattr(cfg.data, "max_length", 2048) or 2048)
        examples, ties = [], 0
        for index, row in enumerate(rows):
            try:
                examples.append(
                    build_preference_example(
                        row,
                        self.tokenizer,
                        max_length=max_length,
                        max_prompt_length=max_length // 2,
                    )
                )
            except IdenticalPairError:
                ties += 1
            except PreferenceRowError as exc:
                raise PreferenceRowError(f"dpo row {index}: {exc}") from exc
        return examples, ties

    @staticmethod
    def _to_arrays(examples: list) -> tuple[Any, Any]:
        import mlx.core as mx

        tokens, mask = collate_pairs(examples)
        return mx.array(tokens), mx.array(mask)

    def _reference_logps(self, examples: list, batch_size: int) -> tuple[Any, Any]:
        """Log-probs of the still-untrained policy: the reference model."""
        import mlx.core as mx

        self.model.eval()
        chosen, rejected = [], []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            tokens, mask = self._to_arrays(chunk)
            logps = mx.stop_gradient(
                sequence_logps(self.model, tokens, mask, head_split=self._head_split)
            )
            mx.eval(logps)
            chosen.append(logps[: len(chunk)])
            rejected.append(logps[len(chunk) :])
            _clear_allocator_cache()
        self.model.train()
        return mx.concatenate(chosen), mx.concatenate(rejected)

    _head_split: Optional[tuple] = None

    def _verified_head_split(self, sample: list) -> Optional[tuple]:
        """The chunked head path, if it reproduces the full model on ``sample``.

        An architecture that post-processes logits after the head (soft-capping,
        scaling) would silently train against different log-probs; one batch of
        real data decides, and a mismatch costs memory, not correctness.
        """
        import mlx.core as mx

        split = resolve_head_split(self.model)
        if split is None or not sample:
            if split is None:
                console.print(
                    "[yellow]MLX DPO: computing log-probs over the full vocabulary "
                    "for every position (no frozen vocab head found); long "
                    "sequences need more memory[/]"
                )
            return split
        self.model.eval()
        tokens, mask = self._to_arrays(sample)
        full = sequence_logps(self.model, tokens, mask)
        chunked = sequence_logps(self.model, tokens, mask, head_split=split)
        same = mx.allclose(full, chunked, rtol=1e-3, atol=1e-2).item()
        self.model.train()
        if not same:
            console.print(
                "[yellow]MLX DPO: this architecture changes logits after the vocab "
                "head; using the full-vocabulary path (more memory)[/]"
            )
            return None
        return split

    # -- loop -------------------------------------------------------------

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        self._require_mlx()

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map

        from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer

        cfg = self.config
        tcfg = cfg.training
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX backend: setup(dataset) must be called before train()")

        seed = int(tcfg.seed) if tcfg.seed is not None else 42
        data_seed = int(tcfg.data_seed) if tcfg.data_seed is not None else seed
        mx.random.seed(seed)

        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(tcfg.batch_size) if isinstance(tcfg.batch_size, int) else 1
        accum = max(1, int(tcfg.gradient_accumulation_steps))
        beta = float(tcfg.dpo_beta)

        train_examples, train_ties = self._examples(list(self._dataset.get("train", [])))
        val_examples, val_ties = self._examples(list(self._dataset.get("val", [])))
        skipped = train_ties + val_ties
        if skipped:
            console.print(
                f"[yellow]MLX DPO: skipped {skipped} dpo row(s) with identical "
                "chosen and rejected (a tie carries no preference signal)[/]"
            )
        plan = plan_batches(len(train_examples), batch_size, int(tcfg.epochs), data_seed)
        iters = len(plan)
        if accum > 1:
            # Like MLX SFT: no partial accumulation group is ever flushed.
            iters = max(accum, iters - iters % accum)
            plan = (plan * math.ceil(iters / len(plan)))[:iters]
        total_updates = max(1, iters // accum)
        batches_per_epoch = max(1, len(train_examples) // batch_size)

        self._apply_lora(self.model)
        if tcfg.gradient_checkpointing:
            from mlx_lm.tuner.trainer import grad_checkpoint

            grad_checkpoint(self.model.layers[0])
        self._head_split = self._verified_head_split(train_examples[:batch_size])

        # Before any adapter weights move: this is the base model.
        ref_train = self._reference_logps(train_examples, batch_size)
        ref_val = (
            self._reference_logps(val_examples, batch_size) if val_examples else None
        )
        if resume_from_checkpoint is not None:
            self._load_checkpoint_weights(resume_from_checkpoint)

        optimizer_plan = plan_optimizer(
            lr=float(tcfg.lr),
            optimizer=str(getattr(tcfg, "optimizer", "adamw_torch")),
            scheduler=str(getattr(tcfg, "scheduler", "cosine")),
            warmup_ratio=float(getattr(tcfg, "warmup_ratio", 0.0) or 0.0),
            weight_decay=float(getattr(tcfg, "weight_decay", 0.0) or 0.0),
            total_updates=total_updates,
        )
        for warning in optimizer_plan.warnings:
            console.print(f"[yellow]MLX backend: {warning}[/]")
        optimizer = _clipping_optimizer(
            build_optimizer(optimizer_plan), float(tcfg.max_grad_norm)
        )

        def loss_fn(model, tokens, mask, ref_chosen, ref_rejected):
            logps = sequence_logps(model, tokens, mask, head_split=self._head_split)
            half = ref_chosen.shape[0]
            loss, metrics = dpo_loss(
                logps[:half], logps[half:], ref_chosen, ref_rejected, beta
            )
            return loss, metrics

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        def evaluate() -> Optional[dict]:
            if not val_examples:
                return None
            self.model.eval()
            losses, accs = [], []
            for start in range(0, len(val_examples), batch_size):
                chunk = val_examples[start : start + batch_size]
                tokens, mask = self._to_arrays(chunk)
                end = start + len(chunk)
                loss, metrics = loss_fn(
                    self.model, tokens, mask, ref_val[0][start:end], ref_val[1][start:end]
                )
                mx.eval(loss, metrics["reward_accuracy"])
                losses.append(loss.item() * len(chunk))
                accs.append(metrics["reward_accuracy"].item() * len(chunk))
            self.model.train()
            return {
                "val_loss": sum(losses) / len(val_examples),
                "val_reward_accuracy": sum(accs) / len(val_examples),
            }

        def save(path: Path) -> None:
            mx.save_safetensors(str(path), dict(tree_flatten(self.model.trainable_parameters())))

        save_steps = int(getattr(tcfg, "save_steps", 0) or 0)
        logging_steps = max(1, int(getattr(tcfg, "logging_steps", 10) or 10))
        epochs = float(tcfg.epochs)
        losses: list = []
        final_eval: Optional[dict] = None
        logged_at = 0
        if display is not None:
            display.start(iters)

        self.model.train()
        grads = None
        group_loss = group_acc = group_margin = 0.0
        t0 = tic = time.time()
        try:
            for it, batch in enumerate(plan, start=1):
                chunk = [train_examples[i] for i in batch]
                tokens, mask = self._to_arrays(chunk)
                idx = mx.array(batch)
                (loss, metrics), step_grads = loss_and_grad(
                    self.model, tokens, mask, ref_train[0][idx], ref_train[1][idx]
                )
                grads = (
                    step_grads
                    if grads is None
                    else tree_map(lambda a, b: a + b, grads, step_grads)
                )
                mx.eval(loss, metrics["reward_accuracy"], metrics["reward_margin"], grads)
                # mlx-lm's default (`clear_cache_threshold: 0`): free the
                # allocator cache every iteration. Batches have a new padded
                # width almost every step, so cached buffers rarely fit the
                # next one; kept, they grew the process to ~40 GB on a run
                # whose largest step needs under 4 GB.
                _clear_allocator_cache()
                group_loss += loss.item()
                group_acc += metrics["reward_accuracy"].item()
                group_margin += metrics["reward_margin"].item()

                if it % accum:
                    continue
                if accum > 1:
                    grads = tree_map(lambda g: g / accum, grads)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                grads = None
                update = it // accum
                step_loss = group_loss / accum
                losses.append(step_loss)
                reward_acc, margin = group_acc / accum, group_margin / accum
                group_loss = group_acc = group_margin = 0.0

                # Evaluate once per epoch: when this update crossed an epoch
                # boundary, or on the last update.
                crossed = it // batches_per_epoch > (it - accum) // batches_per_epoch
                val = evaluate() if crossed or it == iters else None
                if val is not None:
                    final_eval = val

                if update % logging_steps == 0 or it == iters or val is not None:
                    now = time.time()
                    speed = (it - logged_at) / max(now - tic, 1e-9)
                    tic, logged_at = now, it
                    self._report(
                        display, tracker, run_id,
                        step=it, epoch=it / iters * epochs,
                        loss=step_loss, lr=self._current_lr(optimizer, tcfg.lr),
                        speed=speed, reward_acc=reward_acc, margin=margin, val=val,
                    )

                if save_steps > 0 and it % save_steps == 0:
                    save(output_dir / f"{it:07d}_adapters.safetensors")
        finally:
            if display is not None:
                display.stop()
        duration = time.time() - t0

        save(output_dir / "adapters.safetensors")
        self._write_adapter_config(
            output_dir, batch_size=batch_size, iters=iters, accum=accum,
            optimizer_plan=optimizer_plan, seed=seed, data_seed=data_seed,
        )

        result = {
            **summarize_training_loss([{"loss": loss} for loss in losses]),
            "losses": losses,
            "total_steps": iters,
            "duration_secs": duration,
            "duration": f"{duration:.0f}s",
            "output_dir": str(output_dir),
            "skipped_identical_pairs": skipped,
        }
        if final_eval is not None:
            result.update(final_eval)
        console.print(f"[green]MLX DPO training complete:[/] {output_dir}")
        return result

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _current_lr(optimizer: Any, fallback: float) -> float:
        try:
            return float(optimizer.learning_rate)
        except (TypeError, ValueError, AttributeError):
            return float(fallback)

    @staticmethod
    def _report(display, tracker, run_id, *, step, epoch, loss, lr, speed,
                reward_acc, margin, val) -> None:
        val_loss = val["val_loss"] if val else None
        try:
            import mlx.core as mx

            peak = (getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory)()
            gpu_mem = f"{peak / 1e9:.3f} GB"
        except Exception:  # noqa: BLE001 — telemetry must not kill a run
            gpu_mem = ""
        if display is not None:
            display.update(
                step=step, epoch=epoch, loss=loss, lr=lr, speed=speed,
                gpu_mem=gpu_mem, val_loss=val_loss,
            )
        else:
            msg = (
                f"step {step}: loss {loss:.4f}, reward acc {reward_acc:.2f}, "
                f"margin {margin:.4f}"
            )
            if val:
                msg += (
                    f", val loss {val['val_loss']:.4f}, "
                    f"val reward acc {val['val_reward_accuracy']:.2f}"
                )
            console.print(f"[dim]{msg}[/]")
        if tracker is not None and run_id:
            try:
                tracker.log_metrics(
                    run_id=run_id, step=step, epoch=epoch, loss=loss, lr=lr,
                    speed=speed, gpu_mem=gpu_mem, val_loss=val_loss,
                )
            except Exception:  # noqa: BLE001 — telemetry must not kill a run
                pass

    def _write_adapter_config(self, output_dir: Path, *, batch_size, iters, accum,
                              optimizer_plan, seed, data_seed) -> None:
        """The same loadable shape MLX SFT writes, plus what DPO ran with."""
        cfg = self.config
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "model": str(cfg.base),
                    "num_layers": len(getattr(self.model, "layers", [])),
                    "batch_size": batch_size,
                    "iters": iters,
                    "learning_rate": float(cfg.training.lr),
                    "max_seq_length": int(getattr(cfg.data, "max_length", 2048) or 2048),
                    "adapter_path": str(output_dir),
                    "lora_parameters": build_mlx_adapter_config(
                        cfg.training.lora, adapter_path=str(output_dir)
                    )["lora_parameters"],
                    "grad_accumulation_steps": accum,
                    **optimizer_plan.as_metadata(),
                    "max_grad_norm": float(cfg.training.max_grad_norm),
                    "task": "dpo",
                    "dpo_beta": float(cfg.training.dpo_beta),
                    "dpo_loss_type": "sigmoid",
                    "reference": "precomputed from the base model",
                    "seed": seed,
                    "data_seed": data_seed,
                },
                indent=2,
            )
        )
