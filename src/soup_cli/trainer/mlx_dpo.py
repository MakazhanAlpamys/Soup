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


class PrefixUnstableError(PreferenceRowError):
    """The template renders the prompt differently once a completion follows."""


def _normalise_row(row: dict) -> dict:
    """The row in ONE of TRL's two layouts, via the loader's own converter.

    ``soup_cli.data.formats._convert_dpo`` is the contract every backend gets
    from ``load_dataset`` (#1866): text (three strings) or explicit
    conversational (a message-list prompt, answers without the prompt turns).
    It is idempotent, so rows the loader already converted pass through, and
    rows handed straight to ``setup()`` get the same reading and the same
    refusals, which name the field.
    """
    from soup_cli.data.formats import _convert_dpo

    try:
        return _convert_dpo(row)
    except (TypeError, ValueError) as exc:
        raise PreferenceRowError(str(exc)) from exc


def build_preference_example(
    row: dict,
    tokenizer: Any,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str = "keep_start",
) -> PreferenceExample:
    """Tokenize one ``{prompt, chosen, rejected}`` row the way TRL 0.29 does.

    Differences from TRL, on purpose: a tie (``chosen == rejected``) raises
    :class:`IdenticalPairError` so the caller can skip it (TRL trains it at a
    constant loss, zero gradient), and a template that is not prefix-stable for
    the row raises :class:`PrefixUnstableError` where TRL warns and slices
    anyway. Per-row ``tools`` / ``chat_template_kwargs`` are not applied.
    """
    row = _normalise_row(row)
    prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
    if chosen == rejected:
        raise IdenticalPairError(
            "dpo row has identical chosen and rejected completions; there is "
            "no completion to prefer"
        )

    if isinstance(prompt, list):
        prompt_ids = _as_ids(
            tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        )
        full_chosen = _as_ids(tokenizer.apply_chat_template(prompt + chosen, tokenize=True))
        full_rejected = _as_ids(tokenizer.apply_chat_template(prompt + rejected, tokenize=True))
    else:
        eos = getattr(tokenizer, "eos_token", None) or ""

        def with_eos(text: str) -> str:
            return text if not eos or text.endswith(eos) else text + eos

        prompt_ids = list(tokenizer.encode(prompt))
        full_chosen = list(tokenizer.encode(prompt + with_eos(chosen)))
        full_rejected = list(tokenizer.encode(prompt + with_eos(rejected)))

    if full_chosen == full_rejected:
        raise IdenticalPairError(
            "dpo row has identical chosen and rejected completions; there is "
            "no completion to prefer"
        )
    n = len(prompt_ids)
    if full_chosen[:n] != prompt_ids or full_rejected[:n] != prompt_ids:
        raise PrefixUnstableError(
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

    def _examples(self, rows: list, split: str = "train") -> tuple[list, int]:
        """Tokenized pairs, and how many tied rows were skipped.

        A row the template is not prefix-stable for is skipped too, counted and
        reported with the first offender, rather than ending a run whose model
        has already loaded: with a BPE tokenizer one text row whose prompt ends
        in a space is enough. A row the converter refuses still stops the run,
        with its index: that is malformed data, not a tokenizer quirk.
        """
        cfg = self.config
        max_length = int(getattr(cfg.data, "max_length", 2048) or 2048)
        examples, ties, unstable = [], 0, []
        for index, row in enumerate(rows):
            try:
                examples.append(self._build_example(row, max_length))
            except IdenticalPairError:
                ties += 1
            except PrefixUnstableError:
                unstable.append(index)
            except PreferenceRowError as exc:
                raise PreferenceRowError(
                    f"{self.objective.lower()} {split} row {index}: {exc}"
                ) from exc
        if unstable:
            console.print(
                f"[yellow]MLX {self.objective}: skipped {len(unstable)} {split} row(s) whose "
                "completion could not be located (the chat template or tokenizer "
                "renders the prompt differently once a completion follows; first: "
                f"row {unstable[0]})[/]"
            )
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
        total = len(examples)
        console.print(
            f"[dim]MLX {self.objective}: computing reference log-probs for {total} "
            f"{self.reference_unit}[/]"
        )
        every = max(batch_size, (total // 10) // batch_size * batch_size)
        for start in range(0, total, batch_size):
            if start and start % every == 0:
                console.print(f"[dim]  reference log-probs: {start}/{total}[/]")
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
                    f"[yellow]MLX {self.objective}: computing log-probs over the full vocabulary "
                    "for every position (no frozen vocab head found); long "
                    "sequences need more memory[/]"
                )
            return split
        self.model.eval()
        tokens, mask = self._verification_arrays(sample)
        full = sequence_logps(self.model, tokens, mask)
        chunked = sequence_logps(self.model, tokens, mask, head_split=split)
        same = mx.allclose(full, chunked, rtol=1e-3, atol=1e-2).item()
        self.model.train()
        if not same:
            console.print(
                f"[yellow]MLX {self.objective}: this architecture changes logits after the vocab "
                "head; using the full-vocabulary path (more memory)[/]"
            )
            return None
        return split

    # -- loop -------------------------------------------------------------

    # -- objective hooks --------------------------------------------------
    #
    # The loop below is shared by every native preference objective on MLX
    # (DPO here, ORPO in mlx_orpo.py, SimPO in mlx_simpo.py). A subclass
    # changes the tokenized example, the arrays a batch needs, what is
    # prepared before the first update, the loss and its metrics -- nothing
    # else.

    #: Name used in messages and the adapter metadata.
    objective = "DPO"
    #: Metrics ``_loss`` returns, in report order; all are batch means.
    metric_names: tuple = ("reward_accuracy", "reward_margin")

    #: An objective whose loss has no finite value on some inputs (ORPO's
    #: log1mexp at a mean log-prob of 0) stops the run instead of training on
    #: NaN. DPO keeps its documented behaviour: a diverged run reports NaN.
    fail_on_nonfinite = False

    def _build_example(self, row: dict, max_length: int) -> Any:
        return build_preference_example(
            row, self.tokenizer, max_length=max_length, max_prompt_length=max_length // 2,
        )

    def _batch_arrays(self, examples: list) -> tuple:
        return self._to_arrays(examples)

    def _verification_arrays(self, examples: list) -> tuple:
        """``(tokens, completion mask)`` for the head-split check."""
        return self._to_arrays(examples)

    #: What one precomputed reference entry scores, for the progress message.
    reference_unit = "pair(s)"

    def _resolve_batch_size(self) -> int:
        """``training.batch_size`` as this loop runs it (``auto`` is 1 here)."""
        tcfg = self.config.training
        return int(tcfg.batch_size) if isinstance(tcfg.batch_size, int) else 1

    def _plan_batches(self, n_rows: int, batch_size: int, epochs: int, seed: int) -> list:
        """Every epoch's batches of example indices, in training order."""
        return plan_batches(n_rows, batch_size, epochs, seed)

    def _check_train_rows(self, n_rows: int, batch_size: int, ties: int) -> None:
        """Refuse a training split this objective cannot batch."""
        if n_rows < batch_size:
            raise ValueError(
                f"MLX {self.objective}: {n_rows} training pair(s) remain, fewer than "
                f"training.batch_size ({batch_size})"
                + (f", after skipping {ties} tie(s)" if ties else "")
                + f". Lower training.batch_size to at most {max(1, n_rows)} "
                "or add rows."
            )

    def _on_plan_cut(self, dropped: list, kept: int, accum: int) -> None:
        """Called with the batches an accumulation cut leaves untrained; the
        shuffled plans of DPO and ORPO drop different rows each run, silently."""

    def _check_finite(self, loss: Any, split: str) -> None:
        import math as _math

        value = loss.item()
        if self.fail_on_nonfinite and not _math.isfinite(value):
            raise FloatingPointError(
                f"MLX {self.objective}: nonfinite loss ({value}) on the {split} split; "
                "lower training.lr or check the rows"
            )

    def _prepare_objective(self, train_examples: list, val_examples: list,
                           batch_size: int) -> None:
        """Run before the checkpoint loads: DPO's reference is the base model."""
        self._ref = {"train": self._reference_logps(train_examples, batch_size)}
        if val_examples:
            self._ref["val"] = self._reference_logps(val_examples, batch_size)

    def _loss(self, model: Any, arrays: tuple, split: str, rows: Any) -> tuple:
        tokens, mask = arrays
        ref_chosen, ref_rejected = (r[rows] for r in self._ref[split])
        logps = sequence_logps(model, tokens, mask, head_split=self._head_split)
        half = ref_chosen.shape[0]
        return dpo_loss(
            logps[:half], logps[half:], ref_chosen, ref_rejected,
            float(self.config.training.dpo_beta),
        )

    def _format_metrics(self, metrics: dict) -> str:
        return (
            f"reward acc {metrics['reward_accuracy']:.2f}, "
            f"margin {metrics['reward_margin']:.4f}"
        )

    def _objective_metadata(self) -> dict:
        return {
            "task": "dpo",
            "dpo_beta": float(self.config.training.dpo_beta),
            "dpo_loss_type": "sigmoid",
            "reference": "precomputed from the base model",
        }

    # -- loop -------------------------------------------------------------

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        self._require_mlx()

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map

        from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer

        cfg = self.config
        tcfg = cfg.training
        name = self.objective
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX backend: setup(dataset) must be called before train()")

        from soup_cli.utils.seeding import resolve_training_seed

        seed = resolve_training_seed(tcfg)
        data_seed = seed if tcfg.data_seed is None else int(tcfg.data_seed)
        mx.random.seed(seed)

        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = self._resolve_batch_size()
        accum = max(1, int(tcfg.gradient_accumulation_steps))

        train_examples, train_ties = self._examples(list(self._dataset.get("train", [])))
        val_examples, val_ties = self._examples(list(self._dataset.get("val", [])), "val")
        skipped = train_ties + val_ties
        if skipped:
            console.print(
                f"[yellow]MLX {name}: skipped {skipped} {name.lower()} row(s) with "
                "identical chosen and rejected (a tie carries no preference signal)[/]"
            )
        self._check_train_rows(len(train_examples), batch_size, train_ties)
        epoch_count = max(1, int(tcfg.epochs))
        plan = self._plan_batches(len(train_examples), batch_size, epoch_count, data_seed)
        batches_per_epoch = max(1, len(plan) // epoch_count)
        iters = len(plan)
        if accum > 1:
            # Like MLX SFT: no partial accumulation group is ever flushed.
            cut = max(accum, iters - iters % accum)
            if cut < iters:
                self._on_plan_cut(plan[cut:], cut, accum)
            iters = cut
            plan = (plan * math.ceil(iters / len(plan)))[:iters]
        total_updates = max(1, iters // accum)

        self._apply_lora(self.model)
        if tcfg.gradient_checkpointing:
            from mlx_lm.tuner.trainer import grad_checkpoint

            grad_checkpoint(self.model.layers[0])
        self._head_split = self._verified_head_split(train_examples[:batch_size])

        # Before any adapter weights move: this is the base model.
        self._prepare_objective(train_examples, val_examples, batch_size)
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

        def loss_fn(model, arrays, rows):
            return self._loss(model, arrays, "train", rows)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        names = self.metric_names

        def evaluate() -> Optional[dict]:
            if not val_examples:
                return None
            self.model.eval()
            total = {"loss": 0.0, **{k: 0.0 for k in names}}
            for start in range(0, len(val_examples), batch_size):
                chunk = val_examples[start : start + batch_size]
                arrays = self._batch_arrays(chunk)
                rows = mx.arange(start, start + len(chunk))
                loss, metrics = self._loss(self.model, arrays, "val", rows)
                mx.eval(loss, *(metrics[k] for k in names))
                self._check_finite(loss, "val")
                # Weighted by the chunk's size: a partial last chunk counts
                # for its rows, not as a whole batch.
                total["loss"] += loss.item() * len(chunk)
                for k in names:
                    total[k] += metrics[k].item() * len(chunk)
            self.model.train()
            out = {"val_loss": total["loss"] / len(val_examples)}
            out.update({f"val_{k}": total[k] / len(val_examples) for k in names})
            return out

        def save(path: Path) -> None:
            # Written beside the target and renamed into place: a crash mid-write
            # never leaves a complete-looking adapter file.
            partial = path.with_name(path.stem + ".partial.safetensors")
            mx.save_safetensors(str(partial), dict(tree_flatten(self.model.trainable_parameters())))
            partial.replace(path)

        # Written now as well as at the end, so a step checkpoint of a run that
        # stops early still sits beside the metadata `--resume` checks.
        self._write_adapter_config(
            output_dir, batch_size=batch_size, iters=iters, accum=accum,
            optimizer_plan=optimizer_plan, seed=seed, data_seed=data_seed,
        )

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
        group = {"loss": 0.0, **{k: 0.0 for k in names}}
        t0 = tic = time.time()
        try:
            for it, batch in enumerate(plan, start=1):
                chunk = [train_examples[i] for i in batch]
                arrays = self._batch_arrays(chunk)
                (loss, metrics), step_grads = loss_and_grad(self.model, arrays, mx.array(batch))
                grads = (
                    step_grads
                    if grads is None
                    else tree_map(lambda a, b: a + b, grads, step_grads)
                )
                mx.eval(loss, *(metrics[k] for k in names), grads)
                # mlx-lm's default (`clear_cache_threshold: 0`): free the
                # allocator cache every iteration. Batches have a new padded
                # width almost every step, so cached buffers rarely fit the
                # next one; kept, they grew the process to ~40 GB on a run
                # whose largest step needs under 4 GB.
                _clear_allocator_cache()
                self._check_finite(loss, "train")
                group["loss"] += loss.item()
                for k in names:
                    group[k] += metrics[k].item()

                # Counted in iterations, as mlx-lm's `steps_per_save` is for MLX
                # SFT, so the file names line up with `--resume auto`. Checked
                # before the accumulation gate: inside it, a cadence that is not
                # a multiple of the group size skipped saves. Mid-group, the
                # weights saved are those of the last update.
                if save_steps > 0 and it % save_steps == 0:
                    save(output_dir / f"{it:07d}_adapters.safetensors")
                if it % accum:
                    continue
                if accum > 1:
                    grads = tree_map(lambda g: g / accum, grads)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                grads = None
                update = it // accum
                step = {k: v / accum for k, v in group.items()}
                losses.append(step["loss"])
                group = {k: 0.0 for k in group}

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
                        loss=step["loss"], lr=self._current_lr(optimizer, tcfg.lr),
                        speed=speed, metrics=step, val=val,
                    )

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
        console.print(f"[green]MLX {name} training complete:[/] {output_dir}")
        return result

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _current_lr(optimizer: Any, fallback: float) -> float:
        try:
            return float(optimizer.learning_rate)
        except (TypeError, ValueError, AttributeError):
            return float(fallback)

    def _report(self, display, tracker, run_id, *, step, epoch, loss, lr, speed,
                metrics, val) -> None:
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
            msg = f"step {step}: loss {loss:.4f}, {self._format_metrics(metrics)}"
            if val:
                msg += f", val loss {val['val_loss']:.4f}"
                if "val_reward_accuracy" in val:
                    msg += f", val reward acc {val['val_reward_accuracy']:.2f}"
            console.print(f"[dim]{msg}[/]")
        if tracker is not None and run_id:
            try:
                tracker.log_metrics(
                    run_id=run_id, step=step, epoch=epoch, loss=loss, lr=lr,
                    speed=speed, gpu_mem=gpu_mem, val_loss=val_loss,
                )
            except Exception:  # noqa: BLE001 — telemetry must not kill a run
                pass
        if display is not None:
            # `soup ui` / GET /api/train/stream, in MLX SFT's shape. Only with a
            # display, as on the transformers path. Reward accuracy and margin
            # have no event field: they are printed without a display and
            # returned in train()'s result.
            try:
                from soup_cli.utils.sse_train_stream import TrainEvent
                from soup_cli.utils.train_event_buffer import push_train_event

                push_train_event(
                    TrainEvent(
                        type="metric", step=step, epoch=float(epoch), loss=float(loss),
                        lr=float(lr), val_loss=val_loss,
                    )
                )
            except Exception:  # noqa: BLE001 — telemetry must not kill a run
                pass

    def _write_adapter_config(self, output_dir: Path, *, batch_size, iters, accum,
                              optimizer_plan, seed, data_seed) -> None:
        """The same loadable shape MLX SFT writes, plus what the objective ran with."""
        cfg = self.config
        target = output_dir / "adapter_config.json"
        partial = output_dir / "adapter_config.json.partial"
        partial.write_text(
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
                    **self._objective_metadata(),
                    "seed": seed,
                    "data_seed": data_seed,
                },
                indent=2,
            )
        )
        partial.replace(target)
