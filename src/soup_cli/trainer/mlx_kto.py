"""MLX KTO trainer — unpaired preference alignment on Apple Silicon.

The objective is TRL 0.29.1's ``experimental/kto/kto_trainer.py``, the trainer
Soup's transformers backend runs for ``task: kto`` (``trainer/kto.py`` forwards
``kto_beta`` and ``max_length``; everything else stays at ``KTOConfig``'s
defaults: ``loss_type="kto"``, both class weights 1.0):

- ``r`` = policy minus reference **summed** completion log-probability
  (``get_batch_logps(..., average_log_prob=False)``);
- ``z`` = ``mean(policy_KL - reference_KL)`` over the batch's KL rows,
  detached, then clamped at 0 (``kto_loss``);
- desirable rows lose ``w_d * (1 - sigmoid(beta * (r - z)))``, undesirable rows
  ``w_u * (1 - sigmoid(beta * (z - r)))``; the loss is the mean over every row
  of the batch (TRL concatenates both classes, then takes ``nanmean``), so a
  class missing from a batch simply contributes nothing.

The KL rows are TRL's: each example's prompt with the completion of the
previous example in its group, the first taking the last (``_get_kl_dataset``,
mapped with ``batch_size=per_device_train_batch_size``; a last, shorter group
rotates within itself, and a group of one is its own KL row). TRL's KTO samples
batches **sequentially** (``_get_train_sampler`` returns a ``SequentialSampler``)
and keeps the last partial batch, so a training batch is exactly one KL group;
this trainer plans its batches the same way.

The loop, LoRA attach, optimizer plan, clipping, seeding, chunked head, adapter
writer and reference precompute are MLX DPO's
(:class:`~soup_cli.trainer.mlx_dpo.MLXDPOTrainerWrapper`): the reference
log-probs of every matched and KL row are computed once from the base model
before the first update (LoRA ``B`` starts at zero), and the policy's KL
log-probs are computed live, without gradient.

Tokenization is TRL's KTO ``_tokenize`` / ``_process_tokens``: rows are rendered
to text first (``apply_chat_template`` with TRL's common-prefix rule), the
completion is located with TRL's one-token merge rule, BOS is prepended to the
prompt and EOS appended to the completion when missing, and an over-long row
loses the end of its completion (the prompt is never cut; there is no prompt
cap in 0.29.1). A row whose completion keeps no token is refused, where TRL
would train the prompt plus a bare EOS (or slice past the prompt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

from soup_cli.trainer.mlx_dpo import (
    MLXDPOTrainerWrapper,
    PreferenceRowError,
    sequence_logps,
)
from soup_cli.trainer.mlx_orpo import _common_prefix, _tokenized_answer

console = Console()

#: ``KTOConfig``'s defaults in TRL 0.29.1, which Soup's transformers path keeps.
KTO_LOSS_TYPE = "kto"
DESIRABLE_WEIGHT = 1.0
UNDESIRABLE_WEIGHT = 1.0

#: TRL refuses a per-device batch of 1: the KL row would be the matched row.
KTO_MIN_BATCH = 2


@dataclass(frozen=True)
class KtoExample:
    """One tokenized row: its prompt, its completion and its KL completion.

    The KL row shares the prompt; its completion is another row's, re-cut to
    this prompt's room, as TRL's second ``_process_tokens`` pass does.
    """

    prompt_ids: list
    completion_ids: list
    kl_completion_ids: list
    label: bool


# --------------------------------------------------------------------------
# Tokenization (TRL 0.29.1 KTO _tokenize / _process_tokens / _get_kl_dataset)
# --------------------------------------------------------------------------


def _normalise_row(row: dict) -> dict:
    """The row in one layout via the loader's converter.

    Labels are read by the loader's KTO converter (``_convert_kto``) when
    ``load_dataset`` loads the file, exactly as on the transformers backend, so
    every row that reaches the trainer that way already has a bool label. The
    bool check below only guards rows handed to ``setup()`` without the loader:
    those are not given a second, different reading.
    """
    from soup_cli.data.formats import _convert_kto

    if "label" not in row:
        raise PreferenceRowError("the row has no 'label' (true for desirable, false for not)")
    label = row["label"]
    if not isinstance(label, bool):
        raise PreferenceRowError(
            f"label must be true or false, got {label!r} ({type(label).__name__}); "
            "rows that bypass the loader are not converted"
        )
    try:
        return _convert_kto(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreferenceRowError(str(exc)) from exc


def render_kto_text(row: dict, tokenizer: Any) -> tuple[str, str]:
    """``(prompt, completion)`` text, as ``trl.data_utils.apply_chat_template``.

    Text rows pass through. Conversational rows render the prompt with a
    generation prompt (or ``continue_final_message`` after an assistant turn),
    then ``prompt + completion``; the prompt is narrowed to its common prefix
    with that rendering and the completion is what follows it.
    """
    prompt, completion = row["prompt"], row["completion"]
    if not isinstance(prompt, list):
        return prompt, completion
    last_role = prompt[-1]["role"]
    if last_role in ("user", "tool"):
        kwargs = {"add_generation_prompt": True, "continue_final_message": False}
    elif last_role == "assistant":
        kwargs = {"add_generation_prompt": False, "continue_final_message": True}
    else:
        raise PreferenceRowError(f"the prompt ends with an unsupported role {last_role!r}")
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, **kwargs)
    full = tokenizer.apply_chat_template(prompt + completion, tokenize=False)
    prompt_text = _common_prefix(prompt_text, full)
    return prompt_text, full[len(prompt_text):]


def process_kto_tokens(prompt_ids: list, answer_ids: list, tokenizer: Any,
                       max_length: int) -> tuple[list, list, int]:
    """``(prompt_ids, completion_ids, answer tokens kept)`` after TRL's ``_process_tokens``.

    The room is ``max_length`` less one for a BOS and one for an EOS that are
    still to be added; a row past it loses the end of its answer. BOS is then
    prepended to the prompt and EOS appended to the (cut) answer when missing.
    A prompt longer than the room keeps no answer token here, where TRL's
    negative slice would keep some and train past ``max_length``;
    :func:`build_kto_examples` refuses such a matched row.
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise PreferenceRowError("the tokenizer has no eos_token_id to end the completion")
    prompt_ids, answer_ids = list(prompt_ids), list(answer_ids)
    limit = int(max_length)
    if prompt_ids and bos != prompt_ids[0]:
        limit -= 1
    if answer_ids and eos != answer_ids[-1]:
        limit -= 1
    if len(prompt_ids) + len(answer_ids) > limit:
        answer_ids = answer_ids[: max(0, limit - len(prompt_ids))]
    kept = len(answer_ids)
    if bos is not None and (not prompt_ids or bos != prompt_ids[0]):
        prompt_ids = [bos] + prompt_ids
    if not answer_ids or eos != answer_ids[-1]:
        answer_ids = answer_ids + [eos]
    return prompt_ids, answer_ids, kept


def kl_partners(n_rows: int, group_size: int) -> list:
    """Which row's completion each row's KL row borrows.

    TRL's ``_get_kl_dataset`` rotates the completions of each mapped batch by
    one (``[last] + all[:-1]``) and ``Dataset.map(batched=True, batch_size=n)``
    hands it consecutive groups of ``n`` rows, the last one shorter when ``n``
    does not divide the dataset; a last group of one keeps its own completion.
    """
    size = max(1, int(group_size))
    partners = []
    for start in range(0, n_rows, size):
        stop = min(n_rows, start + size)
        partners.extend([stop - 1] + list(range(start, stop - 1)))
    return partners


def build_kto_examples(rows: list, tokenizer: Any, *, max_length: int,
                       group_size: int) -> list:
    """Tokenize ``{prompt, completion, label}`` rows and their KL rows like TRL 0.29.1.

    Raises :class:`PreferenceRowError` (prefixed ``row <index>:``) for a row
    with no usable label or no completion token, so a run never starts on
    shifted KL groups.
    """
    raw = []
    for index, row in enumerate(rows):
        try:
            norm = _normalise_row(row)
            prompt, completion = render_kto_text(norm, tokenizer)
            prompt_ids, answer_ids = _tokenized_answer(tokenizer, prompt, completion)
            if not answer_ids:
                raise PreferenceRowError(
                    "the completion is empty: there is no completion token to score"
                )
            full_prompt, full_answer, kept = process_kto_tokens(
                prompt_ids, answer_ids, tokenizer, max_length
            )
            if not kept:
                raise PreferenceRowError(
                    f"no completion token fits in data.max_length ({max_length}) after the "
                    f"{len(full_prompt)}-token prompt"
                )
        except PreferenceRowError as exc:
            raise PreferenceRowError(f"row {index}: {exc}") from exc
        raw.append((full_prompt, full_answer, list(answer_ids), norm["label"]))

    examples = []
    for (prompt_ids, completion_ids, _, label), j in zip(raw, kl_partners(len(raw), group_size)):
        # TRL's KL pass re-processes this (BOS-prefixed) prompt with the
        # partner's uncut answer.
        _, kl_ids, _ = process_kto_tokens(prompt_ids, raw[j][2], tokenizer, max_length)
        examples.append(KtoExample(prompt_ids, completion_ids, kl_ids, label))
    return examples


def collate_kto(examples: list) -> tuple[list, list]:
    """``(tokens, completion mask)``: the matched rows, then the KL rows,
    right-padded with 0. The mask marks the completion tokens whose
    log-probs are summed; padding is never read."""
    seqs, masks = [], []
    for side in ("completion_ids", "kl_completion_ids"):
        for ex in examples:
            completion = getattr(ex, side)
            seqs.append(list(ex.prompt_ids) + list(completion))
            masks.append([0] * len(ex.prompt_ids) + [1] * len(completion))
    width = max(len(s) for s in seqs)
    return ([s + [0] * (width - len(s)) for s in seqs],
            [m + [0] * (width - len(m)) for m in masks])


def sequential_batches(n_rows: int, batch_size: int, epochs: int) -> list:
    """TRL's KTO batches: dataset order every epoch, the last partial batch kept
    (``SequentialSampler``, ``DataLoader(drop_last=False)``)."""
    size = max(1, int(batch_size))
    one = [list(range(i, min(n_rows, i + size))) for i in range(0, n_rows, size)]
    return [list(b) for _ in range(max(1, int(epochs))) for b in one]


# --------------------------------------------------------------------------
# Scoring and loss
# --------------------------------------------------------------------------


def kto_logps(model: Any, tokens: Any, mask: Any, n: int, *,
              head_split: Any = None) -> tuple[Any, Any]:
    """``(policy logps, policy KL logps)`` for a collated batch of ``n`` examples.

    Two forwards, as TRL runs them: the KL rows first and without gradient
    (``_compute_kl_logps`` under ``torch.no_grad``), then the matched rows.
    """
    import mlx.core as mx

    kl = mx.stop_gradient(
        sequence_logps(model, tokens[n:], mask[n:], head_split=head_split)
    )
    return sequence_logps(model, tokens[:n], mask[:n], head_split=head_split), kl


def kto_loss(
    policy_logps: Any,
    policy_kl_logps: Any,
    reference_logps: Any,
    reference_kl_logps: Any,
    labels: Any,
    beta: float,
    *,
    desirable_weight: float = DESIRABLE_WEIGHT,
    undesirable_weight: float = UNDESIRABLE_WEIGHT,
) -> tuple[Any, dict]:
    """TRL 0.29.1's ``kto_loss`` with ``loss_type="kto"``, averaged over the batch.

    ``1 - sigmoid(x)`` is computed as ``sigmoid(-x)``. Metrics are sums over
    the batch divided by its row count (``chosen_share`` is the desirable
    fraction), so they stay finite for a single-label batch and average
    correctly over accumulation and validation chunks; a class mean is the
    ratio of its ``*_per_row`` metric to its share.
    """
    import mlx.core as mx

    labels = labels.astype(mx.bool_)
    kl = (policy_kl_logps - reference_kl_logps).astype(mx.float32).mean()
    z = mx.maximum(mx.stop_gradient(kl), 0.0)
    r = (policy_logps - reference_logps).astype(mx.float32)
    chosen = desirable_weight * mx.sigmoid(-beta * (r - z))
    rejected = undesirable_weight * mx.sigmoid(-beta * (z - r))
    loss = mx.where(labels, chosen, rejected).mean()
    rows = labels.size
    is_chosen = labels.astype(mx.float32)
    rewards = beta * mx.stop_gradient(r)
    logps = mx.stop_gradient(policy_logps).astype(mx.float32)
    metrics = {
        "kl": z,
        "chosen_share": is_chosen.mean(),
        "chosen_rewards_per_row": (rewards * is_chosen).sum() / rows,
        "rejected_rewards_per_row": (rewards * (1 - is_chosen)).sum() / rows,
        "chosen_logps_per_row": (logps * is_chosen).sum() / rows,
        "rejected_logps_per_row": (logps * (1 - is_chosen)).sum() / rows,
    }
    return loss, metrics


def class_means(metrics: dict, prefix: str = "") -> dict:
    """TRL's logged ``rewards/chosen`` etc. from the per-row metrics: each class
    mean is its per-row sum over its share; a class with no rows is left out."""
    share = metrics[f"{prefix}chosen_share"]
    out = {}
    for side, part in (("chosen", share), ("rejected", 1.0 - share)):
        if part > 1e-12:
            for what in ("rewards", "logps"):
                out[f"{prefix}{what}_{side}"] = metrics[f"{prefix}{side}_{what}_per_row"] / part
    if f"{prefix}rewards_chosen" in out and f"{prefix}rewards_rejected" in out:
        out[f"{prefix}rewards_margins"] = (
            out[f"{prefix}rewards_chosen"] - out[f"{prefix}rewards_rejected"]
        )
    return out


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class MLXKTOTrainerWrapper(MLXDPOTrainerWrapper):
    """KTO on MLX: MLX DPO's loop and reference precompute with KTO's rows and loss."""

    objective = "KTO"
    reference_unit = "row(s) and their KL rows"
    metric_names = (
        "kl", "chosen_share", "chosen_rewards_per_row", "rejected_rewards_per_row",
        "chosen_logps_per_row", "rejected_logps_per_row",
    )
    fail_on_nonfinite = True

    def _resolve_batch_size(self) -> int:
        """Never below 2, ``auto`` included (TRL refuses 1; see #1420)."""
        return max(KTO_MIN_BATCH, super()._resolve_batch_size())

    def _plan_batches(self, n_rows: int, batch_size: int, epochs: int, seed: int) -> list:
        """Dataset order, the last partial batch kept: TRL's sequential sampler.
        ``seed`` does not reorder anything, as on the transformers path."""
        return sequential_batches(n_rows, batch_size, epochs)

    def _check_train_split_size(self, dataset: dict) -> None:
        """Before the model loads: at least two rows (one KL group); fewer
        rows than ``batch_size`` train as one partial batch, as in TRL."""
        rows = len(dataset.get("train") or [])
        if 0 < rows < KTO_MIN_BATCH:
            self._check_train_rows(rows, self._resolve_batch_size(), 0)

    def _check_train_rows(self, n_rows: int, batch_size: int, ties: int) -> None:
        """Two rows or more: below ``batch_size`` they form one partial batch
        and one KL group, as TRL trains them."""
        if n_rows < KTO_MIN_BATCH:
            raise ValueError(
                f"MLX {self.objective}: the training split has {n_rows} row(s); KTO needs "
                f"at least {KTO_MIN_BATCH}, because a row's KL completion is another "
                "row's. Add rows."
            )

    def _on_plan_cut(self, dropped: list, kept: int, accum: int) -> None:
        """Batches run in dataset order, so the cut always drops the same rows."""
        rows = sorted({i for batch in dropped for i in batch})
        first, last = kept + 1, kept + len(dropped)
        span = f"{first}" if first == last else f"{first}-{last}"
        console.print(
            f"[yellow]MLX {self.objective}: gradient_accumulation_steps={accum} keeps "
            f"{kept} of {kept + len(dropped)} batches; batch(es) {span} of the run "
            f"({len(rows)} row(s): {', '.join(map(str, rows))}) are not trained. "
            "Rows run in dataset order, so these are the same rows every run; "
            "TRL instead updates on the partial window at each epoch's end.[/]"
        )

    def _examples(self, rows: list, split: str = "train") -> tuple[list, int]:
        """Every row, tokenized with its KL row; nothing is skipped, so the KL
        groups are TRL's. A bad row stops the run before the model trains."""
        if not rows:
            if split == "train":
                raise ValueError(f"MLX {self.objective}: the train split has no rows")
            return [], 0
        max_length = int(getattr(self.config.data, "max_length", 2048) or 2048)
        try:
            examples = build_kto_examples(
                rows, self.tokenizer, max_length=max_length,
                group_size=self._resolve_batch_size(),
            )
        except PreferenceRowError as exc:
            raise PreferenceRowError(f"{self.objective.lower()} {split} {exc}") from exc
        return examples, 0

    @staticmethod
    def _to_arrays(examples: list) -> tuple[Any, Any]:
        # DPO's reference precompute reads this: the matched rows' reference
        # log-probs come back as its first half, the KL rows' as its second.
        import mlx.core as mx

        tokens, mask = collate_kto(examples)
        return mx.array(tokens), mx.array(mask)

    def _batch_arrays(self, examples: list) -> tuple:
        import mlx.core as mx

        return (*self._to_arrays(examples), mx.array([bool(e.label) for e in examples]))

    def _loss(self, model: Any, arrays: tuple, split: str, rows: Any) -> tuple:
        tokens, mask, labels = arrays
        ref, ref_kl = (r[rows] for r in self._ref[split])
        policy, policy_kl = kto_logps(model, tokens, mask, ref.shape[0],
                                      head_split=self._head_split)
        return kto_loss(policy, policy_kl, ref, ref_kl, labels,
                        float(self.config.training.kto_beta))

    def _format_metrics(self, metrics: dict) -> str:
        means = class_means(metrics)
        parts = [f"kl {metrics['kl']:.4f}"]
        for side in ("chosen", "rejected"):
            value = means.get(f"rewards_{side}")
            parts.append(f"rewards/{side} " + ("n/a" if value is None else f"{value:.4f}"))
        return ", ".join(parts)

    def _objective_metadata(self) -> dict:
        return {
            "task": "kto",
            "kto_beta": float(self.config.training.kto_beta),
            "kto_loss_type": KTO_LOSS_TYPE,
            "desirable_weight": DESIRABLE_WEIGHT,
            "undesirable_weight": UNDESIRABLE_WEIGHT,
            "reference": "precomputed from the base model",
            "kl_rows": "completions rotated by one within each kl_group_size rows",
            "kl_group_size": self._resolve_batch_size(),
            "batch_order": "dataset order (TRL's sequential sampler)",
        }

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        result = super().train(display=display, tracker=tracker, run_id=run_id,
                               resume_from_checkpoint=resume_from_checkpoint)
        if "val_chosen_share" in result:
            result.update(class_means(result, prefix="val_"))
        return result

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """Warm start only from a KTO adapter of the same base.

        The checkpoint's directory must hold the ``adapter_config.json`` an MLX
        KTO run writes (before its first step, so step checkpoints have it
        too). An adapter of another task, another base, or no metadata at all
        is refused before any tensor loads; MLX SFT's loader then checks the
        tensor names and shapes. The reference was already computed from the
        base model, before this runs.
        """
        import json
        from pathlib import Path

        name = self.objective
        meta_path = Path(checkpoint_path).resolve().parent / "adapter_config.json"
        if not meta_path.is_file():
            raise ValueError(
                f"MLX {name}: no adapter_config.json beside {checkpoint_path}; cannot "
                "confirm it is a task='kto' adapter of this base model"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"MLX {name}: unreadable {meta_path}: {exc}") from exc
        if meta.get("task") != "kto":
            raise ValueError(
                f"MLX {name}: {checkpoint_path} was trained with task="
                f"{meta.get('task')!r}, not 'kto'; start a fresh run or resume with "
                "the matching task"
            )
        if str(meta.get("model")) != str(self.config.base):
            raise ValueError(
                f"MLX {name}: {checkpoint_path} belongs to base {meta.get('model')!r}, "
                f"not {self.config.base!r}"
            )
        super()._load_checkpoint_weights(checkpoint_path)
