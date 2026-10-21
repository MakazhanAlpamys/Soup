"""MLX ORPO trainer — reference-free odds-ratio preference optimization on Apple Silicon.

The objective is TRL 0.29.1's ``experimental/orpo/orpo_trainer.py``, the trainer
Soup's transformers backend runs for ``task: orpo``:

- ``a``, ``b``: the **mean** log-probability of the chosen / rejected completion
  tokens (``get_batch_logps(..., average_log_prob=True)``);
- ``log_odds = (a - b) - (log1mexp(a) - log1mexp(b))``, with TRL's stable
  ``log1mexp`` (``log1p(-exp(x))`` below ``-ln 2``, ``log(-expm1(x))`` above);
- loss = chosen NLL - ``orpo_beta * mean(logsigmoid(log_odds))``, where the NLL
  is a token-mean cross-entropy over the **whole** chosen sequence, prompt
  included (every position whose target is a real token), across the batch.

No reference model exists and nothing is precomputed. The loop, LoRA attach,
optimizer plan, clipping, seeding, chunked head and adapter writer are MLX DPO's
(:class:`~soup_cli.trainer.mlx_dpo.MLXDPOTrainerWrapper`); this module adds the
tokenized example, the two masks, the loss and its metrics.

Tokenization follows TRL's ORPO ``tokenize_row`` rather than DPO's: rows are
rendered to text first (``apply_chat_template`` with TRL's common-prefix rule),
the answer is located with ``build_tokenized_answer`` (which tolerates a merge
of the last prompt token into the answer), BOS is prepended and EOS appended
when missing. Over-long rows are then cut by Soup's cap (prompt to
``max_length // 2``, ``keep_end`` as ``ORPOConfig`` defaults, completions to the
remaining room), the same cap ``enforce_preference_sequence_limit`` applies to
the transformers path. TRL's own in-tokenizer answer cut, which can empty the
shorter answer, is not reproduced; on rows that fit, the two backends see the
same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import takewhile
from typing import Any

from rich.console import Console

from soup_cli.trainer._trl_compat import _truncate_tokens
from soup_cli.trainer.mlx_dpo import (
    DEFAULT_LOGIT_CHUNK,
    MLXDPOTrainerWrapper,
    PreferenceRowError,
    PrefixUnstableError,
    _as_ids,
    _normalise_row,
)

console = Console()

#: ``ORPOConfig.truncation_mode``'s default in TRL 0.29.1.
ORPO_TRUNCATION_MODE = "keep_end"
_LN2 = 0.6931471805599453


@dataclass(frozen=True)
class OrpoExample:
    """One tokenized pair: two full sequences and where each answer starts.

    The prompt part of the chosen and rejected sequences can differ by one
    token, when the tokenizer merges the last prompt token into the answer;
    TRL tolerates exactly that, and so does this.
    """

    chosen_ids: list
    chosen_prompt_len: int
    rejected_ids: list
    rejected_prompt_len: int


# --------------------------------------------------------------------------
# Tokenization (TRL 0.29.1 ORPOTrainer.tokenize_row / build_tokenized_answer)
# --------------------------------------------------------------------------


def _common_prefix(a: str, b: str) -> str:
    return "".join(x for x, _ in takewhile(lambda p: p[0] == p[1], zip(a, b)))


def render_preference_text(row: dict, tokenizer: Any) -> tuple[str, str, str]:
    """``(prompt, chosen, rejected)`` text, as ``trl.data_utils.apply_chat_template``.

    Text rows pass through. Conversational rows render the prompt with a
    generation prompt (or ``continue_final_message`` after an assistant turn),
    then each ``prompt + answer``; the prompt is narrowed to its common prefix
    with the chosen rendering, the chosen answer taken after it, then narrowed
    again against the rejected rendering -- the same order TRL uses.
    """
    prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
    if not isinstance(prompt, list):
        return prompt, chosen, rejected
    last_role = prompt[-1]["role"]
    if last_role in ("user", "tool"):
        kwargs = {"add_generation_prompt": True, "continue_final_message": False}
    elif last_role == "assistant":
        kwargs = {"add_generation_prompt": False, "continue_final_message": True}
    else:
        raise PreferenceRowError(f"the prompt ends with an unsupported role {last_role!r}")
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, **kwargs)
    chosen_full = tokenizer.apply_chat_template(prompt + chosen, tokenize=False)
    prompt_text = _common_prefix(prompt_text, chosen_full)
    chosen_text = chosen_full[len(prompt_text):]
    rejected_full = tokenizer.apply_chat_template(prompt + rejected, tokenize=False)
    prompt_text = _common_prefix(prompt_text, rejected_full)
    rejected_text = rejected_full[len(prompt_text):]
    return prompt_text, chosen_text, rejected_text


def _encode(tokenizer: Any, text: str) -> list:
    return _as_ids(tokenizer.encode(text, add_special_tokens=False))


def _tokenized_answer(tokenizer: Any, prompt: str, answer: str) -> tuple[list, list]:
    """``(prompt_ids, answer_ids)`` for ``prompt + answer``: TRL's
    ``build_tokenized_answer``. The answer starts one token earlier when the
    tokenizer merged the last prompt token into it."""
    full = _encode(tokenizer, prompt + answer)
    prompt_ids = _encode(tokenizer, prompt)
    start = len(prompt_ids)
    if prompt_ids != full[:start]:
        start -= 1
    return full[:start], full[start:]


def build_orpo_example(
    row: dict,
    tokenizer: Any,
    *,
    max_length: int,
    max_prompt_length: int,
    truncation_mode: str = ORPO_TRUNCATION_MODE,
) -> OrpoExample:
    """Tokenize one preference row the way TRL 0.29.1's ORPO trainer does.

    A tie (``chosen == rejected``) is kept: ORPO's odds-ratio term is a finite
    constant there (``log_odds = 0``) and the NLL still trains, which is what
    TRL does. A pair whose two prompt renderings differ by more than one token
    raises :class:`PrefixUnstableError` (TRL raises a ``ValueError`` there) so
    the caller can skip it with a count.
    """
    prompt, chosen, rejected = render_preference_text(_normalise_row(row), tokenizer)
    chosen_prompt, chosen_answer = _tokenized_answer(tokenizer, prompt, chosen)
    rejected_prompt, rejected_answer = _tokenized_answer(tokenizer, prompt, rejected)
    diff_tokens = sum(a != b for a, b in zip(chosen_prompt, rejected_prompt))
    if diff_tokens > 1 or abs(len(chosen_prompt) - len(rejected_prompt)) > 1:
        raise PrefixUnstableError(
            "the chosen and rejected renderings tokenize the prompt differently "
            "beyond the last token, so the answers cannot be located"
        )

    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    sides = []
    pairs = ((chosen_prompt, chosen_answer), (rejected_prompt, rejected_answer))
    for prompt_ids, answer_ids in pairs:
        prompt_ids, answer_ids = list(prompt_ids), list(answer_ids)
        if bos is not None and (not prompt_ids or prompt_ids[0] != bos):
            prompt_ids = [bos] + prompt_ids
        if eos is not None and (not answer_ids or answer_ids[-1] != eos):
            answer_ids = answer_ids + [eos]
        # Soup's cap (enforce_preference_sequence_limit's ORPO branch).
        prompt_ids = _truncate_tokens(prompt_ids, max_prompt_length, truncation_mode)
        answer_ids = answer_ids[: max(0, max_length - len(prompt_ids))]
        if not answer_ids:
            raise PreferenceRowError(
                f"no answer token fits in data.max_length ({max_length}) after the "
                f"{len(prompt_ids)}-token prompt"
            )
        sides.append((prompt_ids, answer_ids))
    (cp, ca), (rp, ra) = sides
    return OrpoExample(
        chosen_ids=cp + ca, chosen_prompt_len=len(cp),
        rejected_ids=rp + ra, rejected_prompt_len=len(rp),
    )


def collate_orpo(examples: list) -> tuple[list, list, list]:
    """``(tokens, completion_mask, sequence_mask)``: the chosen rows, then the
    rejected rows, right-padded with 0.

    ``completion_mask[i][t]`` marks a completion token at ``t`` (the preference
    score); ``sequence_mask[i][t]`` marks every real token (the NLL). Both are
    over token positions; :func:`token_logps` scores position ``t >= 1``.
    """
    seqs = [(e.chosen_ids, e.chosen_prompt_len) for e in examples]
    seqs += [(e.rejected_ids, e.rejected_prompt_len) for e in examples]
    width = max(len(ids) for ids, _ in seqs)
    tokens, completion, sequence = [], [], []
    for ids, prompt_len in seqs:
        pad = width - len(ids)
        tokens.append(list(ids) + [0] * pad)
        completion.append([0] * prompt_len + [1] * (len(ids) - prompt_len) + [0] * pad)
        sequence.append([1] * len(ids) + [0] * pad)
    return tokens, completion, sequence


# --------------------------------------------------------------------------
# Scoring and loss
# --------------------------------------------------------------------------


def token_logps(
    model: Any,
    tokens: Any,
    *,
    head_split: Any = None,
    chunk_size: int = DEFAULT_LOGIT_CHUNK,
) -> Any:
    """Log-probability of ``tokens[:, t]`` given the tokens before it, for
    ``t = 1 .. T-1``: shape ``(B, T-1)``.

    ORPO's NLL covers the prompt, so unlike DPO every position is scored. With
    ``head_split`` the backbone runs once and the vocab head runs in
    checkpointed chunks, so a batch's full-vocabulary logits are never all held.
    """
    import mlx.core as mx

    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def score(logits: Any, tgt: Any) -> Any:
        logits = logits.astype(mx.float32)
        picked = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
        return picked - mx.logsumexp(logits, axis=-1)

    if head_split is None:
        return score(model(inputs), targets)
    backbone, head = head_split
    hidden = backbone(inputs)
    width = int(hidden.shape[1])
    step = max(1, int(chunk_size))
    parts = []
    for lo in range(0, width, step):
        hi = min(width, lo + step)
        chunk = mx.checkpoint(lambda h, t=targets[:, lo:hi]: score(head(h), t))
        parts.append(chunk(hidden[:, lo:hi]))
    return mx.concatenate(parts, axis=1)


def log1mexp(x: Any) -> Any:
    """``log(1 - exp(x))`` for ``x <= 0``, TRL's branch at ``-ln 2``.

    ``mx.where`` evaluates both branches, so each one gets its argument
    clamped to its own side of ``-ln 2``: the unused branch then stays finite
    and leaks no NaN into the value or the gradient. At ``x = 0`` the result is
    ``-inf``, as in TRL; the trainer stops on the nonfinite loss.
    """
    import mlx.core as mx

    low = mx.minimum(x, -_LN2)
    high = mx.maximum(x, -_LN2)
    return mx.where(x < -_LN2, mx.log1p(-mx.exp(low)), mx.log(-mx.expm1(high)))


def orpo_loss(
    chosen_mean_logps: Any,
    rejected_mean_logps: Any,
    chosen_nll: Any,
    beta: float,
) -> tuple[Any, dict]:
    """TRL's ORPO loss and its metrics from per-pair mean completion log-probs.

    Returns ``(loss, metrics)``: ``nll_loss``, ``log_odds_ratio``
    (``mean(logsigmoid(log_odds))``), ``log_odds_chosen`` (``mean(log_odds)``),
    ``reward_accuracy`` and ``reward_margin`` (rewards are ``beta * logp``, as
    TRL logs them). The trainer stops on a nonfinite loss instead of training on it.
    """
    import mlx.core as mx
    import mlx.nn as nn

    a = chosen_mean_logps.astype(mx.float32)
    b = rejected_mean_logps.astype(mx.float32)
    log_odds = (a - b) - (log1mexp(a) - log1mexp(b))
    ratio = nn.log_sigmoid(log_odds)
    loss = chosen_nll - beta * ratio.mean()
    metrics = {
        "nll_loss": chosen_nll,
        "log_odds_ratio": ratio.mean(),
        "log_odds_chosen": log_odds.mean(),
        "reward_accuracy": (a > b).astype(mx.float32).mean(),
        "reward_margin": (beta * (a - b)).mean(),
    }
    return loss, metrics


def orpo_terms(model: Any, tokens: Any, completion: Any, sequence: Any, *,
               head_split: Any = None) -> tuple[Any, Any, Any]:
    """``(chosen mean logps, rejected mean logps, chosen NLL)`` for a collated batch."""
    import mlx.core as mx

    logps = token_logps(model, tokens, head_split=head_split)
    comp = completion[:, 1:].astype(mx.float32)
    seq = sequence[:, 1:].astype(mx.float32)
    mean_logps = (logps * comp).sum(axis=-1) / comp.sum(axis=-1)
    half = tokens.shape[0] // 2
    chosen_seq = seq[:half]
    nll = -(logps[:half] * chosen_seq).sum() / chosen_seq.sum()
    return mean_logps[:half], mean_logps[half:], nll


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class MLXORPOTrainerWrapper(MLXDPOTrainerWrapper):
    """ORPO on MLX: MLX DPO's loop with ORPO's tokens, masks and loss."""

    objective = "ORPO"
    metric_names = (
        "nll_loss", "log_odds_ratio", "log_odds_chosen", "reward_accuracy", "reward_margin",
    )

    def _build_example(self, row: dict, max_length: int) -> OrpoExample:
        return build_orpo_example(
            row, self.tokenizer, max_length=max_length, max_prompt_length=max_length // 2,
        )

    def _batch_arrays(self, examples: list) -> tuple:
        import mlx.core as mx

        tokens, completion, sequence = collate_orpo(examples)
        return mx.array(tokens), mx.array(completion), mx.array(sequence)

    def _verification_arrays(self, examples: list) -> tuple:
        # The head-split check compares completion scores on the DPO-shaped
        # batch; any token positions will do, so reuse the chosen sequences.
        import mlx.core as mx

        tokens, completion, _ = collate_orpo(examples)
        return mx.array(tokens), mx.array(completion)

    def _prepare_objective(self, train_examples: list, val_examples: list,
                           batch_size: int) -> None:
        """Reference-free: nothing to precompute."""

    fail_on_nonfinite = True

    def _loss(self, model: Any, arrays: tuple, split: str, rows: Any) -> tuple:
        tokens, completion, sequence = arrays
        a, b, nll = orpo_terms(model, tokens, completion, sequence, head_split=self._head_split)
        return orpo_loss(a, b, nll, float(self.config.training.orpo_beta))

    def _format_metrics(self, metrics: dict) -> str:
        return (
            f"nll {metrics['nll_loss']:.4f}, log-odds ratio "
            f"{metrics['log_odds_ratio']:.4f}, reward acc {metrics['reward_accuracy']:.2f}"
        )

    def _objective_metadata(self) -> dict:
        return {
            "task": "orpo",
            "orpo_beta": float(self.config.training.orpo_beta),
            "orpo_loss": "trl-0.29.1 odds ratio + chosen NLL",
            "reference": "none (reference-free)",
            "truncation_mode": ORPO_TRUNCATION_MODE,
        }

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """Warm start only from an ORPO adapter of the same base.

        The checkpoint's directory must hold the ``adapter_config.json`` an MLX
        ORPO run writes (it is written before the first step, so step
        checkpoints of a run that stopped early have it too). A DPO or SFT
        adapter, a different base model, or no metadata at all is refused
        before any tensor loads; the tensor names and shapes are then checked
        by MLX SFT's loader.
        """
        import json
        from pathlib import Path

        meta_path = Path(checkpoint_path).resolve().parent / "adapter_config.json"
        if not meta_path.is_file():
            raise ValueError(
                f"MLX ORPO: no adapter_config.json beside {checkpoint_path}; cannot "
                "confirm it is an ORPO adapter of this base model"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"MLX ORPO: unreadable {meta_path}: {exc}") from exc
        if meta.get("task") != "orpo":
            raise ValueError(
                f"MLX ORPO: {checkpoint_path} was trained with task="
                f"{meta.get('task')!r}, not 'orpo'; start a fresh run or resume "
                "with the matching task"
            )
        if str(meta.get("model")) != str(self.config.base):
            raise ValueError(
                f"MLX ORPO: {checkpoint_path} belongs to base {meta.get('model')!r}, "
                f"not {self.config.base!r}"
            )
        super()._load_checkpoint_weights(checkpoint_path)
