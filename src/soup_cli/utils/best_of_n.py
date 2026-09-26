"""Best-of-N rejection sampling: sample N, a judge picks the winner (v0.71.31).

Sampling either loads a local ``transformers`` model or calls an injected raw-
completion provider (torch-lazy, inside ``sample_candidates``); judging reuses
the project's ``JudgeEvaluator`` *pointwise* (score each candidate, argmax).
The provider / judge / build path is PURE (NO top-level torch) so it is CPU-
unit-testable.

Output rows are SFT chat rows ``{"messages": [...], "_best_of_n": {...}}`` with
provenance under the reserved ``_best_of_n`` key; ``build_dpo_pair`` optionally
emits a winner-vs-loser preference pair.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from soup_cli.eval.judge import JudgeScore


class PointwiseJudge(Protocol):
    """Anything that can score one response (has a ``.weighted_score``)."""

    def evaluate(self, prompt: str, response: str) -> "JudgeScore": ...


class BestOfNRuntimeError(RuntimeError):
    """A sampler or judge backend failed and the checkpoint can be resumed."""


@dataclass(frozen=True)
class BestOfNPick:
    """The judged winner among N candidates + the per-candidate scores."""

    winner_idx: int
    winner: str
    scores: "tuple[float, ...]"


def sample_candidates(
    model: "Optional[PreTrainedModel]",
    tokenizer: "Optional[PreTrainedTokenizerBase]",
    prompt: str,
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    device: Optional[str] = None,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> "list[str]":
    """Sample ``n`` continuations locally or through a raw-completion provider."""
    if generate_fn is not None:
        try:
            return [generate_fn(prompt).strip() for _ in range(n)]
        except Exception as exc:
            raise BestOfNRuntimeError("provider sampler failed") from exc

    if model is None or tokenizer is None:
        raise ValueError("local best-of-N sampling requires a model and tokenizer")

    import torch

    try:
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True so an attention_mask is passed to generate() — without it
        # transformers warns of unreliable output when pad_token == eos_token.
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        if device:
            enc = enc.to(device)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=pad_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out
        ]
    except Exception as exc:
        raise BestOfNRuntimeError("local sampler failed") from exc


def judge_pick_best(
    prompt: str, candidates: "list[str]", evaluator: "PointwiseJudge"
) -> BestOfNPick:
    """Score each candidate pointwise; argmax wins (ties -> lowest index)."""
    if not candidates:
        raise ValueError("no candidates to judge")
    scores = []
    for index, candidate in enumerate(candidates):
        try:
            raw_score = evaluator.evaluate(prompt, candidate).weighted_score
        except Exception as exc:
            raise BestOfNRuntimeError("judge backend failed") from exc
        if isinstance(raw_score, bool):
            raise ValueError(f"candidate {index} judge score must be a finite number")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate {index} judge score must be a finite number"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"candidate {index} judge score must be finite")
        scores.append(score)
    winner_idx = max(range(len(scores)), key=lambda i: scores[i])
    return BestOfNPick(
        winner_idx=winner_idx, winner=candidates[winner_idx], scores=tuple(scores)
    )


def build_sft_row(prompt: str, pick: BestOfNPick, *, judge_model: str) -> dict:
    """A chat SFT row (prompt -> winner) with best-of-N provenance."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": pick.winner},
        ],
        "_best_of_n": {
            "n": len(pick.scores),
            "winner_idx": pick.winner_idx,
            "judge_model": judge_model,
            "scores": list(pick.scores),
        },
    }


def build_dpo_pair(
    prompt: str, pick: BestOfNPick, candidates: "list[str]"
) -> Optional[dict]:
    """Winner vs lowest-scored candidate as a DPO pair; None if they coincide."""
    loser_idx = min(range(len(pick.scores)), key=lambda i: pick.scores[i])
    if loser_idx == pick.winner_idx:
        return None
    return {"prompt": prompt, "chosen": pick.winner, "rejected": candidates[loser_idx]}
