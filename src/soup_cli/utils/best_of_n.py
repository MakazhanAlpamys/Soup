"""Best-of-N rejection sampling: sample N, a judge picks the winner (v0.71.31).

Sampling loads a local ``transformers`` model (torch-lazy, inside
``sample_candidates``); judging reuses the project's ``JudgeEvaluator``
*pointwise* (score each candidate, argmax). The judge / build half is PURE (NO
top-level torch) so it is CPU-unit-testable.

Output rows are SFT chat rows ``{"messages": [...], "_best_of_n": {...}}`` with
provenance under the reserved ``_best_of_n`` key; ``build_dpo_pair`` optionally
emits a winner-vs-loser preference pair.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BestOfNPick:
    """The judged winner among N candidates + the per-candidate scores."""

    winner_idx: int
    winner: str
    scores: tuple  # tuple[float, ...]


def sample_candidates(
    model,
    tokenizer,
    prompt: str,
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> list:
    """Sample ``n`` diverse continuations for ``prompt`` (do_sample)."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if device:
        inputs = inputs.to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=pad_id,
        )
    prompt_len = inputs.shape[1]
    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        for seq in out
    ]


def judge_pick_best(prompt: str, candidates: list, evaluator) -> BestOfNPick:
    """Score each candidate pointwise; argmax wins (ties -> lowest index)."""
    if not candidates:
        raise ValueError("no candidates to judge")
    scores = [float(evaluator.evaluate(prompt, c).weighted_score) for c in candidates]
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


def build_dpo_pair(prompt: str, pick: BestOfNPick, candidates: list) -> Optional[dict]:
    """Winner vs lowest-scored candidate as a DPO pair; None if they coincide."""
    loser_idx = min(range(len(pick.scores)), key=lambda i: pick.scores[i])
    if loser_idx == pick.winner_idx:
        return None
    return {"prompt": prompt, "chosen": pick.winner, "rejected": candidates[loser_idx]}
