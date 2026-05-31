"""Human evaluation — terminal-based A/B comparison with Elo ratings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ELO_K = 32
ELO_DEFAULT = 1500


@dataclass
class HumanJudgment:
    """A single human judgment in an A/B comparison."""

    prompt: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: str  # "a", "b", or "tie"


@dataclass
class EloRating:
    """Elo rating for a model."""

    model: str
    rating: float = ELO_DEFAULT
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total: int = 0


@dataclass
class HumanEvalResults:
    """Aggregated results from human evaluation."""

    judgments: list[HumanJudgment] = field(default_factory=list)
    ratings: dict[str, EloRating] = field(default_factory=dict)

    def compute_ratings(self) -> None:
        """Compute Elo ratings from all judgments."""
        self.ratings = {}
        for judgment in self.judgments:
            self._ensure_model(judgment.model_a)
            self._ensure_model(judgment.model_b)

            rating_a = self.ratings[judgment.model_a]
            rating_b = self.ratings[judgment.model_b]

            # Expected scores
            exp_a = _expected_score(rating_a.rating, rating_b.rating)
            exp_b = _expected_score(rating_b.rating, rating_a.rating)

            # Actual scores
            if judgment.winner == "a":
                actual_a, actual_b = 1.0, 0.0
                rating_a.wins += 1
                rating_b.losses += 1
            elif judgment.winner == "b":
                actual_a, actual_b = 0.0, 1.0
                rating_a.losses += 1
                rating_b.wins += 1
            else:
                actual_a, actual_b = 0.5, 0.5
                rating_a.ties += 1
                rating_b.ties += 1

            rating_a.rating += ELO_K * (actual_a - exp_a)
            rating_b.rating += ELO_K * (actual_b - exp_b)
            rating_a.total += 1
            rating_b.total += 1

    def _ensure_model(self, model: str) -> None:
        if model not in self.ratings:
            self.ratings[model] = EloRating(model=model)

    def to_dict(self) -> dict:
        """Serialize results for storage."""
        return {
            "judgments": [
                {
                    "prompt": j.prompt,
                    "response_a": j.response_a,
                    "response_b": j.response_b,
                    "model_a": j.model_a,
                    "model_b": j.model_b,
                    "winner": j.winner,
                }
                for j in self.judgments
            ],
            "ratings": {
                name: {
                    "rating": r.rating,
                    "wins": r.wins,
                    "losses": r.losses,
                    "ties": r.ties,
                    "total": r.total,
                }
                for name, r in sorted(
                    self.ratings.items(),
                    key=lambda item: item[1].rating,
                    reverse=True,
                )
            },
        }


def _expected_score(rating_self: float, rating_opponent: float) -> float:
    """Elo expected score calculation."""
    return 1.0 / (1.0 + math.pow(10, (rating_opponent - rating_self) / 400.0))


MAX_PROMPTS = 10_000


def load_prompts(path: Path) -> list[dict]:
    """Load evaluation prompts from JSONL.

    Each line: {"prompt": "...", "category": "..."}
    Capped at MAX_PROMPTS to prevent unbounded memory usage.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    prompts: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            if len(prompts) >= MAX_PROMPTS:
                raise ValueError(
                    f"Prompts file exceeds maximum of {MAX_PROMPTS} entries "
                    f"(stopped at line {line_num})"
                )

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {exc}"
                ) from exc
            if "prompt" not in row:
                raise ValueError(
                    f"Line {line_num}: missing required field 'prompt'"
                )
            prompts.append(row)
    return prompts


def save_results(results: HumanEvalResults, path: Path) -> None:
    """Save human eval results to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results.to_dict(), fh, indent=2)


def load_results(path: Path) -> HumanEvalResults:
    """Load human eval results from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    eval_results = HumanEvalResults()
    for jdata in data.get("judgments", []):
        eval_results.judgments.append(HumanJudgment(
            prompt=jdata["prompt"],
            response_a=jdata["response_a"],
            response_b=jdata["response_b"],
            model_a=jdata["model_a"],
            model_b=jdata["model_b"],
            winner=jdata["winner"],
        ))

    eval_results.compute_ratings()
    return eval_results


def run_human_eval_session(
    prompts: list[dict],
    model_a_name: str,
    model_b_name: str,
    generate_a: Optional[object] = None,
    generate_b: Optional[object] = None,
    responses_a: Optional[list[str]] = None,
    responses_b: Optional[list[str]] = None,
) -> HumanEvalResults:
    """Run an interactive human evaluation session.

    Either provide generate functions or pre-computed responses.
    When running non-interactively (testing), pass responses directly.
    """
    results = HumanEvalResults()

    for idx, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]

        if responses_a and responses_b:
            resp_a = responses_a[idx] if idx < len(responses_a) else ""
            resp_b = responses_b[idx] if idx < len(responses_b) else ""
        else:
            resp_a = generate_a(prompt) if generate_a else ""
            resp_b = generate_b(prompt) if generate_b else ""

        results.judgments.append(HumanJudgment(
            prompt=prompt,
            response_a=resp_a,
            response_b=resp_b,
            model_a=model_a_name,
            model_b=model_b_name,
            winner="tie",  # Default; interactive UI sets this
        ))

    results.compute_ratings()
    return results
