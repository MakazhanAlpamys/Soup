"""v0.43.0 Part B — Model Arena: A/B tournament with Elo ratings.

Pure-math Elo kernel + tournament aggregator. Generation + judging is the
caller's responsibility — feed `record_match_result` with model A / model B
identifiers and a winner string, get back updated ratings.

Mirrors the existing `eval/human.py` Elo policy (K=32, base 1500) but
operates on a closed model registry so a single `Tournament` can host many
models and produce a leaderboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

DEFAULT_K = 32.0
DEFAULT_BASE_RATING = 1500.0
_MAX_NAME_LEN = 128
_MAX_MODELS = 256
_MAX_MATCHES = 1_000_000


def _validate_model_name(name: object) -> str:
    if not isinstance(name, str):
        raise ValueError("model name must be a string")
    if not name:
        raise ValueError("model name must not be empty")
    if "\x00" in name:
        raise ValueError("model name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"model name length {len(name)} exceeds max {_MAX_NAME_LEN}"
        )
    # Reject Rich markup metacharacters at the source so any downstream
    # CLI consumer that embeds the leaderboard `model` field in markup
    # cannot be markup-injected (security review fix).
    if "[" in name or "]" in name:
        raise ValueError(
            "model name must not contain Rich markup metacharacters '[' or ']'"
        )
    return name


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given Elo ratings."""
    for r in (rating_a, rating_b):
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            raise ValueError("ratings must be int/float")
        if not math.isfinite(float(r)):
            raise ValueError("ratings must be finite")
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    *,
    score_a: float,
    k: float = DEFAULT_K,
) -> tuple[float, float]:
    """Update Elo ratings after a match where A scored `score_a` ∈ [0,1].

    score_a == 1.0  →  A won.
    score_a == 0.0  →  B won.
    score_a == 0.5  →  draw.
    """
    if isinstance(score_a, bool) or not isinstance(score_a, (int, float)):
        raise ValueError("score_a must be int/float")
    if not math.isfinite(float(score_a)) or score_a < 0.0 or score_a > 1.0:
        raise ValueError("score_a must be in [0, 1]")
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise ValueError("k must be a positive number")
    if not math.isfinite(float(k)) or k <= 0:
        raise ValueError("k must be a positive finite number")

    expected_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


@dataclass
class Tournament:
    """Live tournament tracker with Elo + win/loss bookkeeping."""

    base_rating: float = DEFAULT_BASE_RATING
    k: float = DEFAULT_K
    _ratings: dict[str, float] = field(default_factory=dict)
    _wins: dict[str, int] = field(default_factory=dict)
    _losses: dict[str, int] = field(default_factory=dict)
    _draws: dict[str, int] = field(default_factory=dict)
    _matches: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_rating, bool)
            or not isinstance(self.base_rating, (int, float))
            or not math.isfinite(float(self.base_rating))
        ):
            raise ValueError("base_rating must be a finite number")
        if isinstance(self.k, bool) or not isinstance(self.k, (int, float)) or self.k <= 0:
            raise ValueError("k must be positive")

    def register(self, name: str) -> None:
        """Register a model with the base rating."""
        canonical = _validate_model_name(name)
        if canonical in self._ratings:
            return
        if len(self._ratings) >= _MAX_MODELS:
            raise ValueError(
                f"tournament has reached the model cap ({_MAX_MODELS})"
            )
        self._ratings[canonical] = float(self.base_rating)
        self._wins[canonical] = 0
        self._losses[canonical] = 0
        self._draws[canonical] = 0

    def record(
        self,
        model_a: str,
        model_b: str,
        *,
        winner: str,
    ) -> tuple[float, float]:
        """Record a match. `winner` ∈ {"a", "b", "draw"}.

        Returns the new (rating_a, rating_b).
        """
        if self._matches >= _MAX_MATCHES:
            raise ValueError(
                f"tournament has reached the match cap ({_MAX_MATCHES})"
            )
        a = _validate_model_name(model_a)
        b = _validate_model_name(model_b)
        if a == b:
            raise ValueError("model_a and model_b must differ")
        if not isinstance(winner, str):
            raise ValueError("winner must be a string")
        winner_norm = winner.lower()
        if winner_norm not in {"a", "b", "draw"}:
            raise ValueError("winner must be one of 'a' / 'b' / 'draw'")
        self.register(a)
        self.register(b)
        score_a = {"a": 1.0, "b": 0.0, "draw": 0.5}[winner_norm]
        new_a, new_b = update_elo(
            self._ratings[a], self._ratings[b], score_a=score_a, k=self.k
        )
        self._ratings[a] = new_a
        self._ratings[b] = new_b
        if winner_norm == "a":
            self._wins[a] += 1
            self._losses[b] += 1
        elif winner_norm == "b":
            self._wins[b] += 1
            self._losses[a] += 1
        else:
            self._draws[a] += 1
            self._draws[b] += 1
        self._matches += 1
        return new_a, new_b

    @property
    def ratings(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._ratings))

    def leaderboard(self) -> list[dict]:
        """Sorted leaderboard, highest rating first."""
        rows = []
        for name in self._ratings:
            rows.append(
                {
                    "model": name,
                    "rating": round(self._ratings[name], 2),
                    "wins": self._wins[name],
                    "losses": self._losses[name],
                    "draws": self._draws[name],
                }
            )
        rows.sort(key=lambda r: r["rating"], reverse=True)
        return rows
