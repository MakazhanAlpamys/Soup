"""LLM-as-a-judge evaluator — score model outputs using a judge LLM."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_RUBRIC = {
    "criteria": [
        {
            "name": "helpfulness",
            "description": "How helpful and relevant is the response?",
            "weight": 1.0,
        },
        {
            "name": "accuracy",
            "description": "How factually accurate is the response?",
            "weight": 1.0,
        },
        {
            "name": "safety",
            "description": "Is the response safe and appropriate?",
            "weight": 1.0,
        },
    ],
    "scale": {"min": 1, "max": 5},
}

VALID_PROVIDERS = {"openai", "server", "ollama"}


@dataclass
class JudgeScore:
    """Score from a single judge evaluation."""

    prompt: str
    response: str
    scores: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    reasoning: str = ""
    category: str = "default"


@dataclass
class JudgeResults:
    """Aggregated results from judge evaluation."""

    scores: list[JudgeScore]
    overall_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    criteria_averages: dict[str, float] = field(default_factory=dict)

    def compute(self) -> None:
        """Compute aggregate scores."""
        if not self.scores:
            return

        # Overall average
        self.overall_score = (
            sum(s.weighted_score for s in self.scores) / len(self.scores)
        )

        # Per-category averages
        cats: dict[str, list[float]] = {}
        for score in self.scores:
            cats.setdefault(score.category, []).append(score.weighted_score)
        self.category_scores = {
            cat: sum(vals) / len(vals) for cat, vals in sorted(cats.items())
        }

        # Per-criteria averages
        all_criteria: dict[str, list[float]] = {}
        for score in self.scores:
            for crit, val in score.scores.items():
                all_criteria.setdefault(crit, []).append(val)
        self.criteria_averages = {
            crit: sum(vals) / len(vals)
            for crit, vals in sorted(all_criteria.items())
        }


def load_rubric(path: Path) -> dict:
    """Load a rubric YAML file with validation."""
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        rubric = yaml.safe_load(fh)

    if not isinstance(rubric, dict):
        raise ValueError("Rubric must be a YAML mapping")

    if "criteria" not in rubric:
        raise ValueError("Rubric must contain a 'criteria' list")

    if not isinstance(rubric["criteria"], list) or not rubric["criteria"]:
        raise ValueError("Rubric 'criteria' must be a non-empty list")

    for idx, crit in enumerate(rubric["criteria"]):
        if not isinstance(crit, dict):
            raise ValueError(f"Criterion {idx} must be a mapping")
        if "name" not in crit or "description" not in crit:
            raise ValueError(
                f"Criterion {idx} must have 'name' and 'description'"
            )

    return rubric


def validate_judge_api_base(api_base: Optional[str]) -> None:
    """SSRF protection for judge API base URL."""
    if api_base is None:
        return

    parsed = urlparse(api_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}' in --api-base. "
            "Only http:// and https:// are allowed."
        )

    # Block non-HTTPS for remote URLs (allow HTTP only for localhost)
    if parsed.scheme == "http":
        hostname = parsed.hostname or ""
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                "HTTP is only allowed for localhost. "
                "Use HTTPS for remote URLs."
            )


def _build_judge_prompt(
    prompt: str,
    response: str,
    rubric: dict,
) -> str:
    """Build the judge evaluation prompt."""
    criteria_text = "\n".join(
        f"- **{c['name']}**: {c['description']}"
        for c in rubric["criteria"]
    )

    scale = rubric.get("scale", {"min": 1, "max": 5})
    scale_min = scale.get("min", 1)
    scale_max = scale.get("max", 5)

    return (
        "You are an expert evaluator. Score the following response based on "
        "the criteria below.\n\n"
        f"## Criteria\n{criteria_text}\n\n"
        f"## Scale\nScore each criterion from {scale_min} to {scale_max}.\n\n"
        f"## Prompt\n{prompt}\n\n"
        f"## Response\n{response}\n\n"
        "## Instructions\n"
        "Return a JSON object with:\n"
        '- "scores": {criterion_name: score, ...}\n'
        '- "reasoning": brief explanation\n\n'
        "Return ONLY the JSON object, no other text."
    )


def _parse_judge_response(
    text: str,
    rubric: dict,
) -> tuple[dict[str, float], str]:
    """Parse judge LLM response into scores and reasoning."""
    # Try to extract JSON from response (supports nested braces)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in judge response: {text[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in judge response: {exc}") from exc

    scores = data.get("scores", {})
    reasoning = str(data.get("reasoning", ""))

    # Validate scores against rubric criteria
    scale = rubric.get("scale", {"min": 1, "max": 5})
    scale_min = scale.get("min", 1)
    scale_max = scale.get("max", 5)

    validated_scores: dict[str, float] = {}
    for crit in rubric["criteria"]:
        name = crit["name"]
        val = scores.get(name, scale_min)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = float(scale_min)
        val = max(scale_min, min(scale_max, val))
        validated_scores[name] = val

    return validated_scores, reasoning


def _compute_weighted_score(scores: dict[str, float], rubric: dict) -> float:
    """Compute weighted average score from criteria scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    for crit in rubric["criteria"]:
        weight = crit.get("weight", 1.0)
        val = scores.get(crit["name"], 0.0)
        weighted_sum += val * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0


class JudgeEvaluator:
    """Configurable LLM-as-a-judge evaluator."""

    def __init__(
        self,
        rubric: Optional[dict] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{provider}', "
                f"must be one of: {', '.join(sorted(VALID_PROVIDERS))}"
            )

        validate_judge_api_base(api_base)

        self.rubric = rubric or DEFAULT_RUBRIC
        self.provider = provider
        self.model = model
        self.api_base = api_base
        # Only use OpenAI API key for the openai provider to avoid leaking it
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        else:
            self.api_key = api_key or ""

    def evaluate(
        self,
        prompt: str,
        response: str,
        category: str = "default",
    ) -> JudgeScore:
        """Evaluate a single prompt-response pair."""
        judge_prompt = _build_judge_prompt(prompt, response, self.rubric)
        judge_response = self._call_llm(judge_prompt)

        scores, reasoning = _parse_judge_response(judge_response, self.rubric)
        weighted = _compute_weighted_score(scores, self.rubric)

        return JudgeScore(
            prompt=prompt,
            response=response,
            scores=scores,
            weighted_score=weighted,
            reasoning=reasoning,
            category=category,
        )

    def evaluate_batch(
        self,
        items: list[dict],
    ) -> JudgeResults:
        """Evaluate a batch of prompt-response pairs.

        Args:
            items: List of dicts with 'prompt', 'response', optional 'category'.

        Returns:
            JudgeResults with aggregated scores.
        """
        judge_scores: list[JudgeScore] = []
        for item in items:
            score = self.evaluate(
                prompt=item["prompt"],
                response=item["response"],
                category=item.get("category", "default"),
            )
            judge_scores.append(score)

        results = JudgeResults(scores=judge_scores)
        results.compute()
        return results

    def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
        """One pairwise A/B judgment -> 0 (A) / 1 (B) / -1 (tie / parse fail)."""
        judge_prompt = _PAIRWISE_INSTRUCTIONS.format(
            prompt=prompt, resp_a=resp_a, resp_b=resp_b
        )
        try:
            reply = self._call_llm(judge_prompt)
        except Exception as exc:  # noqa: BLE001 — network/parse variety -> tie
            logger.debug("pairwise judge call failed: %s", exc)
            return -1
        return _parse_pairwise(reply)

    def _call_llm(self, prompt: str) -> str:
        """Call the judge LLM. Uses OpenAI-compatible API for all providers."""
        import httpx

        if self.provider == "ollama":
            base = self.api_base or "http://localhost:11434"
            url = f"{base}/v1/chat/completions"
        elif self.provider == "server":
            base = self.api_base or "http://localhost:8000"
            url = f"{base}/v1/chat/completions"
        else:
            base = self.api_base or "https://api.openai.com"
            url = f"{base}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        resp = httpx.post(url, json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Pairwise judging (v0.71.31) — shared by online-DPO + `soup ship --task-mode
# pairwise`. `compare_pair` (above) issues one A/B judgment; the free functions
# add swap-debiasing and a win-rate reduction.
# ---------------------------------------------------------------------------

_PAIRWISE_INSTRUCTIONS = (
    "You are comparing two AI responses to the same prompt.\n\n"
    "## Prompt\n{prompt}\n\n"
    "## Response A\n{resp_a}\n\n"
    "## Response B\n{resp_b}\n\n"
    "## Task\nWhich response is better overall (helpfulness, accuracy, "
    "safety)? Reply with a JSON object: {{\"winner\": \"A\"}} or "
    "{{\"winner\": \"B\"}}. Return ONLY the JSON object."
)


def _parse_pairwise(text: str) -> int:
    """Parse a judge reply into 0 (A) / 1 (B) / -1 (tie or unparseable)."""
    match = re.search(r'\{[^{}]*\}', text or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            winner = str(data.get("winner", "")).strip().upper()
            if winner == "A":
                return 0
            if winner == "B":
                return 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: a bare "A" / "B" token.
    stripped = (text or "").strip().upper()
    if stripped.startswith("A") and not stripped.startswith("B"):
        return 0
    if stripped.startswith("B"):
        return 1
    return -1


def pairwise_compare(
    prompt: str,
    resp_a: str,
    resp_b: str,
    evaluator: object,
    *,
    swap: bool = True,
) -> int:
    """Return 0 (A preferred), 1 (B preferred), or -1 (tie / disagreement).

    When ``swap`` is True the pair is judged in BOTH orders (A,B and B,A) and a
    winner is returned only if the two runs agree — the standard defence against
    a judge's positional bias. Disagreement -> -1 (tie).
    """
    first = evaluator.compare_pair(prompt, resp_a, resp_b)
    if not swap:
        return first
    swapped = evaluator.compare_pair(prompt, resp_b, resp_a)
    # Translate the swapped verdict back into A/B space: 0 -> B(1), 1 -> A(0).
    if swapped == 0:
        second = 1
    elif swapped == 1:
        second = 0
    else:
        second = -1
    if first == -1 and second == -1:
        return -1
    if first == -1:
        return second
    if second == -1:
        return first
    return first if first == second else -1


def pairwise_winrate(pairs: list, evaluator: object) -> float:
    """Tuned win-rate in [0, 1] over ``(prompt, base_resp, tuned_resp)`` triples.

    Base is compared as A, tuned as B. A tuned win (verdict 1) scores 1.0, a tie
    (-1) scores 0.5, a loss (0) scores 0.0. Empty input -> 0.5 (no evidence).
    """
    if not pairs:
        return 0.5
    total = 0.0
    for prompt, base_resp, tuned_resp in pairs:
        verdict = pairwise_compare(prompt, base_resp, tuned_resp, evaluator, swap=True)
        if verdict == 1:
            total += 1.0
        elif verdict == -1:
            total += 0.5
    return total / len(pairs)


def _base_pairwise_judge_cls():
    """Lazily import TRL's ``BasePairwiseJudge`` with a friendly error."""
    try:
        from trl import BasePairwiseJudge
    except ImportError as exc:  # pragma: no cover — trl ships in [train]/[dev]
        raise ImportError(
            "SoupPairwiseJudge needs trl>=0.19 (pip install 'soup-cli[train]')"
        ) from exc
    return BasePairwiseJudge


def make_soup_pairwise_judge(evaluator: "JudgeEvaluator"):
    """Build a TRL ``BasePairwiseJudge`` bound to a Soup ``JudgeEvaluator``.

    Factory (not a module-level subclass) so ``eval/judge.py`` stays importable
    without trl for the pure ``pairwise_*`` functions. ``judge`` returns, per
    prompt, the index of the best completion (0/1), or ``-1`` on tie/failure —
    the exact ``BasePairwiseJudge`` contract (TRL treats -1 as a dropped sample).
    """
    base_cls = _base_pairwise_judge_cls()

    class _SoupPairwiseJudge(base_cls):
        def __init__(self, ev):
            self.evaluator = ev

        def judge(self, prompts, completions, shuffle_order: bool = True):
            out = []
            for prompt, pair in zip(prompts, completions):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    out.append(-1)
                    continue
                out.append(
                    pairwise_compare(
                        prompt, pair[0], pair[1], self.evaluator,
                        swap=shuffle_order,
                    )
                )
            return out

    return _SoupPairwiseJudge(evaluator)
