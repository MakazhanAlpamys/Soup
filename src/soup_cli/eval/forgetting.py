"""Catastrophic forgetting detection (Part G of v0.25.0).

Runs lightweight mini benchmarks against a model during training and flags
significant drops in general-knowledge accuracy from the pre-training baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

MiniBenchmark = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Built-in mini benchmarks (kept intentionally small — expand to 100 in prod)
# ---------------------------------------------------------------------------

MINI_MMLU: MiniBenchmark = [
    {"question": "What is 2 + 2? (A) 3 (B) 4 (C) 5", "answer": "B"},
    {"question": "The capital of France is: (A) London (B) Berlin (C) Paris", "answer": "C"},
    {"question": "Water freezes at: (A) 0C (B) 50C (C) 100C", "answer": "A"},
    {"question": "Photosynthesis uses: (A) oxygen (B) carbon dioxide (C) nitrogen", "answer": "B"},
    {"question": "Atomic number of hydrogen is: (A) 1 (B) 2 (C) 3", "answer": "A"},
]

MINI_COMMON_SENSE: MiniBenchmark = [
    {
        "question": "If it is raining, you should bring a: (A) hat (B) umbrella (C) fan",
        "answer": "B",
    },
    {"question": "You eat breakfast in the: (A) morning (B) evening (C) night", "answer": "A"},
    {"question": "Fish live in: (A) trees (B) water (C) sand", "answer": "B"},
    {"question": "The sun rises in the: (A) west (B) south (C) east", "answer": "C"},
    {"question": "Ice melts when: (A) heated (B) frozen (C) pressed", "answer": "A"},
]

MINI_INSTRUCTION: MiniBenchmark = [
    {"question": "Respond with just the word 'ok'.", "answer": "ok"},
    {"question": "Answer in one word: color of grass?", "answer": "green"},
    {"question": "Answer yes or no: Is fire hot?", "answer": "yes"},
    {"question": "Reply with the number three.", "answer": "3"},
    {"question": "Say only: done", "answer": "done"},
]

MINI_BENCHMARKS: dict[str, MiniBenchmark] = {
    "mini_mmlu": MINI_MMLU,
    "mini_common_sense": MINI_COMMON_SENSE,
    "mini_instruction": MINI_INSTRUCTION,
}


@dataclass
class ForgettingResult:
    """Outcome of a single forgetting eval."""

    step: int
    accuracy: float
    baseline: float
    delta: float
    warning_level: Literal["green", "yellow", "red"]


class ForgettingDetector:
    """Run a mini benchmark periodically and report accuracy drops."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        benchmark: str = "mini_mmlu",
        threshold: float = 0.10,
    ) -> None:
        if benchmark not in MINI_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{benchmark}'. "
                f"Options: {', '.join(MINI_BENCHMARKS.keys())}"
            )
        self.generate_fn = generate_fn
        self.benchmark_name = benchmark
        self.benchmark = MINI_BENCHMARKS[benchmark]
        self.threshold = threshold
        self._baseline_accuracy: Optional[float] = None

    def _evaluate(self) -> float:
        correct = 0
        for item in self.benchmark:
            output = self.generate_fn(item["question"])
            if not isinstance(output, str):
                continue
            if item["answer"].strip().lower() in output.strip().lower():
                correct += 1
        return correct / len(self.benchmark) if self.benchmark else 0.0

    def run_baseline(self) -> float:
        """Compute the baseline accuracy before training starts."""
        self._baseline_accuracy = self._evaluate()
        return self._baseline_accuracy

    def _build_result(self, step: int, accuracy: float) -> ForgettingResult:
        baseline = self._baseline_accuracy if self._baseline_accuracy is not None else accuracy
        delta = baseline - accuracy
        if delta <= self.threshold:
            level: Literal["green", "yellow", "red"] = "green"
        elif delta <= self.threshold * 2:
            level = "yellow"
        else:
            level = "red"
        return ForgettingResult(
            step=step,
            accuracy=accuracy,
            baseline=baseline,
            delta=delta,
            warning_level=level,
        )

    def check_forgetting(self, step: int) -> ForgettingResult:
        """Run the mini benchmark and compare against the baseline."""
        if self._baseline_accuracy is None:
            self.run_baseline()
        current = self._evaluate()
        return self._build_result(step=step, accuracy=current)
