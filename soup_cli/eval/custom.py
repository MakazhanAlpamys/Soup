"""Custom eval task runner — load JSONL eval sets, score model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAX_EVAL_TASKS = 10_000
MAX_REGEX_PATTERN_LEN = 1_000
MAX_REGEX_INPUT_LEN = 50_000

VALID_SCORING = {"exact", "contains", "regex", "semantic"}


@dataclass
class EvalTask:
    """A single evaluation task loaded from JSONL."""

    prompt: str
    expected: str = ""
    category: str = "default"
    scoring: str = "exact"
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task: EvalTask
    output: str
    score: float
    matched: bool


@dataclass
class EvalResults:
    """Aggregated results from a full eval run."""

    results: list[EvalResult]
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    category_scores: dict[str, dict] = field(default_factory=dict)

    def compute(self) -> None:
        """Compute aggregate scores from individual results."""
        self.total = len(self.results)
        self.correct = sum(1 for r in self.results if r.matched)
        self.accuracy = self.correct / self.total if self.total > 0 else 0.0

        # Per-category breakdown
        cats: dict[str, list[EvalResult]] = {}
        for result in self.results:
            cat = result.task.category
            cats.setdefault(cat, []).append(result)

        self.category_scores = {}
        for cat, cat_results in sorted(cats.items()):
            cat_correct = sum(1 for r in cat_results if r.matched)
            cat_total = len(cat_results)
            self.category_scores[cat] = {
                "total": cat_total,
                "correct": cat_correct,
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
            }


def load_eval_tasks(path: "Path | str") -> list[EvalTask]:
    """Load eval tasks from a JSONL file with schema validation.

    Each line must be a JSON object with at least a 'prompt' field.
    Optional: 'expected', 'category', 'scoring'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval tasks file not found: {path}")

    if not path.suffix.lower() == ".jsonl":
        raise ValueError(f"Expected .jsonl file, got: {path.suffix}")

    tasks: list[EvalTask] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            if len(tasks) >= MAX_EVAL_TASKS:
                raise ValueError(
                    f"Eval file exceeds maximum of {MAX_EVAL_TASKS} tasks "
                    f"(stopped at line {line_num})"
                )

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_num}: expected JSON object, got {type(row).__name__}"
                )

            if "prompt" not in row:
                raise ValueError(
                    f"Line {line_num}: missing required field 'prompt'"
                )

            scoring = row.get("scoring", "exact")
            if scoring not in VALID_SCORING:
                raise ValueError(
                    f"Line {line_num}: invalid scoring '{scoring}', "
                    f"must be one of: {', '.join(sorted(VALID_SCORING))}"
                )

            tasks.append(EvalTask(
                prompt=str(row["prompt"]),
                expected=str(row.get("expected", "")),
                category=str(row.get("category", "default")),
                scoring=scoring,
                metadata={
                    k: v for k, v in row.items()
                    if k not in ("prompt", "expected", "category", "scoring")
                },
            ))

    if not tasks:
        raise ValueError(f"No eval tasks found in {path}")

    return tasks


def score_exact(output: str, expected: str) -> bool:
    """Exact string match (case-insensitive, stripped)."""
    return output.strip().lower() == expected.strip().lower()


def score_contains(output: str, expected: str) -> bool:
    """Check if expected string is contained in output (case-insensitive)."""
    return expected.strip().lower() in output.strip().lower()


def score_regex(output: str, expected: str) -> bool:
    """Check if output matches expected regex pattern.

    Guards against ReDoS: caps pattern and input length.
    """
    if len(expected) > MAX_REGEX_PATTERN_LEN:
        return False
    truncated_output = output[:MAX_REGEX_INPUT_LEN]
    try:
        return bool(re.search(expected, truncated_output, re.IGNORECASE))
    except re.error:
        return False


def score_semantic(output: str, expected: str) -> float:
    """Compute similarity via Jaccard token overlap (bag-of-words).

    This is a lexical metric, not embedding-based semantic similarity.
    Returns a float 0-1. Considered a match if similarity >= 0.5.
    """
    out_tokens = set(output.strip().lower().split())
    exp_tokens = set(expected.strip().lower().split())
    if not out_tokens or not exp_tokens:
        return 0.0
    intersection = out_tokens & exp_tokens
    union = out_tokens | exp_tokens
    return len(intersection) / len(union)


SCORING_FUNCTIONS = {
    "exact": score_exact,
    "contains": score_contains,
    "regex": score_regex,
}


def score_task(task: EvalTask, output: str) -> EvalResult:
    """Score a single task output against the expected answer."""
    if task.scoring == "semantic":
        similarity = score_semantic(output, task.expected)
        matched = similarity >= 0.5
        return EvalResult(
            task=task, output=output, score=similarity, matched=matched,
        )

    scorer = SCORING_FUNCTIONS.get(task.scoring, score_exact)
    matched = scorer(output, task.expected)
    return EvalResult(
        task=task, output=output, score=1.0 if matched else 0.0, matched=matched,
    )


def run_eval(
    model_path: str,
    tasks: list[EvalTask],
    generate_fn: Optional[object] = None,
) -> EvalResults:
    """Run evaluation on a list of tasks using a model.

    Args:
        model_path: Path to model directory.
        tasks: List of EvalTask objects.
        generate_fn: Optional callable(prompt: str) -> str. If None, uses
            a default pipeline from transformers.

    Returns:
        EvalResults with per-task scores and aggregates.
    """
    if generate_fn is None:
        generate_fn = _create_default_generator(model_path)

    results: list[EvalResult] = []
    for task in tasks:
        output = generate_fn(task.prompt)
        result = score_task(task, output)
        results.append(result)

    eval_results = EvalResults(results=results)
    eval_results.compute()
    return eval_results


def _create_default_generator(model_path: str):
    """Create a default text generation function using transformers pipeline."""
    from rich.console import Console
    from rich.panel import Panel
    from transformers import pipeline

    console = Console()
    console.print(Panel(
        f"[yellow]Loading model from: {model_path}[/]\n"
        "This will execute model code from the specified path.",
        title="Model Loading",
        border_style="yellow",
    ))

    pipe = pipeline(
        "text-generation",
        model=model_path,
        max_new_tokens=512,
        do_sample=False,
    )

    def generate(prompt: str) -> str:
        out = pipe(prompt, return_full_text=False)
        return out[0]["generated_text"] if out else ""

    return generate
