"""Eval-Gated Training (v0.26.0 Part B).

Declarative ``evals/gate.yaml`` suites define per-task thresholds; the gate
runs them at epoch boundaries during training (or post-hoc via
``soup eval gate``) and surfaces pass / fail / regression verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from soup_cli.utils.paths import is_under_cwd


@dataclass(frozen=True)
class GateTaskResult:
    name: str
    score: Optional[float]
    threshold: float
    baseline: Optional[float]
    delta: Optional[float]
    passed: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    regression: bool
    task_results: list[GateTaskResult]


class GateTask(BaseModel):
    """One gate task — e.g. a custom eval + threshold."""

    type: Literal["judge", "custom", "benchmark"] = Field(
        description="Task type: judge | custom | benchmark",
    )
    name: str = Field(description="Task name (used as baseline key)")
    threshold: float = Field(
        ge=0.0, le=1000.0,
        description="Minimum score to pass (scale depends on scorer)",
    )
    # type=custom
    tasks: Optional[str] = Field(
        default=None, description="JSONL file of custom eval tasks",
    )
    scorer: Optional[Literal["exact", "contains", "regex", "semantic"]] = Field(
        default=None, description="Scorer for type=custom",
    )
    # type=judge
    prompts: Optional[str] = Field(
        default=None, description="JSONL prompts for LLM judge",
    )
    judge_model: Optional[str] = Field(
        default=None, description="Judge model URL (e.g. ollama://llama3.1)",
    )
    # type=benchmark
    benchmark: Optional[str] = Field(
        default=None, description="Registered benchmark id (e.g. mini_mmlu)",
    )

    @field_validator("tasks", "prompts")
    @classmethod
    def _clean_path_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("path field contains null byte")
        return value

    @field_validator("judge_model")
    @classmethod
    def _valid_judge_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # Allowlist of schemes — SSRF hardening consistent with the project.
        allowed = ("ollama://", "https://", "http://localhost", "http://127.0.0.1")
        if not value.startswith(allowed):
            raise ValueError(
                f"judge_model URL '{value}' uses disallowed scheme - "
                "use ollama://, https://, or http://localhost"
            )
        return value


class EvalSuite(BaseModel):
    """Parsed ``evals/gate.yaml``."""

    suite: str = Field(description="Suite name for display / logs")
    tasks: list[GateTask] = Field(default_factory=list)


def load_suite(path: str) -> EvalSuite:
    """Load and validate an eval suite from disk."""
    suite_path = Path(path)
    if not is_under_cwd(suite_path):
        raise ValueError(
            f"eval-gate suite '{path}' is outside cwd - refusing to load"
        )
    if not suite_path.exists():
        raise FileNotFoundError(f"eval-gate suite not found: {path}")
    data = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    return EvalSuite(**data)


def resolve_baseline(spec: Optional[str]) -> dict[str, float]:
    """Resolve a baseline specifier to a ``{task_name: score}`` map.

    - ``None`` / ``""``: return empty map
    - ``registry://<id>``: look up eval_results for that entry via the
      experiment tracker, keyed by benchmark
    - filesystem path: JSON mapping of ``{name: score}``
    """
    if not spec:
        return {}

    if spec.startswith("registry://"):
        from soup_cli.registry.store import RegistryStore

        ref = spec[len("registry://"):]
        with RegistryStore() as store:
            # ``resolve`` strips the scheme itself; pass the raw ref so the
            # error path below reports what the user typed.
            entry_id = store.resolve(ref)
            if entry_id is None:
                raise ValueError(
                    f"registry baseline not found: {ref} (use `soup registry list`)"
                )
            rows = store.get_eval_results(entry_id)
        return {
            row.get("benchmark", ""): float(row.get("score", 0.0))
            for row in rows
            if row.get("benchmark") and row.get("score") is not None
        }

    # Filesystem path
    baseline_path = Path(spec)
    if not is_under_cwd(baseline_path):
        raise ValueError(
            f"baseline file '{spec}' is outside cwd - refusing to load"
        )
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline file not found: {spec}")
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in baseline file: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"baseline file must be a JSON object mapping name -> score; "
            f"got {type(data).__name__}"
        )
    return {str(k): float(v) for k, v in data.items()}


def _parse_judge_url(judge_model: str) -> tuple[str, str, Optional[str]]:
    """Split a ``judge_model`` URL into ``(provider, model, api_base)``.

    Examples:
      ``ollama://llama3.1`` -> ("ollama", "llama3.1", None)
      ``http://localhost:8000/Qwen2.5`` -> ("server", "Qwen2.5", "http://localhost:8000")
      ``https://api.openai.com/gpt-4o-mini`` -> ("openai", "gpt-4o-mini", "https://api.openai.com")
    """
    if judge_model.startswith("ollama://"):
        return ("ollama", judge_model[len("ollama://"):], None)
    # http(s):// — last path segment is the model id; the rest is api_base.
    # Defence-in-depth: GateTask._valid_judge_url already restricts the
    # scheme to ollama:// / https:// / http://localhost / http://127.0.0.1.
    # We match prefixes in the same order here. We do NOT include a bare
    # ``http://`` catch-all — if validation is ever bypassed and a non-loopback
    # http URL reaches us, the trailing ``raise ValueError`` will fire.
    for prefix, default_provider in (
        ("http://localhost", "server"),
        ("http://127.0.0.1", "server"),
        ("https://", "openai"),
    ):
        if judge_model.startswith(prefix):
            try:
                base, model = judge_model.rsplit("/", 1)
            except ValueError as exc:
                raise ValueError(
                    f"judge_model '{judge_model}' missing model id"
                ) from exc
            if not model:
                raise ValueError(f"judge_model '{judge_model}' missing model id")
            return (default_provider, model, base)
    raise ValueError(f"judge_model '{judge_model}' uses unsupported scheme")


def _run_judge_task(
    task: GateTask, generate_fn: Callable[[str], str],
) -> float:
    """Run a type=judge task. Generates a completion per prompt, then asks
    the judge model to score the (prompt, response) pair on a 1-10 scale.
    Aggregate score is normalised to [0, 1] (mean / 10).
    """
    if not task.prompts:
        raise ValueError(f"task '{task.name}' is type=judge but 'prompts' is missing")
    if not task.judge_model:
        raise ValueError(
            f"task '{task.name}' is type=judge but 'judge_model' is missing"
        )

    prompts_path = Path(task.prompts)
    if not is_under_cwd(prompts_path):
        raise ValueError(f"prompts file '{task.prompts}' is outside cwd")
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {task.prompts}")

    from soup_cli.eval.judge import JudgeEvaluator

    provider, model, api_base = _parse_judge_url(task.judge_model)
    evaluator = JudgeEvaluator(provider=provider, model=model, api_base=api_base)

    items: list[dict] = []
    with prompts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {task.prompts}: {exc}"
                ) from exc
            prompt = row.get("prompt", "")
            response = generate_fn(prompt)
            items.append({
                "prompt": prompt,
                "response": response,
                "category": row.get("category", "default"),
            })

    if not items:
        return 0.0

    results = evaluator.evaluate_batch(items)
    # results.overall_score is on a 1-10 scale; normalise to [0, 1].
    overall = float(getattr(results, "overall_score", 0.0))
    return max(0.0, min(1.0, overall / 10.0))


def _run_benchmark_task(
    task: GateTask, generate_fn: Callable[[str], str],
) -> float:
    """Run a type=benchmark task using the existing forgetting-mini-benchmark."""
    if not task.benchmark:
        raise ValueError(
            f"task '{task.name}' is type=benchmark but 'benchmark' is missing"
        )
    from soup_cli.eval import forgetting

    runner = getattr(forgetting, "run_mini_benchmark", None)
    if runner is None:
        raise RuntimeError(
            "mini-benchmark runner unavailable - "
            "install [eval] extras or update soup-cli"
        )
    score = runner(benchmark=task.benchmark, generate_fn=generate_fn)
    return max(0.0, min(1.0, float(score)))


def _run_custom_task(
    task: GateTask, generate_fn: Callable[[str], str],
) -> float:
    """Run a type=custom task and return its aggregate score in [0, 1]."""
    from dataclasses import replace

    from soup_cli.eval.custom import load_eval_tasks, score_task

    if not task.tasks:
        raise ValueError(f"task '{task.name}' is type=custom but 'tasks' is missing")
    tasks = load_eval_tasks(task.tasks)
    if not tasks:
        return 0.0
    # Override scoring if the suite specified one (EvalTask.scoring field)
    if task.scorer is not None:
        tasks = [replace(t, scoring=task.scorer) for t in tasks]
    total = 0.0
    for eval_task in tasks:
        output = generate_fn(eval_task.prompt)
        result = score_task(eval_task, output)
        total += float(result.score)
    return total / len(tasks)


def run_gate(
    suite: EvalSuite,
    *,
    generate_fn: Callable[[str], str],
    baseline: Optional[dict[str, float]] = None,
    regression_threshold: float = 0.05,
) -> GateResult:
    """Run every task in ``suite`` and return an aggregate verdict.

    ``generate_fn`` accepts a prompt and returns the model output. Injecting
    this keeps the gate testable without a live model.
    """
    baseline = baseline or {}
    task_results: list[GateTaskResult] = []
    any_regressed = False
    any_failed_threshold = False

    for task in suite.tasks:
        score: Optional[float]
        error: Optional[str] = None
        try:
            if task.type == "custom":
                score = _run_custom_task(task, generate_fn)
            elif task.type == "judge":
                score = _run_judge_task(task, generate_fn)
            elif task.type == "benchmark":
                score = _run_benchmark_task(task, generate_fn)
            else:
                # Pydantic Literal already restricts task.type, so this is a
                # belt-and-braces fallthrough.
                raise ValueError(f"unknown task type: {task.type}")
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
            score = None
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — surface as score=None, never silent pass
            score = None
            error = f"{type(exc).__name__}: {exc}"

        if score is None:
            # Failed evaluation never silently passes the gate.
            passed_threshold = False
            base_score = baseline.get(task.name)
            delta = None
            regressed = False
        else:
            passed_threshold = score >= task.threshold
            base_score = baseline.get(task.name)
            delta = None
            regressed = False
            if base_score is not None:
                delta = score - base_score
                if delta < -abs(regression_threshold):
                    regressed = True

        if not passed_threshold:
            any_failed_threshold = True
        if regressed:
            any_regressed = True

        task_results.append(GateTaskResult(
            name=task.name,
            score=score,
            threshold=task.threshold,
            baseline=base_score,
            delta=delta,
            passed=passed_threshold and not regressed,
            error=error,
        ))

    return GateResult(
        passed=not (any_failed_threshold or any_regressed),
        regression=any_regressed,
        task_results=task_results,
    )
