"""Quant-Lobotomy Checker (v0.26.0 Part D).

Orchestrates a before/after eval comparison: run the same suite against two
model backends (safetensors / gguf / awq / gptq / mlx) and surface the delta.
Focused on the 'did quantization eat accuracy?' question.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

from soup_cli.utils.paths import is_under_cwd as _is_under_cwd

if TYPE_CHECKING:
    from rich.table import Table


def classify_delta(delta: float, *, minor: float = 0.02, major: float = 0.05) -> str:
    """Return one of OK / MINOR / MAJOR based on absolute drop in score."""
    if delta >= 0:
        return "OK"
    drop = -delta
    if drop < minor:
        return "OK"
    if drop < major:
        return "MINOR"
    return "MAJOR"


@dataclass(frozen=True)
class QuantCheckRow:
    task: str
    before: float
    after: float
    delta: float
    verdict: str


@dataclass(frozen=True)
class QuantCheckResult:
    rows: list[QuantCheckRow]

    def to_json(self) -> str:
        return json.dumps(
            {"rows": [asdict(r) for r in self.rows]},
            indent=2,
        )


def _score_tasks(tasks_file: str, generate_fn: Callable[[str], str]) -> float:
    """Load a JSONL task file and return the average score."""
    from soup_cli.eval.custom import load_eval_tasks, score_task

    tasks = load_eval_tasks(tasks_file)
    if not tasks:
        return 0.0
    total = 0.0
    for task in tasks:
        output = generate_fn(task.prompt)
        total += float(score_task(task, output).score)
    return total / len(tasks)


def run_quant_check(
    *,
    before_gen: Callable[[str], str],
    after_gen: Callable[[str], str],
    tasks_file: str,
    task_name: str = "default",
) -> QuantCheckResult:
    """Run ``tasks_file`` through both models and return the row."""
    before = _score_tasks(tasks_file, before_gen)
    after = _score_tasks(tasks_file, after_gen)
    delta = after - before
    verdict = classify_delta(delta)
    return QuantCheckResult(rows=[QuantCheckRow(
        task=task_name, before=before, after=after,
        delta=delta, verdict=verdict,
    )])


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def render_table(result: QuantCheckResult) -> "Table":
    """Render a QuantCheckResult as a Rich Table (for the CLI)."""
    from rich.table import Table

    table = Table(title="Quant check")
    table.add_column("Task", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict")
    for row in result.rows:
        verdict = row.verdict
        colored = {
            "OK": f"[green]{verdict}[/]",
            "MINOR": f"[yellow]{verdict}[/]",
            "MAJOR": f"[red]{verdict}[/]",
        }.get(verdict, verdict)
        table.add_row(
            row.task,
            f"{row.before:.3f}",
            f"{row.after:.3f}",
            f"{row.delta:+.3f}",
            colored,
        )
    return table


def render_markdown(result: QuantCheckResult) -> str:
    lines = [
        "| Task | Before | After | Delta | Verdict |",
        "|------|--------|-------|-------|---------|",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.task} | {row.before:.3f} | {row.after:.3f} | "
            f"{row.delta:+.3f} | {row.verdict} |"
        )
    return "\n".join(lines)


FORMAT_RENDERERS: dict[str, Callable[[QuantCheckResult], Union["Table", str]]] = {
    "table": render_table,
    "json": lambda r: r.to_json(),
    "markdown": render_markdown,
}


def render(
    result: QuantCheckResult, *, fmt: Literal["table", "json", "markdown"],
) -> Union["Table", str]:
    renderer = FORMAT_RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"unknown format '{fmt}'. Use table | json | markdown")
    return renderer(result)


def ensure_format(fmt: str) -> None:
    """Validate format string, raise ValueError on unknown."""
    if fmt not in FORMAT_RENDERERS:
        raise ValueError(f"unknown format '{fmt}'. Use table | json | markdown")


def make_model_generator(
    model_path: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return a ``generate_fn(prompt) -> str`` backed by a transformers model.

    Lazy-loaded so the CLI stays cold-start fast. The model is loaded once
    and reused across calls. ``temperature=0`` enables greedy decoding for
    reproducible eval scores.
    """
    if max_new_tokens < 1 or max_new_tokens > 16384:
        raise ValueError("max_new_tokens must be in [1, 16384]")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=False
    )
    model.eval()

    def _generate(prompt: str) -> str:
        if not prompt:
            return ""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        do_sample = temperature > 0.0
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
        # Strip the prompt prefix from the decoded text.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


def stub_generator(label: str) -> Callable[[str], str]:
    """Return a deterministic stub generator so the CLI has something runnable.

    Real model loading is wired post-v0.26.0 — once then, pass in real
    ``before_gen`` / ``after_gen`` callables instead of this stub.
    """
    def _stub(prompt: str) -> str:  # noqa: ARG001
        return f"[stub:{label}]"
    return _stub


# ---------------------------------------------------------------------------
# Path containment (realpath + commonpath)
# ---------------------------------------------------------------------------


def is_under_cwd(path: Path) -> bool:
    """Backwards-compat alias — use :func:`soup_cli.utils.paths.is_under_cwd`."""
    return _is_under_cwd(path)


def resolve_model_ref(
    ref: str, *, kinds: Optional[tuple[str, ...]] = None,
) -> Optional[str]:
    """Resolve ``registry://<id>`` to an artifact path, or pass through a file.

    When ``kinds`` is provided, only artifacts whose ``kind`` field matches
    are returned. This prevents picking the wrong artifact when an entry
    has multiple attachments (e.g. adapter + GGUF + safetensors).
    """
    if ref.startswith("registry://"):
        from soup_cli.registry.store import RegistryStore

        with RegistryStore() as store:
            eid = store.resolve(ref)
            if eid is None:
                return None
            arts = store.get_artifacts(eid)
        if not arts:
            return None
        if kinds:
            filtered = [a for a in arts if a.get("kind") in kinds]
            if not filtered:
                return None
            return filtered[0]["path"]
        return arts[0]["path"]
    return ref
