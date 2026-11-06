"""``soup eval retention`` -- accumulated forgetting across a series of versions.

``soup ship`` judges a candidate against ONE base on a fixed general suite, so an
update that quietly erodes a task an earlier update taught looks fine every time.
This command keeps a ledger of control tasks (held-out task-eval JSONL files, the
format ``soup ship --task-eval`` reads) and the score every recorded version got on
each, and judges the candidate against the BEST score each task ever had.

Attached to the existing ``soup eval`` Typer app via :func:`register` (the same
pattern as the v0.55.0 / v0.65.0 / v0.71.10 eval modules). The ledger format and
the arithmetic live in ``utils/retention_ledger.py``; this file owns the live run
(one model in memory, reused for every task), the Rich output and the exit codes
(``0`` no regression, ``2`` regression, ``3`` usage or input error, ``1`` runtime
error). Heavy imports happen inside functions.
"""

from __future__ import annotations

import json
from typing import Dict, List, NoReturn, Optional

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
    GateCommand,
)
from soup_cli.utils.retention_ledger import DEFAULT_LEDGER_PATH, DEFAULT_THRESHOLD
from soup_cli.utils.terminal import for_terminal

#: Earlier versions shown as table columns; the JSON report always has all of them.
_MAX_TABLE_VERSIONS = 8


def _fail(console: Console, message: str, code: int) -> NoReturn:
    console.print(f"[red]Error:[/] {for_terminal(message)}")
    raise typer.Exit(code)


def _score_text(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _mean_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _signed_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _print_report(console: Console, report, *, recorded: bool, ledger_path: str) -> None:
    """The task x version table, the summary and the one-line verdict."""
    earlier = report.versions[:-1]
    shown = earlier[-_MAX_TABLE_VERSIONS:]
    table = Table(
        title=f"Retention: {for_terminal(report.label)} vs {len(earlier)} earlier version(s)",
        show_header=True,
    )
    table.add_column("Task")
    for version in shown:
        table.add_column(for_terminal(version), justify="right")
    table.add_column(f"{for_terminal(report.label)} (now)", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Drop", justify="right")
    table.add_column("Status")
    colour = {"new": "dim", "ok": "green", "regressed": "red"}
    for row in report.tasks:
        cells = [for_terminal(row.name)]
        cells.extend(_score_text(row.scores.get(version)) for version in shown)
        cells.append(_score_text(row.current))
        cells.append(_score_text(row.best_previous))
        cells.append("-" if row.drop is None else f"{row.drop:+.3f}")
        status = row.status.upper()
        cells.append(f"[{colour[row.status]}]{status}[/]")
        table.add_row(*cells)
    console.print(table)
    if len(earlier) > len(shown):
        console.print(
            f"[dim]{len(earlier) - len(shown)} earlier version(s) not shown; "
            "--output has the full matrix.[/]"
        )

    summary = report.summary()
    console.print(
        f"Tasks: {summary['tasks']} ({summary['comparable_tasks']} comparable, "
        f"{summary['new_tasks']} new) | Regressions: {summary['regressions']} "
        f"(threshold {report.threshold:.3f})"
    )
    console.print(
        f"Mean forgetting: {_mean_text(summary['mean_forgetting'])} | "
        f"Backward transfer: {_signed_text(summary['backward_transfer'])}"
    )
    if not report.comparable:
        console.print(
            "[dim]No earlier version scored these tasks; this version becomes their "
            "first point.[/]"
        )
    if report.regressions:
        named = ", ".join(
            f"{for_terminal(row.name)} ({row.drop:+.3f})" for row in report.regressions
        )
        console.print(f"[red]RETENTION REGRESSION:[/] {named}")
    else:
        console.print("[green]NO RETENTION REGRESSION[/]")
    if recorded:
        console.print(
            f"[green]Recorded[/] version {for_terminal(report.label)} in "
            f"{for_terminal(ledger_path)}"
        )
    else:
        console.print("[dim]Dry run: pass --record to append this version to the ledger.[/]")


def _load_control_tasks(ledger) -> Dict[str, list]:
    """Parse every registered control file with the loader ``soup ship --task-eval`` uses."""
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.utils.retention_ledger import LedgerError

    loaded: Dict[str, list] = {}
    for task in ledger.tasks:
        try:
            rows = load_eval_tasks(task.path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise LedgerError(
                f"control task {task.name!r} cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        loaded[task.name] = rows
    return loaded


def _build_generator(model: str, adapter: Optional[str], device: Optional[str]):
    """ONE generator over ``model`` (+ ``adapter``): the factory ``soup ship`` uses."""
    from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS
    from soup_cli.utils import live_eval

    # Same precision rule as `soup ship` without --config: bf16 on CUDA, fp32 elsewhere.
    resolved = live_eval.resolve_device(device)
    dtype = "bfloat16" if resolved.startswith("cuda") else "float32"
    return live_eval.make_generator(
        model,
        adapter=adapter,
        device=device,
        max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS,
        dtype=dtype,
    )


def _run(
    console: Console,
    *,
    ledger_path: str,
    model: str,
    adapter: Optional[str],
    label: str,
    add_task: Optional[List[str]],
    record: bool,
    threshold: float,
    output: Optional[str],
    device: Optional[str],
) -> int:
    from soup_cli.utils import retention_ledger as rl
    from soup_cli.utils.paths import (
        atomic_write_text,
        check_report_output_path,
        ensure_distinct_paths,
    )

    # 1. Flags: every usage error is found before a file is hashed or a model loaded.
    try:
        threshold = rl.validate_threshold(threshold)
        label = rl.validate_label(label)
        model = rl.validate_model_ref(model, "--model")
        if adapter is not None:
            adapter = rl.validate_model_ref(adapter, "--adapter")
        specs = [rl.parse_task_spec(spec) for spec in (add_task or [])]
        names = [name for name, _ in specs]
        duplicate = sorted({name for name in names if names.count(name) > 1})
        if duplicate:
            raise rl.LedgerError(f"--add-task names {duplicate[0]!r} twice")
        if output is not None:
            check_report_output_path(output, "--output")
            ensure_distinct_paths([("--ledger", ledger_path)], [("--output", output)])
    except (rl.LedgerError, ValueError, TypeError) as exc:
        _fail(console, str(exc), EXIT_USAGE_ERROR)

    # 2. The ledger (a missing one is only created when --add-task starts it).
    try:
        ledger = rl.load_ledger(ledger_path)
    except rl.LedgerNotFoundError as exc:
        if not specs:
            _fail(
                console,
                f"{exc}; start a ledger with --add-task NAME=PATH",
                EXIT_USAGE_ERROR,
            )
        console.print(f"[dim]No ledger at {for_terminal(ledger_path)}; starting a new one.[/]")
        ledger = rl.Ledger()
    except (rl.LedgerError, OSError) as exc:
        _fail(console, f"cannot read --ledger: {exc}", EXIT_USAGE_ERROR)
    if ledger.has_label(label):
        _fail(
            console,
            f"version label {for_terminal(repr(label))} is already recorded in the ledger; "
            "labels must be unique",
            EXIT_USAGE_ERROR,
        )

    # 3. Register new control tasks, then prove every control file is the one on record.
    now = rl.utc_timestamp()
    try:
        for name, path in specs:
            ledger, status = rl.register_task(ledger, name, path, label=label, now=now)
            console.print(f"[dim]Control task {for_terminal(name)}: {status}.[/]")
        if not ledger.tasks:
            raise rl.LedgerError(
                "the ledger has no control tasks; register one with --add-task NAME=PATH"
            )
        rl.verify_ledger_files(ledger)
        if output is not None:
            ensure_distinct_paths(
                [(f"control task {task.name}", task.path) for task in ledger.tasks],
                [("--output", output)],
            )
        control = _load_control_tasks(ledger)
    except (rl.LedgerError, ValueError, TypeError, OSError) as exc:
        _fail(console, str(exc), EXIT_USAGE_ERROR)

    # 4. ONE model in memory, scored on every task.
    from soup_cli.eval.custom import run_eval

    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    try:
        generate_fn = _build_generator(model, adapter, device)
        for task in ledger.tasks:
            result = run_eval(model, control[task.name], generate_fn=generate_fn)
            scores[task.name] = float(result.accuracy)
            counts[task.name] = int(result.total)
            console.print(
                f"[dim]  {for_terminal(task.name)}: {result.accuracy:.3f} "
                f"({result.correct}/{result.total})[/]"
            )
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        _fail(
            console,
            f"retention run failed: {type(exc).__name__}: {exc}",
            EXIT_RUNTIME_ERROR,
        )

    # 5. A control file edited mid-run would fake the series; refuse before reporting.
    try:
        rl.verify_ledger_files(ledger)
        report = rl.compute_retention(
            ledger, scores, label=label, threshold=threshold, examples=counts
        )
    except rl.LedgerError as exc:
        _fail(console, f"{exc} (nothing was recorded)", EXIT_USAGE_ERROR)

    # 6. Record, then write the report with the truth about whether it was recorded.
    recorded = False
    record_error: Optional[str] = None
    if record:
        try:
            version = rl.new_version(
                label=label, model=model, adapter=adapter, scores=scores, now=now
            )
            rl.save_ledger(rl.append_version(ledger, version), ledger_path)
            recorded = True
        except (rl.LedgerError, OSError) as exc:
            record_error = f"cannot record in --ledger: {type(exc).__name__}: {exc}"
    _print_report(console, report, recorded=recorded, ledger_path=ledger_path)
    if output is not None:
        payload = {**report.to_dict(), "recorded": recorded}
        try:
            atomic_write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", output, field="--output"
            )
            console.print(f"[green]Wrote report[/] {for_terminal(output)}")
        except (ValueError, OSError) as exc:
            _fail(
                console,
                f"cannot write --output: {type(exc).__name__}: {exc}",
                EXIT_RUNTIME_ERROR,
            )
    if record_error is not None:
        _fail(console, record_error, EXIT_RUNTIME_ERROR)
    return EXIT_GATE_FAILED if report.regressions else EXIT_OK


def register(app: typer.Typer, console: Console) -> None:
    """Attach ``soup eval retention`` to the eval Typer app."""

    @app.command(name="retention", cls=GateCommand)
    def retention_cmd(
        model: str = typer.Option(
            ..., "--model", "-m",
            help="Base or merged model (HF id or path) to evaluate.",
        ),
        label: str = typer.Option(
            ..., "--label",
            help="Version label for this candidate (unique in the ledger), e.g. v3.",
        ),
        ledger: str = typer.Option(
            DEFAULT_LEDGER_PATH, "--ledger",
            help="Retention ledger JSON (cwd-contained; created by --add-task + --record).",
        ),
        adapter: Optional[str] = typer.Option(
            None, "--adapter",
            help="LoRA adapter directory to evaluate on top of --model.",
        ),
        add_task: Optional[List[str]] = typer.Option(
            None, "--add-task",
            help=(
                "Register a control task as NAME=PATH (a task-eval JSONL, the format of "
                "soup ship --task-eval). Repeatable. Its sha256 is pinned: a later change "
                "to the file is refused."
            ),
        ),
        record: bool = typer.Option(
            False, "--record",
            help="Append this version's scores (and any new tasks) to the ledger.",
        ),
        threshold: float = typer.Option(
            DEFAULT_THRESHOLD, "--threshold",
            help="A drop below the best earlier score larger than this regresses (0.0-1.0).",
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Write the full task x version matrix and summary as JSON.",
        ),
        device: Optional[str] = typer.Option(
            None, "--device", help="Device for the live run (cuda / cpu).",
        ),
    ) -> None:
        """Measure accumulated forgetting across a series of versions (exit 0 / 2 / 3 / 1).

        Scores the candidate on EVERY control task in the ledger with one loaded model
        and compares each task with the best score any recorded version ever had.
        Exit 0 = no regression, 2 = a task dropped by more than --threshold,
        3 = usage or input error, 1 = runtime error.
        """
        code = _run(
            console,
            ledger_path=ledger,
            model=model,
            adapter=adapter,
            label=label,
            add_task=add_task,
            record=record,
            threshold=threshold,
            output=output,
            device=device,
        )
        raise typer.Exit(code)
