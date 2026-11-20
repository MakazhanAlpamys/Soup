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
error). The model is loaded the way ``soup ship`` loads it (``--config`` quantization,
``--chat-template-kwargs``), and that load is stamped on every recorded version.
Heavy imports happen inside functions.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Tuple

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

#: Characters of the first generation error quoted in the "generation failed" message.
_MAX_ERROR_CHARS = 200


def _error(console: Console, message: str) -> None:
    console.print(f"[red]Error:[/] {for_terminal(message)}")


def _fail(console: Console, message: str, code: int) -> NoReturn:
    _error(console, message)
    raise typer.Exit(code)


def _score_text(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _mean_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _signed_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _decimals_text(value: float, *, signed: bool = False) -> str:
    """``value`` with three to six decimals.

    The arithmetic rounds a drop to six places and the verdict compares THAT number with
    the threshold, so the table prints every digit that matters (``+0.333334``) and only
    pads a short one to three (``+0.150``).
    """
    text = f"{value:+.6f}" if signed else f"{value:.6f}"
    whole, _, fraction = text.partition(".")
    return f"{whole}.{fraction.rstrip('0').ljust(3, '0')}"


def _print_report(
    console: Console,
    report,
    *,
    recorded: bool,
    ledger_path: str,
    record_requested: bool = False,
) -> None:
    """The task x version table, the summary and the one-line verdict.

    ``record_requested`` is whether ``--record`` was passed: a run that asked for a record
    and failed to write it is not a dry run, and says so elsewhere.
    """
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
        cells.append("-" if row.drop is None else _decimals_text(row.drop, signed=True))
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
        f"(threshold {_decimals_text(report.threshold)})"
    )
    console.print(
        f"Mean forgetting: {_mean_text(summary['mean_forgetting'])} | "
        f"Backward transfer: {_signed_text(summary['backward_transfer'])}"
    )
    if not earlier:
        console.print(
            "[yellow]Warning:[/] the ledger has no earlier version, so every task is new and "
            "nothing can regress. If this is not your first run, the ledger was lost or "
            "reset; pass --require-history to refuse that."
        )
    elif not report.comparable:
        console.print(
            "[dim]No earlier version scored these tasks; this version becomes the baseline "
            "entry for each.[/]"
        )
    if report.regressions:
        named = ", ".join(
            f"{for_terminal(row.name)} ({_decimals_text(row.drop, signed=True)})"
            for row in report.regressions
        )
        console.print(f"[red]RETENTION REGRESSION:[/] {named}")
    else:
        console.print("[green]NO RETENTION REGRESSION[/]")
    if recorded:
        console.print(
            f"[green]Recorded[/] version {for_terminal(report.label)} in "
            f"{for_terminal(ledger_path)}"
        )
    elif not record_requested:
        console.print("[dim]Dry run: pass --record to append this version to the ledger.[/]")


def _load_control_tasks(ledger) -> Dict[str, list]:
    """Parse every registered control file from the very bytes its SHA-256 was checked on.

    The file is read once (no-follow, size-capped), hashed against the ledger and parsed
    from that same buffer with the loader ``soup ship --task-eval`` uses, so a swap
    between a hash check and a second open by path cannot feed other rows to the model.
    """
    from soup_cli.eval.custom import parse_eval_task_lines
    from soup_cli.utils import retention_ledger as rl

    loaded: Dict[str, list] = {}
    for task in ledger.tasks:
        try:
            suffix = Path(task.path).suffix
            if suffix.lower() != ".jsonl":
                raise ValueError(f"Expected .jsonl file, got: {suffix}")
            data = rl.read_control_bytes(task.path, f"control task {task.name} path")
            digest = hashlib.sha256(data).hexdigest()
            if digest != task.sha256:
                raise rl.LedgerError(
                    rl.changed_message(task.name, task.sha256, digest, task.path)
                )
            with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8") as stream:
                loaded[task.name] = parse_eval_task_lines(stream, Path(task.path))
        except rl.LedgerError:
            raise
        except (ValueError, OSError) as exc:
            raise rl.LedgerError(
                f"control task {task.name!r} cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
    return loaded


def _parse_load_flags(config: Optional[str], chat_template_kwargs: Optional[str]):
    """``(quantization, template options)`` read exactly the way ``soup ship`` reads them.

    The helpers are ship's own, so ``--config`` (only ``training.quantization`` is used,
    and only 4bit / 8bit) and ``--chat-template-kwargs`` cannot drift from it. A bad value
    is a usage error (exit 3) that ship prints; nothing is loaded yet.
    """
    from soup_cli.commands import ship

    quantization = None
    if config is not None:
        soup_config, _ship_config = ship._parse_ship_config(config)
        quantization = ship._live_eval_quantization_from_config(soup_config)
    return quantization, ship._parse_chat_template_kwargs(chat_template_kwargs)


def _resolve_numerics(device: Optional[str], quantization: Optional[str], template):
    """The stamp of this run: what ``soup ship`` would load, as a ``Numerics``."""
    from soup_cli.commands import ship
    from soup_cli.utils import live_eval
    from soup_cli.utils import retention_ledger as rl

    # One helper decides the precision for ship's console line, its verdict and this stamp:
    # "4bit" / "8bit" when quantized; otherwise native bf16 on capable CUDA cards,
    # fp16 on older CUDA cards, and fp32 elsewhere.
    precision = ship._live_eval_numerics(quantization, device)
    family = live_eval.resolve_device(device).split(":")[0].strip().lower()
    return rl.make_numerics(
        dtype=None if quantization else precision,
        device=family,
        quantization=quantization,
        chat_template_kwargs=template,
    )


def _numerics_line(numerics) -> str:
    return ", ".join(
        f"{key} {'none' if value is None else value}" for key, value in numerics.to_dict().items()
    )


class _CountingGenerator:
    """Wrap a generator so a FAILED generation can be told from a wrong answer.

    ``eval/generation.iter_outputs`` turns an exception, a malformed batch or a non-string
    output into a ``None`` the scorer marks wrong, so an out-of-memory error on every
    sample would read as accuracy 0.000 and a regression. This wrapper counts those events
    (an empty string is a real, wrong answer, not a failure) while passing every call and
    every output through unchanged, so ``iter_outputs`` keeps its own batch and retry logic.
    ``batch`` / ``batch_size`` exist on the wrapper only when the model's generator has them.
    """

    def __init__(self, generate: Callable[[str], str]) -> None:
        self._generate = generate
        self.failures = 0
        self.first_error: Optional[str] = None
        batch = getattr(generate, "batch", None)
        if callable(batch):
            self._batch = batch
            self.batch = self._counted_batch
            size = getattr(generate, "batch_size", None)
            if size is not None:
                self.batch_size = size

    def _note(self, error: str) -> None:
        if self.first_error is None:
            self.first_error = error[:_MAX_ERROR_CHARS]

    def __call__(self, prompt: str):
        try:
            output = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - counted, then raised to iter_outputs
            self.failures += 1
            self._note(f"{type(exc).__name__}: {exc}")
            raise
        if not isinstance(output, str):
            self.failures += 1
            self._note(f"the generator returned {type(output).__name__}, not text")
        return output

    def _counted_batch(self, prompts: list):
        # A raising batch is not counted: iter_outputs retries its prompts one by one
        # through __call__, and only the prompts that still fail are failures.
        outputs = self._batch(prompts)
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            self.failures += len(prompts)
            self._note("the batch generator returned the wrong number of outputs")
        else:
            bad = sum(1 for output in outputs if not isinstance(output, str))
            if bad:
                self.failures += bad
                self._note("the batch generator returned an output that is not text")
        return outputs


def _build_generator(
    model: str, adapter: Optional[str], device: Optional[str], numerics, template
):
    """ONE generator over ``model`` (+ ``adapter``), called the way ``soup ship`` calls it."""
    from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS
    from soup_cli.utils import live_eval

    options: Dict[str, object] = {}
    if template is not None:
        options["chat_template_kwargs"] = template
    return live_eval.make_generator(
        model,
        adapter=adapter,
        device=device,
        max_new_tokens=BEHAVIOURAL_MAX_NEW_TOKENS,
        dtype=numerics.dtype,
        quantization=numerics.quantization,
        **options,
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
    config: Optional[str],
    chat_template_kwargs: Optional[str],
    require_history: bool,
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
    quantization, template = _parse_load_flags(config, chat_template_kwargs)

    # 2. The ledger (a missing one is only created when --add-task starts it).
    loaded: Optional[rl.Ledger] = None
    try:
        ledger = rl.load_ledger(ledger_path)
        loaded = ledger
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
            f"version label {label!r} is already recorded in the ledger; "
            "labels must be unique",
            EXIT_USAGE_ERROR,
        )
    if require_history and not ledger.versions:
        _fail(
            console,
            "--require-history: the ledger holds no earlier version (a lost or reset ledger "
            "would make every task new and nothing could regress); restore the ledger, or "
            "leave the flag out for the first run of a series",
            EXIT_USAGE_ERROR,
        )

    # 3. Scores are only comparable under the numerics they were measured with.
    try:
        numerics = _resolve_numerics(device, quantization, template)
        rl.check_numerics(ledger, numerics)
    except (rl.LedgerError, ValueError) as exc:
        _fail(console, str(exc), EXIT_USAGE_ERROR)
    console.print(f"[dim]Numerics: {for_terminal(_numerics_line(numerics))}[/]")

    # 4. Register new control tasks, then prove every control file is the one on record.
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

    # 5. ONE model in memory, scored on every task. A failed generation is not a wrong
    #    answer: the first task with one stops the run, and nothing is recorded.
    from soup_cli.eval.custom import run_eval

    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    failed: Optional[Tuple[str, int, int]] = None
    counter: Optional[_CountingGenerator] = None
    try:
        counter = _CountingGenerator(_build_generator(model, adapter, device, numerics, template))
        for task in ledger.tasks:
            before = counter.failures
            result = run_eval(model, control[task.name], generate_fn=counter)
            if counter.failures > before:
                failed = (task.name, counter.failures - before, int(result.total))
                break
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
    if failed is not None and counter is not None:
        name, lost, total = failed
        _fail(
            console,
            f"generation failed for {lost} of {total} samples on control task {name!r} "
            f"(first error: {counter.first_error}); a failed generation is not a wrong answer, "
            "so no score was kept and nothing was recorded. Check the model, the adapter and "
            "the device (an out-of-memory error shows up like this).",
            EXIT_RUNTIME_ERROR,
        )

    # 6. A control file edited mid-run would fake the series; refuse before reporting.
    try:
        rl.verify_ledger_files(ledger)
        report = rl.compute_retention(
            ledger, scores, label=label, threshold=threshold, examples=counts
        )
    except rl.LedgerError as exc:
        _fail(console, f"{exc} (nothing was recorded)", EXIT_USAGE_ERROR)

    # 7. Record (only if the file is still what step 2 read), then write the report with the
    #    truth about whether it was recorded. A failed write is an error line, never a
    #    reason to hide a regression: the exit code stays 2 when a task regressed.
    recorded = False
    errors: List[str] = []
    if record:
        try:
            version = rl.new_version(
                label=label, model=model, adapter=adapter, scores=scores, now=now,
                numerics=numerics,
            )
            rl.save_ledger_if_unchanged(
                rl.append_version(ledger, version), ledger_path, loaded, label=label
            )
            recorded = True
        except (rl.LedgerError, OSError) as exc:
            detail = str(exc) if isinstance(exc, rl.LedgerError) else (
                f"{type(exc).__name__}: {exc}"
            )
            errors.append(f"cannot record in --ledger: {detail}")
    _print_report(
        console, report, recorded=recorded, ledger_path=ledger_path, record_requested=record
    )
    if output is not None:
        payload = {**report.to_dict(), "recorded": recorded, "numerics": numerics.to_dict()}
        try:
            atomic_write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", output, field="--output"
            )
            console.print(f"[green]Wrote report[/] {for_terminal(output)}")
        except (ValueError, OSError) as exc:
            errors.append(f"cannot write --output: {type(exc).__name__}: {exc}")
    for message in errors:
        _error(console, message)
    if report.regressions:
        return EXIT_GATE_FAILED
    return EXIT_RUNTIME_ERROR if errors else EXIT_OK


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
                "to the file is refused. Re-pass an existing NAME with the new PATH of a "
                "moved file to update the stored path."
            ),
        ),
        record: bool = typer.Option(
            False, "--record",
            help="Append this version's scores (and any new tasks) to the ledger.",
        ),
        threshold: float = typer.Option(
            DEFAULT_THRESHOLD, "--threshold",
            help=(
                "A drop below the best earlier score larger than this regresses "
                "(0.0-1.0, an absolute accuracy difference as a fraction)."
            ),
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Write the full task x version matrix and summary as JSON.",
        ),
        device: Optional[str] = typer.Option(
            None, "--device", help="Device for the live run (cuda / cpu).",
        ),
        config: Optional[str] = typer.Option(
            None, "--config",
            help=(
                "soup.yaml of the training run: its training.quantization (4bit / 8bit) is "
                "reused for the live load, as soup ship --config does. Nothing else in the "
                "file is used."
            ),
        ),
        chat_template_kwargs: Optional[str] = typer.Option(
            None, "--chat-template-kwargs",
            help='Live chat-template variables as a JSON object, e.g. {"enable_thinking":false}.',
        ),
        require_history: bool = typer.Option(
            False, "--require-history",
            help=(
                "Refuse (exit 3) unless the ledger already holds a recorded version. Use it "
                "in CI: a lost or reset ledger would otherwise make every task new."
            ),
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
            config=config,
            chat_template_kwargs=chat_template_kwargs,
            require_history=require_history,
        )
        raise typer.Exit(code)
