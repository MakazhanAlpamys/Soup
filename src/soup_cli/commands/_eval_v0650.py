"""v0.65.0 — `soup eval behavior` / `capability` / `checklist` / `irt-subset`.

Subcommand bundle attached to the existing ``soup eval`` Typer app via
:func:`register`. Mirrors the v0.55.0 / v0.61.0 registration pattern so
``commands/eval.py`` stays under length cap.
"""
from __future__ import annotations

import json
import os
import stat
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# v0.56.0 evidence-loader policy (review M6 fix).
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024  # 16 MiB
_MAX_RUN_ID_LEN = 256


def _validate_run_id(run_id: object) -> str:
    """Validate a CLI-passed run_id (review M5 fix).

    Run IDs are echoed into output JSON payloads and used as report keys,
    so reject null bytes / control chars / oversize before they propagate.
    """
    if not isinstance(run_id, str):
        raise typer.BadParameter("run_id must be a string")
    if "\x00" in run_id:
        raise typer.BadParameter("run_id must not contain null bytes")
    if not run_id:
        raise typer.BadParameter("run_id must not be empty")
    if len(run_id) > _MAX_RUN_ID_LEN:
        raise typer.BadParameter(
            f"run_id too long ({len(run_id)} > {_MAX_RUN_ID_LEN})"
        )
    return run_id


def _read_evidence_json(path: str, *, console: Console) -> dict:
    """Read + parse an evidence JSON file with size cap (review M6 fix).

    Uses ``O_NOFOLLOW`` (POSIX) + ``os.fstat`` on the SAME descriptor for
    size enforcement (review H-NEW-2 fix — defends against an attacker
    swapping the file between the helper's lstat and our open).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "--evidence")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise typer.BadParameter(
            f"cannot open --evidence: {type(exc).__name__}"
        ) from exc
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode):  # impossible under O_NOFOLLOW, defence-in-depth
            raise typer.BadParameter("--evidence must not be a symlink")
        if st.st_size > _MAX_EVIDENCE_BYTES:
            raise typer.BadParameter(
                f"--evidence too large ({st.st_size} > {_MAX_EVIDENCE_BYTES})"
            )
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as fh:
            raw = fh.read()
            fd = -1
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
    data = json.loads(raw)
    if not isinstance(data, dict):
        console.print("[red]Evidence must be a JSON object.[/]")
        raise typer.Exit(2)
    return data


def _write_json_output(
    payload: dict, output: str, *, console: Console, field: str = "--output",
) -> None:
    """Write ``payload`` to ``output`` atomically (review L6 dedup helper)."""
    from soup_cli.utils.paths import (
        atomic_write_text,
        enforce_under_cwd_and_no_symlink,
    )

    try:
        enforce_under_cwd_and_no_symlink(output, field)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid {field}:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    atomic_write_text(json.dumps(payload, indent=2), output, field=field)
    console.print(f"[green]Wrote {escape(output)}[/]")


def register(app: typer.Typer, console: Console) -> None:
    """Attach v0.65.0 subcommands to the existing ``soup eval`` app."""

    @app.command(name="behavior")
    def behavior_cmd(
        run_id: str = typer.Argument(..., help="Run identifier."),
        battery: str = typer.Option(
            "xstest", "--battery", "-b",
            help="Battery: xstest / harmbench / jailbreakbench / elephant / syceval.",
        ),
        evidence: Optional[str] = typer.Option(
            None, "--evidence", "-e",
            help=(
                "Path to a JSON file with "
                "{pre_responses, post_responses, oracle} arrays."
            ),
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Where to write the rendered BehaviorDiffReport JSON.",
        ),
    ) -> None:
        """Score a run on a bundled behaviour battery (pre/post diff)."""
        from soup_cli.utils.behavior_battery import (
            compute_behavior_diff,
            get_battery_spec,
            validate_battery_name,
        )

        # M5 fix — validate run_id BEFORE anything else.
        try:
            _validate_run_id(run_id)
        except typer.BadParameter as exc:
            console.print(f"[red]Invalid run_id:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        try:
            canonical = validate_battery_name(battery)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Invalid battery:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        spec = get_battery_spec(canonical)
        console.print(Panel(
            f"[bold]{escape(spec.name)}[/]\n{escape(spec.description)}\n"
            f"Axis: {escape(spec.primary_axis)}",
            title="Behaviour Battery",
            border_style="cyan",
        ))

        if evidence is None:
            # No evidence: emit neutral OK report (matches v0.56.0 diagnose
            # policy when no probes are supplied).
            console.print(
                "[yellow]No --evidence supplied; emitting neutral OK report.[/]"
            )
            payload = {
                "run_id": run_id, "battery": canonical,
                "pre": {"value": 1.0, "verdict": "OK", "num_probes": 0},
                "post": {"value": 1.0, "verdict": "OK", "num_probes": 0},
                "delta": 0.0, "overall": "OK",
            }
            if output:
                _write_json_output(payload, output, console=console)
            return

        try:
            data = _read_evidence_json(evidence, console=console)
        except (typer.BadParameter, OSError, json.JSONDecodeError) as exc:
            console.print(
                f"[red]Failed to read evidence:[/] {escape(str(exc))}"
            )
            raise typer.Exit(2) from exc

        try:
            report = compute_behavior_diff(
                run_id=run_id,
                battery=canonical,
                pre_responses=data.get("pre_responses") or [],
                post_responses=data.get("post_responses") or [],
                oracle=data.get("oracle") or [],
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Diff failed:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        table = Table(title=f"Behaviour Diff — {canonical}")
        table.add_column("Stage", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Verdict")
        table.add_row("Pre", f"{report.pre.value:.3f}", report.pre.verdict)
        table.add_row("Post", f"{report.post.value:.3f}", report.post.verdict)
        table.add_row("Δ", f"{report.delta:+.3f}", report.overall)
        console.print(table)

        if output:
            _write_json_output(report.to_dict(), output, console=console)

        if report.overall == "MAJOR":
            raise typer.Exit(2)

    @app.command(name="capability")
    def capability_cmd(
        run_id: str = typer.Argument(..., help="Run identifier."),
        suite: str = typer.Option(
            "fast", "--suite", "-s",
            help="Profile: full / fast / math / code.",
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Where to write the rendered CapabilityReport JSON.",
        ),
    ) -> None:
        """Run a bundled capability profile (MMLU-Pro / GPQA / AIME / ...)."""
        from soup_cli.utils.capability_suite import (
            list_suites,
            resolve_suite,
            validate_suite_name,
        )

        try:
            _validate_run_id(run_id)
        except typer.BadParameter as exc:
            console.print(f"[red]Invalid run_id:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        try:
            canonical = validate_suite_name(suite)
        except (TypeError, ValueError) as exc:
            console.print(
                f"[red]Invalid --suite:[/] {escape(str(exc))} "
                f"(valid: {', '.join(list_suites())})"
            )
            raise typer.Exit(2) from exc

        benchmarks = resolve_suite(canonical)
        table = Table(title=f"Capability Suite — {canonical}")
        table.add_column("Benchmark")
        table.add_column("lm-eval task")
        for b in benchmarks:
            table.add_row(escape(b.name), escape(b.lm_eval_task))
        console.print(table)

        payload = {
            "run_id": run_id,
            "suite": canonical,
            "benchmarks": [{"name": b.name, "task": b.lm_eval_task} for b in benchmarks],
            "note": (
                "Live lm-eval-harness wiring is operator-driven; "
                "the listed tasks are pre-validated friendly defaults."
            ),
        }
        if output:
            _write_json_output(payload, output, console=console)

    @app.command(name="checklist")
    def checklist_cmd(
        spec_path: str = typer.Argument(
            ..., help="Path to CheckList DSL YAML.",
        ),
        evidence: Optional[str] = typer.Option(
            None, "--evidence", "-e",
            help="Optional JSON with operator-supplied per-test responses.",
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Where to write the rendered CheckListReport JSON.",
        ),
    ) -> None:
        """Run CheckList MFT / INV / DIR behavioural tests."""
        from soup_cli.utils.checklist_dsl import (
            load_checklist_spec,
            run_checklist_spec,
        )

        try:
            spec = load_checklist_spec(spec_path)
        except (TypeError, ValueError, OSError) as exc:
            console.print(f"[red]Failed to load spec:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        evidence_map = None
        if evidence is not None:
            try:
                evidence_map = _read_evidence_json(evidence, console=console)
            except (typer.BadParameter, OSError, json.JSONDecodeError) as exc:
                console.print(
                    f"[red]Failed to read evidence:[/] {escape(str(exc))}"
                )
                raise typer.Exit(2) from exc

        try:
            report = run_checklist_spec(spec, evidence=evidence_map)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]CheckList run failed:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        table = Table(title="CheckList Tests")
        table.add_column("Name", style="bold")
        table.add_column("Kind")
        table.add_column("Passed", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Verdict")
        for result in report.results:
            table.add_row(
                escape(result.name),
                result.kind,
                str(result.passed),
                str(result.total),
                result.verdict,
            )
        console.print(table)
        console.print(
            f"Overall: [bold]{report.overall}[/]"
        )

        if output:
            _write_json_output(report.to_dict(), output, console=console)

        if report.overall == "MAJOR":
            raise typer.Exit(2)

    @app.command(name="irt-subset")
    def irt_subset_cmd(
        responses_path: str = typer.Argument(
            ..., help="Path to JSONL with rows {item_id, correct(bool), score?}.",
        ),
        size: str = typer.Option(
            "small", "--size", "-z",
            help="Subset profile: full / small / tiny.",
        ),
        output: Optional[str] = typer.Option(
            None, "--output", "-o",
            help="Where to write the rendered IrtSubsetPlan JSON.",
        ),
    ) -> None:
        """Pick a minimum-cost eval subset that preserves ranking power."""
        from soup_cli.utils.irt import (
            IRT_PROFILES,
            fit_difficulty,
            load_response_rows,
            pick_irt_subset,
        )

        if size not in IRT_PROFILES:
            console.print(
                f"[red]Invalid --size: {escape(size)} "
                f"(valid: {', '.join(sorted(IRT_PROFILES))})[/]"
            )
            raise typer.Exit(2)

        try:
            rows = load_response_rows(responses_path)
        except (TypeError, ValueError, OSError) as exc:
            console.print(f"[red]Failed to load responses:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        try:
            difficulty = fit_difficulty(rows)
            plan = pick_irt_subset(difficulty, size=size)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]IRT fit failed:[/] {escape(str(exc))}")
            raise typer.Exit(2) from exc

        console.print(Panel(
            f"Profile: [bold]{escape(plan.size)}[/]\n"
            f"Selected: {len(plan.item_ids)} / {plan.total_items}\n"
            f"Approx cost cut: {plan.cost_ratio:.1%}",
            title="IRT Subset",
            border_style="green",
        ))

        if output:
            _write_json_output(plan.to_dict(), output, console=console)
