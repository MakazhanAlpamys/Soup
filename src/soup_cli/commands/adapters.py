"""soup adapters — LoRA adapter management (list, info, compare, diff, merge, blame, branch)."""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)

# Strip C0/DEL control bytes from adapter-config-derived text before it reaches
# the terminal. rich.markup.escape() only neutralises Rich's [...] tags, not raw
# ANSI/OSC/ESC bytes that could spoof a title bar or hide a refusal (mirrors
# commands/data_doctor.py::_for_terminal). Keep tab/LF/CR.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


def _for_terminal(text: str) -> str:
    return text.translate(_CONTROL_STRIP_TABLE)


def _find_adapters(directory: Path, max_depth: int = 6) -> list[Path]:
    """Recursively find directories containing adapter_config.json.

    Limited to max_depth levels and 200 results to prevent runaway scans.
    """
    adapters = []
    for config_file in directory.rglob("adapter_config.json"):
        try:
            depth = len(config_file.relative_to(directory).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            adapters.append(config_file.parent)
        if len(adapters) >= 200:
            break
    return sorted(adapters)


def _read_adapter_config(adapter_path: Path) -> dict:
    """Read adapter_config.json from an adapter directory."""
    config_file = adapter_path / "adapter_config.json"
    with open(config_file, encoding="utf-8") as fh:
        return json.load(fh)


def _get_adapter_size(adapter_path: Path) -> str:
    """Get total size of adapter files on disk."""
    total_bytes = 0
    for file_path in adapter_path.iterdir():
        if file_path.is_file():
            total_bytes += file_path.stat().st_size

    if total_bytes < 1024:
        return f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


@app.command(name="list")
def list_adapters(
    directory: str = typer.Argument(
        ".", help="Directory to scan for adapters (recursive)"
    ),
):
    """Scan a directory for LoRA adapters and list them."""
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        console.print(f"[red]Directory not found: {directory}[/]")
        raise typer.Exit(1)

    adapters = _find_adapters(dir_path)

    if not adapters:
        console.print("[dim]No adapters found in this directory.[/]")
        return

    table = Table(title=f"Adapters in {dir_path}")
    table.add_column("Path", style="bold")
    table.add_column("Base Model")
    table.add_column("LoRA r")
    table.add_column("Type")
    table.add_column("Size")

    for adapter_path in adapters:
        try:
            config = _read_adapter_config(adapter_path)
            base = config.get("base_model_name_or_path", "unknown")
            lora_r = str(config.get("r", "?"))
            peft_type = config.get("peft_type", "?")
            size = _get_adapter_size(adapter_path)

            # Shorten path relative to scan directory
            try:
                rel_path = adapter_path.relative_to(dir_path)
            except ValueError:
                rel_path = adapter_path
            # Escape every adapter-config-sourced value — a crafted
            # base_model_name_or_path like "[link=evil]click[/]" would
            # otherwise render as live Rich markup in the terminal.
            table.add_row(
                escape(str(rel_path)),
                escape(str(base)),
                escape(lora_r),
                escape(str(peft_type)),
                size,
            )
        except (json.JSONDecodeError, OSError):
            table.add_row(escape(str(adapter_path)), "[red]error[/]", "-", "-", "-")

    console.print(table)
    console.print(f"\n[dim]Found {len(adapters)} adapter(s).[/]")


@app.command()
def info(
    adapter: str = typer.Argument(..., help="Path to adapter directory"),
):
    """Show detailed metadata for a LoRA adapter."""
    adapter_path = Path(adapter).resolve()
    config_file = adapter_path / "adapter_config.json"

    if not adapter_path.exists():
        console.print(f"[red]Adapter not found: {adapter}[/]")
        raise typer.Exit(1)

    if not config_file.exists():
        console.print(f"[red]No adapter_config.json in: {adapter}[/]")
        raise typer.Exit(1)

    config = _read_adapter_config(adapter_path)
    size = _get_adapter_size(adapter_path)

    base_model = config.get("base_model_name_or_path", "unknown")
    lora_r = config.get("r", "?")
    lora_alpha = config.get("lora_alpha", "?")
    lora_dropout = config.get("lora_dropout", "?")
    peft_type = config.get("peft_type", "?")
    task_type = config.get("task_type", "?")
    target_modules = config.get("target_modules", [])

    if isinstance(target_modules, list):
        modules_str = ", ".join(target_modules)
    else:
        modules_str = str(target_modules)

    # Escape every adapter-config-sourced value before embedding into
    # Rich markup. A crafted base_model_name_or_path like
    # "[link=http://evil]click[/]" would otherwise render as a live
    # clickable link in the terminal.
    info_text = (
        f"Base model: [bold]{escape(str(base_model))}[/]\n"
        f"PEFT type:  [bold]{escape(str(peft_type))}[/]\n"
        f"Task:       [bold]{escape(str(task_type))}[/]\n"
        f"LoRA rank:  [bold]{escape(str(lora_r))}[/], "
        f"alpha: [bold]{escape(str(lora_alpha))}[/], "
        f"dropout: [bold]{escape(str(lora_dropout))}[/]\n"
        f"Targets:    [bold]{escape(modules_str)}[/]\n"
        f"Size on disk: [bold]{size}[/]"
    )

    console.print(Panel(info_text, title=f"Adapter Info -- {escape(adapter_path.name)}"))


@app.command()
def compare(
    adapter1: str = typer.Argument(..., help="Path to first adapter"),
    adapter2: str = typer.Argument(..., help="Path to second adapter"),
):
    """Compare two LoRA adapters side-by-side."""
    path1 = Path(adapter1).resolve()
    path2 = Path(adapter2).resolve()

    for label, path in [("Adapter 1", path1), ("Adapter 2", path2)]:
        config_file = path / "adapter_config.json"
        if not path.exists():
            console.print(f"[red]{label} not found: {path}[/]")
            raise typer.Exit(1)
        if not config_file.exists():
            console.print(f"[red]{label} has no adapter_config.json: {path}[/]")
            raise typer.Exit(1)

    config1 = _read_adapter_config(path1)
    config2 = _read_adapter_config(path2)

    table = Table(title="Adapter Comparison")
    table.add_column("Field", style="bold")
    table.add_column(path1.name, justify="center")
    table.add_column(path2.name, justify="center")

    # Fields to compare
    fields = [
        ("Base model", "base_model_name_or_path"),
        ("PEFT type", "peft_type"),
        ("Task type", "task_type"),
        ("LoRA rank (r)", "r"),
        ("LoRA alpha", "lora_alpha"),
        ("LoRA dropout", "lora_dropout"),
        ("Target modules", "target_modules"),
    ]

    for label, key in fields:
        val1 = config1.get(key, "-")
        val2 = config2.get(key, "-")

        # Format lists
        if isinstance(val1, list):
            val1 = ", ".join(str(item) for item in val1)
        if isinstance(val2, list):
            val2 = ", ".join(str(item) for item in val2)

        # Escape always at the value layer; decoration wraps after.
        # Mirrors v0.57.0 `adapters diff` / `info` policy — equal-value
        # rows must NOT skip escape just because the highlight branch
        # doesn't fire (otherwise a shared crafted value like
        # "[link=evil]click[/]" still injects live markup).
        val1_str = escape(str(val1))
        val2_str = escape(str(val2))

        # Highlight differences (already-escaped values wrap in yellow).
        if val1_str != val2_str:
            val1_str = f"[yellow]{val1_str}[/]"
            val2_str = f"[yellow]{val2_str}[/]"

        table.add_row(label, val1_str, val2_str)

    # Add size comparison
    size1 = _get_adapter_size(path1)
    size2 = _get_adapter_size(path2)
    table.add_row("Size on disk", size1, size2)

    console.print(table)


@app.command()
def diff(
    adapter_a: str = typer.Argument(..., help="Path to first adapter"),
    adapter_b: str = typer.Argument(..., help="Path to second adapter"),
    top_k: int = typer.Option(10, "--top-k", min=1, max=200,
                              help="Number of top changed projections to report"),
    output_format: str = typer.Option("table", "--format",
                                      help="Output format: table | json | markdown"),
    output: str = typer.Option(None, "--output", "-o",
                               help="Write report to file (json/markdown only)"),
):
    """Per-layer ΔW Frobenius diff + effective-rank drift (v0.57.0)."""
    from soup_cli.utils.adapter_diff import (
        compute_adapter_diff,
        render_report_json,
        render_report_markdown,
    )
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    fmt = output_format.lower()
    if fmt not in ("table", "json", "markdown"):
        console.print(f"[red]Unknown --format: {escape(fmt)}[/]")
        raise typer.Exit(2)

    try:
        report = compute_adapter_diff(adapter_a, adapter_b, top_k=top_k)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    if fmt == "json":
        text = render_report_json(report)
    elif fmt == "markdown":
        text = render_report_markdown(report)
    else:
        text = None

    if output is not None:
        if fmt == "table":
            console.print("[red]--output requires --format json or markdown[/]")
            raise typer.Exit(2)
        enforce_under_cwd_and_no_symlink(output, "output")
        # Atomic write via tempfile + os.replace — a crash mid-write must
        # not leave a partial report at the target path (review fix MEDIUM).
        import os as _os
        import tempfile as _tf
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tf.mkstemp(dir=str(target.parent), prefix=".tmp_")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            _os.replace(tmp, str(target))
        except Exception:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
            raise
        console.print(f"[green]Wrote {fmt} report to {escape(output)}[/]")
        return

    if fmt != "table":
        console.print(text)
        return

    # Default Rich table
    table = Table(title=f"Adapter diff: {escape(report.adapter_a)} vs {escape(report.adapter_b)}")
    table.add_column("Layer", style="bold")
    table.add_column("ΔW Frobenius", justify="right")
    table.add_column("Relative", justify="right")
    for layer in sorted(report.per_layer, key=lambda d: d.frobenius, reverse=True)[:top_k]:
        table.add_row(
            escape(layer.name),
            f"{layer.frobenius:.4f}",
            f"{layer.relative:.2%}",
        )
    console.print(table)
    if report.effective_rank_a is not None and report.effective_rank_b is not None:
        console.print(
            f"Effective rank: A={report.effective_rank_a:.2f}, "
            f"B={report.effective_rank_b:.2f}"
        )
    console.print(
        f"Shared layers: {report.shared_layers} | "
        f"only-in-A: {len(report.only_in_a)} | only-in-B: {len(report.only_in_b)}"
    )


@app.command()
def merge(
    adapters: list[str] = typer.Argument(..., help="Two or more adapter paths to merge"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for merged adapter"),
    strategy: str = typer.Option("linear", "--strategy",
                                 help="linear | ties | dare | svd | cmaes"),
    weights: str = typer.Option(None, "--weights",
                                help="Comma-separated weights (default: equal)"),
    density: float = typer.Option(0.2, "--density",
                                  help="Trim density for ties/dare in (0, 1]"),
    seed: int = typer.Option(0, "--seed", help="Random seed for dare / cmaes"),
    rank: int = typer.Option(None, "--rank", help="SVD rank (svd strategy only)"),
    eval_suite: Optional[str] = typer.Option(
        None, "--eval",
        help=(
            "Path to eval suite (required for --strategy cmaes). "
            "Used by the evolutionary loop to score candidate merges."
        ),
    ),
    budget: str = typer.Option(
        "1h", "--budget",
        help="Wall-clock budget for cmaes (60s..24h, e.g. 1h, 30m)",
    ),
    population: int = typer.Option(
        8, "--population", min=2, max=256,
        help="cmaes population size per generation",
    ),
    max_generations: int = typer.Option(
        20, "--max-generations", min=1, max=10_000,
        help="cmaes generation cap",
    ),
    license_ids: list[str] = typer.Option(
        None, "--license",
        help=(
            "SPDX license id per adapter (repeatable; same order as inputs). "
            "v0.60.0 refuses merges with conflicting licenses unless "
            "--license-override <reason> is passed."
        ),
    ),
    license_override: str = typer.Option(
        None, "--license-override",
        help=(
            "Free-text justification (>=8 chars) to merge across a license "
            "conflict. Logged to the audit log for legal review (v0.71.2)."
        ),
    ),
    allow_unscanned: bool = typer.Option(
        False, "--allow-unscanned",
        help=(
            "Skip the v0.71.2 backdoor-scan gate. By default `adapters merge` "
            "refuses to merge an input whose `adapters scan` returns FAIL (or "
            "that cannot be scanned)."
        ),
    ),
    canary: Optional[str] = typer.Option(
        None, "--canary",
        help=(
            "Canary-suite JSON to compute a live OK/MINOR/MAJOR verdict for "
            "the merged adapter vs the first input (v0.71.4 #172). Shape: "
            '{"baseline_scores": [...], "candidate_scores": [...]}.'
        ),
    ),
    strict_verdict: bool = typer.Option(
        False, "--strict-verdict",
        help="Exit 2 when the canary verdict is MAJOR (CI gate).",
    ),
):
    """Merge LoRA adapters via linear / ties / dare / svd (v0.57.0, v0.60.0 license gate)."""
    from soup_cli.utils.adapter_merge import SUPPORTED_STRATEGIES, merge_adapters
    from soup_cli.utils.license_matrix import (
        check_license_compat,
        extract_license_from_adapter,
        record_license_override,
        validate_license_override_reason,
    )

    if strategy not in SUPPORTED_STRATEGIES:
        console.print(
            f"[red]Unknown --strategy: {escape(strategy)}. "
            f"Choose from: {', '.join(SUPPORTED_STRATEGIES)}[/]"
        )
        raise typer.Exit(2)

    if len(adapters) < 2:
        console.print("[red]Need at least 2 adapter paths to merge[/]")
        raise typer.Exit(2)

    # cmaes argument validation runs FIRST (before any file I/O / scanning):
    # a user who forgot --eval should get that error immediately, not a scan
    # error on their inputs. The cmaes *search* itself still runs after the
    # security gates below (v0.71.4 review — preserves the "reject missing
    # --eval before file-handling" contract while gating the merge).
    if strategy == "cmaes":
        if eval_suite is None:
            console.print(
                "[red]--strategy cmaes requires --eval <suite> "
                "(path to a YAML/JSONL eval).[/]"
            )
            raise typer.Exit(2)
        if canary is not None or strict_verdict:
            console.print(
                "[red]--canary / --strict-verdict are not supported with "
                "--strategy cmaes; the --eval suite already drives the search.[/]"
            )
            raise typer.Exit(2)

    # v0.71.2 #192 — backdoor-scan gate. Refuse to merge any input whose
    # spectral scan returns FAIL (or that cannot be scanned at all) unless the
    # operator passes --allow-unscanned. WARN is advisory-only. Runs BEFORE the
    # strategy dispatch (incl. cmaes, which v0.71.4 #220 made weight-writing) so
    # EVERY strategy that produces a merged adapter is gated uniformly.
    if not allow_unscanned:
        from soup_cli.utils.adapter_scan import scan_adapter

        for adapter_path in adapters:
            try:
                scan_report = scan_adapter(adapter_path)
            except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
                console.print(
                    f"[red]Could not scan {escape(adapter_path)} "
                    f"({escape(str(exc))}).[/]\n"
                    "[dim]Pass --allow-unscanned to merge anyway.[/]"
                )
                raise typer.Exit(3) from exc
            if scan_report.overall == "FAIL":
                console.print(
                    f"[red]Backdoor scan FAILED for {escape(adapter_path)}: "
                    f"{escape(scan_report.summary)}[/]\n"
                    "[dim]Pass --allow-unscanned to merge anyway "
                    "(not recommended).[/]"
                )
                raise typer.Exit(3)
            if scan_report.overall == "WARN":
                console.print(
                    f"[yellow]Backdoor scan WARN for {escape(adapter_path)}: "
                    f"{escape(scan_report.summary)} — proceeding.[/]"
                )

    # v0.60.0 Part E: license-conflict gate. Operators may declare a license
    # per adapter (--license, repeatable) OR rely on v0.71.2 #187
    # auto-extraction from adapter_config.json / config.json / model-card
    # frontmatter. On a conflict, --license-override <reason> proceeds and the
    # decision is recorded to the audit log (#190). Runs before the strategy
    # dispatch so cmaes (#220, weight-writing) is gated like every other merge.
    gate_licenses: Optional[list[str]] = None
    if license_ids:
        if len(license_ids) != len(adapters):
            console.print(
                f"[red]--license count {len(license_ids)} must match "
                f"adapter count {len(adapters)}[/]"
            )
            raise typer.Exit(2)
        gate_licenses = list(license_ids)
    else:
        # #187 — auto-detect. Only run the gate if EVERY adapter resolves to a
        # known license; partial info would make the conflict check misleading.
        extracted = [extract_license_from_adapter(a) for a in adapters]
        if all(e is not None for e in extracted):
            gate_licenses = [str(e) for e in extracted]
            console.print(
                "[dim]Auto-detected licenses: "
                f"{escape(', '.join(gate_licenses))}[/]"
            )
        elif license_override is not None:
            console.print(
                "[yellow]--license-override given but licenses could not be "
                "auto-detected for every adapter; no conflict gate triggered.[/]"
            )

    if gate_licenses is not None:
        try:
            license_report = check_license_compat(gate_licenses)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]License check failed: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        if not license_report.ok:
            if license_override is None:
                console.print(
                    f"[red]License conflict refused: "
                    f"{escape(license_report.reason)}[/]"
                )
                console.print(
                    "[dim]Pass --license-override '<legal-cleared justification>' "
                    "(>=8 chars) to proceed.[/]"
                )
                raise typer.Exit(3)
            try:
                cleared = validate_license_override_reason(license_override)
            except (TypeError, ValueError) as exc:
                console.print(
                    f"[red]Invalid --license-override: {escape(str(exc))}[/]"
                )
                raise typer.Exit(2) from exc
            # #190 — persist the override decision for legal review.
            record_license_override(gate_licenses, cleared)
            console.print(
                f"[yellow]License conflict overridden:[/] "
                f"{escape(license_report.reason)}\n"
                f"[dim]Reason: {escape(cleared)} (recorded to audit log)[/]"
            )

    # v0.67.0 Part A: cmaes evolutionary search requires an eval suite +
    # budget. v0.71.4 #220 lifts this from plan-only to a live loop: the eval
    # suite is auto-wired into an eval_fn that materialises + scores each
    # candidate merge, and the best-weighted merge is written to --output.
    # (Arg validation already happened above, before the security gates.)
    if strategy == "cmaes":
        assert eval_suite is not None  # narrowed by the early arg-check
        from soup_cli.utils.cmaes_merge import (
            build_cmaes_eval_fn,
            build_cmaes_plan,
            run_cmaes_merge,
        )

        try:
            cmaes_plan = build_cmaes_plan(
                adapters=adapters,
                eval_suite=eval_suite,
                budget_spec=budget,
                population_size=population,
                max_generations=max_generations,
                seed=seed,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            console.print(f"[red]Invalid cmaes plan: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

        console.print(
            f"[cyan]Running CMA-ES merge[/] "
            f"(pop={cmaes_plan.population_size}, "
            f"max_gen={cmaes_plan.max_generations}, "
            f"budget={cmaes_plan.budget_seconds}s)..."
        )
        console.print(
            "[dim]The default scorer reloads the base model per candidate "
            "(pop x generations loads); pass a small --population for large "
            "models, or wire a cached scorer.[/]"
        )
        try:
            eval_fn = build_cmaes_eval_fn(cmaes_plan)
            result = run_cmaes_merge(cmaes_plan, eval_fn=eval_fn)
        except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as exc:
            console.print(f"[red]cmaes merge failed: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

        # Write the best-weighted merge to --output (linear with the winning
        # simplex weights).
        try:
            report = merge_adapters(
                adapters, output, strategy="linear",
                weights=list(result.best_weights),
            )
        except (FileNotFoundError, TypeError, ValueError, RuntimeError, OSError) as exc:
            console.print(f"[red]Could not write merged adapter: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

        conv_color = "green" if result.converged else "yellow"
        best_w = ", ".join(f"{w:.3f}" for w in result.best_weights)
        console.print(
            Panel(
                f"Strategy:        [bold]cmaes[/]\n"
                f"Adapters:        {len(cmaes_plan.adapters)}\n"
                f"Eval suite:      [bold]{escape(os.path.basename(cmaes_plan.eval_suite))}[/]\n"
                f"Generations:     [bold]{result.generations_run}[/] / "
                f"{cmaes_plan.max_generations}\n"
                f"Evaluations:     [bold]{result.evaluations}[/]\n"
                f"Best score:      [bold]{result.best_score:.4f}[/]\n"
                f"Best weights:    [dim]{escape(best_w)}[/]\n"
                f"Converged:       [{conv_color}]{result.converged}[/]\n"
                f"Merged layers:   [bold]{report.merged_layers}[/]\n"
                f"Output:          [bold]{escape(report.output_dir)}[/]",
                title="Adapter merge — cmaes",
            )
        )
        return

    parsed_weights = None
    if weights:
        try:
            parsed_weights = [float(w.strip()) for w in weights.split(",")]
        except ValueError as exc:
            console.print(f"[red]Invalid --weights: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        report = merge_adapters(
            adapters,
            output,
            strategy=strategy,  # type: ignore[arg-type]
            weights=parsed_weights,
            density=density,
            seed=seed,
            rank=rank,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    # v0.71.4 #172 — live canary verdict. With --canary, score the merged
    # adapter vs the first input and classify OK/MINOR/MAJOR; without it the
    # verdict stays UNKNOWN (advisory).
    verdict = report.verdict
    verdict_suffix = " (pass --canary <suite.json> to compute)"
    if canary is not None:
        from soup_cli.utils.adapter_merge import predict_merged_verdict
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(canary, "canary")
            verdict = predict_merged_verdict(report, canary)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Canary verdict failed: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        verdict_suffix = ""

    verdict_color = {"OK": "green", "MINOR": "yellow", "MAJOR": "red"}.get(
        verdict, "yellow"
    )
    panel = Panel(
        f"Strategy:       [bold]{escape(report.strategy)}[/]\n"
        f"Inputs:         {len(report.adapters)}\n"
        f"Merged layers:  [bold]{report.merged_layers}[/]\n"
        f"Skipped layers: {len(report.skipped_layers)}\n"
        f"Verdict:        [{verdict_color}]{verdict}[/]{verdict_suffix}\n"
        f"Output:         [bold]{escape(report.output_dir)}[/]",
        title="Adapter merge",
    )
    console.print(panel)

    if strict_verdict and verdict == "MAJOR":
        raise typer.Exit(2)


@app.command()
def arithmetic(
    expression: str = typer.Argument(
        ...,
        help='Task-vector expression, e.g. "coder + 0.5*math - toxic".',
    ),
    adapter: list[str] = typer.Option(
        ...,
        "--adapter",
        help=(
            "name=path mapping for each adapter referenced in the expression "
            "(repeatable). Names use [A-Za-z0-9_.-]."
        ),
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output directory for the merged adapter"
    ),
    rank: Optional[int] = typer.Option(
        None,
        "--rank",
        min=1,
        help=(
            "Truncated-SVD rank cap. Passing it routes the merge through the "
            "exact concat path and caps the output rank at N via SVD (works for "
            "same-rank AND mixed-rank inputs). Without it, mixed-rank inputs keep "
            "the full concatenated rank (Σ rᵢ) and same-rank inputs use the fast "
            "element-wise path."
        ),
    ),
    allow_unscanned: bool = typer.Option(
        False,
        "--allow-unscanned",
        help=(
            "Skip the backdoor-scan gate (v0.71.2 #192). By default refuses an "
            "input whose `adapters scan` returns FAIL or cannot be scanned."
        ),
    ),
    allow_cross_base: bool = typer.Option(
        False,
        "--allow-cross-base",
        help=(
            "Allow combining adapters trained on different base models. By "
            "default a base-model mismatch is refused (task vectors from "
            "different bases are not comparable)."
        ),
    ),
):
    """Task-vector arithmetic over LoRA adapters (add / scale / negate) (v0.71.34).

    Applies task arithmetic (arXiv:2212.04089) to LoRA deltas. Same-rank adapters
    take a fast element-wise path; mixed-rank adapters take an exact
    concatenation path (#305) so ``B_out @ A_out = Σ cᵢ·(Bᵢ@Aᵢ)`` for any
    per-adapter rank — pass ``--rank N`` to cap the concatenated rank via
    truncated SVD. Example:

        soup adapters arithmetic "coder + 0.5*math - toxic" \\
            --adapter coder=./coder --adapter math=./math \\
            --adapter toxic=./toxic -o ./merged

    Exit 0 = success, 1 = any refusal (bad expression, unknown adapter, shape
    mismatch, cross-base without --allow-cross-base, scan FAIL, path escape).
    """
    import re

    from soup_cli.utils.adapter_arithmetic import (
        ArithmeticReport,
        _detect_lora_rank,
        merge_task_arithmetic,
        merge_task_arithmetic_concat,
        parse_expression,
        read_adapter_base,
        read_adapter_lora_scaling,
    )
    from soup_cli.utils.adapter_diff import load_adapter_weights
    from soup_cli.utils.adapter_merge import write_merged_adapter
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    # 1. Parse the --adapter name=path map.
    name_re = re.compile(r"^[A-Za-z0-9_.\-]+$")
    mapping: dict[str, str] = {}
    for spec in adapter:
        if "=" not in spec:
            console.print(
                f"[red]--adapter must be name=path, got {escape(spec)}[/]"
            )
            raise typer.Exit(1)
        name, _, path = spec.partition("=")
        name, path = name.strip(), path.strip()
        if not name_re.match(name):
            console.print(
                f"[red]Invalid adapter name {escape(name)!r} "
                "(use [A-Za-z0-9_.-]).[/]"
            )
            raise typer.Exit(1)
        if not path:
            console.print(f"[red]--adapter {escape(name)} has an empty path[/]")
            raise typer.Exit(1)
        if name in mapping:
            console.print(f"[red]Duplicate --adapter name {escape(name)}[/]")
            raise typer.Exit(1)
        try:
            enforce_under_cwd_and_no_symlink(path, "adapter")
        except (ValueError, OSError) as exc:
            console.print(
                f"[red]Adapter path refused for {escape(name)}: "
                f"{escape(str(exc))}[/]"
            )
            raise typer.Exit(1) from exc
        mapping[name] = path

    # 2. Parse the expression against the declared names.
    try:
        terms = parse_expression(expression, set(mapping))
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Bad expression: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    referenced = [t.name for t in terms]
    ref_paths = [mapping[n] for n in referenced]

    # 3. Backdoor-scan gate (mirrors `adapters merge`, #192). Refuse FAIL /
    #    un-scannable inputs unless --allow-unscanned; WARN is advisory.
    if not allow_unscanned:
        from soup_cli.utils.adapter_scan import scan_adapter

        for name, path in zip(referenced, ref_paths):
            try:
                scan_report = scan_adapter(path)
            except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
                console.print(
                    f"[red]Could not scan {escape(name)} "
                    f"({escape(str(exc))}).[/]\n"
                    "[dim]Pass --allow-unscanned to proceed anyway.[/]"
                )
                raise typer.Exit(1) from exc
            if scan_report.overall == "FAIL":
                console.print(
                    f"[red]Backdoor scan FAILED for {escape(name)}: "
                    f"{escape(scan_report.summary)}[/]\n"
                    "[dim]Pass --allow-unscanned to proceed anyway "
                    "(not recommended).[/]"
                )
                raise typer.Exit(1)
            if scan_report.overall == "WARN":
                console.print(
                    f"[yellow]Backdoor scan WARN for {escape(name)}: "
                    f"{escape(scan_report.summary)} — proceeding.[/]"
                )

    # 4. Same-base check (unless overridden).
    bases: dict[str, str | None] = {}
    for name, path in zip(referenced, ref_paths):
        try:
            bases[name] = read_adapter_base(path)
        except (ValueError, OSError) as exc:
            console.print(
                f"[red]Could not read base model for {escape(name)}: "
                f"{escape(str(exc))}[/]"
            )
            raise typer.Exit(1) from exc
    distinct = {b for b in bases.values() if b is not None}
    if len(distinct) > 1:
        if not allow_cross_base:
            listed = ", ".join(
                f"{escape(n)}={escape(_for_terminal(str(b)))}"
                for n, b in bases.items()
            )
            console.print(
                f"[red]Base-model mismatch across adapters: {listed}.[/]\n"
                "[dim]Task vectors from different bases are not comparable. "
                "Pass --allow-cross-base to proceed anyway.[/]"
            )
            raise typer.Exit(1)
        console.print(
            "[yellow]--allow-cross-base: combining adapters from different "
            "base models (results may be meaningless).[/]"
        )

    # 5. Output containment.
    try:
        enforce_under_cwd_and_no_symlink(output, "output")
    except (ValueError, OSError) as exc:
        console.print(f"[red]Output path refused: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    # 6. Load, signed-merge, write.
    try:
        weights_list = [load_adapter_weights(mapping[n]) for n in referenced]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Could not load adapter weights: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    coeffs = [t.coeff for t in terms]

    # Route: mixed-rank inputs (or an explicit --rank truncation) take the exact
    # concat path (#305); same-rank inputs keep the fast element-wise path.
    ranks = [_detect_lora_rank(w) for w in weights_list]
    distinct_ranks = {r for r in ranks if r is not None}
    use_concat = len(distinct_ranks) > 1 or rank is not None
    config_overrides: Optional[dict[str, object]] = None

    if use_concat:
        # Bake each adapter's own decode-time scaling (lora_alpha/r) into its A
        # block so a single-scaling output reproduces Σ cᵢ·(effective ΔWᵢ). Use
        # an explicit `is None` fallback so a valid scaling of 0.0 (lora_alpha=0,
        # r>0 — an intentionally zeroed task vector) is NOT silently promoted to
        # 1.0 by `or`.
        scalings = []
        for n in referenced:
            s = read_adapter_lora_scaling(mapping[n])
            scalings.append(s if s is not None else 1.0)
        try:
            merged, skipped, new_rank = merge_task_arithmetic_concat(
                weights_list, coeffs, scalings=scalings, rank=rank
            )
        except ValueError as exc:
            console.print(f"[red]Merge failed: {escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        if not merged:
            console.print(
                "[red]No shared LoRA A/B pairs across the referenced adapters — "
                "nothing to merge.[/]"
            )
            raise typer.Exit(1)
        # Output is a single-scaling adapter: r == lora_alpha == new_rank. Clear
        # any per-module rank_pattern/alpha_pattern (v0.39.0) carried over from
        # the template source — the concat output is uniform-rank, so stale
        # per-module overrides would mis-scale it.
        config_overrides = {
            "r": new_rank,
            "lora_alpha": new_rank,
            "rank_pattern": {},
            "alpha_pattern": {},
        }
    else:
        try:
            merged, skipped = merge_task_arithmetic(weights_list, coeffs)
        except ValueError as exc:
            console.print(f"[red]Merge failed: {escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        if not merged:
            console.print(
                "[red]No shared LoRA tensors across the referenced adapters — "
                "nothing to merge.[/]"
            )
            raise typer.Exit(1)
    try:
        write_merged_adapter(
            output, ref_paths[0], merged, config_overrides=config_overrides
        )
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Could not write merged adapter: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    report = ArithmeticReport(
        expression=expression,
        terms=tuple(terms),
        output_dir=output,
        merged_layers=len(merged),
        skipped_layers=skipped,
        base_model=next(iter(distinct)) if len(distinct) == 1 else None,
    )
    terms_str = "  ".join(
        f"[bold]{t.coeff:+g}[/]·{escape(t.name)}" for t in report.terms
    )
    panel = Panel(
        f"Expression:     {escape(report.expression)}\n"
        f"Terms:          {terms_str}\n"
        f"Base model:     {escape(_for_terminal(str(report.base_model)))}\n"
        f"Merged tensors: [bold]{report.merged_layers}[/]\n"
        f"Skipped tensors:{len(report.skipped_layers)}\n"
        f"Output:         [bold]{escape(report.output_dir)}[/]",
        title="Adapter arithmetic",
    )
    console.print(panel)


@app.command()
def blame(
    adapter_dir: str = typer.Argument(..., help="Path to trained adapter"),
    dataset: str = typer.Option(..., "--dataset", help="Training JSONL the adapter was built on"),
    layer: str = typer.Option(..., "--layer", help="Layer to attribute (e.g. q_proj.7)"),
    budget: str = typer.Option("4h", "--budget", help="Wall-clock budget (e.g. 4h, 30m)"),
    num_shards: int = typer.Option(10, "--shards", min=2, max=100,
                                   help="Number of dataset shards for leave-one-out"),
    top_k: int = typer.Option(50, "--top-k", min=1, max=10000,
                              help="Number of top influencer rows to report (v0.66.0)"),
    plan_only: bool = typer.Option(False, "--plan-only",
                                   help="Print plan and exit without running"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Write JSON blame result to path (v0.66.0)",
    ),
):
    """Attribute weight movement to dataset rows via DataInf influence (v0.66.0).

    v0.57.0 shipped the plan-only surface. v0.66.0 (closes #171) lifts the
    runner with a DataInf-style influence-function approximation. Pass
    ``--plan-only`` to inspect the plan without running the analysis.
    """
    from soup_cli.utils.blame import (
        parse_budget,
        plan_blame,
        render_blame_json,
        render_blame_markdown,
        run_blame,
    )

    try:
        seconds = parse_budget(budget)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid --budget: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        plan = plan_blame(
            adapter_dir, dataset,
            layer=layer, budget_seconds=seconds, num_shards=num_shards,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    table = Table(title=f"Blame plan: {escape(plan.layer)}")
    table.add_column("Shard", justify="right")
    table.add_column("Hold-out offset", justify="right")
    table.add_column("Hold-out rows", justify="right")
    table.add_column("Projected seconds", justify="right")
    for shard in plan.shards:
        table.add_row(
            str(shard.shard_id),
            str(shard.holdout_offset),
            str(shard.holdout_size),
            str(shard.projected_seconds),
        )
    console.print(table)
    status_color = "green" if plan.feasible else "yellow"
    console.print(
        f"Budget {plan.budget_seconds}s, "
        f"{plan.per_shard_seconds}s/shard — "
        f"[{status_color}]{escape(plan.reason)}[/]"
    )

    if plan_only or not plan.feasible:
        return

    # v0.66.0: live runner via DataInf-style influence approximation.
    try:
        result = run_blame(plan, top_k=top_k)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Blame failed: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_blame_markdown(result))

    if output is not None:
        from soup_cli.utils.paths import atomic_write_text

        try:
            atomic_write_text(render_blame_json(result), output,
                              field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Cannot write --output: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote blame JSON → {escape(output)}[/]")


@app.command()
def branch(
    name: str = typer.Argument(..., help="Branch name (alphanumeric + ._-)"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to soup.yaml (required unless --from-registry)",
    ),
    base: Optional[str] = typer.Option(
        None, "--base", help="Base model id (required unless --from-registry)",
    ),
    dataset: str = typer.Option(None, "--dataset", help="Training dataset path (optional)"),
    attach_to_registry: Optional[str] = typer.Option(
        None, "--attach-to-registry",
        help=(
            "Registry entry id/ref to attach this branch pointer to as a "
            "branch_ref artifact + link via registry_entry_id (v0.71.4)."
        ),
    ),
    from_registry: Optional[str] = typer.Option(
        None, "--from-registry",
        help=(
            "Derive config + base_model + dataset hash from a Registry entry "
            "instead of --config/--base (v0.71.4)."
        ),
    ),
):
    """Snapshot a training environment as a comparable branch (v0.57.0).

    Pass ``--from-registry <id>`` to derive everything from a v0.26 Registry
    entry, or ``--attach-to-registry <id>`` to link a fresh snapshot into the
    lineage DAG (v0.71.4 #173).
    """
    from soup_cli.utils.adapter_branch import (
        attach_branch_to_registry,
        branch_from_registry,
        create_branch,
        load_branch,
    )

    try:
        if from_registry is not None:
            snap = branch_from_registry(name, from_registry)
        else:
            if not config:
                console.print(
                    "[red]--config is required (or use --from-registry)[/]"
                )
                raise typer.Exit(2)
            if not base:
                console.print(
                    "[red]--base is required (or use --from-registry)[/]"
                )
                raise typer.Exit(2)
            snap = create_branch(
                name, config_path=config, base_model=base, dataset_path=dataset,
            )
            if attach_to_registry is not None:
                attach_branch_to_registry(name, attach_to_registry)
                snap = load_branch(name)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    reg_line = (
        f"\nRegistry: [dim]{escape(snap.registry_entry_id)}[/]"
        if snap.registry_entry_id
        else ""
    )
    console.print(
        Panel(
            f"Name:    [bold]{escape(snap.name)}[/]\n"
            f"Base:    [bold]{escape(snap.base_model)}[/]\n"
            f"Config:  [bold]{escape(snap.config_path)}[/]\n"
            f"SHA:     [dim]{snap.config_sha256[:16]}...[/]\n"
            f"Dataset SHA: [dim]"
            f"{snap.dataset_sha256[:16] + '...' if snap.dataset_sha256 else '—'}[/]\n"
            f"Version: {snap.soup_version}{reg_line}",
            title="Adapter branch",
        )
    )


@app.command()
def checkout(
    name: str = typer.Argument(..., help="Branch name to check out"),
    output: str = typer.Option("soup.yaml", "--output", "-o",
                               help="Where to write the restored config"),
):
    """Restore a snapshotted branch's config into cwd (v0.57.0)."""
    from soup_cli.utils.adapter_branch import load_branch, write_checkout

    try:
        snap = load_branch(name)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        target = write_checkout(snap, output)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        f"[green]Restored branch {escape(snap.name)} → {escape(str(target))}[/]"
    )


@app.command(name="branches")
def list_branches_cmd():
    """List all snapshotted branches (v0.57.0)."""
    from soup_cli.utils.adapter_branch import list_branches, load_branch

    names = list_branches()
    if not names:
        console.print("[dim]No branches yet.[/]")
        return

    table = Table(title=f"Adapter branches ({len(names)})")
    table.add_column("Name", style="bold")
    table.add_column("Base model")
    table.add_column("Config SHA")
    table.add_column("Data SHA")
    for branch_name in names:
        try:
            snap = load_branch(branch_name)
            ds = snap.dataset_sha256[:8] + "..." if snap.dataset_sha256 else "—"
            table.add_row(
                escape(snap.name),
                escape(snap.base_model),
                snap.config_sha256[:8] + "...",
                ds,
            )
        except (ValueError, FileNotFoundError, OSError):
            table.add_row(escape(branch_name), "[red]error[/]", "-", "-")
    console.print(table)


@app.command()
def scan(
    adapter: str = typer.Argument(..., help="Path to adapter directory"),
    output_format: str = typer.Option(
        "text", "--format",
        help="Output format: text | json",
    ),
):
    """Spectral backdoor scan over LoRA adapter weights (v0.60.0).

    Flags rank-1 dominance, energy concentration, NaN/Inf, and Frobenius-norm
    outliers. Exit codes: 0=OK, 1=WARN, 3=FAIL. Failure means a likely
    backdoor pattern and ``adapters merge`` will refuse this adapter unless
    ``--allow-unscanned`` is passed.
    """
    from soup_cli.utils.adapter_scan import render_report_text, scan_adapter

    fmt = output_format.lower()
    if fmt not in ("text", "json"):
        console.print(f"[red]Unknown --format: {escape(fmt)}[/]")
        raise typer.Exit(2)

    try:
        report = scan_adapter(adapter)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    if fmt == "json":
        from dataclasses import asdict

        payload = {
            "adapter": report.adapter,
            "overall": report.overall,
            "summary": report.summary,
            "findings": [asdict(f) for f in report.findings],
        }
        console.print_json(data=payload)
    else:
        console.print(escape(render_report_text(report)))

    if report.overall == "FAIL":
        raise typer.Exit(3)
    if report.overall == "WARN":
        raise typer.Exit(1)


@app.command()
def sign(
    adapter: str = typer.Argument(..., help="Path to adapter directory"),
    backend: str = typer.Option(
        "unsigned", "--backend",
        help="Signing backend: unsigned | ed25519 (sigstore is infra-blocked)",
    ),
    key: Optional[str] = typer.Option(
        None, "--key",
        help="ed25519 private-key PEM path (or set SOUP_SIGNING_KEY).",
    ),
    generate_key: Optional[str] = typer.Option(
        None, "--generate-key",
        help="Generate a fresh ed25519 keypair, persist the private key here "
             "(PEM, 0600), and sign with it.",
    ),
):
    """Compute manifest + write ``.soup-signature.json`` (v0.60.0, ed25519 v0.71.2).

    ``unsigned`` (default) gives offline tamper detection via a Merkle-root
    hash. ``ed25519`` (v0.71.2 #185) adds a real detached signature over that
    root — pass ``--key <priv.pem>``, set ``SOUP_SIGNING_KEY``, or use
    ``--generate-key <out.pem>``. ``sigstore`` keyless signing is infra-blocked
    (needs an OIDC identity provider + Fulcio/Rekor network).
    """
    from soup_cli.utils.adapter_sign import sign_adapter

    try:
        record = sign_adapter(
            adapter, backend=backend, key_path=key, generate_key_path=generate_key
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except NotImplementedError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if generate_key and record.backend == "ed25519":
        console.print(
            f"[yellow]Generated ed25519 private key -> {escape(generate_key)} "
            "(keep it secret; verify with the matching public key).[/]"
        )

    console.print(
        Panel(
            f"Adapter:    [bold]{escape(record.manifest.adapter)}[/]\n"
            f"Backend:    [bold]{escape(record.backend)}[/]\n"
            f"Files:      [bold]{len(record.manifest.files)}[/]\n"
            f"Merkle:     [dim]{record.merkle_root[:16]}...[/]\n"
            f"Signed at:  [dim]{escape(record.signed_at)}[/]",
            title="Adapter signed",
        )
    )


@app.command()
def verify(
    adapter: str = typer.Argument(..., help="Path to adapter directory"),
    strict: bool = typer.Option(
        False, "--strict",
        help="Exit 3 on any verification failure (CI-friendly)",
    ),
    public_key: Optional[str] = typer.Option(
        None, "--public-key",
        help="Trusted ed25519 public-key PEM. When set, the embedded signing "
             "key must match it (genuine authentication, not just consistency).",
    ),
):
    """Verify ``.soup-signature.json`` against current files (v0.60.0, ed25519 v0.71.2).

    For ``ed25519``-signed adapters the detached signature is verified against
    the embedded public key (self-consistency); pass ``--public-key
    <trusted.pem>`` to additionally require the signer's key match a key you
    trust out of band.

    Exit codes:
      0  signature present and matches
      1  signature absent or mismatch (lenient mode)
      3  signature absent or mismatch with --strict
    """
    from soup_cli.utils.adapter_sign import verify_adapter

    try:
        report = verify_adapter(
            adapter, strict=strict, trusted_public_key=public_key
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        # Strict mode raises; non-strict gives a report.
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(3) from exc
    except TypeError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    status_color = "green" if report.valid else "yellow"
    panel = Panel(
        f"Adapter:    [bold]{escape(report.adapter)}[/]\n"
        f"Valid:      [{status_color}]{report.valid}[/]\n"
        f"Backend:    [bold]{escape(report.backend or '—')}[/]\n"
        f"Reason:     {escape(report.reason)}",
        title="Adapter verify",
    )
    console.print(panel)
    if report.findings:
        for finding in report.findings:
            console.print(f"  [yellow]- {escape(finding)}[/]")

    if not report.valid:
        raise typer.Exit(1)


@app.command(name="check-safetensors")
def check_safetensors(
    adapter: str = typer.Argument(..., help="Path to adapter / model directory"),
    strict: bool = typer.Option(
        False, "--strict",
        help="Exit 3 on any unsafe (pickle / PyTorch-classic) weight file",
    ),
):
    """Refuse pickle / PyTorch-classic weights at the boundary (v0.60.0).

    Exit codes:
      0  all weights are safetensors
      1  unsafe weights found (lenient mode — advisory)
      3  unsafe weights found (--strict — CI gate)
    """
    from soup_cli.utils.strict_safetensors import check_strict_safetensors

    try:
        report = check_strict_safetensors(adapter, strict=strict)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        # Strict mode raises with a friendly file-name advisory.
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(3) from exc
    except TypeError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if report.ok:
        console.print(
            Panel(
                f"Adapter:    [bold]{escape(report.model_dir)}[/]\n"
                f"Status:     [green]OK[/]\n"
                f"Reason:     {escape(report.reason)}",
                title="Strict safetensors check",
            )
        )
        return

    console.print(
        Panel(
            f"Adapter:    [bold]{escape(report.model_dir)}[/]\n"
            f"Status:     [yellow]UNSAFE[/]\n"
            f"Reason:     {escape(report.reason)}",
            title="Strict safetensors check",
        )
    )
    for path in report.unsafe_files:
        console.print(f"  [yellow]- {escape(path)}[/]")
    raise typer.Exit(1)


@app.command(name="pr")
def adapter_pr(
    title: str = typer.Argument(..., help="Short PR title (e.g. 'add-support-tone')"),
    base_sha: str = typer.Option(..., "--base-sha", help="64-hex SHA of the base model"),
    adapter_path: str = typer.Option(..., "--adapter", help="Path to candidate adapter"),
    eval_json: Optional[str] = typer.Option(
        None, "--eval",
        help=(
            "Path to JSON eval-deltas: list of "
            '{"metric": ..., "baseline": ..., "candidate": ...}'
        ),
    ),
    samples_json: Optional[str] = typer.Option(
        None, "--samples",
        help=(
            "Path to JSON sample diffs: list of "
            '{"prompt": ..., "baseline_output": ..., "candidate_output": ...}'
        ),
    ),
    dataset_diff_path: Optional[str] = typer.Option(
        None, "--dataset-diff",
        help="Path to a text file containing the dataset diff",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Write rendered Markdown to path (default: stdout)",
    ),
    format_: str = typer.Option(
        "markdown", "--format", "-f", help="markdown | json",
    ),
    push: Optional[str] = typer.Option(
        None, "--push",
        help=(
            "Post the rendered Markdown as a comment on a GitHub PR: "
            "owner/repo#N (e.g. MakazhanAlpamys/Soup#42). Auth via "
            "GITHUB_TOKEN / GH_TOKEN (v0.71.4)."
        ),
    ),
):
    """Render a GitHub-style PR for an adapter (v0.67.0 Part D).

    The PR = ``{base SHA, dataset diff, adapter file, eval report}``
    rendered as a Markdown document with eval-delta tables and sample
    diffs, ready to drop into a GitHub PR description or comment. Pass
    ``--push owner/repo#N`` to post it directly as a PR comment (v0.71.4).
    """
    from soup_cli.utils.adapter_pr import (
        build_adapter_pr,
        post_pr_comment,
        render_pr_json,
        render_pr_markdown,
        write_pr_markdown,
    )
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if format_ not in ("markdown", "json"):
        console.print(f"[red]Unknown --format: {escape(format_)}[/]")
        raise typer.Exit(2)

    # Load deltas + samples + dataset_diff lazily; each accepts None.
    pr_input_cap = 8 * 1024 * 1024  # 8 MiB per --eval/--samples/--dataset-diff

    def _load_json_list(path: Optional[str], field: str) -> list:
        if path is None:
            return []
        enforce_under_cwd_and_no_symlink(path, field=field)
        real = os.path.realpath(path)
        if os.path.getsize(real) > pr_input_cap:
            raise ValueError(f"{field} exceeds 8 MiB cap")
        with open(real, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(f"{field} must contain a JSON list")
        return raw

    try:
        deltas = _load_json_list(eval_json, "eval")
        samples = _load_json_list(samples_json, "samples")
        dataset_diff = ""
        if dataset_diff_path is not None:
            enforce_under_cwd_and_no_symlink(
                dataset_diff_path, field="dataset_diff"
            )
            real_diff = os.path.realpath(dataset_diff_path)
            if os.path.getsize(real_diff) > pr_input_cap:
                raise ValueError("dataset_diff exceeds 8 MiB cap")
            with open(real_diff, encoding="utf-8") as fh:
                dataset_diff = fh.read()
        pr = build_adapter_pr(
            title=title,
            base_sha=base_sha,
            adapter_path=adapter_path,
            dataset_diff=dataset_diff,
            deltas=deltas,
            samples=samples,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if format_ == "json":
        rendered = render_pr_json(pr)
    else:
        rendered = render_pr_markdown(pr)

    # v0.71.4 #223 — publish the rendered Markdown as a GitHub PR comment.
    # Always posts Markdown (a JSON blob is not a useful PR comment), even
    # when --format json was chosen for the local render/write.
    if push is not None:
        try:
            url = post_pr_comment(push, render_pr_markdown(pr))
        except RuntimeError as exc:
            # Missing token / gh failure — user-actionable, exit 1.
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(
            f"[green]Posted PR comment to {escape(push)}[/]"
            + (f" -> {escape(url)}" if url else "")
        )
        if output is None:
            return

    if output is None:
        console.print(rendered)
        return

    try:
        if format_ == "markdown":
            write_pr_markdown(pr, output)
        else:
            from soup_cli.utils.paths import atomic_write_text

            atomic_write_text(rendered, output, field="pr output")
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"PR:     [bold]{escape(pr.title)}[/]\n"
            f"Format: [bold]{escape(format_)}[/]\n"
            f"Output: [bold]{escape(output)}[/]",
            title="Adapter PR rendered",
        )
    )


@app.command(name="bisect")
def adapter_bisect(
    history: list[str] = typer.Argument(
        ...,
        help=(
            "Ordered checkpoint identifiers (oldest first). At least 2. "
            "Example: `soup adapters bisect ckpt-100 ckpt-200 ckpt-300 ckpt-400`"
        ),
    ),
    eval_command: str = typer.Option(
        ...,
        "--eval-command",
        help=(
            "Shell command template; '{ckpt}' is replaced with each "
            "checkpoint id. Exit code 0 means PASS, non-zero means FAIL. "
            "Example: 'soup eval custom --model {ckpt} --tasks tasks.jsonl'"
        ),
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help="Print the plan and exit without running the bisect",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Write JSON result to path",
    ),
):
    """Binary-search a training history to find the first failing checkpoint.

    Composes with v0.66 Part B influence-blame: once the boundary is
    found, ``soup adapters blame`` can attribute the regression to
    specific dataset rows.
    """
    from soup_cli.utils.adapter_bisect import build_bisect_plan, run_bisect

    if not isinstance(history, list) or len(history) < 2:
        console.print("[red]Need at least 2 checkpoint ids[/]")
        raise typer.Exit(2)

    try:
        plan = build_bisect_plan(history)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid history: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if "{ckpt}" not in eval_command:
        console.print(
            "[red]--eval-command must include `{ckpt}` placeholder[/]"
        )
        raise typer.Exit(2)

    if plan_only:
        console.print(
            Panel(
                f"History:  [bold]{len(plan.history)} checkpoints[/]\n"
                f"Eval:     [bold]{escape(eval_command)}[/]\n"
                f"Probes:   ~[bold]{_estimated_probes(len(plan.history))}[/] iterations",
                title="Adapter bisect (plan)",
            )
        )
        return

    # Execute via subprocess; exit 0 means OK.
    import shlex
    import subprocess  # noqa: S404 — argv list mode, no shell

    def _eval_fn(ckpt: str) -> bool:
        cmd_str = eval_command.replace("{ckpt}", shlex.quote(ckpt))
        # Safer: split via shlex and use argv mode (no shell interpolation).
        argv = shlex.split(cmd_str)
        try:
            result = subprocess.run(  # noqa: S603
                argv, capture_output=True, timeout=3600, check=False
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            console.print(
                f"[yellow]Eval failed for {escape(ckpt)}: "
                f"{escape(str(exc))}[/]"
            )
            return False
        return result.returncode == 0

    try:
        result = run_bisect(plan, eval_fn=_eval_fn)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if result.verdict == "ALL_OK":
        console.print(
            Panel(
                f"Verdict: [green]ALL_OK[/]\n"
                f"Probes:  {result.probes}",
                title="Adapter bisect",
            )
        )
    else:
        console.print(
            Panel(
                f"Verdict:      [red]BROKEN_AT[/]\n"
                f"First broken: [bold]{escape(result.first_broken or '?')}[/]\n"
                f"Probes:       {result.probes}",
                title="Adapter bisect",
            )
        )

    if output is not None:
        from soup_cli.utils.paths import atomic_write_text

        data = {
            "verdict": result.verdict,
            "first_broken": result.first_broken,
            "probes": result.probes,
            "steps": [
                {"checkpoint": s.checkpoint, "ok": s.ok}
                for s in result.steps
            ],
        }
        atomic_write_text(
            json.dumps(data, indent=2, sort_keys=True),
            output,
            field="bisect output",
        )

    if result.verdict == "BROKEN_AT":
        raise typer.Exit(2)


def _estimated_probes(n: int) -> int:
    """Approximate number of probes for n checkpoints (~log2(n) + 2)."""
    import math

    if n <= 2:
        return n
    return max(2, math.ceil(math.log2(n))) + 2
