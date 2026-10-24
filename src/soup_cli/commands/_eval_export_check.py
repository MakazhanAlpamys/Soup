"""v0.76.x #2048 — `soup eval export-check`.

Verifies that an exported model artifact (merged safetensors directory)
still reproduces the behavioral delta of the LoRA adapter it came from.

Attached to the existing ``soup eval`` Typer app via :func:`register`,
following the v0.55.0 / v0.61.0 / v0.65.0 / v0.71.10 pattern so
``commands/eval.py`` stays under length cap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from soup_cli.utils.exit_codes import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
    GateCommand,
)
from soup_cli.utils.terminal import for_terminal


def register(app: typer.Typer, console: Console) -> None:
    """Attach ``soup eval export-check`` to *app*."""

    @app.command(name="export-check", cls=GateCommand)
    def export_check_cmd(
        adapter: str = typer.Option(
            ...,
            "--adapter",
            help=(
                "Path to the LoRA adapter directory used as the reference. "
                "Its base model is auto-detected from adapter_config.json "
                "unless --base is supplied."
            ),
        ),
        exported: str = typer.Option(
            ...,
            "--exported",
            help=(
                "Path to the exported artifact to test — a merged safetensors "
                "directory (from 'soup merge'). Standalone GGUF, ONNX, and MLX "
                "fused directories are refused with a diagnostic."
            ),
        ),
        base: Optional[str] = typer.Option(
            None,
            "--base",
            help=(
                "Base model path or HuggingFace Hub id. "
                "Auto-detected from adapter_config.json when absent."
            ),
        ),
        prompts: Optional[str] = typer.Option(
            None,
            "--prompts",
            help=(
                "Optional JSONL or plain-text file of prompts (one per line, "
                "or JSONL rows with a 'prompt' key). "
                "Defaults to a built-in set of generic probes when absent."
            ),
        ),
        min_agreement: float = typer.Option(
            0.80,
            "--min-agreement",
            min=0.0,
            max=1.0,
            help=(
                "Minimum required average character-overlap agreement between "
                "the exported model and adapter reference on prompts where the "
                "adapter changes behavior. Range [0.0, 1.0]. Default 0.80."
            ),
        ),
        fmt: str = typer.Option(
            "table",
            "--format",
            help="Output format: table | json | markdown",
        ),
        allow_stub: bool = typer.Option(
            False,
            "--allow-stub",
            help=(
                "Use deterministic stub generators if live model loading fails "
                "(for CI smoke-testing only; stubs do not verify fidelity and fail the gate)."
            ),
        ),
    ) -> None:
        """Verify that an exported model preserved the adapter's behavioral delta.

        Compares greedy outputs of base, base+adapter (reference), and the
        exported artifact on the same prompts.  Exits 0 when the delta
        survived, 2 when it was lost or agreement fell below --min-agreement,
        and 3 on input validation errors.
        """
        from soup_cli.eval.export_check import (
            classify_artifact,
            ensure_format,
            load_prompts_file,
            render,
            run_export_check,
            stub_generator,
        )
        from soup_cli.utils.paths import is_under_cwd

        # --- Validate format ---
        try:
            ensure_format(fmt)
        except ValueError as exc:
            console.print(f"[red]{for_terminal(str(exc))}[/]")
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # --- Validate paths within cwd ---
        adapter_path = Path(adapter)
        exported_path = Path(exported)

        for label, path_obj in (("--adapter", adapter_path), ("--exported", exported_path)):
            if not is_under_cwd(path_obj):
                console.print(
                    f"[red]{label} '{for_terminal(str(path_obj))}' is outside the "
                    "current working directory — refusing.[/]"
                )
                raise typer.Exit(EXIT_USAGE_ERROR)
            if not path_obj.exists():
                console.print(
                    f"[red]{label} not found: {for_terminal(str(path_obj))}[/]"
                )
                raise typer.Exit(EXIT_USAGE_ERROR)

        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            console.print(
                "[red]--adapter must be a LoRA adapter directory containing "
                "adapter_config.json.[/]"
            )
            raise typer.Exit(EXIT_USAGE_ERROR)

        if prompts is not None:
            prompts_path = Path(prompts)
            if not is_under_cwd(prompts_path):
                console.print(
                    f"[red]--prompts '{for_terminal(prompts)}' is outside the "
                    "current working directory — refusing.[/]"
                )
                raise typer.Exit(EXIT_USAGE_ERROR)
            if not prompts_path.exists():
                console.print(f"[red]--prompts not found: {for_terminal(prompts)}[/]")
                raise typer.Exit(EXIT_USAGE_ERROR)

        # --- Validate exported artifact format ---
        try:
            classify_artifact(exported_path)
        except ValueError as exc:
            console.print(f"[red]{for_terminal(str(exc))}[/]")
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # --- Detect base model ---
        resolved_base: Optional[str] = base
        if resolved_base is None:
            adapter_config = adapter_path / "adapter_config.json"
            if adapter_config.exists():
                import json

                try:
                    with open(adapter_config, encoding="utf-8") as fh:
                        cfg = json.load(fh)
                    resolved_base = cfg.get("base_model_name_or_path")
                except (json.JSONDecodeError, OSError):
                    pass

        if not resolved_base:
            console.print(
                "[red]Cannot detect base model. "
                "Supply --base or ensure adapter_config.json contains "
                "'base_model_name_or_path'.[/]"
            )
            raise typer.Exit(EXIT_USAGE_ERROR)

        # --- Load prompts ---
        prompt_list: Optional[list[str]] = None
        if prompts is not None:
            try:
                prompt_list = load_prompts_file(prompts)
            except (ValueError, OSError) as exc:
                console.print(
                    f"[red]Cannot read --prompts '{for_terminal(prompts)}': "
                    f"{for_terminal(str(exc))}[/]"
                )
                raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # --- Build generators ---
        is_stub = False
        base_gen = ref_gen = exported_gen = None

        try:
            from soup_cli.eval.export_check import (
                make_hf_adapter_generator,
                make_hf_generator,
            )

            if fmt == "table":
                console.print("[dim]Loading base model...[/]")
            base_gen = make_hf_generator(resolved_base)
            if fmt == "table":
                console.print("[dim]Loading reference model (adapter)...[/]")
            ref_gen = make_hf_adapter_generator(str(adapter_path), resolved_base)
            if fmt == "table":
                console.print("[dim]Loading exported model...[/]")
            exported_gen = make_hf_generator(str(exported_path))
            if fmt == "table":
                console.print("[green]All three models loaded.[/]\n")

        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            if not allow_stub:
                console.print(
                    f"[red]Failed to load models: {for_terminal(str(exc))}\n"
                    "Pass --allow-stub to exercise rendering with deterministic stubs "
                    "(stubs always fail; they do not verify real fidelity).[/]"
                )
                raise typer.Exit(EXIT_RUNTIME_ERROR) from exc
            if fmt == "table":
                console.print(
                    f"[yellow]Model loading failed ({for_terminal(str(exc))}); "
                    "using deterministic stubs "
                    "because --allow-stub was passed.  "
                    "This does NOT verify real export fidelity and will fail the gate.[/]"
                )
            is_stub = True
            base_gen = stub_generator("base")
            ref_gen = stub_generator("ref")
            exported_gen = stub_generator("exported")

        # --- Run comparison ---
        try:
            result = run_export_check(
                base_gen=base_gen,
                ref_gen=ref_gen,
                exported_gen=exported_gen,
                prompts=prompt_list,
                min_agreement=min_agreement,
                stub=is_stub,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[red]Export check failed unexpectedly: {for_terminal(str(exc))}[/]"
            )
            raise typer.Exit(EXIT_RUNTIME_ERROR) from exc

        # --- Render output ---
        rendered = render(result, fmt=fmt)
        if isinstance(rendered, str):
            console.print(rendered, markup=False, highlight=False, soft_wrap=True)
        else:
            console.print(rendered)

        # --- Summary panel (table format only) ---
        if fmt == "table":
            if result.passed:
                summary = "[green]EXPORT CHECK PASSED[/] — adapter delta preserved."
            elif result.active_prompt_count == 0:
                summary = (
                    "[red]EXPORT CHECK FAILED[/] — the adapter changed none of the "
                    "tested prompts, so export fidelity cannot be verified. "
                    "Supply adapter-specific --prompts."
                )
            elif result.delta_lost:
                summary = "[red]EXPORT CHECK FAILED[/] — adapter delta lost on at least one prompt."
            else:
                summary = (
                    f"[red]EXPORT CHECK FAILED[/] — "
                    f"overall agreement {result.overall_agreement:.2f} "
                    f"< required {min_agreement:.2f}."
                )
            console.print(
                Panel(summary, border_style="green" if result.passed else "red")
            )

        raise typer.Exit(EXIT_OK if result.passed and not is_stub else EXIT_GATE_FAILED)
