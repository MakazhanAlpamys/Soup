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
                "Path to prompts file (JSONL or plain text, one per line). "
                "Must be provided: give --prompts to specify representative evaluation prompts."
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

        if prompts is None:
            console.print(
                "[red]Missing required evaluation prompts: give --prompts with a path to "
                "a JSONL or plain-text prompts file.[/]"
            )
            raise typer.Exit(EXIT_USAGE_ERROR)

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
        try:
            prompt_list = load_prompts_file(prompts)
        except (ValueError, OSError) as exc:
            console.print(
                f"[red]Cannot read --prompts '{for_terminal(prompts)}': "
                f"{for_terminal(str(exc))}[/]"
            )
            raise typer.Exit(EXIT_USAGE_ERROR) from exc

        # --- Probe model loading once to catch errors before sequential evaluation ---
        is_stub = False
        _probe_error: Optional[Exception] = None

        try:
            from soup_cli.eval.export_check import (  # noqa: F401 – import probe only
                make_hf_adapter_evaluator,
                make_hf_evaluator,
            )
        except ImportError as exc:
            _probe_error = exc

        if _probe_error is not None:
            if not allow_stub:
                console.print(
                    f"[red]Failed to load models: {for_terminal(str(_probe_error))}\n"
                    "Pass --allow-stub to exercise rendering with deterministic stubs "
                    "(stubs always fail; they do not verify real fidelity).[/]"
                )
                raise typer.Exit(EXIT_RUNTIME_ERROR) from _probe_error
            if fmt == "table":
                console.print(
                    f"[yellow]Model loading failed ({for_terminal(str(_probe_error))}); "
                    "using deterministic stubs "
                    "because --allow-stub was passed.  "
                    "This does NOT verify real export fidelity and will fail the gate.[/]"
                )
            is_stub = True

        # --- Build factories for sequential loading ---
        from soup_cli.eval.export_check import (
            FunctionModelEvaluator,
            HFCausalLMEvaluator,
            make_hf_adapter_evaluator,
            make_hf_evaluator,
            run_export_check_from_factories,
        )

        if is_stub:
            # Stub path: factories return deterministic FunctionModelEvaluator stubs
            def _base_factory_stub() -> FunctionModelEvaluator:
                return FunctionModelEvaluator(stub_generator("base"))

            def _ref_factory_stub() -> FunctionModelEvaluator:
                return FunctionModelEvaluator(stub_generator("ref"))

            def _exp_factory_stub() -> FunctionModelEvaluator:
                return FunctionModelEvaluator(stub_generator("exported"))

            base_factory = _base_factory_stub
            ref_factory = _ref_factory_stub
            exported_factory = _exp_factory_stub
        else:
            _resolved_base_cap = resolved_base
            _adapter_path_str = str(adapter_path)
            _exported_path_str = str(exported_path)
            _fmt_cap = fmt
            _console_cap = console

            def _base_factory() -> HFCausalLMEvaluator:
                if _fmt_cap == "table":
                    _console_cap.print("[dim]Loading base model...[/]")
                return make_hf_evaluator(_resolved_base_cap)

            def _ref_factory() -> HFCausalLMEvaluator:
                if _fmt_cap == "table":
                    _console_cap.print("[dim]Loading reference model (adapter)...[/]")
                return make_hf_adapter_evaluator(_adapter_path_str, _resolved_base_cap)

            def _exp_factory() -> HFCausalLMEvaluator:
                if _fmt_cap == "table":
                    _console_cap.print("[dim]Loading exported model...[/]")
                return make_hf_evaluator(_exported_path_str)

            base_factory = _base_factory
            ref_factory = _ref_factory
            exported_factory = _exp_factory

        # --- Run comparison (sequential: one model in memory at a time) ---
        try:
            result = run_export_check_from_factories(
                base_factory=base_factory,
                ref_factory=ref_factory,
                exported_factory=exported_factory,
                prompts=prompt_list,
                min_agreement=min_agreement,
                stub=is_stub,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            # Model loading failed inside a factory (e.g. missing weights, bad path).
            if not allow_stub:
                console.print(
                    f"[red]Failed to load models: {for_terminal(str(exc))}\n"
                    "Pass --allow-stub to exercise rendering with deterministic stubs "
                    "(stubs always fail; they do not verify real fidelity).[/]"
                )
                raise typer.Exit(EXIT_RUNTIME_ERROR) from exc
            # Retry with stubs
            if fmt == "table":
                console.print(
                    f"[yellow]Model loading failed ({for_terminal(str(exc))}); "
                    "using deterministic stubs "
                    "because --allow-stub was passed.  "
                    "This does NOT verify real export fidelity and will fail the gate.[/]"
                )
            is_stub = True
            stub_base = FunctionModelEvaluator(stub_generator("base"))
            stub_ref = FunctionModelEvaluator(stub_generator("ref"))
            stub_exp = FunctionModelEvaluator(stub_generator("exported"))
            try:
                result = run_export_check_from_factories(
                    base_factory=lambda: stub_base,
                    ref_factory=lambda: stub_ref,
                    exported_factory=lambda: stub_exp,
                    prompts=prompt_list,
                    min_agreement=min_agreement,
                    stub=True,
                )
            except Exception as exc2:  # noqa: BLE001
                console.print(
                    f"[red]Export check failed unexpectedly: {for_terminal(str(exc2))}[/]"
                )
                raise typer.Exit(EXIT_RUNTIME_ERROR) from exc2
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
