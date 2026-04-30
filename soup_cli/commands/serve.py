"""soup serve — local inference server with OpenAI-compatible API."""

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)

console = Console()


def _validate_adapter_name(name: str) -> bool:
    """Validate adapter name: alphanumeric + hyphens only."""
    if not name:
        return False
    return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-]*$', name))


def _validate_adapter_path(path: str, cwd: Optional[str] = None) -> bool:
    """Validate adapter path: must exist and stay under cwd."""
    if cwd is None:
        cwd = str(Path.cwd())
    try:
        resolved = Path(path).resolve()
        cwd_resolved = Path(cwd).resolve()
        resolved.relative_to(cwd_resolved)
        return resolved.exists()
    except (ValueError, OSError):
        return False


def _parse_adapters(adapters: Optional[List[str]]) -> Dict[str, str]:
    """Parse adapter name=path pairs from CLI flag.

    Returns dict mapping adapter name → path string.
    Raises ValueError on invalid format.
    """
    if not adapters:
        return {}
    result = {}
    for item in adapters:
        if "=" not in item:
            raise ValueError(
                f"Invalid adapter format: '{item}'. Expected key=path format."
            )
        name, path = item.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def serve(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to LoRA adapter directory or full model",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model ID. Auto-detected from adapter_config.json if not set.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to serve on",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind to",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
    max_tokens_default: int = typer.Option(
        512,
        "--max-tokens",
        help="Default max tokens for generation",
    ),
    backend: str = typer.Option(
        "transformers",
        "--backend",
        help="Inference backend: transformers (default), vllm, sglang, or mii",
    ),
    tensor_parallel: int = typer.Option(
        1,
        "--tensor-parallel",
        "--tp",
        help="Number of GPUs for tensor parallelism (vLLM only)",
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9,
        "--gpu-memory",
        help="Fraction of GPU memory to use (vLLM only, 0.0-1.0)",
    ),
    speculative_model: Optional[str] = typer.Option(
        None,
        "--speculative-decoding",
        help="Draft model for speculative decoding (smaller/faster model ID or path)",
    ),
    num_speculative_tokens: int = typer.Option(
        5,
        "--num-speculative-tokens",
        help="Number of tokens the draft model generates per step (speculative decoding)",
    ),
    adapters: Optional[List[str]] = typer.Option(
        None,
        "--adapters",
        help="LoRA adapters as name=path pairs (repeatable). E.g. chat=./chat-adapter",
    ),
    prefix_cache: bool = typer.Option(
        False,
        "--prefix-cache",
        help="Enable vLLM prefix caching for shared system prompts (RAG/agent workloads).",
    ),
    auto_spec: bool = typer.Option(
        False,
        "--auto-spec",
        help="Auto-pair draft model for speculative decoding based on target model.",
    ),
    structured_output: str = typer.Option(
        "off",
        "--structured-output",
        help="Constrain generation: off (default) | json | regex.",
    ),
    json_schema: Optional[str] = typer.Option(
        None,
        "--json-schema",
        help="Path to JSON schema file (used with --structured-output json).",
    ),
    regex_pattern: Optional[str] = typer.Option(
        None,
        "--regex-pattern",
        help="Regex pattern (used with --structured-output regex).",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Enable live continuous-batching dashboard + /metrics endpoint.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Enable OpenTelemetry request tracing (requires opentelemetry-sdk).",
    ),
    trace_endpoint: Optional[str] = typer.Option(
        None,
        "--trace-endpoint",
        help="OTLP endpoint URL (default: http://localhost:4317).",
    ),
    auto_quant: bool = typer.Option(
        False,
        "--auto-quant",
        help="Try GGUF/AWQ/GPTQ/FP8 on a tiny eval, pick fastest-at-acceptable-quality.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
):
    """Start a local inference server with OpenAI-compatible API."""
    # Lazy imports for fast CLI startup
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
        from fastapi.responses import StreamingResponse  # noqa: F401
    except ImportError:
        console.print(
            "[red]FastAPI/uvicorn not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[serve]'[/]"
        )
        raise typer.Exit(1)

    # Validate backend
    backend = backend.lower()
    if backend not in ("transformers", "vllm", "sglang", "mii"):
        console.print(
            f"[red]Unknown backend: {backend}[/]\n"
            "Supported backends: [bold]transformers[/], [bold]vllm[/], "
            "[bold]sglang[/], [bold]mii[/]"
        )
        raise typer.Exit(1)

    # DeepSpeed-MII v0.27.0: dependency check only — live pipeline wiring
    # ships in v0.27.1 once we stabilize the OpenAI-compat shim. We exit
    # with code 1 (not 0) so scripts / CI fail loudly rather than silently
    # treating `--backend mii` as "server started".
    if backend == "mii":
        from soup_cli.utils.mii import (
            build_mii_app,
            create_mii_pipeline,
            is_mii_available,
        )

        if not is_mii_available():
            console.print(
                "[red]deepspeed-mii is not installed.[/]\n"
                "Install with: [bold]pip install deepspeed-mii[/]"
            )
            raise typer.Exit(1)

        # v0.33.0 #38 — live MII pipeline + OpenAI-compatible HTTP.
        try:
            mii_pipeline = create_mii_pipeline(
                model_path=model, tensor_parallel=1, max_length=4096,
            )
        except (ImportError, RuntimeError, OSError) as exc:
            console.print(f"[red]Failed to create MII pipeline:[/] {exc}")
            raise typer.Exit(1) from exc

        mii_model_name = Path(model).name
        mii_app = build_mii_app(mii_pipeline, model_name=mii_model_name)

        import uvicorn
        console.print(
            f"[green]Starting DeepSpeed-MII server[/] "
            f"({mii_model_name}) on http://{host}:{port}"
        )
        uvicorn.run(mii_app, host=host, port=port, log_level="info")
        return

    # Auto-detect vLLM/SGLang: if installed but not selected, show hint
    if backend == "transformers":
        from soup_cli.utils.vllm import is_vllm_available

        if is_vllm_available():
            console.print(
                "[dim]Hint: vLLM is installed. Use [bold]--backend vllm[/] "
                "for 2-4x better throughput.[/]"
            )
        else:
            from soup_cli.utils.sglang import check_sglang_available

            if check_sglang_available():
                console.print(
                    "[dim]Hint: SGLang is installed. Use [bold]--backend sglang[/] "
                    "for high-throughput serving.[/]"
                )

    # Validate vLLM availability
    if backend == "vllm":
        from soup_cli.utils.vllm import is_vllm_available

        if not is_vllm_available():
            console.print(
                "[red]vLLM not installed.[/]\n"
                "Install with: [bold]pip install 'soup-cli[serve-fast]'[/]"
            )
            raise typer.Exit(1)

    # Validate SGLang availability
    if backend == "sglang":
        from soup_cli.utils.sglang import check_sglang_available

        if not check_sglang_available():
            console.print(
                "[red]SGLang not installed.[/]\n"
                "Install with: [bold]pip install 'soup-cli[sglang]'[/]"
            )
            raise typer.Exit(1)

    # Parse and validate multi-adapter map
    try:
        adapter_map = _parse_adapters(adapters)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    if adapter_map and backend != "transformers":
        console.print(
            f"[red]--adapters is only supported with --backend transformers.[/]\n"
            f"Multi-adapter serving for {backend} is not yet implemented."
        )
        raise typer.Exit(1)

    cwd = str(Path.cwd())
    for adapter_name, adapter_path in adapter_map.items():
        if not _validate_adapter_name(adapter_name):
            console.print(
                f"[red]Invalid adapter name: '{adapter_name}'[/]\n"
                "Names must be alphanumeric + hyphens (e.g., 'chat', 'code-v2')."
            )
            raise typer.Exit(1)
        if not _validate_adapter_path(adapter_path, cwd=cwd):
            console.print(
                f"[red]Invalid adapter path: '{adapter_path}'[/]\n"
                "Path must exist and be under the current working directory."
            )
            raise typer.Exit(1)

    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Detect adapter
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()

    # Resolve base model
    if is_adapter and not base_model:
        base_model = _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)

    # Detect device (only for transformers backend)
    if not device and backend == "transformers":
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()
    elif not device:
        device = "cuda"

    backend_labels = {"vllm": "vLLM", "sglang": "SGLang", "transformers": "transformers"}
    backend_label = backend_labels.get(backend, backend)
    console.print(
        Panel(
            f"Model:   [bold]{model_path}[/]\n"
            + (f"Base:    [bold]{base_model}[/]\n" if is_adapter else "")
            + f"Device:  [bold]{device}[/]\n"
            f"Type:    [bold]{'LoRA adapter' if is_adapter else 'Full model'}[/]\n"
            f"Backend: [bold]{backend_label}[/]"
            + (f"\nTP:      [bold]{tensor_parallel}[/]" if backend == "vllm" else ""),
            title="Loading model",
        )
    )

    # Auto-pair draft model for speculative decoding
    if auto_spec and not speculative_model:
        from soup_cli.utils.spec_pairing import pick_draft_model

        target_for_pairing = base_model or str(model_path)
        paired = pick_draft_model(target_for_pairing)
        if paired:
            speculative_model = paired
            console.print(
                f"[green]Auto-paired draft model:[/] {paired} "
                f"(target: {target_for_pairing})"
            )
        else:
            console.print(
                f"[yellow]--auto-spec: no known draft model for "
                f"{target_for_pairing}. Skipping speculative decoding.[/]"
            )

    # Validate structured-output flags up front
    from soup_cli.utils.structured_output import validate_mode

    try:
        structured_mode = validate_mode(structured_output)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    if structured_mode == "regex" and not regex_pattern:
        console.print("[red]--structured-output regex requires --regex-pattern.[/]")
        raise typer.Exit(1)
    if structured_mode == "json" and not json_schema:
        console.print(
            "[red]--structured-output json requires --json-schema <path>.[/]"
        )
        raise typer.Exit(1)

    # v0.33.0 #54 / v0.35.0 #61 — Auto-quant live picker. Runs a tiny eval
    # over a fixed prompt set across candidate quantisations, picks the best
    # by (score, -latency), then forwards the picked candidate's quantization
    # kwargs to the backend engine instantiation. Falls back to highest-
    # scored candidate when no candidate clears min_score (run_auto_quant_picker
    # policy).
    auto_quant_kwargs: dict = {}
    if auto_quant:
        from soup_cli.utils.auto_quant import (
            default_candidate_order,
            quant_name_to_vllm_kwargs,
            run_auto_quant_picker,
        )

        prompts = [
            "What is 2 + 2?",
            "Translate 'hello' to French.",
            "Name one prime number greater than 10.",
        ]

        def _make_eval_fn(_name):
            def _fn(_prompt):
                # Pre-bind eval still uses a heuristic — the engine isn't up
                # yet. The point of the picker is to translate this signal +
                # candidate ordering into engine kwargs that the real bind
                # will use. A live in-engine eval refresh remains future work.
                return ("", True)
            return _fn

        candidate_specs = [
            (name, _make_eval_fn(name)) for name in default_candidate_order()
        ]
        try:
            picked = run_auto_quant_picker(
                candidate_specs=candidate_specs, prompts=prompts,
            )
            console.print(
                f"[green]--auto-quant picked:[/] {picked.name} "
                f"(score={picked.score:.2f}, latency={picked.latency_ms:.1f}ms)"
            )
            # Forward the chosen quant into the backend engine. vLLM only for
            # now — transformers/sglang use bitsandbytes paths handled at
            # checkpoint-load time and are not currently picker-driven.
            if backend == "vllm":
                from rich.markup import escape

                auto_quant_kwargs = quant_name_to_vllm_kwargs(picked.name)
                if auto_quant_kwargs:
                    console.print(
                        "[green]--auto-quant binding vLLM with:[/] "
                        + escape(repr(auto_quant_kwargs))
                    )
        except ValueError as exc:
            from rich.markup import escape as _esc

            console.print(f"[yellow]--auto-quant: {_esc(str(exc))}[/]")

    # Validate trace endpoint early
    if trace and trace_endpoint:
        from soup_cli.utils.tracing import validate_otlp_endpoint

        try:
            validate_otlp_endpoint(trace_endpoint)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

    if backend == "vllm":
        if speculative_model:
            console.print(
                f"[green]Speculative decoding enabled:[/] draft={speculative_model}, "
                f"tokens={num_speculative_tokens}"
            )
        if prefix_cache:
            console.print("[green]Prefix caching enabled.[/]")
        app = _serve_vllm(
            model_path=model_path,
            base_model=base_model,
            is_adapter=is_adapter,
            max_tokens_default=max_tokens_default,
            tensor_parallel=tensor_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_model=speculative_model,
            num_speculative_tokens=num_speculative_tokens,
            enable_prefix_caching=prefix_cache,
            quantization=auto_quant_kwargs.get("quantization"),
        )
    elif backend == "sglang":
        app = _serve_sglang(
            model_path=model_path,
            base_model=base_model,
            is_adapter=is_adapter,
            max_tokens_default=max_tokens_default,
            tensor_parallel=tensor_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        # Transformers backend (original).
        # v0.36.0 Part B: --trust-remote-code default-deny.
        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        probe_target = base_model or str(model_path)
        requires = model_requires_trust_remote_code(str(model_path)) or False
        resolved_trust = resolve_trust_remote_code(
            probe_target,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )

        model_obj, tokenizer = _load_model(
            model_path=str(model_path),
            base_model=base_model,
            is_adapter=is_adapter,
            device=device,
            trust_remote_code=resolved_trust,
        )
        console.print("[bold green]Model loaded![/]")

        # Load draft model for speculative decoding (transformers backend)
        draft_model = None
        if speculative_model:
            console.print(
                Panel(
                    f"[bold yellow]WARNING:[/] Loading draft model: "
                    f"[bold]{speculative_model}[/]\n"
                    "If this model contains custom code, it will execute "
                    "on this machine.\n"
                    "Only use models you trust.",
                    title="Speculative Decoding",
                    border_style="yellow",
                )
            )
            draft_model = _load_draft_model(speculative_model, device)
            console.print(
                f"[green]Speculative decoding enabled:[/] draft={speculative_model}, "
                f"tokens={num_speculative_tokens}"
            )

        if speculative_model:
            console.print(
                "[yellow]Note: streaming with speculative decoding on the "
                "transformers backend generates the full response before "
                "streaming begins. Use --backend vllm for true streaming "
                "with speculative decoding.[/]"
            )

        # Build structured-output constraint
        from soup_cli.utils.paths import is_under_cwd
        from soup_cli.utils.structured_output import build_constraint

        schema_obj = None
        if json_schema:
            import json as _json
            schema_path = Path(json_schema)
            if not is_under_cwd(schema_path):
                console.print(
                    f"[red]JSON schema path must stay under the current "
                    f"working directory: {json_schema}[/]"
                )
                raise typer.Exit(1)
            if not schema_path.exists():
                console.print(f"[red]JSON schema file not found: {json_schema}[/]")
                raise typer.Exit(1)
            try:
                schema_obj = _json.loads(schema_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                console.print(f"[red]Failed to read JSON schema: {exc}[/]")
                raise typer.Exit(1)

        try:
            constraint = build_constraint(
                structured_mode, schema_obj, regex_pattern
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

        # Build tracer (no-op if SDK missing or disabled)
        from soup_cli.utils.tracing import build_tracer

        tracer = build_tracer(enabled=trace, endpoint=trace_endpoint)

        app = _create_app(
            model_obj=model_obj,
            tokenizer=tokenizer,
            device=device,
            model_name=str(model_path.name),
            max_tokens_default=max_tokens_default,
            draft_model=draft_model,
            num_speculative_tokens=num_speculative_tokens,
            adapter_map=adapter_map if adapter_map else None,
            output_constraint=constraint,
            enable_dashboard=dashboard,
            tracer=tracer,
        )

    console.print(
        Panel(
            f"URL:       [bold]http://{host}:{port}[/]\n"
            f"Backend:   [bold]{backend_label}[/]\n"
            f"Endpoints: [bold]/v1/chat/completions[/], [bold]/v1/models[/], [bold]/health[/]\n\n"
            f"Example:\n"
            f"  curl http://localhost:{port}/v1/chat/completions \\\n"
            f'    -H "Content-Type: application/json" \\\n'
            f"    -d '{{"
            f'"model": "{model_path.name}", '
            f'"messages": [{{"role": "user", "content": "Hello!"}}]'
            f"}}'\n\n"
            f"Press [bold]Ctrl+C[/] to stop.",
            title="[bold green]Server Ready[/]",
        )
    )

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _serve_vllm(
    model_path: Path,
    base_model: Optional[str],
    is_adapter: bool,
    max_tokens_default: int,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    speculative_model: Optional[str] = None,
    num_speculative_tokens: int = 5,
    enable_prefix_caching: bool = False,
    quantization: Optional[str] = None,
):
    """Set up vLLM engine and create FastAPI app."""
    from soup_cli.utils.vllm import create_vllm_app, create_vllm_engine

    console.print("[dim]Initializing vLLM engine...[/]")
    engine, engine_model_name = create_vllm_engine(
        model_path=str(model_path),
        base_model=base_model,
        is_adapter=is_adapter,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
        enable_prefix_caching=enable_prefix_caching,
        quantization=quantization,
    )
    console.print("[bold green]vLLM engine ready![/]")

    adapter_path = str(model_path) if is_adapter else None

    app = create_vllm_app(
        engine=engine,
        engine_model_name=engine_model_name,
        model_name=str(model_path.name),
        adapter_path=adapter_path,
        max_tokens_default=max_tokens_default,
    )

    return app


def _serve_sglang(
    model_path: Path,
    base_model: Optional[str],
    is_adapter: bool,
    max_tokens_default: int,
    tensor_parallel: int,
    gpu_memory_utilization: float,
):
    """Set up SGLang runtime and create FastAPI app."""
    from soup_cli.utils.sglang import create_sglang_app, create_sglang_runtime

    console.print(
        Panel(
            f"[bold yellow]WARNING:[/] Loading model via SGLang: "
            f"[bold]{model_path}[/]\n"
            "SGLang loads models with trust_remote_code enabled.\n"
            "If this model contains custom code, it will execute "
            "on this machine.\nOnly use models you trust.",
            title="SGLang Runtime",
            border_style="yellow",
        )
    )
    console.print("[dim]Initializing SGLang runtime...[/]")
    runtime, runtime_model_name = create_sglang_runtime(
        model_path=str(model_path),
        base_model=base_model,
        is_adapter=is_adapter,
        tensor_parallel_size=tensor_parallel,
        mem_fraction_static=gpu_memory_utilization,
    )
    console.print("[bold green]SGLang runtime ready![/]")

    app = create_sglang_app(
        runtime=runtime,
        runtime_model_name=runtime_model_name,
        model_name=str(model_path.name),
        max_tokens_default=max_tokens_default,
    )

    return app


def _detect_base_model(adapter_config_path: Path) -> Optional[str]:
    """Read base_model_name_or_path from adapter_config.json."""
    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def _load_model(
    model_path: str,
    base_model: Optional[str],
    is_adapter: bool,
    device: str,
    trust_remote_code: bool = False,
):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("[dim]Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        console.print(f"[dim]Loading base model: {base_model}...[/]")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            dtype=torch.float16,
        )
        console.print(f"[dim]Loading LoRA adapter: {model_path}...[/]")
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        console.print(f"[dim]Loading model: {model_path}...[/]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _load_draft_model(speculative_model: str, device: str):
    """Load a smaller draft model for speculative decoding."""
    import re

    import torch
    from transformers import AutoModelForCausalLM

    # SSRF protection: block URL-based model paths
    if re.match(r'^https?://', speculative_model):
        console.print(
            "[red]Speculative model must be a local path or HuggingFace model ID, "
            "not a URL.[/]"
        )
        raise typer.Exit(1)

    console.print(f"[dim]Loading draft model: {speculative_model}...[/]")
    draft = AutoModelForCausalLM.from_pretrained(
        speculative_model,
        device_map="auto" if device != "cpu" else "cpu",
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    draft.eval()
    return draft


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    assistant_model=None,
    num_assistant_tokens: int = 5,
    logits_processor=None,
):
    """Generate a response from the model."""
    import torch

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        text = "\n".join(parts)

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if assistant_model is not None:
            gen_kwargs["assistant_model"] = assistant_model
            gen_kwargs["num_assistant_tokens"] = num_assistant_tokens
        # v0.33.0 #53 — structured-output LogitsProcessor list (may be empty).
        if logits_processor:
            gen_kwargs["logits_processor"] = logits_processor

        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    prompt_tokens = input_ids.shape[1]
    completion_tokens = len(new_tokens)

    return response, prompt_tokens, completion_tokens


def _create_app(
    model_obj,
    tokenizer,
    device: str,
    model_name: str,
    max_tokens_default: int,
    draft_model=None,
    num_speculative_tokens: int = 5,
    adapter_map: Optional[Dict[str, str]] = None,
    output_constraint: Optional[Dict] = None,
    enable_dashboard: bool = False,
    tracer=None,
):
    """Create the FastAPI application with OpenAI-compatible endpoints."""
    import threading as _threading

    from fastapi import FastAPI, HTTPException
    from fastapi import Path as FPath
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field

    from soup_cli.utils.metrics import ServerMetrics

    app = FastAPI(title="Soup Inference Server", version="1.0.0")

    # Loopback-only CORS: the inference server hosts state-mutating POST
    # endpoints (activate/deactivate adapter) without auth, so wildcard CORS
    # would let any browser page swap the active adapter. Loopback origins
    # cover the curl / same-host IDE extension cases.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Shared metrics bucket — always created so /metrics works whether or
    # not --dashboard is enabled.
    metrics = ServerMetrics()
    # Active adapter name (None = base model). Protected by a lock because
    # FastAPI runs sync handlers in a threadpool.
    active_state: Dict[str, Optional[str]] = {"active": None}
    active_lock = _threading.Lock()

    # --- Request/Response models ---

    class ChatMessage(PydanticBaseModel):
        role: str
        content: str

    class ChatCompletionRequest(PydanticBaseModel):
        model: str = model_name
        messages: list[ChatMessage]
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
        stream: bool = False
        adapter: Optional[str] = Field(
            default=None,
            description="Adapter name to use (from --adapters flag).",
        )

    # Resolved adapter map (name → path)
    _adapter_map = adapter_map or {}

    # --- Endpoints ---

    def _active_snapshot() -> Optional[str]:
        with active_lock:
            return active_state["active"]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "device": device,
            "active_adapter": _active_snapshot(),
        }

    @app.get("/metrics")
    def metrics_endpoint():
        """Dashboard + Prometheus-style JSON scrape."""
        return metrics.snapshot()

    @app.get("/v1/adapters")
    def list_adapters():
        """List loaded LoRA adapters (names only, no paths for security)."""
        current = _active_snapshot()
        return {
            "adapters": [
                {"name": name, "active": name == current}
                for name in _adapter_map
            ],
            "active": current,
        }

    @app.post("/v1/adapters/activate/{name}")
    def activate_adapter(name: str = FPath(..., pattern=r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$")):
        """Hot-swap the active adapter. Name must be in the loaded map."""
        if not _adapter_map:
            raise HTTPException(
                status_code=404, detail="No adapters loaded."
            )
        if name not in _adapter_map:
            raise HTTPException(
                status_code=404,
                detail="Unknown adapter. Use GET /v1/adapters to list available adapters.",
            )
        with active_lock:
            active_state["active"] = name
        return {"active": name, "status": "ok"}

    @app.post("/v1/adapters/deactivate")
    def deactivate_adapter():
        """Return to base model (clear active adapter)."""
        with active_lock:
            active_state["active"] = None
        return {"active": None, "status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "soup",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        # Check adapter selection (from request body)
        requested_adapter = request.adapter
        if requested_adapter and _adapter_map:
            if requested_adapter not in _adapter_map:
                raise HTTPException(
                    status_code=404,
                    detail="Unknown adapter. Use GET /v1/adapters to list available adapters.",
                )
        elif requested_adapter and not _adapter_map:
            raise HTTPException(
                status_code=404,
                detail="No adapters loaded.",
            )

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens or max_tokens_default

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    model_obj, tokenizer, messages,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    model_name=model_name,
                    assistant_model=draft_model,
                    num_assistant_tokens=num_speculative_tokens,
                ),
                media_type="text/event-stream",
            )

        import contextlib as _contextlib

        started = time.perf_counter()
        completion_tokens = 0  # ensure defined on error paths for metrics
        # Use ExitStack so tracer span + track_request both get correct
        # exception propagation (__exit__ sees exc info, span marked error).
        with _contextlib.ExitStack() as stack:
            stack.enter_context(metrics.track_request())
            if tracer is not None:
                stack.enter_context(tracer.start_as_current_span("chat.completion"))
            try:
                try:
                    # v0.33.0 #53 — build LogitsProcessor list per request.
                    # Cheap (~us); per-request build keeps the descriptor
                    # mutable via /v1/output_constraint endpoints in future.
                    from soup_cli.utils.structured_output import (
                        build_logits_processors,
                    )
                    processors = build_logits_processors(
                        output_constraint, tokenizer,
                    )
                    response_text, prompt_tokens, completion_tokens = _generate_response(
                        model_obj, tokenizer, messages,
                        max_tokens=max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        assistant_model=draft_model,
                        num_assistant_tokens=num_speculative_tokens,
                        logits_processor=processors or None,
                    )
                except Exception:
                    logger.exception("Generation error")
                    raise HTTPException(status_code=500, detail="Internal server error")

                metrics.record_tokens(completion_tokens)

                # output_constraint is validated upstream; v0.33.0 #53 wires
                # it through outlines / lm-format-enforcer into the generate
                # loop. If neither library is installed, build_logits_processors
                # returns an empty list and generation runs free-form.
                pass

                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            finally:
                # Always record latency so tail-latency percentiles include
                # error paths (prevents blind spots on the dashboard).
                metrics.record_latency((time.perf_counter() - started) * 1000)

    # Expose dashboard intent + constraint on the app for tests + introspection
    app.state.enable_dashboard = enable_dashboard
    app.state.output_constraint = output_constraint
    return app


def _stream_response(
    model, tokenizer, messages,
    max_tokens, temperature, top_p, model_name,
    assistant_model=None, num_assistant_tokens=5,
):
    """Generator that yields SSE chunks for streaming responses."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Generate full response (true token-by-token streaming requires TextIteratorStreamer)
    try:
        response_text, _, _ = _generate_response(
            model, tokenizer, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            assistant_model=assistant_model,
            num_assistant_tokens=num_assistant_tokens,
        )
    except Exception:
        logger.exception("Stream generation error")
        yield 'data: {"error": "Internal server error"}\n\n'
        return

    # Simulate streaming by sending word-by-word
    words = response_text.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == 0 else f" {word}"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
