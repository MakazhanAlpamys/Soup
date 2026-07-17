"""soup serve — local inference server with OpenAI-compatible API."""

import contextlib
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import typer

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator
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
    # realpath + commonpath containment (is_under) — Path.resolve() +
    # relative_to() breaks on Windows 8.3 short names.
    from soup_cli.utils.paths import is_under

    if cwd is None:
        cwd = str(Path.cwd())
    if not is_under(path, cwd):
        return False
    try:
        return Path(path).resolve().exists()
    except OSError:
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
        "127.0.0.1",
        "--host",
        help=(
            "Host to bind to. Defaults to loopback (127.0.0.1); the server "
            "exposes an unauthenticated code-exec tool endpoint, so binding a "
            "public interface (0.0.0.0) should be paired with --tool-auth-token."
        ),
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
    trace_log: Optional[str] = typer.Option(
        None,
        "--trace-log",
        help=(
            "Append per-request {prompt, response, latency_ms, tokens, ts} "
            "to JSONL at this path. Path must stay under cwd. Rotates at "
            "100 MB (one backup retained). Added in v0.40.3 (#33)."
        ),
    ),
    trace_log_cap_mb: int = typer.Option(
        100,
        "--trace-log-cap-mb",
        help="Rotation cap in MB for --trace-log (1 - 10000). Default 100.",
    ),
    record_thumbs: Optional[str] = typer.Option(
        None,
        "--record-thumbs",
        help=(
            "Auto-capture thumbs-up/down feedback into a local-rl SQLite at "
            "this path via POST /v1/thumbs. Path must stay under cwd. "
            "Transformers backend only. v0.71.1 (#230)."
        ),
    ),
    reasoning_parser: Optional[str] = typer.Option(
        None,
        "--reasoning-parser",
        help=(
            "Strip reasoning-trace blocks from responses. One of: "
            "deepseek-r1 | qwen3 | phi4 | openthinker. v0.53.9 #98."
        ),
    ),
    steer: Optional[str] = typer.Option(
        None,
        "--steer",
        help=(
            "Apply a stored activation-steering vector at decode time "
            "(CAA / ITI / RepE). Pass the name registered via "
            "`soup steer train`. Schema-only in v0.62.0; live decode hook "
            "ships in v0.62.1."
        ),
    ),
    steer_strength: float = typer.Option(
        1.0,
        "--steer-strength",
        help=(
            "Steering strength multiplier (|s| <= 10.0). Ignored when "
            "--steer is unset. v0.62.0 Part C."
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Source hub for the base model: hf (default) / modelscope / "
            "modelers. Non-HF hubs require the matching SDK (v0.53.10 #152)."
        ),
    ),
    bank: Optional[str] = typer.Option(
        None,
        "--bank",
        help=(
            "Path to a VeRA / VB-LoRA vector bank (JSON). Multi-tenant LoRA "
            "serving at MB-per-user: the per-token delta is v_u ⊙ Px, routed "
            "by the X-User-Id request header. Requires --backend transformers. "
            "(v0.71.12 #221)"
        ),
    ),
    bank_strength: float = typer.Option(
        1.0,
        "--bank-strength",
        help="Vector-bank delta strength multiplier. Ignored when --bank is unset.",
    ),
    mole: Optional[str] = typer.Option(
        None,
        "--mole",
        help=(
            "Path to a MoLE training output dir (mole_gate.pt + "
            "mole_manifest.json). Serves the base + N frozen task LoRAs with "
            "per-token gate blending at decode time. Requires --backend "
            "transformers; not combinable with --bank / --steer / --adapters. "
            "(v0.71.17 #259)"
        ),
    ),
    kv_cache_type: Optional[str] = typer.Option(
        None,
        "--kv-cache-type",
        help=(
            "KV-cache type for decoding: q8_0 (8-bit quantized, needs hqq) / "
            "bf16 / f16 (cache dtype) / fp8 (vLLM+Hopper only). Transformers "
            "backend only — vLLM / SGLang routing is in the blocked tail. "
            "v0.71.14 (#140)."
        ),
    ),
    tool_auth_token: Optional[str] = typer.Option(
        None,
        "--tool-auth-token",
        help=(
            "Require 'Authorization: Bearer <token>' on the code-exec tool "
            "endpoints (/v1/tools/python, /v1/tools/web_search). Strongly "
            "recommended whenever --host is not loopback, since those "
            "endpoints run caller-supplied Python in a best-effort sandbox."
        ),
    ),
):
    """Start a local inference server with OpenAI-compatible API."""
    # Security: the transformers backend exposes a best-effort code-exec tool
    # endpoint (/v1/tools/python). Binding a non-loopback host without a tool
    # auth token puts that endpoint on the network unauthenticated.
    if host not in {"127.0.0.1", "localhost", "::1"} and not tool_auth_token:
        from rich.markup import escape as _rich_escape

        console.print(
            f"[bold yellow]Security warning:[/] binding non-loopback host "
            f"'{_rich_escape(str(host))}' exposes the unauthenticated code-exec "
            "tool endpoint (/v1/tools/python) to the network. Pass "
            "[bold]--tool-auth-token <secret>[/] to require a bearer token, or "
            "use [bold]--host 127.0.0.1[/] (the default)."
        )
    # v0.71.12 #221 — validate `--bank` up front (path containment + backend)
    # so a typo / bad path surfaces before backend init.
    if bank is not None:
        from rich.markup import escape as _rich_escape

        from soup_cli.utils.paths import is_under_cwd

        if "\x00" in bank or not is_under_cwd(bank):
            console.print(
                f"[red]Invalid --bank path:[/] {_rich_escape(str(bank))} "
                "(must be under the current directory, no null bytes)."
            )
            raise typer.Exit(code=2)
        if backend.lower() != "transformers":
            console.print(
                "[red]--bank requires --backend transformers[/] "
                "(the bank installs a forward hook on the loaded model; "
                "vLLM / SGLang / MII are not supported)."
            )
            raise typer.Exit(code=2)
    # v0.71.17 #259 — validate `--mole` up front (path containment + backend +
    # mutual-exclusion) so a misconfig surfaces before any model load. The MoLE
    # runtime loads its OWN base + adapters + gate in the transformers branch.
    if mole is not None:
        from rich.markup import escape as _rich_escape

        from soup_cli.utils.paths import is_under_cwd

        if "\x00" in mole or not is_under_cwd(mole):
            console.print(
                f"[red]Invalid --mole path:[/] {_rich_escape(str(mole))} "
                "(must be under the current directory, no null bytes)."
            )
            raise typer.Exit(code=2)
        if backend.lower() != "transformers":
            console.print(
                "[red]--mole requires --backend transformers[/] "
                "(MoLE blends per-token over N task adapters with a custom "
                "decode loop; vLLM / SGLang / MII are not supported)."
            )
            raise typer.Exit(code=2)
        conflicts = [
            name
            for name, val in (
                ("--bank", bank),
                ("--steer", steer),
                ("--adapters", adapters),
                ("--speculative-decoding", speculative_model),
            )
            if val
        ]
        if conflicts:
            console.print(
                "[red]--mole cannot be combined with:[/] "
                f"{_rich_escape(', '.join(conflicts))} "
                "(MoLE replaces the served model with its own per-token blend)."
            )
            raise typer.Exit(code=2)
    # v0.62.0 Part C / v0.71.10 #201 — validate `--steer` name + strength up
    # front so a typo surfaces before backend init. The live decode hook is
    # installed in the transformers branch after model load.
    if steer is not None:
        from rich.markup import escape as _rich_escape

        from soup_cli.utils.steering import (
            validate_steering_name,
            validate_steering_strength,
        )

        try:
            validate_steering_name(steer)
            validate_steering_strength(steer_strength)
        except (TypeError, ValueError) as exc:
            # Escape the exception message — it embeds the operator-
            # supplied --steer value via {value!r}, which would otherwise
            # let a crafted name inject Rich markup (security review M1).
            console.print(
                f"[red]Invalid --steer:[/] {_rich_escape(str(exc))}"
            )
            raise typer.Exit(code=2) from exc
        if backend.lower() != "transformers":
            console.print(
                "[red]--steer requires --backend transformers[/] "
                "(activation steering installs a forward hook on the loaded "
                "model; vLLM / SGLang / MII are not supported)."
            )
            raise typer.Exit(code=2)

    # v0.71.14 #140 — resolve `--kv-cache-type` up front (before model load) so
    # an invalid type / fp8-on-Ampere / vLLM-backend / missing-quant-backend
    # surfaces immediately. The resolved runtime is threaded into the
    # transformers branch below (model dtype + generate kwargs).
    resolved_kv_runtime = None
    if kv_cache_type is not None:
        from rich.markup import escape as _rich_escape

        from soup_cli.utils.kv_cache import (
            apply_kv_cache_type,
            quantized_cache_backend_available,
        )

        cc = None
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                cc = _torch.cuda.get_device_capability(0)
        except Exception as _cc_exc:  # noqa: BLE001 — torch missing / no CUDA
            # cc stays None → the fp8 gate falls back to the generic
            # vLLM-only message instead of the precise capability one.
            logger.debug("kv-cache CUDA capability probe failed: %r", _cc_exc)
            cc = None
        try:
            resolved_kv_runtime = apply_kv_cache_type(
                kv_cache_type, backend=backend.lower(), compute_capability=cc
            )
        except (TypeError, ValueError, NotImplementedError, RuntimeError) as exc:
            console.print(
                f"[red]--kv-cache-type:[/] {_rich_escape(str(exc))}"
            )
            raise typer.Exit(code=2) from exc
        if (
            resolved_kv_runtime.requires_quant_backend
            and quantized_cache_backend_available() is None
        ):
            console.print(
                "[red]--kv-cache-type q8_0[/] needs a quantized-cache backend. "
                "Install one with [bold]pip install hqq[/] "
                "(or optimum-quanto)."
            )
            raise typer.Exit(code=2)

    # v0.53.10 #152 — pre-fetch base from a non-HF hub before serve starts.
    if hub and hub != "hf":
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        try:
            model, base_model = apply_hub_to_cli_model(
                model, base_model, hub, console=console
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2) from exc
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    # Lazy imports for fast CLI startup
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
        from fastapi.responses import StreamingResponse  # noqa: F401
    except ImportError:
        console.print(
            "[red]FastAPI/uvicorn not installed.[/]\n"
            "Install with: [bold]pip install \"soup-cli\\[serve]\"[/]"
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
                "Install with: [bold]pip install \"soup-cli\\[serve-fast]\"[/]"
            )
            raise typer.Exit(1)

    # Validate SGLang availability
    if backend == "sglang":
        from soup_cli.utils.sglang import check_sglang_available

        if not check_sglang_available():
            console.print(
                "[red]SGLang not installed.[/]\n"
                "Install with: [bold]pip install \"soup-cli\\[sglang]\"[/]"
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
        from rich.markup import escape as _esc

        from soup_cli.utils.spec_pairing import pick_draft_model

        # A paired value can come from the local draft registry (a file that
        # may be edited outside this invocation), so strip control bytes and
        # escape Rich markup before printing — escape() alone leaves raw
        # ESC/OSC sequences live (mirrors commands/draft.py::_for_terminal).
        _ctrl = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
        _ctrl[0x7F] = None

        def _safe(value: str) -> str:
            return _esc(str(value).translate(_ctrl))

        target_for_pairing = base_model or str(model_path)
        paired = pick_draft_model(target_for_pairing)
        if paired:
            speculative_model = paired
            console.print(
                f"[green]Auto-paired draft model:[/] {_safe(paired)} "
                f"(target: {_safe(target_for_pairing)})"
            )
        else:
            console.print(
                f"[yellow]--auto-spec: no known draft model for "
                f"{_safe(target_for_pairing)}. Skipping speculative decoding.[/]"
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

    # v0.36.0 Part B: --trust-remote-code default-deny, resolved ONCE for
    # every backend. vLLM previously loaded with an unconditional
    # trust_remote_code=True (arbitrary repo code, zero notice) — resolve the
    # same gate + warning panel the transformers path uses so no backend
    # silently executes an untrusted repo's code.
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    _trust_probe_target = base_model or str(model_path)
    _trust_requires = model_requires_trust_remote_code(str(model_path)) or False
    resolved_trust = resolve_trust_remote_code(
        _trust_probe_target,
        requested=trust_remote_code,
        console=console,
        requires_remote_code=_trust_requires,
    )

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
            trust_remote_code=resolved_trust,
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
        # Transformers backend (original). ``resolved_trust`` was computed
        # once above (v0.36.0 Part B default-deny) and shared across backends.

        # v0.71.17 #259 — serve-time MoLE loads its OWN base + N task LoRAs +
        # gate; the `model` CLI arg is the base, the manifest supplies adapters
        # + gate geometry. Bypasses _load_model entirely.
        mole_runtime = None
        if mole is not None:
            from rich.markup import escape as _esc

            from soup_cli.utils.mole_routing import load_mole_for_serve

            try:
                # `--model` is the MoLE gate/manifest dir; the base model comes
                # from `--base` (operator override) or, when None, the manifest's
                # recorded base (load_mole_for_serve handles the fallback).
                mole_runtime = load_mole_for_serve(
                    mole,
                    base=base_model,
                    device=device,
                    trust_remote_code=resolved_trust,
                )
            except (TypeError, ValueError, OSError, FileNotFoundError) as exc:
                console.print(f"[red]--mole:[/] {_esc(str(exc))}")
                raise typer.Exit(2) from exc
            model_obj = mole_runtime.model
            tokenizer = mole_runtime.tokenizer
            console.print(
                f"[green]MoLE serve active:[/] "
                f"adapters={len(mole_runtime.adapter_names)}, "
                f"top_k={mole_runtime.gate.top_k}, gate=loaded"
            )
        else:
            model_obj, tokenizer = _load_model(
                model_path=str(model_path),
                base_model=base_model,
                is_adapter=is_adapter,
                device=device,
                trust_remote_code=resolved_trust,
                kv_cache_dtype=(
                    resolved_kv_runtime.model_dtype if resolved_kv_runtime else None
                ),
            )
        console.print("[bold green]Model loaded![/]")
        if resolved_kv_runtime is not None:
            console.print(
                f"[green]KV cache:[/] {resolved_kv_runtime.kv_cache_type} "
                f"— {resolved_kv_runtime.note}"
            )

        # v0.71.33 — actually load the --adapters map into the model so
        # /v1/adapters/activate + the per-request `adapter` field switch the
        # served weights (previously validated + tracked but never applied).
        peft_adapter_names: set = set()
        if adapter_map:
            from rich.markup import escape as _esc

            try:
                model_obj, peft_adapter_names = _load_named_adapters(
                    model_obj, adapter_map
                )
            except Exception as exc:  # noqa: BLE001 — surface any PEFT error
                console.print(
                    f"[red]Failed to load --adapters:[/] {_esc(str(exc))}"
                )
                raise typer.Exit(1) from exc
            console.print(
                "[green]Adapters ready:[/] "
                + ", ".join(sorted(peft_adapter_names))
            )

        # v0.71.10 #201 — install the activation-steering decode hook. The
        # handle persists for the server's lifetime (process-global model).
        if steer is not None:
            from rich.markup import escape as _esc

            from soup_cli.utils.steering import (
                install_steering_hook,
                load_steering_artifact,
                resolve_steering_dir,
            )

            try:
                steer_dir = resolve_steering_dir(steer)
                loaded_steer = load_steering_artifact(steer_dir)
                install_steering_hook(
                    model_obj, loaded_steer, strength=steer_strength
                )
            except (TypeError, ValueError, OSError) as exc:
                console.print(f"[red]--steer:[/] {_esc(str(exc))}")
                raise typer.Exit(2) from exc
            console.print(
                f"[green]Steering active:[/] {_esc(loaded_steer.name)} "
                f"({_esc(loaded_steer.method)}, layer {loaded_steer.layer}, "
                f"strength {steer_strength})"
            )

        # v0.71.12 #221 — load a VeRA / VB-LoRA bank + install the per-user
        # decode hook. The active user is selected per request via X-User-Id.
        loaded_bank = None
        if bank is not None:
            from rich.markup import escape as _esc

            from soup_cli.utils.vector_bank import (
                apply_bank_to_serve,
                load_bank,
            )

            try:
                bank_obj = load_bank(bank)
                loaded_bank = apply_bank_to_serve(bank_obj)
                loaded_bank.install_serve_hook(
                    model_obj, strength=bank_strength
                )
            except (TypeError, ValueError, OSError) as exc:
                console.print(f"[red]--bank:[/] {_esc(str(exc))}")
                raise typer.Exit(2) from exc
            console.print(
                f"[green]Vector bank active:[/] {_esc(loaded_bank.name)} "
                f"({len(loaded_bank._user_vectors)} users, "
                f"dim={loaded_bank.vector_dim}, strength={bank_strength})"
            )

        # Load draft model for speculative decoding (transformers backend)
        draft_model = None
        if speculative_model:
            from rich.markup import escape as _esc

            _spec_display = _esc(str(speculative_model))
            console.print(
                Panel(
                    f"[bold yellow]WARNING:[/] Loading draft model: "
                    f"[bold]{_spec_display}[/]\n"
                    "If this model contains custom code, it will execute "
                    "on this machine.\n"
                    "Only use models you trust.",
                    title="Speculative Decoding",
                    border_style="yellow",
                )
            )
            draft_model = _load_draft_model(speculative_model, device)
            console.print(
                f"[green]Speculative decoding enabled:[/] draft={_spec_display}, "
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

        # v0.40.3 (#33 (b)) — passive request log.
        trace_log_writer = None
        if trace_log is not None:
            from soup_cli.monitoring.trace_logger import TraceLogWriter

            try:
                trace_log_writer = TraceLogWriter(
                    trace_log, cap_mb=trace_log_cap_mb,
                )
            except (TypeError, ValueError) as exc:
                from rich.markup import escape as _escape

                console.print(f"[red]--trace-log:[/] {_escape(str(exc))}")
                raise typer.Exit(1) from exc
            console.print(
                f"[green]Request trace log:[/] {trace_log_writer.path} "
                f"(cap {trace_log_cap_mb} MB)"
            )

        # v0.53.9 #98 — validate reasoning parser name once at startup.
        resolved_reasoning_parser: Optional[str] = None
        if reasoning_parser:
            from soup_cli.utils.reasoning_parser import validate_parser_name

            try:
                resolved_reasoning_parser = validate_parser_name(reasoning_parser)
            except (TypeError, ValueError) as exc:
                console.print(f"[red]--reasoning-parser:[/] {exc}")
                raise typer.Exit(1) from exc

        # v0.71.1 #230 — auto-capture thumbs feedback into a local-rl SQLite.
        record_thumbs_db: Optional[str] = None
        if record_thumbs is not None:
            from rich.markup import escape as _escape

            from soup_cli.utils.local_rl import init_local_rl_db, validate_db_path

            try:
                validate_db_path(record_thumbs)
                init_local_rl_db(record_thumbs)
            except (TypeError, ValueError) as exc:
                console.print(f"[red]--record-thumbs:[/] {_escape(str(exc))}")
                raise typer.Exit(1) from exc
            record_thumbs_db = record_thumbs
            console.print(f"[green]Thumbs feedback log:[/] {_escape(record_thumbs)}")

        app = _create_app(
            model_obj=model_obj,
            tokenizer=tokenizer,
            device=device,
            model_name=str(model_path.name),
            max_tokens_default=max_tokens_default,
            draft_model=draft_model,
            num_speculative_tokens=num_speculative_tokens,
            adapter_map=adapter_map if adapter_map else None,
            peft_adapter_names=peft_adapter_names,
            output_constraint=constraint,
            enable_dashboard=dashboard,
            tracer=tracer,
            trace_log_writer=trace_log_writer,
            reasoning_parser=resolved_reasoning_parser,
            record_thumbs_db=record_thumbs_db,
            auth_token=tool_auth_token,
            loaded_bank=loaded_bank,
            mole_runtime=mole_runtime,
            kv_cache_generate_kwargs=(
                _plain_kv_kwargs(resolved_kv_runtime.generate_kwargs)
                if resolved_kv_runtime
                else None
            ),
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
    trust_remote_code: bool = False,
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
        trust_remote_code=trust_remote_code,
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
    kv_cache_dtype: Optional[str] = None,
):
    """Load model and tokenizer.

    ``kv_cache_dtype`` (v0.71.14 #140) selects the model compute dtype so the
    transformers DynamicCache runs in it: ``"bfloat16"`` → bf16, else the
    default float16. The bf16/f16 ``kv_cache_type`` values map here; q8_0 uses
    a quantized cache via generate kwargs and leaves the model dtype unchanged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dtype = (
        torch.bfloat16 if kv_cache_dtype == "bfloat16" else torch.float16
    )

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
            dtype=load_dtype,
        )
        console.print(f"[dim]Loading LoRA adapter: {model_path}...[/]")
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        console.print(f"[dim]Loading model: {model_path}...[/]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            dtype=load_dtype,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _load_named_adapters(model_obj, adapter_map: Dict[str, str]):
    """Load the ``--adapters name=path`` map into ``model_obj`` for hot-swap.

    Returns ``(model_obj, adapter_names)``. The returned model is a PeftModel
    carrying every named adapter; ``adapter_names`` is the set actually loaded.
    Request-time selection is done by :func:`_adapter_scope`.

    Without this, ``--adapters`` / ``POST /v1/adapters/activate`` / the
    per-request ``adapter`` field were validated + tracked but NEVER applied —
    every request silently ran the startup model (v0.71.33 fix).
    """
    from peft import PeftModel

    names = list(adapter_map)
    already_peft = isinstance(model_obj, PeftModel)
    for idx, name in enumerate(names):
        path = adapter_map[name]
        if idx == 0 and not already_peft:
            # Wrap the plain base model into a multi-adapter PeftModel.
            model_obj = PeftModel.from_pretrained(
                model_obj, path, adapter_name=name
            )
        else:
            model_obj.load_adapter(path, adapter_name=name)
        console.print(f"[dim]Loaded adapter '{name}' from {path}[/]")
    model_obj.eval()
    return model_obj, set(names)


@contextlib.contextmanager
def _adapter_scope(model, lock, names, requested, active):
    """Select the LoRA adapter for one generation, serialized by ``lock``.

    ``requested`` (request body ``adapter`` field) overrides ``active`` (the
    ``/v1/adapters/activate`` selection). A name not in ``names`` (or ``None``)
    runs the base model with adapters disabled. No-op when no named adapters
    were loaded (``names`` empty), so the ordinary single-model serve path is
    completely unaffected.

    The lock spans the whole generation because the PeftModel is process-global
    and ``set_adapter`` mutates shared state — two concurrent requests on
    different adapters would otherwise race. Generation is one blocking call in
    every path (chat / stream / completions all generate-then-return), so this
    serializes adapter-selected requests but never holds across true streaming.
    """
    if not names or lock is None:
        yield
        return
    name = requested or active
    with lock:
        if name and name in names and hasattr(model, "set_adapter"):
            model.set_adapter(name)
            yield
        elif hasattr(model, "disable_adapter"):
            # No (or unknown) adapter selected → base model for this request.
            with model.disable_adapter():
                yield
        else:
            yield


def _load_draft_model(speculative_model: str, device: str):
    """Load a smaller draft model for speculative decoding."""
    import os
    import re

    import torch
    from rich.markup import escape
    from transformers import AutoModelForCausalLM

    # SSRF protection: block URL-based model paths
    if re.match(r'^https?://', speculative_model):
        console.print(
            "[red]Speculative model must be a local path or HuggingFace model ID, "
            "not a URL.[/]"
        )
        raise typer.Exit(1)

    # A locally-registered draft (soup draft distill, v0.71.33) can be selected
    # here via --auto-spec. Refuse pickle / PyTorch-classic weights before
    # from_pretrained torch.load's them — a poisoned ~/.soup/drafts.json entry
    # must not become a load-time RCE.
    if os.path.isdir(speculative_model):
        from soup_cli.utils.strict_safetensors import assert_safe_top_level_weights

        assert_safe_top_level_weights(speculative_model)

    console.print(f"[dim]Loading draft model: {escape(speculative_model)}...[/]")
    draft = AutoModelForCausalLM.from_pretrained(
        speculative_model,
        device_map="auto" if device != "cpu" else "cpu",
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    draft.eval()
    return draft


def _plain_kv_kwargs(mapping: Any) -> Dict[str, Any]:
    """Deep-convert a (possibly MappingProxyType-nested) mapping to plain dicts.

    The v0.71.14 #140 ``kv_cache_type`` runtime stores ``generate_kwargs`` as
    immutable ``MappingProxyType`` (incl. a nested ``cache_config``). transformers
    ``generate`` builds the quantized cache from a plain ``cache_config`` dict,
    so convert before threading it in.
    """
    out: Dict[str, Any] = {}
    for key, value in dict(mapping).items():
        if isinstance(value, Mapping):
            out[key] = _plain_kv_kwargs(value)
        else:
            out[key] = value
    return out


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
    ngram_config: Any = None,
    kv_cache_generate_kwargs: Optional[Dict[str, Any]] = None,
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
        # v0.53.6 #104 — n-gram speculative decoding (transformers backend).
        # Mutually exclusive with a real draft `assistant_model`.
        if ngram_config is not None and assistant_model is None:
            # HF Transformers >= 4.38 supports prompt-lookup decoding via
            # `prompt_lookup_num_tokens`. We expose `num_draft_tokens` as
            # the user-facing knob; n-gram size + prompt_lookup_max are
            # validated upstream by `validate_ngram_config`.
            try:
                gen_kwargs["prompt_lookup_num_tokens"] = int(
                    ngram_config.num_draft_tokens
                )
            except (TypeError, AttributeError):
                # Schema gate at construction time enforces shape; this
                # is defence-in-depth.
                pass

        # v0.71.14 #140 — KV-cache-type generate kwargs (quantized cache for
        # q8_0). bf16/f16 map to the model load dtype, not here, so this is
        # empty for those. Merged last so it can't be clobbered.
        if kv_cache_generate_kwargs:
            gen_kwargs.update(kv_cache_generate_kwargs)

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
    peft_adapter_names: Optional[set] = None,
    output_constraint: Optional[Dict] = None,
    enable_dashboard: bool = False,
    tracer=None,
    trace_log_writer=None,
    ngram_config: Any = None,
    web_search_config: Any = None,
    web_search_backend: Any = None,
    auth_token: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    record_thumbs_db: Optional[str] = None,
    loaded_bank: Any = None,
    mole_runtime: Any = None,
    kv_cache_generate_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create the FastAPI application with OpenAI-compatible endpoints.

    Args:
        auth_token: optional Bearer-token gate for the v0.53.7 tool
            endpoints (``/v1/tools/python`` + ``/v1/tools/web_search``).
            When ``None`` (default), endpoints inherit the server's
            loopback-only CORS trust boundary. When set, callers must
            supply ``Authorization: Bearer <token>``.
    """
    import threading as _threading

    from fastapi import FastAPI, Header, HTTPException
    from fastapi import Path as FPath
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field

    def _check_tool_auth(authorization: Optional[str]) -> None:
        """v0.53.7 H-A: gate tool endpoints when ``auth_token`` is set."""
        if not auth_token:
            return
        expected = f"Bearer {auth_token}"
        if not authorization or authorization != expected:
            raise HTTPException(
                status_code=401, detail="Invalid or missing bearer token"
            )

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
    # v0.71.33 — adapters actually loaded into the PeftModel + the lock that
    # serializes set_adapter + generate (the model is process-global).
    _peft_adapter_names = peft_adapter_names or set()
    _generation_lock = threading.Lock()

    # v0.71.12 #221 — VeRA / VB-LoRA bank loaded at startup (or None). The
    # active user is selected per request via the X-User-Id header.
    _loaded_bank = loaded_bank
    # v0.71.17 #259 — serve-time MoLE runtime (or None). When set, the chat
    # handler routes generation through the per-token gate-blend decode loop.
    _mole_runtime = mole_runtime

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
    def chat_completions(
        request: ChatCompletionRequest,
        x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    ):
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

        # v0.71.12 #221 — select the active VeRA / VB-LoRA user for this
        # request. set_active_user(None) (or an unknown id) self-clears, so
        # a request without the X-User-Id header never inherits the previous
        # request's per-user steering.
        if _loaded_bank is not None:
            _loaded_bank.set_active_user(x_user_id)

        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens or max_tokens_default

        if request.stream:
            stream_started = time.perf_counter()
            return StreamingResponse(
                _stream_response(
                    model_obj, tokenizer, messages,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    model_name=model_name,
                    assistant_model=draft_model,
                    num_assistant_tokens=num_speculative_tokens,
                    trace_log_writer=trace_log_writer,
                    started=stream_started,
                    kv_cache_generate_kwargs=kv_cache_generate_kwargs,
                    mole_runtime=_mole_runtime,
                    loaded_bank=_loaded_bank,
                    x_user_id=x_user_id,
                    adapter_lock=_generation_lock,
                    adapter_names=_peft_adapter_names,
                    requested_adapter=requested_adapter,
                    active_adapter=_active_snapshot(),
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
                    if _mole_runtime is not None:
                        # v0.71.17 #259 — per-token MoLE gate-blend decode.
                        (
                            response_text,
                            prompt_tokens,
                            completion_tokens,
                        ) = _mole_runtime.generate_text(
                            messages,
                            max_tokens=max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                        )
                    else:
                        # v0.71.33 — select the request's LoRA adapter (base =
                        # disabled) under the generation lock for the duration
                        # of generate().
                        with _adapter_scope(
                            model_obj, _generation_lock, _peft_adapter_names,
                            requested_adapter, _active_snapshot(),
                        ):
                            (
                                response_text,
                                prompt_tokens,
                                completion_tokens,
                            ) = _generate_response(
                                model_obj, tokenizer, messages,
                                max_tokens=max_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                assistant_model=draft_model,
                                num_assistant_tokens=num_speculative_tokens,
                                logits_processor=processors or None,
                                ngram_config=ngram_config,
                                kv_cache_generate_kwargs=kv_cache_generate_kwargs,
                            )
                except Exception:
                    logger.exception("Generation error")
                    raise HTTPException(status_code=500, detail="Internal server error")

                metrics.record_tokens(completion_tokens)

                # v0.53.9 #98 — strip reasoning-trace blocks if configured.
                if reasoning_parser is not None:
                    from soup_cli.utils.reasoning_parser import strip_reasoning

                    response_text = strip_reasoning(
                        response_text, reasoning_parser,
                    )

                # output_constraint is validated upstream; v0.33.0 #53 wires
                # it through outlines / lm-format-enforcer into the generate
                # loop. If neither library is installed, build_logits_processors
                # returns an empty list and generation runs free-form.
                pass

                # v0.40.3 (#33 (b)) — passive request log; never breaks
                # the request handler on disk / serialisation issues.
                if trace_log_writer is not None:
                    last_user = next(
                        (m["content"] for m in reversed(messages)
                         if m.get("role") == "user"),
                        "",
                    )
                    trace_log_writer.record(
                        prompt=str(last_user),
                        response=response_text,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        tokens=completion_tokens,
                    )

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

    # ----- v0.53.6 #102 — Anthropic /v1/messages route -----
    # Reuses the v0.45.0 utils/anthropic_messages converter + the existing
    # chat_completions handler. Live on transformers backend only this
    # release (vLLM /v1/messages tracked for v0.53.7).
    @app.post("/v1/messages")
    def anthropic_messages(payload: dict) -> dict:
        from soup_cli.utils.anthropic_messages import (
            from_anthropic,
            validate_anthropic_payload,
        )

        # v0.53.7 #102: streaming live (Anthropic event shape).
        wants_stream = isinstance(payload, dict) and bool(payload.get("stream"))

        try:
            validate_anthropic_payload(payload)
            openai_payload = from_anthropic(payload)
            # Drop the OpenAI-side ``stream`` field — the streaming path is
            # handled below using the Anthropic event shape, not OpenAI SSE.
            openai_payload.pop("stream", None)
            request = ChatCompletionRequest(**openai_payload)
        except (TypeError, ValueError) as exc:
            # Security: do not echo internal validator/converter details
            # to the HTTP body. Log server-side for operator debugging.
            logger.debug("/v1/messages invalid request: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as exc:  # noqa: BLE001 — pydantic ValidationError shape
            logger.debug("/v1/messages pydantic error: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid request")

        chat_response = chat_completions(request)

        # Map OpenAI chat response back to Anthropic shape.
        text = ""
        if isinstance(chat_response, dict):
            try:
                text = chat_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = ""
        usage = (
            chat_response.get("usage", {}) if isinstance(chat_response, dict) else {}
        )

        msg_id = (
            chat_response.get("id", "")
            if isinstance(chat_response, dict)
            else ""
        )
        out_model = openai_payload.get("model", model_name)
        in_tokens = int(usage.get("prompt_tokens", 0) or 0)
        out_tokens = int(usage.get("completion_tokens", 0) or 0)

        if wants_stream:
            # v0.53.7 #102 streaming live — emit Anthropic event-shape SSE.
            return StreamingResponse(
                _stream_anthropic_messages(
                    msg_id=msg_id,
                    model=out_model,
                    text=text,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                },
            )

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": out_model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            },
        }

    # ----- v0.53.6 #103 / v0.53.7 — Server-side tool endpoints (live) -----
    # python / bash route through the RLVR sandbox (v0.25.0 + v0.33.0 #21
    # OS-level isolation). web_search enforces a domain allowlist; default
    # is deny-all per the v0.45.0 Part B schema.
    tool_max_code_len = 64 * 1024
    tool_max_query_len = 1024
    tool_max_results = 16

    @app.post("/v1/tools/python")
    def tool_python(
        payload: dict,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        _check_tool_auth(authorization)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid request")
        code = payload.get("code")
        if not isinstance(code, str) or not code:
            raise HTTPException(status_code=400, detail="Invalid request")
        if len(code) > tool_max_code_len:
            raise HTTPException(status_code=400, detail="Invalid request")
        try:
            from soup_cli.trainer.rewards import _run_code_sandbox

            stdout = _run_code_sandbox(code)
        except Exception as exc:  # noqa: BLE001
            logger.debug("/v1/tools/python sandbox error: %s", exc)
            raise HTTPException(status_code=500, detail="Internal server error")
        return {
            "stdout": stdout if stdout is not None else "",
            "stderr": "",
            "exit_code": 0 if stdout is not None else 1,
            "timed_out": stdout is None,
        }

    @app.post("/v1/tools/bash")
    def tool_bash(payload: dict) -> dict:  # noqa: ARG001 — payload unused on stub
        # v0.53.7 review-fix C1: bash spawns ``/bin/sh -c`` which escapes
        # the RLVR sandbox's OS-level isolation (``unshare(CLONE_NEWNET)``
        # / macOS ``sandbox-exec``); a caller can reach
        # ``http://169.254.169.254/...`` from the child shell. Reverted to
        # 501 until container/namespace work lands in v0.53.9.
        raise HTTPException(
            status_code=501,
            detail=(
                "Server-side tool 'bash' live execution deferred to "
                "v0.53.9 — sandbox isolation requires container/namespace "
                "work."
            ),
        )

    @app.post("/v1/tools/web_search")
    def tool_web_search(
        payload: dict,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        _check_tool_auth(authorization)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid request")
        query = payload.get("query")
        if not isinstance(query, str) or not query:
            raise HTTPException(status_code=400, detail="Invalid request")
        if len(query) > tool_max_query_len:
            raise HTTPException(status_code=400, detail="Invalid request")
        max_results = payload.get("max_results", 5)
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise HTTPException(status_code=400, detail="Invalid request")
        if max_results < 1 or max_results > tool_max_results:
            raise HTTPException(status_code=400, detail="Invalid request")

        # Domain allowlist is threaded into the server constructor — when
        # absent, default to deny-all per the v0.45.0 schema.
        cfg = getattr(app.state, "web_search_config", None)
        allowlist: tuple = ()
        if cfg is not None and hasattr(cfg, "domain_allowlist"):
            allowlist = tuple(cfg.domain_allowlist)
        if not allowlist:
            raise HTTPException(
                status_code=403, detail="web_search disabled (empty domain allowlist)"
            )
        # The actual search backend is operator-configurable — v0.53.7 ships
        # the security gate + 403 default and a SearXNG-style placeholder
        # that returns an empty result set when no upstream is configured.
        # Operators wanting a live search engine can patch
        # ``app.state.web_search_backend`` with a callable
        # ``(query, max_results, allowlist) -> list[dict]``.
        backend = getattr(app.state, "web_search_backend", None)
        results: list = []
        if callable(backend):
            try:
                raw_results = backend(query, max_results, allowlist)
            except Exception as exc:  # noqa: BLE001
                logger.debug("/v1/tools/web_search backend error: %s", exc)
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                )
            for r in (raw_results or [])[:max_results]:
                if isinstance(r, dict) and "url" in r:
                    # Re-check domain allowlist for backend results.
                    url = r.get("url", "")
                    if not isinstance(url, str):
                        continue
                    from urllib.parse import urlparse

                    host = urlparse(url).hostname or ""
                    host = host.lower()
                    allowed = False
                    for entry in allowlist:
                        if entry.startswith("."):
                            if host == entry[1:] or host.endswith(entry):
                                allowed = True
                                break
                        elif host == entry:
                            allowed = True
                            break
                    if allowed:
                        # M-D: strip null bytes from snippet so a backend
                        # cannot inject embedded NULs through to clients.
                        snippet = str(r.get("snippet", "")).replace("\x00", "")
                        results.append(
                            {
                                "url": url,
                                "snippet": snippet[:512],
                            }
                        )
        return {"results": results}

    @app.post("/v1/thumbs")
    def record_thumb_endpoint(
        payload: dict,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """v0.71.1 #230 — record thumbs-up/down feedback into local-rl SQLite.

        Stateless by design: the client POSTs the full {prompt, response,
        thumb} (mirrors the ``soup local-rl record`` CLI). Returns 404 when
        ``--record-thumbs`` was not passed at startup.
        """
        _check_tool_auth(authorization)
        db = getattr(app.state, "record_thumbs_db", None)
        if not db:
            raise HTTPException(
                status_code=404, detail="thumbs recording not enabled"
            )
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid request")
        prompt = payload.get("prompt")
        response = payload.get("response")
        thumb = payload.get("thumb")
        if (
            not isinstance(prompt, str)
            or not isinstance(response, str)
            or not isinstance(thumb, str)
        ):
            raise HTTPException(status_code=400, detail="Invalid request")
        from soup_cli.utils.local_rl import record_thumb

        try:
            record_thumb(db_path=db, prompt=prompt, response=response, thumb=thumb)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid request") from exc
        return {"status": "ok", "thumb": thumb}

    # Expose dashboard intent + constraint on the app for tests + introspection
    app.state.enable_dashboard = enable_dashboard
    app.state.output_constraint = output_constraint
    app.state.trace_log_writer = trace_log_writer
    app.state.web_search_config = web_search_config
    app.state.web_search_backend = web_search_backend
    app.state.record_thumbs_db = record_thumbs_db
    return app


def _sanitise_sse_field(value: str, *, max_len: int) -> str:
    """v0.53.7 M-A: strip CR/LF/NUL + cap len before embedding in SSE.

    SSE wire framing uses ``\\n`` boundaries; a ``\\n`` in a header-derived
    field would close the data block early and allow a caller-controlled
    new event to be injected into the stream.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\r", "").replace("\n", "").replace("\x00", "")
    return cleaned[:max_len]


def _stream_anthropic_messages(
    *,
    msg_id: str,
    model: str,
    text: str,
    input_tokens: int,
    output_tokens: int,
) -> "Generator[str, None, None]":
    """v0.53.7 #102 — yield Anthropic event-shape SSE frames.

    Emits the canonical 4-event sequence:
    - ``message_start``: opens the message envelope.
    - ``content_block_delta``: one frame per word (best-effort streaming;
      the underlying handler ran the full generation eagerly).
    - ``message_delta`` + ``message_stop``: closes the stream.
    """
    import json as _json

    # M-A: sanitise caller-influenced fields before SSE embedding.
    msg_id = _sanitise_sse_field(msg_id, max_len=64)
    model = _sanitise_sse_field(model, max_len=200)

    def _frame(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"

    yield _frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "usage": {
                    "input_tokens": int(input_tokens),
                    "output_tokens": 0,
                },
            },
        },
    )

    # Stream word-by-word so SSE consumers see incremental progress, even
    # though the underlying generation is already complete.
    words = (text or "").split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == 0 else f" {word}"
        yield _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk_text},
            },
        )

    yield _frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": int(output_tokens)},
        },
    )
    yield _frame("message_stop", {"type": "message_stop"})


def _stream_response(
    model, tokenizer, messages,
    max_tokens, temperature, top_p, model_name,
    assistant_model=None, num_assistant_tokens=5,
    trace_log_writer=None, started=None,
    kv_cache_generate_kwargs=None,
    mole_runtime=None,
    loaded_bank=None, x_user_id=None,
    adapter_lock=None, adapter_names=None,
    requested_adapter=None, active_adapter=None,
):
    """Generator that yields SSE chunks for streaming responses."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Generate full response (true token-by-token streaming requires TextIteratorStreamer)
    completion_tokens_for_log = 0
    try:
        # v0.71.17 #260 — re-select the per-user bank inside the generator's own
        # context. set_active_user runs in the sync endpoint thread, but this
        # generator is iterated later by the async event loop in a DIFFERENT
        # contextvars.Context, so the endpoint's selection is not visible here.
        # Re-set so streamed requests still apply per-user steering (and fail
        # closed to no-steering for an unknown / absent id).
        if loaded_bank is not None:
            loaded_bank.set_active_user(x_user_id)
        if mole_runtime is not None:
            # v0.71.17 #259 — MoLE blends per-token via a custom decode loop;
            # generate the full text then simulate streaming (same shape as the
            # transformers path, which also generates-then-streams).
            response_text, _, completion_tokens_for_log = mole_runtime.generate_text(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # v0.71.33 — select the request's LoRA adapter under the generation
            # lock (resolved in the endpoint, applied here where generate runs).
            with _adapter_scope(
                model, adapter_lock, adapter_names,
                requested_adapter, active_adapter,
            ):
                response_text, _, completion_tokens_for_log = _generate_response(
                    model, tokenizer, messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    assistant_model=assistant_model,
                    num_assistant_tokens=num_assistant_tokens,
                    kv_cache_generate_kwargs=kv_cache_generate_kwargs,
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

    # v0.40.3 (#33 (b)) — passive request log on the streaming path. Latency
    # measured from the BEFORE-`_generate_response` mark passed in by the
    # chat_completions handler. Skipped if writer is None or `started` is
    # missing. Errors swallowed (passive log).
    if trace_log_writer is not None and started is not None:
        try:
            last_user = next(
                (m["content"] for m in reversed(messages)
                 if m.get("role") == "user"),
                "",
            )
            trace_log_writer.record(
                prompt=str(last_user),
                response=response_text,
                latency_ms=(time.perf_counter() - started) * 1000,
                tokens=int(completion_tokens_for_log),
                extra={"stream": True},
            )
        except Exception:  # noqa: BLE001 — passive log never blocks SSE
            logger.debug("trace_log streaming record failed", exc_info=True)
