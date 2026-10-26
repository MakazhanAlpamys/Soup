"""soup export — convert a model to GGUF format for Ollama / llama.cpp."""

import json
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.config.deprecation import EXPORT_FORMATS_REMOVED_IN
from soup_cli.utils.terminal import for_terminal, strip_control

console = Console()

SUPPORTED_FORMATS = (
    "gguf", "onnx", "tensorrt",
    # v0.52.0 Part D — BitNet 1.58-bit + TQ1_0 GGUF.
    # Live llama.cpp conversion (v0.71.20 #134); needs a built llama.cpp toolchain.
    "bitnet", "tq1_0",
    # v0.53.1 #142 — TorchAO PTQ live wiring (Int4WeightOnly / Int8DynActInt4 /
    # Float8DynActFloat8 / NVFP4). Requires --quant-config <yaml>.
    "torchao",
    # v0.53.1 #139 — UD/IQ/Apple-ARM GGUFs via llama.cpp imatrix.
    "gguf-ud",
)
GGUF_QUANT_TYPES = ("q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16", "f32")
LLAMA_CPP_DIR_NAME = "llama.cpp"
# Pin to a known release tag for supply-chain safety
LLAMA_CPP_TAG = "b5270"


def export(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to model directory (full model or LoRA adapter)",
    ),
    fmt: str = typer.Option(
        "gguf",
        "--format",
        "-f",
        help=(
            "Export format: gguf, onnx, tensorrt, bitnet, tq1_0, torchao, "
            "gguf-ud (torchao needs --quant-config)"
        ),
    ),
    quant: str = typer.Option(
        "q4_k_m",
        "--quant",
        "-q",
        help="Quantization type: q4_0, q4_k_m, q5_k_m, q8_0, f16, f32",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Default: <model-name>.<quant>.gguf",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model ID (for LoRA adapters). Auto-detected if not set.",
    ),
    llama_cpp_path: Optional[str] = typer.Option(
        None,
        "--llama-cpp",
        help="Path to llama.cpp directory. Auto-detected or cloned to ~/.soup/llama.cpp",
    ),
    onnx_task: str = typer.Option(
        "text-generation",
        "--onnx-task",
        help="ONNX export task: text-generation (causal LM) or feature-extraction (embedding)",
    ),
    deploy: Optional[str] = typer.Option(
        None,
        "--deploy",
        help="Auto-deploy after export. Currently supported: ollama",
    ),
    deploy_name: Optional[str] = typer.Option(
        None,
        "--deploy-name",
        help="Model name for deployment (used with --deploy)",
    ),
    calibration_data: Optional[str] = typer.Option(
        None,
        "--calibration-data",
        help="Path to calibration JSONL for imatrix GGUF export.",
    ),
    registry_id: Optional[str] = typer.Option(
        None,
        "--registry-id",
        help="Attach exported artifact to this registry entry "
        "(default: auto-match by source --model output dir)",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
    quant_config: Optional[str] = typer.Option(
        None,
        "--quant-config",
        help=(
            "Path to YAML for torchao PTQ export (v0.53.1 #142). "
            "Required when --format=torchao."
        ),
    ),
    gguf_flavour: Optional[str] = typer.Option(
        None,
        "--gguf-flavour",
        help=(
            "Advanced GGUF format flag — UD-Q*_K_XL / IQ*_M / Q4_0_4_4 / etc. "
            "Required when --format=gguf-ud (v0.53.1 #139)."
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Source hub for the base model when --model is a LoRA adapter: "
            "hf (default) / modelscope / modelers (v0.53.10 #152)."
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Stream GGUF converter and quantizer logs while they run.",
    ),
    no_install_deps: bool = typer.Option(
        False, "--no-install-deps",
        help="GGUF only: never invoke pip; refuse missing conversion dependencies before work.",
    ),
):
    """Export a model to GGUF, ONNX, TensorRT-LLM, BitNet, or TorchAO format."""
    # v0.53.10 #152 — pre-fetch the base model from a non-HF hub. ``model``
    # is typically a local merged dir / adapter dir; only ``base`` is rewritten.
    if hub and hub != "hf":
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        try:
            _, base = apply_hub_to_cli_model(model, base, hub, console=console)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2) from exc
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    model_path = Path(model)

    # --- Validate ---
    if fmt not in SUPPORTED_FORMATS:
        removed_notice = ""
        if fmt in ("awq", "gptq"):
            removed_notice = (
                f"\n[yellow]--format {fmt} was removed in v{EXPORT_FORMATS_REMOVED_IN}.[/]"
            )
        console.print(
            f"[red]Unsupported format: {fmt}[/]{removed_notice}\n"
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
        raise typer.Exit(1)

    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    if (verbose or no_install_deps) and fmt != "gguf":
        flags = [name for name, enabled in (
            ("--verbose", verbose), ("--no-install-deps", no_install_deps),
        ) if enabled]
        console.print(f"[red]{' and '.join(flags)} require --format gguf.[/]")
        raise typer.Exit(2)

    model_name = Path(model).name

    # --- Deploy target validation ---
    if deploy:
        if fmt not in ("gguf", "gguf-ud", "bitnet", "tq1_0"):
            console.print(
                f"[red]--deploy is not supported for format '{fmt}'. "
                "Supported formats for deploy: gguf, gguf-ud, bitnet, tq1_0[/]"
            )
            raise typer.Exit(1)
        if deploy != "ollama":
            console.print(
                f"[red]Unsupported deploy target: {deploy}[/]\n"
                "Supported: ollama"
            )
            raise typer.Exit(1)

    # --- ONNX export path ---
    if fmt == "onnx":
        output_path = _export_onnx(model_path, output, base, onnx_task, trust_remote_code)
        if output_path is not None:
            _maybe_attach_export(
                artifact_path=str(output_path), kind="onnx",
                explicit_id=registry_id, source_model=str(Path(model)),
            )
        return

    # --- TensorRT-LLM export path ---
    if fmt == "tensorrt":
        output_path = _export_tensorrt(model_path, output, base, trust_remote_code)
        if output_path is not None:
            _maybe_attach_export(
                artifact_path=str(output_path), kind="tensorrt",
                explicit_id=registry_id, source_model=str(Path(model)),
            )
        return

    # --- TorchAO PTQ export path (v0.53.1 #142) ---
    if fmt == "torchao":
        output_path = _export_torchao_cli(
            model_path, output, quant_config, trust_remote_code,
        )
        if output_path is not None:
            _maybe_attach_export(
                artifact_path=str(output_path), kind="torchao",
                explicit_id=registry_id, source_model=str(Path(model)),
            )
        return

    # --- Advanced GGUF export path (UD / IQ / Apple-ARM, v0.53.1 #139) ---
    if fmt == "gguf-ud":
        output_path = _export_gguf_advanced(
            model_path=model_path,
            output=output,
            base=base,
            gguf_flavour=gguf_flavour,
            calibration_data=calibration_data,
            llama_cpp_path=llama_cpp_path,
            trust_remote_code=trust_remote_code,
        )
        if output_path is not None:
            _maybe_attach_export(
                artifact_path=str(output_path), kind="gguf",
                explicit_id=registry_id, source_model=str(Path(model)),
            )
            if deploy:
                _auto_deploy_ollama(output_path, model_name, deploy, deploy_name)
        return

    # --- BitNet 1.58-bit / TQ1_0 GGUF (v0.71.20 #134) ---
    # Live llama.cpp TQ1_0 ternary export. Requires a built llama.cpp
    # toolchain; the convert/quantize binaries surface a friendly
    # FileNotFoundError when absent (infra-blocked, mirrors gguf-ud).
    if fmt in ("bitnet", "tq1_0"):
        output_path = _export_bitnet_gguf(
            model_path=model_path,
            output=output,
            base=base,
            export_format=fmt,
            llama_cpp_path=llama_cpp_path,
            trust_remote_code=trust_remote_code,
        )
        if output_path is not None:
            _maybe_attach_export(
                artifact_path=str(output_path), kind="gguf",
                explicit_id=registry_id, source_model=str(Path(model)),
            )
            if deploy:
                _auto_deploy_ollama(output_path, model_name, deploy, deploy_name)
        return

    if quant not in GGUF_QUANT_TYPES:
        console.print(
            f"[red]Unsupported quantization: {quant}[/]\n"
            f"Supported: {', '.join(GGUF_QUANT_TYPES)}"
        )
        raise typer.Exit(1)

    if not (model_path / "adapter_config.json").exists():
        _check_gguf_capabilities(model_path, llama_cpp_path, probe_selected=not no_install_deps)

    # --- Resolve the destination before any merge/convert work ---
    # #1445 — a missing output parent used to surface as FileNotFoundError
    # from mkdtemp AFTER an adapter was merged, and the cleanup then deleted
    # the merged model. Create the parent up front (what ``_run_convert``
    # already did for the f16 path) so no finished work is discarded.
    if output:
        output_path = Path(output)
    else:
        output_path = Path(model).parent / f"{model_name}.{quant}.gguf"

    # Managed environments must refuse missing deps before a merge or toolchain clone.
    if no_install_deps:
        console.print("[dim]GGUF: dependencies (managed environment)...[/]")
        _install_convert_deps(install=False)
        if not (model_path / "adapter_config.json").exists():
            _check_gguf_capabilities(model_path, llama_cpp_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Check if LoRA adapter (needs merge first) ---
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    merge_dir = None

    if is_adapter:
        console.print("[dim]GGUF: merge adapter...[/]")
        console.print("[yellow]LoRA adapter detected - merging with base model first...[/]")
        base_model = base or _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)

        merge_dir = model_path.parent / f".soup_merge_tmp_{model_path.name}"
        _merge_adapter(
            str(model_path), base_model, str(merge_dir), trust_remote_code,
        )
        model_path = merge_dir

        try:
            _check_gguf_capabilities(model_path, llama_cpp_path)
        except typer.Exit:
            shutil.rmtree(merge_dir, ignore_errors=True)
            raise

    # --- Find llama.cpp ---
    console.print("[dim]GGUF: toolchain lookup...[/]")
    llama_dir = _find_llama_cpp(llama_cpp_path)
    # #144 G2 — this used to run only after an auto-clone, so `--llama-cpp` and an
    # already-present ~/.soup/llama.cpp died on `ModuleNotFoundError:
    # sentencepiece`. Here it covers every branch, and returns immediately when
    # the packages are already importable.
    if not no_install_deps:
        console.print("[dim]GGUF: conversion dependencies...[/]")
        _install_convert_deps()
    tool_options = {"verbose": True} if verbose else {}
    convert_options = dict(tool_options)

    # --- Convert to GGUF ---
    console.print(
        Panel(
            f"Model:  [bold]{model_path}[/]\n"
            f"Format: [bold]{fmt}[/]\n"
            f"Quant:  [bold]{quant}[/]\n"
            f"Output: [bold]{output_path}[/]",
            title="Export Plan",
        )
    )

    try:
        # Step 1: Convert HF model to GGUF (f16)
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            console.print(
                f"[red]convert_hf_to_gguf.py not found in {llama_dir}[/]\n"
                "Make sure llama.cpp is properly cloned."
            )
            raise typer.Exit(1)

        from soup_cli.utils.gguf_mtp import needs_no_mtp

        try:
            if needs_no_mtp(model_path, convert_script):
                convert_options["no_mtp"] = True
        except ValueError as exc:
            from soup_cli.utils.terminal import for_terminal

            console.print(f"[red]{for_terminal(str(exc))}[/]")
            raise typer.Exit(1) from exc

        if quant in ("f16", "f32"):
            # Direct conversion without quantization
            outtype = "f16" if quant == "f16" else "f32"
            console.print(f"[dim]GGUF: convert — Converting to GGUF ({outtype})...[/]")
            _run_convert(convert_script, model_path, output_path, outtype, **convert_options)
        else:
            # Convert to f16 first, then quantize.
            #
            # #144 G1 — the intermediate used to be `{model_name}.f16.gguf` next to
            # the output, which is EXACTLY the default output name of a
            # `--quant f16` export, and it was unlinked when quantisation finished.
            # So exporting q4_0 DELETED a previously exported f16 GGUF, even with an
            # unrelated --output. A private temp directory cannot collide with any
            # file the user owns, and removing the whole directory keeps the
            # multi-gigabyte intermediate from being left behind.
            tmp_dir = Path(
                tempfile.mkdtemp(prefix=".soup_gguf_", dir=str(output_path.parent))
            )
            try:
                f16_path = tmp_dir / f"{model_name}.f16.gguf"
                console.print("[dim]GGUF: convert — Converting to GGUF (f16)...[/]")
                _run_convert(convert_script, model_path, f16_path, "f16", **convert_options)

                # Quantize
                console.print(f"[dim]GGUF: quantize — Quantizing to {quant}...[/]")
                _run_quantize(llama_dir, f16_path, output_path, quant, **tool_options)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    finally:
        # Clean up temporary merge directory
        if merge_dir and merge_dir.exists():
            console.print("[dim]Cleaning up temporary merge files...[/]")
            shutil.rmtree(merge_dir, ignore_errors=True)

    if not output_path.exists():
        console.print("[red]Export failed - output file not created.[/]")
        raise typer.Exit(1)

    # v0.33.0 #35: optional auto-attach to registry entry
    _maybe_attach_export(
        artifact_path=str(output_path), kind="gguf",
        explicit_id=registry_id, source_model=str(Path(model)),
    )

    file_size = output_path.stat().st_size
    size_str = _format_size(file_size)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Size:   [bold]{size_str}[/]\n"
            f"Quant:  [bold]{quant}[/]\n\n"
            f"Use with Ollama:\n"
            f"  1. Create a Modelfile:\n"
            f"     [bold]echo 'FROM {output_path}' > Modelfile[/]\n"
            f"  2. Create the model:\n"
            f"     [bold]ollama create {model_name} -f Modelfile[/]\n"
            f"  3. Run it:\n"
            f"     [bold]ollama run {model_name}[/]",
            title="[bold green]Export Complete![/]",
        )
    )

    # --- Auto-deploy to Ollama if requested ---
    if deploy:
        _auto_deploy_ollama(output_path, model_name, deploy, deploy_name)


def _detect_base_model(adapter_config_path: Path) -> Optional[str]:
    """Read base_model_name_or_path from adapter_config.json."""
    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def _merge_adapter(
    adapter_path: str,
    base_model: str,
    output_dir: str,
    trust_remote_code: bool = False,
):
    """Merge LoRA adapter with base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    requires = model_requires_trust_remote_code(adapter_path) or False
    trc = resolve_trust_remote_code(
        base_model,
        requested=trust_remote_code,
        console=console,
        requires_remote_code=requires,
    )

    console.print(f"[dim]Loading base model: {for_terminal(base_model)}...[/]")
    from soup_cli.utils.run_provenance import recorded_base_kwargs

    revision_kwargs = recorded_base_kwargs(adapter_path, base_model)
    if revision_kwargs:
        console.print(f"[dim]Base revision: {revision_kwargs['revision']} (recorded run)[/]")
    else:
        console.print("[dim]Base revision: unknown (legacy, local source or base override)[/]")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=trc,
        device_map="cpu",
        **revision_kwargs,
    )

    console.print(f"[dim]Loading LoRA adapter: {adapter_path}...[/]")
    model = PeftModel.from_pretrained(model, adapter_path)

    console.print("[dim]Merging weights...[/]")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trc)
    tokenizer.save_pretrained(str(out))
    console.print("[green]Adapter merged successfully.[/]")


# llama.cpp's own requirements.txt pins `torch~=2.2.1` against the CPU wheel
# index plus an old `transformers`. Installing it into the user's interpreter
# silently DOWNGRADES a CUDA torch to CPU-only and breaks their training setup
# (observed live on Windows during the v0.71.35 GGUF validation: torch
# 2.5.1+cu -> 2.2.2+cpu, transformers 4.57 -> 4.46). Soup's `[train]` extra
# already provides torch / transformers / numpy, so install ONLY the extra
# packages the convert script needs, unpinned, and never touch the rest.
_CONVERT_EXTRA_DEPS = ("gguf", "sentencepiece", "protobuf")


def _missing_convert_deps() -> list[str]:
    """Name missing convert packages without importing the heavy training stack."""
    import importlib.util

    missing = []
    for module, package in (("gguf", "gguf"), ("sentencepiece", "sentencepiece"),
                            ("google.protobuf", "protobuf")):
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(package)
    return missing


def _convert_deps_present() -> bool:
    """Are the convert script's extra deps already importable?"""
    return not _missing_convert_deps()


def _install_convert_deps(*, install: bool = True) -> None:
    """Install the convert script's extra deps without disturbing torch.

    #144 G2 — this used to run ONLY after an auto-clone, so `--llama-cpp /path`
    and an already-present `~/.soup/llama.cpp` both skipped it and every
    conversion died on `ModuleNotFoundError: No module named 'sentencepiece'`.
    It is now called on every path, and returns immediately when the packages are
    already importable so an ordinary export does not shell out to pip.
    """
    if not install:
        missing = _missing_convert_deps()
        if missing:
            console.print(
                "[red]--no-install-deps: missing GGUF conversion packages: "
                f"{', '.join(missing)}.[/] Install them in the managed environment and retry."
            )
            raise typer.Exit(1)
        return
    if _convert_deps_present():
        return
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *_CONVERT_EXTRA_DEPS],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # Non-fatal: the user may already have them, or manage deps themselves.
        detail = getattr(exc, "stderr", "") or type(exc).__name__
        console.print(
            "[yellow]Could not auto-install the GGUF convert dependencies "
            f"({', '.join(_CONVERT_EXTRA_DEPS)}).[/]\n"
            f"[dim]{escape(str(detail)[:200])}[/]\n"
            "Install them manually if the conversion fails."
        )


def _existing_llama_cpp(user_path: Optional[str] = None) -> Optional[Path]:
    """Select an existing explicit, environment or cached checkout without cloning."""
    from soup_cli.utils.constants import SOUP_DIR

    # 1. User-specified path
    if user_path:
        path = Path(user_path)
        if path.exists():
            return path
        console.print(f"[red]llama.cpp not found at: {path}[/]")
        raise typer.Exit(1)

    # 2. LLAMA_CPP_PATH env var
    import os

    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 3. Check ~/.soup/llama.cpp
    # SOUP_DIR is a bare name (".soup"), so it MUST be anchored to the home
    # directory the way tracker.py / registry/store.py do. Using it relatively
    # made the lookup cwd-dependent: llama.cpp was never found in the canonical
    # ~/.soup, and the auto-clone dropped a fresh ~200 MB checkout into whatever
    # directory the user happened to run from (v0.71.35 GGUF validation).
    soup_llama = Path.home() / SOUP_DIR / LLAMA_CPP_DIR_NAME
    if soup_llama.exists() and (soup_llama / "convert_hf_to_gguf.py").exists():
        return soup_llama
    return None


def _check_gguf_capabilities(
    model: Path, user_path: Optional[str], *, probe_selected: bool = True,
) -> None:
    from soup_cli.utils.gguf_capabilities import (
        model_text_architecture,
        pinned_text_architectures,
        probe_text_architectures,
    )

    try:
        architecture = model_text_architecture(model)
        selected = _existing_llama_cpp(user_path)
        if selected is None:
            supported = pinned_text_architectures(LLAMA_CPP_TAG)
            toolchain = f"default llama.cpp {LLAMA_CPP_TAG}"
        else:
            if not probe_selected:
                return
            supported = probe_text_architectures(selected / "convert_hf_to_gguf.py")
            toolchain = f"selected converter {selected}"
            if supported is None:
                console.print(
                    "[dim]GGUF: capability probe inconclusive; the selected converter "
                    "will validate the architecture during conversion.[/]"
                )
                return
        if architecture not in supported:
            raise ValueError(
                f"Architecture {architecture!r} is not supported for text conversion by "
                f"{toolchain}. Select a compatible checkout with --llama-cpp <dir>."
            )
    except ValueError as exc:
        console.print(f"[red]{for_terminal(str(exc))}[/]")
        raise typer.Exit(1) from exc


def _find_llama_cpp(user_path: Optional[str] = None) -> Path:
    """Find or clone llama.cpp, without installing conversion dependencies (#144 G2)."""
    from soup_cli.utils.constants import SOUP_DIR

    selected = _existing_llama_cpp(user_path)
    if selected is not None:
        return selected
    soup_llama = Path.home() / SOUP_DIR / LLAMA_CPP_DIR_NAME

    # 4. Auto-clone
    console.print("[yellow]llama.cpp not found. Cloning to ~/.soup/llama.cpp...[/]")
    console.print("[dim]This is a one-time setup for GGUF export.[/]")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG,
             "https://github.com/ggerganov/llama.cpp.git", str(soup_llama)],
            check=True,
            capture_output=True,
            text=True,
        )
        console.print("[green]llama.cpp cloned successfully.[/]")
        return soup_llama
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Failed to clone llama.cpp: {exc.stderr}[/]")
        console.print(
            "Please clone manually:\n"
            f"  [bold]git clone https://github.com/ggerganov/llama.cpp.git {soup_llama}[/]\n"
            "Or specify path: [bold]--llama-cpp /path/to/llama.cpp[/]"
        )
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print(
            "[red]git not found.[/] Please install git or clone llama.cpp manually:\n"
            f"  [bold]git clone https://github.com/ggerganov/llama.cpp.git {soup_llama}[/]"
        )
        raise typer.Exit(1)


def _run_export_tool(cmd: list[str], label: str, *, verbose: bool) -> None:
    """Keep quiet capture by default; stream merged output with a bounded error tail."""
    if not verbose:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]{label} failed:[/]")
            console.print(strip_control(result.stderr), markup=False, highlight=False)
            raise typer.Exit(1)
        return

    tail: deque[str] = deque(maxlen=100)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            text = strip_control(line.rstrip("\r\n"))
            tail.append(text[-2000:])
            console.print(text, markup=False, highlight=False)
        returncode = process.wait()
    except BaseException:
        # A cancelled export or output failure must not leave the tool running.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if returncode != 0:
        console.print(f"[red]{label} failed (exit {returncode}):[/]")
        console.print("\n".join(tail), markup=False, highlight=False)
        raise typer.Exit(1)


def _run_convert(
    script: Path, model_dir: Path, output_path: Path, outtype: str, *, verbose: bool = False,
    no_mtp: bool = False,
) -> None:
    """Run llama.cpp convert_hf_to_gguf.py script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]
    if no_mtp:
        cmd.append("--no-mtp")
    _run_export_tool(cmd, "Conversion", verbose=verbose)


def _run_quantize(
    llama_dir: Path, input_path: Path, output_path: Path, quant_type: str, *, verbose: bool = False,
) -> None:
    """Run llama-quantize (or llama.cpp/build/bin/llama-quantize)."""
    # Try to find the quantize binary
    quantize_bin = _find_quantize_binary(llama_dir)
    if not quantize_bin:
        console.print(
            "[red]llama-quantize binary not found.[/]\n"
            "Build llama.cpp first:\n"
            f"  [bold]cd {llama_dir} && make llama-quantize[/]\n"
            "Or use [bold]--quant f16[/] to skip quantization."
        )
        raise typer.Exit(1)

    cmd = [str(quantize_bin), str(input_path), str(output_path), quant_type.upper()]
    _run_export_tool(cmd, "Quantization", verbose=verbose)


# MSVC / Xcode are multi-config generators: they nest binaries under a
# per-configuration subdirectory (build/bin/Release/llama-quantize.exe) rather
# than the flat build/bin/ that single-config generators (Make/Ninja) produce.
# Without these, `soup export --format gguf` cannot find a correctly-built
# llama.cpp on Windows (v0.71.35 GGUF-on-Windows validation, #70/#144).
_CMAKE_CONFIG_DIRS = ("Release", "RelWithDebInfo", "MinSizeRel", "Debug")


def _find_quantize_binary(llama_dir: Path) -> Optional[Path]:
    """Find the llama-quantize binary."""
    # Check common locations
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "llama-quantize.exe",
        llama_dir / "llama-quantize",
        llama_dir / "llama-quantize.exe",
        llama_dir / "build" / "llama-quantize",
    ]
    # Multi-config generator layouts (MSVC on Windows, Xcode on macOS).
    for config in _CMAKE_CONFIG_DIRS:
        candidates.append(llama_dir / "build" / "bin" / config / "llama-quantize")
        candidates.append(llama_dir / "build" / "bin" / config / "llama-quantize.exe")
        candidates.append(llama_dir / "build" / config / "llama-quantize.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Check if it's in PATH
    which_result = shutil.which("llama-quantize")
    if which_result:
        return Path(which_result)

    return None


def _export_onnx(
    model_path: Path, output: Optional[str], base: Optional[str],
    task: str = "text-generation",
    trust_remote_code: bool = False,
) -> Optional[Path]:
    """Export model to ONNX format via optimum."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        console.print(
            "[red]optimum not installed.[/]\n"
            "Install with: [bold]pip install \"soup-cli\\[onnx]\"[/]\n"
            "Or directly: [bold]pip install optimum[onnx][/]"
        )
        raise typer.Exit(1)

    # Check if LoRA adapter
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    merge_dir = None
    source_path = model_path

    if is_adapter:
        console.print("[yellow]LoRA adapter detected - merging with base model first...[/]")
        base_model = base or _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)
        merge_dir = model_path.parent / f".soup_merge_tmp_{model_path.name}"
        _merge_adapter(str(model_path), base_model, str(merge_dir), trust_remote_code)
        source_path = merge_dir

    output_path = Path(output) if output else model_path.parent / f"{model_path.name}_onnx"

    console.print(
        Panel(
            f"Model:  [bold]{source_path}[/]\n"
            f"Format: [bold]ONNX[/]\n"
            f"Output: [bold]{output_path}[/]",
            title="Export Plan",
        )
    )

    try:
        console.print(
            "[yellow]Warning: ONNX export may execute custom model code "
            "if the model uses trust_remote_code.[/]"
        )
        console.print("[dim]Exporting to ONNX...[/]")
        main_export(
            model_name_or_path=str(source_path),
            output=str(output_path),
            task=task,
        )
    except Exception as exc:
        console.print(f"[red]ONNX export failed:[/] {exc}")
        raise typer.Exit(1)
    finally:
        if merge_dir and merge_dir.exists():
            console.print("[dim]Cleaning up temporary merge files...[/]")
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Format: [bold]ONNX[/]\n\n"
            f"Use with ONNX Runtime:\n"
            f"  [bold]from optimum.onnxruntime import ORTModelForCausalLM[/]\n"
            f"  [bold]model = ORTModelForCausalLM.from_pretrained('{output_path}')[/]",
            title="[bold green]ONNX Export Complete![/]",
        )
    )
    return output_path


def _export_tensorrt(
    model_path: Path,
    output: Optional[str],
    base: Optional[str],
    trust_remote_code: bool = False,
) -> Optional[Path]:
    """Export model to TensorRT-LLM format."""
    # TensorRT-LLM uses trtllm-build CLI from the tensorrt_llm package
    trtllm_available = False
    try:
        import tensorrt_llm  # noqa: F401

        trtllm_available = True
    except ImportError:
        pass

    if not trtllm_available:
        console.print(
            "[red]tensorrt_llm not installed.[/]\n"
            "Install with: [bold]pip install \"soup-cli\\[tensorrt]\"[/]\n"
            "Or follow: https://github.com/NVIDIA/TensorRT-LLM#installation"
        )
        raise typer.Exit(1)

    try:
        import tensorrt_llm.commands.convert_checkpoint  # noqa: F401
    except ImportError:
        console.print(
            "[red]tensorrt_llm.commands.convert_checkpoint not found in the "
            "installed tensorrt_llm package.[/]\n"
            "This entry point does not exist in current TensorRT-LLM releases; "
            "checkpoint conversion now ships as a per-architecture "
            "examples/<arch>/convert_checkpoint.py script in the NVIDIA/TensorRT-LLM "
            "source tree instead.\n"
            "See: https://github.com/NVIDIA/TensorRT-LLM#installation"
        )
        raise typer.Exit(1)

    # Check if LoRA adapter
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    merge_dir = None
    source_path = model_path

    if is_adapter:
        console.print("[yellow]LoRA adapter detected - merging with base model first...[/]")
        base_model = base or _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)
        merge_dir = model_path.parent / f".soup_merge_tmp_{model_path.name}"
        _merge_adapter(str(model_path), base_model, str(merge_dir), trust_remote_code)
        source_path = merge_dir

    output_path = Path(output) if output else model_path.parent / f"{model_path.name}_trt"

    console.print(
        Panel(
            f"Model:  [bold]{source_path}[/]\n"
            f"Format: [bold]TensorRT-LLM[/]\n"
            f"Output: [bold]{output_path}[/]",
            title="Export Plan",
        )
    )

    try:
        # Step 1: Convert HF model to TensorRT-LLM checkpoint
        console.print("[dim]Converting to TensorRT-LLM checkpoint...[/]")
        ckpt_dir = output_path / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "tensorrt_llm.commands.convert_checkpoint",
                    "--model_dir", str(source_path),
                    "--output_dir", str(ckpt_dir),
                    "--dtype", "float16",
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            console.print("[red]Python executable not found.[/]")
            raise typer.Exit(1)
        if result.returncode != 0:
            console.print(
                f"[red]Checkpoint conversion failed:[/]\n{result.stderr}"
            )
            raise typer.Exit(1)

        # Step 2: Build TensorRT engine
        console.print("[dim]Building TensorRT engine...[/]")
        engine_dir = output_path / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "trtllm-build",
                    "--checkpoint_dir", str(ckpt_dir),
                    "--output_dir", str(engine_dir),
                    "--gemm_plugin", "float16",
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            console.print(
                "[red]trtllm-build not found in PATH.[/]\n"
                "Ensure tensorrt_llm is installed and "
                "trtllm-build is available."
            )
            raise typer.Exit(1)
        if result.returncode != 0:
            console.print(
                f"[red]TensorRT engine build failed:[/]\n{result.stderr}"
            )
            raise typer.Exit(1)

    finally:
        if merge_dir and merge_dir.exists():
            console.print("[dim]Cleaning up temporary merge files...[/]")
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Format: [bold]TensorRT-LLM[/]\n\n"
            f"Use with TensorRT-LLM:\n"
            f"  [bold]import tensorrt_llm[/]\n"
            f"  [bold]runner = tensorrt_llm.ModelRunner.from_dir('{engine_dir}')[/]",
            title="[bold green]TensorRT-LLM Export Complete![/]",
        )
    )
    return output_path


def _auto_deploy_ollama(
    output_path: Path, model_name: str, deploy_target: str, deploy_name: Optional[str]
):
    """Auto-deploy a GGUF file to Ollama after export."""
    if deploy_target != "ollama":
        console.print(
            f"[red]Unsupported deploy target: {deploy_target}[/]\n"
            "Supported: ollama"
        )
        raise typer.Exit(1)

    from soup_cli.utils.ollama import (
        create_modelfile,
        deploy_to_ollama,
        detect_ollama,
        validate_model_name,
    )

    ollama_name = deploy_name or f"soup-{model_name}"

    valid, err = validate_model_name(ollama_name)
    if not valid:
        console.print(f"[red]Invalid deploy name:[/] {err}")
        raise typer.Exit(1)

    version = detect_ollama()
    if not version:
        console.print(
            "[red]Ollama not found -- skipping deploy.[/]\n"
            "Install from: [bold]https://ollama.com[/]"
        )
        raise typer.Exit(1)

    console.print(
        f"\n[green]OK[/] Ollama v{version} detected"
        f" -- deploying as [bold]{ollama_name}[/]"
    )
    console.print(
        "[yellow]Warning:[/] This will overwrite any existing Ollama model "
        f"named '{ollama_name}'."
    )

    # Auto-detect template from soup.yaml, fall back to chatml
    from soup_cli.commands.deploy import _auto_detect_template

    resolved_template = _auto_detect_template() or "chatml"
    modelfile = create_modelfile(gguf_path=output_path, template=resolved_template)
    success, message = deploy_to_ollama(ollama_name, modelfile)
    if not success:
        console.print(f"[red]Deploy failed:[/] {message}")
        raise typer.Exit(1)

    console.print(f"[green]OK[/] Deployed to Ollama: [bold]{ollama_name}[/]")
    console.print(f"Run: [bold]ollama run {ollama_name}[/]")


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    value: float = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _maybe_attach_export(
    *, artifact_path: str, kind: str,
    explicit_id: Optional[str], source_model: str,
) -> None:
    """Attach an exported artifact to a registry entry.

    Resolution order:
      1. ``--registry-id`` (explicit override)
      2. Auto-match by source model output dir
    Silent no-op if no match is found and no explicit id was given. Failures
    are surfaced as warnings, never as a hard CLI exit (the export itself
    succeeded).
    """
    from soup_cli.registry.attach import attach_artifact, lookup_entry_by_output_dir

    entry_id = explicit_id
    if entry_id is None:
        entry_id = lookup_entry_by_output_dir(source_model)
    if entry_id is None:
        return

    target = Path(artifact_path)
    files_to_attach: list[Path] = []
    if target.is_file():
        files_to_attach = [target]
    elif target.is_dir():
        engine_dir = target / "engine"
        if engine_dir.is_dir():
            files_to_attach.extend(sorted(engine_dir.glob("*.engine")))

        for ext in ("*.safetensors", "*.onnx", "*.onnx_data", "*.pt", "*.bin", "*.engine"):
            for f in sorted(target.glob(ext)):
                if f not in files_to_attach:
                    files_to_attach.append(f)

    if not files_to_attach:
        files_to_attach = [target]

    attached_names: list[str] = []
    try:
        for f in files_to_attach:
            attach_artifact(entry_id, path=str(f), kind=kind)
            attached_names.append(for_terminal(f.name))
    except (ValueError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]Could not attach export to registry "
            f"'{entry_id}':[/] {exc}"
        )
        return

    names_str = ", ".join(attached_names)
    console.print(
        f"[green]Attached export to registry entry '{entry_id}' as {kind} ({names_str}).[/]"
    )


# --- v0.53.1 #142 — TorchAO PTQ export CLI dispatch -------------------------


def _export_torchao_cli(
    model_path: Path,
    output: Optional[str],
    quant_config: Optional[str],
    trust_remote_code: bool,
) -> Optional[Path]:
    """Dispatch ``soup export --format torchao``.

    Per v0.53.0 ``validate_quant_config_path`` docstring contract:
    enforce cwd containment + ``os.lstat + S_ISLNK`` rejection at CLI
    dispatch time, not in the schema validator.
    """
    if quant_config is None:
        console.print(
            "[red]--format torchao requires --quant-config <yaml>[/]\n"
            "Example: [bold]soup export --format torchao "
            "--quant-config q.yaml --model ./merged[/]"
        )
        raise typer.Exit(2)

    from soup_cli.utils.save_formats import (
        export_torchao,
        load_quant_config,
        validate_torchao_scheme,
    )
    from soup_cli.utils.torchao_compat import TORCHAO_MIN_VERSION

    try:
        cfg_data = load_quant_config(quant_config)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{for_terminal(str(exc))}[/]")
        raise typer.Exit(2)

    scheme_raw = cfg_data.get("scheme")
    if not isinstance(scheme_raw, str):
        console.print(
            "[red]quant_config must declare a top-level 'scheme: <name>' field.[/]"
        )
        raise typer.Exit(2)
    try:
        scheme = validate_torchao_scheme(scheme_raw)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2)

    if output is None:
        output_path = model_path.parent / f"{model_path.name}.torchao.{scheme}"
    else:
        output_path = Path(output)

    console.print(Panel(
        f"Model:  [bold]{model_path}[/]\n"
        f"Scheme: [bold]{scheme}[/]\n"
        f"Output: [bold]{output_path}[/]",
        title="TorchAO PTQ Export",
    ))

    try:
        export_torchao(
            model_dir=str(model_path),
            output_dir=str(output_path),
            scheme=scheme,
            quant_config_data={k: v for k, v in cfg_data.items() if k != "scheme"},
            trust_remote_code=trust_remote_code,
        )
    except (ImportError, RuntimeError) as exc:
        console.print(f"[red]TorchAO export failed: {exc}[/]")
        console.print(
            "Try: [bold]pip install torchao[/] "
            f"(Soup needs torchao>={TORCHAO_MIN_VERSION})"
        )
        raise typer.Exit(1)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2)

    console.print(Panel(
        f"Output: [bold]{output_path}[/]\n"
        f"Scheme: [bold]{scheme}[/]",
        title="[bold green]TorchAO Export Complete[/]",
    ))
    return output_path


# --- v0.53.1 #139 — Advanced GGUF export CLI dispatch -----------------------


def _export_bitnet_gguf(
    *,
    model_path: Path,
    output: Optional[str],
    base: Optional[str],
    export_format: str,
    llama_cpp_path: Optional[str],
    trust_remote_code: bool,
) -> Optional[Path]:
    """Dispatch ``soup export --format bitnet | tq1_0`` (v0.71.20 #134).

    Reuses the v0.53.1 gguf convert→quantize pipeline with the TQ1_0 ternary
    flavour. Pre-merges a LoRA adapter when one is detected. Requires a built
    llama.cpp toolchain.
    """
    from soup_cli.utils.bitnet import export_bitnet_gguf, validate_bitnet_export

    try:
        canonical = validate_bitnet_export(export_format)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2)

    if output is None:
        output_path = model_path.parent / f"{model_path.name}.{canonical}.gguf"
    else:
        output_path = Path(output)

    adapter_config_path = model_path / "adapter_config.json"
    merge_dir = None
    if adapter_config_path.exists():
        console.print("[yellow]LoRA adapter detected — merging first...[/]")
        base_model = base or _detect_base_model(adapter_config_path)
        if not base_model:
            console.print("[red]Cannot detect base model. Pass --base.[/]")
            raise typer.Exit(2)
        merge_dir = model_path.parent / f".soup_merge_tmp_{model_path.name}"
        _merge_adapter(
            str(model_path), base_model, str(merge_dir), trust_remote_code,
        )
        source_model_dir = merge_dir
    else:
        source_model_dir = model_path

    llama_dir = _find_llama_cpp(llama_cpp_path)

    console.print(Panel(
        f"Model:   [bold]{source_model_dir}[/]\n"
        f"Format:  [bold]{canonical}[/] (TQ1_0 ternary)\n"
        f"Output:  [bold]{output_path}[/]",
        title="BitNet GGUF Export",
    ))

    try:
        export_bitnet_gguf(
            model_dir=str(source_model_dir),
            output_path=str(output_path),
            export_format=canonical,
            llama_cpp_dir=str(llama_dir),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]BitNet GGUF export failed: {exc}[/]")
        raise typer.Exit(1)
    finally:
        if merge_dir and merge_dir.exists():
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(Panel(
        f"Output: [bold]{output_path}[/]\n"
        f"Format: [bold]{canonical}[/]",
        title="[bold green]BitNet GGUF Export Complete[/]",
    ))
    return output_path


def _export_gguf_advanced(
    *,
    model_path: Path,
    output: Optional[str],
    base: Optional[str],
    gguf_flavour: Optional[str],
    calibration_data: Optional[str],
    llama_cpp_path: Optional[str],
    trust_remote_code: bool,
) -> Optional[Path]:
    """Dispatch ``soup export --format gguf-ud --gguf-flavour <...>``.

    Routes through llama.cpp's ``imatrix`` + ``quantize`` binaries. Supports
    UD-Q*_K_XL ladder, IQ*_M family, Apple/ARM Q4_0_4_4 / Q4_NL etc.
    """
    if gguf_flavour is None:
        console.print(
            "[red]--format gguf-ud requires --gguf-flavour <UD-Q4_K_XL | IQ2_M | "
            "Q4_0_4_4 | ...>[/]"
        )
        raise typer.Exit(2)

    from soup_cli.utils.gguf_quant import (
        export_advanced_gguf,
        is_advanced_gguf_format,
    )

    if not is_advanced_gguf_format(gguf_flavour):
        console.print(
            f"[red]Unknown gguf_flavour {gguf_flavour!r}. "
            "See soup_cli.utils.gguf_quant.ALL_ADVANCED_GGUF_FORMATS.[/]"
        )
        raise typer.Exit(2)

    # Calibration data path (UD / IQ require it; Apple/ARM Q4_0_4_4 doesn't).
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink
    if calibration_data is not None:
        try:
            enforce_under_cwd_and_no_symlink(
                calibration_data, "calibration_data",
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2)
        if not Path(calibration_data).is_file():
            console.print(
                f"[red]Calibration data file not found: {calibration_data}[/]"
            )
            raise typer.Exit(2)

    if output is None:
        output_path = (
            model_path.parent / f"{model_path.name}.{gguf_flavour}.gguf"
        )
    else:
        output_path = Path(output)

    # Pre-merge LoRA adapter if needed
    adapter_config_path = model_path / "adapter_config.json"
    merge_dir = None
    if adapter_config_path.exists():
        console.print("[yellow]LoRA adapter detected — merging first...[/]")
        base_model = base or _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model. Pass --base.[/]"
            )
            raise typer.Exit(2)
        merge_dir = model_path.parent / f".soup_merge_tmp_{model_path.name}"
        _merge_adapter(
            str(model_path), base_model, str(merge_dir), trust_remote_code,
        )
        source_model_dir = merge_dir
    else:
        source_model_dir = model_path

    llama_dir = _find_llama_cpp(llama_cpp_path)

    console.print(Panel(
        f"Model:    [bold]{source_model_dir}[/]\n"
        f"Flavour:  [bold]{gguf_flavour}[/]\n"
        f"Calib:    [bold]{calibration_data or '(none — Apple/ARM)'}[/]\n"
        f"Output:   [bold]{output_path}[/]",
        title="Advanced GGUF Export",
    ))

    try:
        export_advanced_gguf(
            model_dir=str(source_model_dir),
            output_path=str(output_path),
            flavour=gguf_flavour,
            calibration_data=calibration_data,
            llama_cpp_dir=str(llama_dir),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Advanced GGUF export failed: {exc}[/]")
        raise typer.Exit(1)
    finally:
        if merge_dir and merge_dir.exists():
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(Panel(
        f"Output: [bold]{output_path}[/]\n"
        f"Flavour: [bold]{gguf_flavour}[/]",
        title="[bold green]Advanced GGUF Export Complete[/]",
    ))
    return output_path
