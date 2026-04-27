"""soup export — convert a model to GGUF format for Ollama / llama.cpp."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

SUPPORTED_FORMATS = ("gguf", "onnx", "tensorrt", "awq", "gptq")
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
        help="Export format: gguf, onnx, tensorrt",
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
    bits: int = typer.Option(
        4,
        "--bits",
        help="Quantization bits for AWQ/GPTQ: 4 or 8",
    ),
    group_size: int = typer.Option(
        128,
        "--group-size",
        help="Group size for AWQ/GPTQ quantization",
    ),
    calibration_data: Optional[str] = typer.Option(
        None,
        "--calibration-data",
        help="Path to calibration JSONL for AWQ/GPTQ (default: use built-in sample)",
    ),
    calibration_samples: int = typer.Option(
        128,
        "--calibration-samples",
        help="Number of calibration samples for AWQ/GPTQ",
    ),
    registry_id: Optional[str] = typer.Option(
        None,
        "--registry-id",
        help="Attach exported artifact to this registry entry "
        "(default: auto-match by source --model output dir)",
    ),
):
    """Export a model to GGUF, ONNX, TensorRT-LLM, AWQ, or GPTQ format."""
    model_path = Path(model)

    # --- Validate ---
    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    if fmt not in SUPPORTED_FORMATS:
        console.print(
            f"[red]Unsupported format: {fmt}[/]\n"
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
        raise typer.Exit(1)

    # --- ONNX export path ---
    if fmt == "onnx":
        _export_onnx(model_path, output, base, onnx_task)
        return

    # --- TensorRT-LLM export path ---
    if fmt == "tensorrt":
        _export_tensorrt(model_path, output, base)
        return

    # --- AWQ export path ---
    if fmt == "awq":
        _export_awq(
            model_path, output, base, bits, group_size,
            calibration_data, calibration_samples,
        )
        return

    # --- GPTQ export path ---
    if fmt == "gptq":
        _export_gptq(
            model_path, output, base, bits, group_size,
            calibration_data, calibration_samples,
        )
        return

    if quant not in GGUF_QUANT_TYPES:
        console.print(
            f"[red]Unsupported quantization: {quant}[/]\n"
            f"Supported: {', '.join(GGUF_QUANT_TYPES)}"
        )
        raise typer.Exit(1)

    # --- Check if LoRA adapter (needs merge first) ---
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    merge_dir = None

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
        _merge_adapter(str(model_path), base_model, str(merge_dir))
        model_path = merge_dir

    # --- Find llama.cpp ---
    llama_dir = _find_llama_cpp(llama_cpp_path)

    # --- Convert to GGUF ---
    model_name = Path(model).name
    if output:
        output_path = Path(output)
    else:
        output_path = Path(model).parent / f"{model_name}.{quant}.gguf"

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

        if quant in ("f16", "f32"):
            # Direct conversion without quantization
            outtype = "f16" if quant == "f16" else "f32"
            console.print(f"[dim]Converting to GGUF ({outtype})...[/]")
            _run_convert(convert_script, model_path, output_path, outtype)
        else:
            # Convert to f16 first, then quantize
            f16_path = output_path.parent / f"{model_name}.f16.gguf"
            console.print("[dim]Converting to GGUF (f16)...[/]")
            _run_convert(convert_script, model_path, f16_path, "f16")

            # Quantize
            console.print(f"[dim]Quantizing to {quant}...[/]")
            _run_quantize(llama_dir, f16_path, output_path, quant)

            # Clean up intermediate f16 file
            if f16_path.exists() and f16_path != output_path:
                f16_path.unlink()

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


def _merge_adapter(adapter_path: str, base_model: str, output_dir: str):
    """Merge LoRA adapter with base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"[dim]Loading base model: {base_model}...[/]")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
    )

    console.print(f"[dim]Loading LoRA adapter: {adapter_path}...[/]")
    model = PeftModel.from_pretrained(model, adapter_path)

    console.print("[dim]Merging weights...[/]")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(out))
    console.print("[green]Adapter merged successfully.[/]")


def _find_llama_cpp(user_path: Optional[str] = None) -> Path:
    """Find or clone llama.cpp directory."""
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
    soup_llama = Path(SOUP_DIR) / LLAMA_CPP_DIR_NAME
    if soup_llama.exists() and (soup_llama / "convert_hf_to_gguf.py").exists():
        return soup_llama

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
        # Install Python requirements for the convert script
        requirements = soup_llama / "requirements.txt"
        if requirements.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements), "-q"],
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


def _run_convert(script: Path, model_dir: Path, output_path: Path, outtype: str):
    """Run llama.cpp convert_hf_to_gguf.py script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Conversion failed:[/]\n{result.stderr}")
        raise typer.Exit(1)


def _run_quantize(llama_dir: Path, input_path: Path, output_path: Path, quant_type: str):
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Quantization failed:[/]\n{result.stderr}")
        raise typer.Exit(1)


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
):
    """Export model to ONNX format via optimum."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        console.print(
            "[red]optimum not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[onnx]'[/]\n"
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
        _merge_adapter(str(model_path), base_model, str(merge_dir))
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


def _export_tensorrt(model_path: Path, output: Optional[str], base: Optional[str]):
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
            "Install with: [bold]pip install 'soup-cli[tensorrt]'[/]\n"
            "Or follow: https://github.com/NVIDIA/TensorRT-LLM#installation"
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
        _merge_adapter(str(model_path), base_model, str(merge_dir))
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


def _validate_output_path(output: Optional[str]) -> Optional[Path]:
    """Validate output path stays under cwd (path traversal protection)."""
    if output is None:
        return None
    out_path = Path(output).resolve()
    cwd = Path.cwd().resolve()
    try:
        out_path.relative_to(cwd)
    except ValueError:
        console.print("[red]Output path must be under the current working directory.[/]")
        raise typer.Exit(1)
    return out_path


def _validate_calibration_path(calibration_data: Optional[str]) -> Optional[Path]:
    """Validate calibration data path stays under cwd."""
    if calibration_data is None:
        return None
    cal_path = Path(calibration_data).resolve()
    cwd = Path.cwd().resolve()
    try:
        cal_path.relative_to(cwd)
    except ValueError:
        console.print("[red]Calibration data path must be under the current working directory.[/]")
        raise typer.Exit(1)
    if not cal_path.exists():
        console.print(f"[red]Calibration data not found: {cal_path}[/]")
        raise typer.Exit(1)
    return cal_path


def _load_calibration_texts(cal_path: Optional[Path], max_samples: int = 128) -> list:
    """Load calibration texts from JSONL file."""
    if cal_path is None:
        return []
    texts = []
    with open(cal_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Support "text" field or concatenate all string values
                if "text" in row:
                    texts.append(str(row["text"]))
                else:
                    texts.append(" ".join(str(v) for v in row.values() if v))
            except json.JSONDecodeError:
                continue
            if len(texts) >= max_samples:
                break
    return texts


def _export_awq(
    model_path: Path,
    output: Optional[str],
    base: Optional[str],
    bits: int = 4,
    group_size: int = 128,
    calibration_data: Optional[str] = None,
    calibration_samples: int = 128,
) -> None:
    """Export model to AWQ format via autoawq."""
    # Validate bits
    valid_bits = {4, 8}
    if bits not in valid_bits:
        console.print(
            f"[red]Invalid --bits {bits}. Must be one of: {sorted(valid_bits)}[/]"
        )
        raise typer.Exit(1)

    # Validate output path (security: path traversal protection)
    validated_output = _validate_output_path(output)

    # Validate calibration path (security: path traversal protection)
    cal_path = _validate_calibration_path(calibration_data)

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        console.print(
            "[red]autoawq not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[awq]'[/]\n"
            "Or directly: [bold]pip install autoawq[/]"
        )
        raise typer.Exit(1)

    from transformers import AutoTokenizer

    # Check if LoRA adapter — merge first
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
        _merge_adapter(str(model_path), base_model, str(merge_dir))
        source_path = merge_dir

    default_out = model_path.parent / f"{model_path.name}_awq"
    output_path = validated_output if validated_output else default_out

    console.print(
        Panel(
            f"Model:      [bold]{source_path}[/]\n"
            f"Format:     [bold]AWQ[/]\n"
            f"Bits:       [bold]{bits}[/]\n"
            f"Group size: [bold]{group_size}[/]\n"
            f"Output:     [bold]{output_path}[/]",
            title="Export Plan",
        )
    )

    try:
        console.print(
            Panel(
                "[yellow]Warning:[/] Loading model with trust_remote_code=True.\n"
                "This may execute custom code from the model directory.",
                title="Security Notice",
            )
        )
        console.print("[dim]Loading model for AWQ quantization...[/]")
        model = AutoAWQForCausalLM.from_pretrained(str(source_path))
        tokenizer = AutoTokenizer.from_pretrained(str(source_path), trust_remote_code=True)

        quant_config = {"zero_point": True, "q_group_size": group_size, "w_bit": bits}

        # Load calibration data if provided
        calib_data = (
            _load_calibration_texts(cal_path, max_samples=calibration_samples)
            if cal_path else None
        )

        console.print(f"[dim]Quantizing to AWQ {bits}-bit (group_size={group_size})...[/]")
        if calib_data:
            model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
        else:
            model.quantize(tokenizer, quant_config=quant_config)

        console.print("[dim]Saving quantized model...[/]")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

    except Exception as exc:
        console.print(f"[red]AWQ export failed:[/] {exc}")
        raise typer.Exit(1)
    finally:
        if merge_dir and merge_dir.exists():
            console.print("[dim]Cleaning up temporary merge files...[/]")
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Format: [bold]AWQ {bits}-bit[/]\n\n"
            f"Use with vLLM:\n"
            f"  [bold]from vllm import LLM[/]\n"
            f"  [bold]llm = LLM(model='{output_path}', quantization='awq')[/]",
            title="[bold green]AWQ Export Complete![/]",
        )
    )


def _export_gptq(
    model_path: Path,
    output: Optional[str],
    base: Optional[str],
    bits: int = 4,
    group_size: int = 128,
    calibration_data: Optional[str] = None,
    calibration_samples: int = 128,
) -> None:
    """Export model to GPTQ format via auto-gptq."""
    # Validate bits
    valid_bits = {4, 8}
    if bits not in valid_bits:
        console.print(
            f"[red]Invalid --bits {bits}. Must be one of: {sorted(valid_bits)}[/]"
        )
        raise typer.Exit(1)

    # Validate output path (security: path traversal protection)
    validated_output = _validate_output_path(output)

    # Validate calibration path (security: path traversal protection)
    cal_path = _validate_calibration_path(calibration_data)

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        console.print(
            "[red]auto-gptq not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[gptq]'[/]\n"
            "Or directly: [bold]pip install auto-gptq[/]"
        )
        raise typer.Exit(1)

    from transformers import AutoTokenizer

    # Check if LoRA adapter — merge first
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
        _merge_adapter(str(model_path), base_model, str(merge_dir))
        source_path = merge_dir

    default_out = model_path.parent / f"{model_path.name}_gptq"
    output_path = validated_output if validated_output else default_out

    console.print(
        Panel(
            f"Model:      [bold]{source_path}[/]\n"
            f"Format:     [bold]GPTQ[/]\n"
            f"Bits:       [bold]{bits}[/]\n"
            f"Group size: [bold]{group_size}[/]\n"
            f"Output:     [bold]{output_path}[/]",
            title="Export Plan",
        )
    )

    try:
        console.print(
            Panel(
                "[yellow]Warning:[/] Loading model with trust_remote_code=True.\n"
                "This may execute custom code from the model directory.",
                title="Security Notice",
            )
        )
        console.print("[dim]Loading model for GPTQ quantization...[/]")
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False,
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            str(source_path), quantize_config=quantize_config
        )
        tokenizer = AutoTokenizer.from_pretrained(str(source_path), trust_remote_code=True)

        # Load calibration data if provided
        calib_data = None
        if cal_path:
            texts = _load_calibration_texts(
                cal_path, max_samples=calibration_samples,
            )
            calib_data = [tokenizer(t, return_tensors="pt") for t in texts]

        console.print(f"[dim]Quantizing to GPTQ {bits}-bit (group_size={group_size})...[/]")
        if calib_data:
            model.quantize(calib_data)
        else:
            model.quantize(tokenizer)

        console.print("[dim]Saving quantized model...[/]")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

    except Exception as exc:
        console.print(f"[red]GPTQ export failed:[/] {exc}")
        raise typer.Exit(1)
    finally:
        if merge_dir and merge_dir.exists():
            console.print("[dim]Cleaning up temporary merge files...[/]")
            shutil.rmtree(merge_dir, ignore_errors=True)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Format: [bold]GPTQ {bits}-bit[/]\n\n"
            f"Use with vLLM:\n"
            f"  [bold]from vllm import LLM[/]\n"
            f"  [bold]llm = LLM(model='{output_path}', quantization='gptq')[/]",
            title="[bold green]GPTQ Export Complete![/]",
        )
    )


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
    try:
        attach_artifact(entry_id, path=artifact_path, kind=kind)
    except (ValueError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]Could not attach export to registry "
            f"'{entry_id}':[/] {exc}"
        )
        return
    console.print(
        f"[green]Attached export to registry entry '{entry_id}' as {kind}.[/]"
    )
