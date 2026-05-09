"""soup diff — compare outputs of two models side-by-side."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def diff(
    model_a: str = typer.Option(
        ...,
        "--model-a",
        "-a",
        help="Path to first model (LoRA adapter or full model)",
    ),
    model_b: str = typer.Option(
        ...,
        "--model-b",
        "-b",
        help="Path to second model (LoRA adapter or full model)",
    ),
    prompts: Optional[str] = typer.Option(
        None,
        "--prompts",
        "-p",
        help="Path to prompts file (JSONL with 'prompt' field, or one prompt per line)",
    ),
    prompt: Optional[list[str]] = typer.Option(
        None,
        "--prompt",
        help="Single prompt to compare (can be repeated)",
    ),
    base_a: Optional[str] = typer.Option(
        None,
        "--base-a",
        help="Base model for model A (auto-detected for LoRA adapters)",
    ),
    base_b: Optional[str] = typer.Option(
        None,
        "--base-b",
        help="Base model for model B (auto-detected for LoRA adapters)",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        help="Maximum tokens to generate per response",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSONL file",
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
    """Compare outputs of two models side-by-side on the same prompts."""
    # Validate model paths
    path_a = Path(model_a)
    path_b = Path(model_b)
    if not path_a.exists():
        console.print(f"[red]Model A not found: {path_a}[/]")
        raise typer.Exit(1)
    if not path_b.exists():
        console.print(f"[red]Model B not found: {path_b}[/]")
        raise typer.Exit(1)

    # Collect prompts
    prompt_list = _collect_prompts(prompts, prompt)
    if not prompt_list:
        console.print("[red]No prompts provided. Use --prompts or --prompt.[/]")
        raise typer.Exit(1)

    # Detect device
    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    console.print(
        Panel(
            f"Model A:  [bold]{path_a}[/]\n"
            f"Model B:  [bold]{path_b}[/]\n"
            f"Prompts:  [bold]{len(prompt_list)}[/]\n"
            f"Device:   [bold]{device}[/]",
            title="Diff Plan",
        )
    )

    # Load models
    console.print("[dim]Loading Model A...[/]")
    model_obj_a, tokenizer_a = _load_model(
        str(path_a), base_a, device, trust_remote_code,
    )
    console.print("[dim]Loading Model B...[/]")
    model_obj_b, tokenizer_b = _load_model(
        str(path_b), base_b, device, trust_remote_code,
    )
    console.print("[green]Both models loaded.[/]\n")

    # Run comparison
    results = []
    for idx, prompt_text in enumerate(prompt_list):
        console.print(f"[bold]--- Prompt {idx + 1}/{len(prompt_list)} ---[/]")
        console.print(f"[dim]{prompt_text}[/]\n")

        messages = [{"role": "user", "content": prompt_text}]

        response_a = _generate(
            model_obj_a, tokenizer_a, messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        response_b = _generate(
            model_obj_b, tokenizer_b, messages,
            max_tokens=max_tokens, temperature=temperature,
        )

        # Side-by-side display
        panel_a = Panel(
            response_a or "[dim]<empty>[/]",
            title=f"[blue]Model A: {path_a.name}[/]",
            border_style="blue",
            width=console.width // 2 - 1,
        )
        panel_b = Panel(
            response_b or "[dim]<empty>[/]",
            title=f"[green]Model B: {path_b.name}[/]",
            border_style="green",
            width=console.width // 2 - 1,
        )
        console.print(Columns([panel_a, panel_b]))

        # Metrics
        metrics = _compute_metrics(response_a, response_b)
        metrics_str = (
            f"Length: A={metrics['len_a']} / B={metrics['len_b']}  |  "
            f"Words: A={metrics['words_a']} / B={metrics['words_b']}  |  "
            f"Overlap: {metrics['word_overlap']:.0%}"
        )
        console.print(f"[dim]{metrics_str}[/]\n")

        results.append({
            "prompt": prompt_text,
            "response_a": response_a,
            "response_b": response_b,
            "metrics": metrics,
        })

    # Summary
    _display_summary(results, path_a.name, path_b.name)

    # Save results
    if output:
        out_path = Path(output)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(f"[dim]Results saved to {out_path}[/]")


def _collect_prompts(prompts_file: Optional[str], prompt_args: Optional[list[str]]) -> list[str]:
    """Collect prompts from file and/or CLI arguments."""
    result = []

    if prompts_file:
        path = Path(prompts_file)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Try JSONL
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "prompt" in obj:
                            result.append(obj["prompt"])
                            continue
                    except json.JSONDecodeError:
                        pass
                    # Plain text
                    result.append(line)

    if prompt_args:
        result.extend(prompt_args)

    return result


def _load_model(
    model_path: str,
    base_model: Optional[str],
    device: str,
    trust_remote_code: bool = False,
):
    """Load a model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    path = Path(model_path)
    adapter_config_path = path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()

    if is_adapter and not base_model:
        try:
            with open(adapter_config_path, encoding="utf-8") as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError):
            pass

    if is_adapter and not base_model:
        console.print(f"[red]Cannot detect base model for {path}. Use --base-a/--base-b.[/]")
        raise typer.Exit(1)

    probe_target = base_model or model_path
    requires = model_requires_trust_remote_code(model_path) or False
    trc = resolve_trust_remote_code(
        probe_target,
        requested=trust_remote_code,
        console=console,
        requires_remote_code=requires,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trc,
            device_map="auto",
            dtype=torch.float16,
        )
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trc,
            device_map="auto",
            dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _generate(model, tokenizer, messages, max_tokens=256, temperature=0.7) -> str:
    """Generate a response from the model."""
    import torch

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
            gen_kwargs["top_p"] = 0.9
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _compute_metrics(response_a: str, response_b: str) -> dict:
    """Compute comparison metrics between two responses."""
    words_a = set(response_a.lower().split())
    words_b = set(response_b.lower().split())

    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

    return {
        "len_a": len(response_a),
        "len_b": len(response_b),
        "words_a": len(response_a.split()),
        "words_b": len(response_b.split()),
        "word_overlap": overlap,
    }


def _display_summary(results: list[dict], name_a: str, name_b: str):
    """Display summary statistics for the comparison."""
    if not results:
        return

    table = Table(title="Comparison Summary")
    table.add_column("Metric", style="bold")
    table.add_column(f"Model A ({name_a})", justify="right", style="blue")
    table.add_column(f"Model B ({name_b})", justify="right", style="green")

    avg_len_a = sum(r["metrics"]["len_a"] for r in results) / len(results)
    avg_len_b = sum(r["metrics"]["len_b"] for r in results) / len(results)
    avg_words_a = sum(r["metrics"]["words_a"] for r in results) / len(results)
    avg_words_b = sum(r["metrics"]["words_b"] for r in results) / len(results)
    avg_overlap = sum(r["metrics"]["word_overlap"] for r in results) / len(results)

    table.add_row("Avg length (chars)", f"{avg_len_a:.0f}", f"{avg_len_b:.0f}")
    table.add_row("Avg words", f"{avg_words_a:.0f}", f"{avg_words_b:.0f}")
    table.add_row("Avg word overlap", f"{avg_overlap:.0%}", f"{avg_overlap:.0%}")

    console.print(table)
