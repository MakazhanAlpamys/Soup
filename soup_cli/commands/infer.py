"""soup infer — batch inference on a list of prompts."""

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def infer(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to model (LoRA adapter or full model)",
    ),
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input JSONL file (each line: {\"prompt\": \"...\"})",
    ),
    output_file: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to output JSONL file for results",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model for LoRA adapter (auto-detected if not set)",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        min=1,
        max=16384,
        help="Maximum tokens to generate per response (1-16384)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0 = greedy)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
):
    """Run batch inference on a JSONL file of prompts."""
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/]")
        raise typer.Exit(1)

    # Validate model path
    model_path = Path(model)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/]")
        raise typer.Exit(1)

    # Read prompts
    prompts = _read_prompts(input_path)
    if not prompts:
        console.print("[red]No prompts found in input file.[/]")
        console.print("[dim]Expected JSONL with {\"prompt\": \"...\"} or plain text lines.[/]")
        raise typer.Exit(1)

    # Detect device
    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    console.print(
        Panel(
            f"Model:    [bold]{model_path}[/]\n"
            f"Input:    [bold]{input_path}[/] ({len(prompts)} prompts)\n"
            f"Output:   [bold]{output_file}[/]\n"
            f"Device:   [bold]{device}[/]\n"
            f"Tokens:   [bold]{max_tokens}[/]\n"
            f"Temp:     [bold]{temperature}[/]",
            title="Batch Inference",
        )
    )

    # Load model
    console.print(
        "[yellow]Warning: loading model with trust_remote_code=True. "
        "Only use models you trust.[/]"
    )
    console.print("[dim]Loading model...[/]")
    model_obj, tokenizer = _load_model(str(model_path), base, device)
    console.print("[green]Model loaded.[/]\n")

    # Run inference — stream results to disk as they are generated
    output_path = Path(output_file)
    total_tokens = 0
    num_results = 0
    start_time = time.time()

    with (
        open(output_path, "w", encoding="utf-8") as out_f,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("Generating...", total=len(prompts))

        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            response, token_count = _generate(
                model_obj, tokenizer, messages,
                max_tokens=max_tokens, temperature=temperature,
            )

            result = {
                "prompt": prompt_text,
                "response": response,
                "tokens_generated": token_count,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            total_tokens += token_count
            num_results += 1
            progress.update(task, advance=1)

    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    console.print(
        Panel(
            f"Prompts:       [bold]{num_results}[/]\n"
            f"Total tokens:  [bold]{total_tokens}[/]\n"
            f"Duration:      [bold]{elapsed:.1f}s[/]\n"
            f"Throughput:    [bold]{tokens_per_sec:.1f} tok/s[/]\n"
            f"Output:        [bold]{output_path}[/]",
            title="[bold green]Inference Complete![/]",
        )
    )


def _read_prompts(path: Path) -> list[str]:
    """Read prompts from a JSONL or plain text file."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # Try JSONL
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "prompt" in obj:
                    prompts.append(obj["prompt"])
                    continue
            except json.JSONDecodeError:
                pass
            # Plain text
            prompts.append(line)
    return prompts


def _load_model(model_path: str, base_model: Optional[str], device: str) -> tuple:
    """Load a model and tokenizer (reuses diff.py pattern)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        console.print(
            f"[red]Cannot detect base model for {path}. Use --base.[/]"
        )
        raise typer.Exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        base_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
        )
        model_obj = PeftModel.from_pretrained(base_obj, model_path)
    else:
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _generate(
    model, tokenizer, messages, max_tokens=256, temperature=0.7,
) -> tuple[str, int]:
    """Generate a response from the model. Returns (text, token_count)."""
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
    token_count = new_tokens.shape[0]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response_text, token_count
