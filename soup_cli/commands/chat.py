"""soup chat — interactive chat with a fine-tuned model."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def chat(
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
    device: Optional[str] = typer.Option(
        None,
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum number of tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0 = greedy, 1.0 = creative)",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the conversation",
    ),
):
    """Chat with a fine-tuned model in the terminal."""
    model_path = Path(model)

    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Detect if it's a LoRA adapter or full model
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

    # Detect device
    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    console.print(
        Panel(
            f"Model:  [bold]{model_path}[/]\n"
            + (f"Base:   [bold]{base_model}[/]\n" if is_adapter else "")
            + f"Device: [bold]{device}[/]\n"
            f"Type:   [bold]{'LoRA adapter' if is_adapter else 'Full model'}[/]",
            title="Loading model",
        )
    )

    # Load model + tokenizer
    model_obj, tokenizer = _load_model(
        model_path=str(model_path),
        base_model=base_model,
        is_adapter=is_adapter,
        device=device,
    )

    console.print("[bold green]Model loaded![/] Type your message. Commands:")
    console.print("  [dim]/quit[/]  — exit chat")
    console.print("  [dim]/clear[/] — reset conversation history")
    console.print("  [dim]/system <text>[/] — set system prompt")
    console.print()

    # Chat loop
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = console.input("[bold blue]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye![/]")
            break
        elif user_input.lower() == "/clear":
            history = []
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
            console.print("[dim]Conversation cleared.[/]\n")
            continue
        elif user_input.lower().startswith("/system "):
            new_system = user_input[8:].strip()
            # Remove old system prompt if exists
            history = [msg for msg in history if msg["role"] != "system"]
            history.insert(0, {"role": "system", "content": new_system})
            console.print(f"[dim]System prompt set: {new_system}[/]\n")
            continue

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Generate response
        response = _generate(
            model_obj, tokenizer, history,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )

        history.append({"role": "assistant", "content": response})
        console.print(f"[bold green]Assistant:[/] {response}\n")


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
):
    """Load model and tokenizer. Supports LoRA adapters and full models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("[dim]Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        console.print(f"[dim]Loading base model: {base_model}...[/]")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        console.print(f"[dim]Loading LoRA adapter: {model_path}...[/]")
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        console.print(f"[dim]Loading model: {model_path}...[/]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _generate(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """Generate a response from the model given message history."""
    import torch

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
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

    # Decode only new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
