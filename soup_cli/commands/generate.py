"""soup data generate — generate synthetic training data using LLMs."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

logger = logging.getLogger(__name__)

console = Console()


def generate(
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="System prompt describing what kind of data to generate",
    ),
    count: int = typer.Option(
        100,
        "--count",
        "-n",
        help="Number of examples to generate",
    ),
    output: str = typer.Option(
        "generated.jsonl",
        "--output",
        "-o",
        help="Output file path",
    ),
    fmt: str = typer.Option(
        "alpaca",
        "--format",
        "-f",
        help="Output format: alpaca, sharegpt, chatml",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        help="LLM provider: openai, local, server (local OpenAI-compatible server)",
    ),
    model_name: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="Model name (OpenAI model ID or local model path)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="[deprecated] Use OPENAI_API_KEY env var instead",
        envvar="OPENAI_API_KEY",
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="Custom API base URL (must use HTTPS for remote APIs)",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        help="Number of examples per API call",
    ),
    temperature: float = typer.Option(
        0.8,
        "--temperature",
        "-t",
        help="Sampling temperature for generation",
    ),
    dedup_with: Optional[str] = typer.Option(
        None,
        "--dedup-with",
        help="Path to existing dataset to deduplicate against",
    ),
    seed_file: Optional[str] = typer.Option(
        None,
        "--seed",
        help="Path to seed examples file (JSONL) to guide generation",
    ),
):
    """Generate synthetic training data using an LLM."""
    valid_formats = ("alpaca", "sharegpt", "chatml")
    if fmt not in valid_formats:
        console.print(f"[red]Invalid format: {fmt}. Must be one of: {', '.join(valid_formats)}[/]")
        raise typer.Exit(1)

    valid_providers = ("openai", "local", "server")
    if provider not in valid_providers:
        console.print(
            f"[red]Invalid provider: {provider}. Must be one of: {', '.join(valid_providers)}[/]"
        )
        raise typer.Exit(1)

    # Load seed examples if provided
    seed_examples = []
    if seed_file:
        seed_path = Path(seed_file)
        if not seed_path.exists():
            console.print(f"[red]Seed file not found: {seed_path}[/]")
            raise typer.Exit(1)
        from soup_cli.data.loader import load_raw_data

        seed_examples = load_raw_data(seed_path)
        console.print(f"[dim]Loaded {len(seed_examples)} seed examples[/]")

    # Load existing data for dedup
    existing_texts = set()
    if dedup_with:
        dedup_path = Path(dedup_with)
        if not dedup_path.exists():
            console.print(f"[red]Dedup file not found: {dedup_path}[/]")
            raise typer.Exit(1)
        from soup_cli.data.loader import load_raw_data

        existing_data = load_raw_data(dedup_path)
        for row in existing_data:
            existing_texts.add(_row_to_text(row))
        console.print(f"[dim]Loaded {len(existing_texts)} existing examples for dedup[/]")

    # Generate
    console.print(f"[dim]Generating {count} examples using {provider}/{model_name}...[/]")

    all_examples = []
    duplicates = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=count)

        remaining = count
        while remaining > 0:
            current_batch = min(batch_size, remaining)

            try:
                batch = _generate_batch(
                    prompt=prompt,
                    count=current_batch,
                    fmt=fmt,
                    provider=provider,
                    model_name=model_name,
                    api_key=api_key,
                    api_base=api_base,
                    temperature=temperature,
                    seed_examples=seed_examples,
                )
            except Exception as exc:
                console.print(f"[red]Generation error: {exc}[/]")
                raise typer.Exit(1)

            # Validate and dedup
            for example in batch:
                if not _validate_example(example, fmt):
                    continue
                text = _row_to_text(example)
                if text in existing_texts:
                    duplicates += 1
                    continue
                existing_texts.add(text)
                all_examples.append(example)

            generated_this_round = len(batch)
            remaining -= current_batch
            progress.update(task, advance=generated_this_round)

    # Write output
    out_path = Path(output)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    console.print(
        f"\n[green]Generated {len(all_examples)} examples[/]\n"
        f"Format:     [bold]{fmt}[/]\n"
        f"Output:     [bold]{out_path}[/]\n"
        + (f"Duplicates: [yellow]{duplicates} removed[/]\n" if duplicates > 0 else "")
    )


def _generate_batch(
    prompt: str,
    count: int,
    fmt: str,
    provider: str,
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: float,
    seed_examples: list[dict],
) -> list[dict]:
    """Generate a batch of examples using the specified provider."""
    if provider == "openai":
        return _generate_openai(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            seed_examples=seed_examples,
        )
    elif provider == "local":
        return _generate_local(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            temperature=temperature,
            seed_examples=seed_examples,
        )
    elif provider == "server":
        return _generate_server(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            api_base=api_base,
            temperature=temperature,
            seed_examples=seed_examples,
        )
    return []


def _build_generation_prompt(prompt: str, count: int, fmt: str, seed_examples: list) -> str:
    """Build the prompt for data generation."""
    format_spec = {
        "alpaca": (
            'Each example must be a JSON object with keys: '
            '"instruction", "input" (can be empty string), "output".'
        ),
        "sharegpt": (
            'Each example must be a JSON object with key "conversations", '
            'which is a list of objects with "from" (human/gpt) and "value".'
        ),
        "chatml": (
            'Each example must be a JSON object with key "messages", '
            'which is a list of objects with "role" (user/assistant) and "content".'
        ),
    }

    system_msg = (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality training examples.\n\n"
        f"Topic/Instructions: {prompt}\n\n"
        f"Format: {format_spec[fmt]}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )

    if seed_examples:
        seed_str = json.dumps(seed_examples[:3], ensure_ascii=False, indent=2)
        system_msg += f"\n\nHere are some seed examples to guide the style:\n{seed_str}"

    return system_msg


def _generate_openai(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: float,
    seed_examples: list[dict],
) -> list[dict]:
    """Generate examples using OpenAI-compatible API."""
    import os

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or pass --api-key."
        )

    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for OpenAI generation. Install: pip install httpx")

    base_url = api_base or "https://api.openai.com/v1"

    # Validate api_base to prevent SSRF (block non-HTTPS remote URLs)
    if api_base:
        from urllib.parse import urlparse

        parsed = urlparse(api_base)
        is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
        if not is_local and parsed.scheme != "https":
            raise ValueError(
                f"api_base must use HTTPS for remote APIs (got {parsed.scheme}://). "
                "HTTP is only allowed for localhost."
            )

    generation_prompt = _build_generation_prompt(prompt, count, fmt, seed_examples)

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": f"Generate {count} training examples now."},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        logger.debug("API error response: %s", response.text)
        raise ValueError(
            f"API returned {response.status_code}. Check your API key and model name."
        )

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return _parse_json_array(content)


def _generate_local(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    temperature: float,
    seed_examples: list[dict],
) -> list[dict]:
    """Generate examples using a local model via transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    generation_prompt = _build_generation_prompt(prompt, count, fmt, seed_examples)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": f"Generate {count} training examples now."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{generation_prompt}\n\nGenerate {count} training examples now.\n\n"

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=4096,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _parse_json_array(content)


def _generate_server(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    api_base: Optional[str],
    temperature: float,
    seed_examples: list[dict],
) -> list[dict]:
    """Generate examples using a local OpenAI-compatible server (soup serve, Ollama, etc.).

    Unlike the 'openai' provider, no API key is required. Connects to a running
    local inference server via its OpenAI-compatible /v1/chat/completions endpoint.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for server generation. Install: pip install httpx")

    base_url = api_base or "http://localhost:8000/v1"

    # Validate api_base to prevent SSRF — only allow http/https, block remote non-HTTPS
    if api_base:
        from urllib.parse import urlparse

        parsed = urlparse(api_base)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"api_base must use HTTP or HTTPS scheme (got {parsed.scheme}://)"
            )
        is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
        if not is_local and parsed.scheme != "https":
            raise ValueError(
                f"api_base must use HTTPS for remote APIs (got {parsed.scheme}://). "
                "HTTP is only allowed for localhost."
            )

    # Strip trailing /v1 if present (we add it to the endpoint path)
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    generation_prompt = _build_generation_prompt(prompt, count, fmt, seed_examples)

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": f"Generate {count} training examples now."},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=300.0,
    )

    if response.status_code != 200:
        logger.debug("Server error response: %s", response.text)
        raise ValueError(
            f"Server returned {response.status_code}. "
            "Check that the server is running and the model name is correct."
        )

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return _parse_json_array(content)


def _parse_json_array(content: str) -> list[dict]:
    """Parse a JSON array from LLM output, handling markdown code blocks."""
    content = content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Try to find JSON array in content
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1 and end > start:
        content = content[start:end + 1]

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Try line-by-line JSON objects
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    return results


def _validate_example(example: dict, fmt: str) -> bool:
    """Validate a single generated example matches the expected format."""
    if fmt == "alpaca":
        return "instruction" in example and "output" in example
    elif fmt == "sharegpt":
        convos = example.get("conversations", [])
        return len(convos) >= 2
    elif fmt == "chatml":
        msgs = example.get("messages", [])
        return len(msgs) >= 2
    return False


def _row_to_text(row: dict) -> str:
    """Convert a row to a text string for dedup comparison."""
    return " ".join(str(v) for v in row.values() if v)
